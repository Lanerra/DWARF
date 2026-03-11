"""
DSQG Attention V3 — d52 Optimized Kernel (J=44)
================================================

Optimized version of d50 pure geometry kernel with:
  - @triton.autotune on forward and backward kernels
  - BLOCK_N, num_warps, num_stages tuning per (HD, N)
  - Preloaded pos_bias scalars and sparse scale_embed vectors
  - HD=48 support for 27M model (D=384, H=8, HD=48)

Offset set: δ ∈ [0..40, 128, 384, 1536] (44 total)

Score formula:
  score[n,j,h] = Q[n,h]·K[n−δⱼ,h] / √HD + pos_bias[j,h] + Q[n,h]·SE[j] / √HD

Usage:
  from dsqg_d52_opt import dsqg_attention_v3, DSQGAttentionModule
"""

import math
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

_SPARSE_LIST = [128, 384, 1536]
ALL_OFFSETS  = list(range(41)) + _SPARSE_LIST
assert len(ALL_OFFSETS) == 44


def _next_pow2(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ─────────────────────────────────────────────────────────────────────────────
# ⚙ Autotune configurations
# ─────────────────────────────────────────────────────────────────────────────

_fwd_configs = [
    triton.Config({'BLOCK_N': 8},  num_warps=2, num_stages=2),
    triton.Config({'BLOCK_N': 8},  num_warps=4, num_stages=3),
    triton.Config({'BLOCK_N': 16}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_N': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_N': 16}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_N': 16}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_N': 16}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_N': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_N': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 32}, num_warps=8, num_stages=4),
]

_compute_D_configs = [
    triton.Config({'BLOCK_N': 32},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
]


def _zero_dpb_dse(nargs):
    nargs['DPB'].zero_()
    nargs['DSE'].zero_()


_bwd_dq_configs = [
    triton.Config({'BLOCK_N': 16},  num_warps=2, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 16},  num_warps=4, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 32},  num_warps=4, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 32},  num_warps=4, num_stages=3, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=3, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=3, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2, pre_hook=_zero_dpb_dse),
    triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3, pre_hook=_zero_dpb_dse),
]

_bwd_dkdv_configs = [
    triton.Config({'BLOCK_M': 8},  num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 8},  num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 16}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=3),
]


# ─────────────────────────────────────────────────────────────────────────────
# ⏩ Forward Kernel (autotuned)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_fwd_configs, key=['HD', 'N'])
@triton.jit
def _fwd_v3_opt(
    Q, K, V, POS_BIAS, SE, OUT, LSE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H;           h   = bh % H
    n0  = blk * BLOCK_N

    ns  = n0 + tl.arange(0, BLOCK_N)
    nm  = ns < N
    ds  = tl.arange(0, BLOCK_HD)
    dm  = ds < HD
    sc  = 1.0 / (HD ** 0.5)

    qb  = Q + b * stride_qb + h * stride_qh
    kb  = K + b * stride_kb + h * stride_kh
    vb  = V + b * stride_vb + h * stride_vh

    q = tl.load(qb + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    mi  = tl.full([BLOCK_N], float('-inf'), tl.float32)
    li  = tl.zeros([BLOCK_N], tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # ── Phase 1: Dense δ=0..40 ──────────────────────────────────────────
    # Scale_embed (44×HD, ≤8.4 KB) fits in L1 cache; sequential access
    # ensures cache residency across iterations. Pipelining via num_stages
    # overlaps loads with compute.
    for d in tl.static_range(41):
        kp  = ns - d
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0)

        se_d = tl.load(SE + d * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        s   = tl.sum(q * (kt.to(tl.float32) + se_d[None, :]), axis=1) * sc
        s  += tl.load(POS_BIAS + d * stride_pbi + h * stride_pbh)

        s   = tl.where(val, s, float('-inf'))

        mn  = tl.maximum(mi, s)
        cor = tl.exp(mi - mn)
        p   = tl.exp(s  - mn)
        li  = li * cor + p
        acc = acc * cor[:,None] + p[:,None] * vt.to(tl.float32)
        mi  = mn

    # ── Phase 2: Sparse δ=(128, 384, 1536) ──────────────────────────────
    # Preload sparse scale_embed and pos_bias before the loop
    _sparse_se_0 = tl.load(SE + 41 * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
    _sparse_se_1 = tl.load(SE + 42 * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
    _sparse_se_2 = tl.load(SE + 43 * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
    _sparse_pb_0 = tl.load(POS_BIAS + 41 * stride_pbi + h * stride_pbh)
    _sparse_pb_1 = tl.load(POS_BIAS + 42 * stride_pbi + h * stride_pbh)
    _sparse_pb_2 = tl.load(POS_BIAS + 43 * stride_pbi + h * stride_pbh)

    for si in tl.static_range(3):
        sd  = (128, 384, 1536)[si]
        se_j = (_sparse_se_0, _sparse_se_1, _sparse_se_2)[si]
        pb_j = (_sparse_pb_0, _sparse_pb_1, _sparse_pb_2)[si]

        kp  = ns - sd
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0)

        s   = tl.sum(q * (kt.to(tl.float32) + se_j[None, :]), axis=1) * sc
        s  += pb_j

        s   = tl.where(val, s, float('-inf'))

        mn  = tl.maximum(mi, s)
        cor = tl.exp(mi - mn)
        p   = tl.exp(s  - mn)
        li  = li * cor + p
        acc = acc * cor[:,None] + p[:,None] * vt.to(tl.float32)
        mi  = mn

    ls  = tl.where(li > 0.0, li, 1.0)
    out = acc / ls[:,None]
    lse = mi + tl.log(ls)

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             out.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# ⏪ Backward: D[n] = dot(dout[n], out[n])  (autotuned)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_compute_D_configs, key=['HD', 'N'])
@triton.jit
def _compute_D_v3_opt(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_db,  stride_dh,  stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh = tl.program_id(0); blk = tl.program_id(1)
    b  = bh // H; h = bh % H
    n0 = blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N); nm = ns < N
    ds = tl.arange(0, BLOCK_HD);     dm = ds < HD
    do = tl.load(DO + b*stride_dob + h*stride_doh
                 + ns[:,None]*stride_don + ds[None,:]*stride_dod,
                 mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    o  = tl.load(O  + b*stride_ob  + h*stride_oh
                 + ns[:,None]*stride_on  + ds[None,:]*stride_od,
                 mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    tl.store(D + b*stride_db + h*stride_dh + ns*stride_dn,
             tl.sum(do * o, axis=1), mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# ⏪ Backward: dQ + dPOS_BIAS + dSCALE_EMBED  (autotuned, pre_hook zeros DPB/DSE)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_bwd_dq_configs, key=['HD', 'N'])
@triton.jit
def _bwd_dq_v3_opt(
    Q, K, V, PB, SE, DO, O, LSE, Dv, DQ, DPB, DSE,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_lb,  stride_lh,  stride_ln,
    stride_Db,  stride_Dh,  stride_Dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dpbi, stride_dpbh,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dsei, stride_dsed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H; h = bh % H
    n0  = blk * BLOCK_N
    ns  = n0 + tl.arange(0, BLOCK_N); nm = ns < N
    ds  = tl.arange(0, BLOCK_HD);     dm = ds < HD
    sc  = 1.0 / (HD ** 0.5)

    qb  = Q  + b*stride_qb + h*stride_qh
    kb  = K  + b*stride_kb + h*stride_kh
    vb  = V  + b*stride_vb + h*stride_vh
    dob = DO + b*stride_dob + h*stride_doh

    q   = tl.load(qb  + ns[:,None]*stride_qn  + ds[None,:]*stride_qd,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    do  = tl.load(dob + ns[:,None]*stride_don  + ds[None,:]*stride_dod,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b*stride_lb + h*stride_lh + ns*stride_ln, mask=nm, other=0.0)
    Dval= tl.load(Dv  + b*stride_Db + h*stride_Dh + ns*stride_Dn, mask=nm, other=0.0)
    dq  = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in tl.static_range(44):
        delta = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                 128, 384, 1536)[i]
        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        kt_se = kt + se_i[None, :]

        s     = tl.sum(q * kt_se, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s     = tl.where(val, s, float('-inf'))

        alpha = tl.where(val, tl.exp(s - lse), 0.0)
        ds_v  = alpha * (tl.sum(do * vt, axis=1) - Dval)

        dq   += ds_v[:,None] * kt_se * sc

        tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                      tl.sum(tl.where(val, ds_v, 0.0), axis=0))

        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
        tl.atomic_add(DSE + i * stride_dsei + ds * stride_dsed,
                      tl.where(dm, dse_i, 0.0))

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# ⏪ Backward: dK/dV  (autotuned)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=_bwd_dkdv_configs, key=['HD', 'N'])
@triton.jit
def _bwd_dkdv_v3_opt(
    Q, K, V, PB, SE, DO, LSE, Dv, DK, DV,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb,  stride_lh,  stride_ln,
    stride_Db,  stride_Dh,  stride_Dn,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H; h = bh % H
    m0  = blk * BLOCK_M
    ms  = m0 + tl.arange(0, BLOCK_M); mm = ms < N
    ds  = tl.arange(0, BLOCK_HD);     dm = ds < HD
    sc  = 1.0 / (HD ** 0.5)

    kb  = K  + b*stride_kb + h*stride_kh
    vb  = V  + b*stride_vb + h*stride_vh
    qb  = Q  + b*stride_qb + h*stride_qh
    dob = DO + b*stride_dob + h*stride_doh

    kt  = tl.load(kb + ms[:,None]*stride_kn + ds[None,:]*stride_kd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    vt  = tl.load(vb + ms[:,None]*stride_vn + ds[None,:]*stride_vd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    dk  = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv  = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)

    for i in tl.static_range(44):
        delta = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                 128, 384, 1536)[i]
        np_  = ms + delta
        val  = (np_ < N) & mm

        qn   = tl.load(qb  + np_[:,None]*stride_qn  + ds[None,:]*stride_qd,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don  = tl.load(dob + np_[:,None]*stride_don  + ds[None,:]*stride_dod,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn   = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        s    = tl.sum(qn * (kt + se_i[None, :]), axis=1) * sc
        s   += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s    = tl.where(val, s, float('-inf'))

        alpha = tl.where(val, tl.exp(s - lsen), 0.0)
        ds_v  = alpha * (tl.sum(don * vt, axis=1) - Dn)
        dv   += alpha[:,None] * don
        dk   += ds_v[:,None] * qn * sc

    tl.store(DK + b*stride_dkb + h*stride_dkh
             + ms[:,None]*stride_dkn + ds[None,:]*stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])
    tl.store(DV + b*stride_dvb + h*stride_dvh
             + ms[:,None]*stride_dvn + ds[None,:]*stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# 🔄 Autograd wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (44, H)
        assert scale_embed.shape == (44, HD)
        BLOCK_HD = _next_pow2(HD)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        grid = lambda META: (B * H, triton.cdiv(N, META['BLOCK_N']))

        _fwd_v3_opt[grid](
            q, k, v, pos_bias, scale_embed, out, lse,
            q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
            k.stride(0),   k.stride(1),   k.stride(2),   k.stride(3),
            v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0),    pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD, BLOCK_HD=BLOCK_HD,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        ctx.BLOCK_HD = BLOCK_HD
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pb, se, out, lse = ctx.saved_tensors
        B, H, N, HD = q.shape
        BHD = ctx.BLOCK_HD
        dout = dout.contiguous()

        D   = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        grid_D = lambda META: (B * H, triton.cdiv(N, META['BLOCK_N']))
        _compute_D_v3_opt[grid_D](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_HD=BHD,
        )

        dq   = torch.empty_like(q)
        dpb  = torch.zeros_like(pb)
        dse  = torch.zeros_like(se)
        grid_dq = lambda META: (B * H, triton.cdiv(N, META['BLOCK_N']))
        _bwd_dq_v3_opt[grid_dq](
            q, k, v, pb, se, dout, out, lse, D, dq, dpb, dse,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            dpb.stride(0),  dpb.stride(1),
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            dse.stride(0),  dse.stride(1),
            H=H, N=N, HD=HD, BLOCK_HD=BHD,
        )

        dk  = torch.empty_like(k)
        dv  = torch.empty_like(v)
        grid_dkdv = lambda META: (B * H, triton.cdiv(N, META['BLOCK_M']))
        _bwd_dkdv_v3_opt[grid_dkdv](
            q, k, v, pb, se, dout, lse, D, dk, dv,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dk.stride(0),   dk.stride(1),   dk.stride(2),   dk.stride(3),
            dv.stride(0),   dv.stride(1),   dv.stride(2),   dv.stride(3),
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            H=H, N=N, HD=HD, BLOCK_HD=BHD,
        )
        return dq, dk, dv, dpb, dse


def dsqg_attention_v3(q, k, v, pos_bias, scale_embed):
    """
    q, k, v:      [B, H, N, HD]  bfloat16
    pos_bias:     [44, H]         float32 — global learned frequency prior
    scale_embed:  [44, HD]        float32 — Q-dynamic matched-filter embeddings
    Returns:      [B, H, N, HD]  same dtype as input
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV3.apply(q, k, v, pos_bias.float(), scale_embed.float())
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 📦 Drop-in module
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionV3(nn.Module):
    """
    DSQG V3 with d50 pure geometry (J=44, offsets [0..40, 128, 384, 1536]).

    Parameter: scale_embed [44, HD]
      Initialized to zeros → starts as pure pos_bias (backward compatible).
      Learns to encode which Q directions predict high attention at each offset.

    Score for offset j:
      score[n,h,j] = Q[n,h]·K[n-δⱼ,h]/√HD + pos_bias[j,h] + Q[n,h]·SE[j]/√HD
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        HD = self.head_dim

        if offsets is None:
            offsets = ALL_OFFSETS
        assert list(offsets) == ALL_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))

        self.scale_embed = nn.Parameter(torch.zeros(44, HD))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        out = dsqg_attention_v3(q, k, v, self.pos_bias, self.scale_embed)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
            se = self.scale_embed.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean':   se.abs().mean().item(),
            'scale_embed_abs_max':    se.abs().max().item(),
        }


DSQGAttentionModule = DSQGAttentionV3


# ─────────────────────────────────────────────────────────────────────────────
# 📐 Reference implementation (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_v3(q, k, v, pos_bias, scale_embed):
    B, H, N, HD = q.shape
    sc  = HD ** -0.5
    off = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)
    kp  = F.pad(k.float(), (0, 0, 1536, 0))
    vp  = F.pad(v.float(), (0, 0, 1536, 0))
    ni  = torch.arange(N, device=q.device)
    gi  = 1536 - off[None,:] + ni[:,None]
    Ka  = kp[:, :, gi, :]; Va = vp[:, :, gi, :]
    s   = (q.float().unsqueeze(3) * Ka).sum(-1) * sc
    s  += pos_bias.T[None,:,None,:]
    q_dyn = (q.float().unsqueeze(3) * scale_embed[None,None,:,:]).sum(-1) * sc
    s  += q_dyn
    s   = s.masked_fill((ni[:,None] < off[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a   = F.softmax(s, dim=-1)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 🧪 Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    print("=" * 72)
    print("DSQG V3 d52 Optimized — Correctness Tests")
    print("  J=44, δ∈[0..40, 128, 384, 1536]")
    print("  Autotuned BLOCK_N × num_warps × num_stages")
    print("=" * 72)

    cfgs = [
        (1,  8,   64, 32, "HD=32 tiny"),
        (2,  8,  512, 32, "HD=32 mid (all offsets)"),
        (4,  8, 2047, 32, "HD=32 14M shape"),
        (2,  8,  128, 48, "HD=48 27M shape"),
        (2,  8,  512, 48, "HD=48 mid (all offsets)"),
        (2,  8, 2047, 48, "HD=48 near-full"),
    ]

    ok_all = True
    for B, H, N, HD, lbl in cfgs:
        torch.manual_seed(42)
        q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05

        ref = _reference_v3(q.detach(), k.detach(), v.detach(), pb, se)
        out = dsqg_attention_v3(q.detach().clone(), k.detach().clone(),
                                v.detach().clone(), pb, se)
        forward_error = (ref.float() - out.float()).abs().max().item()

        qr, kr, vr = [t.clone().detach().requires_grad_(True) for t in (q, k, v)]
        pb_r = pb.clone().detach().requires_grad_(True)
        se_r = se.clone().detach().requires_grad_(True)
        _reference_v3(qr, kr, vr, pb_r, se_r).sum().backward()
        dqr = qr.grad.clone(); dkr = kr.grad.clone(); dvr = vr.grad.clone()
        dpbr = pb_r.grad.clone(); dser = se_r.grad.clone()

        qt, kt2, vt = [t.clone().detach().requires_grad_(True) for t in (q, k, v)]
        pb_t = pb.clone().detach().requires_grad_(True)
        se_t = se.clone().detach().requires_grad_(True)
        dsqg_attention_v3(qt, kt2, vt, pb_t, se_t).sum().backward()
        error_qkv = max(
            (qt.grad.float() - dqr.float()).abs().max().item(),
            (kt2.grad.float() - dkr.float()).abs().max().item(),
            (vt.grad.float() - dvr.float()).abs().max().item(),
        )
        error_pb = (pb_t.grad.float() - dpbr.float()).abs().max().item()
        error_se = (se_t.grad.float() - dser.float()).abs().max().item()

        ok = max(forward_error, error_qkv, error_pb, error_se) < 0.05
        if not ok:
            ok_all = False
        print(f"  {lbl:28s}  fwd={forward_error:.4f}  "
              f"bwd_qkv={error_qkv:.4f}  bwd_pb={error_pb:.4f}  "
              f"bwd_se={error_se:.4f}  {'PASS ✓' if ok else 'FAIL ✗'}")

    # ── Autotune warmup ─────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("⚙ Autotune warmup (forward + backward at training shapes)")
    print("-" * 72)

    for HD in [32, 48]:
        B_w, H_w, N_w = 2, 8, 2048
        torch.manual_seed(0)
        q  = torch.randn(B_w, H_w, N_w, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B_w, H_w, N_w, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B_w, H_w, N_w, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H_w,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        pb_p = pb.clone().detach().requires_grad_(True)
        se_p = se.clone().detach().requires_grad_(True)
        out = dsqg_attention_v3(q, k, v, pb_p, se_p)
        out.sum().backward()
        torch.cuda.synchronize()
        print(f"  HD={HD:3d}  warmup complete  shape=[{B_w},{H_w},{N_w},{HD}]")

    # ── Timing comparison ───────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("⏱ Timing comparison (B=8, H=8, N=2048)")
    print("-" * 72)

    for HD in [32, 48]:
        B_t, H_t, N_t = 8, 8, 2048
        torch.manual_seed(0)
        q  = torch.randn(B_t, H_t, N_t, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B_t, H_t, N_t, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B_t, H_t, N_t, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H_t,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05

        for _ in range(5):
            dsqg_attention_v3(q, k, v, pb, se)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iterations = 100
        start_event.record()
        for _ in range(iterations):
            dsqg_attention_v3(q, k, v, pb, se)
        end_event.record()
        torch.cuda.synchronize()
        fwd_ms = start_event.elapsed_time(end_event) / iterations
        fwd_tok_per_sec = B_t * N_t / (fwd_ms / 1000)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        pb_p = pb.clone().detach().requires_grad_(True)
        se_p = se.clone().detach().requires_grad_(True)
        for _ in range(5):
            out = dsqg_attention_v3(q, k, v, pb_p, se_p)
            out.sum().backward()
        torch.cuda.synchronize()

        bwd_iterations = 50
        start_event.record()
        for _ in range(bwd_iterations):
            out = dsqg_attention_v3(q, k, v, pb_p, se_p)
            out.sum().backward()
        end_event.record()
        torch.cuda.synchronize()
        fwd_bwd_ms = start_event.elapsed_time(end_event) / bwd_iterations
        fwd_bwd_tok_per_sec = B_t * N_t / (fwd_bwd_ms / 1000)

        print(f"  HD={HD:3d}  fwd: {fwd_ms:.3f} ms  ({fwd_tok_per_sec/1e6:.2f}M tok/s)  "
              f"fwd+bwd: {fwd_bwd_ms:.3f} ms  ({fwd_bwd_tok_per_sec/1e6:.2f}M tok/s)")

    print("\n" + "=" * 72)
    print(f"{'ALL PASSED ✓' if ok_all else 'SOME FAILED ✗'}")
    print("=" * 72)
    return ok_all


if __name__ == "__main__":
    run_tests()
