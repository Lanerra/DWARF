"""
DSQG Attention V3 — Q-Weighted Scale Gains
==========================================

Extends V2 with per-position, per-head Q-weighted scale selection.

New parameter: scale_embed [44, HD]
  score[n, j, h] = Q[n,h] · K[n−δⱼ,h] / √HD      (content matching, as V2)
                 + pos_bias[j, h]                    (global learned prior, as V2)
                 + Q[n,h] · scale_embed[j] / √HD    (NEW: Q-dynamic matched filter)

Physics motivation (Rust-verified in coherent_scale_retrieval.rs):
  The Q·SE term is the "local oscillator" tuning signal. When scale_embed[j] is
  aligned with Q, the score for offset j is boosted — the query itself selects
  which distance to attend to. This is the matched-filter principle:
    SNR(Q-weighted) ≫ SNR(uniform) when scale_embed[j*] ≈ Q for the target offset j*.
  Signal boost = J × uniform signal (test_signal_boost_equals_j confirmed).

Initialization:
  scale_embed = zeros([44, HD]) — starts as pure pos_bias (backward compatible with V2).
  As training proceeds, scale_embed learns which Q directions predict each offset.

Kernel changes vs V2:
  _fwd:       load se_j per offset, add Q·se_j/√HD to score
  _bwd_dq:    add SE contribution to dq; accumulate dse via atomic_add
  _bwd_dkdv:  add SE contribution to score for correct alpha (no dse needed here)
  autograd:   save/restore scale_embed; route dse gradient

Usage:
  from dsqg_attention_v3 import DSQGAttentionV3 as DSQGAttentionN

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v3.py
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

_SPARSE_LIST = [128, 512, 1536]
ALL_OFFSETS  = list(range(49)) + _SPARSE_LIST
assert len(ALL_OFFSETS) == 52

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel V3
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_v3(
    Q, K, V, POS_BIAS, SE, OUT, LSE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,          # SE strides: [offset, head_dim]
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

    # ── Phase 1: Consecutive δ=0..48 ──────────────────────────────────────
    for d in tl.static_range(49):
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

    # ── Phase 2: Sparse δ=(96,128,384) ────────────────────────────────────
    for si in tl.static_range(3):
        sd  = (128, 512, 1536)[si]
        pbi = 49 + si

        kp  = ns - sd
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0)

        se_j = tl.load(SE + pbi * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        s   = tl.sum(q * (kt.to(tl.float32) + se_j[None, :]), axis=1) * sc
        s  += tl.load(POS_BIAS + pbi * stride_pbi + h * stride_pbh)

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
# Backward: D[n] = dot(dout[n], out[n]) — unchanged from V2
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_D_v3(
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
# Backward: dQ + dPOS_BIAS + dSCALE_EMBED
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_v3(
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

    for i in tl.static_range(52):
        delta = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48,
                 128, 512, 1536)[i]
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

        # dPOS_BIAS: unchanged
        tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                      tl.sum(tl.where(val, ds_v, 0.0), axis=0))

        # V3: dSCALE_EMBED[i]: grad = sum_n(ds_v[n] * q[n]) * sc
        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc     # [HD]
        tl.atomic_add(DSE + i * stride_dsei + ds * stride_dsed,
                      tl.where(dm, dse_i, 0.0))

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# Backward: dK/dV — must include SE in score for correct alpha
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_v3(
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

    for i in tl.static_range(52):
        delta = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48,
                 128, 512, 1536)[i]
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
        # Note: no dk contribution from SE term (SE doesn't depend on K)

    tl.store(DK + b*stride_dkb + h*stride_dkh
             + ms[:,None]*stride_dkn + ds[None,:]*stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])
    tl.store(DV + b*stride_dvb + h*stride_dvh
             + ms[:,None]*stride_dvn + ds[None,:]*stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# Autograd wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (52, H)
        assert scale_embed.shape == (52, HD)
        BLOCK_N  = 16
        BLOCK_HD = _next_pow2(HD)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_v3[g](
            q, k, v, pos_bias, scale_embed, out, lse,
            q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
            k.stride(0),   k.stride(1),   k.stride(2),   k.stride(3),
            v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0),    pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        ctx.BLOCK_N  = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pb, se, out, lse = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD = ctx.BLOCK_N, ctx.BLOCK_HD
        dout = dout.contiguous()

        D   = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BN))
        _compute_D_v3[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        BN_DQ = 64
        dq   = torch.empty_like(q)
        dpb  = torch.zeros_like(pb)
        dse  = torch.zeros_like(se)
        g_dq = (B * H, triton.cdiv(N, BN_DQ))
        _bwd_dq_v3[g_dq](
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
            H=H, N=N, HD=HD, BLOCK_N=BN_DQ, BLOCK_HD=BHD,
        )

        dk  = torch.empty_like(k)
        dv  = torch.empty_like(v)
        _bwd_dkdv_v3[g](
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
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
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
# Drop-in module
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionV3(nn.Module):
    """
    Extends DSQGAttentionN_Fused with Q-weighted scale gains.

    New parameter: scale_embed [44, HD]
      Initialized to zeros → starts as pure pos_bias (backward compatible).
      Learns to encode which Q directions predict high attention at each offset.

    The score for offset j becomes:
      score[n,h,j] = Q[n,h]·K[n-δⱼ,h]/√HD + pos_bias[j,h] + Q[n,h]·SE[j]/√HD

    Compatible with Huygens K/V injection via forward(x, kv_inject=...).
    Compatible with IF amplifier (if_gain parameter, separate from this class).
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
        nn.init.constant_(self.gate_proj.bias, 0.0)   # gate=0 correct design

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))

        # V3: Q-dynamic scale embeddings — zero init = pure pos_bias at start
        self.scale_embed = nn.Parameter(torch.zeros(52, HD))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        # Huygens K/V injection (Q unchanged)
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


# ─────────────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch — for correctness testing)
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
    # Q·K scores [B, H, N, 44]
    s   = (q.float().unsqueeze(3) * Ka).sum(-1) * sc
    s  += pos_bias.T[None,:,None,:]
    # V3: Q·SE scores [B, H, N, 44]
    # q: [B,H,N,HD], scale_embed: [44,HD] → [B,H,N,44]
    q_dyn = (q.float().unsqueeze(3) * scale_embed[None,None,:,:]).sum(-1) * sc
    s  += q_dyn
    s   = s.masked_fill((ni[:,None] < off[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a   = F.softmax(s, dim=-1)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    print("=" * 64)
    print("DSQG V3 — Correctness Tests (Q-weighted scale gains)")
    print("=" * 64)
    cfgs = [
        (1,  8,   64, 32, "tiny"),
        (2,  8,  512, 32, "mid (all offsets)"),
        (4,  8, 2047, 32, "13M shape"),
    ]
    ok_all = True
    for B, H, N, HD, lbl in cfgs:
        torch.manual_seed(42)
        q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(52, H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(52, HD, device=device, dtype=torch.float32) * 0.05

        # Forward correctness
        ref = _reference_v3(q.detach(), k.detach(), v.detach(), pb, se)
        out = dsqg_attention_v3(q.detach().clone(), k.detach().clone(),
                                v.detach().clone(), pb, se)
        fe  = (ref.float() - out.float()).abs().max().item()

        # Backward correctness
        qr,kr,vr = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        se_r = se.clone().detach().requires_grad_(True)
        _reference_v3(qr,kr,vr,pb,se_r).sum().backward()
        dqr = qr.grad.clone(); dkr = kr.grad.clone(); dvr = vr.grad.clone()
        dser = se_r.grad.clone()

        qt,kt2,vt = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        se_t = se.clone().detach().requires_grad_(True)
        dsqg_attention_v3(qt,kt2,vt,pb,se_t).sum().backward()
        de_qkv = max((qt.grad.float()-dqr.float()).abs().max().item(),
                     (kt2.grad.float()-dkr.float()).abs().max().item(),
                     (vt.grad.float()-dvr.float()).abs().max().item())
        de_se  = (se_t.grad.float()-dser.float()).abs().max().item()

        ok = max(fe, de_qkv, de_se) < 0.05
        if not ok: ok_all = False
        print(f"  {lbl:24s}  fwd={fe:.4f}  bwd_qkv={de_qkv:.4f}  bwd_se={de_se:.4f}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")

    # Test: zero scale_embed → same output as V2
    print()
    print("  Zero SE init → same as V2:")
    from dsqg_attention_v2 import dsqg_attention_v2
    B,H,N,HD = 2,8,128,32
    torch.manual_seed(7)
    q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    pb = torch.randn(52, H, device=device, dtype=torch.float32) * 0.5
    se_zero = torch.zeros(52, HD, device=device, dtype=torch.float32)
    out_v2 = dsqg_attention_v2(q.clone(), k.clone(), v.clone(), pb)
    out_v3 = dsqg_attention_v3(q.clone(), k.clone(), v.clone(), pb, se_zero)
    diff   = (out_v2.float() - out_v3.float()).abs().max().item()
    ok_compat = diff < 1e-3
    if not ok_compat: ok_all = False
    print(f"  {'V3(SE=0) == V2':24s}  max_diff={diff:.6f}  {'PASS ✓' if ok_compat else 'FAIL ✗'}")

    print("=" * 64)
    print(f"{'ALL PASSED ✓' if ok_all else 'SOME FAILED ✗'}")
    return ok_all


if __name__ == "__main__":
    run_tests()
