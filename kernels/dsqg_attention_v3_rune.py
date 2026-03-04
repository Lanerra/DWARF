"""
DSQG Attention V3-RunE — Dense 1-39 + Octave Blocks at 64-512
==============================================================

Derived from dsqg_attention_v3_rund.py. Two changes vs Run D:
  1. Offset set: dense 1-39 replaces sparse {1,2,4,8,16}
  2. J=79 (vs J=53)

Design principle — "full coverage":
  Run D showed that sparse short-range offsets {1,2,4,8,16} hurt both PPL
  (natural language needs dense short-range attention for coreference, syntax,
  local agreement) and short-range passkey (δ_eff for d=4,8,16 not covered).

  Run E restores dense 1-31 (matching V3/condU's natural-language modeling
  power) and extends to 39 (covering δ_eff=39 for d=32 passkey). The octave
  blocks at 64-512 are kept from Run D, providing direct δ_eff coverage for
  all five long-range octave passkey distances.

  Rust verification (offset_optimizer::run_e_analysis) confirms:
  - δ_eff directly covered for d=1,2,4,8,16,32,64,128,256,512 (10/12)
  - d=1024,1536 via multi-hop synthesis (proven to work in Run D)
  - Short-range density (δ=1-31): 31 offsets — matches V3 baseline
  - Aggregate path-count score: 24× Run D

Offset set (J=79):
  Base:        {0}                              (1)  — self-attention component
  Dense 1-39:  {1,2,...,39}                    (39)  — NL modeling + d=1-32 δ_eff
  Gap:         {48}                             (1)  — combination-tone anchor
  Block d≈64:  {64,65,...,71}                   (8)  — δ_eff for d=64
  Gap:         {96}                             (1)  — combination-tone anchor
  Block d≈128: {128,129,...,135}                (8)  — δ_eff for d=128
  Gap:         {192}                            (1)  — combination-tone anchor
  Block d≈256: {256,257,...,263}                (8)  — δ_eff for d=256
  Gap:         {384}                            (1)  — combination-tone anchor
  Block d≈512: {512,513,...,519}                (8)  — δ_eff for d=512
  Tail:        {768, 1024, 1536}                (3)  — sparse very long-range
  Total J=79

Comparison to Run D (J=53):
  Run D used sparse {1,2,4,8,16} for short range — caused 3.3 PPL penalty
  vs condU baseline and left d=4,8,16 passkey distances without δ_eff.
  Run E restores the full short-range palette while keeping all octave blocks.
  Expected: ~52-53 PPL (near V3 baseline), passkey ≥40% for all d=1-512.

Init: near-zero pos_bias (zeros) — same as Run D.
  All heads start equal; specialization emerges from gradient dynamics.
  Avoids linspace locality prior that delayed passkey signal to ep3 in Run C.

Smoke test:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v3_rune.py

Training:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/dsqg_hybrid_13m_2048_rune.py \\
    2>&1 | tee benchmarks/logs/dsqg_hybrid_13m_2048_rune_run.log
"""

import math
import warnings
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

# <<< RUNE: dense 1-39 + octave blocks at 64-512, J=79
_BASE   = [0]
_DENSE  = list(range(1, 40))        # dense 1-39: NL modeling + all short δ_eff
_D64    = list(range(64, 72))       # dense octave 1: δ=64-71
_D128   = list(range(128, 136))     # dense octave 2: δ=128-135
_D256   = list(range(256, 264))     # dense octave 3: δ=256-263
_D512   = list(range(512, 520))     # dense octave 4: δ=512-519
_GAPS   = [48, 96, 192, 384]
_TAIL   = [768, 1024, 1536]
ALL_OFFSETS = sorted(_BASE + _DENSE + [48] + _D64 + [96] +
                     _D128 + [192] + _D256 + [384] + _D512 + _TAIL)
assert len(ALL_OFFSETS) == 79, f"Expected 79 offsets, got {len(ALL_OFFSETS)}"
_MAX_OFFSET = 1536   # unchanged — same KV buffer size as V3/RunD

# Triton compile-time tuple (must match ALL_OFFSETS exactly)
_OFFSETS_TUPLE = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    48, 64, 65, 66, 67, 68, 69, 70, 71, 96,
    128, 129, 130, 131, 132, 133, 134, 135, 192, 256,
    257, 258, 259, 260, 261, 262, 263, 384, 512, 513,
    514, 515, 516, 517, 518, 519, 768, 1024, 1536,
)
assert list(_OFFSETS_TUPLE) == ALL_OFFSETS, "Tuple/list mismatch — check _OFFSETS_TUPLE"

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_rune(
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

    # <<< RUNE: static_range(79) + dense-1-39 + octave-block offset tuple
    for i in tl.static_range(79):
        delta = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            48, 64, 65, 66, 67, 68, 69, 70, 71, 96,
            128, 129, 130, 131, 132, 133, 134, 135, 192, 256,
            257, 258, 259, 260, 261, 262, 263, 384, 512, 513,
            514, 515, 516, 517, 518, 519, 768, 1024, 1536,
        )[i]

        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        pb_i = tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.float32)

        s   = tl.sum(q * kt, axis=1) * sc + pb_i
        s  += tl.sum(q * se_i[None, :], axis=1) * sc
        s   = tl.where(val, s, float('-inf'))

        mi_new = tl.maximum(mi, s)
        alpha  = tl.exp(mi - mi_new)
        mi     = mi_new
        li     = li * alpha + tl.exp(s - mi)
        acc    = acc * alpha[:,None] + tl.exp(s - mi)[:,None] * vt

    out = acc / tl.maximum(li[:,None], 1e-6)
    lse = mi + tl.log(tl.maximum(li, 1e-6))

    ob  = OUT + b * stride_ob + h * stride_oh
    lb  = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             out.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# Backward dQ kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_rune(
    Q, K, V, PB, SE, DO, LSE, Dv, DQ, DPB, DSE,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
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

    # <<< RUNE: static_range(79)
    for i in tl.static_range(79):
        delta = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            48, 64, 65, 66, 67, 68, 69, 70, 71, 96,
            128, 129, 130, 131, 132, 133, 134, 135, 192, 256,
            257, 258, 259, 260, 261, 262, 263, 384, 512, 513,
            514, 515, 516, 517, 518, 519, 768, 1024, 1536,
        )[i]
        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        q_dyn = tl.sum(q * se_i[None, :], axis=1) * sc

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s    += q_dyn
        s     = tl.where(val, s, float('-inf'))

        alpha = tl.where(val, tl.exp(s - lse), 0.0)
        ds_v  = alpha * (tl.sum(do * vt, axis=1) - Dval)

        dq   += ds_v[:,None] * kt * sc
        dq   += ds_v[:,None] * se_i[None, :] * sc

        tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                      tl.sum(tl.where(val, ds_v, 0.0), axis=0))

        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
        tl.atomic_add(DSE + i * stride_dsei + ds * stride_dsed,
                      tl.where(dm, dse_i, 0.0))

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# Backward dK/dV kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_rune(
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

    # <<< RUNE: static_range(79)
    for i in tl.static_range(79):
        delta = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            48, 64, 65, 66, 67, 68, 69, 70, 71, 96,
            128, 129, 130, 131, 132, 133, 134, 135, 192, 256,
            257, 258, 259, 260, 261, 262, 263, 384, 512, 513,
            514, 515, 516, 517, 518, 519, 768, 1024, 1536,
        )[i]
        np_  = ms + delta
        val  = (np_ < N) & mm

        qn   = tl.load(qb  + np_[:,None]*stride_qn  + ds[None,:]*stride_qd,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don  = tl.load(dob + np_[:,None]*stride_don  + ds[None,:]*stride_dod,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn   = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        q_dyn = tl.sum(qn * se_i[None, :], axis=1) * sc

        s    = tl.sum(qn * kt, axis=1) * sc
        s   += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s   += q_dyn
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
# Autograd wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnRunE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        J = pos_bias.shape[0]   # 79
        assert pos_bias.shape    == (J, H),  f"pos_bias must be [{J}, {H}]"
        assert scale_embed.shape == (J, HD), f"scale_embed must be [{J}, {HD}]"
        assert q.dtype == torch.bfloat16

        BN  = 128 if HD <= 64 else 64
        BHD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        _fwd_rune[(B*H, triton.cdiv(N, BN))](
            q, k, v, pos_bias, scale_embed, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        ctx.BN = BN; ctx.BHD = BHD
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, pb, se, out, lse = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD = ctx.BN, ctx.BHD
        grad_out = grad_out.contiguous()

        Dv  = (grad_out.float() * out.float()).sum(-1)
        g   = (B * H, triton.cdiv(N, BN))
        dq  = torch.empty_like(q)
        dpb = torch.zeros_like(pb)
        dse = torch.zeros_like(se)

        _bwd_dq_rune[g](
            q, k, v, pb, se, grad_out, lse, Dv, dq, dpb, dse,
            q.stride(0),        q.stride(1),        q.stride(2),        q.stride(3),
            k.stride(0),        k.stride(1),        k.stride(2),        k.stride(3),
            v.stride(0),        v.stride(1),        v.stride(2),        v.stride(3),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            lse.stride(0),      lse.stride(1),      lse.stride(2),
            Dv.stride(0),       Dv.stride(1),       Dv.stride(2),
            dq.stride(0),       dq.stride(1),       dq.stride(2),       dq.stride(3),
            dpb.stride(0),      dpb.stride(1),
            pb.stride(0),       pb.stride(1),
            se.stride(0),       se.stride(1),
            dse.stride(0),      dse.stride(1),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _bwd_dkdv_rune[g](
            q, k, v, pb, se, grad_out, lse, Dv, dk, dv,
            q.stride(0),        q.stride(1),        q.stride(2),        q.stride(3),
            k.stride(0),        k.stride(1),        k.stride(2),        k.stride(3),
            v.stride(0),        v.stride(1),        v.stride(2),        v.stride(3),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            lse.stride(0),      lse.stride(1),      lse.stride(2),
            Dv.stride(0),       Dv.stride(1),       Dv.stride(2),
            dk.stride(0),       dk.stride(1),       dk.stride(2),       dk.stride(3),
            dv.stride(0),       dv.stride(1),       dv.stride(2),       dv.stride(3),
            pb.stride(0),       pb.stride(1),
            se.stride(0),       se.stride(1),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
        )

        return dq, dk, dv, dpb, dse


def dsqg_attention_v3_rune(q, k, v, pos_bias, scale_embed):
    """
    q, k, v:      [B, H, N, HD]  bfloat16
    pos_bias:     [79, H]         float32  — near-zero (zeros) init recommended
    scale_embed:  [79, HD]        float32
    Returns:      [B, H, N, HD]  same dtype as input
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnRunE.apply(q, k, v, pos_bias.float(), scale_embed.float())
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch — correctness testing only)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_rune(q, k, v, pos_bias, scale_embed):
    B, H, N, HD = q.shape
    sc  = HD ** -0.5
    off = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)
    kp  = F.pad(k.float(), (0, 0, _MAX_OFFSET, 0))
    vp  = F.pad(v.float(), (0, 0, _MAX_OFFSET, 0))
    ni  = torch.arange(N, device=q.device)
    gi  = _MAX_OFFSET - off[None, :] + ni[:, None]   # [N, J]
    Ka  = kp[:, :, gi, :]; Va = vp[:, :, gi, :]      # [B, H, N, J, HD]
    s   = (q.float().unsqueeze(3) * Ka).sum(-1) * sc  # [B, H, N, J]
    s  += pos_bias.T[None, :, None, :]                 # [B, H, N, J]
    q_dyn = (q.float().unsqueeze(3) * scale_embed[None, None, :, :]).sum(-1) * sc
    s  += q_dyn
    s   = s.masked_fill((ni[:, None] < off[None, :]).unsqueeze(0).unsqueeze(0),
                        float('-inf'))
    a   = F.softmax(s, dim=-1)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(device='cuda'):
    J = len(ALL_OFFSETS)
    print(f"V3-RunE smoke test: J={J}, max_offset={_MAX_OFFSET}")
    print(f"  Dense 1-39: {list(range(1,40))}")
    print(f"  Octave blocks: [64-71] [128-135] [256-263] [512-519]")
    print(f"  Gap/tail: {{48,96,192,384,768,1024,1536}}")
    B, H, N, HD = 2, 8, 512, 32
    torch.manual_seed(42)
    q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    pb = torch.zeros(J, H,  device=device, dtype=torch.float32)   # near-zero init
    se = torch.zeros(J, HD, device=device, dtype=torch.float32)

    # Forward
    ref = _reference_rune(q.detach(), k.detach(), v.detach(), pb, se)
    out = dsqg_attention_v3_rune(q.detach().clone(), k.detach().clone(),
                                  v.detach().clone(), pb, se)
    fe  = (ref.float() - out.float()).abs().max().item()
    print(f"  Forward max error: {fe:.5f}  {'PASS ✓' if fe < 0.01 else 'FAIL ✗'}")

    # Backward
    qt, kt2, vt = [t.clone().detach().requires_grad_(True) for t in (q, k, v)]
    dsqg_attention_v3_rune(qt, kt2, vt, pb, se).sum().backward()
    dQ = qt.grad.float().abs().max().item()
    dK = kt2.grad.float().abs().max().item()
    dV = vt.grad.float().abs().max().item()
    nan_flag = any(g is None or g.isnan().any() for g in (qt.grad, kt2.grad, vt.grad))
    print(f"  Backward: dQ={dQ:.4f}  dK={dK:.4f}  dV={dV:.4f}  "
          f"{'NaN DETECTED ✗' if nan_flag else 'PASS ✓'}")
    print(f"  Smoke test complete.")
    return fe < 0.01 and not nan_flag


if __name__ == "__main__":
    run_smoke_test()
