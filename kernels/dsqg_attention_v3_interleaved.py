"""
DSQG Attention V3-Interleaved — Interleaved Dense+Sparse Offset Design
=======================================================================

Derived from dsqg_attention_v3.py. The only change is the offset set.

V3 offset set (J=44):
  range(0,33) ∪ {48,64,96,128,192,256,384,512,768,1024,1536}
  Dense 0-32, then sparse. Gap at δ=33-47 causes d=32 passkey ceiling.

V3-Interleaved offset set (J=40):
  {0,...,15} ∪ {32,...,48} ∪ {64,128,256,512,768,1024,1536}
  Dense window 1 (δ=0-15), dense window 2 (δ=32-48, fills the gap),
  sparse long-range (δ=64-1536).

Motivation (from condU 35M pure DSQG, epochs 5-9):
  d=32 passkey stuck at 0% while d=16 reaches 60%. Root cause: the kernel
  has no offset at δ=33-47 (gap between dense 0-32 and sparse 48). The
  retrieval cue is ~4 tokens wide, so d=32 requires attending to δ≈36,
  which falls in the gap. Dense window 2 covers δ=32-48 directly.

Trade-off vs V3:
  Lost: δ=16-31 (16 dense offsets) → gap now at δ=16-31
  Gained: δ=33-48 (17 dense offsets) → gap at δ=33-47 filled
  The full-attention layer at position 5 compensates for DSQG coverage gaps
  in the hybrid architecture (condU). Pure DSQG would see d=16 degraded.

Max offset unchanged (1536): KV buffer size identical to V3. No other
structural changes. Backward-compatible with the same training loop.

All changes vs dsqg_attention_v3.py are marked with # <<< INTERLEAVED.
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

# <<< INTERLEAVED: two dense windows + sparse long-range, J=40
_DENSE_1    = list(range(16))           # δ=0-15  (16 offsets)
_DENSE_2    = list(range(32, 49))       # δ=32-48 (17 offsets) — fills the gap
_SPARSE     = [64, 128, 256, 512, 768, 1024, 1536]  # (7 offsets)
ALL_OFFSETS = _DENSE_1 + _DENSE_2 + _SPARSE
assert len(ALL_OFFSETS) == 40           # <<< INTERLEAVED: was 44
_MAX_OFFSET = 1536                      # unchanged — same KV buffer size

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel (V3-Interleaved)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_v3i(
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

    # <<< INTERLEAVED: static_range(40) and updated offset tuple
    for i in tl.static_range(40):
        delta = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                 64,128,256,512,768,1024,1536)[i]

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
# Backward dQ kernel  (ported from _bwd_dq_v3; only offsets changed)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_v3i(
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

    # <<< INTERLEAVED: static_range(40) + interleaved offset tuple
    for i in tl.static_range(40):
        delta = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                 64,128,256,512,768,1024,1536)[i]
        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        q_dyn = tl.sum(q * se_i[None, :], axis=1) * sc   # [BLOCK_N]

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s    += q_dyn
        s     = tl.where(val, s, float('-inf'))

        # Correct softmax backward: alpha zeroed for invalid positions (no div-by-zero)
        alpha = tl.where(val, tl.exp(s - lse), 0.0)
        ds_v  = alpha * (tl.sum(do * vt, axis=1) - Dval)

        dq   += ds_v[:,None] * kt * sc
        dq   += ds_v[:,None] * se_i[None, :] * sc

        # dPOS_BIAS gradient
        tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                      tl.sum(tl.where(val, ds_v, 0.0), axis=0))

        # dSCALE_EMBED gradient
        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
        tl.atomic_add(DSE + i * stride_dsei + ds * stride_dsed,
                      tl.where(dm, dse_i, 0.0))

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# Backward dK/dV kernel  (ported from _bwd_dkdv_v3; only offsets changed)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_v3i(
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

    # <<< INTERLEAVED: static_range(40) + interleaved offset tuple
    for i in tl.static_range(40):
        delta = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                 64,128,256,512,768,1024,1536)[i]
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

class _DSQGFnV3I(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        J = pos_bias.shape[0]  # 40
        assert pos_bias.shape    == (J, H),  f"pos_bias must be [{J}, {H}]"
        assert scale_embed.shape == (J, HD), f"scale_embed must be [{J}, {HD}]"
        assert q.dtype == torch.bfloat16

        BN  = 128 if HD <= 64 else 64
        BHD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        _fwd_v3i[(B*H, triton.cdiv(N, BN))](
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

        # D[b,h,n] = (grad_out[b,h,n] · out[b,h,n]).sum(HD)
        Dv = (grad_out.float() * out.float()).sum(-1)

        g   = (B * H, triton.cdiv(N, BN))
        dq  = torch.empty_like(q)
        dpb = torch.zeros_like(pb)
        dse = torch.zeros_like(se)

        _bwd_dq_v3i[g](
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

        _bwd_dkdv_v3i[g](
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


def dsqg_attention_v3_interleaved(q, k, v, pos_bias, scale_embed):
    """
    q, k, v:      [B, H, N, HD]  bfloat16
    pos_bias:     [40, H]         float32
    scale_embed:  [40, HD]        float32
    Returns:      [B, H, N, HD]  same dtype as input
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV3I.apply(q, k, v, pos_bias.float(), scale_embed.float())
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch — for correctness testing only)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_v3i(q, k, v, pos_bias, scale_embed):
    B, H, N, HD = q.shape
    sc  = HD ** -0.5
    off = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)
    kp  = F.pad(k.float(), (0, 0, _MAX_OFFSET, 0))
    vp  = F.pad(v.float(), (0, 0, _MAX_OFFSET, 0))
    ni  = torch.arange(N, device=q.device)
    gi  = _MAX_OFFSET - off[None,:] + ni[:,None]
    Ka  = kp[:, :, gi, :]; Va = vp[:, :, gi, :]
    s   = (q.float().unsqueeze(3) * Ka).sum(-1) * sc
    s  += pos_bias.T[None,:,None,:]
    q_dyn = (q.float().unsqueeze(3) * scale_embed[None,None,:,:]).sum(-1) * sc
    s  += q_dyn
    s   = s.masked_fill((ni[:,None] < off[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a   = F.softmax(s, dim=-1)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    J = len(ALL_OFFSETS)  # 40
    print(f"V3-Interleaved smoke test: J={J}, offsets={ALL_OFFSETS}")

    B, H, N, HD = 2, 8, 512, 32
    torch.manual_seed(42)
    q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    pb = torch.randn(J, H,  device=device, dtype=torch.float32) * 0.5
    se = torch.randn(J, HD, device=device, dtype=torch.float32) * 0.05

    ref = _reference_v3i(q.detach(), k.detach(), v.detach(), pb, se)
    out = dsqg_attention_v3_interleaved(q.detach().clone(), k.detach().clone(),
                                        v.detach().clone(), pb, se)
    fwd_err = (ref.float() - out.float()).abs().max().item()
    print(f"  Forward max error: {fwd_err:.5f}  {'PASS ✓' if fwd_err < 0.05 else 'FAIL ✗'}")

    qg = q.clone().detach().requires_grad_(True)
    kg = k.clone().detach().requires_grad_(True)
    vg = v.clone().detach().requires_grad_(True)
    se_g = se.clone().detach().requires_grad_(True)
    dsqg_attention_v3_interleaved(qg, kg, vg, pb, se_g).sum().backward()
    print(f"  Backward: dQ={qg.grad.abs().max():.4f}  dK={kg.grad.abs().max():.4f}"
          f"  dV={vg.grad.abs().max():.4f}  dSE={se_g.grad.abs().max():.4f}")
    print("  Smoke test complete.")
    sys.exit(0 if fwd_err < 0.05 else 1)
