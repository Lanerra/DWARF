"""
DSQG Attention V2 — Full Synthesis Kernel
==========================================
Synthesis of Gemini 3.1 Pro + GPT-5.2-Thinking + Opus 4.6 designs (Feb 28 2026).

Improvements over V1 (Opus baseline):
  Forward — consecutive offsets (δ=0..32, 33 iters):
    tl.make_block_ptr + tl.advance: Triton sees the sequential access pattern,
    enabling hardware prefetch + compiler pipelining.  Each iteration advances
    K/V pointer backward by 1; L2 cache handles the BLOCK_N-1 element overlap.
    tl.static_range(33) unrolls the loop for the compiler.
  Forward — sparse offsets (δ=48..1536, 11 iters):
    Individual HBM loads, same as V1.  tl.static_range embeds deltas as
    compile-time constants (no OFFSETS_PTR tensor needed).
  Backward: unchanged from V1 — LSE recompute (no alpha), key-centric dkdv.

Memory footprint (B=128, H=8, N=2047, HD=32):
  Naive PyTorch K_all+V_all:  ~27.5 GB/step (5 DSQG layers)
  V2 saved state:              ~142 MB  (LSE 8.4 MB + out 134 MB)

Testing:
  .venv/bin/python3 ~/.openclaw/workspace/kernels/dsqg_attention_v2.py
  .venv/bin/python3 ~/.openclaw/workspace/kernels/dsqg_attention_v2.py bench

Integration (one line in training script after imports):
  from dsqg_attention_v2 import DSQGAttentionN_Fused as DSQGAttentionN
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Suppress Triton's tl.advance reassignment warning (false positive in static_range loops)
warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Offset constants — must match _COND_N_OFFSETS in all training scripts
# ─────────────────────────────────────────────────────────────────────────────
_SPARSE_LIST   = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
ALL_OFFSETS    = list(range(33)) + _SPARSE_LIST
assert len(ALL_OFFSETS) == 44

# ─────────────────────────────────────────────────────────────────────────────

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd(
    Q, K, V, POS_BIAS, OUT, LSE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    H: tl.constexpr, N,               HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    """
    Grid: (B*H, cdiv(N, BLOCK_N))

    Phase 1 — Consecutive δ=0..32 (33 iters, tl.static_range, unrolled):
      tl.make_block_ptr + tl.advance tells Triton about the sequential
      access pattern.  Hardware prefetch + L2 reuse on BLOCK_N-1 overlap.

    Phase 2 — Sparse δ=48..1536 (11 iters, constexpr deltas):
      Individual HBM loads.

    Saves out [B,H,N,HD] bf16 + LSE [B,H,N] fp32.  No alpha tensor.
    """
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

    # Load Q tile
    q = tl.load(qb + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    # Block pointers for consecutive block: start at n0 (delta=0 key position),
    # then advance backward by 1 each iteration so delta=d loads [n0-d, n0-d+BLOCK_N)
    k_bptr = tl.make_block_ptr(
        base=kb, shape=(N, HD), strides=(stride_kn, stride_kd),
        offsets=(n0, 0), block_shape=(BLOCK_N, BLOCK_HD), order=(1, 0))
    v_bptr = tl.make_block_ptr(
        base=vb, shape=(N, HD), strides=(stride_vn, stride_vd),
        offsets=(n0, 0), block_shape=(BLOCK_N, BLOCK_HD), order=(1, 0))

    mi  = tl.full([BLOCK_N], float('-inf'), tl.float32)
    li  = tl.zeros([BLOCK_N], tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # ── Phase 1: Consecutive δ=0..32 ──────────────────────────────────────────
    for d in tl.static_range(33):
        kt  = tl.load(k_bptr, boundary_check=(0, 1), padding_option='zero')
        vt  = tl.load(v_bptr, boundary_check=(0, 1), padding_option='zero')
        val = (ns - d >= 0) & nm

        s   = tl.sum(q * kt.to(tl.float32), axis=1) * sc
        s  += tl.load(POS_BIAS + d * stride_pbi + h * stride_pbh)
        s   = tl.where(val, s, float('-inf'))

        mn  = tl.maximum(mi, s)
        cor = tl.exp(mi - mn)
        p   = tl.exp(s  - mn)
        li  = li * cor + p
        acc = acc * cor[:,None] + p[:,None] * vt.to(tl.float32)
        mi  = mn

        k_bptr = tl.advance(k_bptr, (-1, 0))
        v_bptr = tl.advance(v_bptr, (-1, 0))

    # ── Phase 2: Sparse δ=48..1536 ────────────────────────────────────────────
    for si in tl.static_range(11):
        sd = (48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536)[si]
        pbi = 33 + si

        kp  = ns - sd
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0)

        s   = tl.sum(q * kt.to(tl.float32), axis=1) * sc
        s  += tl.load(POS_BIAS + pbi * stride_pbi + h * stride_pbh)
        s   = tl.where(val, s, float('-inf'))

        mn  = tl.maximum(mi, s)
        cor = tl.exp(mi - mn)
        p   = tl.exp(s  - mn)
        li  = li * cor + p
        acc = acc * cor[:,None] + p[:,None] * vt.to(tl.float32)
        mi  = mn

    # Normalize + store
    ls  = tl.where(li > 0.0, li, 1.0)
    out = acc / ls[:,None]
    lse = mi + tl.log(ls)

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             out.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# Backward: D[n] = dot(dout[n], out[n])
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_D(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_db,  stride_dh,  stride_dn,
    H: tl.constexpr, N,               HD: tl.constexpr,
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
# Backward: dQ (query-centric, recomputes alpha from LSE)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq(
    Q, K, V, PB, DO, O, LSE, Dv, DQ, DPB,
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
    H: tl.constexpr, N,               HD: tl.constexpr,
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
    dob = DO + b*stride_dob+ h*stride_doh

    q   = tl.load(qb  + ns[:,None]*stride_qn  + ds[None,:]*stride_qd,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    do  = tl.load(dob + ns[:,None]*stride_don  + ds[None,:]*stride_dod,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b*stride_lb + h*stride_lh + ns*stride_ln, mask=nm, other=0.0)
    Dval= tl.load(Dv  + b*stride_Db + h*stride_Dh + ns*stride_Dn, mask=nm, other=0.0)
    dq  = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in tl.static_range(44):
        delta = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                 20,21,22,23,24,25,26,27,28,29,30,31,32,
                 48,64,96,128,192,256,384,512,768,1024,1536)[i]
        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s     = tl.where(val, s, float('-inf'))

        alpha = tl.where(val, tl.exp(s - lse), 0.0)
        ds_v  = alpha * (tl.sum(do * vt, axis=1) - Dval)
        dq   += ds_v[:,None] * kt * sc

        tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                      tl.sum(tl.where(val, ds_v, 0.0), axis=0))

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])


# ─────────────────────────────────────────────────────────────────────────────
# Backward: dK/dV (key-centric, NO atomics)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv(
    Q, K, V, PB, DO, LSE, Dv, DK, DV,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb,  stride_lh,  stride_ln,
    stride_Db,  stride_Dh,  stride_Dn,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    H: tl.constexpr, N,               HD: tl.constexpr,
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
    dob = DO + b*stride_dob+ h*stride_doh

    kt  = tl.load(kb + ms[:,None]*stride_kn + ds[None,:]*stride_kd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    vt  = tl.load(vb + ms[:,None]*stride_vn + ds[None,:]*stride_vd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    dk  = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv  = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)

    for i in tl.static_range(44):
        delta = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                 20,21,22,23,24,25,26,27,28,29,30,31,32,
                 48,64,96,128,192,256,384,512,768,1024,1536)[i]
        np_  = ms + delta
        val  = (np_ < N) & mm

        qn   = tl.load(qb  + np_[:,None]*stride_qn  + ds[None,:]*stride_qd,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don  = tl.load(dob + np_[:,None]*stride_don  + ds[None,:]*stride_dod,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn   = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        s    = tl.sum(qn * kt, axis=1) * sc
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
# Autograd wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16, f"forward expects bfloat16 after wrapper cast, got {q.dtype}"
        assert pos_bias.shape == (44, H)
        BLOCK_N  = 128 if HD <= 64 else 64
        BLOCK_HD = _next_pow2(HD)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd[g](
            q, k, v, pos_bias, out, lse,
            q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
            k.stride(0),   k.stride(1),   k.stride(2),   k.stride(3),
            v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
        )
        ctx.save_for_backward(q, k, v, pos_bias, out, lse)
        ctx.BLOCK_N  = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pb, out, lse = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD = ctx.BLOCK_N, ctx.BLOCK_HD
        dout = dout.contiguous()

        D   = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BN))
        _compute_D[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        dq  = torch.empty_like(q)
        dpb = torch.zeros_like(pb)
        _bwd_dq[g](
            q, k, v, pb, dout, out, lse, D, dq, dpb,
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
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        dk  = torch.empty_like(k)
        dv  = torch.empty_like(v)
        _bwd_dkdv[g](
            q, k, v, pb, dout, lse, D, dk, dv,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dk.stride(0),   dk.stride(1),   dk.stride(2),   dk.stride(3),
            dv.stride(0),   dv.stride(1),   dv.stride(2),   dv.stride(3),
            pb.stride(0),   pb.stride(1),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
        )
        return dq, dk, dv, dpb


def dsqg_attention_v2(q, k, v, pos_bias):
    """
    q, k, v:   [B, H, N, HD]  bfloat16 (or float32 — auto-cast)
    pos_bias:  [44, H]        float32
    Returns:   [B, H, N, HD]  same dtype as input
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFn.apply(q, k, v, pos_bias)
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in module
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionN_Fused(nn.Module):
    """
    Drop-in for DSQGAttentionN.  One integration line in training script:
        from dsqg_attention_v2 import DSQGAttentionN_Fused as DSQGAttentionN
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        if offsets is None:
            offsets = ALL_OFFSETS
        assert list(offsets) == ALL_OFFSETS, f"Kernel requires {ALL_OFFSETS}; got {list(offsets)}"
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        out      = dsqg_attention_v2(q, k, v, self.pos_bias)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
        return {'pos_bias_abs_mean': pb.abs().mean().item(),
                'pos_bias_abs_max':  pb.abs().max().item(),
                'pos_bias_mean_per_head': pb.mean(0).tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation
# ─────────────────────────────────────────────────────────────────────────────

def _reference(q, k, v, pos_bias):
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
    s   = s.masked_fill((ni[:,None] < off[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a   = F.softmax(s, dim=-1)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    print("=" * 64)
    print("DSQG V2 — Correctness Tests")
    print("=" * 64)
    cfgs = [
        (1,  8,   64, 32, "tiny"),
        (2,  8,  512, 32, "mid (all offsets)"),
        (4,  8, 2047, 32, "13M shape"),
        (4,  8, 2047, 80, "85M shape (HD=80)"),
    ]
    ok_all = True
    for B, H, N, HD, lbl in cfgs:
        torch.manual_seed(42)
        q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44,H, device=device, dtype=torch.float32) * 0.5

        ref = _reference(q.detach(), k.detach(), v.detach(), pb)
        out = dsqg_attention_v2(q.detach().clone(), k.detach().clone(), v.detach().clone(), pb)
        fe  = (ref.float() - out.float()).abs().max().item()

        qr,kr,vr = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        _reference(qr,kr,vr,pb).sum().backward()
        dqr,dkr,dvr = qr.grad.clone(), kr.grad.clone(), vr.grad.clone()

        qt,kt2,vt = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        dsqg_attention_v2(qt,kt2,vt,pb).sum().backward()
        de = max((qt.grad.float()-dqr.float()).abs().max().item(),
                 (kt2.grad.float()-dkr.float()).abs().max().item(),
                 (vt.grad.float()-dvr.float()).abs().max().item())

        ok = max(fe, de) < 0.05
        if not ok: ok_all = False
        print(f"  {lbl:24s}  fwd={fe:.4f}  bwd={de:.4f}  {'PASS ✓' if ok else 'FAIL ✗'}")

    print("=" * 64)
    print(f"{'ALL PASSED ✓' if ok_all else 'SOME FAILED ✗'}")
    return ok_all


def run_benchmark(device='cuda', warmup=5, iters=20):
    import time
    print("\n" + "=" * 64 + "\nDSQG V2 Benchmark\n" + "=" * 64)
    for B,H,N,HD,lbl in [
        (64,  8, 2047, 32, "13M @ B=64  (matches current training)"),
        (128, 8, 2047, 32, "13M @ B=128"),
        (64,  8, 2047, 80, "85M @ B=64"),
    ]:
        toks = B * N
        torch.manual_seed(0)
        q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        pb = torch.randn(44,H, device=device, dtype=torch.float32) * 0.5

        for _ in range(warmup):
            dsqg_attention_v2(q,k,v,pb).sum().backward()
            q.grad = k.grad = v.grad = None

        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            dsqg_attention_v2(q,k,v,pb).sum().backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize(); el = time.perf_counter() - t0

        ms  = el / iters * 1000
        tps = toks / (el / iters)
        torch.cuda.reset_peak_memory_stats()
        q.grad = k.grad = v.grad = None; torch.cuda.empty_cache()
        dsqg_attention_v2(q,k,v,pb).sum().backward()
        mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  {lbl}")
        print(f"  {toks:,} tok/iter  fwd+bwd: {ms:.1f}ms  → {tps/1e6:.2f}M tok/s  peak: {mb:.0f}MB")
    print("\n" + "=" * 64)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "bench":
        run_benchmark()
    else:
        run_tests()
