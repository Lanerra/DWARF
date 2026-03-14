"""
DSQG Attention d41-J16D — V3 kernel with J16D relay-optimal offsets
====================================================================

Based on dsqg_attention_d41s3 (V3 Q-Weighted Scale Gains) but with the J16D
relay-optimal offset set (16 offsets, max_hops=2) instead of the original 45.

ALL_OFFSETS = [1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

Key changes vs d41s3:
- ALL_OFFSETS: 45 → 16 (J16D relay-optimal set)
- Forward: two-phase (42 consecutive + 3 sparse) → single static_range(16)
  All offsets use direct indexed K/V loads (no block-pointer advance trick).
- Backward dq/dkdv: static_range(16) with J16D offset tuple
- pos_bias shape: [16, H]  (was [45, H])
- scale_embed shape: [16, HD]  (was [45, HD])
- NaN guards on online-softmax (from v6): handles n=0 where no offsets are valid

Usage:
  from dsqg_attention_d41j16d import dsqg_attention_d41j16d

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_d41j16d.py
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

ALL_OFFSETS = [1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
J = len(ALL_OFFSETS)
assert J == 16
MAX_DELTA = max(ALL_OFFSETS)

_OFFSET_TUPLE = (1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_d41j16d(
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

    for i in tl.static_range(16):
        delta = (1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)

        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None, :], axis=1) * sc

        s     = tl.where(val, s, float('-inf'))

        mn    = tl.maximum(mi, s)
        cor   = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p     = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        li    = li * cor + p
        acc   = acc * cor[:,None] + p[:,None] * vt
        mi    = mn

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
def _compute_D_d41j16d(
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
def _bwd_dq_d41j16d(
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

    for i in tl.static_range(16):
        delta = (1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
        kp  = ns - delta
        val = (kp >= 0) & nm

        kt  = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt  = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                      mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
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
# Backward: dK/dV — must include SE in score for correct alpha
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_d41j16d(
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

    for i in tl.static_range(16):
        delta = (1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
        np_  = ms + delta
        val  = (np_ < N) & mm

        qn   = tl.load(qb  + np_[:,None]*stride_qn  + ds[None,:]*stride_qd,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don  = tl.load(dob + np_[:,None]*stride_don  + ds[None,:]*stride_dod,
                       mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn   = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
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

class _DSQGFnD41J16D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (16, H),  f"pos_bias: {pos_bias.shape}"
        assert scale_embed.shape == (16, HD), f"scale_embed: {scale_embed.shape}"
        BLOCK_N  = 128 if HD <= 64 else 64
        BLOCK_HD = _next_pow2(HD)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_d41j16d[g](
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
        _compute_D_d41j16d[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        dq   = torch.empty_like(q)
        dpb  = torch.zeros_like(pb)
        dse  = torch.zeros_like(se)
        _bwd_dq_d41j16d[g](
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
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
        )

        dk  = torch.empty_like(k)
        dv  = torch.empty_like(v)
        _bwd_dkdv_d41j16d[g](
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


def dsqg_attention_d41j16d(q, k, v, pos_bias, scale_embed):
    """
    q, k, v:      [B, H, N, HD]  bfloat16
    pos_bias:     [16, H]         float32
    scale_embed:  [16, HD]        float32
    Returns:      [B, H, N, HD]  same dtype as input
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnD41J16D.apply(q, k, v, pos_bias.float(), scale_embed.float())
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in module
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionD41J16D(nn.Module):
    """
    DSQG attention with J16D relay-optimal offsets + Q-weighted scale gains.

    Parameters:
      pos_bias     [16, H]   global learned frequency prior
      scale_embed  [16, HD]  Q-dynamic matched-filter embeddings (zero-init)
      if_gain      [H]       per-head IF amplifier gain (1.0-init)

    Compatible with Huygens K/V injection via forward(x, kv_inject=...).
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
        assert len(offsets) == 16
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(16, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

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

        out = dsqg_attention_d41j16d(q, k, v, self.pos_bias, self.scale_embed)
        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()
            se   = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean':   se.abs().mean().item(),
            'scale_embed_abs_max':    se.abs().max().item(),
            'if_gain':                gain.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch — for correctness testing)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_d41j16d(q, k, v, pos_bias, scale_embed):
    B, H, N, HD = q.shape
    sc  = HD ** -0.5
    off = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)
    kp  = F.pad(k.float(), (0, 0, MAX_DELTA, 0))
    vp  = F.pad(v.float(), (0, 0, MAX_DELTA, 0))
    ni  = torch.arange(N, device=q.device)
    gi  = MAX_DELTA - off[None,:] + ni[:,None]
    Ka  = kp[:, :, gi, :]
    Va  = vp[:, :, gi, :]
    s   = (q.float().unsqueeze(3) * Ka).sum(-1) * sc
    s  += pos_bias.T[None,:,None,:]
    q_dyn = (q.float().unsqueeze(3) * scale_embed[None,None,:,:]).sum(-1) * sc
    s  += q_dyn
    s   = s.masked_fill((ni[:,None] < off[None,:]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a   = F.softmax(s, dim=-1)
    a   = torch.nan_to_num(a, nan=0.0)
    return (a.unsqueeze(-1) * Va).sum(3).to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    print("=" * 64)
    print("DSQG d41-J16D — Correctness Tests (V3 kernel + J16D offsets)")
    print("=" * 64)
    cfgs = [
        (1,  8,   64, 32, "tiny"),
        (2,  8,  512, 32, "mid (all offsets active)"),
        (4,  8, 2047, 32, "14M shape"),
    ]
    ok_all = True
    for B, H, N, HD, lbl in cfgs:
        torch.manual_seed(42)
        q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(16,H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(16,HD, device=device, dtype=torch.float32) * 0.05

        ref = _reference_d41j16d(q.detach(), k.detach(), v.detach(), pb, se)
        out = dsqg_attention_d41j16d(q.detach().clone(), k.detach().clone(),
                                     v.detach().clone(), pb, se)
        fe  = (ref.float() - out.float()).abs().max().item()

        qr,kr,vr = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        se_r = se.clone().detach().requires_grad_(True)
        _reference_d41j16d(qr,kr,vr,pb,se_r).sum().backward()
        dqr = qr.grad.clone(); dkr = kr.grad.clone(); dvr = vr.grad.clone()
        dser = se_r.grad.clone()

        qt,kt2,vt = [t.clone().detach().requires_grad_(True) for t in (q,k,v)]
        se_t = se.clone().detach().requires_grad_(True)
        dsqg_attention_d41j16d(qt,kt2,vt,pb,se_t).sum().backward()
        de_qkv = max((qt.grad.float()-dqr.float()).abs().max().item(),
                     (kt2.grad.float()-dkr.float()).abs().max().item(),
                     (vt.grad.float()-dvr.float()).abs().max().item())
        de_se  = (se_t.grad.float()-dser.float()).abs().max().item()

        ok = max(fe, de_qkv, de_se) < 0.05
        if not ok: ok_all = False
        print(f"  {lbl:24s}  fwd={fe:.4f}  bwd_qkv={de_qkv:.4f}  bwd_se={de_se:.4f}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")

    print()
    print("  Zero SE init check (output should be non-degenerate):")
    B,H,N,HD = 2,8,128,32
    torch.manual_seed(7)
    q  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    k  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    v  = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    pb = torch.randn(16,H, device=device, dtype=torch.float32) * 0.5
    se_zero = torch.zeros(16,HD, device=device, dtype=torch.float32)
    out_z = dsqg_attention_d41j16d(q.clone(), k.clone(), v.clone(), pb, se_zero)
    ref_z = _reference_d41j16d(q, k, v, pb, se_zero)
    diff  = (out_z.float() - ref_z.float()).abs().max().item()
    ok_z  = diff < 1e-3
    if not ok_z: ok_all = False
    print(f"  {'zero SE triton vs ref':24s}  max_diff={diff:.6f}  {'PASS ✓' if ok_z else 'FAIL ✗'}")

    print("=" * 64)
    print(f"{'ALL PASSED ✓' if ok_all else 'SOME FAILED ✗'}")
    return ok_all


if __name__ == "__main__":
    run_tests()
