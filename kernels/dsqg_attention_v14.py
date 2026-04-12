"""
DSQG Attention V14 — K-tile coalesced, 2-pass forward for reduced HBM bandwidth
================================================================================

Preserves exact numerical equivalence with V8 while reducing effective HBM
K/V loads via a two-pass forward structure:
  Pass 1: Compute all J=24 scores (K loads only — L1 stays warm with K tiles)
  Pass 2: Softmax over offset dimension
  Pass 3: Weighted V accumulation (V loads only — L1 stays warm with V tiles)

By separating K and V loads into distinct passes, L1 cache thrashing between
K and V tiles is eliminated, yielding ~6x effective HBM load reduction for
small offsets that share the same K/V tiles.

Backward: reuses V8's backward kernels (same numerical path, same gradients).

    ALL_OFFSETS = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]
    J=24, J_SMALL=14, J_LARGE=10, R_PLANES=2

Usage:
  from dsqg_attention_v14 import dsqg_attention_v14, DSQGAttentionV14

Testing:
  cd /home/dlewis3/Desktop/AI/DWARF && \\
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u kernels/test_v14.py
"""

import os
import sys
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
from dsqg_attention_v8_h100 import (
    ALL_OFFSETS, J, J_SMALL, J_LARGE, R_PLANES, MAX_DELTA,
    _next_pow2, npci_rotate, _reference_v8,
    _compute_D_v8, _bwd_dq_v8, _bwd_dkdv_v8,
)

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

BLOCK_J = 32


@triton.jit
def _fwd_v14(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_lb,  stride_lh,  stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb,  stride_yh,  stride_yn,
    stride_zb,  stride_zh,  stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H;           h   = bh % H
    n0  = blk * BLOCK_N

    ns  = n0 + tl.arange(0, BLOCK_N)
    nm  = ns < N
    sc  = 1.0 / (HD ** 0.5)
    ds  = tl.arange(0, BLOCK_HD)
    dm  = ds < HD
    ji  = tl.arange(0, BLOCK_J)

    qb  = Q + b * stride_qb + h * stride_qh
    kb  = K + b * stride_kb + h * stride_kh
    vb  = V + b * stride_vb + h * stride_vh
    yb  = Y_PRE + b * stride_yb + h * stride_yh
    zb  = Z_PRE + b * stride_zb + h * stride_zh

    q   = tl.load(qb + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    y0  = tl.load(yb + ns*stride_yn + 0, mask=nm, other=0.0)
    y1  = tl.load(yb + ns*stride_yn + 1, mask=nm, other=0.0)

    col0 = (ds == 0);  col1 = (ds == 1)
    col2 = (ds == 2);  col3 = (ds == 3)
    f0   = tl.where(col0, 1.0, 0.0)
    f1   = tl.where(col1, 1.0, 0.0)
    f2   = tl.where(col2, 1.0, 0.0)
    f3   = tl.where(col3, 1.0, 0.0)

    # ─── Pass 1: Score computation (K loads only — no V in L1) ───
    scores = tl.full([BLOCK_N, BLOCK_J], float('-inf'), tl.float32)

    for i in tl.static_range(24):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
                 48, 64, 96, 192, 384, 512, 768, 1024)[i]
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        scores = tl.where(ji[None,:] == i, s[:,None], scores)

    # ─── Pass 2: Softmax over offset dimension ───
    valid_j = (ji[None,:] < 24)
    mi  = tl.max(tl.where(valid_j, scores, float('-inf')), axis=1)
    safe_mi = tl.where(mi > float('-inf'), mi, 0.0)
    e   = tl.exp(scores - safe_mi[:,None])
    e   = tl.where(valid_j, e, 0.0)
    li  = tl.sum(e, axis=1)
    ls  = tl.where(li > 0.0, li, 1.0)
    lse = tl.where(li > 0.0, safe_mi + tl.log(ls), mi)
    alpha = e / ls[:,None]

    # ─── Pass 3: Weighted V accumulation (V loads only — no K in L1) ───
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in tl.static_range(24):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
                 48, 64, 96, 192, 384, 512, 768, 1024)[i]
        kp    = ns - delta
        val   = (kp >= 0) & nm

        alpha_i = tl.sum(alpha * tl.where(ji[None,:] == i, 1.0, 0.0), axis=1)

        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        if i < 14:
            acc += alpha_i[:,None] * vt
        else:
            pi  = i - 14
            z0  = tl.load(zb + kp*stride_zn + 0, mask=val, other=0.0)
            z1  = tl.load(zb + kp*stride_zn + 1, mask=val, other=0.0)

            pb0 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0 * z0
            theta1 = pb1 + pg1 * y1 * z1

            cos0 = tl.cos(theta0);  sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1);  sin1 = tl.sin(theta1)

            v0  = tl.sum(vt * f0[None,:], axis=1)
            v1  = tl.sum(vt * f1[None,:], axis=1)
            vt  = tl.where(col0[None,:], (cos0 * v0 - sin0 * v1)[:,None], vt)
            vt  = tl.where(col1[None,:], (sin0 * v0 + cos0 * v1)[:,None], vt)

            v2  = tl.sum(vt * f2[None,:], axis=1)
            v3  = tl.sum(vt * f3[None,:], axis=1)
            vt  = tl.where(col2[None,:], (cos1 * v2 - sin1 * v3)[:,None], vt)
            vt  = tl.where(col3[None,:], (sin1 * v2 + cos1 * v3)[:,None], vt)

            acc += alpha_i[:,None] * vt

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             acc.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


class _DSQGFnV14(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (J, H)
        assert scale_embed.shape == (J, HD)
        assert phase_base.shape  == (J_LARGE, H, R_PLANES)
        assert phase_gain.shape  == (J_LARGE, H, R_PLANES)
        assert y_pre.shape       == (B, H, N, R_PLANES)
        assert z_pre.shape       == (B, H, N, R_PLANES)

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BN, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BN, _num_warps, _num_stages = 64,  8, 2
            else:       BN, _num_warps, _num_stages = 64,  4, 2
        elif HD <= 128:
            if _sm90:   BN, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BN, _num_warps, _num_stages = 64,  4, 2
            else:       BN, _num_warps, _num_stages = 32,  4, 2
        elif HD <= 256:
            if _sm90:   BN, _num_warps, _num_stages = 32,  4, 3
            elif _sm89: BN, _num_warps, _num_stages = 32,  4, 2
            else:       BN, _num_warps, _num_stages = 16,  4, 2
        else:
            if _sm90:   BN, _num_warps, _num_stages = 16,  4, 3
            elif _sm89: BN, _num_warps, _num_stages = 16,  4, 2
            else:       BN, _num_warps, _num_stages = 8,   4, 2

        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BN))

        _fwd_v14[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            pos_bias.stride(0),    pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            phase_base.stride(0),  phase_base.stride(1),
            phase_gain.stride(0),  phase_gain.stride(1),
            y_pre.stride(0),       y_pre.stride(1),       y_pre.stride(2),
            z_pre.stride(0),       z_pre.stride(1),       z_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BLOCK_HD, BLOCK_J=BLOCK_J,
            num_warps=_num_warps, num_stages=_num_stages,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre, out, lse)
        ctx.BLOCK_N    = BN
        ctx.BLOCK_HD   = BLOCK_HD
        ctx.num_warps  = _num_warps
        ctx.num_stages = _num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD, NW, NS = ctx.BLOCK_N, ctx.BLOCK_HD, ctx.num_warps, ctx.num_stages
        dout = dout.contiguous()

        D  = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g  = (B * H, triton.cdiv(N, BN))

        _compute_D_v8[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0),    D.stride(1),    D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = (N + BN - 1) // BN
        _dev     = q.device

        dq      = torch.zeros_like(q)
        dy_pre  = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B * H, blocks_n, J,       device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, J * HD,  device=_dev, dtype=torch.float32)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            blocks_n * J,   J,
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            blocks_n*J*HD,  J*HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, J).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, J, HD).sum(dim=(0, 1, 2)).contiguous()

        dk     = torch.zeros_like(k)
        dv     = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)
        phase_base_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        phase_gain_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        stride_buf_bh  = blocks_n * J_LARGE * 2
        stride_buf_blk = J_LARGE * 2

        _bwd_dkdv_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
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
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            stride_buf_bh, stride_buf_blk,
            dz_pre.stride(0),     dz_pre.stride(1),     dz_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, J_LARGE, 2).sum(dim=(0, 2))
            return r.permute(1, 0, 2).contiguous()
        d_phase_base = _reduce_phase_buf(phase_base_buf)
        d_phase_gain = _reduce_phase_buf(phase_gain_buf)

        return (dq, dk, dv,
                dpb, dse, d_phase_base, d_phase_gain, dy_pre, dz_pre)


def dsqg_attention_v14(q, k, v, pos_bias, scale_embed,
                       phase_base, phase_gain, y_pre, z_pre):
    """
    Drop-in replacement for dsqg_attention_v8 with K-tile coalesced forward.

    q, k, v:       [B, H, N, HD]   bfloat16
    pos_bias:      [24, H]          float32
    scale_embed:   [24, HD]         float32
    phase_base:    [10, H, 2]       float32
    phase_gain:    [10, H, 2]       float32
    y_pre:         [B, H, N, 2]    float32
    z_pre:         [B, H, N, 2]    float32
    Returns:       [B, H, N, HD]   bfloat16
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    original_dtype = q.dtype
    if original_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV14.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
    )
    return out if original_dtype == torch.bfloat16 else out.to(original_dtype)


class DSQGAttentionV14(nn.Module):
    """
    DSQG V14: K-tile coalesced forward, same architecture as V8.
    Drop-in replacement for DSQGAttentionV8_H100 with faster forward pass.
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim
        assert HD >= 4

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

        self.phase_base   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))
        self.phase_gain   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes   = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        sc    = HD ** -0.5
        y_pre = torch.einsum('bhnd,rd->bhnr',
                             q, self.query_probes.to(q.dtype)).mul(sc).float().contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr',
                             k, self.key_probes.to(k.dtype)).mul(sc).float().contiguous()

        out = dsqg_attention_v14(q, k, v,
                                 self.pos_bias, self.scale_embed,
                                 self.phase_base, self.phase_gain,
                                 y_pre, z_pre)

        out      = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))
