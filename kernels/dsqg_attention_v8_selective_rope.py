"""
DSQG Attention V8 Selective RoPE — RoPE rotation for distal (J_LARGE) offsets
=============================================================================

Wraps V8-dynamic-J Triton kernels with Selective RoPE:
  J_SMALL offsets (δ ≤ 28): no rotation on Q, K
  J_LARGE offsets (δ ≥ 48): RoPE rotation on Q and K before kernel

RoPE frequencies: freq_i = 1 / (10000 ** (2i / HD))
RoPE angles: θ_i(n) = n * freq_i (absolute position)
Net relative rotation in QK dot product: δ * freq_i

For mixed groups (both small and large offsets), two kernel passes
are merged via log-sum-exp weighting.

pos_bias: UNCHANGED.  MOVT on values: UNCHANGED.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_dynamic_j import (
    _fwd_v8, _compute_D_v8, _bwd_dq_v8, _bwd_dkdv_v8,
    npci_rotate, R_PLANES, _next_pow2,
)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embedding (pure PyTorch, differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_rope(x, delta):
    """
    Apply Selective RoPE using offset distance as position argument.
    x: [B, H, N, HD] → [B, H, N, HD], same dtype.
    freq_i = 1/(10000^(2i/HD)), θ_i = delta * freq_i (fixed per offset, not per token).
    Using offset distance (not absolute token position) keeps angles small and
    gradients well-conditioned in BF16.
    """
    B, H, N, HD = x.shape
    half_dim = HD // 2
    x_float = x.float()
    indices = torch.arange(half_dim, device=x.device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (2.0 * indices / HD))
    angles = delta * freqs          # shape: (half_dim,) — same angle for all positions
    cos_a = torch.cos(angles).view(1, 1, 1, half_dim)
    sin_a = torch.sin(angles).view(1, 1, 1, half_dim)
    x_even = x_float[..., 0::2]
    x_odd  = x_float[..., 1::2]
    rotated = torch.empty_like(x_float)
    rotated[..., 0::2] = x_even * cos_a - x_odd * sin_a
    rotated[..., 1::2] = x_even * sin_a + x_odd * cos_a
    return rotated.to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function — returns (out, lse) for log-sum-exp merging
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnGroupedWithLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                offsets_dev, j_val, j_small, j_large):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape == (j_val, H)
        assert scale_embed.shape == (j_val, HD)
        assert phase_base.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert phase_gain.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert y_pre.shape == (B, H, N, R_PLANES)
        assert z_pre.shape == (B, H, N, R_PLANES)

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 8, 2
            else:       BLOCK_N, _nw, _ns = 64, 4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 4, 2
            else:       BLOCK_N, _nw, _ns = 32, 4, 2
        elif HD <= 256:
            if _sm90:   BLOCK_N, _nw, _ns = 32, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 32, 4, 2
            else:       BLOCK_N, _nw, _ns = 16, 4, 2
        else:
            if _sm90:   BLOCK_N, _nw, _ns = 16, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 16, 4, 2
            else:       BLOCK_N, _nw, _ns = 8, 4, 2

        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_v8[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=_nw, num_stages=_ns,
        )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev)
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _nw
        ctx.num_stages = _ns
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse, offsets_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD = ctx.BLOCK_N, ctx.BLOCK_HD
        NW, NS = ctx.num_warps, ctx.num_stages
        j_val, j_small, j_large = ctx.j_val, ctx.j_small, ctx.j_large
        dout = dout.contiguous()

        D = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BN))

        _compute_D_v8[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = (N + BN - 1) // BN
        _dev = q.device

        dq = torch.zeros_like(q)
        dy_pre = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B * H, blocks_n, j_val,
                              device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, j_val * HD,
                              device=_dev, dtype=torch.float32)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            blocks_n * j_val, j_val,
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            blocks_n * j_val * HD, j_val * HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(
            dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(
            dim=(0, 1, 2)).contiguous()

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)

        if j_large > 0:
            phase_base_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                         device=_dev, dtype=torch.float32)
            phase_gain_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                         device=_dev, dtype=torch.float32)
        else:
            phase_base_buf = torch.empty(B * H, blocks_n, 1,
                                         device=_dev, dtype=torch.float32)
            phase_gain_buf = torch.empty(B * H, blocks_n, 1,
                                         device=_dev, dtype=torch.float32)

        stride_buf_bh = blocks_n * max(j_large, 1) * 2
        stride_buf_blk = max(j_large, 1) * 2

        _bwd_dkdv_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            stride_buf_bh, stride_buf_blk,
            dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )

        if j_large > 0:
            def _reduce_phase_buf(buf):
                r = buf.view(B, H, blocks_n, j_large, 2).sum(dim=(0, 2))
                return r.permute(1, 0, 2).contiguous()
            d_phase_base = _reduce_phase_buf(phase_base_buf)
            d_phase_gain = _reduce_phase_buf(phase_gain_buf)
        else:
            d_phase_base = torch.zeros_like(phase_base)
            d_phase_gain = torch.zeros_like(phase_gain)

        return (dq, dk, dv, dpb, dse, d_phase_base, d_phase_gain,
                dy_pre, dz_pre, None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Functional wrappers
# ─────────────────────────────────────────────────────────────────────────────

def dsqg_attention_grouped_with_lse(q, k, v, pos_bias, scale_embed,
                                     phase_base, phase_gain, y_pre, z_pre,
                                     offsets_dev, j_val, j_small, j_large):
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out, lse = _DSQGFnGroupedWithLSE.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
        offsets_dev, j_val, j_small, j_large,
    )
    if orig != torch.bfloat16:
        out = out.to(orig)
    return out, lse


def dsqg_attention_grouped(q, k, v, pos_bias, scale_embed,
                            phase_base, phase_gain, y_pre, z_pre,
                            offsets_dev, j_val, j_small, j_large):
    out, _ = dsqg_attention_grouped_with_lse(
        q, k, v, pos_bias, scale_embed,
        phase_base, phase_gain, y_pre, z_pre,
        offsets_dev, j_val, j_small, j_large)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Module — DSQGAttentionGroupedSelectiveRoPE
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionGroupedSelectiveRoPE(nn.Module):
    """
    DSQG attention with Selective RoPE for distal offsets.
    Same constructor signature as DSQGAttentionGrouped.

    J_SMALL offsets (δ ≤ 28): no RoPE rotation on Q, K.
    J_LARGE offsets (δ ≥ 48): RoPE rotation on Q and K.
    pos_bias, MOVT on values: unchanged.

    For mixed groups, two kernel passes are merged via log-sum-exp.
    """
    def __init__(self, embedding_dim, num_heads, offsets, j_small, j_large,
                 seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        HD = self.head_dim
        j_val = len(offsets)
        self.j_val = j_val
        self.j_small = j_small
        self.j_large = j_large
        assert HD >= 4

        self.register_buffer(
            'offsets_dev',
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

        small_offsets = [d for d in offsets if d <= 28]
        large_offsets = [d for d in offsets if d >= 48]
        self.register_buffer(
            'offsets_dev_small',
            torch.tensor(small_offsets if small_offsets else [0],
                         dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'offsets_dev_large',
            torch.tensor(large_offsets if large_offsets else [0],
                         dtype=torch.int32),
            persistent=False,
        )

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets],
                                  dtype=torch.float32)
        self.pos_bias = nn.Parameter(
            -delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(j_val, HD))
        self.if_gain = nn.Parameter(torch.ones(num_heads))

        self.phase_base = nn.Parameter(
            torch.zeros(max(j_large, 1), num_heads, R_PLANES))
        # SO(2) Lie group parameterization: (x, y) on unit circle → no 2π wrap-arounds.
        # init=(1,0) → identity rotation. Optimizer works in unbounded Euclidean space.
        pgv = torch.zeros(max(j_large, 1), num_heads, R_PLANES, 2)
        pgv[..., 0] = 1.0   # cos component → identity
        pgv[..., 1] = 0.0   # sin component
        self.phase_gain_vec = nn.Parameter(pgv)
        # Keep phase_gain as a zero tensor for backward-compat with kernel signature
        self.register_buffer('phase_gain',
            torch.zeros(max(j_large, 1), num_heads, R_PLANES), persistent=False)

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        sc = HD ** -0.5
        y_pre = torch.einsum(
            'bhnd,rd->bhnr', q, self.query_probes.to(q.dtype)
        ).mul(sc).float().contiguous()
        z_pre = torch.einsum(
            'bhnd,rd->bhnr', k, self.key_probes.to(k.dtype)
        ).mul(sc).float().contiguous()

        if self.j_small > 0 and self.j_large > 0:
            pb_small = self.pos_bias[:self.j_small]
            se_small = self.scale_embed[:self.j_small]
            pb_large = self.pos_bias[self.j_small:]
            se_large = self.scale_embed[self.j_small:]
            dummy_phase = torch.zeros(
                1, H, R_PLANES, device=x.device, dtype=torch.float32)

            out_s, lse_s = dsqg_attention_grouped_with_lse(
                q, k, v, pb_small, se_small,
                dummy_phase, dummy_phase,
                y_pre, z_pre,
                self.offsets_dev_small, self.j_small, self.j_small, 0)

            # SO(2): derive phase_gain from unit-circle parameterization
            pgv = self.phase_gain_vec
            pgv_norm = pgv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pgv_unit = pgv / pgv_norm
            phase_gain_eff = pgv_unit[..., 0].contiguous()  # (J_LARGE, H, R_PLANES)

            # RoPE disabled: use unrotated q/k (log-sum-exp merge path was NaN source)
            out_l, lse_l = dsqg_attention_grouped_with_lse(
                q, k, v, pb_large, se_large,
                self.phase_base, phase_gain_eff,
                y_pre, z_pre,
                self.offsets_dev_large, self.j_large, 0, self.j_large)

            max_lse = torch.maximum(lse_s.detach(), lse_l.detach())
            w_s = torch.exp(lse_s.detach() - max_lse)
            w_l = torch.exp(lse_l.detach() - max_lse)
            denom = (w_s + w_l).unsqueeze(-1).clamp(min=1e-30)
            out = (w_s.unsqueeze(-1) * out_s.float()
                   + w_l.unsqueeze(-1) * out_l.float()) / denom
            out = out.to(q.dtype)

        elif self.j_large > 0:
            pgv = self.phase_gain_vec
            pgv_norm = pgv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pgv_unit = pgv / pgv_norm
            phase_gain_eff = pgv_unit[..., 0].contiguous()
            # RoPE disabled: use unrotated q/k
            out = dsqg_attention_grouped(
                q, k, v,
                self.pos_bias, self.scale_embed,
                self.phase_base, phase_gain_eff,
                y_pre, z_pre,
                self.offsets_dev, self.j_val, 0, self.j_large)

        else:
            out = dsqg_attention_grouped(
                q, k, v,
                self.pos_bias, self.scale_embed,
                self.phase_base, self.phase_gain,
                y_pre, z_pre,
                self.offsets_dev, self.j_val, self.j_small, 0)

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))
