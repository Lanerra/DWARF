"""
DSQG Persistent Block V1 — Layer-fused module for pure DSQG models
===================================================================

Replaces per-layer DSQGBlockTriadic loop with a single module that owns
ALL L layers' parameters internally. Key advantages:

  1. Activation checkpointing via recomputation: stores only x_in per
     checkpointed segment, recomputes forward during backward
  2. All parameters packed as [L, ...] tensors for contiguous memory
  3. Single module interface: DSQGPersistentBlock(config).forward(x) -> x
  4. Per-layer triadic offset groups (GROUP_A/B/C/T cycling)

Option A (v1): Projections (qkv, out, gate, FFN) remain as PyTorch ops
using cuBLAS. The persistent kernel handles the attention inner loop via
existing V8 dynamic-J Triton kernels. Projection matmuls are already
tensor-core optimal via cuBLAS.

Usage:
  from dsqg_persistent_v1 import DSQGPersistentBlock

  block = DSQGPersistentBlock(
      num_layers=12, embedding_dim=512, num_heads=8, ffn_dim=1024,
      layer_configs=[
          {'offsets': GROUP_A, 'j_small': 17, 'j_large': 15},
          {'offsets': GROUP_B, 'j_small': 0,  'j_large': 32},
          ...
      ],
  )
  x_out = block(x)  # [B, N, D] -> [B, N, D]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

import pathlib as _pl
import os, sys

_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_dynamic_j import (
    _fwd_v8, _compute_D_v8, _bwd_dq_v8, _bwd_dkdv_v8,
    npci_rotate, R_PLANES, _next_pow2,
)
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

_EMA_FLOOR = 1e-5

def _causal_ema(xi, ema_factor, floor=_EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)

def _agc_normalize(pool, eps=1e-6):
    """Adaptive Gradient Clipping normalisation on the EMA pool."""
    norm = pool.norm(dim=-1, keepdim=True).clamp(min=eps)
    return pool / norm


class PreIFModule(nn.Module):
    """Per-layer preIF (Interference) module — EMA causal pool → k/v delta injection."""
    def __init__(self, embedding_dim, num_heads, ema_init=0.020833):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.inter_norm  = nn.LayerNorm(embedding_dim)
        self.inter_gate  = nn.Linear(embedding_dim, embedding_dim)
        self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ema_factor  = nn.Parameter(torch.full((1,), ema_init))

    def forward(self, x):
        """Returns (k_delta, v_delta) tensors shaped [B, H, N, HD]."""
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        xi   = self.inter_norm(x)
        pool = _causal_ema(xi, self.ema_factor.abs() + _EMA_FLOOR)
        pool = _agc_normalize(pool)
        inter  = torch.sigmoid(self.inter_gate(xi)) * pool
        k_delta = self.inter_k_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v_delta = self.inter_v_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        return k_delta, v_delta
import triton


def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets), (
        f"J_SMALL({j_small}) + J_LARGE({j_large}) != J({len(offsets)})")
    return j_small, j_large


class _GroupedAttentionFn(torch.autograd.Function):
    """Autograd wrapper for V8 dynamic-J DSQG attention with per-layer offsets."""

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                offsets_dev, j_val, j_small, j_large):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64, 8, 2
            else:       BLOCK_N, _num_warps, _num_stages = 64, 4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 32, 4, 2
        elif HD <= 256:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 32, 4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 32, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 16, 4, 2
        else:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 16, 4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 16, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 8, 4, 2

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
            num_warps=_num_warps, num_stages=_num_stages,
        )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev)
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _num_warps
        ctx.num_stages = _num_stages
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
        return out

    @staticmethod
    def backward(ctx, dout):
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
        dpb_buf = torch.empty(B * H, blocks_n, j_val, device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, j_val * HD, device=_dev, dtype=torch.float32)

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
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(dim=(0, 1, 2)).contiguous()

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


def _grouped_attention(q, k, v, pos_bias, scale_embed,
                       phase_base, phase_gain, y_pre, z_pre,
                       offsets_dev, j_val, j_small, j_large):
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _GroupedAttentionFn.apply(
        q.contiguous().clone(),
        k.contiguous().clone(),
        v.contiguous().clone(),
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
        offsets_dev, j_val, j_small, j_large,
    )
    return out if orig == torch.bfloat16 else out.to(orig)


class DSQGPersistentBlock(nn.Module):
    """
    Persistent DSQG block: owns all L DSQG layers in a single module.

    Parameters are stored per-layer using nn.ParameterList for variable-size
    attention params (offsets differ across triadic groups) and packed [L, ...]
    tensors for fixed-size weights (norms, projections, FFN).

    Forward loop iterates over layers with optional gradient checkpointing.
    Projections use PyTorch cuBLAS (Option A). Attention uses V8 dynamic-J
    Triton kernels.
    """

    def __init__(self, num_layers, embedding_dim, num_heads, ffn_dim,
                 layer_configs, dropout=0.1, scale_embed_init_val=0.0,
                 checkpoint_every=1, ema_init=0.020833):
        """
        Args:
            num_layers: number of DSQG layers
            embedding_dim: model dimension D
            num_heads: number of attention heads H
            ffn_dim: FFN hidden dimension
            layer_configs: list of L dicts, each with:
                'offsets': list[int] — offset deltas for this layer
                'j_small': int — number of small offsets (delta <= 28)
                'j_large': int — number of large offsets (delta >= 48)
                'has_if':  bool  — whether this layer has preIF (default False)
            dropout: dropout probability
            scale_embed_init_val: initial value for scale_embed (0.0 = zero-init)
            checkpoint_every: gradient-checkpoint every N layers (1 = all)
            ema_init: initial EMA alpha for preIF layers
        """
        super().__init__()
        assert len(layer_configs) == num_layers

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.ffn_dim = ffn_dim
        self.checkpoint_every = checkpoint_every
        HD = self.head_dim
        H = num_heads
        D = embedding_dim

        assert HD >= 4

        self.norm1_weight = nn.Parameter(torch.ones(num_layers, D))
        self.norm1_bias = nn.Parameter(torch.zeros(num_layers, D))
        self.norm2_weight = nn.Parameter(torch.ones(num_layers, D))
        self.norm2_bias = nn.Parameter(torch.zeros(num_layers, D))

        self.qkv_weight = nn.Parameter(torch.empty(num_layers, 3 * D, D))
        self.qkv_bias = nn.Parameter(torch.zeros(num_layers, 3 * D))
        self.out_weight = nn.Parameter(torch.empty(num_layers, D, D))
        self.out_bias = nn.Parameter(torch.zeros(num_layers, D))
        self.gate_weight = nn.Parameter(torch.empty(num_layers, D, D))
        self.gate_bias = nn.Parameter(torch.zeros(num_layers, D))

        self.ffn_fc1_weight = nn.Parameter(torch.empty(num_layers, ffn_dim, D))
        self.ffn_fc1_bias = nn.Parameter(torch.zeros(num_layers, ffn_dim))
        self.ffn_fc2_weight = nn.Parameter(torch.empty(num_layers, D, ffn_dim))
        self.ffn_fc2_bias = nn.Parameter(torch.zeros(num_layers, D))

        self.if_gain = nn.Parameter(torch.ones(num_layers, H))

        self.query_probes = nn.Parameter(torch.zeros(num_layers, R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(num_layers, R_PLANES, HD))
        self.npci_theta_k = nn.Parameter(torch.zeros(num_layers, H))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_layers, H))

        self.dropout_p = dropout

        # preIF modules — only instantiated for layers with has_if=True
        self._layer_has_if = [bool(cfg.get('has_if', False)) for cfg in layer_configs]
        self.preif_modules = nn.ModuleDict({
            str(i): PreIFModule(embedding_dim, num_heads, ema_init=ema_init)
            for i, cfg in enumerate(layer_configs) if cfg.get('has_if', False)
        })

        self._layer_j_val = []
        self._layer_j_small = []
        self._layer_j_large = []

        self.pos_bias_list = nn.ParameterList()
        self.scale_embed_list = nn.ParameterList()
        self.phase_base_list = nn.ParameterList()
        self.phase_gain_vec_list = nn.ParameterList()  # SO(2) MOVT: (x,y) unit-circle vectors

        offsets_list = []
        for i, cfg in enumerate(layer_configs):
            offsets = cfg['offsets']
            j_small = cfg['j_small']
            j_large = cfg['j_large']
            j_val = len(offsets)
            assert j_small + j_large == j_val

            self._layer_j_val.append(j_val)
            self._layer_j_small.append(j_small)
            self._layer_j_large.append(j_large)

            alphas = torch.linspace(0.2, 2.0, H)
            delta_vals = torch.tensor(
                [math.log(1.0 + d) for d in offsets], dtype=torch.float32)
            self.pos_bias_list.append(
                nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0)))

            self.scale_embed_list.append(
                nn.Parameter(torch.zeros(j_val, HD)))

            self.phase_base_list.append(
                nn.Parameter(torch.zeros(max(j_large, 1), H, R_PLANES)))
            # SO(2) MOVT: reparameterize phase_gain as 2D unit-circle vector (x,y).
            # init=(1,0) → identity rotation; eliminates 2π periodic boundaries.
            _pgv_init = torch.zeros(max(j_large, 1), H, R_PLANES, 2)
            _pgv_init[..., 0] = 1.0
            self.phase_gain_vec_list.append(nn.Parameter(_pgv_init))

            offsets_list.append(
                torch.tensor(offsets, dtype=torch.int32))

        self._offsets_list = offsets_list
        self._offsets_registered = False

        self._init_weights(scale_embed_init_val)

    def _register_offset_buffers(self, device):
        if self._offsets_registered:
            return
        for i, off in enumerate(self._offsets_list):
            self.register_buffer(
                f'_offsets_dev_{i}', off.to(device), persistent=False)
        self._offsets_registered = True

    def _init_weights(self, scale_embed_init_val):
        for i in range(self.num_layers):
            nn.init.normal_(self.qkv_weight[i], 0, 0.02)
            nn.init.normal_(self.out_weight[i], 0, 0.02)
            nn.init.normal_(self.gate_weight[i], 0, 0.02)
            nn.init.normal_(self.ffn_fc1_weight[i], 0, 0.02)
            nn.init.normal_(self.ffn_fc2_weight[i], 0, 0.02)

            nn.init.normal_(self.phase_base_list[i], 0.0, 0.01)
            nn.init.normal_(self.query_probes[i], 0.0, 0.01)
            nn.init.normal_(self.key_probes[i], 0.0, 0.01)
            nn.init.normal_(self.phase_gain_vec_list[i], 0.0, 0.001)
            with torch.no_grad():
                self.phase_gain_vec_list[i].data = (
                    self.phase_gain_vec_list[i].data /
                    self.phase_gain_vec_list[i].data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                )
            if scale_embed_init_val != 0.0:
                nn.init.constant_(self.scale_embed_list[i], scale_embed_init_val)

    def _layer_norm(self, x, layer_index, which):
        if which == 1:
            w = self.norm1_weight[layer_index]
            b = self.norm1_bias[layer_index]
        else:
            w = self.norm2_weight[layer_index]
            b = self.norm2_bias[layer_index]
        return F.layer_norm(x, (self.embedding_dim,), w, b)

    def _single_layer_forward(self, x, layer_index):
        B, N, D = x.shape
        H = self.num_heads
        HD = self.head_dim
        li = layer_index

        normed = self._layer_norm(x, li, 1)

        qkv = F.linear(normed, self.qkv_weight[li], self.qkv_bias[li])
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        # preIF: inject k_delta/v_delta via EMA causal pool
        if self._layer_has_if[li]:
            k_delta, v_delta = self.preif_modules[str(li)](x)
            k_delta = k_delta.contiguous().clone()
            v_delta = v_delta.contiguous().clone()
            k = npci_rotate(k, k_delta, self.npci_theta_k[li])
            v = npci_rotate(v, v_delta, self.npci_theta_v[li])

        sc = HD ** -0.5
        y_pre = torch.einsum(
            'bhnd,rd->bhnr', q, self.query_probes[li].to(q.dtype)
        ).mul(sc).float().contiguous()
        z_pre = torch.einsum(
            'bhnd,rd->bhnr', k, self.key_probes[li].to(k.dtype)
        ).mul(sc).float().contiguous()

        offsets_dev = getattr(self, f'_offsets_dev_{li}')
        j_val = self._layer_j_val[li]
        j_small = self._layer_j_small[li]
        j_large = self._layer_j_large[li]

        # SO(2) MOVT: derive scalar phase_gain from unit-circle vector
        _pgv = self.phase_gain_vec_list[li]
        _pgv_unit = _pgv / _pgv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        phase_gain = _pgv_unit[..., 0].contiguous()

        attn_out = _grouped_attention(
            q, k, v,
            self.pos_bias_list[li], self.scale_embed_list[li],
            self.phase_base_list[li], phase_gain,
            y_pre, z_pre,
            offsets_dev, j_val, j_small, j_large,
        )

        attn_out = attn_out * self.if_gain[li].view(1, H, 1, 1)
        attn_flat = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(F.linear(normed, self.gate_weight[li], self.gate_bias[li]))
        projected = F.linear(attn_flat * gate, self.out_weight[li], self.out_bias[li])
        projected = F.dropout(projected, p=self.dropout_p, training=self.training)
        x = x + projected

        normed2 = self._layer_norm(x, li, 2)
        ffn_out = F.linear(normed2, self.ffn_fc1_weight[li], self.ffn_fc1_bias[li])
        ffn_out = F.gelu(ffn_out)
        ffn_out = F.dropout(ffn_out, p=self.dropout_p, training=self.training)
        ffn_out = F.linear(ffn_out, self.ffn_fc2_weight[li], self.ffn_fc2_bias[li])
        x = x + ffn_out

        return x

    def forward(self, x):
        self._register_offset_buffers(x.device)

        for i in range(self.num_layers):
            should_checkpoint = (
                self.training
                and self.checkpoint_every > 0
                and i % self.checkpoint_every == 0
            )
            if should_checkpoint:
                x = grad_ckpt(
                    self._single_layer_forward, x, i,
                    use_reentrant=False,
                )
            else:
                x = self._single_layer_forward(x, i)
        return x

    def copy_layer_from_triadic_block(self, layer_index, triadic_block):
        """
        Copy weights from a DSQGBlockTriadic (or equivalent) into this
        persistent block at the given layer index. Useful for migration
        from layer-by-layer models.
        """
        li = layer_index
        attn = triadic_block.attn

        with torch.no_grad():
            self.norm1_weight[li].copy_(triadic_block.norm1.weight)
            self.norm1_bias[li].copy_(triadic_block.norm1.bias)
            self.norm2_weight[li].copy_(triadic_block.norm2.weight)
            self.norm2_bias[li].copy_(triadic_block.norm2.bias)

            self.qkv_weight[li].copy_(attn.qkv_proj.weight)
            self.qkv_bias[li].copy_(attn.qkv_proj.bias)
            self.out_weight[li].copy_(attn.out_proj.weight)
            self.out_bias[li].copy_(attn.out_proj.bias)
            self.gate_weight[li].copy_(attn.gate_proj.weight)
            self.gate_bias[li].copy_(attn.gate_proj.bias)

            self.ffn_fc1_weight[li].copy_(triadic_block.ffn.fc1.weight)
            self.ffn_fc1_bias[li].copy_(triadic_block.ffn.fc1.bias)
            self.ffn_fc2_weight[li].copy_(triadic_block.ffn.fc2.weight)
            self.ffn_fc2_bias[li].copy_(triadic_block.ffn.fc2.bias)

            self.if_gain[li].copy_(attn.if_gain)
            self.query_probes[li].copy_(attn.query_probes)
            self.key_probes[li].copy_(attn.key_probes)
            self.npci_theta_k[li].copy_(attn.npci_theta_k)
            self.npci_theta_v[li].copy_(attn.npci_theta_v)

            self.pos_bias_list[li].data.copy_(attn.pos_bias)
            self.scale_embed_list[li].data.copy_(attn.scale_embed)
            self.phase_base_list[li].data.copy_(attn.phase_base)
            self.phase_gain_vec_list[li].data.copy_(attn.phase_gain_vec)

            if self._layer_has_if[li] and hasattr(triadic_block, 'interference') and triadic_block.interference:
                pif = self.preif_modules[str(li)]
                pif.inter_norm.weight.data.copy_(triadic_block.inter_norm.weight)
                pif.inter_norm.bias.data.copy_(triadic_block.inter_norm.bias)
                pif.inter_gate.weight.data.copy_(triadic_block.inter_gate.weight)
                pif.inter_gate.bias.data.copy_(triadic_block.inter_gate.bias)
                pif.inter_k_proj.weight.data.copy_(triadic_block.inter_k_proj.weight)
                pif.inter_k_proj.bias.data.copy_(triadic_block.inter_k_proj.bias)
                pif.inter_v_proj.weight.data.copy_(triadic_block.inter_v_proj.weight)
                pif.inter_v_proj.bias.data.copy_(triadic_block.inter_v_proj.bias)
                pif.ema_factor.data.copy_(triadic_block.ema_factor)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
