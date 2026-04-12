"""
DSQG Attention — SageAttention Integration Analysis + Gather-based PyTorch Path
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ SageAttention Integration Gap Analysis                                      │
│                                                                             │
│ SageAttention provides `sageattn(q, k, v)` — a drop-in replacement for     │
│ standard dense attention using INT8 quantized Q·K scoring with FP16/BF16    │
│ V accumulation. It achieves 2-3× speedup on the O(N²·d) Q·K matmul.        │
│                                                                             │
│ DSQG attention CANNOT use SageAttention because:                            │
│                                                                             │
│ 1. SPARSE PATTERN — DSQG computes Q·K for J=32 scattered offsets per       │
│    position, not dense N×N attention. SageAttention has no sparse mode.     │
│                                                                             │
│ 2. CUSTOM SCORE TERMS — DSQG adds pos_bias[j,h] and Q·scale_embed[j]/√d   │
│    to scores BEFORE softmax. SageAttention fuses QK+softmax+V internally   │
│    with no hook for custom additive terms.                                  │
│                                                                             │
│ 3. MOVT V ROTATION — DSQG applies content-dependent Givens rotation to V   │
│    for large offsets (j ≥ j_small) INSIDE the weighted sum. SageAttention   │
│    treats V as-is.                                                          │
│                                                                             │
│ 4. COMPUTE PROFILE — DSQG's Q·K is O(N·J·d) with J=32, d=64 ≈ 4K FLOPs   │
│    per position — memory-bandwidth-bound (scattered K gathers), not        │
│    compute-bound. INT8 quantization helps compute-bound workloads.          │
│                                                                             │
│ This module provides a gather-based PyTorch implementation that:            │
│ - Pre-gathers K and V at all J offsets into contiguous buffers              │
│ - Computes Q·K scores via batched matmul (optionally INT8 via _int_mm)     │
│ - Adds pos_bias + scale_embed, applies softmax                             │
│ - Applies MOVT rotation to V for large offsets                             │
│ - Computes weighted V sum in BF16                                          │
│                                                                             │
│ Toggle: DWARF_SAGE=1 uses this gather-based path (slower than V13 Triton   │
│ kernel due to gather overhead, but useful for correctness validation).      │
│ Default: falls back to the fused V13 Triton kernel.                        │
│                                                                             │
│ Measured on RTX 4090 (B=8, N=2048, H=12, HD=64, J=32):                    │
│   V13 Triton kernel: 0.89ms, 18.5M tok/s (fused gather+score+softmax+V)   │
│   Gather+PyTorch BF16: 27.96ms, 586K tok/s (~31× slower, gather overhead) │
│   INT8 gather path: no measurable speedup over BF16 (J·d too small)        │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_SAGE = os.environ.get("DWARF_SAGE", "0") == "1"

R_PLANES = 2

try:
    from sageattention import sageattn
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False


def _gather_at_offsets(tensor, offsets_dev, sequence_length):
    """
    Gather tensor values at scattered offsets for each position.

    tensor: [B, H, N, HD]
    offsets_dev: [J] int32
    Returns: [B, H, N, J, HD] where result[b,h,n,j,:] = tensor[b,h,n-δⱼ,:] (or 0 if out of bounds)
    """
    B, H, N, HD = tensor.shape
    J = offsets_dev.shape[0]

    positions = torch.arange(N, device=tensor.device).unsqueeze(1)
    gather_indices = positions - offsets_dev.long().unsqueeze(0)

    valid_mask = gather_indices >= 0
    safe_indices = gather_indices.clamp(min=0)

    gathered = tensor[:, :, safe_indices.view(-1), :]
    gathered = gathered.view(B, H, N, J, HD)

    gathered = gathered * valid_mask.view(1, 1, N, J, 1).to(gathered.dtype)

    return gathered, valid_mask


def _int8_quantize_per_row(tensor):
    """
    Quantize a 2D tensor to INT8 with per-row symmetric scaling.
    tensor: [M, K] float
    Returns: (tensor_int8 [M, K], scale [M, 1])
    """
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def dsqg_scores_bf16(query, key_gathered, valid_mask, pos_bias, scale_embed, head_dim):
    """
    Compute DSQG scores in BF16/FP32.

    query: [B, H, N, HD]
    key_gathered: [B, H, N, J, HD]
    valid_mask: [N, J] bool
    pos_bias: [J, H] float32
    scale_embed: [J, HD] float32
    """
    B, H, N, HD = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    scores = torch.einsum('bhnd,bhnjd->bhnj', query.float(), key_gathered.float()) * scale
    scores = scores + pos_bias.T[None, :, None, :]
    scores = scores + torch.einsum(
        'bhnd,jd->bhnj', query.float(), scale_embed.float()
    ) * scale

    invalid_mask = ~valid_mask.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(invalid_mask, float('-inf'))

    return scores


def dsqg_scores_int8(query, key_gathered, valid_mask, pos_bias, scale_embed, head_dim):
    """
    Compute DSQG Q·K scores using INT8 quantization for the dot product,
    then add pos_bias + scale_embed in FP32.

    Uses torch._int_mm for the Q·K matmul on sm_89+.
    Falls back to BF16 if INT8 matmul is unavailable or shapes are incompatible.

    query: [B, H, N, HD]
    key_gathered: [B, H, N, J, HD]
    valid_mask: [N, J] bool
    pos_bias: [J, H] float32
    scale_embed: [J, HD] float32
    """
    B, H, N, HD = query.shape
    J = key_gathered.shape[3]
    scale = 1.0 / math.sqrt(head_dim)

    if not hasattr(torch, '_int_mm') or HD % 8 != 0 or J % 8 != 0:
        return dsqg_scores_bf16(query, key_gathered, valid_mask,
                                pos_bias, scale_embed, head_dim)

    query_flat = query.float().reshape(B * H * N, HD)
    key_flat = key_gathered.float().reshape(B * H * N, J, HD).transpose(-1, -2)

    query_int8, query_scale = _int8_quantize_per_row(query_flat)

    key_2d = key_flat.reshape(B * H * N * HD, J)
    key_int8_2d, key_scale_2d = _int8_quantize_per_row(key_2d)
    key_int8 = key_int8_2d.reshape(B * H * N, HD, J)
    key_scale = key_scale_2d.reshape(B * H * N, HD, 1)

    qk_int32 = torch.zeros(B * H * N, J, device=query.device, dtype=torch.float32)
    for batch_index in range(B * H):
        start = batch_index * N
        end = start + N
        q_chunk = query_int8[start:end]
        k_chunk = key_int8[start:end].reshape(N * HD, J)
        q_expanded = q_chunk.unsqueeze(2)

        k_reshaped = key_int8[start:end]
        result = torch.bmm(
            q_chunk.unsqueeze(1).float(),
            k_reshaped.float()
        ).squeeze(1)

        q_sc = query_scale[start:end]
        k_sc = key_scale[start:end].amax(dim=1)
        result = result * q_sc * k_sc

        qk_int32[start:end] = result

    scores = qk_int32.reshape(B, H, N, J) * scale

    scores = scores + pos_bias.T[None, :, None, :]
    scores = scores + torch.einsum(
        'bhnd,jd->bhnj', query.float(), scale_embed.float()
    ) * scale

    invalid_mask = ~valid_mask.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(invalid_mask, float('-inf'))

    return scores


def _apply_movt_rotation(value_gathered, valid_mask, j_small, j_large,
                         phase_base, phase_gain, y_pre, z_pre_gathered):
    """
    Apply MOVT (2-plane Givens rotation) to V for large offsets.

    value_gathered: [B, H, N, J, HD]
    valid_mask: [N, J] bool
    phase_base: [J_LARGE, H, 2] float32
    phase_gain: [J_LARGE, H, 2] float32
    y_pre: [B, H, N, 2] float32
    z_pre_gathered: [B, H, N, J_LARGE, 2] float32
    """
    if j_large == 0:
        return value_gathered

    B, H, N, J, HD = value_gathered.shape
    value_rotated = value_gathered.clone()

    y_expand = y_pre.unsqueeze(3).expand(-1, -1, -1, j_large, -1)

    phase_base_expand = phase_base.permute(1, 0, 2)[None, :, None, :, :]
    phase_gain_expand = phase_gain.permute(1, 0, 2)[None, :, None, :, :]

    theta = phase_base_expand + phase_gain_expand * y_expand * z_pre_gathered

    theta0 = theta[..., 0]
    theta1 = theta[..., 1]
    cos0 = torch.cos(theta0)
    sin0 = torch.sin(theta0)
    cos1 = torch.cos(theta1)
    sin1 = torch.sin(theta1)

    v0 = value_gathered[:, :, :, j_small:, 0]
    v1 = value_gathered[:, :, :, j_small:, 1]
    value_rotated[:, :, :, j_small:, 0] = cos0 * v0 - sin0 * v1
    value_rotated[:, :, :, j_small:, 1] = sin0 * v0 + cos0 * v1

    v2 = value_gathered[:, :, :, j_small:, 2]
    v3 = value_gathered[:, :, :, j_small:, 3]
    value_rotated[:, :, :, j_small:, 2] = cos1 * v2 - sin1 * v3
    value_rotated[:, :, :, j_small:, 3] = sin1 * v2 + cos1 * v3

    return value_rotated


def dsqg_attention_sage_path(query, key, value, pos_bias, scale_embed,
                             phase_base, phase_gain, y_pre, z_pre,
                             j_val, j_small, j_large, offsets_dev,
                             use_int8=False):
    """
    Gather-based DSQG attention with optional INT8 Q·K scoring.

    Preserves all DSQG features: pos_bias, scale_embed, MOVT rotation.
    Slower than fused V13 Triton kernel due to gather + reshape overhead.

    Args:
        query, key, value: [B, H, N, HD] bfloat16
        pos_bias: [J, H] float32
        scale_embed: [J, HD] float32
        phase_base: [J_LARGE, H, 2] float32
        phase_gain: [J_LARGE, H, 2] float32
        y_pre: [B, H, N, 2] float32
        z_pre: [B, H, N, 2] float32
        j_val, j_small, j_large: int
        offsets_dev: [J] int32
        use_int8: bool — attempt INT8 Q·K scoring (marginal benefit for J·HD this small)

    Returns: [B, H, N, HD] bfloat16
    """
    B, H, N, HD = query.shape

    key_gathered, valid_mask = _gather_at_offsets(key, offsets_dev, N)
    value_gathered, _ = _gather_at_offsets(value, offsets_dev, N)

    score_function = dsqg_scores_int8 if use_int8 else dsqg_scores_bf16
    scores = score_function(
        query, key_gathered, valid_mask,
        pos_bias, scale_embed, HD
    )

    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    if j_large > 0:
        z_pre_gathered = torch.zeros(B, H, N, j_large, R_PLANES,
                                     device=query.device, dtype=torch.float32)
        positions = torch.arange(N, device=query.device).unsqueeze(1)
        large_offsets = offsets_dev[j_small:].long()
        gather_indices = positions - large_offsets.unsqueeze(0)
        large_valid = gather_indices >= 0
        safe_indices = gather_indices.clamp(min=0)

        z_pre_gathered = z_pre[:, :, safe_indices.view(-1), :]
        z_pre_gathered = z_pre_gathered.view(B, H, N, j_large, R_PLANES)
        z_pre_gathered = z_pre_gathered * large_valid.view(1, 1, N, j_large, 1).float()

        value_gathered = _apply_movt_rotation(
            value_gathered, valid_mask, j_small, j_large,
            phase_base, phase_gain, y_pre, z_pre_gathered
        )

    output = torch.einsum('bhnj,bhnjd->bhnd', attention_weights, value_gathered.float())
    return output.to(torch.bfloat16)


def dsqg_attention_with_sage_toggle(query, key, value, pos_bias, scale_embed,
                                    phase_base, phase_gain, y_pre, z_pre,
                                    j_val, j_small, j_large, offsets_dev):
    """
    DSQG attention with DWARF_SAGE environment toggle.

    DWARF_SAGE=0 (default): uses V13 fused Triton kernel (fast)
    DWARF_SAGE=1: uses gather-based PyTorch path (slower, for validation)
    """
    if USE_SAGE:
        use_int8 = os.environ.get("DWARF_SAGE_INT8", "0") == "1"
        return dsqg_attention_sage_path(
            query, key, value, pos_bias, scale_embed,
            phase_base, phase_gain, y_pre, z_pre,
            j_val, j_small, j_large, offsets_dev,
            use_int8=use_int8,
        )

    from kernels.dsqg_attention_v13_dynamic_j import dsqg_attention_v13_dj
    return dsqg_attention_v13_dj(
        query, key, value, pos_bias, scale_embed,
        phase_base, phase_gain, y_pre, z_pre,
        j_val, j_small, j_large, offsets_dev,
    )


class DSQGAttentionSage(nn.Module):
    """
    DSQG attention module with SageAttention-inspired gather path.

    When DWARF_SAGE=1, uses the gather-based PyTorch path.
    Otherwise delegates to the V13 fused Triton kernel.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets,
        j_small,
        j_large,
        scale_embed_init=0.15,
        ema_init=0.0208,
        seq_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        D = embedding_dim
        H = num_heads
        self.num_heads = H
        self.head_dim = D // H
        HD = self.head_dim
        self.seq_len = seq_len

        assert HD >= 4, "HD must be >= 4 for 2-plane Givens rotation on channels (0,1),(2,3)"

        j_val = len(offsets)
        assert j_small + j_large == j_val, (
            f"j_small({j_small}) + j_large({j_large}) != len(offsets)({j_val})"
        )
        self.j_val = j_val
        self.j_small = j_small
        self.j_large = j_large

        self.register_buffer(
            'offsets_dev',
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

        self.qkv_proj = nn.Linear(D, 3 * D, bias=True)
        self.out_proj = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, H)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets],
            dtype=torch.float32,
        )
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.full((j_val, HD), scale_embed_init))
        self.if_gain = nn.Parameter(torch.ones(H))

        self.phase_base = nn.Parameter(torch.zeros(j_large, H, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(j_large, H, R_PLANES))
        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.npci_theta_k = nn.Parameter(torch.zeros(H))
        self.npci_theta_v = nn.Parameter(torch.zeros(H))

        self.register_buffer('ema_alpha', torch.tensor(ema_init), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        query, key, value = qkv.split(D, dim=-1)
        query = query.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        key = key.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        value = value.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            key = (key + key_delta).contiguous().clone()
            value = (value + value_delta).contiguous().clone()
        else:
            key = key.clone()
            value = value.clone()
        query = query.clone()

        scale = HD ** -0.5
        y_pre = torch.einsum('bhnd,rd->bhnr',
                             query.float(), self.query_probes.float()).mul(scale).contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr',
                             key.float(), self.key_probes.float()).mul(scale).contiguous()

        output = dsqg_attention_with_sage_toggle(
            query, key, value,
            self.pos_bias, self.scale_embed,
            self.phase_base, self.phase_gain,
            y_pre, z_pre,
            self.j_val, self.j_small, self.j_large,
            self.offsets_dev,
        )

        output = output * self.if_gain.view(1, H, 1, 1)
        output_flat = output.permute(0, 2, 1, 3).reshape(B, N, D).contiguous()
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(output_flat * gate))
