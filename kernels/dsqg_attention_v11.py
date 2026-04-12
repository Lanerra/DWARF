"""
DSQG Attention V11 — FlexAttention-based implementation
========================================================

Drop-in replacement for DSQGAttentionV8_H100 (V8 Triton kernel) using PyTorch
FlexAttention.

Design rationale:
  - V8 Triton kernel loops over 24 offsets sequentially with scattered HBM loads
  - FlexAttention uses block-sparse patterns with fused kernels
  - Block mask encodes the sparse offset pattern (J=24 fixed offsets)
  - score_mod adds pos_bias and scale_embed contributions

PERFORMANCE NOTE:
  V11 is currently ~40x SLOWER than V8 due to torch.compile overhead and
  irregular indexing patterns in score_mod. FlexAttention is optimized for
  regular patterns (causal, sliding window) — not for our sparse 24-offset
  pattern with per-offset learnable biases.

  The V8 Triton kernel's sequential loop over 24 offsets is actually well-
  suited to this access pattern. Consider using V8 for production training.

  V11 serves as:
  1. Reference implementation for correctness verification
  2. Starting point if PyTorch optimizes irregular score_mod patterns
  3. Educational example of FlexAttention usage with DSQG

NPCI (phase rotation) DROPPED:
  Across 31 trained layers (d768_l32_fa8_best.pt checkpoint):
    phase_gain RMS = 0.0010
    phase_base RMS = 0.0224
    ratio = 4.4%
  The phase_gain signal is effectively zero — NPCI rotation has negligible effect.
  PyTorch 2.6 FlexAttention has no value_mod anyway.

  Kept as UNUSED parameters for checkpoint compatibility:
    - phase_base    [J_LARGE=10, H, 2]
    - phase_gain    [J_LARGE=10, H, 2]
    - query_probes  [2, HD]
    - key_probes    [2, HD]

Offset set: J=24, same as V8
  ALL_OFFSETS = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]

Score computation (exact match to V8):
  For each valid (q_idx, k_idx) pair where diff = q_idx - k_idx is in ALL_OFFSETS:
    score = dot(Q[q_idx], K[k_idx]) / sqrt(HD)
          + pos_bias[j, h]                          # learnable per-offset routing prior [J, H]
          + dot(Q[q_idx], scale_embed[j]) / sqrt(HD)    # content filter (zero-init) [J, HD]
  where j = offset_to_j[diff] maps the offset delta to its index in ALL_OFFSETS.

Usage:
  from dsqg_attention_v11 import DSQGAttentionV11

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/test_v11.py
"""

import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

ALL_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
J = len(ALL_OFFSETS)
J_SMALL = 14
J_LARGE = 10
R_PLANES = 2
MAX_DELTA = max(ALL_OFFSETS)

_BLOCK_MASK_CACHE = {}


def _build_offset_buffers(max_n: int, device: torch.device):
    """
    Build offset lookup buffers for FlexAttention score_mod.

    Returns:
        offset_set: bool tensor [max_n+1] where offset_set[d] = True if d is a valid offset
        offset_to_j: int tensor [max_n+1] mapping delta → j index (-1 for invalid)
    """
    offset_set = torch.zeros(max_n + 1, dtype=torch.bool, device=device)
    offset_to_j = torch.full((max_n + 1,), -1, dtype=torch.int64, device=device)

    for j, delta in enumerate(ALL_OFFSETS):
        if delta <= max_n:
            offset_set[delta] = True
            offset_to_j[delta] = j

    return offset_set, offset_to_j


def _get_or_create_block_mask(B: int, H: int, N: int, device: torch.device, offset_set: torch.Tensor):
    """
    Get cached block mask or create new one.
    Block mask creation is expensive — cache per (N, device).
    """
    cache_key = (N, device)
    if cache_key in _BLOCK_MASK_CACHE:
        cached_mask = _BLOCK_MASK_CACHE[cache_key]
        if cached_mask.shape[0] >= B and cached_mask.shape[1] >= H:
            return cached_mask

    def mask_mod(b, h, q_idx, k_idx):
        diff = q_idx - k_idx
        valid_offset = (diff > 0) & (diff <= MAX_DELTA)
        return valid_offset & offset_set[diff.clamp(0, len(offset_set) - 1)]

    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device
    )
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta_h: torch.Tensor) -> torch.Tensor:
    """
    Apply norm-preserving coupled injection (kept for interface compatibility).

    x, x_delta: [B, H, N, HD]  (same shape)
    theta_h:    [H]             per-head rotation angle (radians)

    Returns x' with ||x'[b,h,n,:]|| = ||x[b,h,n,:]|| exactly.
    """
    theta = theta_h.view(1, -1, 1, 1)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    x_hat = x / x_norm

    parallel = (x_delta * x_hat).sum(dim=-1, keepdim=True) * x_hat
    perp = x_delta - parallel
    perp_norm = perp.norm(dim=-1, keepdim=True)

    mask = perp_norm > x_norm * 1e-10
    u_hat = torch.where(mask, perp / perp_norm.clamp(min=1e-30), torch.zeros_like(perp))

    return torch.cos(theta) * x + torch.sin(theta) * x_norm * u_hat


def dsqg_attention_v11(q, k, v, pos_bias, scale_embed, offset_set, offset_to_j):
    """
    FlexAttention-based DSQG attention.

    Args:
        q, k, v:       [B, H, N, HD]  bfloat16 or float32
        pos_bias:      [J=24, H]      float32
        scale_embed:   [J=24, HD]     float32
        offset_set:    [max_n+1]      bool tensor (module buffer)
        offset_to_j:   [max_n+1]      int64 tensor (module buffer)

    Returns:
        out:           [B, H, N, HD]  same dtype as q
    """
    B, H, N, HD = q.shape
    device = q.device
    dtype = q.dtype

    scale = HD ** -0.5

    q_se = torch.einsum('bhnd,jd->bhnj', q, scale_embed.to(dtype)) * scale

    pos_bias_t = pos_bias.T.to(dtype).contiguous()

    def score_mod(score, b, h, q_idx, k_idx):
        diff = q_idx - k_idx
        diff_clamped = diff.clamp(0, offset_to_j.shape[0] - 1)
        j = offset_to_j[diff_clamped]
        j_safe = j.clamp(0, J - 1)

        pb = pos_bias_t[h, j_safe]
        se = q_se[b, h, q_idx, j_safe]

        return score + pb + se

    block_mask = _get_or_create_block_mask(B, H, N, device, offset_set)

    out = flex_attention(
        q, k, v,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
    )

    return out


class DSQGAttentionV11(nn.Module):
    """
    DSQG V11: FlexAttention-based drop-in replacement for DSQGAttentionV8_H100.

    J=24 relay-optimal offsets, same score computation as V8.
    NPCI rotation dropped (phase_gain effectively zero in trained checkpoints).

    Parameters (checkpoint-compatible with V8):
      pos_bias      [24, H]    global frequency prior
      scale_embed   [24, HD]   Q-matched-filter (zero-init)
      if_gain       [H]        IF amplifier gain (1.0-init)
      phase_base    [10, H, 2] MOVT angles for large offsets — UNUSED, kept for ckpt compat
      phase_gain    [10, H, 2] QK-OVT gains for large offsets — UNUSED, kept for ckpt compat
      query_probes  [2, HD]    y precomputation probes — UNUSED, kept for ckpt compat
      key_probes    [2, HD]    z precomputation probes — UNUSED, kept for ckpt compat
      npci_theta_k  [H]        NPCI K rotation angle — UNUSED, kept for ckpt compat
      npci_theta_v  [H]        NPCI V rotation angle — UNUSED, kept for ckpt compat
    """

    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.seq_len = seq_len
        HD = self.head_dim

        assert HD >= 4, "HD must be >= 4 for checkpoint compatibility with V8"

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain = nn.Parameter(torch.ones(num_heads))

        self.phase_base = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

        max_n = max(seq_len, MAX_DELTA + 1)
        offset_set, offset_to_j = _build_offset_buffers(max_n, torch.device('cpu'))
        self.register_buffer('offset_set', offset_set, persistent=False)
        self.register_buffer('offset_to_j', offset_to_j, persistent=False)

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

        if self.offset_set.device != x.device:
            max_n = max(N, MAX_DELTA + 1)
            offset_set, offset_to_j = _build_offset_buffers(max_n, x.device)
            self.offset_set = offset_set
            self.offset_to_j = offset_to_j

        out = dsqg_attention_v11(
            q, k, v,
            self.pos_bias,
            self.scale_embed,
            self.offset_set,
            self.offset_to_j,
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
            se = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            phb = self.phase_base.detach().cpu()
            phg = self.phase_gain.detach().cpu()
            qp = self.query_probes.detach().cpu()
            kp = self.key_probes.detach().cpu()
            thk = self.npci_theta_k.detach().cpu()
            thv = self.npci_theta_v.detach().cpu()

        plane_diff = (phb[:, :, 0] - phb[:, :, 1]).abs().mean().item()

        return {
            'pos_bias_abs_mean': pb.abs().mean().item(),
            'pos_bias_abs_max': pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean': se.abs().mean().item(),
            'scale_embed_abs_max': se.abs().max().item(),
            'if_gain': gain.tolist(),
            'phase_base_abs_mean': phb.abs().mean().item(),
            'phase_base_abs_max': phb.abs().max().item(),
            'phase_base_local_mean': 0.0,
            'phase_base_sparse_mean': phb.abs().mean().item(),
            'phase_base_by_head': phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_sparse_by_head': phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_plane_diff': plane_diff,
            'phase_base_p0_sparse': phb[:, :, 0].mean(0).tolist(),
            'phase_base_p1_sparse': phb[:, :, 1].mean(0).tolist(),
            'phase_gain_abs_mean': phg.abs().mean().item(),
            'phase_gain_abs_max': phg.abs().max().item(),
            'phase_gain_sparse_mean': phg.abs().mean().item(),
            'phase_gain_by_head': phg.abs().mean(dim=(0, 2)).tolist(),
            'query_probe_norm': qp.norm(dim=1).tolist(),
            'key_probe_norm': kp.norm(dim=1).tolist(),
            'npci_theta_k': thk.tolist(),
            'npci_theta_v': thv.tolist(),
        }


if __name__ == "__main__":
    print("=" * 70)
    print("DSQG Attention V11 — FlexAttention-based implementation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    B, H, N, HD = 2, 12, 512, 64
    D = H * HD

    model = DSQGAttentionV11(D, H, seq_len=N).to(device)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    print("\nBasic forward pass: PASS")
