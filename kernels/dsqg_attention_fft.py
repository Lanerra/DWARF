"""
FFT-DSQG Attention — Dense PyTorch reformulation of DSQG attention
===================================================================

Replaces V8's Triton kernel with pure PyTorch pad-and-slice + einsum ops.
Preserves the exact relay mechanism: pos_bias, scale_embed, MOVT, NPCI,
if_gain, output gate.

Score computation backends:
  "gather" (default) — pad K, contiguous slice per offset, dot product.
     O(J*N*HD), same as V8 but coalesced PyTorch ops instead of Triton
     scattered loads. Memory-efficient: processes one offset at a time.
  "fft" — FFT K, multiply by per-offset phase ramp in frequency domain,
     IFFT back, pointwise Q*K_shifted, sum over HD. Mathematically correct
     but O(J*HD*N*log N) — strictly slower for J=24, N=2048.

FFT Cross-Correlation Analysis
------------------------------
The design brief proposed IFFT(conj(FFT(Q)) * FFT(K)) as cross-correlation.
That computes: xcorr[lag] = sum_t sum_d Q[t,d] * K[(t+lag) mod N, d]
This sums over ALL positions t, yielding ONE scalar per lag.
DSQG needs PER-POSITION scores: score[t,j] = sum_d Q[t,d] * K[t-delta_j, d]
These are fundamentally different (R^N output vs R^(N*J) output).
There is no N threshold where global FFT cross-correlation applies.

The "fft" backend instead uses frequency-domain SHIFTING (phase ramp
multiplication) to compute K_shifted[t] = K[t-delta] per offset, then
does the pointwise dot product. This is correct but adds N*log(N) overhead
per offset vs O(N) for a direct slice.

FlashFFTConv (Fu 2023) / Hyena / H3 use FFT for SSM-style convolutions
where the operation IS a global convolution. DSQG's per-position dot
products are structurally different.

Interface: identical to DSQGAttentionV8 / DSQGAttentionV13.

Usage:
  from dsqg_attention_fft import FFTDSQGAttention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '.')
from kernels.dsqg_attention_v8 import (
    ALL_OFFSETS, J, J_SMALL, J_LARGE, R_PLANES, MAX_DELTA, npci_rotate,
)


def fft_cross_correlation_analysis(q, k):
    """
    Demonstrates what global FFT cross-correlation computes.
    Returns xcorr[b,h,lag] = sum_t sum_d Q[t,d]*K[(t+lag)%N, d]
    This is NOT per-position scores — it sums over all positions.
    Included for mathematical verification only.
    """
    Q_f = torch.fft.rfft(q.float(), dim=2)
    K_f = torch.fft.rfft(k.float(), dim=2)
    xcorr_f = (Q_f.conj() * K_f).sum(dim=-1)
    return torch.fft.irfft(xcorr_f, n=q.shape[2], dim=2)


def _scores_via_gather(q_f, k_padded, pos_bias, scale_embed, positions):
    """
    Compute per-position DSQG scores via pad-and-slice (one offset at a time).
    Memory-efficient: O(B*H*N*HD) peak, not O(B*H*N*J*HD).
    """
    B, H, N, HD = q_f.shape
    sc = HD ** -0.5
    score_list = []
    for j in range(J):
        delta = ALL_OFFSETS[j]
        start = MAX_DELTA - delta
        K_j = k_padded[:, :, start:start + N, :]
        s = (q_f * K_j).sum(dim=-1) * sc
        s = s + pos_bias[j, :][None, :, None]
        s = s + (q_f * scale_embed[j]).sum(dim=-1) * sc
        score_list.append(s)
    scores = torch.stack(score_list, dim=-1)
    offsets_t = torch.tensor(ALL_OFFSETS, device=q_f.device, dtype=torch.long)
    causal_mask = positions[:, None] < offsets_t[None, :]
    return scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))


def _scores_via_fft(q_f, k_f, pos_bias, scale_embed, positions):
    """
    Compute per-position DSQG scores via FFT frequency-domain shifting.
    For each offset delta: shift K by delta in frequency domain (phase ramp),
    IFFT back, then pointwise Q*K_shifted dot product.
    Pads to next power of 2 >= N + MAX_DELTA to avoid circular aliasing.
    Uses FP32 throughout for FFT numerical stability.
    """
    B, H, N, HD = q_f.shape
    sc = HD ** -0.5
    device = q_f.device

    N_fft = 1
    while N_fft < N + MAX_DELTA:
        N_fft *= 2

    q_flat = q_f.reshape(B * H, N, HD)
    k_flat = k_f.reshape(B * H, N, HD)
    k_padded_fft = F.pad(k_flat, (0, 0, 0, N_fft - N))
    K_freq = torch.fft.rfft(k_padded_fft, n=N_fft, dim=1)

    freq_bins = torch.arange(K_freq.shape[1], device=device, dtype=torch.float32)

    score_list = []
    for j in range(J):
        delta = ALL_OFFSETS[j]
        phase_ramp = torch.exp(
            -1j * 2.0 * math.pi * freq_bins * delta / N_fft
        )
        K_shifted_freq = K_freq * phase_ramp[None, :, None]
        k_shifted = torch.fft.irfft(K_shifted_freq, n=N_fft, dim=1)[:, :N, :]
        s = (q_flat * k_shifted).sum(dim=-1) * sc
        s = s.reshape(B, H, N)
        s = s + pos_bias[j, :][None, :, None]
        s = s + (q_f * scale_embed[j]).sum(dim=-1) * sc
        score_list.append(s)

    scores = torch.stack(score_list, dim=-1)
    offsets_t = torch.tensor(ALL_OFFSETS, device=device, dtype=torch.long)
    causal_mask = positions[:, None] < offsets_t[None, :]
    return scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))


def fft_dsqg_attention(q, k, v, pos_bias, scale_embed,
                       phase_base, phase_gain, y_pre, z_pre,
                       score_backend="gather"):
    """
    FFT-DSQG attention — dense PyTorch implementation.

    Args:
        q, k, v:       [B, H, N, HD] bfloat16
        pos_bias:      [J=24, H]     float32
        scale_embed:   [J=24, HD]    float32
        phase_base:    [10, H, 2]    float32
        phase_gain:    [10, H, 2]    float32
        y_pre:         [B, H, N, 2]  float32
        z_pre:         [B, H, N, 2]  float32
        score_backend: "gather" or "fft"
    Returns:
        out: [B, H, N, HD] bfloat16
    """
    B, H, N, HD = q.shape
    device = q.device

    q_f = q.float()
    k_f = k.float()
    positions = torch.arange(N, device=device)

    pos_bias_f = pos_bias.float()
    scale_embed_f = scale_embed.float()
    phase_base_f = phase_base.float()
    phase_gain_f = phase_gain.float()

    k_padded = F.pad(k_f, (0, 0, MAX_DELTA, 0))
    if score_backend == "fft":
        scores = _scores_via_fft(q_f, k_f, pos_bias_f, scale_embed_f, positions)
    else:
        scores = _scores_via_gather(q_f, k_padded, pos_bias_f, scale_embed_f, positions)

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    v_padded = F.pad(v.float(), (0, 0, MAX_DELTA, 0))
    z_padded = F.pad(z_pre, (0, 0, MAX_DELTA, 0))

    acc = torch.zeros(B, H, N, HD, device=device, dtype=torch.float32)
    for j in range(J):
        delta = ALL_OFFSETS[j]
        start = MAX_DELTA - delta
        V_j = v_padded[:, :, start:start + N, :]
        w_j = attn[:, :, :, j].unsqueeze(-1)

        if j >= J_SMALL:
            pi = j - J_SMALL
            z_j = z_padded[:, :, start:start + N, :]

            theta0 = (phase_base_f[pi, :, 0][None, :, None]
                      + phase_gain_f[pi, :, 0][None, :, None]
                      * y_pre[:, :, :, 0] * z_j[:, :, :, 0])
            theta1 = (phase_base_f[pi, :, 1][None, :, None]
                      + phase_gain_f[pi, :, 1][None, :, None]
                      * y_pre[:, :, :, 1] * z_j[:, :, :, 1])

            cos0, sin0 = torch.cos(theta0), torch.sin(theta0)
            cos1, sin1 = torch.cos(theta1), torch.sin(theta1)

            v0, v1 = V_j[:, :, :, 0], V_j[:, :, :, 1]
            v2, v3 = V_j[:, :, :, 2], V_j[:, :, :, 3]

            rot = torch.stack([
                cos0 * v0 - sin0 * v1,
                sin0 * v0 + cos0 * v1,
                cos1 * v2 - sin1 * v3,
                sin1 * v2 + cos1 * v3,
            ], dim=-1)

            V_j = torch.cat([rot, V_j[:, :, :, 4:]], dim=-1) if HD > 4 else rot

        acc = acc + w_j * V_j

    return acc.to(q.dtype)


class FFTDSQGAttention(nn.Module):
    """
    FFT-DSQG: Dense PyTorch attention, drop-in replacement for DSQGAttentionV8.

    Checkpoint-compatible with V8/V13 (same parameter names and shapes).
    Backward via native autograd — no custom backward kernel needed.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets=None,
        scale_embed_init=0.0,
        seq_len=2048,
        dropout=0.1,
        score_backend="gather",
    ):
        super().__init__()
        D = embedding_dim
        H = num_heads
        self.num_heads = H
        self.head_dim = D // H
        HD = self.head_dim
        self.seq_len = seq_len
        self.score_backend = score_backend

        assert HD >= 4, "HD must be >= 4 for 2-plane Givens rotation"

        offsets_list = offsets if offsets is not None else ALL_OFFSETS
        assert len(offsets_list) == J, f"Expected {J} offsets, got {len(offsets_list)}"

        self.register_buffer(
            'offset_tensor',
            torch.tensor(offsets_list, dtype=torch.int32),
            persistent=False,
        )

        self.qkv_proj = nn.Linear(D, 3 * D, bias=True)
        self.out_proj = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, H)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets_list], dtype=torch.float32
        )
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.full((J, HD), scale_embed_init))
        self.if_gain = nn.Parameter(torch.ones(H))

        self.phase_base = nn.Parameter(torch.zeros(J_LARGE, H, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(J_LARGE, H, R_PLANES))
        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.npci_theta_k = nn.Parameter(torch.zeros(H))
        self.npci_theta_v = nn.Parameter(torch.zeros(H))

        self.ema_alpha = nn.Parameter(torch.tensor(0.0208))
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
            'bhnd,rd->bhnr', q.float(), self.query_probes.float()
        ).mul(sc).contiguous()
        z_pre = torch.einsum(
            'bhnd,rd->bhnr', k.float(), self.key_probes.float()
        ).mul(sc).contiguous()

        out = fft_dsqg_attention(
            q, k, v,
            self.pos_bias, self.scale_embed,
            self.phase_base, self.phase_gain,
            y_pre, z_pre,
            score_backend=self.score_backend,
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D).contiguous()
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
            'fft_kernel': f'dense_pytorch_{self.score_backend}',
        }
