"""
Wave Field 85M — A/B/C Ablation
=================================
GPT-2-comparable scale: ~84.5M parameters, 2048-token context.
(File named train_105m_ablation.py for historical reasons; actual count
confirmed 84,487,456 params for A — output projection is weight-tied to
the embedding, so is not counted separately.)

Direct scale-up of the 13.5M ablation in train_2048_wavelet_ablation.py.
Architecture is identical; all three conditions run fresh (no prior baselines
at this scale).

  Condition A: V4, no dispersion           (wave kernel, no β)
  Condition B: V4D, learned β dispersion   (wave kernel + causal β correction)
  Condition C: V4 + causal Morlet wavelet  (wavelet kernel, no β)

Model config (~84.5M):
  vocab_size:     32,000     (BPE, trained on first 50k OpenWebText docs)
  embedding_dim:  640        (vs 256 at 13.5M)
  num_layers:     12         (vs 6)
  num_heads:      8          (head_dim = 80; vs 32 at 13.5M)
  ffn_dim:        2560       (vs 1024; 4× embedding_dim)
  field_size:     4096       (same — G=N assumption decoupled from model size)
  max_seq_len:    2048       (same)

Parameter count breakdown (verified):
  token_embedding:    32000 × 640          = 20.48M  (tied to output proj)
  per layer ×12:
    QKV:              640 × 1920 + 1920    = 1.23M
    out_proj:         640 × 640  + 640     = 0.41M
    gate_proj:        640 × 640  + 640     = 0.41M
    FFN (fc1+fc2):    640×2560+2560 ×2     = 3.28M
    field_coupling:   8 × 8                = <0.001M
    wave/wavelet:     ~0.000M (very small)
    LayerNorms (×2):  640 × 2 × 2          = 0.003M
    subtotal:                             ≈ 5.33M × 12 = 64.0M
  final_norm:                                         = ~0.001M
  output_proj: weight-tied to embedding → 0 extra
  Total:                                             ≈ 84.5M

Training config:
  batch_size:     4   (× grad_accum 8 = effective 32; lower for 24GB GPUs)
  Adjust for H200/A100: batch_size=32, grad_accum=4 → same effective batch
  peak_lr:        3e-4  (cosine, 1-epoch warmup, min 1e-5)
  optimizer:      AdamW (β₁=0.9, β₂=0.95, ε=1e-8, wd=0.1)  — same as 13.5M
  epochs:         10
  AMP:            True (float16 on CUDA; bfloat16 on H200 if supported)
  grad_clip:      1.0

Estimated runtimes (RTX 4090, batch 4×8=32):
  ~6–8 hours per condition (×3 = ~20–24 hours total)
Estimated runtimes (H200 SXM, batch 32×4=128):
  ~2–3 hours per condition (×3 = ~7–9 hours total, ~$25–33 at $3.59/hr)

Results:
  Saved to 105m_ablation_results.json
  Checkpoints: 105m_A_checkpoints/, 105m_B_checkpoints/, 105m_C_checkpoints/

Reference baselines (13.5M for comparison — NOT matched compute):
  Standard 13.5M @ 2048:  PPL 64.5
  Wave V4     13.5M [A]:  PPL 86.8
  Wave V4D    13.5M [B]:  PPL 87.9
  Wave Morlet 13.5M [C]:  pending (run in progress)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────────────────────
# HARDWARE CONFIG — edit here before launching
# ──────────────────────────────────────────────────────────────────────────────
# RTX 4090 (24 GB):   BATCH_SIZE=4,  GRAD_ACCUM=8   → eff. batch 32
# H200 SXM (80 GB):   BATCH_SIZE=32, GRAD_ACCUM=4   → eff. batch 128
# A100 (40 GB):       BATCH_SIZE=16, GRAD_ACCUM=4   → eff. batch 64
BATCH_SIZE = 32
GRAD_ACCUM = 4
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import gc


# ──────────────────────────────────────────────────────────────────────────────
# WAVE FIELD ATTENTION V4D  (Conditions A and B)
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldAttentionV4D(nn.Module):
    """
    Wave Field Attention with V4 stability improvements + optional dispersion.

    V4 improvements (from 4A Lab fork):
      - Log-scale frequency initialization: octave coverage across heads
      - Better damping init: linspace(-2.0, 0.0) — less aggressive decay
      - Full 2π phase coverage: linspace(0, 2π·(H-1)/H) — head diversity
      - Float32 scatter accumulation for AMP numerical stability
      - +0.05 damping floor — stability guard

    V4D additions (ours):
      - Dynamic stride: actual_stride = (G-1)/(N-1) computed at forward time
        (critical for 2048-token context — V4 fork still has this bug)
      - Learned β per head (if use_dispersion=True):
        K_disp(f) = K(f) · exp(i·β·(f/G)²·2π)
        β=0 → non-dispersive; β>0 → normal; β<0 → anomalous
      - Causal correction: dispersion rotates phase, creating acausal components;
        after rotation, irfft, zero t<0, rfft back — pure overhead, zero info gain
    """

    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, dropout=0.1, use_dispersion=True):
        super().__init__()

        self.embedding_dim  = embedding_dim
        self.num_heads      = num_heads
        self.head_dim       = embedding_dim // num_heads
        self.field_size     = field_size
        self.max_seq_len    = max_seq_len
        self.use_dispersion = use_dispersion

        assert embedding_dim % num_heads == 0

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)

        # V4: log-scale frequency init
        self.wave_frequency = nn.Parameter(
            torch.logspace(math.log10(0.1), math.log10(10.0), num_heads)
        )
        # V4: less aggressive damping init
        self.wave_damping = nn.Parameter(
            torch.linspace(-2.0, 0.0, num_heads)
        )
        # V4: full 2π phase coverage
        self.wave_phase = nn.Parameter(
            torch.linspace(0.0, 2.0 * math.pi * (num_heads - 1) / num_heads, num_heads)
        )

        # V4D: per-head dispersion coefficient β
        # Init to ones (active dispersion start); β=0 kills gradients via the guard.
        if use_dispersion:
            self.wave_dispersion = nn.Parameter(torch.ones(num_heads))
        else:
            self.register_buffer('wave_dispersion', torch.zeros(num_heads))

        # Content-dependent gating (bias=2.0 → starts open)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Cross-head field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
        )

        self.scale = math.sqrt(self.head_dim)

    def _build_wave_kernels(self, device):
        """
        Build left-aligned causal wave kernels with V4 stability improvements
        and optional dispersive phase rotation.

          k(t) = exp(-(α+0.05)·t) · cos(ω·t + φ)   [V4: floor on α]
          Normalised by L1 sum (not softmax — preserves negatives for wave character)

        Dispersive phase (frequency domain):
          K_disp(f) = K(f) · exp(i · β · (f/G)² · 2π)
          Followed by causal correction (irfft → zero t<0 → rfft) to prevent
          acausal energy from β-induced phase rotation.
        """
        G = self.field_size
        H = self.num_heads

        t     = torch.arange(G, dtype=torch.float32, device=device)
        alpha = (F.softplus(self.wave_damping) + 0.05).unsqueeze(1)
        omega = self.wave_frequency.abs().unsqueeze(1)
        phi   = self.wave_phase.unsqueeze(1)

        kernels    = torch.exp(-alpha * t.unsqueeze(0)) * torch.cos(omega * t.unsqueeze(0) + phi)
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels    = kernels / kernel_sum

        kernel_fft = torch.fft.rfft(kernels, n=2 * G)

        beta = self.wave_dispersion.to(device)
        if beta.abs().max() > 1e-8:
            f_indices  = torch.arange(G + 1, dtype=torch.float32, device=device)
            f_norm     = f_indices / G
            disp_phase = beta.unsqueeze(1) * f_norm.unsqueeze(0).pow(2) * 2.0 * math.pi
            disp_rot   = torch.complex(torch.cos(disp_phase), torch.sin(disp_phase))
            kernel_fft = kernel_fft * disp_rot

            # Causal correction: zero negative-time tail introduced by dispersion
            kernel_time         = torch.fft.irfft(kernel_fft, n=2 * G)
            kernel_time[:, G:]  = 0.0
            kernel_fft          = torch.fft.rfft(kernel_time, n=2 * G)

        return kernel_fft

    def _wave_convolve(self, field, kernel_fft):
        """Linear convolution via zero-padded FFT. O(G log G) per head."""
        B, H, G, D = field.shape
        pad_size   = 2 * G
        field_t    = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft  = torch.fft.rfft(field_t, n=pad_size)
        conv_fft   = field_fft * kernel_fft.unsqueeze(0)
        convolved  = torch.fft.irfft(conv_fft, n=pad_size)[:, :, :G]
        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _bilinear_scatter(self, values, field_pos_float, B, H, G, head_dim, device):
        """Scatter values onto field using bilinear interpolation (V4: float32)."""
        N      = field_pos_float.shape[0]
        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)
        field  = torch.zeros(B, H, G, head_dim, device=device, dtype=torch.float32)
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        field.scatter_add_(2, idx_lo_exp, (values * w_lo).float())
        field.scatter_add_(2, idx_hi_exp, (values * w_hi).float())
        return field.to(values.dtype)

    def _bilinear_gather(self, field, field_pos_float):
        """Read from field using bilinear interpolation."""
        B, H, G, D = field.shape
        N      = field_pos_float.shape[0]
        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)
        return (torch.gather(field, 2, idx_lo_exp) * w_lo
              + torch.gather(field, 2, idx_hi_exp) * w_hi)

    def _apply_field_coupling(self, field):
        """Static cross-head field coupling."""
        B, H, G, D = field.shape
        coupling   = F.softmax(self.field_coupling, dim=-1)
        field_flat = field.reshape(B, H, G * D)
        coupled    = torch.bmm(coupling.unsqueeze(0).expand(B, -1, -1), field_flat)
        return coupled.reshape(B, H, G, D)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, D = x.shape
        G       = self.field_size
        H       = self.num_heads
        hd      = self.head_dim

        qkv  = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        v = v.view(B, N, H, hd).transpose(1, 2)

        actual_stride = (G - 1) / max(N - 1, 1)
        field_pos     = (torch.arange(N, device=x.device, dtype=torch.float32)
                         * actual_stride).clamp(0, G - 2)

        k_mag   = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag

        field   = self._bilinear_scatter(deposit, field_pos, B, H, G, hd, x.device)
        kfft    = self._build_wave_kernels(x.device)
        field   = self._wave_convolve(field, kfft)
        field   = self._apply_field_coupling(field)

        gate     = torch.sigmoid(self.gate_proj(x))
        gate     = gate.view(B, N, H, hd).transpose(1, 2)
        gathered = self._bilinear_gather(field, field_pos)
        output   = (gathered * gate).transpose(1, 2).reshape(B, N, D)
        output   = self.out_proj(output)

        return output.squeeze(0) if squeeze else output


# ──────────────────────────────────────────────────────────────────────────────
# WAVE FIELD ATTENTION — WAVELET VARIANT  (Condition C)
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldAttentionWavelet(nn.Module):
    """
    Wave Field Attention with V4 improvements + causal Morlet wavelet kernel.

    Replaces the wave kernel (α, ω, φ, β) with a multi-scale Morlet wavelet
    kernel: a learned linear combination of Gaussian-windowed cosines at
    log-spaced scales, convolved causally via zero-padded FFT.

    Why NOT DWT-butterfly:
      The Haar DWT butterfly is non-causal: at every level it mixes field[2k]
      with field[2k+1] (the *next* position), leaking future field positions
      backward through all DWT levels.  Confirmed empirically by C_gaussian:
      PPL 7.3 / Acc 68.9% after epoch 2 but degenerate generation — shortcut
      identical to Condition B β=1 no-causal-fix.

    Kernel design:
      wavelet_s(t) = exp(-t²/(2σ_s²)) · cos(2πt/(2σ_s)) / ‖·‖₁
      σ_s = 2^(levels-1-s)   s=0 → coarsest (σ≈2048), s=L-1 → finest (σ=1)
      kernel[h, t] = Σ_s  gains[h, s] · wavelet_s(t)   t ∈ [0, G-1]

    Properties:
      · Causal by construction — support at t ≥ 0 only
      · Multi-scale — levels independent scale bands per head
      · Gaussian gain init ensures head diversity from step 1
      · Gradient: gains → materialise kernel → rfft → multiply → irfft
      · O(G log G) — same as Condition A
    """

    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, dropout=0.1):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert field_size > 0 and (field_size & (field_size - 1)) == 0, \
            "field_size must be a power of 2"

        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.field_size    = field_size
        self.max_seq_len   = max_seq_len
        self.levels        = int(math.log2(field_size))  # 12 for G=4096

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)

        # Per-head, per-scale gains — the "wavelet kernel"
        # Gaussian init: head 0 → coarsest scale, head H-1 → finest scale.
        # Floor of 0.5 prevents any scale from being completely silenced.
        head_centers   = torch.linspace(0, self.levels - 1, num_heads)
        scale_idx      = torch.arange(self.levels).float()
        sigma          = self.levels / 4.0
        gains_init     = torch.exp(
            -0.5 * ((scale_idx.unsqueeze(0) - head_centers.unsqueeze(1)) / sigma) ** 2
        ) + 0.5
        self.wavelet_gains = nn.Parameter(gains_init)

        # Content-dependent gating (bias=2.0 → starts open)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Cross-head field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
        )

        self.scale = math.sqrt(self.head_dim)

    def _build_causal_wavelet_kernel(self, device: torch.device) -> torch.Tensor:
        """
        Materialise [H, G] causal Morlet kernel → rfft → [H, G+1] complex.

        The kernel is strictly zero for t < 0 by construction (support [0, G-1]),
        so the same zero-padded FFT path as Conditions A/B gives linear (non-
        circular, causal) convolution without any additional correction step.
        """
        G      = self.field_size
        levels = self.levels
        H      = self.num_heads

        t       = torch.arange(G, dtype=torch.float32, device=device)
        s_vals  = torch.arange(levels, dtype=torch.float32, device=device)
        sigma_s = 2.0 ** (levels - 1 - s_vals)                       # [levels]

        # Morlet basis: [G, levels]
        gauss    = torch.exp(-0.5 * (t.unsqueeze(1) / sigma_s.unsqueeze(0)) ** 2)
        cosine   = torch.cos(2.0 * math.pi * t.unsqueeze(1)
                             / (2.0 * sigma_s.unsqueeze(0)))
        wavelets = gauss * cosine
        norms    = wavelets.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
        wavelets = wavelets / norms                                   # [G, levels]

        # [H, G] kernel = gains @ wavelets.T, then L1-normalise per head
        gains  = self.wavelet_gains                                   # [H, levels]
        kernel = gains @ wavelets.t()                                 # [H, G]
        k_norm = kernel.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernel = kernel / k_norm

        # Zero-pad to 2G, keep causal half only (t ≥ 0 already, pad the rest)
        pad         = torch.zeros(H, 2 * G, device=device)
        pad[:, :G]  = kernel
        return torch.fft.rfft(pad, n=2 * G)                          # [H, G+1]

    def _wave_convolve(self, field: torch.Tensor,
                       kernel_fft: torch.Tensor) -> torch.Tensor:
        """Causal linear FFT convolution — identical to Condition A/B."""
        B, H, G, D = field.shape
        pad_size   = 2 * G
        field_t    = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft  = torch.fft.rfft(field_t, n=pad_size)
        conv_fft   = field_fft * kernel_fft.unsqueeze(0)
        convolved  = torch.fft.irfft(conv_fft, n=pad_size)[:, :, :G]
        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _wave_convolve_causal(self, field: torch.Tensor) -> torch.Tensor:
        """Build wavelet kernel and convolve causally."""
        kernel_fft = self._build_causal_wavelet_kernel(field.device)
        return self._wave_convolve(field, kernel_fft)

    def _bilinear_scatter(self, values, field_pos_float, B, H, G, head_dim, device):
        N      = field_pos_float.shape[0]
        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)
        field  = torch.zeros(B, H, G, head_dim, device=device, dtype=torch.float32)
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        field.scatter_add_(2, idx_lo_exp, (values * w_lo).float())
        field.scatter_add_(2, idx_hi_exp, (values * w_hi).float())
        return field.to(values.dtype)

    def _bilinear_gather(self, field, field_pos_float):
        B, H, G, D = field.shape
        N      = field_pos_float.shape[0]
        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)
        return (torch.gather(field, 2, idx_lo_exp) * w_lo
              + torch.gather(field, 2, idx_hi_exp) * w_hi)

    def _apply_field_coupling(self, field):
        B, H, G, D = field.shape
        coupling   = F.softmax(self.field_coupling, dim=-1)
        field_flat = field.reshape(B, H, G * D)
        coupled    = torch.bmm(coupling.unsqueeze(0).expand(B, -1, -1), field_flat)
        return coupled.reshape(B, H, G, D)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, D = x.shape
        G       = self.field_size
        H       = self.num_heads
        hd      = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        v = v.view(B, N, H, hd).transpose(1, 2)

        actual_stride = (G - 1) / max(N - 1, 1)
        field_pos     = (torch.arange(N, device=x.device, dtype=torch.float32)
                         * actual_stride).clamp(0, G - 2)

        k_mag   = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag

        field    = self._bilinear_scatter(deposit, field_pos, B, H, G, hd, x.device)
        field    = self._wave_convolve_causal(field)
        field    = self._apply_field_coupling(field)

        gate     = torch.sigmoid(self.gate_proj(x))
        gate     = gate.view(B, N, H, hd).transpose(1, 2)
        gathered = self._bilinear_gather(field, field_pos)
        output   = (gathered * gate).transpose(1, 2).reshape(B, N, D)
        output   = self.out_proj(output)

        return output.squeeze(0) if squeeze else output

    def gain_summary(self) -> torch.Tensor:
        return self.wavelet_gains.detach().cpu()


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldTransformerV4D(nn.Module):
    """Wave Field Transformer using V4D attention (Conditions A and B)."""

    def __init__(self, vocab_size, embedding_dim=640, num_layers=12,
                 num_heads=8, ffn_dim=2560, field_size=4096,
                 max_seq_len=2050, dropout=0.1, use_dispersion=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.vocab_size     = vocab_size
        self.use_checkpoint = use_checkpoint

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout         = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = WaveFieldAttentionV4D(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                field_size=field_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_dispersion=use_dispersion,
            )
            ffn = nn.Sequential(
                nn.Linear(embedding_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embedding_dim),
                nn.Dropout(dropout),
            )
            self.layers.append(nn.ModuleDict({'attn': attn, 'ffn': ffn}))

        self.norms  = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.final_norm        = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids))

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = x + torch.utils.checkpoint.checkpoint(
                    layer['attn'], self.norms[i](x))
            else:
                x = x + layer['attn'](self.norms[i](x))
            x = x + layer['ffn'](self.norms2[i](x))

        logits = self.output_projection(self.final_norm(x))
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def dispersion_summary(self):
        betas = []
        for layer in self.layers:
            attn = layer['attn']
            if attn.use_dispersion:
                betas.append(attn.wave_dispersion.detach().cpu())
        return torch.stack(betas) if betas else None


class WaveFieldTransformerWavelet(nn.Module):
    """Wave Field Transformer using Morlet wavelet kernel (Condition C)."""

    def __init__(self, vocab_size, embedding_dim=640, num_layers=12,
                 num_heads=8, ffn_dim=2560, field_size=4096,
                 max_seq_len=2050, dropout=0.1,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.vocab_size     = vocab_size
        self.use_checkpoint = use_checkpoint

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout         = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = WaveFieldAttentionWavelet(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                field_size=field_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            ffn = nn.Sequential(
                nn.Linear(embedding_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embedding_dim),
                nn.Dropout(dropout),
            )
            self.layers.append(nn.ModuleDict({'attn': attn, 'ffn': ffn}))

        self.norms  = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.final_norm        = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids))

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = x + torch.utils.checkpoint.checkpoint(
                    layer['attn'], self.norms[i](x))
            else:
                x = x + layer['attn'](self.norms[i](x))
            x = x + layer['ffn'](self.norms2[i](x))

        logits = self.output_projection(self.final_norm(x))
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZER + DATA
# ──────────────────────────────────────────────────────────────────────────────

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


def load_openwebtext(max_docs=100000):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {max_docs:,} docs)...')
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= max_docs: break
        t = item['text'].strip()
        if len(t) > 50: texts.append(t)
        if (i + 1) % 25000 == 0:
            print(f'  {i+1:,} docs...')
    n  = len(texts)
    t1 = int(n * 0.95); t2 = int(n * 0.975)
    print(f'  {n:,} docs | train {t1:,} | val {t2-t1:,} | test {n-t2:,}')
    return {'train': texts[:t1], 'valid': texts[t1:t2], 'test': texts[t2:]}


def encode_lines(lines, tok, max_seq_len):
    data = []
    for line in lines:
        ids = tok.encode(line)
        if len(ids) < 2: continue
        for s in range(0, len(ids) - 1, max_seq_len):
            chunk = ids[s:s + max_seq_len + 1]
            if len(chunk) >= 2:
                data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=True):
    idx     = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
    batches = []
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s + batch_size]
        bx = [data[i][0] for i in bi]; by = [data[i][1] for i in bi]
        ml = max(x.size(0) for x in bx)
        px = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(bx, by)):
            px[i, :x.size(0)] = x; py[i, :y.size(0)] = y
        batches.append((px, py))
    return batches


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPTS = [
    "It was a dark and stormy",
    "The length of the hypotenuse",
    "The President of the United",
    "Once upon a time there was",
    "The results indicate that",
]


@torch.no_grad()
def generate_samples(model, tok, prompts, max_new_tokens=150,
                     temperature=0.0, max_seq_len=2048, device='cuda'):
    model.eval()
    print('\n  ── Generation samples (greedy, 150 tokens) ──')
    for prompt in prompts:
        ids       = tok.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            ctx         = input_ids[:, -max_seq_len:]
            logits, _   = model(ctx)
            next_logits = logits[0, -1, :]
            if temperature == 0.0:
                next_id = next_logits.argmax().item()
            else:
                probs   = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        generated = tok.decode(input_ids[0, len(ids):].tolist())
        short     = generated.replace('\n', ' ')[:120]
        print(f'  [{prompt!r}]\n    → {short!r}')
    print('  ──\n')
    model.train()


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def print_dispersion_analysis(model, name):
    betas = model.dispersion_summary()
    if betas is None:
        print(f'\n  {name}: dispersion disabled')
        return
    L, H = betas.shape
    print(f'\n  ── {name} — learned β values ──')
    print(f'  {"Layer":<8} ' + ' '.join(f'h{h:<5}' for h in range(H)))
    for i in range(L):
        row = '  '.join(f'{betas[i, h].item():+.3f}' for h in range(H))
        print(f'  Layer {i:<2}  {row}')
    all_b = betas.flatten()
    print(f'\n  Summary: mean={all_b.mean():.3f}  std={all_b.std():.3f}  '
          f'min={all_b.min():.3f}  max={all_b.max():.3f}')
    near_zero = (all_b.abs() < 0.1).sum().item()
    normal    = (all_b >  0.1).sum().item()
    anomalous = (all_b < -0.1).sum().item()
    total     = L * H
    print(f'  Head roles: {near_zero}/{total} near-zero  |  '
          f'{normal}/{total} normal (β>0.1)  |  '
          f'{anomalous}/{total} anomalous (β<-0.1)')
    if all_b.std() > 0.1:
        print(f'  ✓ Heads DIFFERENTIATED — multi-scale self-organisation detected')
    else:
        print(f'  ○ Heads relatively uniform')


def print_gain_analysis(model, name, field_size=4096):
    levels = int(math.log2(field_size))
    print(f'\n  ── {name} — learned wavelet gains ──')
    all_gains = []
    for i, layer in enumerate(model.layers):
        attn   = layer['attn']
        gains  = attn.gain_summary()
        all_gains.append(gains)
        mean_g = gains.mean(dim=0)
        row    = '  '.join(f'{v.item():.3f}' for v in mean_g)
        print(f'  Layer {i:<2}  {row}  (head mean, L0=coarsest→L{levels-1}=finest)')
    stacked    = torch.stack(all_gains)
    flat       = stacked.flatten()
    print(f'\n  Overall: mean={flat.mean():.3f}  std={flat.std():.3f}  '
          f'min={flat.min():.3f}  max={flat.max():.3f}')
    scale_std  = stacked.mean(dim=1).std(dim=0).mean().item()
    if scale_std > 0.05:
        print(f'  ✓ Scales DIFFERENTIATED (std={scale_std:.3f}) — multi-scale self-organisation')
    else:
        print(f'  ○ Scales relatively uniform (std={scale_std:.3f})')


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup    = warmup_steps
        self.total     = total_steps
        self.min_lr    = min_lr
        self.base_lr   = optimizer.param_groups[0]['lr']
        self.step_n    = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base_lr * self.step_n / self.warmup
        else:
            p  = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


@torch.no_grad()
def evaluate(model, batches, vocab_size, device, use_amp=False):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                   ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


def train_model(model, train_data, val_data, tok, vocab_size, device,
                name, num_epochs=10, batch_size=32, grad_accum=4,
                peak_lr=3e-4, use_amp=True, save_dir='checkpoints',
                generation_prompts=None, max_seq_len=2048,
                resume_from=None, start_epoch=1):
    os.makedirs(save_dir, exist_ok=True)
    params  = sum(p.numel() for p in model.parameters())
    eff_bs  = batch_size * grad_accum
    print(f'\n  {name}: {params:,} params | batch {batch_size}×{grad_accum}={eff_bs} | lr {peak_lr}')

    opt    = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    spe    = math.ceil(len(train_data) / batch_size)
    sch    = WarmupCosineScheduler(opt, spe * 1, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if resume_from is not None and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, map_location=device, weights_only=True))
        steps_to_skip = (start_epoch - 1) * math.ceil(spe / grad_accum)
        for _ in range(steps_to_skip):
            sch.step()
        print(f'  Resumed from {resume_from} (starting epoch {start_epoch}, {steps_to_skip} scheduler steps skipped)')

    best_vl = float('inf'); best_vp = float('inf')
    best_va = 0; best_ep = 0
    t0 = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb  = 0, 0
        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(batches):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                       ignore_index=-100) / grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            tl += loss.item() * grad_accum; nb += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % 200 == 0:
                print(f'    Step {step+1}/{len(batches)} | Loss {tl/nb:.4f}', flush=True)

        al = tl / max(nb, 1); et = time.time() - et
        vb = create_batches(val_data, batch_size, device, shuffle=False)
        vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp)

        mk = ''
        if vl < best_vl:
            best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            mk = ' * BEST'

        print(f'  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} '
              f'PPL {vp:.1f} Acc {va:.1f}% | {et:.0f}s{mk}', flush=True)

        if generation_prompts:
            generate_samples(model, tok, generation_prompts,
                             max_new_tokens=150, temperature=0.0,
                             max_seq_len=max_seq_len, device=str(device))

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    return {'model_name': name, 'params': params, 'best_ppl': best_vp,
            'best_acc': best_va, 'best_epoch': best_ep,
            'total_time': time.time() - t0}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Wave Field 85M A/B/C Ablation')
    parser.add_argument('--start_condition', choices=['a', 'b', 'c'], default='a',
                        help='Skip conditions before this one (default: a = run all)')
    parser.add_argument('--resume_a_epoch', type=int, default=None,
                        help='Resume Condition A from this epoch (loads 105m_A_checkpoints/best.pt)')
    args = parser.parse_args()

    max_seq_len = 2048

    print('=' * 70)
    print('  WAVE FIELD 85M — A/B/C ABLATION @ 2048 TOKENS')
    print('  Condition A: V4, no dispersion')
    print('  Condition B: V4D, learned β dispersion + causal correction')
    print('  Condition C: V4 + causal Morlet wavelet kernel')
    print('=' * 70)
    print('\n  Config: embedding=640, layers=12, heads=8, ffn=2560, G=4096')
    print('  Verified ~84.5M parameters per condition (weight-tied output proj)')
    print()
    print(f'  Batch config: {BATCH_SIZE}x{GRAD_ACCUM} = eff. {BATCH_SIZE*GRAD_ACCUM}')
    print('  (Edit BATCH_SIZE / GRAD_ACCUM at top of file to change.)')

    splits = load_openwebtext(max_docs=100000)

    bpe_vocab = 32000
    print(f'\nTraining BPE tokenizer (vocab={bpe_vocab})...')
    raw_tok    = train_bpe_tokenizer(splits['train'][:50000], vocab_size=bpe_vocab)
    tok        = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f'  BPE vocab: {vocab_size} tokens')

    print(f'\n  Encoding data (max_seq_len={max_seq_len})...')
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data   = encode_lines(splits['valid'], tok, max_seq_len)
    test_data  = encode_lines(splits['test'],  tok, max_seq_len)
    print(f'  Train {len(train_data):,} | Val {len(val_data):,} | Test {len(test_data):,}')

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'  Device: {device} | AMP: {use_amp}')

    # ── 105M model config ─────────────────────────────────────────────────────
    cfg = dict(
        vocab_size     = vocab_size,
        embedding_dim  = 640,
        num_layers     = 12,
        num_heads      = 8,
        ffn_dim        = 2560,
        field_size     = 4096,
        max_seq_len    = max_seq_len + 2,
        dropout        = 0.1,
        use_checkpoint = True,   # required at this scale
    )
    train_cfg = dict(
        num_epochs  = 10,
        batch_size  = BATCH_SIZE,
        grad_accum  = GRAD_ACCUM,
        peak_lr     = 3e-4,
        use_amp     = use_amp,
        generation_prompts = GENERATION_PROMPTS,
        max_seq_len        = max_seq_len,
    )

    results = {}

    # ── Condition A: V4 no dispersion ─────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION A — V4, no dispersion')
    print(f'{"="*70}')

    resume_a = None
    start_a  = 1
    if args.resume_a_epoch is not None:
        resume_a = os.path.join('105m_A_checkpoints', 'best.pt')
        start_a  = args.resume_a_epoch
        print(f'  Resuming Condition A from epoch {start_a} ({resume_a})')

    model_a = WaveFieldTransformerV4D(**cfg, use_dispersion=False).to(device)
    res_a   = train_model(
        model_a, train_data, val_data, tok, vocab_size, device,
        'Wave V4 105M (no dispersion)', **train_cfg,
        save_dir='105m_A_checkpoints',
        resume_from=resume_a, start_epoch=start_a,
    )
    tb = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_a, ta_a = evaluate(model_a, tb, vocab_size, device, use_amp)
    res_a['test_ppl'] = tp_a; res_a['test_acc'] = ta_a
    print(f'\n  Condition A TEST: PPL {tp_a:.1f} | Acc {ta_a:.1f}%')
    del model_a; gc.collect(); torch.cuda.empty_cache()
    results['A_v4_no_dispersion_105m'] = res_a

    # ── Condition B: V4D with dispersion ──────────────────────────────────────
    if args.start_condition in ('a', 'b'):
        pass  # run B
    else:
        print('\n  Skipping Condition B (--start_condition=c)')
        res_b = {}
    if args.start_condition in ('a', 'b'):
        print(f'\n{"="*70}')
        print('  CONDITION B — V4D, learned β dispersion + causal correction')
        print(f'{"="*70}')

    model_b = WaveFieldTransformerV4D(**cfg, use_dispersion=True).to(device)
    res_b   = train_model(
        model_b, train_data, val_data, tok, vocab_size, device,
        'Wave V4D 105M (with dispersion)', **train_cfg,
        save_dir='105m_B_checkpoints',
    )
    tb = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_b, ta_b = evaluate(model_b, tb, vocab_size, device, use_amp)
    res_b['test_ppl'] = tp_b; res_b['test_acc'] = ta_b
    print(f'\n  Condition B TEST: PPL {tp_b:.1f} | Acc {ta_b:.1f}%')
    print_dispersion_analysis(model_b, 'Condition B (105M)')
    del model_b; gc.collect(); torch.cuda.empty_cache()
    results['B_v4d_with_dispersion_105m'] = res_b

    # ── Condition C: causal Morlet wavelet ────────────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION C — V4 + causal Morlet wavelet kernel')
    print(f'{"="*70}')

    model_c = WaveFieldTransformerWavelet(**cfg).to(device)

    # Causality smoke test — identical to 13.5M run
    with torch.no_grad():
        _attn0    = model_c.layers[0]['attn']
        _G        = _attn0.field_size
        _H        = _attn0.num_heads
        _D        = _attn0.head_dim
        _kfft     = _attn0._build_causal_wavelet_kernel(device)
        _f_future = torch.zeros(1, _H, _G, _D, device=device)
        _f_future[:, :, 1, :] = 1.0
        _out      = _attn0._wave_convolve(_f_future, _kfft)
        _leakage  = _out[:, :, 0, :].abs().max().item()
        assert _leakage < 1e-5, f'CAUSALITY VIOLATION: pos1→pos0 leakage={_leakage:.2e}'
        print(f'  Causality check PASSED: pos1→pos0 leakage = {_leakage:.2e}')
        del _attn0, _kfft, _f_future, _out

    res_c = train_model(
        model_c, train_data, val_data, tok, vocab_size, device,
        'Wave V4 105M + Causal Wavelet', **train_cfg,
        save_dir='105m_C_checkpoints',
    )
    tb = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_c, ta_c = evaluate(model_c, tb, vocab_size, device, use_amp)
    res_c['test_ppl'] = tp_c; res_c['test_acc'] = ta_c
    print(f'\n  Condition C TEST: PPL {tp_c:.1f} | Acc {ta_c:.1f}%')
    print_gain_analysis(model_c, 'Condition C (105M)', field_size=cfg['field_size'])
    del model_c; gc.collect(); torch.cuda.empty_cache()
    results['C_causal_wavelet_105m'] = res_c

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  ABLATION RESULTS — 105M')
    print(f'{"="*70}')

    print(f'\n  {"Model":<45} {"Test PPL":>10} {"Test Acc":>9}')
    print(f'  {"─"*45} {"─"*10} {"─"*9}')
    for label, res in [('Wave V4      105M [A]', res_a),
                       ('Wave V4D     105M [B]', res_b),
                       ('Wave Wavelet 105M [C]', res_c)]:
        print(f'  {label:<45} {res["test_ppl"]:>10.1f} {res["test_acc"]:>8.1f}%')

    print(f'\n  B vs A: {tp_b - tp_a:+.1f} PPL  (dispersion cost at 105M scale)')
    print(f'  C vs A: {tp_c - tp_a:+.1f} PPL  (wavelet vs wave kernel at 105M scale)')

    # Compare to 13.5M baselines
    print(f'\n  ── Cross-scale comparison ──')
    print(f'  {"Model":<40} {"13.5M PPL":>10} {"105M PPL":>10} {"Δ":>6}')
    print(f'  {"─"*40} {"─"*10} {"─"*10} {"─"*6}')
    ref_a_13m = 86.8; ref_b_13m = 87.9
    print(f'  {"Wave V4 (A)":<40} {ref_a_13m:>10.1f} {tp_a:>10.1f} {tp_a - ref_a_13m:>+5.1f}')
    print(f'  {"Wave V4D (B)":<40} {ref_b_13m:>10.1f} {tp_b:>10.1f} {tp_b - ref_b_13m:>+5.1f}')

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        '105m_ablation_results.json',
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
