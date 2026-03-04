"""
Wave Field Condition C_causal — Causal Wavelet Kernel Attention @ 2048 Tokens
==============================================================================
Direct continuation of the A/B dispersion ablation.

  Condition A:        V4 improvements only (no dispersion)       → TEST PPL 86.8
  Condition B:        V4 improvements + learned β dispersion     → TEST PPL 87.9
  Condition C_acausal: V4 + Haar DWT (non-causal) — INVALID      PPL 7.3 ep2
  Condition C_causal:  V4 + causal Morlet wavelet kernel         ← THIS RUN

== Why C_gaussian was invalid ==

C_gaussian (Haar DWT butterfly, Gaussian gain init) reached PPL 7.3 / Acc 68.9%
after epoch 2 but generated "yyyyyyy..." for all prompts — a classic
shortcut-learner signature.  Root cause: the standard Haar DWT butterfly is
NON-CAUSAL.  At every level it mixes field[2k] with field[2k+1] (the *next*
field position), propagating future token information backward through all
levels.  The model learned to exploit this shortcut during training, but it
disappears at inference time (future tokens don't exist yet during generation).
This is the same bug as Condition B's β=1 run, just with weaker leakage
(local vs global) hence slower exploitation.

== Condition C_causal design ==

The fix: abandon DWT as the forward-pass operator.  Instead, parameterise
the convolution kernel as a linear combination of Morlet (Gaussian-windowed
cosine) wavelets at log-spaced scales, fed into the existing causal FFT
convolution path (proven correct in Conditions A and B).

  kernel[h, t] = Σ_s  gains[h, s] · morlet_s(t)     t ∈ [0, G-1]

  morlet_s(t)  = exp(-t² / (2σ_s²)) · cos(2πt / (2σ_s)) / ‖·‖₁
  σ_s          = 2^(L-1-s)    s=0 → coarsest (σ≈2048), s=L-1 → finest (σ=1)

  · Causal by construction — kernel support at t ≥ 0 only
  · Multi-scale — 12 independent scales per head, exactly as DWT intended
  · Same Gaussian gain init maps heads across coarse→fine spectrum
  · Gradient: gains → materialise kernel → rfft → multiply → irfft

  Complexity: O(G log G) — same as Condition A (causal FFT path)

== Per-epoch text generation ==

After each epoch's val evaluation, the model generates continuations of a
fixed prompt set (greedy, temperature=0). This lets us observe qualitative
improvement alongside PPL metrics — especially useful for comparing whether
DWT's multi-scale structure helps coherent text emerge earlier in training.

== What this run answers ==

  Primary: Does DWT-based field convolution match Condition A PPL (86.8)?
  Secondary: Does it exceed it? (multi-scale inductive bias helping)
  Tertiary: When do generations become coherent relative to PPL milestone?
  Bonus: Training wall-clock time vs A (should be faster: O(G) vs O(G log G))

Reference baselines:
  Standard 13.5M @ 2048:  PPL  64.5
  Condition A (V4):        PPL  86.8
  Condition B (V4D):       PPL  87.9

Results will be appended to 2048_dispersion_ablation_results.json.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import gc

# ──────────────────────────────────────────────────────────────────────────────
# DWT HELPERS  (differentiable; autograd tracks through all ops)
# ──────────────────────────────────────────────────────────────────────────────

_SQRT2_INV = 1.0 / math.sqrt(2.0)


def haar_dwt_field(field: torch.Tensor) -> torch.Tensor:
    """
    Full multi-level Haar DWT over the G (position) dimension.

    Input:  [B, H, G, D]  — G must be a power of 2
    Output: [B, H, G, D]  — packed wavelet coefficients:
              [0:1, :]         approximation  (1 coeff,   coarsest scale)
              [1:2, :]         level-1 detail (1 coeff)
              [2:4, :]         level-2 detail (2 coeffs)
              ...
              [G//2:G, :]      level-L detail (G/2 coeffs, finest scale)

    Complexity: O(G) — total ops across all levels sum to 2G.
    """
    B, H, G, D = field.shape
    assert G > 0 and (G & (G - 1)) == 0, f"G={G} must be a power of 2"
    result = field.clone()
    length = G
    while length > 1:
        half = length >> 1
        chunk = result[:, :, :length, :].clone()   # read before overwriting
        even  = chunk[:, :, 0::2, :]               # [B, H, half, D]
        odd   = chunk[:, :, 1::2, :]               # [B, H, half, D]
        result[:, :, :half,   :] = (even + odd) * _SQRT2_INV   # approx
        result[:, :, half:length, :] = (even - odd) * _SQRT2_INV  # detail
        length = half
    return result


def haar_idwt_field(coeffs: torch.Tensor) -> torch.Tensor:
    """
    Full multi-level Haar IDWT — inverse of haar_dwt_field.

    Input/output: [B, H, G, D]
    """
    B, H, G, D = coeffs.shape
    assert G > 0 and (G & (G - 1)) == 0, f"G={G} must be a power of 2"
    result = coeffs.clone()
    length = 2
    while length <= G:
        half = length >> 1
        approx = result[:, :, :half,        :].clone()
        detail = result[:, :, half:length,  :].clone()
        result[:, :, 0:length:2, :] = (approx + detail) * _SQRT2_INV   # even
        result[:, :, 1:length:2, :] = (approx - detail) * _SQRT2_INV   # odd
        length <<= 1
    return result


def apply_wavelet_gains(coeffs: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """
    Apply per-head, per-scale gains to packed DWT coefficients.

    coeffs: [B, H, G, D]
    gains:  [H, levels]  — levels = log2(G)
              gains[:, 0]      → approximation band (index 0)
              gains[:, 1]      → coarsest detail    (index 1)
              gains[:, 2]      → next detail         (indices 2..3)
              ...
              gains[:, L-1]    → finest detail       (indices G/2..G-1)

    This is the wavelet analogue of multiplying by a frequency-domain kernel:
    each scale band's contribution is independently scaled per head.
    """
    B, H, G, D = coeffs.shape
    levels = int(math.log2(G))
    assert gains.shape == (H, levels), \
        f"gains must be [{H}, {levels}], got {list(gains.shape)}"

    result = coeffs.clone()

    # Approximation band: index 0, gains[:, 0]
    result[:, :, 0:1, :] = coeffs[:, :, 0:1, :] * gains[:, 0].view(1, H, 1, 1)

    # Detail bands: level k has band [2^(k-1) : 2^k] and uses gains[:, k]
    # k=1 → [1:2], k=2 → [2:4], ..., k=levels-1 → [G//4:G//2], k=levels-1 finest
    # Note: after approx at 0, coarsest detail is at 1, finest is G//2..G-1
    band_start = 1
    for k in range(1, levels):
        band_end = band_start * 2
        result[:, :, band_start:band_end, :] = \
            coeffs[:, :, band_start:band_end, :] * gains[:, k].view(1, H, 1, 1)
        band_start = band_end

    # Finest detail band: band_start..G, gains[:, levels-1]
    result[:, :, band_start:G, :] = \
        coeffs[:, :, band_start:G, :] * gains[:, levels - 1].view(1, H, 1, 1)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# WAVE FIELD ATTENTION — WAVELET VARIANT
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldAttentionWavelet(nn.Module):
    """
    Wave Field Attention with V4 improvements + causal wavelet kernel (Condition C_causal).

    Replaces the FFT-based wave kernel (α, ω, φ, β) with a multi-scale
    Morlet wavelet kernel: a learned linear combination of Gaussian-windowed
    cosines at log-spaced scales, convolved causally via zero-padded FFT.

    NOTE — why not DWT-butterfly:
      The original DWT-butterfly (_wave_convolve_dwt) was non-causal.  Each
      level mixed field[2k] with field[2k+1] (the *next* position), leaking
      future field positions backward through all DWT levels.  Empirically
      confirmed by C_gaussian: PPL 7.3 / Acc 68.9% after epoch 2 but
      degenerate "yyyyyyy..." generation — same shortcut-learner signature
      as Condition B β=1 no-causal-fix.  This version fixes it.

    V4 improvements retained:
      - Dynamic stride (positions computed from actual N)
      - Log-scale frequency init for QKV projection diversity
        (not applicable here — kept for consistency in init structure)
      - Float32 scatter accumulation for AMP stability
      - Bilinear scatter / gather
      - Static cross-head field coupling
      - Content-dependent gating

    New:
      - wavelet_gains: [H, levels] — Morlet kernel weights per head per scale
        Gaussian init ensures head diversity from step 1 (see __init__)
      - _build_causal_wavelet_kernel: materialise [H, G] kernel, rfft → [H, G+1]
      - _wave_convolve_causal: kernel + zero-padded FFT conv (causal, O(G log G))
    """

    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, dropout=0.1):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert field_size > 0 and (field_size & (field_size - 1)) == 0, \
            "field_size must be a power of 2 for DWT"

        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.field_size    = field_size
        self.max_seq_len   = max_seq_len
        self.levels        = int(math.log2(field_size))  # 12 for G=4096

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)

        # Per-head, per-scale gains — the "wavelet kernel"
        # Shape: [H, levels] — one gain per DWT scale per head
        #
        # Structured initialisation — analogue of V4's logspace frequency init:
        #   Head 0      → peak gain at scale 0 (coarsest, long-range context)
        #   Head H-1    → peak gain at scale levels-1 (finest, local syntax)
        #   Heads 1..H-2 → Gaussian peaks spread across the scale spectrum
        #
        # This ensures head diversity from step 1, just as V4's logspace ω
        # prevents all heads starting with identical wave kernels.
        # A floor of 0.5 prevents any scale from being completely silenced.
        #
        # Flat init (all 1.0) was empirically tested as C_flat_init:
        #   - Val loss 5.16 after epoch 1 (better than A/B's ~5.95 at ep1,
        #     but degenerate "year-year-year" generation — no scale structure)
        #   - 1577s/epoch (3× slower than A — Python loop loses to cuFFT)
        #   - Zero head diversity until coupling matrix differentiates them
        head_centers = torch.linspace(0, self.levels - 1, num_heads)  # [H]
        scale_idx    = torch.arange(self.levels).float()               # [levels]
        sigma        = self.levels / 4.0
        gains_init   = torch.exp(
            -0.5 * ((scale_idx.unsqueeze(0) - head_centers.unsqueeze(1)) / sigma) ** 2
        )
        gains_init   = gains_init + 0.5   # floor: no scale completely silenced
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
        Build a causal multi-scale kernel for FFT convolution.

        The standard Haar DWT butterfly is NOT causal: at every level it mixes
        field position 2k with position 2k+1 (the *next* position), so the
        DWT→gains→IDWT path leaks future field positions into the current
        position's output.  Empirically confirmed by C_gaussian epoch 2:
        PPL 7.3 but "yyyyyyy..." generation — the model exploited the shortcut
        during training but the shortcut disappears at inference time.

        Fix: abandon DWT as the forward-pass operator.  Instead, parameterise
        the kernel *directly* in the wavelet domain and materialise it as a
        causal time-domain vector, then convolve via the same zero-padded FFT
        path used in Conditions A and B.

        Kernel design — Morlet (Gaussian-windowed cosine) basis:

            wavelet_s(t) = exp(-t² / (2 σ_s²)) · cos(2π t / (2 σ_s))
                           ───────────────────────────────────────────
                                          ‖wavelet_s‖₁

            σ_s = 2^(levels-1-s)   s=0 → coarsest (σ≈2048), s=L-1 → finest (σ=1)

            kernel[h, t] = Σ_s  gains[h, s] · wavelet_s(t)   t ∈ [0, G-1]

        Properties:
          · Causal by construction — support at t ≥ 0 only.
          · Multi-scale — 12 independent scale bands per head, just as DWT had.
          · Same Gaussian gain init maps heads to the same coarse→fine spectrum.
          · Gradient flows: gains → materialise kernel → rfft → multiply → irfft.
          · Re-uses the causal FFT conv path (proven correct in A and B).
        """
        G       = self.field_size
        levels  = self.levels          # 12 for G=4096
        H       = self.num_heads

        t = torch.arange(G, dtype=torch.float32, device=device)          # [G]

        # Scale σ_s: scale 0 = coarsest (σ = G/2), scale L-1 = finest (σ = 1)
        s_vals  = torch.arange(levels, dtype=torch.float32, device=device)
        sigma_s = 2.0 ** (levels - 1 - s_vals)                           # [levels]

        # Morlet basis: [G, levels]
        gauss    = torch.exp(-0.5 * (t.unsqueeze(1) / sigma_s.unsqueeze(0)) ** 2)
        cosine   = torch.cos(2.0 * math.pi * t.unsqueeze(1)
                             / (2.0 * sigma_s.unsqueeze(0)))
        wavelets = gauss * cosine                                         # [G, levels]

        # L1-normalise each basis wavelet so gains are scale-invariant
        norms    = wavelets.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
        wavelets = wavelets / norms                                       # [G, levels]

        # kernel[h, t] = gains[h, :] · wavelets[t, :]
        gains    = self.wavelet_gains                                     # [H, levels]
        kernel   = gains @ wavelets.t()                                   # [H, G]

        # L1-normalise per head (matches V4 wave kernel normalisation)
        k_norm   = kernel.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernel   = kernel / k_norm                                        # [H, G]

        # Zero-pad to 2G → linear (non-circular) causal convolution
        pad      = torch.zeros(H, 2 * G, device=device)
        pad[:, :G] = kernel                                               # t ≥ 0 only
        return torch.fft.rfft(pad, n=2 * G)                              # [H, G+1]

    def _wave_convolve(self, field: torch.Tensor,
                       kernel_fft: torch.Tensor) -> torch.Tensor:
        """
        Causal linear FFT convolution — identical to Condition A/B.

        field:      [B, H, G, D]
        kernel_fft: [H, G+1]  complex (from _build_causal_wavelet_kernel)
        returns:    [B, H, G, D]
        """
        B, H, G, D = field.shape
        pad_size   = 2 * G
        # Reshape to [B*D, H, G] for batched 1-D FFT
        field_t    = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft  = torch.fft.rfft(field_t, n=pad_size)
        conv_fft   = field_fft * kernel_fft.unsqueeze(0)
        convolved  = torch.fft.irfft(conv_fft, n=pad_size)[:, :, :G]
        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _wave_convolve_causal(self, field: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale causal field convolution via wavelet kernel + FFT.

        Replaces the original _wave_convolve_dwt which used the non-causal
        Haar DWT butterfly (acausal: mixed field[2k] with field[2k+1]).
        """
        kernel_fft = self._build_causal_wavelet_kernel(field.device)
        return self._wave_convolve(field, kernel_fft)

    def _bilinear_scatter(self, values, field_pos_float, B, H, G, head_dim, device):
        """Scatter values onto field using bilinear interpolation (V4: float32)."""
        N = field_pos_float.shape[0]
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
        N = field_pos_float.shape[0]
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

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        v = v.view(B, N, H, hd).transpose(1, 2)

        # V4: dynamic stride — always span [0, G-1] regardless of N
        actual_stride = (G - 1) / max(N - 1, 1)
        field_pos     = (torch.arange(N, device=x.device, dtype=torch.float32)
                         * actual_stride).clamp(0, G - 2)

        k_mag   = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag

        # Scatter → Causal Wavelet Convolve → Couple → Gather
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
        """Return learned gains: [num_heads, levels]."""
        return self.wavelet_gains.detach().cpu()


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldTransformerWavelet(nn.Module):
    """Wavelet field transformer — drop-in replacement for V4D ablation."""

    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, field_size=4096,
                 max_seq_len=2050, dropout=0.1,
                 use_checkpoint=False, interference_interval=3,
                 **kwargs):   # absorb unused kwargs (e.g. use_dispersion)
        super().__init__()
        self.vocab_size             = vocab_size
        self.use_checkpoint         = use_checkpoint
        self.interference_interval  = interference_interval

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
# TOKENIZER + DATA  (verbatim from train_2048_dispersion_ablation.py)
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
    ds = load_dataset('openwebtext', split='train', streaming=True)
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
    idx = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
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

# Fixed prompts for per-epoch qualitative tracking.
# Greedy (temperature=0) ensures fully deterministic output per epoch —
# differences between epochs reflect only model weight changes, not sampling.
GENERATION_PROMPTS = [
    "It was a dark and stormy",        # narrative register — common literary opener
    "The length of the hypotenuse",    # technical / mathematical
    "The President of the United",     # factual / knowledge-dependent
    "Once upon a time there was",      # simple narrative — should emerge early
    "The results indicate that",       # formal / academic register
]


@torch.no_grad()
def generate_samples(model, tok, prompts, max_new_tokens=150,
                     temperature=0.0, max_seq_len=2048, device='cuda'):
    """
    Generate continuations for each prompt using greedy decoding.
    Prints a labelled block suitable for epoch logs.
    """
    model.eval()
    print('\n  ── Generation samples (greedy, 150 tokens) ──')
    for prompt in prompts:
        ids        = tok.encode(prompt)
        input_ids  = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            ctx    = input_ids[:, -max_seq_len:]
            logits, _ = model(ctx)
            next_logits = logits[0, -1, :]

            if temperature == 0.0:
                next_id = next_logits.argmax().item()
            else:
                probs   = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1)

        generated = tok.decode(input_ids[0, len(ids):].tolist())
        # Truncate at 120 chars for readable log line
        short = generated.replace('\n', ' ')[:120]
        print(f'  [{prompt!r}]\n    → {short!r}')
    print('  ──\n')
    model.train()


# ──────────────────────────────────────────────────────────────────────────────
# WAVELET GAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def print_gain_analysis(model, name, field_size=4096):
    """Print per-layer, per-head wavelet gains and statistics."""
    levels = int(math.log2(field_size))
    print(f'\n  ── {name} — learned wavelet gains ──')
    print(f'  Scale labels (0=approx/coarsest → {levels-1}=finest):')
    print(f'  {"Layer":<8} ' + ' '.join(f'L{k:<4}' for k in range(levels)))
    print(f'  {"─"*8} ' + '─' * (levels * 6))

    all_gains = []
    for i, layer in enumerate(model.layers):
        attn   = layer['attn']
        gains  = attn.gain_summary()   # [H, levels]
        all_gains.append(gains)
        # Print mean gain per scale across heads
        mean_g = gains.mean(dim=0)
        row    = '  '.join(f'{v.item():.3f}' for v in mean_g)
        print(f'  Layer {i:<2}  {row}  (head mean)')

    stacked = torch.stack(all_gains)  # [L, H, levels]
    flat    = stacked.flatten()
    print(f'\n  Overall: mean={flat.mean():.3f}  std={flat.std():.3f}  '
          f'min={flat.min():.3f}  max={flat.max():.3f}')

    # Check scale differentiation — did different scales learn different gains?
    scale_means = stacked.mean(dim=(0, 1))  # [levels] — mean across layers+heads
    scale_std   = stacked.mean(dim=1).std(dim=0).mean().item()  # cross-head variation
    print(f'\n  Scale mean gains: ' + '  '.join(f'L{k}:{v:.3f}' for k, v in enumerate(scale_means)))
    if scale_std > 0.05:
        print(f'  ✓ Scales DIFFERENTIATED (std={scale_std:.3f}) — multi-scale self-organisation')
    else:
        print(f'  ○ Scales relatively uniform (std={scale_std:.3f})')


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# LR SCHEDULER
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


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING  (with per-epoch generation samples)
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, train_data, val_data, tok, vocab_size, device,
                name, num_epochs=10, batch_size=8, grad_accum=4,
                peak_lr=3e-4, use_amp=True, save_dir='checkpoints',
                generation_prompts=None, max_seq_len=2048):
    os.makedirs(save_dir, exist_ok=True)
    params  = sum(p.numel() for p in model.parameters())
    eff_bs  = batch_size * grad_accum
    print(f'\n  {name}: {params:,} params | batch {batch_size}×{grad_accum}={eff_bs} | lr {peak_lr}')

    opt    = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    spe    = math.ceil(len(train_data) / batch_size)
    sch    = WarmupCosineScheduler(opt, spe * 1, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_vl = float('inf'); best_vp = float('inf')
    best_va = 0; best_ep = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
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

        # ── Per-epoch qualitative samples ─────────────────────────────────────
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
    max_seq_len = 2048

    print('=' * 70)
    print('  WAVE FIELD — WAVELET ABLATION @ 2048 TOKENS')
    print('  Condition A: V4 no dispersion          → TEST PPL 86.8  (prior run)')
    print('  Condition B: V4D with dispersion        → TEST PPL 87.9   (prior run)')
    print('  Condition C: V4 + Haar DWT convolution  ← this run')
    print('=' * 70)

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

    # Shared config — identical to A and B for direct comparison
    cfg = dict(
        vocab_size=vocab_size, embedding_dim=256, num_layers=6,
        num_heads=8, ffn_dim=1024, field_size=4096,
        max_seq_len=max_seq_len + 2, dropout=0.1,
        use_checkpoint=True,
    )
    train_cfg = dict(
        num_epochs=10, batch_size=8, grad_accum=4, peak_lr=3e-4,
        use_amp=use_amp,
    )

    results = {}

    # ── Prior results (hardcoded) ─────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION A — prior run results')
    print(f'{"="*70}')
    res_a = {
        'model_name': 'Wave V4 (no dispersion)',
        'params': 13326352,
        'best_ppl': 87.20659856085048,
        'best_acc': 26.250515601918195,
        'best_epoch': 10,
        'total_time': 5242.925741672516,
        'test_ppl': 86.79913927291058,
        'test_acc': 26.286280589779736,
    }
    print(f'  Condition A TEST: PPL {res_a["test_ppl"]:.1f} | Acc {res_a["test_acc"]:.1f}%')
    results['A_v4_no_dispersion'] = res_a

    # Update this once Condition B completes — insert final test results here.
    print(f'\n{"="*70}')
    print('  CONDITION B — prior run results')
    print(f'{"="*70}')
    res_b = {
        'model_name': 'Wave V4D (with dispersion)',
        'params': 13326360,
        'best_ppl': 88.2,
        'best_acc': 26.1,
        'best_epoch': 10,
        'total_time': 10616.0,   # ~10 epochs × 1062s/epoch (causal correction 2× overhead)
        'test_ppl': 87.9,
        'test_acc': 26.2,
        # β analysis: mean=0.766, std=0.304, min=0.420, max=1.752
        # 48/48 heads: normal dispersion (β>0.1), 0 near-zero, 0 anomalous
        # All heads differentiated — multi-scale self-organisation confirmed
        'beta_mean': 0.766, 'beta_std': 0.304, 'beta_min': 0.420, 'beta_max': 1.752,
    }
    print(f'  Condition B TEST: PPL {res_b["test_ppl"]:.1f} | Acc {res_b["test_acc"]:.1f}%')
    results['B_v4d_with_dispersion'] = res_b

    # ── Condition C: Wavelet field attention ──────────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION C_causal — V4 improvements + causal Morlet wavelet kernel')
    print(f'{"="*70}')

    model_c = WaveFieldTransformerWavelet(**cfg).to(device)

    # Causality smoke test: impulse at future field position must NOT affect
    # the output at position 0.  If it does, we have the same shortcut bug.
    with torch.no_grad():
        _attn0    = model_c.layers[0]['attn']
        _G        = _attn0.field_size
        _H        = _attn0.num_heads
        _D        = _attn0.head_dim
        _kfft     = _attn0._build_causal_wavelet_kernel(device)
        _f_future = torch.zeros(1, _H, _G, _D, device=device)
        _f_future[:, :, 1, :] = 1.0          # impulse at position 1 (future)
        _out      = _attn0._wave_convolve(_f_future, _kfft)
        _leakage  = _out[:, :, 0, :].abs().max().item()
        assert _leakage < 1e-5, f'CAUSALITY VIOLATION: pos1→pos0 leakage={_leakage:.2e}'
        print(f'  Causality check PASSED: pos1→pos0 leakage = {_leakage:.2e}')
        del _attn0, _kfft, _f_future, _out

    # Quick parameter count check
    total_params = sum(p.numel() for p in model_c.parameters())
    wavelet_params = sum(
        p.numel() for layer in model_c.layers
        for p in layer['attn'].wavelet_gains.unsqueeze(0)
    )
    print(f'  Wavelet gains: {wavelet_params} params'
          f'  ({int(math.log2(cfg["field_size"]))} scales × {cfg["num_heads"]} heads × {cfg["num_layers"]} layers)')

    res_c = train_model(
        model_c, train_data, val_data, tok, vocab_size, device,
        'Wave V4 + Causal Wavelet Kernel', **train_cfg,
        save_dir='2048_wavelet_checkpoints',
        generation_prompts=GENERATION_PROMPTS,
        max_seq_len=max_seq_len,
    )

    tc = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_c, ta_c = evaluate(model_c, tc, vocab_size, device, use_amp)
    res_c['test_ppl'] = tp_c; res_c['test_acc'] = ta_c
    print(f'\n  Condition C TEST: PPL {tp_c:.1f} | Acc {ta_c:.1f}%')
    print_gain_analysis(model_c, 'Condition C', field_size=cfg['field_size'])

    del model_c; gc.collect(); torch.cuda.empty_cache()
    results['C_causal_wavelet'] = res_c

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  ABLATION RESULTS — A / B / C')
    print(f'{"="*70}')

    ref_std = 64.5
    tp_a    = res_a['test_ppl']
    tp_b    = res_b.get('test_ppl') or float('nan')

    print(f'\n  {"Model":<40} {"Test PPL":>10} {"vs Standard":>12}')
    print(f'  {"─"*40} {"─"*10} {"─"*12}')
    print(f'  {"[Standard 13.5M @ 2048]":<40} {ref_std:>10.1f} {"—":>12}')
    print(f'  {"Wave V4 (no dispersion)  [A]":<40} {tp_a:>10.1f} '
          f'{(tp_a - ref_std)/ref_std*100:>+11.1f}%')
    if not math.isnan(tp_b):
        print(f'  {"Wave V4D (with dispersion) [B]":<40} {tp_b:>10.1f} '
              f'{(tp_b - ref_std)/ref_std*100:>+11.1f}%')
    else:
        print(f'  {"Wave V4D (with dispersion) [B]":<40} {"pending":>10}')
    print(f'  {"Wave V4 + Causal Wavelet  [C]":<40} {tp_c:>10.1f} '
          f'{(tp_c - ref_std)/ref_std*100:>+11.1f}%')

    delta_ca = tp_c - tp_a
    print(f'\n  C vs A: {delta_ca:+.1f} PPL  ', end='')
    if delta_ca < -1.0:
        print('✓ WAVELET HELPS — causal wavelet kernel outperforms FFT baseline')
    elif delta_ca < 1.0:
        print('○ Essentially matched — DWT parity with FFT at lower compute cost')
    else:
        print('✗ DWT trails — multi-scale structure did not help at this scale')

    # Load existing results file and append C
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        '2048_dispersion_ablation_results.json'
    )
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\n  Results appended → {results_path}')


if __name__ == '__main__':
    main()
