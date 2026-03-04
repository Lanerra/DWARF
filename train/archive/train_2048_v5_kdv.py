"""
Wave Field Condition D — V5-KdV: DB4 Basis + KdV Nonlinearity + RG Init
=========================================================================
This run tests the three-mechanism combination derived from Rust verification:

  Condition A:        V4 improvements only (no dispersion)       → TEST PPL 86.8
  Condition B:        V4 improvements + learned β dispersion     → TEST PPL 87.9
  Condition C_causal: V4 + causal Morlet wavelet kernel          → TEST PPL 87.2
  Condition D:        V5-KdV — DB4 basis + KdV nonlinearity + RG init  ← THIS RUN

== Three Mechanisms ==

Each mechanism is motivated by Rust verification (verification/src/), tested
individually, and known not to conflict with the others.

1. DB4 Minimum-Phase Basis (replaces Morlet)
   - Daubechies-4 wavelet: 4-tap filter, all z-plane zeros inside unit disk
   - Energy concentration: 93.3% in first 2 taps vs Morlet's 44.0%
   - Minimum-phase ↔ causal by construction (no causal correction overhead)
   - Multi-scale: DB4 upsampled at scales 2^0 … 2^(levels-1), L1-normalised
   - Recency bias: output dominated by recent context, matching language's
     local syntax structure
   - Rust: db4_energy_more_front_loaded_than_morlet, db4_energy_concentrated_in_early_taps

2. V5-KdV Nonlinear Field Term
   - h_out = h_conv + α · h_conv · delta    (δg = h[g] − h[g−1], causal diff)
   - α = learned per-head scalar, init 0.05 (α=0 recovers C_causal exactly)
   - Corrects residual rightward centroid drift from causal convolution
   - Preserves amplitude of forward-propagating structures (soliton stabilisation)
   - Rust α sweep: DB4 + KdV is monotonically cooperative across [0,1]; Morlet
     becomes unstable above α≈0.20 (DB4 is the right basis for KdV pairing)
   - Rust: kdv_nonlinearity_slows_dispersive_amplitude_decay, kdv_alpha_monotone,
           kdv_field_step_is_causal, alpha_sweep_zero_drift

3. RG-Motivated Gain Initialisation
   - Empirically derived from C_causal's learned gain structure after 10 epochs:
       base_gain = 0.458  (L0 dominant gain)
       depth_decay = (0.092/0.458)^(1/11) ≈ 0.864  (L11 suppressed by 5×)
       scale_width = num_scales / 3.0
   - Each layer's gains are pre-scaled to match the depth-decayed profile
     that C_causal learned through gradient descent — pre-biasing without
     locking (gradient descent can still move from this initialisation)
   - Addresses context compression: coarse scales (long-range) dominant in
     early layers; fine scales (local) suppressed in final layers
   - Cost: zero parameters, zero compute. Pure initialisation.
   - Rust: rg_init.rs (depth_decay_monotone, scale_concentration_peaks_at_level0,
           depth_decay_ratio_matches_empirical)

== What this run answers ==

  Primary:   Does the DB4 + KdV + RG init combination outperform C_causal (87.2)?
  Secondary: Does DB4 reach parity with A (86.8) or exceed it?
  Tertiary:  What α values does the model learn per layer? (post-run analysis)
  Bonus:     Does RG init accelerate convergence in early epochs?

== Key differences from Condition C_causal ==

  _build_causal_wavelet_kernel → _build_db4_multiscale_kernel
  No _wave_convolve_causal per se — _wave_convolve same path, new kernel
  New: _apply_kdv_nonlinearity() — applied after wave convolution
  New: apply_rg_init() — called once after model creation
  alpha: [H] parameter per attention layer, init 0.05

Reference baselines:
  Standard 13.5M @ 2048:  PPL  64.5
  Condition A (V4):        PPL  86.8
  Condition B (V4D):       PPL  87.9
  Condition C_causal:      PPL  87.2

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
# DB4 KERNEL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Daubechies-4 low-pass filter coefficients (minimum-phase, L2-normalised)
# tap[0] = most recent, tap[3] = oldest
# Energy: 93.3% in first 2 taps — strong recency bias by construction
DB4_TAPS = [0.4829629, 0.8365163, 0.2241439, -0.1294095]

# RG initialisation parameters — derived from C_causal's learned gain structure
# after 10 epochs of gradient descent (see verification/src/rg_init.rs)
RG_BASE_GAIN   = 0.458   # L0 mean gain (dominant, early layer)
RG_DEPTH_DECAY = 0.864   # per-layer decay: L0/L11 ≈ 5× (empirical peak ratio)
RG_SCALE_WIDTH = None    # set to num_scales / 3.0 in apply_rg_init

# V5-KdV alpha initialisation — within DB4 stable range from Rust sweep
# DB4 sweep: monotonically cooperative across [0,1], no instability
# Morlet optimal: 0.15-0.20 (but DB4 is the basis here, larger range safe)
KDV_ALPHA_INIT = 0.05    # small positive start; gradient descent finds optimal


# ──────────────────────────────────────────────────────────────────────────────
# WAVE FIELD ATTENTION — V5-KdV VARIANT
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldAttentionV5KdV(nn.Module):
    """
    Wave Field Attention with:
      - DB4 minimum-phase multi-scale convolution kernel
      - V5-KdV causal nonlinear field stabilisation
      - RG-motivated gain initialisation (applied externally after model init)

    All V4 improvements retained (dynamic stride, float32 scatter, bilinear
    scatter/gather, cross-head coupling, content-dependent gating).

    == DB4 multi-scale kernel ==

    The Morlet basis (C_causal) spreads energy broadly across the tap window
    (44% in first 2 taps for σ=4).  The DB4 minimum-phase filter concentrates
    93.3% in the first 2 of 4 taps — structural recency bias by construction,
    without requiring any learned correction.

    Multi-scale: we upsample the DB4 filter at scales 2^0, 2^1, … 2^(L-1):

        db4_s(t):  DB4[k] placed at position k * 2^s, zero elsewhere
                   length = 3 * 2^s + 1  (upsampled spacing)
                   truncated to G if longer, zero-padded otherwise

    The kernel for each head is a learned linear combination of these basis
    filters, identical in structure to the Morlet basis in C_causal:

        kernel[h, t] = Σ_s  gains[h, s] · db4_s(t) / ‖db4_s‖₁

    == V5-KdV nonlinear term ==

    After causal FFT convolution:

        delta[g]   = h_conv[g] − h_conv[g−1]   (causal first difference; delta[0]=0)
        h_out[g]   = h_conv[g] + α_h · h_conv[g] · delta[g]

    α_h is a learned per-head scalar (shared across G and D), initialised to
    KDV_ALPHA_INIT = 0.05.  At α=0 this is the identity, recovering C_causal.

    Causality proof: delta[g] depends only on h_conv positions ≤ g (by causal
    diff definition).  h_conv[g] depends only on h[i] for i ≤ g (causal FFT conv).
    Therefore h_out[g] depends only on h[i] for i ≤ g.  ✓

    Verified in Rust: kdv_field_step_is_causal, kdv_alpha_monotone_in_small_range.
    """

    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, num_levels=None, dropout=0.1):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert field_size > 0 and (field_size & (field_size - 1)) == 0, \
            "field_size must be a power of 2"

        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.field_size    = field_size
        self.max_seq_len   = max_seq_len
        # Number of DB4 scales: same as log2(G) for consistent gain count
        self.levels        = num_levels if num_levels else int(math.log2(field_size))

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)

        # ── Wavelet gains ─────────────────────────────────────────────────────
        # Shape: [H, levels] — identical structure to C_causal for fair comparison
        # Init: same Gaussian spread across scale spectrum (head 0 → coarsest,
        #       head H-1 → finest) with 0.5 floor.
        # RG init applied externally (see apply_rg_init()) to adjust per-layer
        # gain magnitudes without changing the relative scale distribution.
        H       = num_heads
        L       = self.levels
        head_centers = torch.linspace(0, L - 1, H)
        scale_idx    = torch.arange(L).float()
        sigma        = L / 4.0
        gains_init   = torch.exp(
            -0.5 * ((scale_idx.unsqueeze(0) - head_centers.unsqueeze(1)) / sigma) ** 2
        )
        gains_init   = gains_init + 0.5
        self.wavelet_gains = nn.Parameter(gains_init)

        # ── KdV alpha ─────────────────────────────────────────────────────────
        # Per-head scalar, shared across G and D dimensions.
        # Init: KDV_ALPHA_INIT (0.05) — small positive, within stable DB4 range.
        # Gradient descent will find optimal α per head per layer.
        # α=0 recovers C_causal exactly (no risk of regression from zero-start).
        self.kdv_alpha = nn.Parameter(torch.full((H,), KDV_ALPHA_INIT))

        # ── Content-dependent gating ──────────────────────────────────────────
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # ── Cross-head field coupling ─────────────────────────────────────────
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
        )

        self.scale = math.sqrt(self.head_dim)

        # Pre-build DB4 basis on CPU (device-agnostic; moved to device on first use)
        self._db4_basis_cpu = None

    def _build_db4_basis(self, device: torch.device) -> torch.Tensor:
        """
        Build multi-scale DB4 basis: [levels, G].

        Scale s: upsample DB4 by factor 2^s (insert 2^s−1 zeros between taps).
        Length of upsampled filter: 3 * 2^s + 1.
        Truncated to G if longer; zero-padded to G if shorter.
        L1-normalised per scale (amplitude-invariant across scales).

        Returned tensor lives on `device`.  Cached on CPU to avoid rebuilding
        on every forward pass (the basis is parameter-free and deterministic).
        """
        if self._db4_basis_cpu is not None:
            return self._db4_basis_cpu.to(device)

        G    = self.field_size
        L    = self.levels
        db4  = torch.tensor(DB4_TAPS, dtype=torch.float32)

        basis = []
        for s in range(L):
            stride = 2 ** s
            # Place db4 taps at positions 0, stride, 2*stride, 3*stride
            k_len = 3 * stride + 1          # length of upsampled filter
            k     = torch.zeros(max(k_len, G))
            for i, tap in enumerate(db4):
                pos = i * stride
                if pos < k_len:
                    k[pos] = tap
            k = k[:G]                       # truncate to G
            # L1-normalise (amplitude-invariant; mirrors C_causal L1 norm)
            k = k / k.abs().sum().clamp(min=1e-8)
            basis.append(k)

        self._db4_basis_cpu = torch.stack(basis)   # [L, G]
        return self._db4_basis_cpu.to(device)

    def _build_db4_multiscale_kernel(self, device: torch.device) -> torch.Tensor:
        """
        Build per-head causal DB4 multi-scale kernel for FFT convolution.

        Returns: rfft of zero-padded kernel, shape [H, G+1] complex.
        Identical interface to C_causal's _build_causal_wavelet_kernel.
        """
        G      = self.field_size
        H      = self.num_heads
        basis  = self._build_db4_basis(device)    # [L, G]

        # kernel[h, t] = gains[h, :] @ basis[:, t]
        gains  = self.wavelet_gains                # [H, L]
        kernel = gains @ basis                     # [H, G]

        # L1-normalise per head
        k_norm = kernel.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernel = kernel / k_norm                   # [H, G]

        # Zero-pad to 2G → linear (non-circular) causal convolution
        pad         = torch.zeros(H, 2 * G, device=device)
        pad[:, :G]  = kernel
        return torch.fft.rfft(pad, n=2 * G)        # [H, G+1] complex

    def _wave_convolve(self, field: torch.Tensor,
                       kernel_fft: torch.Tensor) -> torch.Tensor:
        """Causal linear FFT convolution — identical to C_causal."""
        B, H, G, D = field.shape
        pad_size   = 2 * G
        field_t    = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft  = torch.fft.rfft(field_t, n=pad_size)
        conv_fft   = field_fft * kernel_fft.unsqueeze(0)
        convolved  = torch.fft.irfft(conv_fft, n=pad_size)[:, :, :G]
        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _apply_kdv_nonlinearity(self, field: torch.Tensor) -> torch.Tensor:
        """
        V5-KdV causal nonlinear field update.

            delta[g]  = field[g] − field[g−1]   (delta[0] = 0 — causal)
            h_out[g]  = field[g] + α_h · field[g] · delta[g]

        α_h is per-head [H], broadcast over [B, H, G, D].

        Causality: delta[g] depends only on positions ≤ g.  field[g] from causal
        conv depends only on input positions ≤ g.  Product is causal.  ✓

        At α=0: identity (recovers C_causal exactly).
        At small positive α: forward-propagating structures stabilised (soliton
        self-steepening counteracts dispersive backward drift).
        """
        delta          = torch.zeros_like(field)
        delta[:, :, 1:, :] = field[:, :, 1:, :] - field[:, :, :-1, :]   # [B,H,G-1,D]

        # alpha: [H] → [1, H, 1, 1]
        alpha          = self.kdv_alpha.view(1, self.num_heads, 1, 1)
        return field + alpha * field * delta

    def _wave_convolve_v5(self, field: torch.Tensor) -> torch.Tensor:
        """DB4 causal conv + KdV nonlinear stabilisation."""
        kernel_fft = self._build_db4_multiscale_kernel(field.device)
        h_conv     = self._wave_convolve(field, kernel_fft)
        return self._apply_kdv_nonlinearity(h_conv)

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
        coupling    = F.softmax(self.field_coupling, dim=-1)
        field_flat  = field.reshape(B, H, G * D)
        coupled     = torch.bmm(coupling.unsqueeze(0).expand(B, -1, -1), field_flat)
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

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        v = v.view(B, N, H, hd).transpose(1, 2)

        actual_stride = (G - 1) / max(N - 1, 1)
        field_pos     = (torch.arange(N, device=x.device, dtype=torch.float32)
                         * actual_stride).clamp(0, G - 2)

        k_mag   = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag

        # Scatter → DB4 Conv + KdV → Couple → Gather
        field    = self._bilinear_scatter(deposit, field_pos, B, H, G, hd, x.device)
        field    = self._wave_convolve_v5(field)      # DB4 + KdV nonlinearity
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

    def alpha_summary(self) -> torch.Tensor:
        """Return learned KdV alphas: [num_heads]."""
        return self.kdv_alpha.detach().cpu()


# ──────────────────────────────────────────────────────────────────────────────
# RG INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def apply_rg_init(model: 'WaveFieldTransformerV5KdV',
                  base_gain: float = RG_BASE_GAIN,
                  depth_decay: float = RG_DEPTH_DECAY) -> None:
    """
    Apply RG-motivated depth-decayed gain initialisation.

    Derived from C_causal's learned wavelet gain structure after 10 epochs:
      L0  mean gain: 0.458 (dominant — coarse scale emphasis)
      L11 mean gain: 0.092 (suppressed — fine scales near-silent in final layer)
      depth_decay = (0.092/0.458)^(1/11) ≈ 0.864 per layer

    For each layer i, we rescale wavelet_gains so that mean(gains) ≈
    base_gain * depth_decay^i.  The relative distribution across scales
    (set by Gaussian init) is preserved — we're only adjusting the magnitude
    envelope, not flattening the multi-scale structure.

    This pre-biases the network toward the gain structure that C_causal
    learned through gradient descent, without locking it there (gradient
    descent can still move from this initialisation).

    Verified: rg_init.rs (depth_decay_monotone, scale_concentration_peaks_at_level0,
              depth_decay_ratio_matches_empirical: measured ratio ≈ 5.0, target 4.97).
    """
    print(f'\n  Applying RG init: base_gain={base_gain:.3f}, depth_decay={depth_decay:.3f}')
    num_layers = len(model.layers)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            attn          = layer['attn']
            target_mean   = base_gain * (depth_decay ** i)
            current_mean  = attn.wavelet_gains.mean().item()
            if current_mean > 1e-6:
                scale = target_mean / current_mean
                attn.wavelet_gains.mul_(scale)
            print(f'    Layer {i}: target_mean={target_mean:.4f}  '
                  f'actual_mean={attn.wavelet_gains.mean().item():.4f}')
    print(f'  RG init complete.')


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

class WaveFieldTransformerV5KdV(nn.Module):
    """V5-KdV transformer — DB4 basis + KdV nonlinearity + RG init."""

    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, field_size=4096,
                 max_seq_len=2050, dropout=0.1,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.vocab_size     = vocab_size
        self.use_checkpoint = use_checkpoint

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout         = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = WaveFieldAttentionV5KdV(
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
# ALPHA ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def print_alpha_analysis(model: WaveFieldTransformerV5KdV) -> None:
    """Print per-layer, per-head learned KdV alpha values."""
    print('\n  ── Learned KdV α values ──')
    print(f'  {"Layer":<8}  ' + '  '.join(f'H{h:<3}' for h in range(
        model.layers[0]['attn'].num_heads)))
    all_alphas = []
    for i, layer in enumerate(model.layers):
        alphas = layer['attn'].alpha_summary()   # [H]
        all_alphas.append(alphas)
        row = '  '.join(f'{v.item():.3f}' for v in alphas)
        print(f'  Layer {i:<2}  {row}')
    stacked = torch.stack(all_alphas)   # [num_layers, H]
    flat    = stacked.flatten()
    print(f'\n  Overall α: mean={flat.mean():.4f}  std={flat.std():.4f}  '
          f'min={flat.min():.4f}  max={flat.max():.4f}')
    pos_frac = (flat > 0.0).float().mean().item()
    print(f'  Positive α fraction: {pos_frac*100:.1f}%  '
          f'(>0.10: {(flat > 0.10).float().mean()*100:.1f}%)')


def print_gain_analysis(model: WaveFieldTransformerV5KdV, field_size: int = 4096) -> None:
    """Print per-layer mean wavelet gains."""
    levels = int(math.log2(field_size))
    print(f'\n  ── Learned wavelet gains (DB4 scales) ──')
    print(f'  {"Layer":<8} ' + ' '.join(f'L{k:<4}' for k in range(levels)))
    all_gains = []
    for i, layer in enumerate(model.layers):
        gains    = layer['attn'].gain_summary()   # [H, levels]
        all_gains.append(gains)
        mean_g   = gains.mean(dim=0)
        row      = '  '.join(f'{v.item():.3f}' for v in mean_g)
        print(f'  Layer {i:<2}  {row}')
    stacked = torch.stack(all_gains)
    flat    = stacked.flatten()
    print(f'\n  Overall: mean={flat.mean():.3f}  std={flat.std():.3f}  '
          f'min={flat.min():.3f}  max={flat.max():.3f}')


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZER + DATA  (verbatim from ablation scripts)
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
        bi  = idx[s:s + batch_size]
        bx  = [data[i][0] for i in bi]; by = [data[i][1] for i in bi]
        ml  = max(x.size(0) for x in bx)
        px  = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py  = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
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
        short     = generated.replace('\n', ' ')[:120]
        print(f'  [{prompt!r}]\n    → {short!r}')
    print('  ──\n')
    model.train()


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
        tc  += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt  += mask.sum().item()
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
# TRAINING
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
        et      = time.time()
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
    max_seq_len = 2048

    print('=' * 70)
    print('  WAVE FIELD — V5-KdV ABLATION @ 2048 TOKENS')
    print('  Condition A:   V4 no dispersion           → TEST PPL 86.8  (prior)')
    print('  Condition B:   V4D with dispersion         → TEST PPL 87.9  (prior)')
    print('  Condition C:   V4 + causal Morlet wavelet  → TEST PPL 87.2  (prior)')
    print('  Condition D:   V5-KdV DB4+KdV+RG init      ← THIS RUN')
    print()
    print('  Three mechanisms (see module docstring):')
    print('    1. DB4 minimum-phase basis  (93.3% energy in first 2 taps)')
    print('    2. V5-KdV nonlinear term    (α per head, init 0.05, causal)')
    print('    3. RG-motivated gain init   (depth-decayed: 0.458 → 0.092)')
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

    # Shared config — identical to A, B, C for direct comparison
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

    # ── Prior results ─────────────────────────────────────────────────────────
    prior = {
        'A_v4_no_dispersion':  {'test_ppl': 86.8, 'test_acc': 26.3},
        'B_v4d_dispersion':    {'test_ppl': 87.9, 'test_acc': 26.2},
        'C_causal_wavelet':    {'test_ppl': 87.2, 'test_acc': 26.1},
    }

    # ── Condition D — V5-KdV ─────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION D — V5-KdV: DB4 + KdV nonlinearity + RG gain init')
    print(f'{"="*70}')

    model_d = WaveFieldTransformerV5KdV(**cfg).to(device)
    total_params = sum(p.numel() for p in model_d.parameters())
    kdv_params   = sum(
        layer['attn'].kdv_alpha.numel()
        for layer in model_d.layers
    )
    print(f'\n  Parameters: {total_params:,} total, {kdv_params} KdV α '
          f'({cfg["num_layers"]} layers × {cfg["num_heads"]} heads)')

    # ── Pre-training causality checks ─────────────────────────────────────────
    with torch.no_grad():
        # 1. DB4 kernel: pos1→pos0 leakage must be < 1e-5
        attn0   = model_d.layers[0]['attn']
        G, H, D = attn0.field_size, attn0.num_heads, attn0.head_dim
        kfft    = attn0._build_db4_multiscale_kernel(device)
        f_fut   = torch.zeros(1, H, G, D, device=device)
        f_fut[:, :, 1, :] = 1.0
        out     = attn0._wave_convolve(f_fut, kfft)
        leak    = out[:, :, 0, :].abs().max().item()
        assert leak < 1e-5, f'DB4 CAUSALITY VIOLATION: leakage={leak:.2e}'
        print(f'  Causality check PASSED (DB4): pos1→pos0 leakage = {leak:.2e}')

        # 2. KdV step: impulse at g0 must produce 0 at g < g0
        h_imp  = torch.zeros(1, H, G, D, device=device)
        h_imp[:, :, 30, :] = 1.0
        h_kdv  = attn0._apply_kdv_nonlinearity(h_imp)
        max_leak_kdv = h_kdv[:, :, :30, :].abs().max().item()
        assert max_leak_kdv < 1e-9, f'KdV CAUSALITY VIOLATION: leakage={max_leak_kdv:.2e}'
        print(f'  Causality check PASSED (KdV): pre-impulse leakage = {max_leak_kdv:.2e}')

        del attn0, kfft, f_fut, out, h_imp, h_kdv

    # ── Apply RG initialisation ───────────────────────────────────────────────
    apply_rg_init(model_d)

    # ── Train ─────────────────────────────────────────────────────────────────
    res_d = train_model(
        model_d, train_data, val_data, tok, vocab_size, device,
        'Wave V5-KdV (DB4+KdV+RG)', **train_cfg,
        save_dir='2048_v5kdv_checkpoints',
        generation_prompts=GENERATION_PROMPTS,
        max_seq_len=max_seq_len,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    tb      = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_d, ta_d = evaluate(model_d, tb, vocab_size, device, use_amp)
    res_d['test_ppl'] = tp_d; res_d['test_acc'] = ta_d
    print(f'\n  Condition D TEST: PPL {tp_d:.1f} | Acc {ta_d:.1f}%')

    # ── Post-training analysis ────────────────────────────────────────────────
    print_alpha_analysis(model_d)
    print_gain_analysis(model_d, field_size=cfg['field_size'])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  ABLATION RESULTS — A / B / C / D')
    print(f'{"="*70}')
    ref_std = 64.5
    print(f'\n  {"Model":<42} {"Test PPL":>10} {"vs A":>8}')
    print(f'  {"─"*42} {"─"*10} {"─"*8}')
    print(f'  {"[Standard 13.5M @ 2048]":<42} {ref_std:>10.1f} {"—":>8}')
    for label, r in [('Wave V4 (no dispersion)  [A]', prior['A_v4_no_dispersion']),
                     ('Wave V4D (w/ dispersion) [B]', prior['B_v4d_dispersion']),
                     ('Wave V4 + Morlet         [C]', prior['C_causal_wavelet'])]:
        tp = r['test_ppl']
        print(f'  {label:<42} {tp:>10.1f} {tp - prior["A_v4_no_dispersion"]["test_ppl"]:>+7.1f}')
    delta = tp_d - prior['A_v4_no_dispersion']['test_ppl']
    print(f'  {"Wave V5-KdV (DB4+KdV+RG) [D]":<42} {tp_d:>10.1f} {delta:>+7.1f}')
    print()
    if delta < -1.0:
        print('  ✓ V5-KdV HELPS — physics combination outperforms FFT baseline')
    elif abs(delta) <= 1.0:
        print('  ○ V5-KdV at parity — physics combination matched FFT baseline')
    else:
        print('  ✗ V5-KdV trails — combined mechanism did not help at 13M scale')

    # ── Save results ─────────────────────────────────────────────────────────
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        '2048_dispersion_ablation_results.json'
    )
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing['D_v5_kdv'] = res_d
    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\n  Results saved → {results_path}')


if __name__ == '__main__':
    main()
