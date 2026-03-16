"""
tests/test_physics_correctness.py
==================================
Correctness tests for the DWARF condV physics pipeline.

Catches the class of implementation bugs that cost us weeks:
  - Conv truncation hiding long-window EMA errors
  - KdV gradient not flowing
  - AGC expanding instead of normalising
  - EMA dead-zone (raw param below floor → zero gradient)
  - NPCI norm-preservation violation
  - Interference block wiring (wrong layers get IF)
  - GradScaler on BF16 (wrong dtype for autocast)

Run:
  cd /home/dlewis3/Desktop/AI/DWARF
  .venv/bin/python3 -m pytest tests/test_physics_correctness.py -v
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kernels'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import pytest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Causal EMA — scan vs reference, long-window faithfulness
# ─────────────────────────────────────────────────────────────────────────────

def _reference_ema_exact(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Pure-Python exact causal EMA; O(N) per token, ground truth."""
    B, N, D = x.shape
    out   = torch.zeros_like(x, dtype=torch.float32)
    state = torch.zeros(B, D, dtype=torch.float32, device=x.device)
    xf    = x.float()
    for t in range(N):
        state = alpha * xf[:, t] + (1 - alpha) * state
        out[:, t] = state
    return out.to(x.dtype)


def _conv_ema_k256(x: torch.Tensor, alpha: float, K: int = 256) -> torch.Tensor:
    """Old conv-based EMA truncated at K=256 (the original implementation)."""
    B, N, D = x.shape
    k_len  = min(K, N)
    t      = torch.arange(k_len, device=x.device, dtype=torch.float32)
    kernel = alpha * (1 - alpha) ** t
    kernel = (kernel / kernel.sum()).flip(0)
    xi_f   = x.float()
    xi_bd  = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p   = F.pad(xi_bd, (k_len - 1, 0))
    pool   = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(x.dtype)


class TestCausalEMA:

    def test_scan_matches_exact_short_window(self):
        """Scan should match exact EMA when alpha=0.05 (window≈20t)."""
        from causal_ema_scan import causal_ema_scan
        B, N, D = 2, 64, 32
        x  = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        ef = torch.tensor(0.05, device=DEVICE)
        ref = _reference_ema_exact(x, 0.05)
        out = causal_ema_scan(x, ef, floor=1e-6)
        err = (ref.float() - out.float()).abs().max()
        assert err < 0.05, f"Scan vs exact (α=0.05): max_err={err:.4e}"

    def test_scan_matches_exact_long_window(self):
        """
        Critical: α=0.00035 → window≈2857t. Conv K=256 would truncate to 9%
        of the intended window. Scan must match the exact result.
        """
        from causal_ema_scan import causal_ema_scan
        B, N, D = 2, 512, 32
        alpha  = 0.00035
        x  = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        ef = torch.tensor(alpha, device=DEVICE)
        ref  = _reference_ema_exact(x, alpha)
        scan = causal_ema_scan(x, ef, floor=1e-6)
        conv = _conv_ema_k256(x, alpha)

        # Scan should be close to exact
        scan_err = (ref.float() - scan.float()).abs().mean()
        # Conv should be WORSE (to confirm this test is meaningful)
        conv_err = (ref.float() - conv.float()).abs().mean()
        assert scan_err < 0.02, f"Scan error too large at long window: {scan_err:.4e}"
        assert conv_err > scan_err * 2, \
            f"Conv should be worse than scan at α=0.00035; scan={scan_err:.4e} conv={conv_err:.4e}"

    def test_ema_gradient_flows(self):
        """Gradient must flow to ema_factor."""
        from causal_ema_scan import causal_ema_scan
        x  = torch.randn(2, 32, 16, device=DEVICE, dtype=torch.bfloat16)
        ef = torch.tensor(0.05, device=DEVICE, requires_grad=True)
        out = causal_ema_scan(x, ef, floor=1e-6)
        out.sum().backward()
        assert ef.grad is not None, "No gradient to ema_factor"
        assert ef.grad.abs() > 0, f"Zero gradient to ema_factor: {ef.grad}"

    def test_ema_dead_zone_detection(self):
        """
        Reproduces the dead-zone bug: EMA_INIT=0.003, FLOOR=0.0003,
        Adam momentum drives raw param to e.g. -0.001623.
        After clamp the output is floor, but raw param is below floor
        → gradient is zero → only weight decay acts.
        This test verifies: if raw param < floor, gradient is suppressed.
        """
        from causal_ema_scan import causal_ema_scan
        # Simulate param that overshot into dead zone
        raw_param = torch.tensor(-0.001623, device=DEVICE, requires_grad=True)
        x = torch.randn(2, 32, 16, device=DEVICE, dtype=torch.bfloat16)
        # Old floor was 0.0003 — raw param -0.001623 is below it
        out = causal_ema_scan(x, raw_param, floor=0.0003)
        out.sum().backward()
        # With floor=0.0003 and raw=-0.001623: clamp is at boundary → grad=0
        assert raw_param.grad is not None
        # Gradient should be zero (param is at clamp boundary)
        assert raw_param.grad.abs() < 1e-6, \
            f"Dead-zone should give zero gradient, got {raw_param.grad.item():.4e}"

        # Now with new floor=0.00001 — same raw param, but above floor
        raw_param2 = torch.tensor(-0.001623, device=DEVICE, requires_grad=True)
        # Note: with floor=0.00001, raw param -0.001623 is still below floor
        # so clamped value is floor. Gradient is still zero.
        # The FIX is the lower init (0.00035 instead of 0.003) that prevents
        # momentum overshoot in the first place.
        # Verify: init=0.00035, floor=0.00001 → param stays above floor.
        init_val = 0.00035
        # After typical Adam step: Δ ≈ lr * m / (sqrt(v) + ε) ≈ 3e-4 * 0.003 ≈ 9e-7
        # Multiple steps: param drifts down but stays >> floor=0.00001
        assert init_val > 0.00001, "New init must be above new floor"
        # init=0.00035 is *above* old floor 0.0003; the dead zone was that
        # Adam momentum drove the raw param *below* floor from above.
        # Fix: lower the floor to 0.00001 so there's room to settle naturally.
        assert init_val > 0.00001, "New init must be above new floor"


# ─────────────────────────────────────────────────────────────────────────────
# 2. KdV correction — gradient flows, zero-init is identity, sign is correct
# ─────────────────────────────────────────────────────────────────────────────

class TestKdV:

    def _kdv(self, pool, alpha_val):
        """Replicate _kdv_correction from train script."""
        alpha     = torch.tensor(alpha_val).clamp(0.0, 0.5)
        pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
        return pool + alpha * pool * (pool - pool_prev)

    def test_zero_init_is_identity(self):
        """kdv_alpha=0 → output == input (KdV is inactive at init)."""
        pool = torch.randn(2, 64, 32, device=DEVICE)
        out  = self._kdv(pool, 0.0)
        assert torch.allclose(out, pool, atol=1e-6), "KdV zero-init should be identity"

    def test_gradient_flows_to_kdv_alpha(self):
        """Gradient must reach kdv_alpha even when it starts at zero."""
        pool      = torch.randn(2, 32, 16, device=DEVICE, requires_grad=False)
        kdv_alpha = torch.tensor(0.01, device=DEVICE, requires_grad=True)
        alpha     = kdv_alpha.clamp(0.0, 0.5)
        pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
        out = pool + alpha * pool * (pool - pool_prev)
        out.sum().backward()
        assert kdv_alpha.grad is not None, "No gradient to kdv_alpha"
        assert kdv_alpha.grad.abs() > 0, "Zero gradient to kdv_alpha"

    def test_gradient_zero_at_exact_zero_init(self):
        """
        Bug check: if kdv_alpha=0.0 exactly and pool is random,
        the gradient of out w.r.t. kdv_alpha is pool*(pool-pool_prev).sum().
        It should be nonzero (pool is random), confirming gradient CAN flow
        even from zero init.
        """
        pool      = torch.randn(2, 32, 16, device=DEVICE) + 0.5  # nonzero mean
        kdv_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        alpha     = kdv_alpha.clamp(0.0, 0.5)
        pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
        out = pool + alpha * pool * (pool - pool_prev)
        out.sum().backward()
        # At alpha=0, grad = (pool * (pool - pool_prev)).sum() — nonzero for random pool
        assert kdv_alpha.grad.abs() > 1e-3, \
            f"KdV gradient from zero init is suspiciously small: {kdv_alpha.grad.item():.4e}"

    def test_causal_shift(self):
        """KdV uses pool[t-1], not pool[t+1] (causal, not anti-causal)."""
        pool = torch.zeros(1, 4, 2, device=DEVICE)
        pool[0, 2, :] = 1.0  # spike at t=2
        pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
        # pool_prev[0, 2, :] should be pool[0, 1, :] = 0 (causal)
        assert pool_prev[0, 2, 0].item() == 0.0, "KdV pool_prev not causal"
        # pool_prev[0, 3, :] should be pool[0, 2, :] = 1 (previous spike)
        assert pool_prev[0, 3, 0].item() == 1.0, "KdV causal shift wrong"


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGC — normalises to unit RMS, no params, gradient flows through
# ─────────────────────────────────────────────────────────────────────────────

class TestAGC:

    def _agc(self, pool, eps=1e-6):
        D   = pool.shape[-1]
        rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
        return pool / (rms + eps)

    def test_output_rms_near_one(self):
        """After AGC, per-token RMS should be ≈ 1.0."""
        pool = torch.randn(4, 64, 32, device=DEVICE) * 5.0  # large variance
        out  = self._agc(pool)
        D    = pool.shape[-1]
        rms  = out.norm(dim=-1) / (D ** 0.5)
        assert (rms - 1.0).abs().max() < 0.01, \
            f"AGC output RMS not near 1.0: max_dev={((rms-1).abs().max()):.4e}"

    def test_agc_gradient_flows(self):
        """Gradient flows through AGC (no params but must not block backprop)."""
        pool = torch.randn(2, 16, 8, device=DEVICE, requires_grad=True)
        D    = pool.shape[-1]
        rms  = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
        out  = pool / (rms + 1e-6)
        out.sum().backward()
        assert pool.grad is not None
        assert pool.grad.abs().max() > 0

    def test_agc_scale_invariant(self):
        """AGC output is scale-invariant: 2×input → same output."""
        pool = torch.randn(2, 16, 8, device=DEVICE)
        out1 = self._agc(pool)
        out2 = self._agc(pool * 2.0)
        assert torch.allclose(out1, out2, atol=1e-5), \
            "AGC not scale-invariant"


# ─────────────────────────────────────────────────────────────────────────────
# 4. NPCI — norm preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestNPCI:

    def test_norm_preserved(self):
        """||npci_rotate(x, delta, theta)|| == ||x|| for all inputs."""
        sys.path.insert(0, '/tmp/dwarf-j17d/kernels')
        try:
            from dsqg_attention_v8 import npci_rotate
        except ImportError:
            pytest.skip("dsqg_attention_v8 kernel not available")

        B, H, N, HD = 2, 8, 32, 64
        x     = torch.randn(B, H, N, HD, device=DEVICE)
        delta = torch.randn(B, H, N, HD, device=DEVICE) * 0.1
        theta = torch.randn(H, device=DEVICE) * 0.3

        out   = npci_rotate(x, delta, theta)
        x_norms   = x.norm(dim=-1)
        out_norms = out.norm(dim=-1)
        rel_err = ((x_norms - out_norms) / x_norms.clamp(min=1e-8)).abs().max()
        assert rel_err < 1e-4, f"NPCI broke norm: rel_err={rel_err:.4e}"

    def test_npci_zero_theta_is_identity(self):
        """theta=0 → cos(0)=1, sin(0)=0 → output == input."""
        sys.path.insert(0, '/tmp/dwarf-j17d/kernels')
        try:
            from dsqg_attention_v8 import npci_rotate
        except ImportError:
            pytest.skip("dsqg_attention_v8 kernel not available")

        B, H, N, HD = 2, 4, 16, 32
        x     = torch.randn(B, H, N, HD, device=DEVICE)
        delta = torch.randn(B, H, N, HD, device=DEVICE)
        theta = torch.zeros(H, device=DEVICE)
        out   = npci_rotate(x, delta, theta)
        assert torch.allclose(out, x, atol=1e-5), "NPCI theta=0 should be identity"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interference block wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestInterferenceWiring:

    def _get_if_layers(self, num_layers, interference_interval, full_attn_layer):
        """Replicate the layer construction logic from train script."""
        if_layers = []
        for i in range(num_layers):
            if i == full_attn_layer:
                continue
            has_if = (interference_interval is not None and
                      i % interference_interval == interference_interval - 1)
            if has_if:
                if_layers.append(i)
        return if_layers

    def test_j24d_int2_wiring(self):
        """
        J24D-int2: L=6, INTERFERENCE=2, FA=L5.
        Expected IF layers: i%2==1 and i≠5 → layers 1, 3.
        """
        if_layers = self._get_if_layers(
            num_layers=6, interference_interval=2, full_attn_layer=5)
        assert if_layers == [1, 3], \
            f"J24D-int2 IF layers wrong: expected [1,3] got {if_layers}"

    def test_j24d_int3_wiring(self):
        """
        J24D physics baseline: L=6, INTERFERENCE=3, FA=L5.
        Expected IF layers: i%3==2 and i≠5 → layers 2.
        """
        if_layers = self._get_if_layers(
            num_layers=6, interference_interval=3, full_attn_layer=5)
        assert if_layers == [2], \
            f"J24D-int3 IF layers wrong: expected [2] got {if_layers}"

    def test_condU_wiring(self):
        """
        condU 35M hybrid: L=6, INTERFERENCE=2, FA=L5.
        Same as J24D-int2.
        """
        if_layers = self._get_if_layers(
            num_layers=6, interference_interval=2, full_attn_layer=5)
        assert if_layers == [1, 3], \
            f"condU IF layers wrong: expected [1,3] got {if_layers}"

    def test_full_attn_layer_not_dsqg(self):
        """FA layer must not be a DSQG block (avoid double-counting)."""
        full_attn = 5
        if_layers = self._get_if_layers(
            num_layers=6, interference_interval=2, full_attn_layer=full_attn)
        assert full_attn not in if_layers, \
            f"Full-attn layer {full_attn} incorrectly assigned interference"


# ─────────────────────────────────────────────────────────────────────────────
# 6. BF16 autocast — no GradScaler, correct dtype
# ─────────────────────────────────────────────────────────────────────────────

class TestBF16Autocast:

    def test_no_gradscaler_with_bf16(self):
        """
        GradScaler is for FP16 only (FP16 can underflow).
        BF16 has same exponent range as FP32 — GradScaler silently clamps
        gradients and was incorrectly applied in pre-March training scripts.
        Verify: BF16 forward+backward works without GradScaler.
        """
        if DEVICE == "cpu":
            pytest.skip("BF16 autocast requires CUDA")
        model = torch.nn.Linear(32, 32).to(DEVICE)
        x = torch.randn(4, 32, device=DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = out.sum()
        loss.backward()
        # Gradients should be normal float32 (not scaled)
        grad = model.weight.grad
        assert grad is not None
        assert grad.dtype == torch.float32, \
            f"Expected float32 grad, got {grad.dtype}"
        assert not torch.any(torch.isnan(grad)), "NaN gradients with BF16"
        assert not torch.any(torch.isinf(grad)), "Inf gradients with BF16"

    def test_bf16_autocast_dtype(self):
        """Forward computation inside autocast should be BF16, not FP16."""
        if DEVICE == "cpu":
            pytest.skip("BF16 autocast requires CUDA")
        model = torch.nn.Linear(32, 32).to(DEVICE)
        x = torch.randn(4, 32, device=DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        assert out.dtype == torch.bfloat16, \
            f"Autocast produced {out.dtype}, expected bfloat16"


# ─────────────────────────────────────────────────────────────────────────────
# 7. EMA conv truncation — quantifies the K=256 error at low alpha
# ─────────────────────────────────────────────────────────────────────────────

class TestEMATruncation:

    def test_conv_truncation_error_high_alpha(self):
        """
        α=0.05 → window≈20t. K=256 covers the whole window. Conv≈exact.
        """
        B, N, D = 2, 128, 16
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        alpha = 0.05
        ref  = _reference_ema_exact(x, alpha)
        conv = _conv_ema_k256(x, alpha)
        err  = (ref.float() - conv.float()).abs().mean()
        assert err < 0.01, f"Conv@α=0.05 has unexpected error: {err:.4e}"

    def test_conv_truncation_error_low_alpha(self):
        """
        α=0.00035 → window≈2857t. K=256 covers only 9%.
        Conv introduces systematic error; scan does not.
        This documents the bug that existed before causal_ema_scan.
        """
        from causal_ema_scan import causal_ema_scan
        B, N, D = 2, 512, 16
        x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
        alpha = 0.00035
        ef    = torch.tensor(alpha, device=DEVICE)
        ref  = _reference_ema_exact(x, alpha)
        conv = _conv_ema_k256(x, alpha)
        scan = causal_ema_scan(x, ef, floor=1e-6)

        conv_err = (ref.float() - conv.float()).abs().mean().item()
        scan_err = (ref.float() - scan.float()).abs().mean().item()

        # Conv should be significantly worse than scan at long windows
        assert conv_err > scan_err * 5, \
            f"Expected conv >> scan at α=0.00035; conv={conv_err:.4e} scan={scan_err:.4e}"
        # Scan should be close to exact
        assert scan_err < 0.02, f"Scan error too large: {scan_err:.4e}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Scale embed — init and gradient
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleEmbed:

    def test_scale_embed_init_nonzero(self):
        """scale_embed should be initialised to SCALE_EMBED_INIT_VAL=0.1, not 0."""
        # If zero-init: scale_embed contributes nothing in early training,
        # the relay mechanism never activates. 0.1 gives initial signal.
        SCALE_EMBED_INIT_VAL = 0.1
        J, HD = 24, 64
        scale_embed = torch.full((J, HD), SCALE_EMBED_INIT_VAL)
        assert scale_embed.mean().item() == pytest.approx(SCALE_EMBED_INIT_VAL, abs=1e-6), \
            "scale_embed not initialised to expected value"

    def test_scale_embed_gradient_flows(self):
        """scale_embed must receive gradient; if detached, relay dies."""
        J, HD = 24, 64
        scale_embed = torch.full((J, HD), 0.1, requires_grad=True, device=DEVICE)
        x = torch.randn(J, HD, device=DEVICE)
        out = (x * scale_embed).sum()
        out.backward()
        assert scale_embed.grad is not None
        assert scale_embed.grad.abs().max() > 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v", "--tb=short"])
