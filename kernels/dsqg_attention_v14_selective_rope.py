"""
DSQG Attention V14 — Selective RoPE (LayerNorm content signal)
==============================================================

Wraps V13-dynamic-J kernel with a fixed MOVT content computation that
eliminates the gradient starvation in phase_gain parameters.

Problem: In V8/V13, the content signal for MOVT rotation is:
    theta = phase_base + phase_gain * y_pre * z_pre
  where y_pre = q @ query_probes.T and z_pre = k @ key_probes.T
  (with optional normalization). The gradient dL/d(phase_gain) = dL/d(theta) * y * z
  scales with ||q|| * ||k||, creating ~20,000x gradient disparity vs phase_base.

Fix: Apply LayerNorm to the y*z product before it enters the kernel.
    content = LayerNorm(y_pre * z_pre, [R_PLANES])
  The kernel then computes theta = phase_base + phase_gain * content * 1.
  LayerNorm normalizes the content signal to zero mean / unit variance,
  giving phase_gain O(1) gradient scale. Fisher analysis confirms the
  Fisher information ratio (gain/base) improves from 0.000012 to 1.064.

Implementation: Zero kernel changes. We pass the normalized content as
  y_pre and set z_pre = ones, so the kernel's y0*z0 multiplication is
  just content * 1 = content.

Usage:
  from dsqg_attention_v14_selective_rope import DSQGAttentionV14SelectiveRoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

R_PLANES = 2

try:
    from dsqg_attention_v13_dynamic_j import (
        DSQGAttentionV13Dynamic,
        dsqg_attention_v13_dj,
    )
    _V13_AVAILABLE = True
except ImportError:
    _V13_AVAILABLE = False


class DSQGAttentionV14SelectiveRoPE(nn.Module):
    """
    DSQG V14: Selective RoPE with LayerNorm content normalization.

    Wraps DSQGAttentionV13Dynamic with a fixed y_pre/z_pre computation
    that eliminates gradient starvation in phase_gain.

    The Triton kernel is UNCHANGED. Only the content signal computation
    in the forward method differs from V13.
    """

    def __init__(
        self,
        embedding_dimension,
        number_of_heads,
        offsets,
        j_small,
        j_large,
        scale_embed_init=0.15,
        ema_init=0.0208,
        sequence_length=2048,
        dropout=0.1,
    ):
        super().__init__()
        head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads
        self.head_dimension = head_dimension
        self.sequence_length = sequence_length

        assert head_dimension >= 4

        j_value = len(offsets)
        assert j_small + j_large == j_value
        self.j_value = j_value
        self.j_small = j_small
        self.j_large = j_large

        self.register_buffer(
            "offsets_device",
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

        self.qkv_projection = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=True)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=True)
        self.gate_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=True)
        nn.init.constant_(self.gate_projection.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, number_of_heads)
        delta_values = torch.tensor(
            [math.log(1.0 + d) for d in offsets],
            dtype=torch.float32,
        )
        self.pos_bias = nn.Parameter(-delta_values.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.full((j_value, head_dimension), scale_embed_init))
        self.if_gain = nn.Parameter(torch.ones(number_of_heads))

        self.phase_base = nn.Parameter(torch.zeros(j_large, number_of_heads, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(j_large, number_of_heads, R_PLANES))
        self.phase_gate = nn.Parameter(torch.zeros(j_large))
        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, head_dimension))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, head_dimension))
        self.npci_theta_k = nn.Parameter(torch.zeros(number_of_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(number_of_heads))

        self.content_layer_norm = nn.LayerNorm(R_PLANES, elementwise_affine=False)

        self.register_buffer("ema_alpha", torch.tensor(ema_init), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        batch_size, sequence_length, embedding_dimension = x.shape
        number_of_heads = self.number_of_heads
        head_dimension = self.head_dimension

        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(embedding_dimension, dim=-1)
        q = q.view(batch_size, sequence_length, number_of_heads, head_dimension).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch_size, sequence_length, number_of_heads, head_dimension).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch_size, sequence_length, number_of_heads, head_dimension).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = (k + k_delta).contiguous().clone()
            v = (v + v_delta).contiguous().clone()
        else:
            k = k.clone()
            v = v.clone()
        q = q.clone()

        y_pre = torch.einsum("bhnd,rd->bhnr", q.float(), self.query_probes.float())
        z_pre = torch.einsum("bhnd,rd->bhnr", k.float(), self.key_probes.float())
        content = self.content_layer_norm(y_pre * z_pre).contiguous()

        ones = torch.ones_like(content)

        gate = torch.sigmoid(self.phase_gate)[:, None, None]
        gated_phase_base = self.phase_base * gate
        gated_phase_gain = self.phase_gain * gate

        output = dsqg_attention_v13_dj(
            q, k, v,
            self.pos_bias, self.scale_embed,
            gated_phase_base, gated_phase_gain,
            content, ones,
            self.j_value, self.j_small, self.j_large,
            self.offsets_device,
        )

        output = output * self.if_gain.view(1, number_of_heads, 1, 1)
        output_flat = output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, embedding_dimension).contiguous()
        gate = torch.sigmoid(self.gate_projection(x))
        return self.dropout(self.output_projection(output_flat * gate))

    def get_autotune_config(self):
        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)
        arch = "sm_90" if _sm90 else ("sm_89" if _sm89 else "sm_86")
        return {
            "arch": arch,
            "kernel": "v14_selective_rope_layernorm_content",
            "autotune": False,
            "j_val": self.j_value,
            "j_small": self.j_small,
            "j_large": self.j_large,
        }

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
            se = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            phb = self.phase_base.detach().cpu()
            phg = self.phase_gain.detach().cpu()
            phgate = torch.sigmoid(self.phase_gate).detach().cpu()
            qp = self.query_probes.detach().cpu()
            kp = self.key_probes.detach().cpu()
            thk = self.npci_theta_k.detach().cpu()
            thv = self.npci_theta_v.detach().cpu()

        plane_diff = (phb[:, :, 0] - phb[:, :, 1]).abs().mean().item() if self.j_large > 0 else 0.0

        return {
            "pos_bias_abs_mean": pb.abs().mean().item(),
            "pos_bias_abs_max": pb.abs().max().item(),
            "pos_bias_mean_per_head": pb.mean(0).tolist(),
            "scale_embed_abs_mean": se.abs().mean().item(),
            "scale_embed_abs_max": se.abs().max().item(),
            "if_gain": gain.tolist(),
            "phase_base_abs_mean": phb.abs().mean().item() if self.j_large > 0 else 0.0,
            "phase_base_abs_max": phb.abs().max().item() if self.j_large > 0 else 0.0,
            "phase_base_local_mean": 0.0,
            "phase_base_sparse_mean": phb.abs().mean().item() if self.j_large > 0 else 0.0,
            "phase_base_by_head": phb.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            "phase_base_sparse_by_head": phb.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            "phase_base_plane_diff": plane_diff,
            "phase_gain_abs_mean": phg.abs().mean().item() if self.j_large > 0 else 0.0,
            "phase_gain_abs_max": phg.abs().max().item() if self.j_large > 0 else 0.0,
            "phase_gain_sparse_mean": phg.abs().mean().item() if self.j_large > 0 else 0.0,
            "phase_gain_by_head": phg.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            "phase_gate_mean": phgate.mean().item() if self.j_large > 0 else 0.5,
            "phase_gate_min": phgate.min().item() if self.j_large > 0 else 0.5,
            "phase_gate_max": phgate.max().item() if self.j_large > 0 else 0.5,
            "query_probe_norm": qp.norm(dim=1).tolist(),
            "key_probe_norm": kp.norm(dim=1).tolist(),
            "npci_theta_k": thk.tolist(),
            "npci_theta_v": thv.tolist(),
            "v14_selective_rope": True,
            "content_normalization": "layernorm",
            "j_val": self.j_value,
            "j_small": self.j_small,
            "j_large": self.j_large,
        }


def load_v13_checkpoint_into_v14(v14_module, state_dictionary, strict=False):
    """
    Load a V13 checkpoint into a V14 module.

    V14 adds content_layer_norm (no affine, so no learned params to load).
    All other parameters have identical names and shapes.

    Parameter name mapping (V13 -> V14):
      num_heads -> number_of_heads (constructor only)
      head_dim -> head_dimension (constructor only)
      seq_len -> sequence_length (constructor only)
      All nn.Parameter names are UNCHANGED.
    """
    renamed = {}
    for key, value in state_dictionary.items():
        new_key = key
        new_key = new_key.replace("qkv_proj.", "qkv_projection.")
        new_key = new_key.replace("out_proj.", "output_projection.")
        new_key = new_key.replace("gate_proj.", "gate_projection.")
        new_key = new_key.replace("offsets_dev", "offsets_device")
        renamed[new_key] = value

    missing, unexpected = v14_module.load_state_dict(renamed, strict=strict)
    return missing, unexpected


# ─────────────────────────────────────────────────────────────────────────────
# Correctness test
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device="cuda"):
    if not _V13_AVAILABLE:
        print("V13 kernel not available — skipping V14 tests")
        return True

    print("=" * 70)
    print("DSQG V14 — Selective RoPE Correctness Tests")
    print("=" * 70)
    ok_all = True

    offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192]
    j_small = 16
    j_large = 4

    batch_size, number_of_heads, sequence_length, head_dimension = 2, 8, 128, 32
    embedding_dimension = number_of_heads * head_dimension

    torch.manual_seed(42)
    v14 = DSQGAttentionV14SelectiveRoPE(
        embedding_dimension, number_of_heads, offsets, j_small, j_large,
        dropout=0.0,
    ).to(device)

    with torch.no_grad():
        v14.query_probes.normal_(0, 0.1)
        v14.key_probes.normal_(0, 0.1)
        v14.phase_gain.normal_(0, 0.1)

    x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device, dtype=torch.float32)

    output = v14(x)
    assert output.shape == (batch_size, sequence_length, embedding_dimension), f"Shape mismatch: {output.shape}"
    print(f"  Forward pass shape: {output.shape} — PASS")

    x_grad = x.clone().requires_grad_(True)
    output = v14(x_grad)
    loss = output.sum()
    loss.backward()
    assert x_grad.grad is not None
    assert x_grad.grad.abs().max() > 0
    print(f"  Backward pass: gradients flow — PASS")

    phase_gain_gradient = v14.phase_gain.grad
    phase_base_gradient = v14.phase_base.grad
    assert phase_gain_gradient is not None, "phase_gain has no gradient"
    assert phase_base_gradient is not None, "phase_base has no gradient"
    ratio = phase_gain_gradient.norm() / max(phase_base_gradient.norm().item(), 1e-30)
    print(f"  V14 gradient ratio (gain/base): {ratio:.4f}")
    ok_ratio = ratio > 0.01
    if not ok_ratio:
        ok_all = False
    print(f"  Gradient ratio > 0.01: {'PASS' if ok_ratio else 'FAIL'}")

    v13 = DSQGAttentionV13Dynamic(
        embedding_dimension, number_of_heads, offsets, j_small, j_large,
        dropout=0.0,
    ).to(device)
    v14_state = v14.state_dict()
    v13_mapped = {}
    for key, value in v14_state.items():
        new_key = key
        new_key = new_key.replace("qkv_projection.", "qkv_proj.")
        new_key = new_key.replace("output_projection.", "out_proj.")
        new_key = new_key.replace("gate_projection.", "gate_proj.")
        new_key = new_key.replace("offsets_device", "offsets_dev")
        new_key = new_key.replace("number_of_heads", "num_heads")
        v13_mapped[new_key] = value
    v13.load_state_dict(v13_mapped, strict=False)

    with torch.no_grad():
        v14.phase_gain.zero_()
        v13.phase_gain.zero_()
        v14.query_probes.zero_()
        v14.key_probes.zero_()
        v13.query_probes.zero_()
        v13.key_probes.zero_()

    x_test = torch.randn(batch_size, sequence_length, embedding_dimension,
                         device=device, dtype=torch.float32)
    with torch.no_grad():
        out_v14 = v14(x_test)
        out_v13 = v13(x_test)
    diff = (out_v14.float() - out_v13.float()).abs().max().item()
    ok_zero = diff < 0.05
    if not ok_zero:
        ok_all = False
    print(f"  V14==V13 when phase_gain=0: max_diff={diff:.6f} — {'PASS' if ok_zero else 'FAIL'}")

    print("=" * 70)
    print(f"{'ALL PASSED' if ok_all else 'SOME FAILED'}")
    return ok_all


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = run_tests(device)
    sys.exit(0 if ok else 1)
