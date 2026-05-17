"""
HISA recompute equivalence tests.

Proves that backward replay variants produce mathematically identical
results to the baseline (non-replay) implementation.

Before replay is implemented: baseline-vs-baseline control test.
After replay: baseline-vs-replay equivalence test.

Run: pytest -q tests/test_hisa_recompute_equivalence.py -k hisa
"""

import os
import pathlib
import sys

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clone_weights(src_model, dst_model):
    """Copy weights from src to dst model."""
    for (n1, p1), (n2, p2) in zip(src_model.named_parameters(), dst_model.named_parameters()):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        p2.data.copy_(p1.data)


def _compare_tensors(a, b, name, atol=1e-2, rtol=1e-3):
    """Compare two tensors in fp32, return max abs error."""
    a_fp32 = a.float().detach()
    b_fp32 = b.float().detach()
    diff = (a_fp32 - b_fp32).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    # Also check relative error where values are large enough
    denom = a_fp32.abs().clamp(min=1e-6)
    rel_err = (diff / denom).max().item()

    status = "OK" if max_err <= atol else "FAIL"
    print(f"  {name}: max_abs={max_err:.2e}  mean_abs={mean_err:.2e}  max_rel={rel_err:.2e}  [{status}]")
    return max_err, rel_err


class TestHISARecomputeEquivalence:
    """
    Exactness harness for HISA kernel replay variants.

    Tests baseline-vs-baseline equivalence (control) and will later test
    baseline-vs-replay equivalence once replay is implemented.
    """

    def setup_method(self):
        from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

        torch.manual_seed(42)
        self.B, self.N, self.D, self.H, self.hd = 2, 256, 64, 4, 16
        self.C, self.k, self.m = 8, 4, 16
        self.device = 'cuda'

        self.model_a = HierarchicalSparseAttentionV15HISA(
            D=self.D, H=self.H, hd=self.hd,
            num_chunks=self.C, top_k_chunks=self.k, hisa_top_m_tokens=self.m,
        ).to(self.device)

        self.model_b = HierarchicalSparseAttentionV15HISA(
            D=self.D, H=self.H, hd=self.hd,
            num_chunks=self.C, top_k_chunks=self.k, hisa_top_m_tokens=self.m,
        ).to(self.device)

        _clone_weights(self.model_a, self.model_b)

    def _run_forward_backward(self, model, x, mode='baseline'):
        """
        Run forward + backward, returning (output, grad_dict).

        mode: 'baseline' (current) or 'replay' (future).
        """
        x = x.clone().detach().requires_grad_(True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            loss = out.float().square().mean()
        loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().detach()
        grads['x'] = x.grad.clone().detach() if x.grad is not None else None

        return out.detach(), grads

    def test_baseline_vs_baseline_control(self):
        """
        Control test: two identical models with identical inputs
        should produce identical outputs and gradients.

        This must pass before any kernel edits.
        """
        torch.manual_seed(123)
        x = torch.randn(self.B, self.N, self.D,
                        device=self.device, dtype=torch.bfloat16)

        out_a, grads_a = self._run_forward_backward(self.model_a, x, mode='baseline')
        out_b, grads_b = self._run_forward_backward(self.model_b, x, mode='baseline')

        # Compare outputs
        max_err, rel_err = _compare_tensors(out_a, out_b, 'output', atol=1e-2)
        assert max_err <= 1e-2, f"Output mismatch: max_abs={max_err:.2e}"

        # Compare gradients
        for name in ['W_q.weight', 'W_k.weight', 'W_v.weight', 'W_o.weight', 'x']:
            ga = grads_a.get(name)
            gb = grads_b.get(name)
            assert ga is not None, f"{name} grad is None (model A)"
            assert gb is not None, f"{name} grad is None (model B)"
            max_err, rel_err = _compare_tensors(ga, gb, f'grad {name}', atol=1e-2)
            assert max_err <= 1e-2, f"Grad {name} mismatch: max_abs={max_err:.2e}"

        # Check no non-finite gradients
        for name, grad in grads_a.items():
            if grad is not None:
                assert torch.isfinite(grad.float()).all(), \
                    f"{name} gradient contains non-finite values"

    def test_baseline_vs_baseline_non_divisible_length(self):
        """
        Test with sequence length not evenly divisible by num_chunks.

        This exercises pad/unpad paths which must be replayed exactly.
        """
        torch.manual_seed(456)
        N = 250  # not divisible by 8 chunks
        x = torch.randn(self.B, N, self.D,
                        device=self.device, dtype=torch.bfloat16)

        out_a, grads_a = self._run_forward_backward(self.model_a, x, mode='baseline')
        out_b, grads_b = self._run_forward_backward(self.model_b, x, mode='baseline')

        max_err, rel_err = _compare_tensors(out_a, out_b, 'output (N=250)', atol=1e-2)
        assert max_err <= 1e-2, f"Output mismatch (N=250): max_abs={max_err:.2e}"

        for name in ['W_q.weight', 'W_k.weight', 'W_v.weight', 'W_o.weight', 'x']:
            ga = grads_a.get(name)
            gb = grads_b.get(name)
            assert ga is not None, f"{name} grad is None (model A, N=250)"
            assert gb is not None, f"{name} grad is None (model B, N=250)"
            max_err, rel_err = _compare_tensors(ga, gb, f'grad {name} (N=250)', atol=1e-2)
            assert max_err <= 1e-2, f"Grad {name} mismatch (N=250): max_abs={max_err:.2e}"

    def test_deterministic_near_tie_routing(self):
        """
        Test with inputs designed to create near-tie routing scores.

        This verifies that top-k selection is deterministic across runs
        with identical inputs and weights.
        """
        from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

        torch.manual_seed(789)

        # Create a model where routing scores will be near-tie
        model_tie = HierarchicalSparseAttentionV15HISA(
            D=self.D, H=self.H, hd=self.hd,
            num_chunks=self.C, top_k_chunks=self.k, hisa_top_m_tokens=self.m,
        ).to(self.device)

        # Make W_k produce similar keys (near-tie scenario)
        with torch.no_grad():
            model_tie.W_k.weight.fill_(0.01)

        x = torch.randn(self.B, self.N, self.D,
                        device=self.device, dtype=torch.bfloat16)

        out_tie, grads_tie = self._run_forward_backward(model_tie, x, mode='baseline')
        out_tie2, grads_tie2 = self._run_forward_backward(model_tie, x, mode='baseline')

        max_err, rel_err = _compare_tensors(out_tie, out_tie2, 'output (near-tie)', atol=1e-2)
        assert max_err <= 1e-2, f"Near-tie output mismatch: max_abs={max_err:.2e}"

    def test_no_nfinite_gradients(self):
        """
        Verify no non-finite gradients in any parameter or input.
        """
        torch.manual_seed(999)
        x = torch.randn(self.B, self.N, self.D,
                        device=self.device, dtype=torch.bfloat16)

        _, grads = self._run_forward_backward(self.model_a, x, mode='baseline')

        for name, grad in grads.items():
            if grad is not None:
                assert torch.isfinite(grad.float()).all(), \
                    f"{name} contains non-finite gradients"
                assert grad.float().norm() > 0, \
                    f"{name} gradient is zero"


class TestHISARecomputeEquivalenceLarge:
    """
    Larger-scale equivalence test with settings closer to production.
    """

    def test_large_baseline_control(self):
        """
        Baseline-vs-baseline with larger dimensions (closer to D768 regime).
        """
        from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

        torch.manual_seed(42)
        B, N, D, H, hd = 2, 512, 768, 16, 48
        C, k, m = 32, 4, 32

        model_a = HierarchicalSparseAttentionV15HISA(
            D=D, H=H, hd=hd,
            num_chunks=C, top_k_chunks=k, hisa_top_m_tokens=m,
        ).to('cuda')
        model_b = HierarchicalSparseAttentionV15HISA(
            D=D, H=H, hd=hd,
            num_chunks=C, top_k_chunks=k, hisa_top_m_tokens=m,
        ).to('cuda')
        _clone_weights(model_a, model_b)

        x = torch.randn(B, N, D, device='cuda', dtype=torch.bfloat16)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_a = model_a(x)
            loss_a = out_a.float().square().mean()
        loss_a.backward()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_b = model_b(x)
            loss_b = out_b.float().square().mean()
        loss_b.backward()

        max_err, rel_err = _compare_tensors(out_a, out_b, 'output (D768)', atol=1e-1)
        assert max_err <= 1e-1, f"Large output mismatch: max_abs={max_err:.2e}"

        # Check key gradients
        for name in ['W_q.weight', 'W_k.weight', 'W_v.weight', 'W_o.weight']:
            ga = dict(model_a.named_parameters())[name].grad
            gb = dict(model_b.named_parameters())[name].grad
            assert ga is not None and gb is not None
            max_err, rel_err = _compare_tensors(ga, gb, f'grad {name} (D768)', atol=1e-1)
            assert max_err <= 1e-1, f"Large grad {name} mismatch: max_abs={max_err:.2e}"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
