"""
V15-HISA DSR smoke tests — backward gradient flow.

This guards against the forward-only failure mode where the Triton launch writes
to a fresh output tensor without an autograd wrapper.
"""

import os
import pathlib
import sys

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn.functional as F


def _hisa_backward_flow():
    from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

    torch.manual_seed(42)
    B, N, D, H, hd = 2, 256, 64, 4, 16
    C, k, m = 8, 4, 16
    device = 'cuda'

    model = HierarchicalSparseAttentionV15HISA(
        D=D, H=H, hd=hd,
        num_chunks=C, top_k_chunks=k, hisa_top_m_tokens=m,
    ).to(device)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
        loss = out.float().square().mean()
    loss.backward()

    required = ['W_q.weight', 'W_k.weight', 'W_v.weight', 'W_o.weight']
    params = dict(model.named_parameters())
    for name in required:
        grad = params[name].grad
        grad_norm = 0.0 if grad is None else grad.float().norm().item()
        print(f'  {name}: grad_norm={grad_norm:.4e}')
        assert grad is not None, f'{name} did not receive gradients'
        assert torch.isfinite(grad).all(), f'{name} gradient contains non-finite values'
        assert grad_norm > 0.0, f'{name} gradient is zero'

    assert x.grad is not None, 'Input did not receive gradients'
    x_grad_norm = x.grad.float().norm().item()
    print(f'  input x: grad_norm={x_grad_norm:.4e}')
    assert x_grad_norm > 0.0, 'Input gradient is zero'
    return True


def _hisa_parameter_update():
    from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

    torch.manual_seed(123)
    B, N, D, H, hd = 2, 256, 64, 4, 16
    C, k, m = 8, 4, 16
    device = 'cuda'

    model = HierarchicalSparseAttentionV15HISA(
        D=D, H=H, hd=hd,
        num_chunks=C, top_k_chunks=k, hisa_top_m_tokens=m,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    target = torch.randn(B, N, D, device=device)

    before = model.W_q.weight.detach().clone()
    entropy_before = None
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
        entropy_before = model._routing_entropy
        loss = F.mse_loss(out.float(), target)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    delta = (model.W_q.weight.detach() - before).float().abs().max().item()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(x)
    entropy_after = model._routing_entropy

    print(f'  W_q update max={delta:.4e}')
    print(f'  routing_entropy before={entropy_before:.6f} after={entropy_after:.6f}')
    assert delta > 0.0, 'W_q did not update after backward/step'
    return True


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA required'
    torch.cuda.set_device(0)

    tests = [
        ('V15-HISA backward gradient flow', _hisa_backward_flow),
        ('V15-HISA parameter update', _hisa_parameter_update),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f'\n{"="*60}')
        print(f'  {name}')
        print(f'{"="*60}')
        try:
            fn()
            print('  PASSED')
            passed += 1
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print(f'\n{"="*60}')
    print(f'  Results: {passed} passed, {failed} failed')
    print(f'{"="*60}')
    sys.exit(1 if failed > 0 else 0)
