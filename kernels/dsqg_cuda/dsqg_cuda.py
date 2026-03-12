"""
DSQG Attention V3 — CUDA sliding-window kernel (d50 Pure Geometry, J=44)

Python loader + autograd wrapper.
Compiles the CUDA kernel on first import via torch.utils.cpp_extension.load.

Usage:
  from kernels.dsqg_cuda import dsqg_attention_v3_cuda
  out = dsqg_attention_v3_cuda(q, k, v, pos_bias, scale_embed)
"""

import os
import time
import torch
from torch.utils.cpp_extension import load

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_FILE = os.path.join(_KERNEL_DIR, "dsqg_cuda_kernel.cu")

_module = load(
    name="dsqg_cuda_ext",
    sources=[_CU_FILE],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-lineinfo",
    ],
    verbose=False,
)


class _DSQGCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape == (44, q.size(1))
        assert scale_embed.shape == (44, q.size(3))

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, lse = _module.forward(q, k, v, pos_bias, scale_embed)
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pos_bias, scale_embed, out, lse = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dk, dv, dpb, dse = _module.backward(
            q, k, v, pos_bias, scale_embed, out, lse, dout
        )
        return dq, dk, dv, dpb, dse


def dsqg_attention_v3_cuda(q, k, v, pos_bias, scale_embed):
    """
    q, k, v:      [B, H, N, HD]  bfloat16
    pos_bias:     [44, H]         float32
    scale_embed:  [44, HD]        float32
    Returns:      [B, H, N, HD]  same dtype as input
    """
    original_dtype = q.dtype
    if original_dtype != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGCudaFn.apply(q, k, v, pos_bias.float(), scale_embed.float())
    return out if original_dtype == torch.bfloat16 else out.to(original_dtype)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(_KERNEL_DIR, ".."))
    sys.path.insert(0, os.path.join(_KERNEL_DIR, "../.."))
    from dsqg_d50_pure import dsqg_attention_v3 as dsqg_triton, _reference_v3

    device = "cuda"

    print("=" * 70)
    print(" DSQG V3 CUDA — Forward Agreement with Triton + Reference")
    print("=" * 70)

    configurations = [
        (1, 8,   64, 32, "tiny HD=32"),
        (2, 8,  512, 32, "mid HD=32"),
        (4, 8, 2047, 32, "14M shape HD=32"),
        (1, 8,   64, 48, "tiny HD=48"),
        (2, 8,  512, 48, "mid HD=48"),
    ]
    all_passed = True
    for B, H, N, HD, label in configurations:
        torch.manual_seed(42)
        q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05

        ref_out = _reference_v3(q.detach(), k.detach(), v.detach(), pb, se)
        triton_out = dsqg_triton(q.detach().clone(), k.detach().clone(),
                                  v.detach().clone(), pb, se)
        cuda_out = dsqg_attention_v3_cuda(q.detach().clone(), k.detach().clone(),
                                           v.detach().clone(), pb, se)

        fwd_err_vs_ref = (ref_out.float() - cuda_out.float()).abs().max().item()
        fwd_err_vs_triton = (triton_out.float() - cuda_out.float()).abs().max().item()

        passed = max(fwd_err_vs_ref, fwd_err_vs_triton) < 0.05
        if not passed:
            all_passed = False
        print(f"  {label:24s}  vs_ref={fwd_err_vs_ref:.4f}  "
              f"vs_triton={fwd_err_vs_triton:.4f}  "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    print("=" * 70)
    print(" DSQG V3 CUDA — Backward Agreement with Triton")
    print("=" * 70)

    backward_configurations = [
        (1, 8,   64, 32, "tiny HD=32"),
        (2, 8,  512, 32, "mid HD=32"),
        (4, 8, 2047, 32, "14M shape HD=32"),
    ]
    for B, H, N, HD, label in backward_configurations:
        torch.manual_seed(42)
        q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05

        # Triton backward
        qt, kt2, vt = [t.clone().detach().requires_grad_(True) for t in (q, k, v)]
        se_t = se.clone().detach().requires_grad_(True)
        dsqg_triton(qt, kt2, vt, pb, se_t).sum().backward()
        dq_triton = qt.grad.clone()
        dk_triton = kt2.grad.clone()
        dv_triton = vt.grad.clone()
        dse_triton = se_t.grad.clone()

        # CUDA backward
        qc, kc, vc = [t.clone().detach().requires_grad_(True) for t in (q, k, v)]
        se_c = se.clone().detach().requires_grad_(True)
        dsqg_attention_v3_cuda(qc, kc, vc, pb, se_c).sum().backward()
        dq_cuda = qc.grad.clone()
        dk_cuda = kc.grad.clone()
        dv_cuda = vc.grad.clone()
        dse_cuda = se_c.grad.clone()

        err_dq = (dq_triton.float() - dq_cuda.float()).abs().max().item()
        err_dk = (dk_triton.float() - dk_cuda.float()).abs().max().item()
        err_dv = (dv_triton.float() - dv_cuda.float()).abs().max().item()
        err_dse = (dse_triton.float() - dse_cuda.float()).abs().max().item()

        err_max = max(err_dq, err_dk, err_dv, err_dse)
        passed = err_max < 0.1
        if not passed:
            all_passed = False
        print(f"  {label:24s}  dq={err_dq:.4f}  dk={err_dk:.4f}  "
              f"dv={err_dv:.4f}  dse={err_dse:.4f}  "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    print("=" * 70)
    print(" DSQG V3 — Performance Comparison (CUDA vs Triton)")
    print("=" * 70)

    perf_configurations = [
        (8, 8, 2048, 32, "B=8 H=8 N=2048 HD=32"),
        (8, 8, 2048, 48, "B=8 H=8 N=2048 HD=48"),
    ]
    for B, H, N, HD, label in perf_configurations:
        torch.manual_seed(42)
        q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb = torch.randn(44, H,  device=device, dtype=torch.float32) * 0.5
        se = torch.randn(44, HD, device=device, dtype=torch.float32) * 0.05

        total_tokens = B * N
        warmup = 10
        repeats = 50

        # Triton
        for _ in range(warmup):
            dsqg_triton(q, k, v, pb, se)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            dsqg_triton(q, k, v, pb, se)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - t0) / repeats
        triton_tps = total_tokens / triton_time

        # CUDA
        for _ in range(warmup):
            dsqg_attention_v3_cuda(q, k, v, pb, se)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            dsqg_attention_v3_cuda(q, k, v, pb, se)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - t0) / repeats
        cuda_tps = total_tokens / cuda_time

        print(f"  {label}")
        print(f"    Triton: {triton_tps/1e3:.1f}k tok/s  ({triton_time*1e3:.2f} ms)")
        print(f"    CUDA:   {cuda_tps/1e3:.1f}k tok/s  ({cuda_time*1e3:.2f} ms)")
        print(f"    Speedup: {cuda_tps/triton_tps:.2f}x")
        print()

    print("=" * 70)
    print(f"{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)
