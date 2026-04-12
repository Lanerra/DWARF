"""
FFT-DSQG vs V8 Benchmark
=========================

Side-by-side comparison of step time, peak VRAM, and MFU estimate.
Tests both score backends (gather, fft) against V8 Triton kernel.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/benchmark_fft_vs_v8.py
"""

import sys
sys.path.insert(0, '.')

import time
import gc
import torch
import torch.nn.functional as F

from kernels.dsqg_attention_v8 import DSQGAttentionV8
from kernels.dsqg_attention_fft import FFTDSQGAttention


def get_gpu_info():
    if not torch.cuda.is_available():
        return "No CUDA"
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f"{name} (sm_{cc[0]}{cc[1]}, {mem_gb:.1f} GB)"


def estimate_flops(B, N, H, HD):
    """
    Estimate FLOPs for one forward+backward step of DSQG attention.

    Forward:
      QKV proj: 2 * B * N * D * 3D
      Score (24 offsets): 24 * B * H * N * HD * 2 (dot product)
      Scale embed gate:  24 * B * H * N * HD * 2
      Softmax: B * H * N * 24 * 5
      Value sum: B * H * N * 24 * HD * 2
      MOVT: ~10 * B * H * N * 20 (sin/cos + rotations)
      Out proj + gate: 2 * B * N * D * D + B * N * D

    Backward ~ 2x forward for attention, 2x for linear layers.
    Total ~ 3x forward.
    """
    D = H * HD
    fwd_attn = (
        24 * B * H * N * HD * 2 * 2
        + B * H * N * 24 * 5
        + B * H * N * 24 * HD * 2
        + 10 * B * H * N * 40
    )
    fwd_linear = 2 * B * N * D * 3 * D + 2 * B * N * D * D + B * N * D
    fwd_total = fwd_attn + fwd_linear
    return fwd_total * 3


def benchmark_model(model, B, N, D, device, warmup=5, repeats=20, label=""):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    torch.cuda.empty_cache()

    for _ in range(warmup):
        x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16) * 0.1
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for i in range(repeats):
        x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16) * 0.1
        optimizer.zero_grad()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'label': label,
        'avg_step_ms': avg_time * 1000,
        'min_step_ms': min_time * 1000,
        'max_step_ms': max_time * 1000,
        'peak_vram_gb': peak_vram,
    }


def run_benchmark(B, N, H, HD, device):
    D = H * HD

    print(f"\n  Config: B={B}, N={N}, H={H}, HD={HD}, D={D}")
    print(f"  {'':4s}{'Model':24s}{'Avg (ms)':>10s}{'Min (ms)':>10s}{'Max (ms)':>10s}"
          f"{'VRAM (GB)':>10s}{'MFU (%)':>10s}")
    print(f"  {'-'*68}")

    flops = estimate_flops(B, N, H, HD)
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" in gpu_name:
        peak_tflops = 165.0
    elif "3090" in gpu_name:
        peak_tflops = 71.0
    elif "H100" in gpu_name or "H200" in gpu_name:
        peak_tflops = 989.0
    else:
        peak_tflops = 100.0

    results = []

    torch.manual_seed(42)
    try:
        v8 = DSQGAttentionV8(D, H, seq_len=N, dropout=0.0).to(device)
        r = benchmark_model(v8, B, N, D, device, label="V8 (Triton)")
        mfu = (flops / (r['avg_step_ms'] / 1000)) / (peak_tflops * 1e12) * 100
        r['mfu'] = mfu
        results.append(r)
        print(f"  {'':4s}{r['label']:24s}{r['avg_step_ms']:10.2f}{r['min_step_ms']:10.2f}"
              f"{r['max_step_ms']:10.2f}{r['peak_vram_gb']:10.2f}{mfu:10.2f}")
        del v8
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {'':4s}{'V8 (Triton)':24s}  ERROR: {e}")

    torch.manual_seed(42)
    try:
        fft_gather = FFTDSQGAttention(D, H, seq_len=N, dropout=0.0, score_backend="gather").to(device)
        r = benchmark_model(fft_gather, B, N, D, device, label="FFT-DSQG (gather)")
        mfu = (flops / (r['avg_step_ms'] / 1000)) / (peak_tflops * 1e12) * 100
        r['mfu'] = mfu
        results.append(r)
        print(f"  {'':4s}{r['label']:24s}{r['avg_step_ms']:10.2f}{r['min_step_ms']:10.2f}"
              f"{r['max_step_ms']:10.2f}{r['peak_vram_gb']:10.2f}{mfu:10.2f}")
        del fft_gather
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {'':4s}{'FFT-DSQG (gather)':24s}  ERROR: {e}")

    torch.manual_seed(42)
    try:
        fft_fft = FFTDSQGAttention(D, H, seq_len=N, dropout=0.0, score_backend="fft").to(device)
        r = benchmark_model(fft_fft, B, N, D, device, label="FFT-DSQG (fft)")
        mfu = (flops / (r['avg_step_ms'] / 1000)) / (peak_tflops * 1e12) * 100
        r['mfu'] = mfu
        results.append(r)
        print(f"  {'':4s}{r['label']:24s}{r['avg_step_ms']:10.2f}{r['min_step_ms']:10.2f}"
              f"{r['max_step_ms']:10.2f}{r['peak_vram_gb']:10.2f}{mfu:10.2f}")
        del fft_fft
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {'':4s}{'FFT-DSQG (fft)':24s}  ERROR: {e}")

    return results


if __name__ == "__main__":
    device = 'cuda'
    assert torch.cuda.is_available(), "CUDA required"

    print("=" * 70)
    print("FFT-DSQG vs V8 Benchmark")
    print("=" * 70)
    print(f"  GPU: {get_gpu_info()}")
    print(f"  PyTorch: {torch.__version__}")

    configs = [
        (2, 2047, 8, 32, "d256 B=2 (13M config)"),
        (4, 2047, 8, 32, "d256 B=4"),
        (8, 2047, 8, 32, "d256 B=8"),
        (2, 512, 12, 64, "d768 B=2 N=512"),
    ]

    all_results = {}
    for B, N, H, HD, name in configs:
        print(f"\n{'='*70}")
        print(f"Benchmark: {name}")
        all_results[name] = run_benchmark(B, N, H, HD, device)

    print(f"\n{'='*70}")
    print("Analysis")
    print("=" * 70)
    print()
    print("Why FFT is slower than V8 for DSQG:")
    print("  1. DSQG computes per-position dot products Q[t]*K[t-d] for J=24 offsets.")
    print("     This is NOT a global cross-correlation — FFT cannot replace the core op.")
    print("  2. V8's Triton kernel fuses score computation + softmax + value aggregation")
    print("     in a single kernel, keeping Q in SRAM registers. The PyTorch version")
    print("     launches ~50 separate kernels (24 for scores, 1 softmax, 24 for values),")
    print("     reading Q from HBM each time.")
    print("  3. The FFT backend adds O(N*log(N)) overhead per offset for the frequency-")
    print("     domain shift, vs O(N) for a direct contiguous slice.")
    print()
    print("When FFT would win:")
    print("  If DSQG used ALL N lags (not just 24), FFT cross-correlation would compute")
    print("  them in O(N*log(N)) vs O(N^2). But with J=24 << N=2048, direct computation")
    print("  at O(J*N) = O(24*N) is already faster than O(N*log(N)) = O(11*N).")
    print("  Threshold: FFT becomes competitive when J > ~log2(N) = 11, but the global")
    print("  cross-correlation still sums over positions, giving wrong semantics for DSQG.")
    print("  There is no N threshold where FFT cross-correlation produces per-position scores.")
