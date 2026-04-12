"""
V8-dynamic-J batch-size sweep benchmark
========================================
Measures tok/s and peak VRAM for V8-dynamic-J at BS=16, 32, 64
with D=1536, H=12, HD=128, N=2048 (D=1536 training config).

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/bench_v8_bs_sweep.py
"""

import sys, time, torch

sys.path.insert(0, "kernels")
from dsqg_attention_v8_dynamic_j import dsqg_attention_v8, ALL_OFFSETS, J, J_LARGE, R_PLANES

DEVICE = "cuda"
H, N, HD = 12, 2048, 128
BATCH_SIZES = [16, 32, 64]
WARMUP = 5
REPS   = 20


def make_inputs(B, device, seed=42):
    torch.manual_seed(seed)
    kw  = dict(device=device, dtype=torch.bfloat16)
    kw2 = dict(device=device, dtype=torch.float32)
    q          = torch.randn(B, H, N, HD, **kw) * 0.1
    k          = torch.randn(B, H, N, HD, **kw) * 0.1
    v          = torch.randn(B, H, N, HD, **kw) * 0.1
    pos_bias   = torch.zeros(J, H, **kw2)
    scale_embed= torch.ones(J, HD, device=device, dtype=torch.float32) * 0.1
    phase_base = torch.zeros(J_LARGE, H, R_PLANES, **kw2)
    phase_gain = torch.zeros(J_LARGE, H, R_PLANES, **kw2)
    y_pre      = torch.zeros(B, H, N, R_PLANES, **kw)
    z_pre      = torch.zeros(B, H, N, R_PLANES, **kw)
    return q, k, v, pos_bias, scale_embed, phase_base, phase_gain, y_pre, z_pre


def bench(B, fwd_only=False):
    inputs = make_inputs(B, DEVICE)
    if not fwd_only:
        inputs = [x.requires_grad_(x.is_floating_point()) for x in inputs]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # warmup
    for _ in range(WARMUP):
        out = dsqg_attention_v8(*inputs)
        if not fwd_only:
            out.sum().backward()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = dsqg_attention_v8(*inputs)
        if not fwd_only:
            out.sum().backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    ms_per_call = (t1 - t0) / REPS * 1000
    tokens_per_sec = B * N / (ms_per_call / 1000)
    return ms_per_call, tokens_per_sec, peak_mb


print(f"\nV8-dynamic-J batch-size sweep  (H={H}, N={N}, HD={HD})")
print("=" * 66)

for B in BATCH_SIZES:
    print(f"\n  BS={B}:")
    try:
        ms_f, tps_f, mem_f = bench(B, fwd_only=True)
        print(f"    Forward only:      {ms_f:8.2f} ms/call  "
              f"{tps_f/1e6:5.2f} Mtok/s  peak={mem_f:.0f} MB")
    except torch.cuda.OutOfMemoryError:
        print(f"    Forward only:      OOM")
        print(f"    Fwd+Bwd:           OOM (skipped)")
        continue
    try:
        ms_b, tps_b, mem_b = bench(B, fwd_only=False)
        print(f"    Forward + Backward:{ms_b:8.2f} ms/call  "
              f"{tps_b/1e6:5.2f} Mtok/s  peak={mem_b:.0f} MB")
    except torch.cuda.OutOfMemoryError:
        print(f"    Forward + Backward: OOM")

print("\nDone.")
