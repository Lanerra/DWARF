"""
bench_v5_cuda_vs_triton.py
Benchmark: dsqg_attention_v5 CUDA kernel vs Triton kernel.

The condU_v5 training script (train_2048_condU_v5.py) hot-swaps the Triton
forward with a compiled CUDA forward+backward when dsqg_cuda.so is present.
This script measures throughput of both on the current GPU.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/bench_v5_cuda_vs_triton.py

Outputs:
    benchmarks/logs/bench_v5_cuda_vs_triton.json
"""
import sys, os, time, json
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'kernels'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'kernels', 'dsqg_cuda'))

RESULT_FILE = os.path.join(REPO_ROOT, 'benchmarks', 'logs', 'bench_v5_cuda_vs_triton.json')
DEVICE = 'cuda'

# ── Model dims matching condU_v5 35M ─────────────────────────────────────────
B, N, H, HD = 2, 2048, 8, 64   # batch, seq, heads, head_dim
D = H * HD                      # 512

# Offset set — same as condU_v5 (44 offsets)
_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
OFFSETS = sorted(set(range(0, _DENSE_LOCAL_W + 1)) | set(_DYADIC_LONG_RANGE))
J = len(OFFSETS)
assert J == 44, f"Expected 44 offsets, got {J}"
J_SPARSE = len(_DYADIC_LONG_RANGE)  # 11 — size of phase_base/phase_gain dim

WARMUP_ITERS = 10
BENCH_ITERS  = 50

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_inputs(device, dtype=torch.bfloat16):
    q           = torch.randn(B, H, N, HD,     device=device, dtype=dtype, requires_grad=True)
    k           = torch.randn(B, H, N, HD,     device=device, dtype=dtype, requires_grad=True)
    v           = torch.randn(B, H, N, HD,     device=device, dtype=dtype, requires_grad=True)
    pos_bias    = torch.randn(J, H,            device=device,              requires_grad=True)
    scale_embed = torch.randn(J, HD,           device=device,              requires_grad=True)
    # phase_base/phase_gain index the J_SPARSE=11 long-range offsets only
    phase_base  = torch.randn(J_SPARSE, H, 2,  device=device,              requires_grad=True)
    phase_gain  = torch.zeros(J_SPARSE, H, 2,  device=device,              requires_grad=True)
    # y_pre/z_pre: running rotation state [B, H, N, R_PLANES=2]
    y_pre       = torch.zeros(B, H, N, 2,      device=device,              requires_grad=True)
    z_pre       = torch.zeros(B, H, N, 2,      device=device,              requires_grad=True)
    return q, k, v, pos_bias, scale_embed, phase_base, phase_gain, y_pre, z_pre


def bench(fn, label, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run fn(), measure fwd+bwd throughput in tokens/s."""
    # warmup
    for _ in range(warmup):
        out = fn()
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_iter = B * N
    tps = tokens_per_iter * iters / elapsed
    ms_per_iter = elapsed / iters * 1000
    print(f"  {label:<40s}  {tps/1e6:7.2f}M tok/s  ({ms_per_iter:.1f} ms/iter)")
    return tps, ms_per_iter


# ── Load both kernel variants ─────────────────────────────────────────────────

print("=" * 70)
print("  V5 CUDA extension vs Triton — condU_v5 kernel benchmark")
print(f"  GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
print(f"  B={B}, N={N}, H={H}, HD={HD}, J={J}")
print("=" * 70)

# Import Triton version
import dsqg_attention_v5 as _v5_module
from dsqg_attention_v5 import DSQGAttentionV5

# Try CUDA version
_cuda_fn = None
try:
    import dsqg_cuda as _dsqg_cuda_ext
    from dsqg_attention_v5_cuda import dsqg_attention_v5_cuda as _cuda_attn_fn
    _cuda_fn = _cuda_attn_fn
    print("[kernel] CUDA extension loaded — will benchmark both")
except ImportError as e:
    print(f"[kernel] CUDA extension not found ({e}) — Triton only")

print()

results = {}

# ── Triton benchmark ──────────────────────────────────────────────────────────
print("Triton kernel:")
_triton_fn = _v5_module.dsqg_attention_v5  # save original Triton function

def run_triton():
    q, k, v, pb, se, phb, phg, y, z = make_inputs(DEVICE)
    return _triton_fn(q, k, v, pb, se, phb, phg, y, z)

tps_triton, ms_triton = bench(run_triton, "Triton V5")
results['triton'] = {'tps': tps_triton, 'ms_per_iter': ms_triton}

# ── CUDA benchmark ────────────────────────────────────────────────────────────
if _cuda_fn is not None:
    print("\nCUDA extension kernel:")

    def run_cuda():
        q, k, v, pb, se, phb, phg, y, z = make_inputs(DEVICE)
        return _cuda_fn(q, k, v, pb, se, phb, phg, y, z)

    tps_cuda, ms_cuda = bench(run_cuda, "CUDA V5 extension")
    results['cuda'] = {'tps': tps_cuda, 'ms_per_iter': ms_cuda}

    ratio = tps_cuda / tps_triton
    winner = "CUDA" if ratio > 1 else "Triton"
    print(f"\n  Ratio CUDA/Triton: {ratio:.3f}×  →  {winner} wins")
    results['ratio_cuda_over_triton'] = ratio
    results['winner'] = winner
else:
    print("\nCUDA extension unavailable — only Triton result recorded.")

# ── Save ──────────────────────────────────────────────────────────────────────
results['gpu'] = torch.cuda.get_device_name(0)
results['config'] = {'B': B, 'N': N, 'H': H, 'HD': HD, 'J': J}
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {RESULT_FILE}")
