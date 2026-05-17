#!/usr/bin/env python3
"""
HISA checkpoint comparison harness.

Shells out to the trainer with a fixed env matrix, parses [BENCH] lines,
and produces a JSON artifact + compact comparison table.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 scripts/bench_hisa_checkpoint_modes.py
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 scripts/bench_hisa_checkpoint_modes.py --saved-state
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 scripts/bench_hisa_checkpoint_modes.py --mode baseline_none
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Fixed benchmark contract
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(PROJECT_ROOT, "train", "train_d512_l10_hisa_h16_v2_l3.py")
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

BASE_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "DWARF_BENCH_ONLY": "1",
    "DWARF_MAX_ACC_STEPS": "25",
    "DWARF_LOG_INTERVAL": "5",
    "DWARF_BS": "2",
    "DWARF_GA": "1",
    "DWARF_COMPILE": "0",
}

# Mode definitions: (name, DWARF_CKPT, HISA_RECOMPUTE)
MODES = {
    "baseline_none":        {"DWARF_CKPT": "none",         "HISA_RECOMPUTE": "none"},
    "trainer_full_attn":    {"DWARF_CKPT": "full_attn",    "HISA_RECOMPUTE": "none"},
    "trainer_all":          {"DWARF_CKPT": "all",          "HISA_RECOMPUTE": "none"},
    "trainer_every_other":  {"DWARF_CKPT": "every_other",  "HISA_RECOMPUTE": "none"},
    # Placeholder hooks for kernel replay variants (Track B):
    "recompute_out_lse":    {"DWARF_CKPT": "full_attn",    "HISA_RECOMPUTE": "out_lse"},
    "recompute_all":        {"DWARF_CKPT": "full_attn",    "HISA_RECOMPUTE": "all"},
}


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except Exception:
        return "unknown"


def run_trainer(env_overrides: dict) -> tuple[str, str]:
    """Run trainer with given env overrides. Returns (stdout, stderr)."""
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [PYTHON, "-u", TRAINER]
    print(f"  Running: {' '.join(cmd)}", flush=True)
    print(f"  Env overrides: {env_overrides}", flush=True)

    start = time.monotonic()
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600, cwd=PROJECT_ROOT
    )
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        print(f"  WARNING: trainer exited with code {proc.returncode} after {elapsed:.0f}s", flush=True)
        # Still try to parse any [BENCH] lines that appeared
        print(f"  stderr preview: {proc.stderr[:500]}", flush=True)

    return proc.stdout, proc.stderr


def parse_bench_lines(text: str) -> dict:
    """Parse [BENCH] lines from trainer output into structured dict."""
    result = {}

    # [BENCH] first_step_ms=... trailing_avg_ms=... steady_tok_s=... approx_compile_overhead_ms=...
    m = re.search(
        r'\[BENCH\]\s+first_step_ms=([\d.]+)\s+'
        r'trailing_avg_ms=([\d.]+)\s+'
        r'steady_tok_s=([\d.]+)\s+'
        r'approx_compile_overhead_ms=([\d.]+)',
        text,
    )
    if m:
        result["first_step_ms"] = float(m.group(1))
        result["trailing_avg_ms"] = float(m.group(2))
        result["steady_tok_s"] = float(m.group(3))
        result["approx_compile_overhead_ms"] = float(m.group(4))

    # [BENCH] peak_vram=...MB compile=... mode=... window=... steps=...
    m = re.search(
        r'\[BENCH\]\s+peak_vram=([\d.]+)MB\s+'
        r'compile=(\w+)\s+'
        r'mode=(\S+)\s+'
        r'window=(\d+)\s+'
        r'steps=(\d+)',
        text,
    )
    if m:
        result["peak_vram"] = float(m.group(1))
        result["compile"] = m.group(2)
        result["mode"] = m.group(3)
        result["window"] = int(m.group(4))
        result["steps"] = int(m.group(5))

    return result


def run_saved_state_probe(env_overrides: dict) -> dict:
    """
    Probe saved autograd state using torch.autograd.graph.saved_tensors_hooks.
    Returns dict with saved_state_bytes, saved_state_count, top_contributors.
    """
    probe_script = os.path.join(PROJECT_ROOT, "scripts", "_probe_hisa_saved_state.py")

    # Generate a temporary probe script that imports the HISA module and measures saved state
    probe_code = '''
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

torch.manual_seed(42)
B, N, D, H, hd = 2, 2048, 768, 64, 12
C, k, m = 32, 4, 32
device = "cuda"

model = HierarchicalSparseAttentionV15HISA(
    D=D, H=H, hd=hd,
    num_chunks=C, top_k_chunks=k, hisa_top_m_tokens=m,
).to(device)
model.train()

x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)

saved_info = {"tensors": [], "total_bytes": 0, "count": 0}

def save_hook(tensor):
    info = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.element_size() * tensor.numel(),
    }
    saved_info["tensors"].append(info)
    saved_info["total_bytes"] += info["bytes"]
    saved_info["count"] += 1
    return tensor

def restore_hook(tensor):
    return tensor

with torch.autograd.graph.saved_tensors_hooks(save_hook, restore_hook):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = out.float().square().mean()
    loss.backward()

# Sort by bytes descending
saved_info["tensors"].sort(key=lambda t: t["bytes"], reverse=True)
top = saved_info["tensors"][:10]

print(f"SAVED_STATE_BYTES={saved_info['total_bytes']}")
print(f"SAVED_STATE_COUNT={saved_info['count']}")
for i, t in enumerate(top):
    print(f"TOP_{i}={t['shape']}|{t['dtype']}|{t['bytes']}")
'''

    with open(probe_script, "w") as f:
        f.write(probe_code)

    env = os.environ.copy()
    env.update(env_overrides)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        proc = subprocess.run(
            [PYTHON, probe_script],
            env=env, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT
        )
        output = proc.stdout

        result = {"saved_state_bytes": 0, "saved_state_count": 0, "top_contributors": []}

        m = re.search(r"SAVED_STATE_BYTES=(\d+)", output)
        if m:
            result["saved_state_bytes"] = int(m.group(1))

        m = re.search(r"SAVED_STATE_COUNT=(\d+)", output)
        if m:
            result["saved_state_count"] = int(m.group(1))

        for m in re.finditer(r"TOP_(\d+)=(\[[\d,\s]+\])\|(\w+)\|(\d+)", output):
            result["top_contributors"].append({
                "index": int(m.group(1)),
                "shape": json.loads(m.group(2)),
                "dtype": m.group(3),
                "bytes": int(m.group(4)),
            })

        return result

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.remove(probe_script)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="HISA checkpoint comparison harness")
    parser.add_argument(
        "--mode", choices=list(MODES.keys()), default=None,
        help="Run a single mode instead of all"
    )
    parser.add_argument(
        "--saved-state", action="store_true",
        help="Run saved-state probe for each mode"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Use DWARF_COMPILE=1 (confirmation pass)"
    )
    args = parser.parse_args()

    # Determine which modes to run
    if args.mode:
        run_modes = [args.mode]
    else:
        run_modes = list(MODES.keys())

    # Set compile mode
    if args.compile:
        BASE_ENV["DWARF_COMPILE"] = "1"

    git_hash = get_git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {}
    saved_state_results = {}

    print(f"HISA Checkpoint Comparison Harness")
    print(f"Git: {git_hash}")
    print(f"Compile: {'ON' if args.compile else 'OFF'}")
    print(f"Modes: {', '.join(run_modes)}")
    print(f"{'='*60}")

    for mode_name in run_modes:
        mode_config = MODES[mode_name]
        env_overrides = {**BASE_ENV}
        env_overrides.update(mode_config)

        print(f"\n--- Mode: {mode_name} ---")
        print(f"  CKPT={mode_config['DWARF_CKPT']} HISA_RECOMPUTE={mode_config['HISA_RECOMPUTE']}")

        stdout, stderr = run_trainer(env_overrides)
        bench = parse_bench_lines(stdout + stderr)

        results[mode_name] = {
            "mode": mode_name,
            "checkpoint": mode_config["DWARF_CKPT"],
            "hisa_recompute": mode_config["HISA_RECOMPUTE"],
            "git_hash": git_hash,
            "compile": args.compile,
            **bench,
        }

        if args.saved_state:
            print(f"  Running saved-state probe...")
            saved_state_results[mode_name] = run_saved_state_probe(env_overrides)
            results[mode_name]["saved_state"] = saved_state_results[mode_name]

        # Print summary
        if bench.get("peak_vram"):
            print(f"  peak_vram={bench['peak_vram']:.0f}MB  tok/s={bench.get('steady_tok_s', 'N/A')}")
        else:
            print(f"  WARNING: No [BENCH] output parsed")

    # Save JSON artifact
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_compile" if args.compile else ""
    if args.saved_state:
        suffix += "_saved"
    if args.mode:
        suffix += f"_{args.mode}"
    output_path = os.path.join(RESULTS_DIR, f"hisa_checkpoint_bench_{timestamp}{suffix}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table sorted by peak VRAM, then tok/s
    print(f"\n{'='*60}")
    print(f"Comparison Table (sorted by peak VRAM, then tok/s)")
    print(f"{'='*60}")

    def sort_key(item):
        name, data = item
        vram = data.get("peak_vram", float('inf'))
        toks = -(data.get("steady_tok_s", 0))  # negative for descending
        return (vram, toks)

    for mode_name, data in sorted(results.items(), key=sort_key):
        vram = data.get("peak_vram", "N/A")
        toks = data.get("steady_tok_s", "N/A")
        first = data.get("first_step_ms", "N/A")
        trailing = data.get("trailing_avg_ms", "N/A")
        ckpt = data.get("checkpoint", "N/A")
        replay = data.get("hisa_recompute", "N/A")
        ss_bytes = data.get("saved_state", {}).get("saved_state_bytes", "N/A")

        vram_str = f"{vram:.0f}MB" if isinstance(vram, (int, float)) else str(vram)
        toks_str = f"{toks:.0f}" if isinstance(toks, (int, float)) else str(toks)
        first_str = f"{first:.0f}ms" if isinstance(first, (int, float)) else str(first)
        trailing_str = f"{trailing:.0f}ms" if isinstance(trailing, (int, float)) else str(trailing)
        ss_str = f"{ss_bytes / 1e6:.1f}MB" if isinstance(ss_bytes, (int, float)) and ss_bytes != "N/A" else str(ss_bytes)

        print(f"  {mode_name:25s} VRAM={vram_str:>10s}  tok/s={toks_str:>6s}  "
              f"first={first_str:>8s}  trail={trailing_str:>8s}  "
              f"ckpt={ckpt}  replay={replay}"
              + (f"  saved={ss_str}" if args.saved_state else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
