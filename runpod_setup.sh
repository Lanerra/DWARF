#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RunPod environment setup for DWARF experiments
#
# Sets up the pod and pre-caches data — does NOT start training.
# Launch training manually after setup completes (see commands at the end).
#
# Usage:
#   bash runpod_setup.sh
#
# After setup, start a training run in a persistent tmux session:
#   tmux new-session -d -s train "python3 -u benchmarks/train_2048_condK.py 2>&1 | tee /workspace/logs/condK_run.log"
#   tmux attach -t train
#
# Requirements: pod with PyTorch base image, CUDA 12+
#   Tested: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Persistent volume (recommended):
#   Mount at /workspace — HuggingFace cache + tokenizer are stored there,
#   so subsequent runs on new pods skip the 20GB OpenWebText download.
#
# Estimated wall-clock on H100 SXM (10 epochs, 2048 tokens, 13M params):
#   ~35–45 min per condition
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_URL="https://github.com/Lanerra/DWARF.git"
WORKSPACE="/workspace"
REPO_DIR="DWARF"

echo "════════════════════════════════════════════════════════"
echo "  DWARF — RunPod Environment Setup"
echo "════════════════════════════════════════════════════════"
echo ""

# ── 1. System deps ────────────────────────────────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q tokenizers datasets tqdm

echo ""
python3 -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── 2. Clone / update repo ────────────────────────────────────────────────────
echo "[2/5] Cloning DWARF repo..."
if [ -d "$REPO_DIR" ]; then
    echo "  Directory exists — pulling latest..."
    cd "$REPO_DIR" && git pull && cd ..
else
    git clone "$REPO_URL"
fi
cd "$REPO_DIR"
echo "  Repo ready at: $(pwd)"
echo ""

# ── 3. HuggingFace cache → persistent volume ──────────────────────────────────
echo "[3/5] Configuring cache paths..."
if [ -d "$WORKSPACE" ]; then
    export HF_HOME="$WORKSPACE/hf_cache"
    mkdir -p "$HF_HOME"
    mkdir -p "$WORKSPACE/logs"
    echo "  HF cache  → $WORKSPACE/hf_cache  (persistent)"
    echo "  Log dir   → $WORKSPACE/logs"
else
    echo "  No /workspace volume detected — using default HF cache"
    echo "  (OpenWebText will re-download on each new pod)"
    mkdir -p benchmarks/logs
fi
echo ""

# ── 4. Pre-download OpenWebText ───────────────────────────────────────────────
# Streaming=True in training scripts so this isn't strictly required, but
# downloading once to the persistent volume avoids the 10-15 min wait on
# every subsequent run. Skip with SKIP_DOWNLOAD=1 if already cached.
echo "[4/5] Pre-caching OpenWebText dataset..."
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "  Skipping (SKIP_DOWNLOAD=1)"
else
    python3 -c "
from datasets import load_dataset
print('  Streaming first 1,000 docs to warm cache...')
ds = load_dataset('openwebtext', split='train', streaming=True)
for i, _ in enumerate(ds):
    if i >= 999: break
print('  Cache warmed. Full dataset streams on demand during training.')
"
fi
echo ""

# ── 5. Ensure tmux available ──────────────────────────────────────────────────
echo "[5/5] Checking tmux..."
if ! command -v tmux &>/dev/null; then
    apt-get install -q -y tmux
    echo "  Installed tmux"
else
    echo "  tmux $(tmux -V | cut -d' ' -f2) available"
fi
echo ""

# ── Ready ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Setup complete. Pod is ready."
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Working directory : $(pwd)"
echo "  Persistent logs   : ${WORKSPACE}/logs/ (if /workspace mounted)"
echo ""
echo "  ── Launch a training run ──────────────────────────────"
echo ""
echo "  # condK+RP (current best — in training queue)"
echo "  tmux new-session -d -s train \\"
echo "    \"python3 -u benchmarks/train_2048_condK_pooling.py 2>&1 | tee /workspace/logs/condKRP_run.log\""
echo ""
echo "  # condL+RP (DSQG, 24 unique dyadic offsets + D4 warm-start)"
echo "  tmux new-session -d -s train \\"
echo "    \"python3 -u benchmarks/train_2048_condLRP.py 2>&1 | tee /workspace/logs/condLRP_run.log\""
echo ""
echo "  # condN (dense-32 local + dyadic long-range, 44 offsets)"
echo "  tmux new-session -d -s train \\"
echo "    \"python3 -u benchmarks/train_2048_condN.py 2>&1 | tee /workspace/logs/condN_run.log\""
echo ""
echo "  # condK (reference — already done locally, rerun for validation)"
echo "  tmux new-session -d -s train \\"
echo "    \"python3 -u benchmarks/train_2048_condK.py 2>&1 | tee /workspace/logs/condK_run.log\""
echo ""
echo "  ── After launching ────────────────────────────────────"
echo "  Attach:   tmux attach -t train"
echo "  Detach:   Ctrl-B then D  (training keeps running)"
echo "  Monitor:  tail -f /workspace/logs/<name>.log"
echo "  Kill:     tmux kill-session -t train"
echo ""
echo "  ── GPU ────────────────────────────────────────────────"
python3 -c "
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'  {torch.cuda.get_device_name(0)}')
    print(f'  {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')
"
echo ""
