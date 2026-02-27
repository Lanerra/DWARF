#!/usr/bin/env bash
# launch_27m_condP_runpod.sh
#
# Run on a fresh RunPod pod AFTER runpod_setup.sh has completed.
# Creates a tmux session called "train27m", launches training, tees to log.
# Safe to run multiple times — won't start a duplicate session if one exists.
#
# Usage (from /workspace/DWARF after setup):
#   bash benchmarks/launch_27m_condP_runpod.sh
#
# Monitor:
#   tmux attach -t train27m
#   tail -f /workspace/logs/27m_condP_run.log
#
# Detach from tmux without killing training: Ctrl-B, then D

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/DWARF"
LOG_DIR="$WORKSPACE/logs"
LOG_FILE="$LOG_DIR/27m_condP_run.log"
SESSION="train27m"
SCRIPT="benchmarks/train_2048_27m_condP.py"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: $REPO_DIR not found. Run runpod_setup.sh first." >&2
    exit 1
fi
if [ ! -f "$REPO_DIR/$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found in $REPO_DIR. Run git pull." >&2
    exit 1
fi
if [ ! -f "$REPO_DIR/benchmarks/results/2048_condI_tokenizer.json" ]; then
    echo "ERROR: condI tokenizer missing at benchmarks/results/." >&2
    echo "       Run: cd $REPO_DIR && git pull" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── GPU check ────────────────────────────────────────────────────────────────
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Start (or resume) tmux session ───────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "Attach with: tmux attach -t $SESSION"
    echo "Log file:    $LOG_FILE"
    exit 0
fi

echo "Starting tmux session: $SESSION"
echo "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "
cd '$REPO_DIR' && \\
echo '=== 27M condP training started at' \$(date) '===' | tee -a '$LOG_FILE' && \\
.venv/bin/python3 -u '$SCRIPT' 2>&1 | tee -a '$LOG_FILE'
echo '=== DONE at' \$(date) '===' | tee -a '$LOG_FILE'
" Enter

echo "Training launched in background."
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION"
echo "  Tail log:  tail -f $LOG_FILE"
echo "  Detach:    Ctrl-B, D   (inside tmux)"
echo ""
echo "Checkpoint saves to: $REPO_DIR/2048_27m_condP_checkpoints/best.pt"
