#!/usr/bin/env bash
# launch_condM_local.sh
#
# Launch condM (13.5M, 5:1 DSQG/full-attention hybrid) on the local RTX 4090.
# Creates a tmux session called "condM", tees output to benchmarks/logs/.
# Safe to run multiple times — won't start a duplicate session if one exists.
#
# Run from repo root:
#   bash benchmarks/launch_condM_local.sh
#
# Monitor:
#   tmux attach -t condM
#   tail -f benchmarks/logs/condM_run.log
#
# Detach from tmux without killing training: Ctrl-B, then D

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/benchmarks/logs"
LOG_FILE="$LOG_DIR/condM_run.log"
SESSION="condM"
SCRIPT="benchmarks/train_2048_condM.py"
VENV="$REPO_DIR/.venv/bin/python3"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found in $REPO_DIR." >&2
    exit 1
fi
if [ ! -f "$REPO_DIR/benchmarks/results/2048_condI_tokenizer.json" ]; then
    echo "ERROR: condI tokenizer missing at benchmarks/results/." >&2
    echo "       Run: cd '$REPO_DIR' && git pull" >&2
    exit 1
fi
if [ ! -f "$VENV" ]; then
    echo "ERROR: Python venv not found at $VENV." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── GPU check ────────────────────────────────────────────────────────────────
echo "=== GPU (nvidia-smi order) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""
echo "Note: CUDA device 0 = RTX 4090 (default, no CUDA_VISIBLE_DEVICES needed)"
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

# No CUDA_VISIBLE_DEVICES — default cuda:0 = RTX 4090 on this machine
tmux send-keys -t "$SESSION" "
cd '$REPO_DIR' && \\
echo '=== condM training started at' \$(date) '===' | tee -a '$LOG_FILE' && \\
'$VENV' -u '$SCRIPT' 2>&1 | tee -a '$LOG_FILE'
echo '=== DONE at' \$(date) '===' | tee -a '$LOG_FILE'
" Enter

echo "Training launched in background."
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION"
echo "  Tail log:  tail -f $LOG_FILE"
echo "  Detach:    Ctrl-B, D   (inside tmux)"
echo ""
echo "Checkpoint saves to: $REPO_DIR/2048_condM_checkpoints/best.pt"
