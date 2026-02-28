#!/usr/bin/env bash
# Run eval_suite.py for all available new models on the RTX 3090 (CUDA_VISIBLE_DEVICES=1)
# Layer 3 is excluded — still training. Run manually once condm_layer3 completes:
#   CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_suite.py --model condm_layer3

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python3"
EVAL="benchmarks/eval_suite.py"
LOG_DIR="benchmarks/logs"
TS=$(date +%Y%m%d_%H%M%S)
RUNNER_LOG="${LOG_DIR}/eval_suite_run_all_${TS}.log"

mkdir -p "$LOG_DIR"

MODELS=(
    "standard_27m"
    "condp_27m"
    "condm_layer0"
    "condm_layer5"
    "condm_27m"
    "condm_85m"
)

echo "====================================================" | tee -a "$RUNNER_LOG"
echo " eval_suite — all models — started $(date)" | tee -a "$RUNNER_LOG"
echo " Device: RTX 3090 (CUDA_VISIBLE_DEVICES=1)" | tee -a "$RUNNER_LOG"
echo "====================================================" | tee -a "$RUNNER_LOG"

for MODEL in "${MODELS[@]}"; do
    echo "" | tee -a "$RUNNER_LOG"
    echo "----------------------------------------------------" | tee -a "$RUNNER_LOG"
    echo " Starting: $MODEL  ($(date))" | tee -a "$RUNNER_LOG"
    echo "----------------------------------------------------" | tee -a "$RUNNER_LOG"
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u "$EVAL" --model "$MODEL" 2>&1 | tee -a "$RUNNER_LOG"
    echo " Finished: $MODEL  ($(date))" | tee -a "$RUNNER_LOG"
done

echo "" | tee -a "$RUNNER_LOG"
echo "====================================================" | tee -a "$RUNNER_LOG"
echo " All evals complete — $(date)" | tee -a "$RUNNER_LOG"
echo " Results in: $LOG_DIR/eval_suite_*.json" | tee -a "$RUNNER_LOG"
echo "====================================================" | tee -a "$RUNNER_LOG"
