#!/usr/bin/env bash
# run_loop_14m.sh — sequential 14M autoresearch probes (8000 steps each)
# Usage: bash autoresearch/run_loop_14m.sh 2>&1 | tee autoresearch/loop_14m.log
set -euo pipefail

STEPS=8000
DEVICE=0   # 4090
VENV=".venv/bin/python3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

run_probe() {
  local dense=$1
  local sparse=$2
  local epochs=${3:-1}
  local sparse_fn=$(echo "$sparse" | tr ',' '_')
  local tag="d${dense}_s${sparse_fn}_e${epochs}"

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Probe: dense=$dense  sparse=[$sparse]  epochs=$epochs"
  echo "  Tag:   $tag"
  echo "  Start: $(date)"
  echo "════════════════════════════════════════════════════════════"

  CUDA_VISIBLE_DEVICES=$DEVICE $VENV -u autoresearch/probe_run.py \
    --dense "$dense" \
    --sparse "$sparse" \
    --steps "$STEPS" \
    --epochs "$epochs" \
    --model-size 14m \
    --tag "$tag" \
    --out autoresearch/results.tsv \
    2>&1 | tee "autoresearch/probe_${tag}.log"

  echo "  Done:  $(date)"
}

echo "Starting 14M autoresearch loop — $(date)"
echo "Steps per probe: $STEPS  (~40 min each, ~7 probes total)"
echo ""

# Baseline first (d41s3 reference)
run_probe 41 "48,128,384"

# Dense=48 variants
run_probe 48 "96,128,384"
run_probe 48 "128,384,1536"
run_probe 48 "128,512,1536"
run_probe 48 "192,512,1536"
run_probe 48 "256,768,1536"

# Narrow dense, long-range sparse (isolates dense-width effect)
run_probe 41 "128,384,1536"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All probes complete — $(date)"
echo "Results: autoresearch/results.tsv"
echo "════════════════════════════════════════════════════════════"
