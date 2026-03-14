# 🔬 DWARF Autoresearch

Automated experiment loop for DSQG attention architecture search.

## Setup

```bash
git checkout -b ar/experiment-name main
```

## Run an experiment

1. Edit the EXPERIMENT KNOBS section at the top of `autoresearch/train_ar.py`
2. Run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 autoresearch/train_ar.py > autoresearch/run.log 2>&1
```

## Read results

```bash
grep -A 20 '^---' autoresearch/run.log
```

## Program

The research program and mechanistic basis for each knob is documented in `autoresearch/program_ar.md`.

## Workflow Rules

**No autocommitting from autoresearcher loops.**
Training scripts read the git hash (`git rev-parse`) for checkpoint naming only.
Results are appended to `results/autoresearch/j_series_results.tsv` manually after review.
Commit manually when a run is worth preserving — not on every candidate evaluated.

**Working directory:** `/home/dlewis3/Desktop/AI/DWARF/` on `main`.
Do not run autoresearcher loops from `/tmp/` clones or feature branches.
