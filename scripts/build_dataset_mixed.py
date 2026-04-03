"""
Build the mixed pretraining dataset: 60% FineWeb-Edu + 25% PG19 + 15% The Stack.

Reads the three pre-encoded shard files (must exist before running):
  logs/fineweb_edu_encoded_2048_v2.pt   — FineWeb-Edu (already built)
  logs/pg19_encoded_2048.pt             — PG19 (build_dataset_pg19.py)
  logs/stack_encoded_2048.pt            — The Stack (build_dataset_stack.py)

Then:
  1. Samples each source at the target ratio (60/25/15)
  2. Interleaves and globally shuffles
  3. Reserves VAL_SPLIT sequences for validation
  4. Saves in exactly the same format as fineweb_edu_encoded_2048_v2.pt
     so training scripts need zero changes — just point to the new file.

Output:
  logs/mixed_encoded_2048_v1.pt   — mixed dataset

Why these ratios:
  60% FineWeb-Edu: maintains comparability with existing baselines;
      general educational web text, strong long-range dependencies.
  25% PG19: public-domain books, very long docs, natural narrative
      long-range dependencies. Good relay activation signal.
  15% The Stack (dedup): code has explicit long-range references
      (function def → call sites). Boosts relay for technical text.

Usage:
    .venv/bin/python3 -u scripts/build_dataset_mixed.py \
        > logs/build_dataset_mixed.log 2>&1 &

This script is CPU-only and completes in seconds once the three shards exist.
"""

import os, time
import torch

# ── Config ────────────────────────────────────────────────────────────────────
FINEWEB_PATH = 'logs/fineweb_edu_encoded_2048_v2.pt'
PG19_PATH    = 'logs/pg19_encoded_2048.pt'
STACK_PATH   = 'logs/stack_encoded_2048.pt'
OUTPUT_PATH  = 'logs/mixed_encoded_2048_v1.pt'

# Sampling ratios (must sum to 1.0)
RATIO_FINEWEB = 0.60
RATIO_PG19    = 0.25
RATIO_STACK   = 0.15
assert abs(RATIO_FINEWEB + RATIO_PG19 + RATIO_STACK - 1.0) < 1e-9, 'Ratios must sum to 1.0'

# Size target: use FineWeb-Edu train split size as the anchor.
# The mixed dataset will have the same total number of training sequences,
# sampled at the above ratios from each source.
VAL_SPLIT    = 5_582   # match existing baseline

os.makedirs('logs', exist_ok=True)

print('Mixed Dataset Builder — 60% FineWeb-Edu / 25% PG19 / 15% The Stack')
print(f'  Ratios: FineWeb={RATIO_FINEWEB:.0%}  PG19={RATIO_PG19:.0%}  Stack={RATIO_STACK:.0%}')
print()

# ── Check inputs ──────────────────────────────────────────────────────────────
for path in [FINEWEB_PATH, PG19_PATH, STACK_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing input shard: {path}\n'
            f'Run the corresponding build_dataset_*.py script first.'
        )

# ── Load shards ───────────────────────────────────────────────────────────────
print('Loading shards...', flush=True)
t0 = time.time()

print(f'  Loading FineWeb-Edu: {FINEWEB_PATH}')
fw_data  = torch.load(FINEWEB_PATH, weights_only=True)
# FineWeb v2 stores train/val split already; extract train seqs
fw_train = fw_data['train']   # [N, 2048] int16 or int32
fw_val   = fw_data['val']
vocab_size = fw_data.get('vocab_size', 32000)
seq_len    = fw_data.get('seq_len', 2048)
print(f'    FineWeb-Edu train: {len(fw_train):,} seqs | val: {len(fw_val):,} seqs')

print(f'  Loading PG19: {PG19_PATH}')
pg_data  = torch.load(PG19_PATH, weights_only=True)
pg_seqs  = pg_data['seqs']    # [N, 2048] — no pre-split
print(f'    PG19: {len(pg_seqs):,} seqs')

print(f'  Loading The Stack: {STACK_PATH}')
st_data  = torch.load(STACK_PATH, weights_only=True)
st_seqs  = st_data['seqs']    # [N, 2048]
print(f'    Stack: {len(st_seqs):,} seqs')

elapsed = time.time() - t0
print(f'  Loaded in {elapsed:.1f}s')
print()

# ── Compute sample counts ──────────────────────────────────────────────────────
# Anchor total to FineWeb-Edu train size (consistent with existing training runs).
total_train = len(fw_train)   # e.g. 445,312 for a Moonshot-scale run
n_fw  = total_train             # use all FineWeb-Edu train seqs
n_pg  = int(n_fw * (RATIO_PG19  / RATIO_FINEWEB))
n_st  = int(n_fw * (RATIO_STACK / RATIO_FINEWEB))
total_mixed = n_fw + n_pg + n_st

print(f'Sampling plan (anchored to FineWeb-Edu train size = {n_fw:,}):')
print(f'  FineWeb-Edu : {n_fw:,} seqs ({n_fw / total_mixed:.1%})')
print(f'  PG19        : {n_pg:,} seqs ({n_pg / total_mixed:.1%})')
print(f'  The Stack   : {n_st:,} seqs ({n_st / total_mixed:.1%})')
print(f'  Total       : {total_mixed:,} seqs')
print()

# Validate we have enough sequences in each source
if len(pg_seqs) < n_pg:
    print(f'WARNING: PG19 only has {len(pg_seqs):,} seqs, wanted {n_pg:,}. '
          f'Using all available.')
    n_pg = len(pg_seqs)

if len(st_seqs) < n_st:
    print(f'WARNING: Stack only has {len(st_seqs):,} seqs, wanted {n_st:,}. '
          f'Using all available.')
    n_st = len(st_seqs)

# ── Sample and interleave ──────────────────────────────────────────────────────
print('Sampling and interleaving...')
t0 = time.time()

# FineWeb: use as-is (already shuffled in the v2 build)
fw_sample  = fw_train[:n_fw]

# PG19: random sample without replacement
pg_perm    = torch.randperm(len(pg_seqs))[:n_pg]
pg_sample  = pg_seqs[pg_perm]

# Stack: random sample without replacement
st_perm    = torch.randperm(len(st_seqs))[:n_st]
st_sample  = st_seqs[st_perm]

# Ensure all tensors are int32 (may differ between shards)
fw_sample  = fw_sample.to(torch.int32)
pg_sample  = pg_sample.to(torch.int32)
st_sample  = st_sample.to(torch.int32)

# Concatenate and globally shuffle
combined   = torch.cat([fw_sample, pg_sample, st_sample], dim=0)
n_combined = len(combined)
perm       = torch.randperm(n_combined)
combined   = combined[perm]

elapsed = time.time() - t0
print(f'  Interleaved and shuffled {n_combined:,} seqs in {elapsed:.1f}s')

# ── Train / val split ─────────────────────────────────────────────────────────
# Use the existing FineWeb-Edu val split (already held out, no contamination).
# The combined tensor is all-train; val comes from FineWeb-Edu's held-out set.
val_seqs   = fw_val
train_seqs = combined

print(f'\nDataset split:')
print(f'  Train : {len(train_seqs):,} seqs ({len(train_seqs) * seq_len / 1e9:.2f}B tokens)')
print(f'  Val   : {len(val_seqs):,} seqs  (reused from FineWeb-Edu val split)')
print()

# ── Save ──────────────────────────────────────────────────────────────────────
# Exact same format as fineweb_edu_encoded_2048_v2.pt — drop-in replacement
save_data = {
    'train':            train_seqs,
    'val':              val_seqs,
    'vocab_size':       vocab_size,
    'seq_len':          seq_len,
    'source_breakdown': {
        'fineweb_edu': n_fw,
        'pg19':        n_pg,
        'stack':       n_st,
    },
    'ratios': {
        'fineweb_edu': RATIO_FINEWEB,
        'pg19':        RATIO_PG19,
        'stack':       RATIO_STACK,
    },
    'tokenizer': 'results/fineweb_tokenizer_32k.json',
}

print(f'Saving to {OUTPUT_PATH}...')
t0 = time.time()
torch.save(save_data, OUTPUT_PATH)
elapsed = time.time() - t0
size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
print(f'Saved: {OUTPUT_PATH} ({size_gb:.1f} GB) in {elapsed:.1f}s')

# ── Chinchilla stats ──────────────────────────────────────────────────────────
train_tokens = len(train_seqs) * seq_len
print()
print('Chinchilla scaling context:')
for params_m, name in [
    (45,  'Moonshot-58M (45M params)'),
    (55,  'Depth16-55M (55M params)'),
    (147, 'DWARF 104M D1024 (147M params)'),
    (267, 'DWARF 267M (267M params)'),
]:
    optimal  = params_m * 1e6 * 20
    coverage = train_tokens / optimal * 100
    needed   = int(optimal // seq_len)
    print(f'  {name}:')
    print(f'    Chinchilla optimal = {optimal/1e9:.2f}B tokens ({needed:,} seqs)')
    print(f'    This dataset       = {train_tokens/1e9:.2f}B tokens = {coverage:.0f}% Chinchilla')

print()
print('To use in training, set:')
print(f'  ENCODED_PATH = \'{OUTPUT_PATH}\'')
print('No other training script changes required.')
