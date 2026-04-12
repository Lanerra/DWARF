"""
Combine FineWeb-Edu (80%) + Wikipedia (20%) into a single shuffled dataset
for continued pretraining of d768_l32.

Prerequisites:
  logs/fineweb_edu_encoded_2048_v2.pt  — already exists
  logs/wikipedia_encoded_2048.pt       — built by build_dataset_wikipedia.py

Output:
  logs/fineweb_wiki_80_20_encoded_2048.pt

Usage:
    .venv/bin/python3 -u scripts/build_dataset_combined.py
"""

import torch, os, math

FW_PATH   = 'logs/fineweb_edu_encoded_2048_v2.pt'
WIKI_PATH = 'logs/wikipedia_encoded_2048.pt'
OUT_PATH  = 'logs/fineweb_wiki_80_20_encoded_2048.pt'

# Target: 80/20 split. Cap FineWeb-Edu at 364K train seqs (same as original run)
# so the continued run sees fresh data, not the same docs again.
FW_TARGET   = 364_000   # new FineWeb-Edu sequences (skip first 364K seen before)
WIKI_TARGET = 91_000    # 20% Wikipedia (= 364K × 0.25 to get 80/20 of combined)
# 364K FW + 91K Wiki = 455K total; 364/455 = 80%, 91/455 = 20% ✓

VAL_SPLIT = 3_000

print('Building combined pretraining dataset (80% FineWeb-Edu + 20% Wikipedia)')
print()

# ── Load FineWeb-Edu ──────────────────────────────────────────────────────────
print(f'Loading FineWeb-Edu: {FW_PATH}')
fw = torch.load(FW_PATH, map_location='cpu')
fw_train = fw['train']   # 2,014,597 seqs
print(f'  Total: {len(fw_train):,} seqs')

# Skip the first 364K (seen in original training) — take a fresh slice
SEEN_FW = 364_000
if len(fw_train) < SEEN_FW + FW_TARGET:
    print(f'  Not enough sequences for fresh slice; using all with shuffle')
    perm = torch.randperm(len(fw_train))
    fw_slice = fw_train[perm[:FW_TARGET]]
else:
    # Take sequences after the first SEEN_FW
    fw_slice = fw_train[SEEN_FW : SEEN_FW + FW_TARGET]
    # Shuffle within the slice
    perm = torch.randperm(len(fw_slice))
    fw_slice = fw_slice[perm]

print(f'  Using: {len(fw_slice):,} fresh sequences (offset {SEEN_FW:,})')
del fw_train, fw

# ── Load Wikipedia ────────────────────────────────────────────────────────────
print(f'\nLoading Wikipedia: {WIKI_PATH}')
wiki = torch.load(WIKI_PATH, map_location='cpu')
wiki_train = wiki['train']
print(f'  Total: {len(wiki_train):,} seqs')

wiki_target = min(WIKI_TARGET, len(wiki_train))
perm = torch.randperm(len(wiki_train))
wiki_slice = wiki_train[perm[:wiki_target]]
print(f'  Using: {len(wiki_slice):,} sequences')
del wiki_train, wiki

# ── Combine and shuffle ───────────────────────────────────────────────────────
combined = torch.cat([fw_slice, wiki_slice], dim=0)
del fw_slice, wiki_slice

perm = torch.randperm(len(combined))
combined = combined[perm]

n_total = len(combined)
fw_count   = FW_TARGET
wiki_count = wiki_target
fw_pct   = fw_count / (fw_count + wiki_count) * 100
wiki_pct = wiki_count / (fw_count + wiki_count) * 100

print(f'\nCombined: {n_total:,} sequences')
print(f'  FineWeb-Edu: {fw_count:,} ({fw_pct:.0f}%)')
print(f'  Wikipedia:   {wiki_count:,} ({wiki_pct:.0f}%)')
print(f'  Total tokens: {n_total * 2048 / 1e9:.2f}B')

# Chinchilla check
params = 196.6e6
optimal_tokens = params * 20
print(f'\nChinchilla (197M params):')
print(f'  Optimal: {optimal_tokens/1e9:.1f}B tokens ({int(optimal_tokens/2048):,} seqs)')
print(f'  Original training: ~{364_000 * 3 * 2048 / 1e9:.1f}B tokens (~57% Chinchilla)')
print(f'  This dataset (3 epochs): ~{n_total * 3 * 2048 / 1e9:.1f}B tokens')
original_pct = 364_000 * 3 * 2048 / optimal_tokens * 100
new_pct = n_total * 3 * 2048 / optimal_tokens * 100
print(f'  After continued training: ~{original_pct + new_pct:.0f}% Chinchilla total')

# ── Val split and save ────────────────────────────────────────────────────────
val   = combined[:VAL_SPLIT]
train = combined[VAL_SPLIT:]

save_data = {
    'train': train, 'val': val,
    'vocab_size': 32000,
    'seq_len': 2048,
    'composition': {'fineweb_edu': fw_count, 'wikipedia': wiki_count},
    'fw_pct': fw_pct, 'wiki_pct': wiki_pct,
    'description': '80% FineWeb-Edu + 20% Wikipedia, shuffled. For d768_l32 continued pretraining.',
}
torch.save(save_data, OUT_PATH)
size_gb = os.path.getsize(OUT_PATH) / 1e9
print(f'\nSaved: {OUT_PATH} ({size_gb:.1f} GB)')
print('Done. Ready for train_d768_l32_continued_4090_bf16.py')
