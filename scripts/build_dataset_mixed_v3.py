"""
Build mixed pretraining dataset v3: 60% FineWeb-Edu + 25% PG19 + 15% The Stack.
All three sources encoded with results/mixed_tokenizer_32k.json.

Source caches (all local, no network needed):
  benchmarks/logs/condm_fineweb_edu_doc_cache.json  — JSON array of 100K FW docs
  benchmarks/logs/pg19_toktraining_docs.jsonl        — 300K PG19 chunks (10K chars each)
  logs/stack_doc_cache.json                          — JSON array of 150K code files

Output: logs/mixed_encoded_2048_v3.pt
  Same dict format as fineweb_edu_encoded_2048_v2.pt — training scripts need zero changes.

Usage:
    .venv/bin/python3 -u scripts/build_dataset_mixed_v3.py \
        > logs/build_dataset_mixed_v3.log 2>&1 &
"""

import json, os, time, multiprocessing
import numpy as np
import torch

TOKENIZER_PATH = 'results/mixed_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/mixed_encoded_2048_v3.pt'
SEQ_LEN        = 2048
VAL_SPLIT      = 5_582
NUM_WORKERS    = 14

CACHE_FW    = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'  # JSON array of strings
CACHE_PG19  = 'benchmarks/logs/pg19_toktraining_docs.jsonl'       # JSONL, field='text'
CACHE_STACK = 'logs/stack_doc_cache.json'                          # JSON array of strings

os.makedirs('logs', exist_ok=True)

print('Mixed Dataset Builder v3 — mixed_tokenizer_32k (all sources re-encoded)')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {OUTPUT_PATH}')
print()

# ── Load tokenizer ────────────────────────────────────────────────────────────
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tok.token_to_id('<|endoftext|>') or 0
vocab_size = tok.get_vocab_size()
print(f'Tokenizer: vocab={vocab_size:,}, EOS={eos_id}')

# ── Worker ────────────────────────────────────────────────────────────────────

def encode_batch(args):
    texts, tokenizer_path, seq_len = args
    tok = Tokenizer.from_file(tokenizer_path)
    seqs = []
    for text in texts:
        ids = tok.encode(text).ids
        if len(ids) < seq_len:
            continue
        seqs.append(np.array(ids[:seq_len], dtype=np.int32))
    return seqs

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json_array(path):
    print(f'  Loading {path} ...')
    t0 = time.time()
    with open(path) as f:
        docs = json.load(f)
    texts = [d for d in docs if isinstance(d, str) and len(d) >= 200]
    print(f'  Loaded {len(texts):,} docs in {time.time()-t0:.1f}s')
    return texts

def load_jsonl(path, field='text', n_max=None):
    print(f'  Loading {path} ...')
    t0 = time.time()
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n_max and i >= n_max:
                break
            try:
                doc = json.loads(line)
                text = doc.get(field, '')
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
            if (i+1) % 100_000 == 0:
                print(f'    {i+1:,} lines ({time.time()-t0:.1f}s)')
    print(f'  Loaded {len(texts):,} docs in {time.time()-t0:.1f}s')
    return texts

# ── Encode ────────────────────────────────────────────────────────────────────

def encode_texts(texts, label):
    t0 = time.time()
    batch_size = max(1, len(texts) // NUM_WORKERS)
    batches = [(texts[i:i+batch_size], TOKENIZER_PATH, SEQ_LEN)
               for i in range(0, len(texts), batch_size)]
    print(f'  [{label}] Encoding {len(texts):,} docs, {len(batches)} batches, {NUM_WORKERS} workers...')
    all_seqs = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap(encode_batch, batches)):
            all_seqs.extend(result)
            if (i+1) % max(1, len(batches)//10) == 0:
                pct = 100*(i+1)/len(batches)
                elapsed = time.time() - t0
                rate = len(all_seqs) / elapsed if elapsed > 0 else 0
                print(f'    {pct:.0f}% — {len(all_seqs):,} seqs ({rate:.0f} seqs/s)')
    elapsed = time.time() - t0
    print(f'  [{label}] Done: {len(all_seqs):,} seqs in {elapsed:.1f}s')
    if not all_seqs:
        return np.zeros((0, SEQ_LEN), dtype=np.int32)
    return np.stack(all_seqs)

# ── Main ──────────────────────────────────────────────────────────────────────

print('=== Loading raw documents ===')
fw_texts    = load_json_array(CACHE_FW)
pg19_texts  = load_jsonl(CACHE_PG19)
stack_texts = load_json_array(CACHE_STACK)
print(f'\nDocs: FW={len(fw_texts):,}  PG19={len(pg19_texts):,}  Stack={len(stack_texts):,}')
print()

print('=== Encoding with mixed_tokenizer_32k ===')
fw_seqs    = encode_texts(fw_texts,    'FineWeb')
pg19_seqs  = encode_texts(pg19_texts,  'PG19')
stack_seqs = encode_texts(stack_texts, 'Stack')
print(f'\nSeqs: FW={len(fw_seqs):,}  PG19={len(pg19_seqs):,}  Stack={len(stack_seqs):,}')

# ── 60/25/15 ratio sampling ───────────────────────────────────────────────────

# Find the largest total respecting all three constraints
max_from_fw    = int(len(fw_seqs)    / 0.60)
max_from_pg19  = int(len(pg19_seqs)  / 0.25)
max_from_stack = int(len(stack_seqs) / 0.15)
total = min(max_from_fw, max_from_pg19, max_from_stack)

n_fw    = int(total * 0.60)
n_pg19  = int(total * 0.25)
n_stack = total - n_fw - n_pg19

print(f'\nSampling at 60/25/15:')
print(f'  FineWeb: {n_fw:,} / {len(fw_seqs):,}')
print(f'  PG19:    {n_pg19:,} / {len(pg19_seqs):,}')
print(f'  Stack:   {n_stack:,} / {len(stack_seqs):,}')
print(f'  Total:   {total:,} sequences  ({total*SEQ_LEN/1e9:.2f}B tokens)')

rng = np.random.default_rng(42)
idx_fw    = rng.choice(len(fw_seqs),    n_fw,    replace=False)
idx_pg19  = rng.choice(len(pg19_seqs),  n_pg19,  replace=False)
idx_stack = rng.choice(len(stack_seqs), n_stack, replace=False)

all_seqs = np.concatenate([
    fw_seqs[idx_fw],
    pg19_seqs[idx_pg19],
    stack_seqs[idx_stack],
], axis=0)

print(f'\nShuffling {len(all_seqs):,} sequences...')
perm = rng.permutation(len(all_seqs))
all_seqs = all_seqs[perm]

val_data   = all_seqs[:VAL_SPLIT]
train_data = all_seqs[VAL_SPLIT:]

print(f'Split: train={len(train_data):,}  val={len(val_data):,}')
print(f'Train tokens: {len(train_data)*SEQ_LEN/1e9:.2f}B')

# ── Chinchilla notes ──────────────────────────────────────────────────────────
for params_m, label in [(45.6,'Moonshot-58M'), (267,'267M-D1024')]:
    chinchilla_seqs = int(20 * params_m * 1e6 / SEQ_LEN)
    pct = 100 * len(train_data) / chinchilla_seqs
    print(f'  {label} ({params_m}M): {pct:.1f}% Chinchilla ({chinchilla_seqs:,} seqs = 100%)')

# ── Save ──────────────────────────────────────────────────────────────────────
print(f'\nSaving to {OUTPUT_PATH} ...')
t0 = time.time()
torch.save({
    'train':          torch.from_numpy(train_data),
    'val':            torch.from_numpy(val_data),
    'vocab_size':     vocab_size,
    'seq_len':        SEQ_LEN,
    'eos_id':         eos_id,
    'tokenizer_path': TOKENIZER_PATH,
    'source_mix':     {'fineweb': 0.60, 'pg19': 0.25, 'stack': 0.15},
    'n_seqs':         {'fineweb': n_fw, 'pg19': n_pg19, 'stack': n_stack},
}, OUTPUT_PATH)
size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
print(f'Saved in {time.time()-t0:.1f}s  ({size_gb:.1f} GB)')
print()
print('Done.')
print(f'  Dataset:   {OUTPUT_PATH}')
print(f'  Tokenizer: {TOKENIZER_PATH}')
