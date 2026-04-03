"""
Mixed pretraining dataset v5: 60% FineWeb-Edu + 25% PG19 + 15% The Stack.
Encoded with results/fineweb_tokenizer_32k.json (NOT mixed_tokenizer).

Use this dataset for mixed-continuation experiments starting from a
D768-L16 checkpoint trained on fineweb_edu_encoded_2048_v2.pt — same
tokenizer throughout, only content distribution changes in ep2+.

FineWeb-Edu: local cache benchmarks/logs/fineweb_mixed_v4_cache.jsonl
PG19:        local cache benchmarks/logs/pg19_toktraining_docs.jsonl
Stack:       local cache logs/stack_doc_cache.json

Target: ~240K train sequences (22% Chinchilla for 111.5M params)
  FW:    144K seqs (60%)
  PG19:   60K seqs (25%)
  Stack:  36K seqs (15%)

Usage:
    cd /home/dlewis3/Desktop/AI/DWARF
    .venv/bin/python3 -u scripts/build_dataset_mixed_v5_fwtok.py \
        > logs/build_dataset_mixed_v5_fwtok.log 2>&1 &
"""

import json, os, time, multiprocessing
import numpy as np
import torch
from tokenizers import Tokenizer

TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/mixed_encoded_2048_fineweb_tok.pt'
SEQ_LEN        = 2048
VAL_SPLIT      = 5_582
NUM_WORKERS    = 14
TARGET_SEQS    = 240_000   # 22% Chinchilla for 111.5M params

CACHE_FW    = 'benchmarks/logs/fineweb_mixed_v4_cache.jsonl'
CACHE_PG19  = 'benchmarks/logs/pg19_toktraining_docs.jsonl'
CACHE_STACK = 'logs/stack_doc_cache.json'

os.makedirs('logs', exist_ok=True)

print('Mixed Dataset Builder v5 — fineweb_tokenizer_32k (all caches local)')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {OUTPUT_PATH}')
print(f'  Target:    {TARGET_SEQS:,} train seqs (22% Chinchilla / 111.5M params)')
print()

tok = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tok.token_to_id('<|endoftext|>') or 0
vocab_size = tok.get_vocab_size()
print(f'Tokenizer: vocab={vocab_size:,}, EOS={eos_id}')
print()

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

def encode_texts(texts, label):
    t0 = time.time()
    batch_size = max(1, len(texts) // NUM_WORKERS)
    batches = [(texts[i:i+batch_size], TOKENIZER_PATH, SEQ_LEN)
               for i in range(0, len(texts), batch_size)]
    print(f'  [{label}] Encoding {len(texts):,} docs, {len(batches)} batches...')
    all_seqs = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap(encode_batch, batches)):
            all_seqs.extend(result)
            if (i+1) % max(1, len(batches)//10) == 0:
                pct = 100*(i+1)/len(batches)
                rate = len(all_seqs)/(time.time()-t0) if time.time()>t0 else 0
                print(f'    {pct:.0f}% — {len(all_seqs):,} seqs ({rate:.0f}/s)')
    print(f'  [{label}] Done: {len(all_seqs):,} seqs in {time.time()-t0:.1f}s')
    return np.stack(all_seqs) if all_seqs else np.zeros((0, SEQ_LEN), dtype=np.int32)

# ── Load sources from local cache ─────────────────────────────────────────────

def load_jsonl(path, field='text'):
    print(f'  Loading {path} ...')
    t0 = time.time()
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
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

def load_json_array(path):
    print(f'  Loading {path} ...')
    t0 = time.time()
    with open(path) as f:
        docs = json.load(f)
    texts = [d for d in docs if isinstance(d, str) and len(d) >= 200]
    print(f'  Loaded {len(texts):,} docs in {time.time()-t0:.1f}s')
    return texts

# ── Main ──────────────────────────────────────────────────────────────────────

print('=== Step 1: Load FineWeb-Edu from cache ===')
fw_texts = load_jsonl(CACHE_FW, field='text')
print()

print('=== Step 2: Load PG19 + Stack ===')
pg19_texts  = load_jsonl(CACHE_PG19)
stack_texts = load_json_array(CACHE_STACK)
print()

print('=== Step 3: Encode all three sources ===')
fw_seqs    = encode_texts(fw_texts,    'FineWeb')
pg19_seqs  = encode_texts(pg19_texts,  'PG19')
stack_seqs = encode_texts(stack_texts, 'Stack')
print(f'\nSeqs available: FW={len(fw_seqs):,}  PG19={len(pg19_seqs):,}  Stack={len(stack_seqs):,}')

# ── Sample at 60/25/15 to hit TARGET_SEQS ────────────────────────────────────
total = TARGET_SEQS
n_fw    = int(total * 0.60)
n_pg19  = int(total * 0.25)
n_stack = total - n_fw - n_pg19

# Sanity check we have enough
assert len(fw_seqs)    >= n_fw,    f'Not enough FW seqs: have {len(fw_seqs)}, need {n_fw}'
assert len(pg19_seqs)  >= n_pg19,  f'Not enough PG19 seqs: have {len(pg19_seqs)}, need {n_pg19}'
assert len(stack_seqs) >= n_stack, f'Not enough Stack seqs: have {len(stack_seqs)}, need {n_stack}'

print(f'\nSampling at 60/25/15 → {total:,} total seqs:')
print(f'  FineWeb: {n_fw:,} / {len(fw_seqs):,}')
print(f'  PG19:    {n_pg19:,} / {len(pg19_seqs):,}')
print(f'  Stack:   {n_stack:,} / {len(stack_seqs):,}')
print(f'  Total:   {total:,} sequences  ({total*SEQ_LEN/1e9:.2f}B tokens)')

rng = np.random.default_rng(42)
idx_fw    = rng.choice(len(fw_seqs),    n_fw,    replace=False)
idx_pg19  = rng.choice(len(pg19_seqs),  n_pg19,  replace=False)
idx_stack = rng.choice(len(stack_seqs), n_stack, replace=False)

all_seqs = np.concatenate([fw_seqs[idx_fw], pg19_seqs[idx_pg19], stack_seqs[idx_stack]])
perm = rng.permutation(len(all_seqs))
all_seqs = all_seqs[perm]

val_data   = all_seqs[:VAL_SPLIT]
train_data = all_seqs[VAL_SPLIT:]

print(f'\nSplit: train={len(train_data):,}  val={len(val_data):,}')
print(f'Train tokens: {len(train_data)*SEQ_LEN/1e9:.2f}B')

for params_m, label in [(45.6,'Moonshot-58M'), (111.5,'D768-L16'), (267,'267M-D1024')]:
    chin = int(20 * params_m * 1e6 / SEQ_LEN)
    pct  = 100 * len(train_data) / chin
    print(f'  {label}: {pct:.1f}% Chinchilla ({chin:,} seqs = 100%)')

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
print('\nDone.')
print(f'  Dataset:   {OUTPUT_PATH}')
print(f'  Tokenizer: {TOKENIZER_PATH}')
