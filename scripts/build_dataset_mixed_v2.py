"""
Re-encode the mixed dataset using mixed_tokenizer_32k.json.

Reads raw text from the three source caches:
  benchmarks/logs/fineweb_toktraining_docs.jsonl  (600K docs available, use N_FINEWEB)
  benchmarks/logs/pg19_toktraining_docs.jsonl     (chunked books)
  logs/stack_doc_cache.json                        (JSON array of code strings)

Encodes with results/mixed_tokenizer_32k.json, filters to sequences >= 2048 tokens,
truncates to exactly 2048, shuffles, reserves VAL_SPLIT for validation.

Output: logs/mixed_encoded_2048_v2.pt
  Same format as fineweb_edu_encoded_2048_v2.pt — training scripts need zero changes.

Usage:
    .venv/bin/python3 -u scripts/build_dataset_mixed_v2.py \
        > logs/build_dataset_mixed_v2.log 2>&1 &
"""

import json, os, time, multiprocessing
import numpy as np
import torch

os.environ['HF_DATASETS_OFFLINE'] = '1'

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = 'results/mixed_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/mixed_encoded_2048_v2.pt'
SEQ_LEN        = 2048
VAL_SPLIT      = 5_582
NUM_WORKERS    = 14

# Source files and doc counts
CACHE_FW    = 'benchmarks/logs/fineweb_toktraining_docs.jsonl'   # JSONL, field='text'
CACHE_PG19  = 'benchmarks/logs/pg19_toktraining_docs.jsonl'      # JSONL, field='text'
CACHE_STACK = 'logs/stack_doc_cache.json'                         # JSON array of strings

# Approximate sequence targets (will be trimmed to actual available)
# We want ~60/25/15 ratio in the final dataset; encode more than needed then sample.
N_FW_DOCS    = 800_000   # encode up to this many FW docs
N_PG19_DOCS  = 300_000   # encode up to this many PG19 chunks
N_STACK_DOCS = 150_000   # encode all stack docs

os.makedirs('logs', exist_ok=True)

print('Mixed Dataset Builder v2 — mixed_tokenizer_32k')
print(f'  Tokenizer:  {TOKENIZER_PATH}')
print(f'  Output:     {OUTPUT_PATH}')
print(f'  SEQ_LEN:    {SEQ_LEN}')
print(f'  Workers:    {NUM_WORKERS}')
print()

# ── Load tokenizer ────────────────────────────────────────────────────────────
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tok.token_to_id('<|endoftext|>') or 0
vocab_size = tok.get_vocab_size()
print(f'Tokenizer loaded: vocab={vocab_size:,}, EOS={eos_id}')

# ── Worker function ───────────────────────────────────────────────────────────

def encode_batch(args):
    """Encode a batch of raw text strings → list of int32 numpy arrays (len==SEQ_LEN)."""
    texts, tokenizer_path, seq_len = args
    tok = Tokenizer.from_file(tokenizer_path)
    seqs = []
    for text in texts:
        ids = tok.encode(text).ids
        if len(ids) < seq_len:
            continue
        seqs.append(np.array(ids[:seq_len], dtype=np.int32))
    return seqs

# ── Load raw docs ─────────────────────────────────────────────────────────────

def load_jsonl_texts(path, field, n_max):
    texts = []
    t0 = time.time()
    print(f'  Loading {path}...')
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_max:
                break
            try:
                doc = json.loads(line)
                text = doc.get(field, '')
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
            if (i+1) % 100_000 == 0:
                print(f'    {i+1:,} lines read ({time.time()-t0:.1f}s)')
    print(f'  Loaded {len(texts):,} docs in {time.time()-t0:.1f}s')
    return texts

def load_json_array_texts(path, n_max):
    print(f'  Loading {path}...')
    t0 = time.time()
    with open(path) as f:
        docs = json.load(f)
    texts = [d for d in docs if isinstance(d, str) and len(d) >= 200][:n_max]
    print(f'  Loaded {len(texts):,} docs in {time.time()-t0:.1f}s')
    return texts

# ── Encode with multiprocessing ───────────────────────────────────────────────

def encode_texts(texts, label):
    """Encode texts in parallel, return numpy array of shape (N, SEQ_LEN)."""
    t0 = time.time()
    batch_size = max(1, len(texts) // NUM_WORKERS)
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((texts[i:i+batch_size], TOKENIZER_PATH, SEQ_LEN))

    print(f'  [{label}] Encoding {len(texts):,} docs in {len(batches)} batches...')
    all_seqs = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap(encode_batch, batches)):
            all_seqs.extend(result)
            if (i+1) % max(1, len(batches)//10) == 0:
                pct = 100*(i+1)/len(batches)
                rate = len(all_seqs) / (time.time()-t0)
                print(f'    {pct:.0f}% — {len(all_seqs):,} seqs ({rate:.0f} seqs/s)')

    elapsed = time.time() - t0
    print(f'  [{label}] Done: {len(all_seqs):,} seqs in {elapsed:.1f}s')
    if not all_seqs:
        return np.zeros((0, SEQ_LEN), dtype=np.int32)
    return np.stack(all_seqs)

# ── Main ──────────────────────────────────────────────────────────────────────

print('=== Loading raw documents ===')
fw_texts    = load_jsonl_texts(CACHE_FW,    'text', N_FW_DOCS)
pg19_texts  = load_jsonl_texts(CACHE_PG19,  'text', N_PG19_DOCS)
stack_texts = load_json_array_texts(CACHE_STACK, N_STACK_DOCS)

print(f'\nLoaded: FW={len(fw_texts):,}  PG19={len(pg19_texts):,}  Stack={len(stack_texts):,}')
print()

print('=== Encoding ===')
fw_seqs    = encode_texts(fw_texts,    'FineWeb')
pg19_seqs  = encode_texts(pg19_texts,  'PG19')
stack_seqs = encode_texts(stack_texts, 'Stack')

print(f'\nEncoded seqs: FW={len(fw_seqs):,}  PG19={len(pg19_seqs):,}  Stack={len(stack_seqs):,}')

# ── Sample to 60/25/15 ratio ──────────────────────────────────────────────────

# Determine total size from available sequences
n_fw    = len(fw_seqs)
n_pg19  = len(pg19_seqs)
n_stack = len(stack_seqs)

# Max total given 60/25/15 constraints
max_from_fw    = int(n_fw    / 0.60)
max_from_pg19  = int(n_pg19  / 0.25)
max_from_stack = int(n_stack / 0.15)
total = min(max_from_fw, max_from_pg19, max_from_stack)

n_take_fw    = int(total * 0.60)
n_take_pg19  = int(total * 0.25)
n_take_stack = total - n_take_fw - n_take_pg19  # remainder to avoid rounding errors

print(f'\nSampling at 60/25/15 ratio:')
print(f'  FineWeb:  {n_take_fw:,} / {n_fw:,} available')
print(f'  PG19:     {n_take_pg19:,} / {n_pg19:,} available')
print(f'  Stack:    {n_take_stack:,} / {n_stack:,} available')
print(f'  Total:    {total:,} sequences')

rng = np.random.default_rng(42)
fw_idx    = rng.choice(n_fw,    n_take_fw,    replace=False)
pg19_idx  = rng.choice(n_pg19,  n_take_pg19,  replace=False)
stack_idx = rng.choice(n_stack, n_take_stack, replace=False)

all_seqs = np.concatenate([
    fw_seqs[fw_idx],
    pg19_seqs[pg19_idx],
    stack_seqs[stack_idx],
], axis=0)

# Global shuffle
print(f'\nShuffling {len(all_seqs):,} sequences...')
perm = rng.permutation(len(all_seqs))
all_seqs = all_seqs[perm]

# ── Train/val split ───────────────────────────────────────────────────────────

val_data   = all_seqs[:VAL_SPLIT]
train_data = all_seqs[VAL_SPLIT:]

print(f'\nSplit: train={len(train_data):,}  val={len(val_data):,}')
print(f'Total tokens: {len(train_data) * SEQ_LEN / 1e9:.2f}B (train)')

# ── Save ──────────────────────────────────────────────────────────────────────

print(f'\nSaving to {OUTPUT_PATH}...')
t0 = time.time()
torch.save({
    'train':          torch.from_numpy(train_data),
    'val':            torch.from_numpy(val_data),
    'vocab_size':     vocab_size,
    'seq_len':        SEQ_LEN,
    'eos_id':         eos_id,
    'tokenizer_path': TOKENIZER_PATH,
    'source_mix':     {'fineweb': 0.60, 'pg19': 0.25, 'stack': 0.15},
    'n_source_docs':  {'fineweb': len(fw_texts), 'pg19': len(pg19_texts),
                       'stack': len(stack_texts)},
}, OUTPUT_PATH)
print(f'Saved in {time.time()-t0:.1f}s')

size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
print(f'File size: {size_gb:.1f} GB')
print()
print(f'Done! {len(train_data):,} train seqs / {len(val_data):,} val seqs')
print(f'Tokenizer: {TOKENIZER_PATH}')
print()
print('Next step: update ENCODED_PATH in training script to:')
print(f"  _encoded_cache = '{OUTPUT_PATH}'")
print(f"  tokenizer_path = '{TOKENIZER_PATH}'")
