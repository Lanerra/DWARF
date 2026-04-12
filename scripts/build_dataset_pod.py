"""
Pod-optimised FineWeb-Edu encoder — streams directly from HuggingFace,
no local doc-cache step.  Stops as soon as TARGET_SEQS sequences are
collected.  Saves to logs/fineweb_edu_encoded_2048_v2.pt (same path the
training scripts expect).

Usage (on RunPod, from /workspace/DWARF):
    pip install datasets tokenizers -q
    python3 -u scripts/build_dataset_pod.py 2>&1 | tee logs/build_pod.log
"""

import math, os, time
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_SEQS    = 350_000      # collect this many 2048-token seqs (headroom over 331K)
SEQ_LEN        = 2048
MIN_TOKENS     = 2048         # skip docs shorter than one sequence
VAL_SPLIT      = 5_582
NUM_WORKERS    = max(4, (os.cpu_count() or 8))   # use all available cores
CHUNK_DOCS     = 5_000        # docs per encode chunk
TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/fineweb_edu_encoded_2048_v2.pt'

os.makedirs('logs', exist_ok=True)
print(f'Pod FineWeb-Edu encoder')
print(f'  Target:    {TARGET_SEQS:,} sequences ({TARGET_SEQS * SEQ_LEN / 1e9:.2f}B tokens)')
print(f'  Workers:   {NUM_WORKERS} / {os.cpu_count()} cores')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {OUTPUT_PATH}')
print()

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_PATH}')
tok_check = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tok_check.get_vocab_size()
EOS_ID = tok_check.token_to_id('<|endoftext|>') or 0
print(f'Tokenizer loaded: vocab={vocab_size:,}  EOS={EOS_ID}')
del tok_check

# ── Worker: encode one chunk of docs → flat int32 numpy array ────────────────
def _encode_chunk(args):
    docs, tok_path, eos_id = args
    tok = Tokenizer.from_file(tok_path)
    parts = []
    for text in docs:
        ids = tok.encode(text).ids
        if len(ids) < 2048:
            continue
        parts.append(ids)
        parts.append([eos_id])
    if not parts:
        return np.empty(0, dtype=np.int32)
    flat = np.fromiter(
        (x for chunk in parts for x in chunk),
        dtype=np.int32,
        count=sum(len(p) for p in parts),
    )
    return flat

# ── Stream docs from HF, batch into chunks, encode in parallel ───────────────
print('Streaming FineWeb-Edu sample-10BT from HuggingFace...')
ds = load_dataset('HuggingFaceFW/fineweb-edu',
                  name='sample-10BT', split='train', streaming=True)

pool       = mp.Pool(NUM_WORKERS)
pending    = []          # list of AsyncResult
doc_buf    = []          # buffer of raw text docs
all_tokens = []          # list of flat np.int32 arrays from finished chunks
total_seqs = 0
docs_seen  = 0
t0         = time.time()

def _flush_buf(buf):
    """Submit current buffer as one parallel encode job."""
    if buf:
        return pool.apply_async(_encode_chunk, ((buf, TOKENIZER_PATH, EOS_ID),))
    return None

def _collect_pending(pending_list, all_tokens_list):
    """Drain completed futures."""
    done, still_pending = [], []
    for fut in pending_list:
        if fut.ready():
            arr = fut.get()
            if len(arr):
                all_tokens_list.append(arr)
            done.append(fut)
        else:
            still_pending.append(fut)
    return still_pending

for row in ds:
    text = row.get('text', '')
    if len(text) < MIN_TOKENS * 3:   # cheap char pre-filter
        continue

    doc_buf.append(text)
    docs_seen += 1

    if len(doc_buf) >= CHUNK_DOCS:
        fut = _flush_buf(doc_buf)
        if fut:
            pending.append(fut)
        doc_buf = []
        pending = _collect_pending(pending, all_tokens)

        # Estimate sequences so far (rough: total tokens / SEQ_LEN)
        total_toks = sum(len(a) for a in all_tokens)
        total_seqs = total_toks // SEQ_LEN
        elapsed = time.time() - t0
        rate    = docs_seen / elapsed
        print(f'  docs_seen={docs_seen:,}  seqs≈{total_seqs:,}  '
              f'pending_chunks={len(pending)}  '
              f'{total_toks/1e9:.2f}B tok  {rate:.0f} docs/s', flush=True)

        if total_seqs >= TARGET_SEQS:
            print(f'  Reached {TARGET_SEQS:,} target sequences — stopping stream.')
            break

# Flush remaining buffer
if doc_buf:
    fut = _flush_buf(doc_buf)
    if fut:
        pending.append(fut)

# Wait for all pending encode jobs
print(f'Waiting for {len(pending)} remaining encode jobs...')
for fut in pending:
    arr = fut.get()
    if len(arr):
        all_tokens.append(arr)
pool.close()
pool.join()

# ── Concatenate, chunk into sequences ────────────────────────────────────────
print('Concatenating token arrays...')
flat = np.concatenate(all_tokens)
del all_tokens
print(f'Total tokens: {len(flat):,} ({len(flat)/1e9:.2f}B)')

n_seqs = len(flat) // SEQ_LEN
flat   = flat[:n_seqs * SEQ_LEN]
tensor = torch.from_numpy(flat.reshape(n_seqs, SEQ_LEN).copy()).long()
del flat
print(f'Total sequences: {n_seqs:,}')

# Shuffle and split
perm   = torch.randperm(n_seqs)
tensor = tensor[perm]
val    = tensor[:VAL_SPLIT]
train  = tensor[VAL_SPLIT:]

print(f'  Train: {len(train):,}  Val: {len(val):,}')

# ── Save ──────────────────────────────────────────────────────────────────────
print(f'Saving to {OUTPUT_PATH}...', flush=True)
torch.save({
    'train': train, 'val': val,
    'vocab_size': vocab_size, 'seq_len': SEQ_LEN,
    'tokenizer': TOKENIZER_PATH,
}, OUTPUT_PATH)
size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
elapsed = time.time() - t0
print(f'Done in {elapsed/60:.1f}m — {OUTPUT_PATH} ({size_gb:.1f} GB)')
print(f'Train seqs: {len(train):,}  ({len(train)*SEQ_LEN/1e9:.2f}B tokens)')
