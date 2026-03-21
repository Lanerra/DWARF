"""
Build a pre-encoded FineWeb-Edu training dataset using the new FineWeb tokenizer.

Encodes qualifying documents (>=2048 tokens) from FineWeb-Edu sample-10BT
into 2048-token sequences, saved as a single .pt tensor for fast training.

Why FineWeb-Edu for training data:
  Higher information density per token than FineWeb proper. At 147M+ scale,
  still superior to raw web text for clean language modeling. Switch to
  FineWeb proper when scaling to 1B+.

Performance note:
  Encoding uses multiprocessing across NUM_WORKERS cores (default: 14 of 16).
  Rule: always use >=8 cores for CPU-bound dataset operations on this machine.

Output:
  benchmarks/logs/fineweb_edu_1m_doc_cache.json   — raw text cache
  logs/fineweb_edu_encoded_2048_v2.pt              — encoded tensor

Usage:
    .venv/bin/python3 -u scripts/build_dataset_fineweb.py \
        > logs/build_dataset_fineweb.log 2>&1 &
"""

import json, os, time, math
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_DOCS        = 1_000_000   # qualifying docs to collect (>=MIN_TOKENS each)
MIN_TOKENS      = 2048        # minimum sequence length
SEQ_LEN         = 2048        # training sequence length
NUM_WORKERS     = 14          # cores to use for encoding (rule: >=8 on this machine)
FW_DATASET      = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
TOKENIZER_PATH  = 'results/fineweb_tokenizer_32k.json'
DOC_CACHE_PATH  = 'benchmarks/logs/fineweb_edu_1m_doc_cache.json'
ENCODED_PATH    = 'logs/fineweb_edu_encoded_2048_v2.pt'
VAL_SPLIT       = 5_582       # reserve this many sequences for validation

os.makedirs('logs', exist_ok=True)
os.makedirs('benchmarks/logs', exist_ok=True)

print('FineWeb-Edu Dataset Builder (v2 — multiprocessing encode)')
print(f'  Source:    {FW_DATASET} / {FW_SUBSET}')
print(f'  Target:    {NUM_DOCS:,} qualifying docs (≥{MIN_TOKENS} tokens)')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {ENCODED_PATH}')
print(f'  Workers:   {NUM_WORKERS} / {mp.cpu_count()} cores')
print()

# ── Load tokenizer ────────────────────────────────────────────────────────────
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f'Tokenizer not found: {TOKENIZER_PATH}\n'
        f'Run scripts/train_tokenizer_fineweb.py first.'
    )
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f'Loaded tokenizer: {vocab_size:,} vocab')

# ── Step 1: Build doc cache ───────────────────────────────────────────────────
def load_or_build_doc_cache():
    if os.path.exists(DOC_CACHE_PATH):
        print(f'Loading existing doc cache: {DOC_CACHE_PATH}')
        with open(DOC_CACHE_PATH) as f:
            docs = json.load(f)
        print(f'  Loaded {len(docs):,} cached documents')
        return docs

    print(f'Streaming FineWeb-Edu to build doc cache ({NUM_DOCS:,} docs)...')
    docs = []
    t0 = time.time()
    ds = load_dataset(FW_DATASET, name=FW_SUBSET, split='train', streaming=True)
    for raw in ds:
        text = raw.get('text', '')
        # Quick pre-filter by char length before expensive tokenization
        if len(text) < MIN_TOKENS * 3:   # ~3 chars/token avg
            continue
        # Full tokenization check
        enc = tokenizer.encode(text)
        if len(enc.ids) < MIN_TOKENS:
            continue
        docs.append(text)
        if len(docs) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = len(docs) / elapsed
            eta = (NUM_DOCS - len(docs)) / rate / 60
            print(f'  {len(docs):,}/{NUM_DOCS:,} docs ({rate:.0f}/s, ETA {eta:.0f}m)')
        if len(docs) >= NUM_DOCS:
            break

    print(f'Saving doc cache to {DOC_CACHE_PATH}...')
    with open(DOC_CACHE_PATH, 'w') as f:
        json.dump(docs, f)
    print(f'Cached {len(docs):,} documents ({os.path.getsize(DOC_CACHE_PATH)/1e9:.1f} GB)')
    return docs

docs = load_or_build_doc_cache()

# ── Step 2: Multiprocessing encode ───────────────────────────────────────────
# Worker: encode a chunk of docs, write to a temp file, return path + token count
# Writing to disk avoids accumulating a giant Python list or numpy array in worker RAM.
def _encode_chunk(args):
    chunk_idx, chunk_docs, tok_path, eos_token, tmp_dir = args
    tok = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id(eos_token)
    out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npy')
    # Encode in batches of 1000 docs, write incrementally
    BATCH = 1000
    total_tokens = 0
    # First pass: count tokens so we can pre-allocate
    counts = []
    for text in chunk_docs:
        n = len(tok.encode(text).ids) + 1  # +1 for EOS
        counts.append(n)
        total_tokens += n
    arr = np.empty(total_tokens, dtype=np.int32)
    pos = 0
    for text, n in zip(chunk_docs, counts):
        ids = tok.encode(text).ids
        arr[pos:pos+n-1] = ids
        arr[pos+n-1] = eos_id
        pos += n
    np.save(out_path, arr)
    del arr
    return out_path, total_tokens

print(f'\nEncoding {len(docs):,} documents using {NUM_WORKERS} workers...')
t0 = time.time()

# Split docs into equal chunks, one per worker
chunk_size = math.ceil(len(docs) / NUM_WORKERS)
chunks = [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
tmp_dir = 'logs/encode_tmp'
os.makedirs(tmp_dir, exist_ok=True)
args = [(i, chunk, TOKENIZER_PATH, '<|endoftext|>', tmp_dir) for i, chunk in enumerate(chunks)]

chunk_files = []
total_tokens_so_far = 0
with mp.Pool(NUM_WORKERS) as pool:
    for i, (out_path, n_tokens) in enumerate(pool.imap(_encode_chunk, args)):
        chunk_files.append(out_path)
        total_tokens_so_far += n_tokens
        elapsed = time.time() - t0
        docs_done = min((i + 1) * chunk_size, len(docs))
        rate = docs_done / elapsed
        eta = (len(docs) - docs_done) / rate / 60 if rate > 0 else 0
        print(f'  Chunk {i+1}/{len(chunks)} done | {docs_done:,}/{len(docs):,} docs | '
              f'{total_tokens_so_far/1e9:.2f}B tokens | ETA {eta:.0f}m', flush=True)

# Merge chunk files sequentially — only one chunk in RAM at a time
elapsed = time.time() - t0
print(f'Encode complete in {elapsed/60:.1f}m | {total_tokens_so_far/1e9:.2f}B tokens total')
print(f'Merging {len(chunk_files)} chunk files...', flush=True)

all_ids = np.concatenate([np.load(f) for f in chunk_files])
for f in chunk_files:
    os.remove(f)
print(f'Merge complete | RAM: {all_ids.nbytes/1e9:.1f} GB')

# Chunk into SEQ_LEN sequences
n_seqs = len(all_ids) // SEQ_LEN
all_ids = all_ids[:n_seqs * SEQ_LEN]
tensor = torch.from_numpy(all_ids.reshape(n_seqs, SEQ_LEN).copy())
del all_ids
print(f'Total sequences: {n_seqs:,} ({n_seqs * SEQ_LEN / 1e9:.2f}B tokens)')

# ── Step 3: Train/val split and save ─────────────────────────────────────────
perm = torch.randperm(n_seqs)
tensor = tensor[perm]

val_seqs   = tensor[:VAL_SPLIT]
train_seqs = tensor[VAL_SPLIT:]

print(f'\nDataset split:')
print(f'  Train: {len(train_seqs):,} sequences ({len(train_seqs)*SEQ_LEN/1e9:.2f}B tokens)')
print(f'  Val:   {len(val_seqs):,} sequences')

save_data = {'train': train_seqs, 'val': val_seqs,
             'vocab_size': vocab_size, 'seq_len': SEQ_LEN,
             'num_source_docs': len(docs),
             'tokenizer': TOKENIZER_PATH}

print(f'Saving to {ENCODED_PATH}...', flush=True)
torch.save(save_data, ENCODED_PATH)
size_gb = os.path.getsize(ENCODED_PATH) / 1e9
print(f'Saved: {ENCODED_PATH} ({size_gb:.1f} GB)')

# Chinchilla stats
print()
print('Chinchilla scaling context:')
train_tokens = len(train_seqs) * SEQ_LEN
for params_m, name in [(45, 'moonshot-58M (45M)'), (147, 'DWARF 104M (147M)'), (351, 'DWARF D=2048 (351M)')]:
    optimal = params_m * 1e6 * 20
    coverage = train_tokens / optimal * 100
    needed = int(optimal // SEQ_LEN)
    print(f'  {name}: Chinchilla optimal = {optimal/1e9:.1f}B tokens ({needed:,} seqs)')
    print(f'    This dataset: {train_tokens/1e9:.1f}B tokens = {coverage:.0f}% Chinchilla')
