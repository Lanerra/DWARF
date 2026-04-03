"""
Build a pre-encoded Wikipedia training dataset using the FineWeb tokenizer.

Streams Wikipedia (20220301.en) from HuggingFace, filters docs >=2048 tokens,
encodes into 2048-token sequences with fineweb_tokenizer_32k.

Why Wikipedia:
  High definition/fact-binding density — structurally similar to FineWeb-Edu.
  "X is Y", "A was born in...", "The process involves..." — relay-compatible patterns.
  Unlike PG19 (narrative) or Stack (code), Wikipedia preserves DWARF's relay gradient.

Output:
  benchmarks/logs/wikipedia_doc_cache.json  — raw text cache
  logs/wikipedia_encoded_2048.pt            — encoded tensor (fineweb_tokenizer_32k)

Usage:
    .venv/bin/python3 -u scripts/build_dataset_wikipedia.py \
        > logs/build_dataset_wikipedia.log 2>&1 &
"""

import json, os, time, math
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_DOCS        = 400_000     # qualifying docs to collect (>=MIN_TOKENS each)
MIN_TOKENS      = 2048
SEQ_LEN         = 2048
NUM_WORKERS     = 14
WIKI_DATASET    = 'wikimedia/wikipedia'
WIKI_SUBSET     = '20231101.en'
TOKENIZER_PATH  = 'results/fineweb_tokenizer_32k.json'
DOC_CACHE_PATH  = 'benchmarks/logs/wikipedia_doc_cache.json'
ENCODED_PATH    = 'logs/wikipedia_encoded_2048.pt'
VAL_SPLIT       = 2_000

os.makedirs('logs', exist_ok=True)
os.makedirs('benchmarks/logs', exist_ok=True)

print('Wikipedia Dataset Builder (fineweb_tokenizer_32k)')
print(f'  Source:    {WIKI_DATASET} / {WIKI_SUBSET}')
print(f'  Target:    {NUM_DOCS:,} qualifying docs (≥{MIN_TOKENS} tokens)')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {ENCODED_PATH}')
print(f'  Workers:   {NUM_WORKERS} / {mp.cpu_count()} cores')
print()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f'Loaded tokenizer: {vocab_size:,} vocab')

def load_or_build_doc_cache():
    if os.path.exists(DOC_CACHE_PATH):
        print(f'Loading existing doc cache: {DOC_CACHE_PATH}')
        with open(DOC_CACHE_PATH) as f:
            docs = json.load(f)
        print(f'  Loaded {len(docs):,} cached documents')
        return docs

    print(f'Streaming Wikipedia to build doc cache ({NUM_DOCS:,} docs)...')
    docs = []
    t0 = time.time()
    ds = load_dataset(WIKI_DATASET, name=WIKI_SUBSET, split='train', streaming=True,
                      trust_remote_code=True)
    for raw in ds:
        text = raw.get('text', '')
        if len(text) < MIN_TOKENS * 3:
            continue
        enc = tokenizer.encode(text)
        if len(enc.ids) < MIN_TOKENS:
            continue
        docs.append(text)
        if len(docs) % 5_000 == 0:
            elapsed = time.time() - t0
            rate = len(docs) / elapsed
            eta = (NUM_DOCS - len(docs)) / rate / 60
            print(f'  {len(docs):,}/{NUM_DOCS:,} docs ({rate:.0f}/s, ETA {eta:.0f}m)', flush=True)
        if len(docs) >= NUM_DOCS:
            break

    print(f'Saving doc cache to {DOC_CACHE_PATH}...')
    with open(DOC_CACHE_PATH, 'w') as f:
        json.dump(docs, f)
    print(f'Cached {len(docs):,} documents ({os.path.getsize(DOC_CACHE_PATH)/1e9:.1f} GB)')
    return docs

docs = load_or_build_doc_cache()

def _encode_chunk(args):
    chunk_idx, chunk_docs, tok_path, eos_token, tmp_dir = args
    tok = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id(eos_token)
    out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npy')
    counts = []
    total_tokens = 0
    for text in chunk_docs:
        n = len(tok.encode(text).ids) + 1
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

chunk_size = math.ceil(len(docs) / NUM_WORKERS)
chunks = [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
tmp_dir = 'logs/encode_tmp_wiki'
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
        print(f'  Chunk {i+1}/{len(chunks)} | {docs_done:,} docs | '
              f'{total_tokens_so_far/1e9:.2f}B tokens', flush=True)

elapsed = time.time() - t0
print(f'Encode complete in {elapsed/60:.1f}m | {total_tokens_so_far/1e9:.2f}B tokens')
print(f'Merging {len(chunk_files)} chunk files...')

all_ids = np.concatenate([np.load(f) for f in chunk_files])
for f in chunk_files:
    os.remove(f)

n_seqs = len(all_ids) // SEQ_LEN
all_ids = all_ids[:n_seqs * SEQ_LEN]
tensor = torch.from_numpy(all_ids.reshape(n_seqs, SEQ_LEN).copy())
del all_ids
print(f'Total sequences: {n_seqs:,} ({n_seqs * SEQ_LEN / 1e9:.2f}B tokens)')

perm = torch.randperm(n_seqs)
tensor = tensor[perm]
val_seqs   = tensor[:VAL_SPLIT]
train_seqs = tensor[VAL_SPLIT:]

print(f'Train: {len(train_seqs):,} sequences | Val: {len(val_seqs):,} sequences')

save_data = {
    'train': train_seqs, 'val': val_seqs,
    'vocab_size': vocab_size, 'seq_len': SEQ_LEN,
    'num_source_docs': len(docs),
    'tokenizer': TOKENIZER_PATH,
    'source': f'{WIKI_DATASET}/{WIKI_SUBSET}',
}
torch.save(save_data, ENCODED_PATH)
print(f'Saved: {ENCODED_PATH} ({os.path.getsize(ENCODED_PATH)/1e9:.1f} GB)')
