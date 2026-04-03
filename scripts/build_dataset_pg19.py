"""
Build a pre-encoded PG19 (Project Gutenberg pre-1919) training dataset.

PG19 contains ~28,752 out-of-copyright books — all public domain, no copyright risk.
Books are long enough that virtually all qualify at >=2048 tokens.
Encodes with fineweb_tokenizer_32k for compatibility with the mixed-dataset builder.

Output (no train/val split — build_dataset_mixed.py handles that):
  logs/pg19_doc_cache.json   — raw text cache (~3-8 GB JSON)
  logs/pg19_encoded_2048.pt  — encoded tensor {'seqs': tensor, ...}

Usage:
    .venv/bin/python3 -u scripts/build_dataset_pg19.py \
        > logs/build_dataset_pg19.log 2>&1 &

Performance note:
  First run downloads ~11GB from HuggingFace (deepmind/pg19) and caches
  as Arrow files locally. Subsequent runs use cache directly.
  Encoding uses 14 of 16 cores via multiprocessing.
"""

import json, os, time, math
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_WORKERS    = 14          # rule: >=8 cores on this machine (7950X has 16)
MIN_TOKENS     = 2048        # minimum sequence length to qualify
SEQ_LEN        = 2048        # training sequence length
TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
DOC_CACHE_PATH = 'logs/pg19_doc_cache.json'
ENCODED_PATH   = 'logs/pg19_encoded_2048.pt'

os.makedirs('logs', exist_ok=True)
os.makedirs('logs/encode_tmp_pg19', exist_ok=True)

print('PG19 Dataset Builder')
print(f'  Source:    emozilla/pg19 (parquet mirror of deepmind/pg19, public domain)')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Output:    {ENCODED_PATH}')
print(f'  Workers:   {NUM_WORKERS}')
print()

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f'Tokenizer not found: {TOKENIZER_PATH}\n'
        f'Expected results/fineweb_tokenizer_32k.json'
    )

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
eos_id     = tokenizer.token_to_id('<|endoftext|>')
print(f'Loaded tokenizer: {vocab_size:,} vocab, EOS={eos_id}')
print()

# ── Step 1: Build doc cache ───────────────────────────────────────────────────
def load_or_build_doc_cache():
    if os.path.exists(DOC_CACHE_PATH):
        print(f'Loading existing doc cache: {DOC_CACHE_PATH}')
        with open(DOC_CACHE_PATH) as f:
            docs = json.load(f)
        print(f'  Loaded {len(docs):,} books')
        return docs

    print('Downloading PG19 from HuggingFace (emozilla/pg19, parquet mirror)...')
    print('  First run: downloads ~11GB, cached locally as Arrow files.')
    ds = load_dataset('emozilla/pg19', split='train')
    print(f'  {len(ds):,} books in train split')

    docs = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        text = row.get('text', '')
        if not text:
            skipped += 1
            continue
        # Quick char-length pre-filter before tokenizing (~4 chars/token)
        if len(text) < MIN_TOKENS * 3:
            skipped += 1
            continue
        # Full token check (most books pass easily)
        enc = tokenizer.encode(text)
        if len(enc.ids) < MIN_TOKENS:
            skipped += 1
            continue
        docs.append(text)
        if len(docs) % 1000 == 0:
            elapsed = time.time() - t0
            print(f'  {len(docs):,}/{len(ds):,} books kept '
                  f'({skipped} skipped, {elapsed:.0f}s)', flush=True)

    print(f'\nSaving doc cache to {DOC_CACHE_PATH}...')
    with open(DOC_CACHE_PATH, 'w') as f:
        json.dump(docs, f)
    cache_gb = os.path.getsize(DOC_CACHE_PATH) / 1e9
    print(f'Cached {len(docs):,} books ({cache_gb:.1f} GB, {skipped} skipped)')
    return docs

docs = load_or_build_doc_cache()
print(f'\nEncoding {len(docs):,} books using {NUM_WORKERS} workers...')

# ── Step 2: Multiprocessing encode ────────────────────────────────────────────
# Each worker encodes its chunk, writes to a temp .npy file to avoid
# accumulating giant Python lists in worker RAM. Merge sequentially after.
def _encode_chunk(args):
    chunk_idx, chunk_docs, tok_path, eos_token, tmp_dir = args
    tok    = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id(eos_token)
    out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npy')

    # Two-pass: count tokens, then fill pre-allocated array
    counts = [len(tok.encode(text).ids) + 1 for text in chunk_docs]  # +1 for EOS
    total  = sum(counts)
    arr    = np.empty(total, dtype=np.int32)
    pos    = 0
    for text, n in zip(chunk_docs, counts):
        ids           = tok.encode(text).ids
        arr[pos:pos+n-1] = ids
        arr[pos+n-1]  = eos_id
        pos += n

    np.save(out_path, arr)
    del arr
    return out_path, total

t0 = time.time()
chunk_size  = math.ceil(len(docs) / NUM_WORKERS)
chunks      = [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
tmp_dir     = 'logs/encode_tmp_pg19'
worker_args = [(i, ch, TOKENIZER_PATH, '<|endoftext|>', tmp_dir)
               for i, ch in enumerate(chunks)]

chunk_files   = []
total_tokens  = 0
with mp.Pool(NUM_WORKERS) as pool:
    for i, (out_path, n_tok) in enumerate(pool.imap(_encode_chunk, worker_args)):
        chunk_files.append(out_path)
        total_tokens += n_tok
        elapsed = time.time() - t0
        docs_done = min((i + 1) * chunk_size, len(docs))
        print(f'  Chunk {i+1}/{len(chunks)} done | '
              f'{docs_done:,}/{len(docs):,} books | '
              f'{total_tokens/1e9:.2f}B tokens | {elapsed:.0f}s', flush=True)

elapsed = time.time() - t0
print(f'\nEncode complete in {elapsed/60:.1f}m | {total_tokens/1e9:.2f}B tokens')
print(f'Merging {len(chunk_files)} chunks...')

all_ids = np.concatenate([np.load(f) for f in chunk_files])
for f in chunk_files:
    os.remove(f)
print(f'Merge done | raw array: {all_ids.nbytes/1e9:.1f} GB')

# ── Step 3: Reshape into sequences and shuffle ────────────────────────────────
n_seqs  = len(all_ids) // SEQ_LEN
all_ids = all_ids[:n_seqs * SEQ_LEN]
tensor  = torch.from_numpy(all_ids.reshape(n_seqs, SEQ_LEN).copy())
del all_ids
tensor  = tensor[torch.randperm(n_seqs)]   # shuffle
print(f'Total sequences: {n_seqs:,} ({n_seqs * SEQ_LEN / 1e9:.2f}B tokens)')

# ── Step 4: Save ──────────────────────────────────────────────────────────────
# Store all seqs (no train/val split) — build_dataset_mixed.py handles that.
save_data = {
    'seqs':             tensor,
    'vocab_size':       vocab_size,
    'seq_len':          SEQ_LEN,
    'num_source_docs':  len(docs),
    'source':           'emozilla/pg19 (mirror of deepmind/pg19)',
    'tokenizer':        TOKENIZER_PATH,
}
print(f'Saving to {ENCODED_PATH}...')
torch.save(save_data, ENCODED_PATH)
size_gb = os.path.getsize(ENCODED_PATH) / 1e9
print(f'Saved: {ENCODED_PATH} ({size_gb:.1f} GB)')
print(f'\nDone. {n_seqs:,} sequences ready for build_dataset_mixed.py')
