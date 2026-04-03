"""
Build a pre-encoded code dataset from bigcode/the-stack-dedup.

Samples Python, JavaScript, and Rust files that are >=2048 tokens long.
These languages have strong long-range dependency structure:
  - Function defined early, called many times throughout the file
  - Import blocks at top, referenced throughout
  - Class definitions, attributes, methods cross-referencing each other

Target: 100K–200K qualifying files (configurable via TARGET_DOCS below).
Encodes with fineweb_tokenizer_32k for compatibility with build_dataset_mixed.py.

Output (no train/val split — build_dataset_mixed.py handles that):
  logs/stack_doc_cache.json   — raw text cache
  logs/stack_encoded_2048.pt  — encoded tensor {'seqs': tensor, ...}

Usage:
    .venv/bin/python3 -u scripts/build_dataset_stack.py \
        > logs/build_dataset_stack.log 2>&1 &

Note:
  bigcode/the-stack-dedup is a gated dataset; requires
      huggingface-cli login
  or HF_TOKEN env var before running. Free to access after agreement.
  Total download is large (~200GB full), but streaming avoids bulk download.
"""

import json, os, time, math
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_DOCS    = 150_000     # qualifying files to collect (<=200K per spec)
MIN_TOKENS     = 2048        # minimum sequence length to qualify
SEQ_LEN        = 2048        # training sequence length
NUM_WORKERS    = 14          # rule: >=8 cores on this machine (7950X has 16)
LANGUAGES      = ['python', 'javascript', 'rust']
TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
DOC_CACHE_PATH = 'logs/stack_doc_cache.json'
ENCODED_PATH   = 'logs/stack_encoded_2048.pt'

# Per-language caps to ensure diversity (total = TARGET_DOCS)
LANG_CAPS = {
    'python':     70_000,   # ~47% — most commonly long
    'javascript': 50_000,   # ~33%
    'rust':       30_000,   # ~20%
}
assert sum(LANG_CAPS.values()) == TARGET_DOCS, 'LANG_CAPS must sum to TARGET_DOCS'

os.makedirs('logs', exist_ok=True)
os.makedirs('logs/encode_tmp_stack', exist_ok=True)

print('The Stack (dedup) Dataset Builder')
print(f'  Source:    bigcode/the-stack-dedup, languages={LANGUAGES}')
print(f'  Target:    {TARGET_DOCS:,} qualifying files (≥{MIN_TOKENS} tokens)')
print(f'  Caps:      ' + ', '.join(f'{k}={v:,}' for k, v in LANG_CAPS.items()))
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
        print(f'  Loaded {len(docs):,} code files from cache')
        return docs

    docs = []
    lang_counts = {lang: 0 for lang in LANGUAGES}
    t0 = time.time()

    for lang in LANGUAGES:
        cap = LANG_CAPS[lang]
        print(f'Streaming {lang} (target {cap:,} files)...', flush=True)
        ds = load_dataset(
            'bigcode/the-stack-dedup',
            data_dir=f'data/{lang}',
            split='train',
            streaming=True,
        )
        skipped = 0
        for row in ds:
            code = row.get('content', '')
            if not code:
                skipped += 1
                continue
            # Quick char-length pre-filter (~2 chars/token for code)
            if len(code) < MIN_TOKENS * 2:
                skipped += 1
                continue
            enc = tokenizer.encode(code)
            if len(enc.ids) < MIN_TOKENS:
                skipped += 1
                continue
            docs.append(code)
            lang_counts[lang] += 1
            if lang_counts[lang] % 5_000 == 0:
                elapsed = time.time() - t0
                total_so_far = sum(lang_counts.values())
                print(f'  [{lang}] {lang_counts[lang]:,}/{cap:,} | '
                      f'total {total_so_far:,}/{TARGET_DOCS:,} | {elapsed:.0f}s',
                      flush=True)
            if lang_counts[lang] >= cap:
                break

        print(f'  [{lang}] done: {lang_counts[lang]:,} collected, {skipped} skipped',
              flush=True)

    print(f'\nSaving doc cache to {DOC_CACHE_PATH}...')
    with open(DOC_CACHE_PATH, 'w') as f:
        json.dump(docs, f)
    cache_gb = os.path.getsize(DOC_CACHE_PATH) / 1e9
    print(f'Cached {len(docs):,} code files ({cache_gb:.1f} GB)')
    return docs

docs = load_or_build_doc_cache()
print(f'\nEncoding {len(docs):,} code files using {NUM_WORKERS} workers...')

# ── Step 2: Multiprocessing encode ────────────────────────────────────────────
def _encode_chunk(args):
    chunk_idx, chunk_docs, tok_path, eos_token, tmp_dir = args
    tok    = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id(eos_token)
    out_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.npy')

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
tmp_dir     = 'logs/encode_tmp_stack'
worker_args = [(i, ch, TOKENIZER_PATH, '<|endoftext|>', tmp_dir)
               for i, ch in enumerate(chunks)]

chunk_files  = []
total_tokens = 0
with mp.Pool(NUM_WORKERS) as pool:
    for i, (out_path, n_tok) in enumerate(pool.imap(_encode_chunk, worker_args)):
        chunk_files.append(out_path)
        total_tokens += n_tok
        elapsed = time.time() - t0
        docs_done = min((i + 1) * chunk_size, len(docs))
        print(f'  Chunk {i+1}/{len(chunks)} done | '
              f'{docs_done:,}/{len(docs):,} files | '
              f'{total_tokens/1e9:.2f}B tokens | {elapsed:.0f}s', flush=True)

elapsed = time.time() - t0
print(f'\nEncode complete in {elapsed/60:.1f}m | {total_tokens/1e9:.2f}B tokens')
print(f'Merging {len(chunk_files)} chunks...')

all_ids = np.concatenate([np.load(f) for f in chunk_files])
for f in chunk_files:
    os.remove(f)
print(f'Merge done | raw array: {all_ids.nbytes/1e9:.1f} GB')

# ── Step 3: Reshape and shuffle ───────────────────────────────────────────────
n_seqs  = len(all_ids) // SEQ_LEN
all_ids = all_ids[:n_seqs * SEQ_LEN]
tensor  = torch.from_numpy(all_ids.reshape(n_seqs, SEQ_LEN).copy())
del all_ids
tensor  = tensor[torch.randperm(n_seqs)]
print(f'Total sequences: {n_seqs:,} ({n_seqs * SEQ_LEN / 1e9:.2f}B tokens)')

# ── Step 4: Save ──────────────────────────────────────────────────────────────
save_data = {
    'seqs':             tensor,
    'vocab_size':       vocab_size,
    'seq_len':          SEQ_LEN,
    'num_source_docs':  len(docs),
    'lang_counts':      LANG_CAPS,
    'source':           'bigcode/the-stack-dedup',
    'tokenizer':        TOKENIZER_PATH,
}
print(f'Saving to {ENCODED_PATH}...')
torch.save(save_data, ENCODED_PATH)
size_gb = os.path.getsize(ENCODED_PATH) / 1e9
print(f'Saved: {ENCODED_PATH} ({size_gb:.1f} GB)')
print(f'\nDone. {n_seqs:,} sequences ready for build_dataset_mixed.py')
