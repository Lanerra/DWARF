"""
build_dataset_congressional.py
================================
Stream and encode the Pile of Law congressional_hearings subset for DWARF pretraining.

Source:  pile-of-law/pile-of-law  data/train.congressional_hearings.jsonl.xz
Format:  JSONL, each line: {"text": "...", ...}
Filter:  documents with ≥ 2048 tokens after encoding with mixed_tokenizer_32k
Output:  logs/congressional_encoded_2048.pt  — same format as other DWARF datasets

Usage:
    cd /home/dlewis3/Desktop/AI/DWARF
    .venv/bin/python3 -u scripts/build_dataset_congressional.py 2>&1 | tee logs/build_congressional.log
"""

import os, sys, json, lzma, math, time, numpy as np
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT   = Path(__file__).parent.parent
TOKENIZER   = REPO_ROOT / 'results' / 'mixed_tokenizer_32k.json'
OUTPUT      = REPO_ROOT / 'logs' / 'congressional_encoded_2048.pt'
CACHE_FILE  = REPO_ROOT / 'benchmarks' / 'logs' / 'congressional_doc_cache.jsonl'

SEQ_LEN    = 2048
VAL_SPLIT  = 1024   # sequences reserved for validation
NUM_WORKERS = 14

# ── tokenizer ────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(TOKENIZER))
EOS_ID = tok.token_to_id('<|endoftext|>') or 0
VOCAB   = tok.get_vocab_size()
print(f'Tokenizer: {TOKENIZER.name}  vocab={VOCAB}  EOS={EOS_ID}')

# ── helpers ───────────────────────────────────────────────────────────────────
def encode_doc(text: str) -> list[int]:
    ids = tok.encode(text).ids
    ids.append(EOS_ID)
    return ids

def chunk_ids(ids: list[int], seq_len: int) -> list[list[int]]:
    seqs = []
    for i in range(0, len(ids) - seq_len, seq_len):
        seqs.append(ids[i:i + seq_len + 1])   # +1 for target shift
    return seqs

def worker_encode(texts: list[str]) -> np.ndarray:
    """Encode a batch of texts, return int32 array of shape (n_seqs, SEQ_LEN+1)."""
    rows = []
    for t in texts:
        ids = encode_doc(t)
        rows.extend(chunk_ids(ids, SEQ_LEN))
    if not rows:
        return np.empty((0, SEQ_LEN + 1), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def _encode_and_save(args):
    """Module-level function (picklable) for multiprocessing pool."""
    idx, texts = args
    arr = worker_encode(texts)
    tmp_dir = REPO_ROOT / 'benchmarks' / 'logs' / '_congressional_chunks'
    path = tmp_dir / f'chunk_{idx:04d}.npy'
    np.save(str(path), arr)
    return idx, len(arr)

# ── download / cache ─────────────────────────────────────────────────────────
def stream_and_cache():
    """Stream the xz file from HuggingFace and cache qualifying texts."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        print(f'Cache exists: {CACHE_FILE}  — skipping download')
        return

    from huggingface_hub import hf_hub_download
    import io, tempfile

    print('Downloading data/train.congressional_hearings.jsonl.xz …')
    t0 = time.time()
    xz_path = hf_hub_download(
        repo_id='pile-of-law/pile-of-law',
        filename='data/train.congressional_hearings.jsonl.xz',
        repo_type='dataset',
        local_dir=str(REPO_ROOT / 'benchmarks' / 'logs' / '_pile_tmp'),
    )
    print(f'  Downloaded in {time.time()-t0:.0f}s: {xz_path}')

    print('Streaming + caching docs ≥ 2048 tokens (quick char-based pre-filter ≥ 8192 chars)…')
    n_total = n_kept = 0
    t0 = time.time()
    with lzma.open(xz_path, 'rt', encoding='utf-8', errors='replace') as fh, \
         open(CACHE_FILE, 'w') as out:
        for line in fh:
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text', obj.get('content', ''))
            if not text or len(text) < 8_192:   # fast pre-filter
                continue
            out.write(json.dumps({'text': text}) + '\n')
            n_kept += 1
            if n_total % 5_000 == 0:
                elapsed = time.time() - t0
                print(f'  {n_total:>8,} docs scanned  {n_kept:>6,} kept  {elapsed:.0f}s', flush=True)

    print(f'Cached {n_kept:,} long docs from {n_total:,} total  ({time.time()-t0:.0f}s)')
    print(f'Cache: {CACHE_FILE}')

# ── encode ────────────────────────────────────────────────────────────────────
def encode_from_cache():
    import torch

    print(f'\nLoading cache: {CACHE_FILE}')
    with open(CACHE_FILE) as f:
        docs = [json.loads(l)['text'] for l in f if l.strip()]
    print(f'  Loaded {len(docs):,} docs')

    # filter by actual token length
    print('Filtering docs with ≥ 2048 tokens …')
    qualifying = []
    for i, d in enumerate(docs):
        if len(encode_doc(d)) >= SEQ_LEN + 1:
            qualifying.append(d)
        if (i + 1) % 5_000 == 0:
            print(f'  {i+1:>6,}/{len(docs):,}  {len(qualifying):,} qualifying', flush=True)
    print(f'  {len(qualifying):,} docs qualify out of {len(docs):,}')

    if not qualifying:
        print('ERROR: no qualifying docs found!')
        sys.exit(1)

    # multiprocessing encode
    print(f'\nEncoding with {NUM_WORKERS} workers …')
    chunk_size = max(1, len(qualifying) // NUM_WORKERS)
    batches = [qualifying[i:i+chunk_size] for i in range(0, len(qualifying), chunk_size)]

    tmp_dir = REPO_ROOT / 'benchmarks' / 'logs' / '_congressional_chunks'
    tmp_dir.mkdir(exist_ok=True)

    t0 = time.time()
    with mp.Pool(NUM_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(_encode_and_save, enumerate(batches)):
            idx, n = r
            results.append(r)
            print(f'  chunk {idx:3d}: {n:,} seqs  ({time.time()-t0:.0f}s)', flush=True)

    # merge
    print('\nMerging chunks …')
    total_seqs = sum(n for _, n in results)
    print(f'  Total sequences: {total_seqs:,}')

    all_arrays = []
    for idx in range(len(batches)):
        path = tmp_dir / f'chunk_{idx:04d}.npy'
        if path.exists():
            all_arrays.append(np.load(str(path)))

    data = np.concatenate(all_arrays, axis=0)
    print(f'  Merged shape: {data.shape}  dtype: {data.dtype}')

    # shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(data))
    data = data[perm]

    # train/val split
    val_n   = min(VAL_SPLIT, len(data) // 10)
    val     = data[:val_n]
    train   = data[val_n:]
    print(f'  Train: {len(train):,}  Val: {len(val):,} seqs')

    # save
    print(f'\nSaving to {OUTPUT} …')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'train': torch.from_numpy(train),
        'val':   torch.from_numpy(val),
        'vocab_size': VOCAB,
        'seq_len':    SEQ_LEN,
        'num_source_docs': len(qualifying),
        'tokenizer_path': str(TOKENIZER),
        'note': 'Pile of Law congressional_hearings ≥2048 tokens, mixed_tokenizer_32k',
    }, str(OUTPUT))
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f'Saved: {OUTPUT}  ({size_mb:.0f} MB)')
    print(f'\nChinchilla context:')
    for params_m, label in [(45.6, 'Moonshot-58M'), (267, '267M'), (478.7, '478M')]:
        chinchilla_seqs = int(20 * params_m * 1e6 / SEQ_LEN)
        pct = 100 * len(train) / chinchilla_seqs
        print(f'  {label} ({params_m}M params): {chinchilla_seqs:,} seqs = 100% Chinchilla → {pct:.1f}% of this dataset')

    # cleanup tmp
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    stream_and_cache()
    encode_from_cache()
    print('\nDone.')
