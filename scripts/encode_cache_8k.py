#!/usr/bin/env python3
"""Encode 100K doc cache with 8K tokenizer → logs/fineweb_edu_encoded_2048_8k.pt"""
import json, time, numpy as np, torch
from multiprocessing import Pool
from tokenizers import Tokenizer

CACHE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOK     = 'results/fineweb_tokenizer_8k.json'
OUT     = 'logs/fineweb_edu_encoded_2048_8k.pt'
SEQ_LEN = 2048
WORKERS = 14

def _worker(args):
    tok_path, docs_chunk = args
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id('<|endoftext|>')
    seqs = []
    for doc in docs_chunk:
        ids = tok.encode(doc).ids + [eos]
        for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
            seqs.append(ids[i:i+SEQ_LEN])
    return np.array(seqs, dtype=np.int32) if seqs else np.empty((0,SEQ_LEN), dtype=np.int32)

print("Loading docs...")
with open(CACHE) as f:
    docs = json.load(f)
print(f"{len(docs):,} docs")

chunks = [docs[i::WORKERS] for i in range(WORKERS)]
t0 = time.time()
print(f"Encoding with {WORKERS} workers...")
with Pool(WORKERS) as pool:
    results = pool.map(_worker, [(TOK, c) for c in chunks])

all_seqs = np.concatenate([r for r in results if len(r)], axis=0)
print(f"Total seqs: {len(all_seqs):,}  ({len(all_seqs)*SEQ_LEN/1e6:.0f}M tokens)")
print(f"Encoded in {time.time()-t0:.1f}s")

# Split train/val
val_n = 2048
perm = np.random.permutation(len(all_seqs))
val   = torch.tensor(all_seqs[perm[:val_n]].astype(np.int64))
train = torch.tensor(all_seqs[perm[val_n:]].astype(np.int64))
torch.save({'train': train, 'val': val}, OUT)
print(f"Saved: train={len(train):,}  val={len(val):,} → {OUT}")
