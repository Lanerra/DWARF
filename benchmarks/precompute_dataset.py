"""
precompute_dataset.py — Pre-tokenize FineWeb-Edu and cache encoded tensors.

Saves train/val/test splits as packed 2048-token sequences to:
  benchmarks/logs/fineweb_encoded_2048.pt

All training scripts check for this file at startup and skip tokenization
if it exists. Pure CPU work — safe to run alongside a GPU training session.

Usage:
  .venv/bin/python3 benchmarks/precompute_dataset.py
"""

import json, os, sys, time
import torch

MAX_SEQ_LEN   = 2048
NUM_DOCS      = 100_000
FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
OUT_FILE      = 'benchmarks/logs/fineweb_encoded_2048.pt'

_script_dir     = os.path.dirname(os.path.abspath(__file__))
_tok_candidates = [
    os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
    os.path.join(_script_dir, '2048_condI_tokenizer.json'),
]

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()

def load_data():
    print(f'Loading docs from cache: {FW_CACHE_FILE}')
    with open(FW_CACHE_FILE) as f:
        texts = json.load(f)
    texts = texts[:NUM_DOCS]
    n = len(texts)
    print(f'  Loaded {n:,} docs')
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95) : int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500 : int(n * 0.95) + 5000],
    }

def encode_split(split_texts, tokenizer, split_name):
    t0 = time.time()
    tokens = []
    for i, text in enumerate(split_texts):
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
        if (i + 1) % 10_000 == 0:
            print(f'  {split_name}: {i+1:,}/{len(split_texts):,} docs ({time.time()-t0:.0f}s)')
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences ({time.time()-t0:.1f}s)')
    return seqs

if __name__ == '__main__':
    if os.path.exists(OUT_FILE):
        print(f'Cache already exists: {OUT_FILE}')
        d = torch.load(OUT_FILE, weights_only=True)
        print(f'  train: {d["train"].shape}  val: {d["val"].shape}  test: {d["test"].shape}')
        sys.exit(0)

    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI tokenizer not found')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    splits = load_data()
    t_total = time.time()
    print(f'\nEncoding (MAX_SEQ_LEN={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, 'Test')

    print(f'\nSaving to {OUT_FILE} ...')
    torch.save({'train': train_data, 'val': val_data, 'test': test_data}, OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f'Done. {size_mb:.0f} MB, total time: {time.time()-t_total:.1f}s')
