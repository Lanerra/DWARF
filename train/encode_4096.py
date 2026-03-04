"""
Standalone dataset encoder for N=4096 context.
Reads the existing FineWeb-Edu doc cache and produces
benchmarks/logs/fineweb_encoded_4096.pt — no GPU required.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))

FW_CACHE_FILE    = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
ENCODED_OUT      = 'benchmarks/logs/fineweb_encoded_4096.pt'
TOKENIZER_PATH   = 'benchmarks/results/2048_condI_tokenizer.json'
MAX_SEQ_LEN      = 4096

def main():
    if os.path.exists(ENCODED_OUT):
        print(f'Cache already exists: {ENCODED_OUT}')
        return

    # ── load doc cache ────────────────────────────────────────────────────────
    print(f'Loading doc cache: {FW_CACHE_FILE}')
    t0 = time.time()
    with open(FW_CACHE_FILE) as fp:
        texts = json.load(fp)
    print(f'  {len(texts):,} docs loaded in {time.time()-t0:.1f}s')

    n = len(texts)
    splits = {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
    }

    # ── tokenizer ─────────────────────────────────────────────────────────────
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOKENIZER_PATH)

    class BPETokenizerWrapper:
        def encode(self, text): return _tok.encode(text).ids

    tokenizer = BPETokenizerWrapper()

    # ── encode ────────────────────────────────────────────────────────────────
    def encode_split(split_texts, name):
        t1 = time.time()
        tokens = []
        for i, text in enumerate(split_texts):
            tokens.extend(tokenizer.encode(text))
            tokens.append(3)   # EOS
            if (i + 1) % 10_000 == 0:
                print(f'  {name}: {i+1:,}/{len(split_texts):,} docs '
                      f'({len(tokens)/1e6:.1f}M tokens so far)')
        n_seqs = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
        data   = torch.tensor(tokens[:n_seqs], dtype=torch.long)
        seqs   = data.view(-1, MAX_SEQ_LEN)
        print(f'  {name}: {len(seqs):,} sequences '
              f'({len(seqs)*MAX_SEQ_LEN/1e6:.1f}M tokens) in {time.time()-t1:.1f}s')
        return seqs

    train_data = encode_split(splits['train'], 'Train')
    val_data   = encode_split(splits['val'],   'Val')
    test_data  = encode_split(splits['test'],  'Test')

    # ── save ──────────────────────────────────────────────────────────────────
    print(f'Saving to {ENCODED_OUT} ...')
    torch.save({'train': train_data, 'val': val_data, 'test': test_data},
               ENCODED_OUT)
    size_mb = os.path.getsize(ENCODED_OUT) / 1024**2
    print(f'Done — {size_mb:.0f} MB, total {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
