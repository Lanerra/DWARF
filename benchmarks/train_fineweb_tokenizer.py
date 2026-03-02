#!/usr/bin/env python3
"""
Train a BPE tokenizer on FineWeb-Edu documents.

Uses the existing doc cache (benchmarks/logs/condm_fineweb_edu_doc_cache.json)
so no HuggingFace streaming is required. Targets 32,000 vocab (same as condI
OWT tokenizer) for drop-in compatibility with all training scripts.

Output: benchmarks/results/fineweb_edu_tokenizer.json

Usage:
    .venv/bin/python3 benchmarks/train_fineweb_tokenizer.py
    .venv/bin/python3 benchmarks/train_fineweb_tokenizer.py --vocab-size 32000
    .venv/bin/python3 benchmarks/train_fineweb_tokenizer.py --max-docs 20000
"""

import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
CACHE_FILE  = REPO_ROOT / 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
OUTPUT_DIR  = REPO_ROOT / 'benchmarks/results'
OUTPUT_FILE = OUTPUT_DIR / 'fineweb_edu_tokenizer.json'

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_MAX_DOCS   = None


def load_texts(cache_path, max_docs=None):
    print(f'Loading FineWeb-Edu cache: {cache_path}')
    with open(cache_path) as f:
        texts = json.load(f)
    if max_docs is not None:
        texts = texts[:max_docs]
    print(f'  {len(texts):,} documents loaded')
    return texts


def train_tokenizer(texts, vocab_size):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    print(f'\nTraining BPE tokenizer (vocab_size={vocab_size:,})...')
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder        = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size     = vocab_size,
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>'],
        show_progress  = True,
        min_frequency  = 2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s | vocab: {tokenizer.get_vocab_size():,}')
    return tokenizer


def verify_tokenizer(tokenizer):
    print('\nVerifying tokenizer...')
    test_cases = [
        'The President of the United States',
        'the secret word is apple .',
        'the weather was mild and the air was still .',
        'Mathematical reasoning requires careful attention to detail.',
        'This study examined the effects of educational interventions.',
    ]
    for text in test_cases:
        ids     = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        ok      = 'OK' if decoded.strip() == text.strip() else 'MISMATCH'
        print(f'  {ok} [{len(ids):2d} tok] {repr(text[:60])}')

    passkey_words = ['apple','banana','orange','cherry','grape',
                     'lemon','mango','peach','plum','berry']
    print('\n  Passkey word single-token check:')
    all_single = True
    for word in passkey_words:
        n_plain = len(tokenizer.encode(word).ids)
        n_space = len(tokenizer.encode(' ' + word).ids)
        best    = min(n_plain, n_space)
        ok      = 'OK' if best == 1 else 'MULTI'
        if best != 1: all_single = False
        print(f'    {ok} {word!r}: plain={n_plain} spaced={n_space}')
    if all_single:
        print('  All passkey words are single tokens — passkey eval fully compatible.')
    else:
        print('  WARNING: Some passkey words are multi-token. Eval uses first token of each.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument('--max-docs',   type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument('--output',     type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    if not CACHE_FILE.exists():
        print(f'ERROR: Cache not found: {CACHE_FILE}')
        sys.exit(1)

    texts     = load_texts(CACHE_FILE, args.max_docs)
    tokenizer = train_tokenizer(texts, args.vocab_size)
    verify_tokenizer(tokenizer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f'\nSaved -> {out_path}')
    print(f'Drop-in replacement: swap 2048_condI_tokenizer.json -> fineweb_edu_tokenizer.json')


if __name__ == '__main__':
    main()
