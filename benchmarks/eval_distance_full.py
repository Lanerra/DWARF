#!/usr/bin/env python3
"""
Extended Distance-Conditioned Loss Evaluation — condM vs condP 27M

Extends eval_distance_loss.py to cover the full 2048-token range, including
the previously unmeasured 1025–2047 bucket.

Uses wikitext-103-raw-v1 (test split, already cached) to build multiple full
2048-token chunks. This gives statistically meaningful per-bucket sample sizes
across all distance ranges including the longest lookback.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_distance_full.py

Results → benchmarks/logs/eval_distance_full_<timestamp>.json
"""

import json, math, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
TOKENIZER    = os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json')
CONDM_CKPT   = os.path.join(REPO_ROOT, 'checkpoints', '2048_condM_checkpoints', 'best.pt')
CONDP27_CKPT = os.path.join(REPO_ROOT, 'checkpoints', '27m_2048__condP_checkpoints', 'best.ptrom')
LOGS_DIR     = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

MAX_SEQ_LEN  = 2048
VOCAB_SIZE   = 32000
N_ARTICLES   = 200   # wikitext-103 test articles to use (gives ~17 full chunks)

# ─── Import architectures from the combined eval script (correct key names) ───

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'eval_combined', os.path.join(SCRIPT_DIR, 'eval_condM_vs_condP27m.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CondMTransformer    = _mod.CondMTransformer
CondP27MTransformer = _mod.CondP27MTransformer
BPETokenizerWrapper = _mod.BPETokenizerWrapper

# ─── Buckets (full 2048-token coverage) ───────────────────────────────────────

BUCKETS = [
    ('0-16',       0,    16),
    ('17-64',      17,   64),
    ('65-256',     65,  256),
    ('257-512',   257,  512),
    ('513-1024',  513, 1024),
    ('1025-2047', 1025, 2047),
]

def bucket_of(pos):
    for name, lo, hi in BUCKETS:
        if lo <= pos <= hi:
            return name
    return None

# ─── Per-sequence loss ────────────────────────────────────────────────────────

@torch.no_grad()
def per_token_loss(model, token_ids, device):
    ids = token_ids.unsqueeze(0).to(device)
    logits = model(ids)
    return F.cross_entropy(logits[0, :-1, :], ids[0, 1:], reduction='none').cpu().tolist()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Verify files
    for p in [CONDM_CKPT, CONDP27_CKPT, TOKENIZER]:
        if not os.path.exists(p):
            print(f'ERROR: not found: {p}'); sys.exit(1)

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER))

    # Load models
    print('\nLoading condM (13M, 5:1 hybrid)...')
    condm = CondMTransformer().to(device)
    condm.load_state_dict(torch.load(CONDM_CKPT, map_location=device, weights_only=True))
    condm.eval()
    print(f'  condM: {condm.param_count():,} params')

    print('Loading condP 27M (pure DSQG)...')
    condp = CondP27MTransformer().to(device)
    condp.load_state_dict(torch.load(CONDP27_CKPT, map_location=device, weights_only=True))
    condp.eval()
    print(f'  condP 27M: {condp.param_count():,} params')

    # Load wikitext-103 (already cached locally)
    print(f'\nLoading wikitext-103-raw-v1 (test, first {N_ARTICLES} non-trivial articles)...')
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    texts = [t for t in ds['text'] if len(t.strip()) > 200][:N_ARTICLES]
    print(f'  Articles: {len(texts)}')

    # Tokenize and concatenate
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t))
    print(f'  Total tokens: {len(all_ids):,}')

    # Chunk into full MAX_SEQ_LEN windows
    chunks = []
    for start in range(0, len(all_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
        chunk = all_ids[start: start + MAX_SEQ_LEN]
        if len(chunk) == MAX_SEQ_LEN:
            chunks.append(torch.tensor(chunk, dtype=torch.long))
    print(f'  Full 2048-token chunks: {len(chunks)}')
    if not chunks:
        print('ERROR: not enough tokens to form a single 2048-token chunk.')
        sys.exit(1)

    results = {
        'meta': {
            'date':            datetime.datetime.now().isoformat(),
            'n_chunks':        len(chunks),
            'total_tokens':    len(all_ids),
            'n_articles':      len(texts),
            'dataset':         'wikitext-103-raw-v1 test',
            'condm_ckpt':      CONDM_CKPT,
            'condp27_ckpt':    CONDP27_CKPT,
            'condm_test_ppl':  54.529,
            'condp27_test_ppl': 52.756,
        }
    }

    # Evaluate both models
    print('\n' + '='*72)
    print('  DISTANCE-CONDITIONED LOSS (full 2048-token range)')
    print('='*72)

    model_results = {}
    for model_name, model in [('condM', condm), ('condP_27M', condp)]:
        print(f'\n  Evaluating {model_name} over {len(chunks)} chunks...')
        t0 = time.time()
        bucket_losses = {name: [] for name, _, _ in BUCKETS}

        for ci, chunk in enumerate(chunks):
            losses = per_token_loss(model, chunk, device)
            for i, loss in enumerate(losses):
                bname = bucket_of(i)
                if bname:
                    bucket_losses[bname].append(loss)
            if (ci + 1) % 5 == 0:
                print(f'    chunk {ci+1}/{len(chunks)}...')

        elapsed = time.time() - t0
        print(f'  Done ({elapsed:.1f}s)')

        bucket_stats = {}
        print(f'\n  {model_name} results:')
        print(f'    {"Bucket":<14} {"Mean loss":>10} {"Mean PPL":>10} {"n tokens":>10}')
        for bname, lo, hi in BUCKETS:
            bl = bucket_losses[bname]
            if not bl:
                bucket_stats[bname] = {'mean_loss': None, 'mean_ppl': None, 'n': 0, 'range': (lo, hi)}
                print(f'    {bname:<14} {"—":>10} {"—":>10} {"0":>10}')
            else:
                ml = sum(bl) / len(bl)
                mp = math.exp(ml)
                bucket_stats[bname] = {'mean_loss': ml, 'mean_ppl': mp, 'n': len(bl), 'range': (lo, hi)}
                print(f'    {bname:<14} {ml:>10.4f} {mp:>10.2f} {len(bl):>10,}')

        model_results[model_name] = bucket_stats

    # Delta table
    print('\n' + '='*72)
    print('  COMPARISON: PPL delta (condM − condP27M)')
    print('  Negative = condM better; Positive = condP27M better')
    print('='*72)
    print(f'\n  {"Bucket":<14} {"condM PPL":>10} {"condP27 PPL":>12} {"Δ PPL":>10} {"winner":>10}')
    print(f'  {"-"*14} {"-"*10} {"-"*12} {"-"*10} {"-"*10}')

    for bname, _, _ in BUCKETS:
        cm = model_results['condM'][bname]
        cp = model_results['condP_27M'][bname]
        if cm['mean_ppl'] is not None and cp['mean_ppl'] is not None:
            delta = cm['mean_ppl'] - cp['mean_ppl']
            winner = 'condM' if delta < 0 else 'condP27M'
            print(f'  {bname:<14} {cm["mean_ppl"]:>10.2f} {cp["mean_ppl"]:>12.2f} '
                  f'{delta:>+10.2f} {winner:>10}')
        else:
            print(f'  {bname:<14} {"—":>10} {"—":>12} {"—":>10} {"—":>10}')

    # Weighted aggregate (cross-check vs known test PPL)
    print('\n  Weighted aggregate (mean loss across all buckets):')
    for model_name in ['condM', 'condP_27M']:
        all_l = []
        for bname, _, _ in BUCKETS:
            s = model_results[model_name][bname]
            if s['mean_loss'] is not None:
                all_l.extend([s['mean_loss']] * s['n'])
        if all_l:
            agg = sum(all_l) / len(all_l)
            print(f'    {model_name}: mean_loss={agg:.4f}  PPL={math.exp(agg):.2f}  '
                  f'(training test PPL for reference: '
                  f'{"54.529" if model_name == "condM" else "52.756"})')

    results['bucket_results'] = model_results

    # Save
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(LOGS_DIR, f'eval_distance_full_{ts}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out}')


if __name__ == '__main__':
    main()
