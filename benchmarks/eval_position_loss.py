#!/usr/bin/env python3
"""
Fine-Grained Per-Position Loss Evaluation

For each model, runs wikitext-103 test set and records cross-entropy loss
at every sequence position (0 to 2047). Outputs:
  - Mean loss at each position (averaged over all test sequences)
  - Smoothed loss curve (window=16)
  - Loss grouped by distance band (finer than eval_suite.py)
  - JSON with full per-position statistics

Tests GPT-5.2-Thinking's prediction: "Distance-conditioned errors should not be
smooth — they should show banded structure tied to DSQG support holes and
path-count sparsity."

Support holes after 3 DSQG hops (as computed by GPT-5.2):
  673-687, 689-703, 737-751, 753-767, 929-943, 945-959, 1985-2015, 2017-2047

If this prediction holds, we expect loss spikes at these positions relative
to neighboring positions.

Usage:
  cd /home/dlewis3/Desktop/AI/DWARF
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_position_loss.py --model condm_layer5
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_position_loss.py --all
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_position_loss.py --all --skip_slow

Results: benchmarks/logs/pos_loss_<model>_<timestamp>.json
"""

import argparse, json, math, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
TOKENIZER   = os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json')
CKPT_ROOT   = os.path.join(REPO_ROOT, 'checkpoints')
LOGS_DIR    = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

MAX_SEQ_LEN = 2048
VOCAB_SIZE  = 32000
N_CHUNKS    = 50   # 50 × 2048 = ~100K tokens — enough for stable per-position stats

# ─── Support hole predictions from GPT-5.2-Thinking analysis ─────────────────
# (positions unreachable after exactly 3 DSQG hops using condN offset set)
# These are the bands we expect to show elevated loss vs. neighbors.
PREDICTED_HOLES = [
    (673, 687), (689, 703), (737, 751), (753, 767),
    (929, 943), (945, 959), (1985, 2015), (2017, 2047),
]

# ─── Model registry ───────────────────────────────────────────────────────────
# Importing the full architecture from eval_suite.py to avoid duplication
sys.path.insert(0, SCRIPT_DIR)
from eval_suite import (
    MODEL_REGISTRY, build_model, load_model, load_tokenizer,
    MAX_SEQ_LEN as EVS_MAX_SEQ_LEN,
)

# ─── Wikitext-103 loader ──────────────────────────────────────────────────────

def load_wikitext_chunks(tokenizer, n_chunks=N_CHUNKS):
    """Load wikitext-103 test split and return full 2048-token chunks."""
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1',
                          split='test', cache_dir=cache_dir)
        texts = [row['text'] for row in ds if row['text'].strip()]
    except Exception as e:
        print(f'  [WARN] Could not load wikitext-103: {e}')
        return None

    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
        if len(tokens) >= n_chunks * MAX_SEQ_LEN:
            break

    n = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    chunks = data.view(-1, MAX_SEQ_LEN)[:n_chunks]
    print(f'  Loaded {len(chunks)} wikitext-103 chunks '
          f'({len(chunks) * MAX_SEQ_LEN:,} tokens)')
    return chunks


# ─── Per-position loss ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_position_loss(model, chunks, device, batch_size=4):
    """
    Returns per_position_loss: tensor of shape [MAX_SEQ_LEN - 1]
    position i = mean loss when predicting token at position i+1
    given context tokens 0..i.
    """
    model.eval()
    pos_loss_sum   = torch.zeros(MAX_SEQ_LEN - 1)
    pos_loss_count = torch.zeros(MAX_SEQ_LEN - 1)

    n_batches = math.ceil(len(chunks) / batch_size)
    for bi in range(n_batches):
        batch = chunks[bi * batch_size: (bi + 1) * batch_size].to(device)
        x = batch[:, :-1]    # [B, N-1]
        y = batch[:,  1:]    # [B, N-1]
        B, N = x.shape

        logits = model(x)    # [B, N, V]
        # Per-token cross-entropy loss, no reduction
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction='none'
        ).reshape(B, N)      # [B, N]

        # Accumulate: position i means predicting token at input position i+1
        for pos in range(N):
            pos_loss_sum[pos]   += loss_per_token[:, pos].sum().item()
            pos_loss_count[pos] += B

        if (bi + 1) % 10 == 0:
            print(f'    {bi+1}/{n_batches} batches...')

    return pos_loss_sum / pos_loss_count.clamp(min=1)


def smooth(x, window=16):
    """Uniform moving average smoothing. Returns list of Python floats."""
    arr = [float(v) for v in (x.tolist() if hasattr(x, 'tolist') else x)]
    padded = arr[:window//2] + arr + arr[-window//2:]
    return [float(sum(padded[i:i+window]) / window) for i in range(len(arr))]


def bucket_positions(per_pos_loss, buckets):
    """Group per-position loss into named buckets."""
    results = {}
    for label, lo, hi in buckets:
        vals = per_pos_loss[lo:hi]
        ppl  = math.exp(min(vals.mean().item(), 20))
        results[label] = {
            'lo': lo, 'hi': hi,
            'mean_loss': vals.mean().item(),
            'ppl':       ppl,
            'n':         len(vals),
        }
    return results


def check_hole_predictions(per_pos_loss):
    """
    For each predicted support hole, compare mean loss in that band
    vs mean loss in the 16-position windows immediately before and after.
    A positive delta means the hole shows elevated loss (prediction confirmed).
    """
    results = []
    pos_loss = per_pos_loss.numpy()
    for lo, hi in PREDICTED_HOLES:
        if hi >= len(pos_loss): continue
        band_loss  = pos_loss[lo:hi].mean()
        before_lo  = max(0, lo - 16)
        after_hi   = min(len(pos_loss), hi + 16)
        before_loss = pos_loss[before_lo:lo].mean() if lo > before_lo else band_loss
        after_loss  = pos_loss[hi:after_hi].mean()  if hi < after_hi  else band_loss
        context_loss = (before_loss + after_loss) / 2
        delta = band_loss - context_loss
        results.append({
            'band':         f'{lo}-{hi}',
            'band_loss':    float(band_loss),
            'context_loss': float(context_loss),
            'delta':        float(delta),
            'prediction':   'confirmed' if delta > 0.02 else 'not_confirmed',
        })
    return results


# ─── Distance buckets ─────────────────────────────────────────────────────────

# Fine-grained (position = context length available when predicting each token)
FINE_BUCKETS = [
    ('0-15',       0,    16),
    ('16-32',     16,    33),    # edge of dense coverage
    ('33-47',     33,    48),    # DSQG blind spot (not directly addressable)
    ('48-63',     48,    64),
    ('64-127',    64,   128),
    ('128-255',  128,   256),
    ('256-511',  256,   512),
    ('512-767',  512,   768),
    ('768-1023', 768,  1024),
    ('1024-1535',1024, 1536),
    ('1536-2046',1536, 2047),
]

COARSE_BUCKETS = [
    ('0-16',       0,    17),
    ('17-64',     17,    65),
    ('65-256',    65,   257),
    ('257-512',  257,   513),
    ('513-1024', 513,  1025),
    ('1025-2046',1025, 2047),
]


# ─── Main evaluation ──────────────────────────────────────────────────────────

def eval_model_position_loss(model_name, device):
    cfg = MODEL_REGISTRY[model_name]
    print(f'\n{"="*60}')
    print(f'  pos_loss: {model_name}  ({cfg["label"]})')
    print(f'{"="*60}')

    tokenizer = load_tokenizer()
    model, n_params = load_model(cfg, device)

    chunks = load_wikitext_chunks(tokenizer, n_chunks=N_CHUNKS)
    if chunks is None:
        print('  [ERROR] Could not load wikitext-103 — skipping')
        return None

    print(f'  Computing per-position loss...')
    t0         = time.time()
    per_pos    = compute_position_loss(model, chunks, device)
    elapsed    = time.time() - t0
    overall_ppl = math.exp(min(per_pos.mean().item(), 20))
    print(f'  Done in {elapsed:.1f}s | Overall PPL: {overall_ppl:.2f}')

    # Coarse and fine buckets
    coarse  = bucket_positions(per_pos, [(l,lo,hi) for l,lo,hi in COARSE_BUCKETS])
    fine    = bucket_positions(per_pos, [(l,lo,hi) for l,lo,hi in FINE_BUCKETS])

    # Support hole predictions
    hole_check = check_hole_predictions(per_pos)
    confirmed  = sum(1 for r in hole_check if r['prediction'] == 'confirmed')
    print(f'\n  Support hole prediction check ({confirmed}/{len(hole_check)} confirmed):')
    for r in hole_check:
        sign = '+' if r['delta'] > 0 else ''
        mark = '✓' if r['prediction'] == 'confirmed' else '·'
        print(f'    {mark} band {r["band"]:10s}: delta={sign}{r["delta"]:.4f}')

    # Print fine-grained table
    print(f'\n  Fine-grained per-position PPL:')
    print(f'  {"Position":15s} {"PPL":>8s}  {"Note":30s}')
    print(f'  {"-"*55}')
    for label, lo, hi in FINE_BUCKETS:
        v   = fine[label]
        note = ''
        if label == '33-47':
            note = '← DSQG 1-hop blind spot'
        elif label == '0-15':
            note = '← minimal context'
        elif label == '1536-2046':
            note = '← beyond max DSQG offset'
        print(f'  {label:15s} {v["ppl"]:>8.2f}  {note:30s}')

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(LOGS_DIR, f'pos_loss_{model_name}_{ts}.json')

    result = {
        'model':          model_name,
        'label':          cfg['label'],
        'n_params':       n_params,
        'n_chunks':       len(chunks),
        'overall_ppl':    overall_ppl,
        'elapsed_s':      elapsed,
        'timestamp':      ts,
        # Full per-position data (2047 values)
        'per_position_loss': per_pos.tolist(),
        'per_position_ppl':  [math.exp(min(v, 20)) for v in per_pos.tolist()],
        # Smoothed (window=16) for plotting
        'smoothed_loss_w16': smooth(per_pos, window=16),
        # Buckets
        'coarse_buckets': coarse,
        'fine_buckets':   fine,
        # Support hole test
        'hole_predictions': hole_check,
        'holes_confirmed':  confirmed,
        'holes_total':      len(hole_check),
    }

    class _NumpySafeEncoder(json.JSONEncoder):
        def default(self, o):
            # Catch any numpy scalar that slipped through explicit casts
            try:
                return float(o)
            except (TypeError, ValueError):
                return super().default(o)

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, cls=_NumpySafeEncoder)
    print(f'\n  Saved → {out_path}')
    return result


def main():
    parser = argparse.ArgumentParser(description='Per-position loss eval')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', choices=list(MODEL_REGISTRY.keys()),
                       help='Single model to evaluate')
    group.add_argument('--all', action='store_true',
                       help='Run all registered models')
    parser.add_argument('--skip_slow', action='store_true',
                        help='Skip condm_27m and condp_27m (slow on 3090)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    if args.model:
        models = [args.model]
    else:
        # Post-condK models most relevant for support structure analysis
        POST_CONDK = [
            'condm_layer0',   # layer 0 ablation
            'condm_layer5',   # condM baseline (most relevant — 5 DSQG before full attn)
            'condm_layer3',   # layer 3 ablation (only if checkpoint exists)
            'condp_27m',      # pure DSQG — no full attention at all
            'standard_27m',   # baseline — no DSQG structure
            'condm_27m',      # 27M hybrid
        ]
        if args.skip_slow:
            POST_CONDK = [m for m in POST_CONDK
                          if m not in ('condm_27m', 'condp_27m')]
        # Only include models whose checkpoint actually exists
        models = []
        for m in POST_CONDK:
            cfg  = MODEL_REGISTRY.get(m)
            if cfg and os.path.exists(cfg['checkpoint']):
                models.append(m)
            else:
                print(f'  [SKIP] {m} — checkpoint not found')

    print(f'\nModels to evaluate: {models}')
    all_results = {}
    for model_name in models:
        try:
            result = eval_model_position_loss(model_name, device)
            if result:
                all_results[model_name] = result
        except Exception as e:
            print(f'  [ERROR] {model_name}: {e}')
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        # Cross-model comparison table
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cmp_path = os.path.join(LOGS_DIR, f'pos_loss_comparison_{ts}.json')
        print(f'\n{"="*60}')
        print('  CROSS-MODEL COMPARISON')
        print(f'{"="*60}')
        print(f'  {"Model":<20} {"PPL":>8}  {"33-47 PPL":>10}  {"holes conf":>12}')
        print(f'  {"-"*55}')
        for name, r in all_results.items():
            blind = r['fine_buckets'].get('33-47', {}).get('ppl', float('nan'))
            print(f'  {name:<20} {r["overall_ppl"]:>8.2f}  '
                  f'{blind:>10.2f}  '
                  f'{r["holes_confirmed"]}/{r["holes_total"]:>2}')
        with open(cmp_path, 'w') as f:
            # Save summary only (not full per-position arrays) to keep small
            summary = {}
            for name, r in all_results.items():
                summary[name] = {k: v for k, v in r.items()
                                 if k not in ('per_position_loss',
                                              'per_position_ppl',
                                              'smoothed_loss_w16')}
            json.dump(summary, f, indent=2)
        print(f'\n  Comparison → {cmp_path}')


if __name__ == '__main__':
    main()
