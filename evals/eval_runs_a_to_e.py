#!/usr/bin/env python3
"""
External benchmark evaluation for Run A through Run E (condX N=4096 + N=2048 series).

Evaluates each checkpoint on HellaSwag, PIQA, ARC-Easy, ARC-Challenge,
Winogrande, and LAMBADA using the same log-likelihood scoring as eval_external.py.

Runs sequentially — intended for CUDA_VISIBLE_DEVICES=1 (RTX 3090) while
Run F trains on CUDA_VISIBLE_DEVICES=0 (RTX 4090).

Runs evaluated:
  Run A — N=4096, V4 kernel, near-zero init         (58.716 PPL, 0%  passkey)
  Run B — N=4096, V4+dispersive, near-zero init     (58.557 PPL, 10% passkey)
  Run C — N=2048, V3-interleaved J=40, linspace     (52.857 PPL, 45% passkey)
  Run D — N=2048, V3-rund J=53, near-zero           (55.488 PPL, 35% passkey)
  Run E — N=2048, V3-rune J=79, near-zero           (52.442 PPL, 25% passkey)

Also includes condU N=4096 baseline (from 2048_condU_13m_4096_checkpoints).

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_runs_a_to_e.py
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_runs_a_to_e.py --runs c d e
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_runs_a_to_e.py --runs a b

Results: benchmarks/logs/eval_external_runs_a_to_e_<timestamp>.json
         (per-run files also written to benchmarks/logs/eval_external_run<X>_<timestamp>.json)
"""

import argparse, importlib.util, json, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR  = os.path.join(REPO_ROOT, 'train')
CKPT_ROOT  = os.path.join(REPO_ROOT, 'checkpoints')
KERNEL_DIR = os.path.join(REPO_ROOT, 'kernels')
LOGS_DIR   = os.path.join(SCRIPT_DIR, 'logs')
TOKENIZER  = os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json')
os.makedirs(LOGS_DIR, exist_ok=True)

# Add kernel dir to path so training scripts can import their kernels
if KERNEL_DIR not in sys.path:
    sys.path.insert(0, KERNEL_DIR)
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

VOCAB_SIZE = 32000

# ── Run registry ──────────────────────────────────────────────────────────────

RUNS = {
    'condu_4096': {
        'label':      'condU 13M N=4096 (baseline)',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_4096_anneal.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condU_13m_4096_checkpoints', 'best.pt'),
        'seq_len':    4096,
        'ppl':        58.617,
        'passkey':    '0%',
        'note':       'condU baseline for condX N=4096 series',
    },
    'a': {
        'label':      'Run A — N=4096, near-zero init, no dispersive',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_4096_anneal.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '4096_dsqg_hybrid_13m_anneal', 'best.pt'),
        'seq_len':    4096,
        'ppl':        58.716,
        'passkey':    '0%',
        'note':       'near-zero init alone insufficient at N=4096',
    },
    'b': {
        'label':      'Run B — N=4096, dispersive kernel, near-zero init',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_4096_disp_anneal.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '4096_dsqg_hybrid_13m_disp_anneal', 'best.pt'),
        'seq_len':    4096,
        'ppl':        58.557,
        'passkey':    '10%',
        'note':       'dispersive mechanism: propagating/evanescent head split',
    },
    'c': {
        'label':      'Run C — N=2048, interleaved J=40, linspace init',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_2048_interleaved.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '2048_dsqg_hybrid_13m_interleaved', 'best.pt'),
        'seq_len':    2048,
        'ppl':        52.857,
        'passkey':    '45%',
        'note':       'd=32 passkey wall broken; d=1024 40%',
    },
    'd': {
        'label':      'Run D — N=2048, octave blocks J=53, near-zero init',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_2048_rund.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '2048_dsqg_hybrid_13m_rund', 'best.pt'),
        'seq_len':    2048,
        'ppl':        55.488,
        'passkey':    '35%',
        'note':       'd=64/256/1024 at 60%; sparse short-range hurt PPL',
    },
    'e': {
        'label':      'Run E — N=2048, dense 1-39 J=79, near-zero init',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_2048_rune.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '2048_dsqg_hybrid_13m_rune', 'best.pt'),
        'seq_len':    2048,
        'ppl':        52.442,
        'passkey':    '25%',
        'note':       'best PPL of series; distraction-by-density killed passkey',
    },
    'f': {
        'label':      'Run F — N=2048, dense 1-15 + gap + octave J=63, near-zero init',
        'script':     os.path.join(TRAIN_DIR, 'dsqg_hybrid_13m_2048_runf.py'),
        'checkpoint': os.path.join(CKPT_ROOT, '2048_dsqg_hybrid_13m_runf', 'best.pt'),
        'seq_len':    2048,
        'ppl':        52.729,
        'passkey':    '21.7%',
        'note':       'cliff-edge design; broad coverage 10/12 distances; d=32:60%',
    },
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_tokenizer():
    from tokenizers import Tokenizer
    return BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER))


# ── Model loader ──────────────────────────────────────────────────────────────

def load_run(run_key, device):
    """Load a run's model from its training script + checkpoint."""
    cfg = RUNS[run_key]
    script_path = cfg['script']
    ckpt_path   = cfg['checkpoint']
    seq_len     = cfg['seq_len']

    print(f'\n  Loading {cfg["label"]}')
    print(f'  Script:     {os.path.relpath(script_path, REPO_ROOT)}')
    print(f'  Checkpoint: {os.path.relpath(ckpt_path, REPO_ROOT)}')

    # Import the training script module — this brings in the kernel and CondUTransformer
    mod_name = f'run_{run_key}_train'
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    # Prevent the __main__ block from running
    mod.__spec__.name = mod_name
    spec.loader.exec_module(mod)

    cls = getattr(mod, 'CondUTransformer', None)
    if cls is None:
        raise RuntimeError(f'CondUTransformer not found in {script_path}')

    model = cls(
        vocab_size=VOCAB_SIZE,
        embedding_dim=mod.EMBEDDING_DIM,
        num_layers=mod.NUM_LAYERS,
        num_heads=mod.NUM_HEADS,
        ffn_dim=mod.FFN_DIM,
        seq_len=seq_len,
        full_attn_layer=mod.FULL_ATTN_LAYER,
        interference_interval=mod.INTERFERENCE,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # Strip torch.compile prefix
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.endswith('causal_mask')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    learnable_missing = [k for k in missing if not k.endswith('causal_mask')]
    if learnable_missing:
        print(f'  WARNING: missing learnable keys: {learnable_missing}')
    if unexpected:
        print(f'  WARNING: unexpected keys: {unexpected}')

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,} | seq_len={seq_len} | PPL={cfg["ppl"]} | passkey={cfg["passkey"]}')
    return model


# ── Benchmark scoring (mirrors eval_external.py) ──────────────────────────────

MAX_EVAL_SEQLEN = 2048  # clamp for eval; Run A/B have 4096 pos_embed but
                         # benchmark examples are all <2048 tokens anyway

@torch.no_grad()
def score_completion(model, tokenizer, context, completion, device):
    ctx_ids  = tokenizer.encode(context)
    comp_ids = tokenizer.encode(' ' + completion)
    if not comp_ids:
        comp_ids = tokenizer.encode(completion)
    if not comp_ids:
        return float('inf')
    full_ids   = (ctx_ids + comp_ids)[-MAX_EVAL_SEQLEN:]
    comp_start = max(0, len(ctx_ids) - (MAX_EVAL_SEQLEN - len(comp_ids)))
    input_ids  = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]],  dtype=torch.long, device=device)
    with torch.amp.autocast('cuda'):
        logits = model(input_ids)
    comp_target = target_ids[0, comp_start:]
    comp_logits = logits[0, comp_start:]
    if len(comp_target) == 0:
        return float('inf')
    return F.cross_entropy(comp_logits, comp_target, reduction='mean').item()


@torch.no_grad()
def eval_multiple_choice(model, tokenizer, examples, device, task_name='', max_examples=None):
    correct = 0; total = 0; t0 = time.time()
    if max_examples:
        examples = examples[:max_examples]
    for i, ex in enumerate(examples):
        scores = [score_completion(model, tokenizer, ex['context'], c, device)
                  for c in ex['choices']]
        if scores.index(min(scores)) == ex['label']:
            correct += 1
        total += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f'    [{task_name}] {i+1}/{len(examples)} acc={correct/total:.3f} eta={eta:.0f}s')
    return {'accuracy': correct / total if total else 0.0, 'correct': correct, 'total': total}


@torch.no_grad()
def eval_lambada(model, tokenizer, examples, device, max_examples=None):
    correct = 0; total = 0
    if max_examples:
        examples = examples[:max_examples]
    for ex in examples:
        ctx_ids = tokenizer.encode(ex['context'])
        if not ctx_ids:
            continue
        input_ids = torch.tensor([ctx_ids[-(MAX_EVAL_SEQLEN - 1):]], dtype=torch.long, device=device)
        with torch.amp.autocast('cuda'):
            logits = model(input_ids)[0, -1]
        pred_word = tokenizer.decode([logits.argmax().item()]).strip()
        target_clean = ex['target'].strip().lower().rstrip('.,;:!?')
        pred_clean   = pred_word.lower().rstrip('.,;:!?')
        if pred_clean == target_clean:
            correct += 1
        total += 1
    return {'accuracy': correct / total if total else 0.0, 'correct': correct, 'total': total}


# ── Benchmark loader (re-use eval_external.py's load_benchmarks) ──────────────

CACHE_DIR = os.path.join(SCRIPT_DIR, 'logs', 'benchmark_cache')

def load_benchmarks(tasks):
    """Load cached benchmark examples from CACHE_DIR (written by download_benchmarks.py)."""
    benchmarks = {}
    for task in tasks:
        cache_file = os.path.join(CACHE_DIR, f'{task}.json')
        if not os.path.exists(cache_file):
            print(f'  [{task}] SKIP — not cached. Run: .venv/bin/python3 benchmarks/download_benchmarks.py')
            continue
        with open(cache_file) as f:
            benchmarks[task] = json.load(f)
    return benchmarks


# ── Per-run evaluation ────────────────────────────────────────────────────────

TASKS = ['hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande', 'lambada']

def evaluate_run(run_key, model, tokenizer, benchmarks, device):
    cfg = RUNS[run_key]
    print(f'\n{"="*60}')
    print(f'  Evaluating: {cfg["label"]}')
    print(f'{"="*60}')

    results = {
        'run':       run_key,
        'label':     cfg['label'],
        'ppl':       cfg['ppl'],
        'passkey':   cfg['passkey'],
        'note':      cfg['note'],
        'tasks':     {},
    }
    task_accs = []

    for task in TASKS:
        if task not in benchmarks:
            print(f'  [{task}] skipped (not loaded)')
            continue
        t0 = time.time()
        examples = benchmarks[task]
        print(f'  [{task}] n={len(examples)} ...', end='', flush=True)

        if task == 'lambada':
            r = eval_lambada(model, tokenizer, examples, device)
        else:
            r = eval_multiple_choice(model, tokenizer, examples, device, task_name=task)

        elapsed = time.time() - t0
        acc = r['accuracy']
        task_accs.append(acc)
        results['tasks'][task] = {**r, 'elapsed_s': elapsed}
        print(f' acc={acc*100:.1f}%  ({r["correct"]}/{r["total"]})  {elapsed:.0f}s')

    mean_acc = sum(task_accs) / len(task_accs) if task_accs else 0.0
    results['mean_accuracy'] = mean_acc
    print(f'  Mean: {mean_acc*100:.1f}%')
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark Runs A through E')
    parser.add_argument('--runs', nargs='+', default=list(RUNS.keys()),
                        help='Which runs to evaluate (default: all). '
                             'Options: condu_4096 a b c d e')
    parser.add_argument('--tasks', nargs='+', default=TASKS,
                        help='Which tasks (default: all 6)')
    args = parser.parse_args()

    # Validate
    invalid = [r for r in args.runs if r not in RUNS]
    if invalid:
        print(f'ERROR: unknown run(s): {invalid}. Available: {list(RUNS.keys())}')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    print('=' * 60)
    print('  DWARF External Benchmarks — Runs A through E')
    print(f'  Device: {device}  Tasks: {args.tasks}')
    print(f'  Runs:   {args.runs}')
    print('=' * 60)

    print('\nLoading tokenizer...')
    tokenizer = load_tokenizer()

    print(f'\nLoading benchmarks: {args.tasks}')
    benchmarks = load_benchmarks(args.tasks)
    for task, examples in benchmarks.items():
        print(f'  {task}: {len(examples):,} examples')

    all_results = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for run_key in args.runs:
        # Load model
        try:
            model = load_run(run_key, device)
        except Exception as e:
            print(f'  ERROR loading run {run_key}: {e}')
            import traceback; traceback.print_exc()
            continue

        # Evaluate
        result = evaluate_run(run_key, model, tokenizer, benchmarks, device)
        all_results.append(result)

        # Save per-run result immediately (so partial runs aren't lost)
        per_run_path = os.path.join(LOGS_DIR, f'eval_external_run{run_key}_{timestamp}.json')
        with open(per_run_path, 'w') as fp:
            json.dump(result, fp, indent=2)
        print(f'  Saved: {os.path.relpath(per_run_path, REPO_ROOT)}')

        # Free GPU memory between runs
        del model
        torch.cuda.empty_cache()

    # Summary table
    print('\n' + '=' * 70)
    print('  SUMMARY — External Benchmarks')
    print('=' * 70)
    header = f'  {"Run":<20} {"PPL":>7} {"PK":>6}  {"HS":>6} {"PQ":>6} {"AE":>6} {"AC":>6} {"WG":>6} {"LB":>6}  {"Mean":>6}'
    print(header)
    print('  ' + '-' * 66)
    for r in all_results:
        t = r['tasks']
        hs = t.get('hellaswag',     {}).get('accuracy', float('nan'))
        pq = t.get('piqa',          {}).get('accuracy', float('nan'))
        ae = t.get('arc_easy',      {}).get('accuracy', float('nan'))
        ac = t.get('arc_challenge', {}).get('accuracy', float('nan'))
        wg = t.get('winogrande',    {}).get('accuracy', float('nan'))
        lb = t.get('lambada',       {}).get('accuracy', float('nan'))
        mn = r['mean_accuracy']
        label = r['label'][:20]
        print(f'  {label:<20} {r["ppl"]:>7.3f} {r["passkey"]:>6}  '
              f'{hs*100:>5.1f}% {pq*100:>5.1f}% {ae*100:>5.1f}% '
              f'{ac*100:>5.1f}% {wg*100:>5.1f}% {lb*100:>5.1f}%  '
              f'{mn*100:>5.1f}%')

    # Save combined results
    combined_path = os.path.join(LOGS_DIR, f'eval_external_runs_a_to_e_{timestamp}.json')
    with open(combined_path, 'w') as fp:
        json.dump({
            'timestamp': timestamp,
            'runs':      args.runs,
            'tasks':     args.tasks,
            'results':   all_results,
        }, fp, indent=2)
    print(f'\n  Combined results -> {os.path.relpath(combined_path, REPO_ROOT)}')


if __name__ == '__main__':
    main()
