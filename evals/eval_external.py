#!/usr/bin/env python3
"""
External NLP Benchmark Evaluator for DWARF checkpoints.

Evaluates any DWARF checkpoint on HellaSwag, PIQA, ARC-Easy, ARC-Challenge,
Winogrande, and LAMBADA using log-likelihood comparison (standard approach
for small language models).

All tasks use the same evaluation strategy:
  For multiple-choice: score each choice by log P(choice | context),
  normalized by choice length in tokens. Pick highest score.
  
  For LAMBADA: greedy accuracy on last word prediction.

Usage:
  # List available models (from eval_suite.py registry)
  .venv/bin/python3 benchmarks/eval_external.py --list

  # Evaluate a registered model
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_external.py \\
    --model condm_layer5

  # Evaluate with an explicit checkpoint path and arch
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_external.py \\
    --checkpoint checkpoints/2048_condU_27m_checkpoints/best.pt \\
    --arch condu_27m \\
    --label "condU 27M"

  # Run only specific tasks
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_external.py \\
    --model condm_layer5 --tasks hellaswag piqa arc_easy

  # Download benchmarks first if not cached:
  .venv/bin/python3 benchmarks/download_benchmarks.py

Supported --arch values for explicit checkpoint loading:
  condm_13m      D=256, H=8, FFN=1024, L=6, full_layer=5
  condm_27m      D=400, H=8, FFN=1600, L=6, full_layer=5
  condm_85m      D=640, H=8, FFN=2560, L=12, full_layer=11
  condu_13m      D=256, H=8, FFN=1024, L=6, full_layer=5, physics=condU
  condu_27m      D=400, H=8, FFN=1600, L=6, full_layer=5, physics=condU
  condu_35m      D=512, H=8, FFN=2048, L=6, full_layer=5, physics=condU
  standard_13m   D=256, H=8, FFN=1024, L=6, no DSQG
  standard_27m   D=400, H=8, FFN=1600, L=6, no DSQG
  standard_85m   D=640, H=8, FFN=2560, L=12, no DSQG

Results: benchmarks/logs/eval_external_<label>_<timestamp>.json
"""

import argparse, json, math, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
CACHE_DIR   = next((p for p in [
    os.path.join(SCRIPT_DIR,  'logs', 'benchmark_cache'),
    os.path.join(REPO_ROOT,   'logs', 'benchmark_cache'),
] if os.path.isdir(p)), os.path.join(SCRIPT_DIR, 'logs', 'benchmark_cache'))
LOGS_DIR    = os.path.join(SCRIPT_DIR, 'logs')
TOKENIZER   = next((p for p in [
    os.path.join(SCRIPT_DIR,  'results', '2048_condI_tokenizer.json'),
    os.path.join(REPO_ROOT,   'results', '2048_condI_tokenizer.json'),
    os.path.join(REPO_ROOT,   'benchmarks', 'results', '2048_condI_tokenizer.json'),
] if os.path.exists(p)), os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json'))
CKPT_ROOT   = os.path.join(REPO_ROOT, 'checkpoints')
MAX_SEQ_LEN = 2048
VOCAB_SIZE  = 32000

# Add kernel dir to path
_kernel_dir = os.path.join(REPO_ROOT, 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

# ── Explicit-checkpoint arch configs ──────────────────────────────────────────

ARCH_CONFIGS = {
    'condm_13m':   {'arch': 'condm', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6,  'full_layer': 5},
    'condm_27m':   {'arch': 'condm', 'D': 400, 'H': 8, 'FFN': 1600, 'L': 6,  'full_layer': 5},
    'condm_85m':   {'arch': 'condm', 'D': 640, 'H': 8, 'FFN': 2560, 'L': 12, 'full_layer': 11},
    'condu_13m':   {'arch': 'condu', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6,  'full_layer': 5},
    'condu_27m':   {'arch': 'condu', 'D': 400, 'H': 8, 'FFN': 1600, 'L': 6,  'full_layer': 5},
    'condu_35m':      {'arch': 'condu', 'D': 512, 'H': 8, 'FFN': 2048, 'L': 6,  'full_layer': 5},
    'condu_35m_pure': {'arch': 'condu', 'D': 512, 'H': 8, 'FFN': 2048, 'L': 6,  'full_layer': -1},
    'condu_v5_38m':   {'arch': 'condu_v5', 'D': 512, 'H': 8, 'FFN': 2048, 'L': 6, 'full_layer': 5, 'interference': 3},
    # condM layer-position ablation
    'condm_13m_L0':  {'arch': 'condm', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6,  'full_layer': 0},
    'condm_13m_L3':  {'arch': 'condm', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6,  'full_layer': 3},
    # condV: condM+condU physics (CondMTransformer from train_2048_condV.py)
    'condv_13m':     {'arch': 'condv', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6,  'full_layer': 5, 'interference': 2},
    # condW: pure DSQG+INT, no full attention (CondWTransformer from train_2048_condW.py)
    'condw_13m':     {'arch': 'condw', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6},
    'standard_13m':{'arch': 'standard', 'D': 256, 'H': 8, 'FFN': 1024, 'L': 6},
    'standard_27m':{'arch': 'standard', 'D': 400, 'H': 8, 'FFN': 1600, 'L': 6},
    'standard_85m':{'arch': 'standard', 'D': 640, 'H': 8, 'FFN': 2560, 'L': 12},
}

TASKS = ['hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande', 'lambada']

# ── Tokenizer ─────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_tokenizer():
    from tokenizers import Tokenizer
    if not os.path.exists(TOKENIZER):
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER}')
    return BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_from_registry(model_name, device):
    """Load a model from eval_suite.py's MODEL_REGISTRY."""
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from eval_suite import MODEL_REGISTRY, build_model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Run with --list to see options.')
    cfg = MODEL_REGISTRY[model_name]
    model = build_model(cfg, device)
    return model, cfg.get('label', model_name)


def load_model_from_arch(arch_name, checkpoint_path, device):
    """Load a model by arch config name + explicit checkpoint path."""
    import math as _math

    if arch_name not in ARCH_CONFIGS:
        raise ValueError(f'Unknown arch: {arch_name}. '
                         f'Options: {list(ARCH_CONFIGS.keys())}')
    cfg = ARCH_CONFIGS[arch_name]
    arch = cfg['arch']
    D, H, FFN, L = cfg['D'], cfg['H'], cfg['FFN'], cfg['L']

    if arch == 'standard':
        from eval_suite import StandardTransformer
        model = StandardTransformer(
            D=D, H=H, FFN=FFN, L=L, seq_len=MAX_SEQ_LEN,
        ).to(device)

    elif arch == 'condm':
        from eval_suite import CondMTransformer
        model = CondMTransformer(
            D=D, H=H, FFN=FFN, L=L, seq_len=MAX_SEQ_LEN,
            full_layer=cfg.get('full_layer', L - 1),
            interf=cfg.get('interference', 3),
        ).to(device)

    elif arch == 'condu':
        # Import from training script — 13M uses CondMTransformer, 27M uses CondUTransformer
        _train_name  = 'train_2048_27m_condU.py' if D == 400 else 'train_2048_condU.py'
        train_script = next((p for p in [
            os.path.join(SCRIPT_DIR,         _train_name),
            os.path.join(REPO_ROOT, 'train', _train_name),
        ] if os.path.exists(p)), os.path.join(SCRIPT_DIR, _train_name))
        import importlib.util
        spec = importlib.util.spec_from_file_location('condu_train', train_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 13M script exports CondMTransformer; 27M script exports CondUTransformer
        cls = getattr(mod, 'CondUTransformer', None) or getattr(mod, 'CondMTransformer')
        model = cls(
            vocab_size=VOCAB_SIZE, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=FFN, seq_len=MAX_SEQ_LEN,
            full_attn_layer=cfg.get('full_layer', L - 1),
            interference_interval=cfg.get('interference', 3),
        ).to(device)

    elif arch == 'condv':
        # condV: condM architecture + full condU physics stack (CondMTransformer in condV script)
        import importlib.util
        train_script = os.path.join(SCRIPT_DIR, 'train_2048_condV.py')
        spec = importlib.util.spec_from_file_location('condv_train', train_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.CondMTransformer
        model = cls(
            vocab_size=VOCAB_SIZE, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=FFN, seq_len=MAX_SEQ_LEN,
            full_attn_layer=cfg.get('full_layer', L - 1),
            interference_interval=cfg.get('interference', 3),
        ).to(device)

    elif arch == 'condu_v5':
        train_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'train'))
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        # Kernel paths are set up inside the training module via __file__
        from train_2048_condU_v5 import CondUV5Transformer
        model = CondUV5Transformer(
            vocab_size=VOCAB_SIZE, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=FFN, seq_len=MAX_SEQ_LEN,
            full_attn_layer=cfg.get('full_layer', 5),
            interference_interval=cfg.get('interference', 3),
        ).to(device)

    elif arch == 'condw':
        # condW: pure DSQG+INT stack, no full attention (CondWTransformer)
        import importlib.util
        train_script = os.path.join(SCRIPT_DIR, 'train_2048_condW.py')
        spec = importlib.util.spec_from_file_location('condw_train', train_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.CondWTransformer
        model = cls(
            vocab_size=VOCAB_SIZE, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=FFN, seq_len=MAX_SEQ_LEN,
            interference_interval=cfg.get('interference', 3),
        ).to(device)

    else:
        raise ValueError(f'Unhandled arch: {arch}')

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both raw state_dicts and checkpoint dicts with 'model_state_dict'
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # torch.compile wraps the model in _orig_mod.* — strip that prefix so
    # compiled checkpoints load into an uncompiled eval model.
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    # Drop causal_mask buffers — they're recomputed at model init and may have
    # changed shape between eval_suite versions ([2048,2048] vs [1,1,2048,2048]).
    # strict=False handles any remaining missing/unexpected keys gracefully.
    state = {k: v for k, v in state.items() if not k.endswith('causal_mask')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        learnable_missing = [k for k in missing if not k.endswith('causal_mask')]
        if learnable_missing:
            print(f'  WARNING: missing learnable keys: {learnable_missing}')
    if unexpected:
        print(f'  WARNING: unexpected keys: {unexpected}')
    model.eval()
    print(f'  Loaded {sum(p.numel() for p in model.parameters()):,} params '
          f'from {checkpoint_path}')
    return model


# ── Log-likelihood scoring ────────────────────────────────────────────────────

@torch.no_grad()
def score_completion(model, tokenizer, context: str, completion: str,
                     device: str) -> float:
    """
    Compute mean log-likelihood of `completion` tokens given `context`.
    Returns the mean (per-token) NLL — lower is better.
    Normalizes by completion length to avoid bias toward shorter options.
    """
    ctx_ids  = tokenizer.encode(context)
    comp_ids = tokenizer.encode(' ' + completion)  # space prefix for subword alignment
    if not comp_ids:
        comp_ids = tokenizer.encode(completion)
    if not comp_ids:
        return float('inf')

    full_ids = (ctx_ids + comp_ids)[-MAX_SEQ_LEN:]
    # Completion starts at position: max(0, len(ctx_ids) + len(comp_ids) - MAX_SEQ_LEN)
    comp_start = max(0, len(ctx_ids) - (MAX_SEQ_LEN - len(comp_ids)))

    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    logits = model(input_ids)  # [1, T, V]

    # Score only the completion tokens
    # The completion tokens in target_ids start at comp_start
    comp_target = target_ids[0, comp_start:]
    comp_logits = logits[0, comp_start:]

    if len(comp_target) == 0:
        return float('inf')

    loss = F.cross_entropy(comp_logits, comp_target, reduction='mean')
    return loss.item()


# ── Task evaluators ───────────────────────────────────────────────────────────

def eval_multiple_choice(model, tokenizer, examples, device,
                         task_name='', max_examples=None):
    """Evaluate multiple-choice task via log-likelihood comparison."""
    model.eval()
    correct = 0
    total   = 0
    t0      = time.time()

    if max_examples:
        examples = examples[:max_examples]

    for i, ex in enumerate(examples):
        context = ex['context']
        choices = ex['choices']
        label   = ex['label']

        scores = []
        for choice in choices:
            score = score_completion(model, tokenizer, context, choice, device)
            scores.append(score)

        pred = scores.index(min(scores))  # lowest NLL = best
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            acc     = correct / total
            eta     = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f'    [{task_name}] {i+1}/{len(examples)} — '
                  f'acc={acc:.3f} | eta {eta:.0f}s')

    acc = correct / total if total > 0 else 0.0
    return {'accuracy': acc, 'correct': correct, 'total': total}


def eval_lambada(model, tokenizer, examples, device, max_examples=None):
    """
    LAMBADA: predict last word of text from preceding context.
    Uses greedy argmax at the last context position.
    """
    model.eval()
    correct = 0
    total   = 0

    if max_examples:
        examples = examples[:max_examples]

    for ex in examples:
        context = ex['context']
        target  = ex['target']

        ctx_ids = tokenizer.encode(context)
        if not ctx_ids:
            continue

        # Greedy: take argmax at last context position
        input_ids = torch.tensor([ctx_ids[-MAX_SEQ_LEN + 1:]], dtype=torch.long, device=device)
        logits    = model(input_ids)[0, -1]  # [V]
        pred_id   = logits.argmax().item()
        pred_word = tokenizer.decode([pred_id]).strip()

        # Check if prediction matches target (case-insensitive, strip punctuation)
        target_clean = target.strip().lower().rstrip('.,;:!?')
        pred_clean   = pred_word.lower().rstrip('.,;:!?')
        if pred_clean == target_clean:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return {'accuracy': acc, 'correct': correct, 'total': total}


# ── Main evaluator ────────────────────────────────────────────────────────────

def run_evaluation(model, tokenizer, device, tasks=None, max_examples=None):
    if tasks is None:
        tasks = TASKS

    all_results = {}
    for task in tasks:
        cache_file = os.path.join(CACHE_DIR, f'{task}.json')
        if not os.path.exists(cache_file):
            print(f'  [{task}] SKIP — not cached. Run download_benchmarks.py first.')
            all_results[task] = {'accuracy': None, 'error': 'not_cached'}
            continue

        with open(cache_file) as f:
            examples = json.load(f)

        print(f'  [{task}] {len(examples):,} examples...', flush=True)
        t0 = time.time()

        if task == 'lambada':
            result = eval_lambada(model, tokenizer, examples, device, max_examples)
        else:
            result = eval_multiple_choice(model, tokenizer, examples, device,
                                          task_name=task, max_examples=max_examples)

        elapsed = time.time() - t0
        result['elapsed_s'] = elapsed
        all_results[task]   = result

        acc = result['accuracy']
        n   = result['total']
        print(f'  [{task}] accuracy = {acc:.4f} ({acc*100:.1f}%)  '
              f'[{n} examples, {elapsed:.0f}s]')

    return all_results


def print_summary(label, results):
    print('\n' + '=' * 60)
    print(f'  {label}')
    print('=' * 60)
    print(f'  {"Task":<22} {"Accuracy":>10}  {"N":>6}')
    print(f'  {"-"*22} {"-"*10}  {"-"*6}')
    accs = []
    for task in TASKS:
        if task not in results:
            continue
        r = results[task]
        if r.get('accuracy') is None:
            print(f'  {task:<22} {"N/A":>10}  {"---":>6}')
            continue
        acc = r['accuracy']
        n   = r.get('total', '?')
        print(f'  {task:<22} {acc*100:>9.1f}%  {n:>6}')
        accs.append(acc)
    if accs:
        mean = sum(accs) / len(accs)
        print(f'  {"-"*22} {"-"*10}  {"-"*6}')
        print(f'  {"mean (present tasks)":<22} {mean*100:>9.1f}%')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='DWARF external benchmark eval')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name from eval_suite.py MODEL_REGISTRY')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Explicit path to checkpoint .pt file')
    parser.add_argument('--arch', type=str, default=None,
                        help=f'Arch config for --checkpoint. '
                             f'Options: {list(ARCH_CONFIGS.keys())}')
    parser.add_argument('--label', type=str, default=None,
                        help='Label for results (default: model/arch name)')
    parser.add_argument('--tasks', nargs='+', default=None,
                        choices=TASKS,
                        help='Tasks to run (default: all)')
    parser.add_argument('--max', type=int, default=None,
                        help='Max examples per task (for quick tests)')
    parser.add_argument('--list', action='store_true',
                        help='List available models from registry and exit')
    args = parser.parse_args()

    if args.list:
        if SCRIPT_DIR not in sys.path:
            sys.path.insert(0, SCRIPT_DIR)
        try:
            from eval_suite import MODEL_REGISTRY
            print('Available models (--model):')
            for name, cfg in MODEL_REGISTRY.items():
                print(f'  {name:<28} {cfg.get("label", "")}')
        except Exception as e:
            print(f'Could not load MODEL_REGISTRY: {e}')
        print(f'\nArch configs (--arch for --checkpoint):')
        for name in ARCH_CONFIGS:
            print(f'  {name}')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 60)
    print('  DWARF External Benchmark Evaluator')
    print('=' * 60)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Tasks: {args.tasks or TASKS}')
    if args.max:
        print(f'  Max examples/task: {args.max}')

    tokenizer = load_tokenizer()
    print(f'  Tokenizer: {TOKENIZER}')

    # Load model
    if args.model:
        print(f'\nLoading from registry: {args.model}')
        model, label = load_model_from_registry(args.model, device)
        label = args.label or label
    elif args.checkpoint and args.arch:
        print(f'\nLoading checkpoint: {args.checkpoint}')
        print(f'Arch: {args.arch}')
        model = load_model_from_arch(args.arch, args.checkpoint, device)
        label = args.label or f'{args.arch} ({os.path.basename(args.checkpoint)})'
    else:
        parser.print_help()
        print('\nERROR: Provide --model OR (--checkpoint + --arch)')
        sys.exit(1)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')
    print(f'  Label: {label}\n')

    # Run evaluation
    results = run_evaluation(model, tokenizer, device,
                             tasks=args.tasks, max_examples=args.max)

    print_summary(label, results)

    # Save results
    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_label = label.replace(' ', '_').replace('/', '_')[:40]
    out_file   = os.path.join(LOGS_DIR, f'eval_external_{safe_label}_{timestamp}.json')
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({
            'label':      label,
            'n_params':   n_params,
            'timestamp':  timestamp,
            'tasks':      args.tasks or TASKS,
            'results':    results,
        }, f, indent=2)
    print(f'\n  Results -> {out_file}')


if __name__ == '__main__':
    main()
