"""
eval_long_context.py — Extended context passkey test for J-series models.

Tests retrieval BEYOND the 2048-token training context window, in increasing
intervals up to 16384 tokens. Uses existing checkpoints with no fine-tuning.

This is a "shouldn't work but let's see" experiment — DWARF's relay chain
theoretically extends beyond training context, but we've never tested it.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -u evals/eval_long_context.py
"""

import os, sys, json, math, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kernels'))

from tokenizers import Tokenizer

TOKENIZER_PATH = os.path.join(ROOT, 'results', '2048_condI_tokenizer.json')

# ── Passkey config (matches training scripts) ─────────────────────────────────
_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape',
                    'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'

# ── Test distances — training window + beyond ─────────────────────────────────
DISTANCES_IN_WINDOW  = [64, 128, 256, 512, 1024, 1536]           # baseline (trained)
DISTANCES_BEYOND     = [2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384]  # OOD
ALL_DISTANCES        = DISTANCES_IN_WINDOW + DISTANCES_BEYOND

TRIALS_PER_DISTANCE  = 20  # enough for 5% granularity

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = [
    {
        'name':       'J20D-V10-L8 (47.1M)',
        'arch':       'j20d_v10_L8',
        'script':     'train/train_j20d_v10_L8_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'checkpoint': 'autoresearch/checkpoints/99437df_j20d_v10_L8_best.pt',
        'D': 512, 'H': 8, 'L': 8, 'FFN': 2048, 'full_layer': 7, 'interference': 2,
    },
    {
        'name':       'J26D-int2 (39.5M)',
        'arch':       'j26d',
        'script':     'train/train_j26d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'checkpoint': 'autoresearch/checkpoints/99437df_j26d_int2_physics_best.pt',
        'D': 512, 'H': 8, 'L': 6, 'FFN': 2048, 'full_layer': 5, 'interference': 2,
    },
    {
        'name':       'Curve 27M (31.6M)',
        'arch':       'curve_27m',
        'script':     'train/train_curve_27m_bf16.py',
        'model_cls':  'CurveTransformer',
        'checkpoint': 'checkpoints/curve_27m_best.pt',
        'D': 512, 'H': 8, 'L': 6, 'FFN': 768, 'full_layer': 5, 'interference': 2,
    },
    {
        'name':       'j24d-int2-physics (39.5M)',
        'arch':       'j24d',
        'script':     'train/train_j24d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'checkpoint': '/tmp/dwarf-j17d/autoresearch/checkpoints/df0d435_j24d_int2_physics_best.pt',
        'D': 512, 'H': 8, 'L': 6, 'FFN': 2048, 'full_layer': 5, 'interference': 2,
    },
]

def load_model(cfg, device):
    import importlib.util
    script = os.path.join(ROOT, cfg['script'])
    spec = importlib.util.spec_from_file_location('train_mod', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls = getattr(mod, cfg['model_cls'])
    model = cls(
        vocab_size=32000,
        embedding_dim=cfg['D'],
        num_layers=cfg['L'],
        num_heads=cfg['H'],
        ffn_dim=cfg['FFN'],
        seq_len=2048,            # trained at 2048; pos_embed fixed at this
        full_attn_layer=cfg['full_layer'],
        interference_interval=cfg['interference'],
        scale_embed_init_val=0.1,
    )

    # Patch forward to clamp position indices to training max (2048).
    # This is the simplest possible "context extension" — positions beyond
    # training window reuse the last trained position embedding.
    # Not interpolation, just clamping — the most naive possible extension.
    _original_forward = model.forward
    _max_pos = 2048

    def _forward_clamped(idx):
        B, N = idx.shape
        # Build positions, clamped to max trained pos
        pos = torch.arange(N, device=idx.device).unsqueeze(0)
        pos = pos.clamp(max=_max_pos - 1)
        # Manually do what the original forward does but with clamped pos
        x = model.drop(model.embedding(idx) + model.pos_embed(pos))
        for block in model.blocks:
            x = block(x)
        return model.out(model.norm(x))

    model.forward = _forward_clamped

    ck_path = os.path.join(ROOT, cfg['checkpoint'])
    state = torch.load(ck_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    # strip _orig_mod prefix if present
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def passkey_at_distance(model, tokenizer, distance, device, n_trials=TRIALS_PER_DISTANCE):
    """Run passkey eval at a given filler distance. Handles sequences > 2048."""
    filler_ids  = tokenizer.encode(_FILLER_SENTENCE).ids
    cue_ids     = tokenizer.encode(_RETRIEVAL_CUE).ids

    correct, skipped = 0, 0
    for i in range(n_trials):
        word    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
        others  = [w for w in _PASSKEY_WORDS if w != word]
        intro   = tokenizer.encode(_INTRO_TEMPLATE.format(word=word)).ids

        # Build filler of exactly `distance` tokens
        filler = []
        while len(filler) < distance:
            filler.extend(filler_ids)
        filler = filler[:distance]

        full_seq = intro + filler + cue_ids

        # No truncation — let it be however long it is
        ids = torch.tensor([full_seq], dtype=torch.long, device=device)

        try:
            logits = model(ids)[:, -1, :]
        except Exception as e:
            # OOM or other error at very long sequences
            skipped += 1
            continue

        cands = [(tokenizer.encode(' ' + w).ids or tokenizer.encode(w).ids)[0]
                 for w in [word] + others[:9]]
        pred_word = ([word] + others[:9])[logits[0][cands].argmax().item()]
        if pred_word == word:
            correct += 1

    valid = n_trials - skipped
    acc = correct / valid if valid > 0 else float('nan')
    return acc, valid, skipped


def run_eval(cfg, device):
    print(f'\n{"─"*70}')
    print(f'  {cfg["name"]}')
    print(f'  Checkpoint: {cfg["checkpoint"]}')
    print(f'{"─"*70}')

    model = load_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params/1e6:.1f}M  |  seq_len (training): 2048')

    tok = Tokenizer.from_file(TOKENIZER_PATH)

    results = {}
    print(f'\n  {"Distance":>10}  {"Seq len":>8}  {"Accuracy":>10}  {"Status"}')
    print(f'  {"─"*50}')

    for d in ALL_DISTANCES:
        seq_len = d + 10  # approx (intro + cue ≈ 10 tokens)
        in_window = d <= 2048
        status = 'in-window' if in_window else 'OOD ⚡'

        t0 = time.time()
        acc, valid, skipped = passkey_at_distance(model, tok, d, device)
        elapsed = time.time() - t0

        acc_str = f'{acc:.1%}' if not math.isnan(acc) else 'ERROR'
        skip_str = f' (skip={skipped})' if skipped > 0 else ''
        print(f'  {d:>10}  {seq_len:>8}  {acc_str:>10}  {status}{skip_str}  [{elapsed:.1f}s]')

        results[d] = {'accuracy': acc, 'valid': valid, 'skipped': skipped,
                      'in_window': in_window}

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print(f'\n{"═"*70}')
    print(f'  DWARF EXTENDED CONTEXT TEST — Beyond Training Window')
    print(f'  Training context: 2048 tokens')
    print(f'  Testing up to: {max(ALL_DISTANCES)} tokens ({max(ALL_DISTANCES)/2048:.1f}× training window)')
    print(f'  Models: {len(MODELS)}  |  Trials/distance: {TRIALS_PER_DISTANCE}')
    print(f'{"═"*70}')

    all_results = {}
    for cfg in MODELS:
        ck = os.path.join(ROOT, cfg['checkpoint'])
        if not os.path.exists(ck):
            print(f'\n  SKIP {cfg["name"]} — checkpoint not found: {ck}')
            continue
        model_results = run_eval(cfg, device)
        all_results[cfg['name']] = {
            'config': cfg,
            'results': model_results,
        }

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f'\n\n{"═"*90}')
    print(f'  SUMMARY — Passkey accuracy by model and distance')
    print(f'  (╟ marks training window boundary at d=2048)')
    print(f'{"═"*90}')

    header = f'  {"Model":<28}'
    for d in ALL_DISTANCES:
        marker = '║' if d == 2048 else ' '
        header += f'{marker}{d:>7}'
    print(header)
    print(f'  {"─"*88}')

    for name, data in all_results.items():
        row = f'  {name:<28}'
        for d in ALL_DISTANCES:
            marker = '║' if d == 2048 else ' '
            r = data['results'].get(d, {})
            acc = r.get('accuracy', float('nan'))
            if math.isnan(acc):
                cell = ' ERROR'
            else:
                cell = f'{acc:.0%}'
            row += f'{marker}{cell:>7}'
        print(row)

    print(f'\n  Legend: ║ = training window boundary | OOD = out-of-distribution')

    # Save results
    out_path = os.path.join(ROOT, 'evals', 'logs', 'eval_long_context_results.json')
    save_data = {k: {**v, 'config': {kk: vv for kk, vv in v['config'].items()
                                      if kk != 'script'}}
                 for k, v in all_results.items()}
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\n  Saved → {out_path}')


if __name__ == '__main__':
    main()
