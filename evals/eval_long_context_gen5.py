"""
eval_long_context_gen5.py — Extended context passkey test for Gen5 champion.

Tests three position encoding strategies beyond the 2048-token training window:

  1. CLAMP      — positions beyond 2048 all map to pos=2047 (naive baseline)
  2. INTERPOLATE — sequence linearly interpolated into [0, 2047]
  3. CYCLIC      — positions wrap modulo 2048 (pos % 2048)
  4. ZERO_POS    — pos_embed zeroed entirely; pure DSQG relay, no absolute position

The question: does the DSQG relay chain inherently generalize beyond training context,
or does it depend on the learned absolute pos_embed?

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -u evals/eval_long_context_gen5.py
"""

import os, sys, json, math, time
import torch
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kernels'))

from tokenizers import Tokenizer

TOKENIZER_PATH = os.path.join(ROOT, 'results', '2048_condI_tokenizer.json')

_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape',
                    'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'

# In-window: up to d=2048. Beyond: 2560→16384.
DISTANCES_IN_WINDOW = [256, 512, 1024, 1536, 2048]
DISTANCES_BEYOND    = [2560, 3072, 4096, 6144, 8192, 12288, 16384]
ALL_DISTANCES       = DISTANCES_IN_WINDOW + DISTANCES_BEYOND

TRIALS_PER_DISTANCE = 20
TRAIN_MAX_POS = 2048

GEN5_CFG = {
    'name':       'Gen5-L8-preIF (45.6M)',
    'script':     'train/train_borg_gen5_L8_preIF_bf16.py',
    'model_cls':  'AutoresearchTransformerPhysics',
    'checkpoint': 'autoresearch/checkpoints/borg_gen5_L8_preIF_best.pt',
    'D': 512, 'H': 8, 'L': 8, 'FFN': 2048, 'full_layer': 2, 'interference': 2,
}

STRATEGIES = ['clamp', 'interpolate', 'cyclic', 'zero_pos']

STRATEGY_DESC = {
    'clamp':       'CLAMP      — positions > 2047 map to 2047 (naive baseline)',
    'interpolate': 'INTERPOLATE — full sequence linearly scaled into [0, 2047]',
    'cyclic':      'CYCLIC      — positions wrap modulo 2048',
    'zero_pos':    'ZERO_POS    — pos_embed zeroed; pure relay, no absolute position',
}


def load_model_base(cfg, device):
    """Load model WITHOUT any forward patch — returns raw model."""
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
        seq_len=TRAIN_MAX_POS,
        full_attn_layer=cfg['full_layer'],
        interference_interval=cfg['interference'],
        scale_embed_init_val=0.1,
    )

    ck_path = os.path.join(ROOT, cfg['checkpoint'])
    state = torch.load(ck_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def patch_forward(model, strategy):
    """Patch model.forward with the chosen position encoding strategy."""

    if strategy == 'clamp':
        def _forward(idx):
            B, N = idx.shape
            pos = torch.arange(N, device=idx.device).unsqueeze(0).clamp(max=TRAIN_MAX_POS - 1)
            x = model.drop(model.embedding(idx) + model.pos_embed(pos))
            for block in model.blocks:
                x = block(x)
            return model.out(model.norm(x))

    elif strategy == 'interpolate':
        def _forward(idx):
            B, N = idx.shape
            if N <= 1:
                pos = torch.zeros(1, N, dtype=torch.long, device=idx.device)
            else:
                # Map [0, N-1] → [0, TRAIN_MAX_POS-1] linearly
                scale = (TRAIN_MAX_POS - 1) / (N - 1)
                pos_f = torch.arange(N, device=idx.device, dtype=torch.float) * scale
                pos = pos_f.round().long().clamp(0, TRAIN_MAX_POS - 1).unsqueeze(0)
            x = model.drop(model.embedding(idx) + model.pos_embed(pos))
            for block in model.blocks:
                x = block(x)
            return model.out(model.norm(x))

    elif strategy == 'cyclic':
        def _forward(idx):
            B, N = idx.shape
            pos = (torch.arange(N, device=idx.device) % TRAIN_MAX_POS).unsqueeze(0)
            x = model.drop(model.embedding(idx) + model.pos_embed(pos))
            for block in model.blocks:
                x = block(x)
            return model.out(model.norm(x))

    elif strategy == 'zero_pos':
        def _forward(idx):
            B, N = idx.shape
            # Zero absolute position — embed tokens only, no positional signal
            x = model.drop(model.embedding(idx))
            for block in model.blocks:
                x = block(x)
            return model.out(model.norm(x))

    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    model.forward = _forward


@torch.no_grad()
def passkey_at_distance(model, tokenizer, distance, device):
    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE).ids

    correct, skipped = 0, 0
    for i in range(TRIALS_PER_DISTANCE):
        word   = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
        others = [w for w in _PASSKEY_WORDS if w != word]
        intro  = tokenizer.encode(_INTRO_TEMPLATE.format(word=word)).ids

        filler = []
        while len(filler) < distance:
            filler.extend(filler_ids)
        filler = filler[:distance]

        full_seq = intro + filler + cue_ids
        ids = torch.tensor([full_seq], dtype=torch.long, device=device)

        try:
            logits = model(ids)[:, -1, :]
        except Exception:
            skipped += 1
            continue

        cands = [(tokenizer.encode(' ' + w).ids or tokenizer.encode(w).ids)[0]
                 for w in [word] + others[:9]]
        pred = ([word] + others[:9])[logits[0][cands].argmax().item()]
        if pred == word:
            correct += 1

    valid = TRIALS_PER_DISTANCE - skipped
    return (correct / valid if valid > 0 else float('nan')), valid, skipped


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print(f'\n{"═"*74}')
    print(f'  Gen5 L=8 preIF — Extended Context Position Strategy Comparison')
    print(f'  Training context: {TRAIN_MAX_POS} tokens')
    print(f'  Testing up to: {max(ALL_DISTANCES)} tokens ({max(ALL_DISTANCES)/TRAIN_MAX_POS:.0f}× training window)')
    print(f'  Strategies: {", ".join(STRATEGIES)}')
    print(f'{"═"*74}\n')

    for desc in STRATEGY_DESC.values():
        print(f'  {desc}')
    print()

    tok = Tokenizer.from_file(TOKENIZER_PATH)
    all_results = {}

    for strategy in STRATEGIES:
        print(f'\n{"─"*74}')
        print(f'  Strategy: {strategy.upper()}')
        print(f'  {STRATEGY_DESC[strategy]}')
        print(f'{"─"*74}')

        model = load_model_base(GEN5_CFG, device)
        patch_forward(model, strategy)

        results = {}
        print(f'\n  {"Distance":>10}  {"Seq len":>8}  {"Accuracy":>10}  {"Status"}')
        print(f'  {"─"*55}')

        for d in ALL_DISTANCES:
            in_window = (d <= TRAIN_MAX_POS)
            status = 'in-window' if in_window else 'OOD ⚡'
            t0 = time.time()
            acc, valid, skipped = passkey_at_distance(model, tok, d, device)
            elapsed = time.time() - t0
            acc_str = f'{acc:.1%}' if not math.isnan(acc) else 'ERROR'
            skip_str = f' (skip={skipped})' if skipped else ''
            print(f'  {d:>10}  {d+10:>8}  {acc_str:>10}  {status}{skip_str}  [{elapsed:.1f}s]')
            results[str(d)] = {'accuracy': acc, 'valid': valid,
                               'skipped': skipped, 'in_window': in_window}

        all_results[strategy] = results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f'\n\n{"═"*90}')
    print(f'  SUMMARY — Gen5 L=8 preIF: passkey accuracy by position strategy')
    print(f'  (║ marks training window boundary)')
    print(f'{"═"*90}')

    header = f'  {"Strategy":<14}'
    for d in ALL_DISTANCES:
        m = '║' if d == TRAIN_MAX_POS else ' '
        header += f'{m}{d:>7}'
    print(header)
    print(f'  {"─"*88}')

    for strategy in STRATEGIES:
        row = f'  {strategy.upper():<14}'
        for d in ALL_DISTANCES:
            m = '║' if d == TRAIN_MAX_POS else ' '
            r = all_results[strategy].get(str(d), {})
            acc = r.get('accuracy', float('nan'))
            cell = f'{acc:.0%}' if not math.isnan(acc) else 'ERR'
            row += f'{m}{cell:>7}'
        print(row)

    print(f'\n  Legend: ║ = training window boundary | OOD = beyond training context')

    out_path = os.path.join(ROOT, 'evals', 'logs', 'eval_long_context_gen5_strategies.json')
    with open(out_path, 'w') as f:
        json.dump({'model': GEN5_CFG['name'], 'strategies': all_results}, f, indent=2)
    print(f'\n  Saved → {out_path}')
    import subprocess
    subprocess.run(['openclaw', 'system', 'event', '--text',
                    'Done: Gen5 long-context position strategy comparison complete',
                    '--mode', 'now'], capture_output=True)


if __name__ == '__main__':
    main()
