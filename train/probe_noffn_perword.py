"""
probe_noffn_perword.py — Per-word passkey breakdown for no_ffn_34m ep2/ep3

Tests Brico's FA-induction hypothesis:
  - FA induction mechanism → roughly uniform ~20% per word at d≤256
  - Memorization artifact → concentrated in 2 specific words at 100%, others 0%

Loads the no_ffn_34m_best.pt checkpoint and runs the passkey eval
broken down by individual passkey word, across distances.
"""

import contextlib, json, os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

# Need to run from DWARF repo root for module imports
REPO_ROOT = '/home/dlewis3/Desktop/AI/DWARF'
sys.path.insert(0, f'{REPO_ROOT}/train')
sys.path.insert(0, f'{REPO_ROOT}')

# Import the model class
from train_no_ffn_34m_4090_bf16 import (
    DWARFNoFFN, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM,
    FULL_ATTN_LAYER, OFFSETS, EMA_INIT, SCALE_EMBED_INIT_VAL,
    SCALE_EMBED_LR_MULT, _PASSKEY_WORDS, _FILLER_SENTENCE,
    _RETRIEVAL_CUE, _INTRO_TEMPLATE, PASSKEY_DISTANCES, TOKENIZER_PATH,
    CHECKPOINT_DIR, MAX_SEQ_LEN
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = f'{CHECKPOINT_DIR}/no_ffn_34m_best.pt'  # ep3 best
N_TRIALS_PER_WORD = 5  # trials per word per distance (5 × 10 words = 50 total, matching original)

def load_model_and_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f'{REPO_ROOT}/{TOKENIZER_PATH}')
    vocab_size = tok.get_vocab_size()

    model = DWARFNoFFN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    )
    print(f'Loading checkpoint: {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    # Handle checkpoint formats
    if 'model' in ckpt:
        state = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    print(f'Model loaded. Device: {DEVICE}')
    return model, tok


def per_word_passkey(model, tokenizer):
    """Run passkey eval, returning per-word accuracy at each distance."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE).ids

    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word).ids or tokenizer.encode(word).ids
        word_token_ids[word] = encoded[0]

    # results[word][d] = (correct, total)
    results = {w: {} for w in _PASSKEY_WORDS}

    for d in PASSKEY_DISTANCES:
        print(f'  d={d}...', end='', flush=True)
        for target_word in _PASSKEY_WORDS:
            others = [w for w in _PASSKEY_WORDS if w != target_word]
            cand_ids = [word_token_ids[target_word]] + [word_token_ids[o] for o in others[:9]]
            correct = 0
            total = N_TRIALS_PER_WORD

            seqs, last_positions = [], []
            for _ in range(N_TRIALS_PER_WORD):
                intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target_word)).ids
                filler = []
                while len(filler) < d:
                    filler.extend(filler_ids)
                full_seq = intro_ids + filler[:d] + cue_ids
                if len(full_seq) >= MAX_SEQ_LEN:
                    # Skip too-long sequences
                    total -= 1
                    continue
                pad = [0] * (MAX_SEQ_LEN - len(full_seq))
                seqs.append(full_seq + pad)
                last_positions.append(len(full_seq) - 1)

            if not seqs:
                results[target_word][d] = (0, 0)
                continue

            ids_t   = torch.tensor(seqs,           dtype=torch.long,  device=DEVICE)
            pos_t   = torch.tensor(last_positions, dtype=torch.long,  device=DEVICE)
            cand_t  = torch.tensor([cand_ids],     dtype=torch.long,  device=DEVICE).expand(len(seqs), -1)

            with torch.no_grad():
                logits = model(ids_t)
            row = torch.arange(ids_t.size(0), device=DEVICE)
            next_logits = logits[row, pos_t, :]
            cand_logits = torch.gather(next_logits, 1, cand_t)
            correct = (cand_logits.argmax(dim=1) == 0).sum().item()
            results[target_word][d] = (correct, len(seqs))

        print(f' done')

    return results


def print_results(results):
    print('\n' + '='*70)
    print('PER-WORD PASSKEY BREAKDOWN — no_ffn_34m best checkpoint')
    print('='*70)
    print(f'\n{"Word":<10}', end='')
    for d in PASSKEY_DISTANCES:
        print(f'  d={d:<6}', end='')
    print()
    print('-' * 70)

    # Per-word rows
    for word in _PASSKEY_WORDS:
        print(f'{word:<10}', end='')
        for d in PASSKEY_DISTANCES:
            c, t = results[word].get(d, (0, 0))
            if t > 0:
                pct = int(100 * c / t)
                print(f'  {pct:>6}%', end='')
            else:
                print(f'  {"N/A":>6}', end='')
        print()

    print('-' * 70)
    # Totals per distance
    print(f'{"TOTAL":<10}', end='')
    for d in PASSKEY_DISTANCES:
        total_c = sum(results[w].get(d, (0,0))[0] for w in _PASSKEY_WORDS)
        total_t = sum(results[w].get(d, (0,0))[1] for w in _PASSKEY_WORDS)
        if total_t > 0:
            pct = int(100 * total_c / total_t)
            print(f'  {pct:>6}%', end='')
        else:
            print(f'  {"N/A":>6}', end='')
    print()

    # Diagnostic: which words contribute most to the d≤256 20%?
    print('\n--- Diagnosis at d=256 (small-context regime) ---')
    for word in _PASSKEY_WORDS:
        c, t = results[word].get(256, (0, 0))
        if t > 0:
            pct = int(100 * c / t)
            marker = '  ← HIGH' if pct >= 60 else ('  ← MED' if pct >= 20 else '')
            print(f'  {word:<10}: {c}/{t} = {pct}%{marker}')

    print('\n--- Diagnosis at d=512 (cliff zone) ---')
    for word in _PASSKEY_WORDS:
        c, t = results[word].get(512, (0, 0))
        if t > 0:
            pct = int(100 * c / t)
            marker = '  ← HIGH' if pct >= 60 else ('  ← MED' if pct >= 20 else '')
            print(f'  {word:<10}: {c}/{t} = {pct}%{marker}')

    print('\n--- Interpretation ---')
    # Check if 20% is concentrated or uniform at d=256
    at_256 = [(w, results[w].get(256, (0,0))) for w in _PASSKEY_WORDS]
    nonzero_at_256 = [(w, c/t) for w, (c, t) in at_256 if t > 0 and c > 0]
    if not nonzero_at_256:
        print('No correct trials at d=256.')
    else:
        accs = [acc for _, acc in nonzero_at_256]
        max_acc = max(accs)
        min_acc = min(accs)
        if max_acc > 0.8:
            print(f'  CONCENTRATED: {len(nonzero_at_256)} word(s) with high accuracy (max={max_acc:.0%})')
            print(f'  → Consistent with MEMORIZATION ARTIFACT hypothesis')
        elif max_acc - min_acc < 0.2:
            print(f'  UNIFORM: all words ~{sum(accs)/len(accs):.0%} accuracy (range {min_acc:.0%}-{max_acc:.0%})')
            print(f'  → Consistent with FA INDUCTION MECHANISM hypothesis')
        else:
            print(f'  MIXED: range {min_acc:.0%}-{max_acc:.0%}')
            print(f'  → Neither hypothesis cleanly fits; need more data')


if __name__ == '__main__':
    os.chdir(REPO_ROOT)
    print(f'Working dir: {os.getcwd()}')
    model, tok = load_model_and_tokenizer()
    print(f'\nRunning per-word passkey breakdown...')
    print(f'  {N_TRIALS_PER_WORD} trials × {len(_PASSKEY_WORDS)} words × {len(PASSKEY_DISTANCES)} distances')
    results = per_word_passkey(model, tok)
    print_results(results)

    # Save raw results
    out_path = f'{REPO_ROOT}/logs/probe_noffn_perword.json'
    with open(out_path, 'w') as f:
        json.dump({
            word: {str(d): list(v) for d, v in dist_results.items()}
            for word, dist_results in results.items()
        }, f, indent=2)
    print(f'\nSaved results to: {out_path}')
