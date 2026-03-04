"""
DWARF condK — Passkey Retrieval Test

Tests whether DWARF's wave field preserves specific content at varying distances,
and specifically whether dyadic offsets are retrieved better than non-dyadic ones.

HYPOTHESIS (from 8:38 AM cron session, Feb 25):
  - DWARF has two memory systems: local KV (exact, dyadic offsets up to 3,072 tokens)
    and global wave field (compressed, propagated, unbounded)
  - condJ/K added Q·K content-dependent retrieval — the version most likely to do
    passkey retrieval at all
  - D4 DWT covers ~35% of attention mass (specific dyadic positions only)
  - Positions at non-dyadic offsets must rely on field interpolation, not direct taps

TESTS:
  1. Baseline (D=0): passkey immediately before retrieval cue — should work
  2. Varying distance: D = 50, 100, 250, 500, 1000, 1500, 1800 tokens of filler
  3. Dyadic vs non-dyadic: specifically test at D4 tap offsets vs gaps between them
  4. Random baseline: unrelated words — model probability without any passkey signal

METRIC:
  - Log-probability of correct passkey token(s) at retrieval position
  - Rank of correct answer among all vocabulary tokens
  - Accuracy: correct passkey is more probable than any of N=9 distractors
  - PPL-equivalent score at retrieval position

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/passkey_test.py

Results logged to benchmarks/passkey_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn.functional as F

# Import the condK model definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We reuse train_2048_condK_pooling.py's model since condK+RP uses condK architecture
# but we load the *condK* (no pooling) checkpoint for the pure condK model.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '2048_condK_checkpoints')
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, '2048_condI_tokenizer.json')
RESULTS_PATH   = os.path.join(SCRIPT_DIR, '2048_passkey_results.json')

MAX_SEQ_LEN = 2048

# ─── Passkey vocabulary: 20 simple words that tokenize to a single BPE token ──
# These are chosen to be clearly distinct from each other and from the filler text.
PASSKEY_WORDS = [
    'apple', 'tiger', 'robot', 'ocean', 'piano',
    'eagle', 'storm', 'river', 'crown', 'flame',
    'ghost', 'clock', 'brick', 'frost', 'sword',
    'light', 'brush', 'stone', 'shelf', 'grape',
]

# Filler text — repetitive neutral sentences unlikely to confuse the passkey signal
FILLER_SENTENCE = 'the weather was mild and the air was still . '
FILLER_TOKENS_PER_REPEAT = None  # Will be computed after tokenizer loads

# Template: passkey introduction + retrieval cue
INTRO_TEMPLATE   = 'the secret word is {word} .'
RETRIEVAL_CUE    = 'the secret word is'

# D4 dyadic offsets (from condK architecture) that will be within a 2048-token context
D4_OFFSETS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
NON_DYADIC_OFFSETS = [10, 20, 40, 80, 160, 320, 640, 1280]  # powers of 2 * odd = not D4

# Test distances in tokens (filler between passkey and retrieval cue)
TEST_DISTANCES = sorted(set(
    [5, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1700, 1850] +
    [d for d in D4_OFFSETS if d < 1900] +
    [d for d in NON_DYADIC_OFFSETS if d < 1900]
))


def load_model_and_tokenizer(device):
    """Load condK model + tokenizer."""
    # Import DWARFTransformer from the condK training script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'condK', os.path.join(SCRIPT_DIR, 'train_2048_condK.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = mod.BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))
    print(f'Tokenizer: {tokenizer.vocab_size()} tokens')

    # Build model
    model = mod.DWARFTransformer(
        vocab_size    = tokenizer.vocab_size(),
        embedding_dim = 256,
        num_layers    = 6,
        num_heads     = 8,
        ffn_dim       = 1024,
        seq_len       = MAX_SEQ_LEN,
        n_scales      = 11,
    ).to(device)

    # Load best checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoint found at {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    n_params = model.param_count()
    print(f'condK model: {n_params:,} parameters, checkpoint: {ckpt_path}')
    return model, tokenizer, mod


def build_passkey_sequence(tokenizer, passkey_word, filler_tokens, filler_length,
                            retrieval_cue_tokens, max_len=MAX_SEQ_LEN):
    """
    Build: [passkey_intro_tokens] + [filler × enough] + [retrieval_cue_tokens]

    Returns (input_ids, target_token_id, n_filler_actual) where:
      - input_ids: the full token sequence (up to max_len-1 tokens, for x in model(x))
      - target_token_id: the first token of the passkey word
      - n_filler_actual: how many filler tokens actually fit
    """
    intro_tokens = tokenizer.encode(INTRO_TEMPLATE.format(word=passkey_word))

    # How many filler repeats fit?
    available = max_len - 1 - len(intro_tokens) - len(retrieval_cue_tokens) - 1
    # We want exactly filler_length filler tokens (pad with zeros if < filler tokens available)
    if filler_length > available:
        return None  # Doesn't fit

    # Pad filler to exact length (repeat filler sentence, then trim/pad)
    filler = []
    while len(filler) < filler_length:
        filler.extend(filler_tokens)
    filler = filler[:filler_length]

    # Full sequence: intro + filler + cue
    # The model predicts token at position len(intro + filler + cue),
    # which should be the passkey word
    full_seq = intro_tokens + filler + retrieval_cue_tokens
    if len(full_seq) >= max_len:
        return None

    return (full_seq, filler_length)


@torch.no_grad()
def score_passkey(model, tokenizer, input_ids, passkey_word, device):
    """
    Run a single forward pass and return logit statistics at the retrieval position.

    Returns:
      log_prob_correct: log probability of the first passkey token
      rank: rank of passkey token among all vocab tokens (1=best)
      top5_words: top 5 predicted tokens as strings
    """
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.amp.autocast('cuda') if device != 'cpu' else torch.no_grad():
        logits = model(x)  # B=1, N, V

    # We want the prediction at the last position (the token after the retrieval cue)
    last_logits = logits[0, -1]  # V

    # Get the first token of the passkey word
    passkey_token_ids = tokenizer.encode(' ' + passkey_word)  # space prefix for BPE
    if not passkey_token_ids:
        passkey_token_ids = tokenizer.encode(passkey_word)
    target_id = passkey_token_ids[0]

    log_probs = F.log_softmax(last_logits, dim=-1)
    log_prob_correct = log_probs[target_id].item()

    # Rank (1-indexed: 1 = most probable)
    sorted_ids = last_logits.argsort(descending=True)
    rank = (sorted_ids == target_id).nonzero(as_tuple=True)[0].item() + 1

    # Top 5 tokens
    top5_ids = sorted_ids[:5].tolist()
    top5_words = []
    for tid in top5_ids:
        try:
            decoded = tokenizer.decode([tid]).strip()
        except Exception:
            decoded = f'[{tid}]'
        top5_words.append(decoded)

    return log_prob_correct, rank, top5_words


@torch.no_grad()
def run_accuracy_test(model, tokenizer, passkey_word, distractors, input_ids, device):
    """
    Multiple-choice accuracy: is the correct passkey more probable than all 9 distractors?
    Returns True if correct passkey ranks #1 among the 10 options.
    """
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.amp.autocast('cuda') if device != 'cpu' else torch.no_grad():
        logits = model(x)
    last_logits = logits[0, -1]

    candidates = [passkey_word] + distractors
    scores = []
    for word in candidates:
        tids = tokenizer.encode(' ' + word)
        if not tids:
            tids = tokenizer.encode(word)
        scores.append(last_logits[tids[0]].item())

    correct_wins = scores[0] == max(scores)
    return correct_wins, scores


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF condK — Passkey Retrieval Test')
    print('  Tests: specific content preservation vs distributional coherence')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    model, tokenizer, mod = load_model_and_tokenizer(device)

    # Tokenize filler sentence and retrieval cue
    filler_tokens = tokenizer.encode(FILLER_SENTENCE)
    retrieval_cue_tokens = tokenizer.encode(RETRIEVAL_CUE)
    print(f'  Filler sentence: {len(filler_tokens)} tokens per repeat')
    print(f'  Retrieval cue: {len(retrieval_cue_tokens)} tokens')
    print(f'  Passkey words ({len(PASSKEY_WORDS)}): {PASSKEY_WORDS[:5]}...')
    print()

    # Verify passkey words are single (or small number of) tokens
    print('  Token counts for passkeys:')
    for w in PASSKEY_WORDS[:5]:
        tids = tokenizer.encode(' ' + w)
        print(f'    "{w}" → {len(tids)} token(s): {tids}')
    print()

    # ── Main test: accuracy at varying distances ──────────────────────────────
    results_by_distance = {}

    print(f'  Running {len(TEST_DISTANCES)} distance tests...')
    print(f'  {"Distance":>10}  {"Accuracy":>10}  {"Mean Rank":>10}  '
          f'{"Mean LogP":>12}  {"Dyadic?":>8}')
    print('  ' + '-' * 58)

    for dist in TEST_DISTANCES:
        accs, ranks, logps = [], [], []

        for i, passkey_word in enumerate(PASSKEY_WORDS):
            # Distractors: all other passkey words
            distractors = [w for w in PASSKEY_WORDS if w != passkey_word][:9]

            result = build_passkey_sequence(
                tokenizer, passkey_word, filler_tokens,
                filler_length=dist,
                retrieval_cue_tokens=retrieval_cue_tokens,
            )
            if result is None:
                continue  # Doesn't fit in context

            input_ids, n_filler = result

            log_prob, rank, top5 = score_passkey(
                model, tokenizer, input_ids, passkey_word, device)
            correct, scores = run_accuracy_test(
                model, tokenizer, passkey_word, distractors, input_ids, device)

            accs.append(int(correct))
            ranks.append(rank)
            logps.append(log_prob)

        if not accs:
            continue

        acc    = sum(accs) / len(accs)
        m_rank = sum(ranks) / len(ranks)
        m_logp = sum(logps) / len(logps)
        is_d4  = dist in D4_OFFSETS

        print(f'  {dist:>10}  {acc:>9.1%}  {m_rank:>10.1f}  '
              f'{m_logp:>12.3f}  {"D4 tap" if is_d4 else "non-D4":>8}')

        results_by_distance[dist] = {
            'distance_tokens':    dist,
            'accuracy_10way':     acc,
            'mean_rank':          m_rank,
            'mean_log_prob':      m_logp,
            'n_tests':            len(accs),
            'is_d4_offset':       is_d4,
        }

    # ── Summary: D4 vs non-D4 comparison ─────────────────────────────────────
    print()
    print('  ' + '=' * 58)
    print('  D4 TAP OFFSETS vs NON-D4 OFFSETS (same distance ranges)')
    print('  ' + '=' * 58)

    d4_results     = {d: v for d, v in results_by_distance.items() if v['is_d4_offset']}
    non_d4_results = {d: v for d, v in results_by_distance.items() if not v['is_d4_offset']}

    def summarize(group, label):
        if not group: return
        accs  = [v['accuracy_10way'] for v in group.values()]
        ranks = [v['mean_rank']      for v in group.values()]
        logps = [v['mean_log_prob']  for v in group.values()]
        print(f'  {label}:')
        print(f'    Mean accuracy: {sum(accs)/len(accs):.1%}')
        print(f'    Mean rank:     {sum(ranks)/len(ranks):.1f}')
        print(f'    Mean log-prob: {sum(logps)/len(logps):.3f}')

    summarize(d4_results, 'D4 tap offsets')
    summarize(non_d4_results, 'Non-D4 offsets')

    # ── Random baseline ───────────────────────────────────────────────────────
    print()
    print('  ' + '=' * 58)
    print('  RANDOM BASELINE (filler only, no passkey planted)')
    print('  ' + '=' * 58)

    baseline_accs = []
    for passkey_word in PASSKEY_WORDS:
        distractors = [w for w in PASSKEY_WORDS if w != passkey_word][:9]
        # Build sequence with NO passkey intro — just filler + cue
        filler = []
        while len(filler) < 200:
            filler.extend(filler_tokens)
        filler = filler[:200]
        input_ids = filler + retrieval_cue_tokens
        if len(input_ids) >= MAX_SEQ_LEN:
            continue
        correct, _ = run_accuracy_test(
            model, tokenizer, passkey_word, distractors, input_ids, device)
        baseline_accs.append(int(correct))

    if baseline_accs:
        baseline_acc = sum(baseline_accs) / len(baseline_accs)
        print(f'  Baseline accuracy (no passkey): {baseline_acc:.1%}  '
              f'(random chance = 10.0%)')

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print('  ' + '=' * 58)
    print('  INTERPRETATION')
    print('  ' + '=' * 58)

    close_results = {d: v for d, v in results_by_distance.items() if d <= 100}
    far_results   = {d: v for d, v in results_by_distance.items() if d >= 500}

    if close_results:
        close_acc = sum(v['accuracy_10way'] for v in close_results.values()) / len(close_results)
        print(f'  Close range (D≤100):  {close_acc:.1%} accuracy')
    if far_results:
        far_acc = sum(v['accuracy_10way'] for v in far_results.values()) / len(far_results)
        print(f'  Far range  (D≥500):   {far_acc:.1%} accuracy')
    print()
    if close_results and far_results:
        if far_acc >= 0.5:
            print('  ✓ DWARF preserves specific content at distance — wave field is doing real work')
        elif far_acc >= 0.2:
            print('  ~ Partial: some content preserved at distance but degraded')
        else:
            print('  ✗ DWARF does not reliably retrieve specific content at distance')
            print('    → Field preserves distributional coherence, not specific tokens')

    if d4_results and non_d4_results:
        d4_acc  = sum(v['accuracy_10way'] for v in d4_results.values()) / len(d4_results)
        nd4_acc = sum(v['accuracy_10way'] for v in non_d4_results.values()) / len(non_d4_results)
        gap = d4_acc - nd4_acc
        print(f'  D4 vs non-D4 gap: {gap:+.1%}')
        if gap > 0.1:
            print('  ✓ D4 tap offsets show better retrieval — sparsity gap is real')
        elif gap < -0.05:
            print('  ~ Non-D4 actually better — field interpolation may be working')
        else:
            print('  ~ No significant D4 vs non-D4 gap at this scale')

    # Save results
    output = {
        'model':           'condK',
        'checkpoint':      CHECKPOINT_DIR,
        'timestamp':       time.strftime('%Y-%m-%d %H:%M:%S'),
        'results_by_distance': results_by_distance,
        'baseline_accuracy':   baseline_acc if baseline_accs else None,
        'd4_offsets_tested':   list(d4_results.keys()),
        'non_d4_offsets_tested': list(non_d4_results.keys()),
        'passkey_words':    PASSKEY_WORDS,
        'test_format':      'completion: "the secret word is X. [filler] the secret word is"',
        'hypothesis':       'Does DWARF wave field preserve specific content vs distributional coherence?',
    }
    with open(RESULTS_PATH, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f'\n  Full results → {RESULTS_PATH}')


if __name__ == '__main__':
    main()
