#!/usr/bin/env python3
"""
probe_relay_compositionality.py — Relay compositionality test for DSQG models.

Tests whether the passkey retrieval mechanism is compositional relay (signal
propagates through intermediate positions before reaching FA layer) or
parametric lookup (signal appears only at query position, directly).

Design from: Explorations/2026-03-16-flan-relay-vs-lookup.md

Test 1 (Layer-by-Layer Signal):
  Plant passkey at distance d from query. Identify a 2-hop relay path:
  passkey_pos → intermediate_pos → query_pos via δ_a and δ_b.
  Capture residual stream at ALL positions at the output of each layer.
  Measure passkey-identity signal at:
    - passkey_pos (should be high throughout: it's where the word is)
    - intermediate_pos (relay hypothesis: builds before FA layer)
    - query_pos (relay hypothesis: builds AT FA layer)
  
  Relay prediction: passkey_pos has signal from L0. intermediate_pos
  builds signal in L1-L3. query_pos builds signal at L4-L5.
  
  Parametric prediction: passkey_pos has signal from L0. intermediate_pos
  shows little. query_pos builds signal at L4-L5 without staging at intermediate.

Test 2 (Out-of-Distribution Distance):
  Test passkey accuracy at d just beyond coverage frontier.
  Relay prediction: graceful degradation (multi-hop paths still available).
  Parametric prediction: sharp cliff at coverage frontier.

Test 3 (Scrambled-Offset Sanity):
  Compare to a version where relay path for d=256 is broken by zeroing
  the intermediate offset (δ=192 ablated). If relay is compositional,
  accuracy should drop specifically for d=256; distances with direct paths
  should be unaffected.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_relay_compositionality.py \\
      --arch j26d_int2_physics \\
      --checkpoint autoresearch/checkpoints/99437df_j26d_int2_physics_best.pt \\
      --out logs/probe_relay_compositionality_j26d.json

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_relay_compositionality.py \\
      --arch j24d_int2_physics \\
      --checkpoint checkpoints/autoresearch/df0d435_j24d_physics_best.pt \\
      --out logs/probe_relay_compositionality_j24d.json
"""

import os, sys, json, time, math, argparse, copy
import importlib.util
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kernels'))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# ── Reuse arch registry from probe_champion ───────────────────────────────────

_ARCH_INFO = {
    'j24d_int2_physics': {
        'script':     'train/train_j24d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'has_physics': True,
    },
    'j26d_int2_physics': {
        'script':     'train/train_j26d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'has_physics': True,
    },
    'curve_27m': {
        'script':     'train/train_curve_27m_bf16.py',
        'model_cls':  'CurveTransformer',
        'has_physics': True,
    },
}

TOKENIZER_PATH = 'results/2048_condI_tokenizer.json'
MAX_SEQ_LEN    = 2048

# Only use words that tokenize to exactly 6 tokens in the intro template
# (ensures passkey_pos=4 invariant holds). 'grape','mango','melon','peach' are 7 tokens.
_PASSKEY_WORDS  = ['apple', 'banana', 'orange', 'cherry', 'lemon', 'plum']
_INTRO_TEMPLATE = 'the secret word is {word} .'
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_RETRIEVAL_CUE  = 'the secret word is'


def load_model(arch, checkpoint_path, device):
    info = _ARCH_INFO[arch]
    spec = importlib.util.spec_from_file_location(
        f'train_{arch}', os.path.join(ROOT, info['script'])
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    offsets = list(getattr(m, '_COND_N_OFFSETS', None) or getattr(m, 'OFFSETS', []))
    cls = getattr(m, info['model_cls'])
    D  = getattr(m, 'EMBEDDING_DIM', 512)
    L  = getattr(m, 'NUM_LAYERS', 6)
    H  = getattr(m, 'NUM_HEADS', 8)
    F_ = getattr(m, 'FFN_DIM', 2048)
    fa = getattr(m, 'FULL_ATTN_LAYER', 5)
    iv = getattr(m, 'INTERFERENCE', 3)
    vs = getattr(m, 'VOCAB_SIZE', 32000)

    model = cls(
        vocab_size=vs, embedding_dim=D, num_layers=L,
        num_heads=H, ffn_dim=F_, seq_len=MAX_SEQ_LEN,
        full_attn_layer=fa, interference_interval=iv,
        scale_embed_init_val=0.1,
    )

    ckpt_path = os.path.join(ROOT, checkpoint_path)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']

    # Handle torch.compile artifacts
    clean = {}
    for k, v in sd.items():
        clean[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(clean, strict=False)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(os.path.join(ROOT, TOKENIZER_PATH))

    return model, offsets, m, tokenizer, fa, L, D


def build_passkey_sequence(tokenizer, word, relay_d, filler_sentence):
    """
    Build a passkey sequence where query_pos - relay_d = passkey_pos EXACTLY.
    
    The intro is 'the secret word is {word} .' = 6 tokens.
    The passkey word is always at position 4 (0-based) in the intro.
    We set filler_len = relay_d - 5 so that:
      query_pos = len(full_seq) - 1 = 5 + relay_d
      passkey_pos = 4
      query_pos - relay_d = 4 = passkey_pos  ✓

    Returns:
      - token_ids: list of ints
      - passkey_pos: index where the passkey WORD token is (always 4)
      - query_pos: index of last token (where we read prediction)
    """
    filler_ids = tokenizer.encode(filler_sentence).ids
    intro_ids  = tokenizer.encode(_INTRO_TEMPLATE.format(word=word)).ids
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE).ids

    # intro must be 6 tokens for formula to hold
    if len(intro_ids) != 6:
        raise AssertionError(f"Expected 6-token intro for word='{word}', got {len(intro_ids)}")
    if len(cue_ids) != 4:
        raise AssertionError(f"Expected 4-token cue, got {len(cue_ids)}")

    filler_len = relay_d - 5  # ensures query_pos - relay_d = passkey_pos = 4
    assert filler_len >= 0, f"relay_d={relay_d} too small (needs >= 5)"

    filler = []
    while len(filler) < filler_len:
        filler.extend(filler_ids)
    filler = filler[:filler_len]

    full_seq = intro_ids + filler + cue_ids
    passkey_pos = 4       # always index 4 in intro
    query_pos   = len(full_seq) - 1

    assert query_pos - relay_d == passkey_pos, (
        f"Position mismatch: query={query_pos} relay_d={relay_d} passkey={passkey_pos}"
    )

    return full_seq, passkey_pos, query_pos


def run_signal_trace(model, tokenizer, offsets, device, D, L,
                     relay_d, relay_hops, n_trials=5):
    """
    Test 1: Layer-by-layer signal trace.
    
    For a 2-hop relay: passkey_pos → intermediate_pos → query_pos
    Measure cosine similarity between each position's residual stream
    and the embedding of the passkey word, at each layer output.
    
    relay_d: distance from passkey to query
    relay_hops: list of (δ_a, δ_b) pairs to trace, e.g. [(192, 64)]
    """
    print(f'\n─── Test 1: Layer-by-Layer Signal Trace (d={relay_d}) ────────────────────')
    
    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids

    # Get word embedding matrix (for cosine sim reference) — keep on CPU for indexing
    embed_weight = model.embedding.weight.detach().cpu()  # [V, D]

    results = {}

    for δ_a, δ_b in relay_hops:
        assert δ_a + δ_b == relay_d, f"Hop mismatch: {δ_a}+{δ_b} != {relay_d}"
        assert δ_a in offsets, f"δ_a={δ_a} not in offset set"
        assert δ_b in offsets, f"δ_b={δ_b} not in offset set"

        hop_key = f'δ_{δ_a}+{δ_b}'
        print(f'\n  Relay path: passkey → +{δ_a} → +{δ_b} → query')

        layer_signals = {
            'passkey_pos':      {f'L{i}': [] for i in range(L)},
            'intermediate_pos': {f'L{i}': [] for i in range(L)},
            'query_pos':        {f'L{i}': [] for i in range(L)},
        }

        for trial_i in range(n_trials):
            word = _PASSKEY_WORDS[trial_i % len(_PASSKEY_WORDS)]

            seq_ids, passkey_pos, query_pos = build_passkey_sequence(
                tokenizer, word, relay_d, _FILLER_SENTENCE
            )

            # Use the actual token ID at passkey_pos (in-context tokenization may differ)
            word_token_id = seq_ids[passkey_pos]
            word_vec = embed_weight[word_token_id].float()  # [D]
            word_vec_norm = F.normalize(word_vec.unsqueeze(0), dim=-1)  # [1, D]

            # Relay chain: query at t, reads from t-δ_b, which reads from t-δ_b-δ_a=t-relay_d
            # For 2-hop path (δ_a, δ_b): intermediate_pos = query_pos - δ_b
            # For the path (64, 192): query(265) → -192 → 73 → -64 → 9 (passkey at 4!)
            # Wait: with new formula, query=100 for relay_d=96, passkey=4
            # path (48,48): intermediate = 100-48=52, then 52-48=4 = passkey ✓
            intermediate_pos = query_pos - δ_b

            if intermediate_pos <= 0 or passkey_pos < 0:
                print(f'  [WARN] Position out of range: intermediate={intermediate_pos}, passkey={passkey_pos}')
                continue

            # Trim to MAX_SEQ_LEN
            if len(seq_ids) > MAX_SEQ_LEN:
                seq_ids = seq_ids[:MAX_SEQ_LEN]
                query_pos = len(seq_ids) - 1

            # Capture residual stream at all positions, at each layer output
            captured = {}
            handles = []
            
            def make_layer_hook(layer_idx):
                def hook(module, inp, out):
                    # out might be a tuple; take first element
                    y = out[0] if isinstance(out, tuple) else out
                    captured[f'L{layer_idx}'] = y.detach().cpu().float()
                return hook

            for li in range(L):
                handles.append(
                    model.blocks[li].register_forward_hook(make_layer_hook(li))
                )

            x_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(x_tensor)

            for h in handles:
                h.remove()

            # Measure cosine sim at each layer and position
            for li in range(L):
                lkey = f'L{li}'
                if lkey not in captured:
                    continue
                hidden = captured[lkey]  # [1, seq_len, D]

                for pos_name, pos_idx in [
                    ('passkey_pos', passkey_pos),
                    ('intermediate_pos', intermediate_pos),
                    ('query_pos', query_pos),
                ]:
                    if pos_idx >= hidden.shape[1]:
                        continue
                    h_vec = hidden[0, pos_idx, :]  # [D]
                    h_vec_norm = F.normalize(h_vec.unsqueeze(0), dim=-1)
                    cos_sim = (h_vec_norm @ word_vec_norm.T).item()
                    layer_signals[pos_name][lkey].append(cos_sim)

        # Average across trials
        print(f'\n  Cosine similarity with passkey word embedding at each layer:')
        print(f'  {"Layer":<8}  {"passkey_pos":>12}  {"intermediate":>13}  {"query_pos":>12}')
        print(f'  {"─"*8}  {"─"*12}  {"─"*13}  {"─"*12}')

        hop_result = {}
        for li in range(L):
            lkey = f'L{li}'
            pk_sims   = layer_signals['passkey_pos'][lkey]
            int_sims  = layer_signals['intermediate_pos'][lkey]
            qpos_sims = layer_signals['query_pos'][lkey]

            pk_mean   = sum(pk_sims)   / len(pk_sims)   if pk_sims   else float('nan')
            int_mean  = sum(int_sims)  / len(int_sims)  if int_sims  else float('nan')
            qpos_mean = sum(qpos_sims) / len(qpos_sims) if qpos_sims else float('nan')

            print(f'  L{li:<7}  {pk_mean:>12.4f}  {int_mean:>13.4f}  {qpos_mean:>12.4f}')
            hop_result[lkey] = {
                'passkey_cos_sim':      round(pk_mean, 4),
                'intermediate_cos_sim': round(int_mean, 4),
                'query_cos_sim':        round(qpos_mean, 4),
            }

        results[hop_key] = hop_result

        # Interpret
        # Find where intermediate_pos signal first exceeds baseline (L0)
        int_sims_by_layer = [hop_result.get(f'L{li}', {}).get('intermediate_cos_sim', 0) for li in range(L)]
        query_sims_by_layer = [hop_result.get(f'L{li}', {}).get('query_cos_sim', 0) for li in range(L)]
        
        baseline_int = int_sims_by_layer[0]
        baseline_q   = query_sims_by_layer[0]
        
        int_buildup  = max(int_sims_by_layer) - baseline_int
        query_buildup = max(query_sims_by_layer) - baseline_q
        int_peak_layer = int_sims_by_layer.index(max(int_sims_by_layer))
        q_peak_layer   = query_sims_by_layer.index(max(query_sims_by_layer))
        
        print(f'\n  Signal buildup summary (δ_{δ_a}+{δ_b}):')
        print(f'    intermediate_pos: max buildup = {int_buildup:+.4f} at L{int_peak_layer}')
        print(f'    query_pos:        max buildup = {query_buildup:+.4f} at L{q_peak_layer}')
        
        if int_buildup > 0.01 and int_peak_layer < q_peak_layer:
            print(f'    → RELAY SIGNATURE: intermediate builds before query  ✓')
            results[hop_key]['interpretation'] = 'relay_compositional'
        elif int_buildup < 0.005:
            print(f'    → PARAMETRIC SIGNATURE: intermediate shows no buildup  ✗')
            results[hop_key]['interpretation'] = 'parametric_lookup'
        else:
            print(f'    → AMBIGUOUS: intermediate builds but not clearly before query')
            results[hop_key]['interpretation'] = 'ambiguous'

    return results


def run_ood_distance_test(model, tokenizer, offsets, device, n_trials=20):
    """
    Test 2: Out-of-distribution distance generalization.
    
    Test passkey accuracy at distances near and beyond the coverage frontier.
    Relay prediction: graceful degradation beyond frontier.
    Parametric prediction: sharp cliff at frontier.
    """
    print(f'\n─── Test 2: OOD Distance Generalization ──────────────────────────────────')

    max_offset = max(offsets)
    # Test distances: at frontier, 10% and 25% beyond
    test_distances = sorted(set([
        max_offset,
        int(max_offset * 1.1),
        int(max_offset * 1.25),
        int(max_offset * 1.5),
        int(max_offset * 2.0),
    ]))
    test_distances = [d for d in test_distances if d < MAX_SEQ_LEN - 50]

    print(f'  Max offset in set: δ={max_offset}')
    print(f'  Test distances: {test_distances}')

    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids
    results = {}

    for d in test_distances:
        if d < 5:
            continue
        correct = 0
        valid = 0
        for i in range(n_trials):
            word = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            try:
                seq_ids, passkey_pos, query_pos = build_passkey_sequence(
                    tokenizer, word, d, _FILLER_SENTENCE
                )
            except AssertionError:
                continue

            if len(seq_ids) > MAX_SEQ_LEN:
                seq_ids = seq_ids[:MAX_SEQ_LEN]
                query_pos = len(seq_ids) - 1
                passkey_pos = min(passkey_pos, query_pos)

            x = torch.tensor([seq_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)  # [1, seq_len, V]

            # Predict next token at query_pos
            pred_id = logits[0, query_pos, :].argmax().item()
            pred_word = tokenizer.decode([pred_id]).strip()
            correct += int(pred_word.lower().strip() == word.lower().strip())
            valid += 1

        acc = correct / valid if valid > 0 else 0.0
        delta_from_max = d - max_offset
        status = 'AT_FRONTIER' if d == max_offset else (
            'BEYOND_FRONTIER' if d > max_offset else 'WITHIN_FRONTIER'
        )
        print(f'  d={d:6d} ({delta_from_max:+6d} from frontier): {acc:.1%}  [{status}]')
        results[str(d)] = {
            'accuracy': round(acc, 3),
            'correct': correct,
            'trials': valid,
            'delta_from_frontier': delta_from_max,
            'status': status,
        }

    # Check for cliff vs graceful degradation
    frontier_acc = results.get(str(max_offset), {}).get('accuracy', 0)
    beyond_accs = [v['accuracy'] for k, v in results.items() if v['status'] == 'BEYOND_FRONTIER']
    if beyond_accs:
        mean_beyond = sum(beyond_accs) / len(beyond_accs)
        degradation = frontier_acc - mean_beyond
        print(f'\n  Frontier accuracy: {frontier_acc:.1%}')
        print(f'  Mean accuracy beyond frontier: {mean_beyond:.1%}')
        print(f'  Degradation: {degradation:.1%}')
        if mean_beyond > 0.15:
            print(f'  → RELAY SIGNATURE: accuracy maintained beyond frontier (graceful degradation)')
            results['interpretation'] = 'relay_generalizes'
        elif degradation > 0.30:
            print(f'  → PARAMETRIC SIGNATURE: sharp cliff at frontier')
            results['interpretation'] = 'parametric_cliff'
        else:
            results['interpretation'] = 'ambiguous'

    return results


def run_path_break_test(model, tokenizer, offsets, device,
                        relay_d, relay_hops, n_trials=10):
    """
    Test 3: Break relay path, verify targeted accuracy drop.
    
    Zero the intermediate offset (δ_a) from pos_bias.
    Relay prediction: accuracy drops for d=relay_d, intact for d with direct paths.
    Parametric prediction: accuracy drops proportionally to coverage loss,
    not specifically for relay_d.
    """
    print(f'\n─── Test 3: Relay Path Break Test (d={relay_d}) ─────────────────────────')

    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids

    def test_distance_acc(d, model, n_t=n_trials):
        if d < 5:
            return 0.0
        correct = 0; valid = 0
        for i in range(n_t):
            word = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            try:
                seq_ids, _, query_pos = build_passkey_sequence(tokenizer, word, d, _FILLER_SENTENCE)
            except AssertionError:
                continue
            if len(seq_ids) > MAX_SEQ_LEN:
                seq_ids = seq_ids[:MAX_SEQ_LEN]
                query_pos = len(seq_ids) - 1
            x = torch.tensor([seq_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)
            pred_id = logits[0, query_pos, :].argmax().item()
            pred_word = tokenizer.decode([pred_id]).strip()
            correct += int(pred_word.lower().strip() == word.lower().strip())
            valid += 1
        return correct / valid if valid > 0 else 0.0

    results = {}

    for δ_a, δ_b in relay_hops:
        hop_key = f'δ_{δ_a}+{δ_b}'
        print(f'\n  Zeroing δ={δ_a} from pos_bias (breaks relay path for d={relay_d})')

        # Find δ_a index in offset set
        if δ_a not in offsets:
            print(f'  [SKIP] δ={δ_a} not in offset set')
            continue

        j_idx = offsets.index(δ_a)

        # Zero δ_a from pos_bias across all DSQG blocks
        saved = {}
        for li, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
                pb = block.attn.pos_bias  # [J, H]
                saved[f'L{li}'] = pb.data[j_idx].clone()
                pb.data[j_idx] = 0.0

        # Test relay_d (should drop)
        acc_relay_d = test_distance_acc(relay_d, model)
        # Test a direct-path distance (should be largely unaffected)
        # Find direct distances: where δ_a itself is the distance
        direct_d = δ_a if δ_a <= max(offsets) else None
        acc_direct = test_distance_acc(direct_d, model) if direct_d else None

        # Restore
        for li, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
                if f'L{li}' in saved:
                    block.attn.pos_bias.data[j_idx] = saved[f'L{li}']

        # Baseline
        acc_relay_d_base  = test_distance_acc(relay_d, model)
        acc_direct_base   = test_distance_acc(direct_d, model) if direct_d else None

        print(f'  d={relay_d} (relay path):  baseline={acc_relay_d_base:.1%}  after_break={acc_relay_d:.1%}  drop={acc_relay_d_base-acc_relay_d:.1%}')
        if acc_direct is not None:
            print(f'  d={direct_d} (direct δ_a): baseline={acc_direct_base:.1%}  after_break={acc_direct:.1%}  drop={acc_direct_base-acc_direct:.1%}')

        results[hop_key] = {
            f'd_{relay_d}_baseline':   round(acc_relay_d_base, 3),
            f'd_{relay_d}_after_break': round(acc_relay_d, 3),
            f'd_{relay_d}_drop':       round(acc_relay_d_base - acc_relay_d, 3),
        }
        if direct_d and acc_direct is not None:
            results[hop_key][f'd_{direct_d}_baseline']    = round(acc_direct_base, 3)
            results[hop_key][f'd_{direct_d}_after_break'] = round(acc_direct, 3)
            results[hop_key][f'd_{direct_d}_drop']        = round(acc_direct_base - acc_direct, 3)

        relay_drop   = acc_relay_d_base - acc_relay_d
        direct_drop  = (acc_direct_base - acc_direct) if acc_direct is not None else 0.0
        
        if relay_drop > 0.20 and relay_drop > (direct_drop * 2):
            print(f'  → RELAY SIGNATURE: targeted drop for relay_d >> direct_d  ✓')
            results[hop_key]['interpretation'] = 'relay_targeted_drop'
        elif relay_drop < 0.05:
            print(f'  → PARAMETRIC SIGNATURE: no targeted drop for relay_d  ✗')
            results[hop_key]['interpretation'] = 'no_relay_dependency'
        else:
            results[hop_key]['interpretation'] = 'ambiguous'

    return results


def run_probe(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, offsets, train_m, tokenizer, fa_layer, L, D = load_model(
        args.arch, args.checkpoint, device
    )

    print(f'\nModel: {args.arch}')
    print(f'Offsets (J={len(offsets)}): {offsets[:8]}...{offsets[-4:]}')
    print(f'Layers: {L}, FA layer: {fa_layer}, D={D}')

    # ── Define relay paths to test ─────────────────────────────────────────────
    # For J24D/J26D (offsets include 64, 96, 192, 384, 512...):
    # d=256 via δ=192+64 (both in set)
    # d=192 via δ=96+96 (both in set)
    # Pick the cleanest 2-hop path where no direct δ = relay_d exists

    # Find distances without a direct offset, but with a clean 2-hop path
    # Require d >= 10 (formula: filler_len = d-5 >= 5)
    relay_candidates = []
    offset_set = set(offsets)
    for d in range(10, 500):
        if d in offset_set:
            continue  # direct path exists — not a pure relay test
        # Find 2-hop paths via (δ_a, δ_b) where δ_a + δ_b = d
        for δ_a in offsets:
            δ_b = d - δ_a
            if δ_b > 0 and δ_b in offset_set:
                relay_candidates.append((d, δ_a, δ_b))
                break

    # Use first relay_d with a clean 2-hop path, prefer d ~ 96-200 (short enough to be clean)
    relay_d = None
    relay_hops = []
    for d, δ_a, δ_b in sorted(relay_candidates, key=lambda x: abs(x[0] - 150)):
        relay_d = d
        relay_hops = [(δ_a, δ_b)]
        print(f'\nRelay test target: d={relay_d}, path: δ_{δ_a}+{δ_b}={relay_d}')
        print(f'  (d={relay_d} not in offset set: confirmed 2-hop relay required)')
        break

    if relay_d is None:
        print('[WARN] No clean 2-hop relay path found. Defaulting to d=192, δ=96+96.')
        relay_d = 192
        relay_hops = [(96, 96)]

    # ── Run tests ─────────────────────────────────────────────────────────────
    all_results = {
        'arch': args.arch,
        'checkpoint': args.checkpoint,
        'offsets': offsets,
        'relay_d': relay_d,
        'relay_hops': relay_hops,
    }

    # Test 1: Layer-by-layer signal trace
    t1 = run_signal_trace(
        model, tokenizer, offsets, device, D, L,
        relay_d, relay_hops, n_trials=args.n_trials
    )
    all_results['test1_signal_trace'] = t1

    # Test 2: OOD distance
    t2 = run_ood_distance_test(model, tokenizer, offsets, device, n_trials=args.n_trials)
    all_results['test2_ood_distance'] = t2

    # Test 3: Path break
    t3 = run_path_break_test(
        model, tokenizer, offsets, device,
        relay_d, relay_hops, n_trials=args.n_trials
    )
    all_results['test3_path_break'] = t3

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n─── Summary ────────────────────────────────────────────────────────────')
    
    interps = []
    for hop_key in all_results.get('test1_signal_trace', {}):
        interp = all_results['test1_signal_trace'][hop_key].get('interpretation')
        if interp:
            print(f'  Test 1 ({hop_key}): {interp}')
            interps.append(interp)
    
    t2_interp = all_results.get('test2_ood_distance', {}).get('interpretation')
    if t2_interp:
        print(f'  Test 2 (OOD generalization): {t2_interp}')
        interps.append(t2_interp)
    
    for hop_key in all_results.get('test3_path_break', {}):
        interp = all_results['test3_path_break'][hop_key].get('interpretation')
        if interp:
            print(f'  Test 3 ({hop_key}): {interp}')
            interps.append(interp)

    relay_votes    = sum(1 for i in interps if 'relay' in i)
    param_votes    = sum(1 for i in interps if 'parametric' in i)
    overall = (
        'relay_compositional' if relay_votes > param_votes
        else ('parametric_lookup' if param_votes > relay_votes else 'ambiguous')
    )
    print(f'\n  Overall: {relay_votes} relay signals, {param_votes} parametric signals')
    print(f'  Conclusion: {overall}')
    all_results['overall_interpretation'] = overall

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved → {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Relay compositionality probe')
    parser.add_argument('--arch',        required=True, choices=sorted(_ARCH_INFO.keys()))
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--out',         required=True)
    parser.add_argument('--n-trials',    type=int, default=10,
                        help='Trials per distance (default 10)')
    args = parser.parse_args()
    run_probe(args)
