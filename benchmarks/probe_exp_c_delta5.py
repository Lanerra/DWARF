"""
probe_exp_c_delta5.py — Exp C: δ=5 ablation probe for J13D v2.

Tests the dual-edged δ=5 hypothesis from doc 41:
  - Prediction: zeroing δ=5 INCREASES d=8, d=16 accuracy (shortcut bypass removed)
  - Prediction: zeroing δ=5 DECREASES d=64, d=256, d=1024 accuracy (amplifier step removed)
  - Result discriminates: shortcut-bypass vs amplifier mechanisms

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_exp_c_delta5.py

Output: benchmarks/logs/probe_exp_c_delta5_j13dv2.json + terminal summary.
"""

import os, sys, json, time, math, copy, contextlib, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kernels'))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
ARCH_SCRIPT  = os.path.join(ROOT, 'train', 'train_borg_j13d_v2_30m_4090_bf16.py')
CHECKPOINT   = os.path.join(ROOT, 'autoresearch', 'checkpoints', 'j13d_v2_30m_ep3_resume.pt')
TOKENIZER_PATH = os.path.join(ROOT, 'results', '2048_condI_tokenizer.json')
DATASET_PATH   = os.path.join(ROOT, 'logs', 'fineweb_encoded_2048.pt')
OUT_PATH       = os.path.join(ROOT, 'benchmarks', 'logs', 'probe_exp_c_delta5_j13dv2.json')

MAX_SEQ_LEN    = 2048
PASSKEY_TRIALS = 50   # matches training script default (5% granularity)
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(device):
    spec = importlib.util.spec_from_file_location('train_j13d_v2', ARCH_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    offsets  = list(m.OFFSETS)
    D  = m.EMBEDDING_DIM
    L  = m.NUM_LAYERS
    H  = m.NUM_HEADS
    F_ = m.FFN_DIM
    fa = m.FULL_ATTN_LAYER
    vs = m.VOCAB_SIZE

    model = m.AutoresearchTransformerPhysics(
        vocab_size=vs, embedding_dim=D, num_layers=L,
        num_heads=H, ffn_dim=F_, seq_len=MAX_SEQ_LEN,
        full_attn_layer=fa,
        scale_embed_init_val=0.1,
    )

    ck    = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state = ck['model_state_dict']
    elif isinstance(ck, dict) and any(k.startswith('embedding') or k.startswith('blocks') for k in ck):
        state = ck  # raw state dict
    else:
        state = ck  # fallback
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print(f'  Loaded {CHECKPOINT}')
    print(f'  Offsets ({len(offsets)}): {offsets}')
    print(f'  Arch: D={D}, L={L}, H={H}, FFN={F_}, FA=L{fa}')
    return model, offsets, m


# ── Passkey eval ────────────────────────────────────────────────────────────────
def passkey_eval(model, tokenizer, device, m_ref):
    words   = list(m_ref._PASSKEY_WORDS)
    intro_t = m_ref._INTRO_TEMPLATE
    filler_s= m_ref._FILLER_SENTENCE
    cue_s   = m_ref._RETRIEVAL_CUE
    filler_ids = tokenizer.encode(filler_s)
    cue_ids    = tokenizer.encode(cue_s)
    results = {}
    with torch.no_grad():
        for d in PASSKEY_DISTANCES:
            correct = 0; n_valid = 0
            for i in range(PASSKEY_TRIALS):
                target   = words[i % len(words)]
                others   = [w for w in words if w != target]
                intro_ids = tokenizer.encode(intro_t.format(word=target))
                available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
                if d > available:
                    continue
                filler = []
                while len(filler) < d:
                    filler.extend(filler_ids)
                full_seq = intro_ids + filler[:d] + cue_ids
                if len(full_seq) >= MAX_SEQ_LEN:
                    continue
                ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                cand_ids = [(tokenizer.encode(' '+w) or tokenizer.encode(w))[0]
                            for w in [target]+others[:9]]
                pred = ([target]+others[:9])[logits[0][cand_ids].argmax().item()]
                correct  += int(pred == target)
                n_valid  += 1
            results[d] = round(correct / n_valid, 3) if n_valid else 0.0
    mean = round(sum(results.values()) / len(results), 3)
    return results, mean


# ── PPL eval ───────────────────────────────────────────────────────────────────
def ppl_eval(model, val_data, device, n=50):
    model.eval()
    total_loss = 0.0; total_tokens = 0
    with torch.no_grad():
        for i in range(min(n, len(val_data))):
            batch = val_data[i:i+1].to(device)
            x, y  = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast('cuda'):
                logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-1)
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
    return round(math.exp(total_loss / total_tokens), 4)


# ── Zero-offset ablation context manager ───────────────────────────────────────
@contextlib.contextmanager
def zero_offset_rows(model, j_indices):
    """Temporarily zero pos_bias[j_idx] and scale_embed[j_idx] across all DSQG layers."""
    saved = {}
    for li, block in enumerate(model.blocks):
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        pb = getattr(attn, 'pos_bias', None)
        se = getattr(attn, 'scale_embed', None)
        if pb is None:
            continue
        saved[(li, 'pb')] = pb.data[j_indices].clone()
        pb.data[j_indices] = 0.0
        if se is not None:
            saved[(li, 'se')] = se.data[j_indices].clone()
            se.data[j_indices] = 0.0
    try:
        yield
    finally:
        for li, block in enumerate(model.blocks):
            attn = getattr(block, 'attn', None)
            if attn is None: continue
            pb = getattr(attn, 'pos_bias', None)
            se = getattr(attn, 'scale_embed', None)
            if pb is None: continue
            pb.data[j_indices] = saved[(li, 'pb')]
            if se is not None and (li, 'se') in saved:
                se.data[j_indices] = saved[(li, 'se')]


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    _raw_tok = Tokenizer.from_file(TOKENIZER_PATH)
    class _TokWrapper:
        def __init__(self, t): self._t = t
        def encode(self, text):
            enc = self._t.encode(text)
            return enc.ids
        def decode(self, ids): return self._t.decode(ids)
        def get_vocab_size(self): return self._t.get_vocab_size()
    tokenizer = _TokWrapper(_raw_tok)
    print(f'Tokenizer: vocab={tokenizer.get_vocab_size()}')

    raw_cache = torch.load(DATASET_PATH, map_location='cpu', weights_only=False)
    if isinstance(raw_cache, dict):
        # May have 'train' and 'val' keys
        val_data = raw_cache.get('val', raw_cache.get('data', next(iter(raw_cache.values()))))
    else:
        val_data = raw_cache
    if len(val_data) < 3:
        # If only a few sequences, use them all; otherwise trim to first 200 for speed
        pass
    elif len(val_data) > 200:
        val_data = val_data[:200]
    print(f'Val data: {len(val_data)} sequences')

    model, offsets, m_ref = load_model(device)
    delta5_idx = offsets.index(5)
    print(f'\nδ=5 is at index {delta5_idx} in offsets: {offsets}')

    results = {
        'experiment':  'exp_c_delta5_ablation',
        'model':       'j13d_v2_30m_ep3',
        'checkpoint':  CHECKPOINT,
        'offsets':     offsets,
        'delta5_idx':  delta5_idx,
        'trials_per_distance': PASSKEY_TRIALS,
    }

    # ── Baseline ───────────────────────────────────────────────────────────────
    print('\n─── Baseline (no ablation) ──────────────────────────────────────────────')
    t0 = time.time()
    pk_by_d, mean_pk = passkey_eval(model, tokenizer, device, m_ref)
    baseline_ppl     = ppl_eval(model, val_data, device)
    print(f'  Passkey: {mean_pk:.1%}  |  PPL: {baseline_ppl}  ({time.time()-t0:.0f}s)')
    for d, v in sorted(pk_by_d.items()):
        print(f'    d={d:5d}: {v:.0%}')
    results['baseline'] = {
        'passkey': mean_pk,
        'ppl': baseline_ppl,
        'passkey_by_d': {str(d): v for d, v in pk_by_d.items()},
    }

    # ── δ=5 ablation ───────────────────────────────────────────────────────────
    print('\n─── δ=5 zeroed ──────────────────────────────────────────────────────────')
    t0 = time.time()
    j_tensor = torch.tensor([delta5_idx])
    with zero_offset_rows(model, j_tensor):
        pk5_by_d, mean_pk5 = passkey_eval(model, tokenizer, device, m_ref)
        ppl5               = ppl_eval(model, val_data, device)
    print(f'  Passkey: {mean_pk5:.1%} ({mean_pk5-mean_pk:+.1%})  |  PPL: {ppl5} ({ppl5-baseline_ppl:+.2f})  ({time.time()-t0:.0f}s)')
    print(f'\n  Per-distance Δ (zero_δ5 - baseline):')
    print(f'  {"d":>6}  {"base":>6}  {"zero_δ5":>8}  {"Δ":>6}  note')
    print(f'  {"──":>6}  {"────":>6}  {"───────":>8}  {"──":>6}')
    for d in sorted(pk_by_d.keys()):
        base  = pk_by_d[d]
        ablat = pk5_by_d[d]
        delta = ablat - base
        pred  = ''
        if d in (8, 16):
            pred = '← expect +'
        elif d in (64, 256, 1024):
            pred = '← expect −'
        elif d in (1, 2):
            pred = '← expect −'
        flag  = '✓' if (
            (d in (8, 16) and delta > 0) or
            (d in (1, 2, 64, 256, 1024) and delta < 0)
        ) else ('✗' if (
            (d in (8, 16) and delta < -0.05) or
            (d in (1, 2, 64, 256, 1024) and delta > 0.05)
        ) else ' ')
        print(f'  {d:6d}  {base:6.1%}  {ablat:8.1%}  {delta:+.0%}  {flag} {pred}')
    results['delta5_ablation'] = {
        'passkey': mean_pk5,
        'ppl': ppl5,
        'passkey_delta': round(mean_pk5 - mean_pk, 3),
        'ppl_delta': round(ppl5 - baseline_ppl, 3),
        'passkey_by_d': {str(d): v for d, v in pk5_by_d.items()},
        'passkey_delta_by_d': {str(d): round(pk5_by_d[d] - pk_by_d[d], 3) for d in pk_by_d},
    }

    # ── Adjacent offset comparison: δ=4 and δ=8 ablations ──────────────────────
    print('\n─── Control ablations: δ=4 and δ=8 (structural neighbors) ─────────────')
    for ctrl_delta in [4, 8]:
        if ctrl_delta not in offsets:
            print(f'  δ={ctrl_delta}: not in offset set, skipping')
            continue
        j_ctrl = offsets.index(ctrl_delta)
        t0 = time.time()
        j_tensor_ctrl = torch.tensor([j_ctrl])
        with zero_offset_rows(model, j_tensor_ctrl):
            pk_ctrl_by_d, mean_pk_ctrl = passkey_eval(model, tokenizer, device, m_ref)
            ppl_ctrl                   = ppl_eval(model, val_data, device, n=25)
        print(f'  δ={ctrl_delta}: passkey={mean_pk_ctrl:.1%} ({mean_pk_ctrl-mean_pk:+.1%})  '
              f'ppl={ppl_ctrl} ({ppl_ctrl-baseline_ppl:+.2f})  ({time.time()-t0:.0f}s)')
        results[f'ctrl_delta{ctrl_delta}_ablation'] = {
            'passkey': mean_pk_ctrl,
            'ppl': ppl_ctrl,
            'passkey_delta': round(mean_pk_ctrl - mean_pk, 3),
            'ppl_delta': round(ppl_ctrl - baseline_ppl, 3),
            'passkey_by_d': {str(d): v for d, v in pk_ctrl_by_d.items()},
        }

    # ── Summary: hypothesis verdict ─────────────────────────────────────────────
    d5_results = results['delta5_ablation']['passkey_delta_by_d']
    shortcut_verdict = all(
        d5_results.get(str(d), 0) > 0
        for d in [8, 16]
    )
    amplifier_verdict = all(
        d5_results.get(str(d), 0) < 0
        for d in [64, 256, 1024]
    )
    print('\n─── Hypothesis verdict ──────────────────────────────────────────────────')
    print(f'  Shortcut-bypass at d=8,16: {"CONFIRMED ✓" if shortcut_verdict else "not confirmed ✗"}')
    print(f'  Amplifier at d=64,256,1024: {"CONFIRMED ✓" if amplifier_verdict else "not confirmed ✗"}')
    if shortcut_verdict and amplifier_verdict:
        print('  → DUAL-EDGED δ=5 hypothesis CONFIRMED: shortcut at mid-range, amplifier at long.')
    elif shortcut_verdict:
        print('  → Shortcut only: δ=5 hurts mid-range but evidence for amplifier role inconclusive.')
    elif amplifier_verdict:
        print('  → Amplifier only: δ=5 helps long-range but mid-range shortcut hypothesis fails.')
    else:
        print('  → Neither prediction confirmed cleanly. Review per-distance data.')
    results['hypothesis_verdict'] = {
        'shortcut_bypass_d8_d16': shortcut_verdict,
        'amplifier_d64_d256_d1024': amplifier_verdict,
    }

    # ── Write output ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {OUT_PATH}')
