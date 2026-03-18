"""
probe_champion.py — Mechanistic ablation probe for DSQG champion models.

Runs on any supported champion checkpoint and generates a full ablation report:
  Section 1 — Offset importance  (single-offset + group ablations)
  Section 2 — Layer residual knockouts
  Section 3 — Head knockouts (penultimate DSQG + full-attn layers)
  Section 4 — Dense neighborhood contribution (PPL only)
  Section 5 — EMA differential (b1 vs b3 isolation) [curve_27m / physics models]
  Section 6 — Interference layer position (L1-IF vs L3-IF) [curve_27m / physics models]
  Section 7 — FA delta magnitude by pre-FA layer depth (residual stream analysis)
  Section 8 — pos_bias asymmetry (offset specialization analysis)
  Section 9 — Scale_embed per-offset contribution (learned scale analysis)
  Section 10 — IF block inference removability (self-erasing scaffold test)

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_champion.py \\
      --arch d41_35m \\
      --checkpoint checkpoints/d41_35m/best.pt \\
      --out logs/probe_d41_35m_results.json

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_champion.py \\
      --arch condx_v2 \\
      --checkpoint checkpoints/condX_v2_35m_bf16/best.pt \\
      --out logs/probe_condX_v2_results.json

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/probe_champion.py \\
      --arch curve_27m \\
      --checkpoint checkpoints/curve_27m_best.pt \\
      --out logs/probe_curve_27m_results.json

Supported --arch values: d41_35m, condx_v2, j24d_int2_physics, curve_27m
"""

import os, sys, json, time, math, argparse, copy, contextlib
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'kernels'))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# ── Architecture registry ──────────────────────────────────────────────────────

_ARCH_INFO = {
    'd41_35m': {
        'script':     'train/train_2048_35m_d41.py',
        'model_cls':  'CondMTransformer',
        'is_bypass':  False,
    },
    'condx_v2': {
        'script':     'train/train_2048_35m_condX_v2_bf16.py',
        'model_cls':  'CondXTransformer',
        'is_bypass':  True,   # has FullCausalAttentionBypass at full_attn_layer
    },
    'j24d_int2_physics': {
        'script':     'train/train_j24d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'j26d_int2_physics': {
        'script':     'train/train_j26d_int2_physics_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
    },
    'j20d_v10_L8': {
        'script':     'train/train_j20d_v10_L8_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'j20d_v10_L12': {
        'script':     'train/train_j20d_v10_L12_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'j20d_v10_L10': {
        'script':     'train/train_j20d_v10_L10_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'curve_27m': {
        'script':     'train/train_curve_27m_bf16.py',
        'model_cls':  'CurveTransformer',
        'is_bypass':  False,
        'has_physics': True,  # has EMA+KdV blocks; enables Sections 5 & 6
    },
    'borg_adapt_warmstart': {
        'script':     'train/train_borg_adapt_13m_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_midattn': {
        'script':     'train/train_borg_midattn_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_lastattn': {
        'script':     'train/train_borg_lastattn_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_midattn_gen2': {
        'script':     'train/train_borg_midattn_unfreeze_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_midfa_L0': {
        'script':     'train/train_borg_midfa_L0_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg2_dual_fa': {
        'script':     'train/train_borg2_dual_fa_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
        'multi_fa':   True,
    },
    'borg_gen4_L11': {
        'script':     'train/train_borg_gen4_L11_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_gen5_L11_preIF': {
        'script':     'train/train_borg_gen5_L11_preIF_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_gen5_L8_preIF': {
        'script':     'train/train_borg_gen5_L8_preIF_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_gen3_L8': {
        'script':     'train/train_borg_gen3_L8_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'borg_L11': {
        'script':     'train/train_borg_L11_bf16.py',
        'model_cls':  'AutoresearchTransformerPhysics',
        'is_bypass':  False,
        'has_physics': True,
    },
    'cond_delta': {
        'script':     'train/train_cond_delta_bf16.py',
        'model_cls':  'AutoresearchTransformerCondDelta',
        'is_bypass':  False,
        'has_physics': True,
    },
}

# ── Dataset / tokenizer constants (condI, shared across all curve models) ─────

TOKENIZER_PATH = 'results/2048_condI_tokenizer.json'
DATASET_PATH   = 'logs/fineweb_encoded_2048.pt'
MAX_SEQ_LEN    = 2048

# ── Passkey eval ───────────────────────────────────────────────────────────────
# NOTE: constants are loaded from the training module in load_model() so they
# match exactly what each model was trained with.  These are fallback defaults.

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 10   # 10 trials → 10% granularity; fast but readable
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'melon', 'peach', 'plum']
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_RETRIEVAL_CUE   = 'the secret word is'


def passkey_eval(model, tokenizer, device):
    """Quick passkey accuracy across all distances. Returns (dict d→float, mean)."""
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    with torch.no_grad():
        for d in PASSKEY_DISTANCES:
            correct = 0; n_valid = 0
            for i in range(PASSKEY_TRIALS):
                target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
                others    = [w for w in _PASSKEY_WORDS if w != target]
                intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
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
                cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                            for w in [target] + others[:9]]
                pred = ([target] + others[:9])[logits[0][cand_ids].argmax().item()]
                correct  += int(pred == target)
                n_valid  += 1
            results[d] = round(correct / n_valid, 3) if n_valid else 0.0
    mean_pk = round(sum(results.values()) / len(results), 3)
    return results, mean_pk


def ppl_eval(model, val_data, device, n_batches=100):
    """Quick PPL on first n_batches of validation data."""
    model.eval()
    total_loss = 0.0; total_tokens = 0
    with torch.no_grad():
        for i in range(min(n_batches, len(val_data))):
            batch = val_data[i:i+1].to(device)
            x, y  = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast('cuda'):
                logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-1
            )
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
    return round(math.exp(total_loss / total_tokens), 4)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(arch, checkpoint_path, device):
    info = _ARCH_INFO[arch]
    spec = importlib.util.spec_from_file_location(
        f'train_{arch}',
        os.path.join(ROOT, info['script'])
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    offsets = list(getattr(m, '_COND_N_OFFSETS', None) or getattr(m, 'OFFSETS', []))
    cls     = getattr(m, info['model_cls'])
    D  = getattr(m, 'EMBEDDING_DIM', 512)
    L  = getattr(m, 'NUM_LAYERS', 6)
    H  = getattr(m, 'NUM_HEADS', 8)
    F_ = getattr(m, 'FFN_DIM', 2048)
    fa = getattr(m, 'FULL_ATTN_LAYER', 5)
    iv = getattr(m, 'INTERFERENCE', 3)
    vs = getattr(m, 'VOCAB_SIZE', 32000)

    # Physics models (AutoresearchTransformerPhysics, CurveTransformer) require
    # scale_embed_init_val; d41/condx_v2 do not accept it.
    extra_kwargs = {}
    if info.get('has_physics') or info['model_cls'] in (
            'AutoresearchTransformerPhysics', 'CurveTransformer',
            'AutoresearchTransformerCondDelta'):
        extra_kwargs['scale_embed_init_val'] = 0.1

    # Multi-FA models (borg2_dual_fa) use full_attn_layers (list) instead of full_attn_layer
    if info.get('multi_fa'):
        fa_layers = getattr(m, 'FULL_ATTN_LAYERS', [2, 5])
        model = cls(
            vocab_size=vs, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=F_, seq_len=MAX_SEQ_LEN,
            full_attn_layers=fa_layers, interference_interval=iv,
            **extra_kwargs,
        )
        model.full_attn_layer = fa_layers[-1]
    else:
        model = cls(
            vocab_size=vs, embedding_dim=D, num_layers=L,
            num_heads=H, ffn_dim=F_, seq_len=MAX_SEQ_LEN,
            full_attn_layer=fa, interference_interval=iv,
            **extra_kwargs,
        )

    ck    = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)
    # Strip torch.compile _orig_mod prefixes if present
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    val_ppl = ck.get('val_ppl', '?')
    print(f'  Loaded {checkpoint_path}  (val_ppl={val_ppl})')
    print(f'  Offsets: J={len(offsets)}, max={max(offsets)}, '
          f'dense=[0..{max(o for o in offsets if o<=50)}], '
          f'sparse={[o for o in offsets if o>50]}')
    print(f'  Arch: D={D}, L={L}, H={H}, FFN={F_}, FA=L{fa}, IF=/{iv}')

    # Override passkey constants from training module (ensures exact match)
    global _PASSKEY_WORDS, _INTRO_TEMPLATE, _FILLER_SENTENCE, _RETRIEVAL_CUE
    global PASSKEY_DISTANCES
    if hasattr(m, '_PASSKEY_WORDS'):
        _PASSKEY_WORDS   = list(m._PASSKEY_WORDS)
        print(f'  Passkey words from script: {_PASSKEY_WORDS[:4]}...')
    if hasattr(m, '_INTRO_TEMPLATE'):
        _INTRO_TEMPLATE  = m._INTRO_TEMPLATE
    if hasattr(m, '_FILLER_SENTENCE'):
        _FILLER_SENTENCE = m._FILLER_SENTENCE
    if hasattr(m, '_RETRIEVAL_CUE'):
        _RETRIEVAL_CUE   = m._RETRIEVAL_CUE
    if hasattr(m, 'PASSKEY_DISTANCES'):
        PASSKEY_DISTANCES = list(m.PASSKEY_DISTANCES)

    return model, offsets, m


# ── Ablation helpers ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def zero_offset_rows(model, j_indices):
    """Temporarily zero pos_bias[j_indices,:] and scale_embed[j_indices,:]
    across all DSQG layers, then restore."""
    saved = {}
    for li, block in enumerate(model.blocks):
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        pb = getattr(attn, 'pos_bias', None)
        se = getattr(attn, 'scale_embed', None)
        if pb is None:
            continue
        key = (li, 'pb')
        saved[key] = pb.data[j_indices].clone()
        pb.data[j_indices] = 0.0
        if se is not None:
            key2 = (li, 'se')
            saved[key2] = se.data[j_indices].clone()
            se.data[j_indices] = 0.0
    try:
        yield
    finally:
        for li, block in enumerate(model.blocks):
            attn = getattr(block, 'attn', None)
            if attn is None:
                continue
            pb = getattr(attn, 'pos_bias', None)
            se = getattr(attn, 'scale_embed', None)
            if pb is None:
                continue
            pb.data[saved[(li, 'pb')].shape[0] if False else slice(None)]  # no-op guard
            # restore via index
            pb.data[j_indices] = saved[(li, 'pb')]
            if se is not None and (li, 'se') in saved:
                se.data[j_indices] = saved[(li, 'se')]


@contextlib.contextmanager
def zero_layer_residual(model, layer_idx):
    """Temporarily bypass a layer's residual contribution."""
    original_forward = model.blocks[layer_idx].forward

    def zeroed_forward(x, *a, **kw):
        return torch.zeros_like(x)

    model.blocks[layer_idx].forward = zeroed_forward
    try:
        yield
    finally:
        model.blocks[layer_idx].forward = original_forward


@contextlib.contextmanager
def zero_head(model, layer_idx, head_idx):
    """Zero one attention head's Q·K logits (set to -inf pre-softmax)
    via a forward hook on the DSQG or full-attn block."""
    handles = []

    block = model.blocks[layer_idx]
    attn  = block.attn

    # For DSQG heads: hook qkv_proj output and mask that head's Q slice
    if hasattr(attn, 'pos_bias'):   # DSQG block
        HD = attn.head_dim if hasattr(attn, 'head_dim') else (
             attn.qkv_proj.weight.shape[1] // attn.num_heads)

        def _hook(module, inp, out):
            # out: [B, N, 3D] — zero Q for head_idx
            out = out.clone()
            start = head_idx * HD
            out[:, :, start:start+HD] = 0.0
            return out

        handles.append(attn.qkv_proj.register_forward_hook(_hook))
    elif hasattr(attn, 'q_proj'):   # FullCausalAttentionBypass (condX-v2)
        HD = attn.head_dim

        def _hook_q(module, inp, out):
            out = out.clone()
            start = head_idx * HD
            out[:, :, start:start+HD] = 0.0
            return out

        handles.append(attn.q_proj.register_forward_hook(_hook_q))
    elif hasattr(attn, 'qkv_proj'):  # Standard FullCausalAttention
        HD = attn.head_dim if hasattr(attn, 'head_dim') else (
             attn.qkv_proj.weight.shape[1] // attn.num_heads)

        def _hook_qkv(module, inp, out):
            out = out.clone()
            start = head_idx * HD
            out[:, :, start:start+HD] = 0.0
            return out

        handles.append(attn.qkv_proj.register_forward_hook(_hook_qkv))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def zero_ema_block(model, ema_block_idx):
    """Zero the EMA output for a specific interference (EMA+KdV) block.

    Interference blocks in physics models (DSQGBlockV8Physics) are identified by
    having an `ema_factor` parameter.  They appear at layers where
    layer_idx % interference_interval == interference_interval - 1.

    ema_block_idx=0 → first interference block (L1 for IF=2, L=6)
    ema_block_idx=1 → second interference block (L3 for IF=2, L=6)

    We zero the interference sub-block output by hooking the inter_gate linear
    and clamping its output to zero, which drives the gated EMA contribution
    to zero while leaving the main DSQG path intact.
    """
    # Find interference blocks in order
    if_blocks = [
        (li, block) for li, block in enumerate(model.blocks)
        if hasattr(block, 'ema_factor')
    ]
    if ema_block_idx >= len(if_blocks):
        raise IndexError(
            f'ema_block_idx={ema_block_idx} out of range '
            f'(found {len(if_blocks)} interference blocks)'
        )
    li, block = if_blocks[ema_block_idx]
    handles = []
    # Hook inter_gate to zero its output → gates the EMA/KV injection to zero
    if hasattr(block, 'inter_gate'):
        def _zero_gate(module, inp, out):
            return torch.zeros_like(out)
        handles.append(block.inter_gate.register_forward_hook(_zero_gate))
    try:
        yield li  # yield the actual layer index for logging
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def zero_if_layer_output(model, ema_block_idx):
    """Zero the *entire residual contribution* of one interference block.

    Unlike zero_ema_block (which only kills the EMA/KV injection path),
    this zeros the full block output so the residual stream passes through
    unmodified — isolating whether the staging function itself matters,
    not just the EMA signal.

    ema_block_idx: 0=first IF block, 1=second IF block (same indexing as above).
    """
    if_blocks = [
        (li, block) for li, block in enumerate(model.blocks)
        if hasattr(block, 'ema_factor')
    ]
    if ema_block_idx >= len(if_blocks):
        raise IndexError(
            f'ema_block_idx={ema_block_idx} out of range '
            f'(found {len(if_blocks)} interference blocks)'
        )
    li, block = if_blocks[ema_block_idx]
    # Use the same approach as zero_layer_residual: replace forward to return zeros
    original_forward = block.forward

    def zeroed_forward(x, *a, **kw):
        return torch.zeros_like(x)

    block.forward = zeroed_forward
    try:
        yield li
    finally:
        block.forward = original_forward


# ── Main probe ─────────────────────────────────────────────────────────────────

def run_probe(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*72}')
    print(f'  Champion Ablation Probe: {args.arch}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'{"="*72}\n')

    # Load tokenizer — wrap so .encode() returns IDs directly (matches training)
    from tokenizers import Tokenizer as _HFTokenizer
    _raw_tok = _HFTokenizer.from_file(os.path.join(ROOT, TOKENIZER_PATH))
    class _TokWrapper:
        def __init__(self, t): self._t = t
        def encode(self, text):
            enc = self._t.encode(text)
            return enc.ids
        def decode(self, ids): return self._t.decode(ids)
        def get_vocab_size(self): return self._t.get_vocab_size()
    tokenizer = _TokWrapper(_raw_tok)
    print(f'Tokenizer: {TOKENIZER_PATH}  (vocab={tokenizer.get_vocab_size()})')

    # Load dataset (val only for PPL)
    cache = torch.load(os.path.join(ROOT, DATASET_PATH), weights_only=True)
    if isinstance(cache, dict):
        val_data = cache['val']
    else:
        n = len(cache)
        val_data = cache[int(n * 0.95):]
    print(f'Val data: {len(val_data)} sequences')

    # Load model
    model, offsets, train_m = load_model(args.arch, args.checkpoint, device)

    results = {
        'experiment':   f'probe_champion_{args.arch}',
        'checkpoint':   args.checkpoint,
        'architecture': args.arch,
        'J':            len(offsets),
        'offsets':      offsets,
        'eval_config': {
            'passkey_trials':  PASSKEY_TRIALS,
            'ppl_batches':     100,
            'passkey_distances': PASSKEY_DISTANCES,
        },
    }

    # ── Baseline ────────────────────────────────────────────────────────────────
    print('\n─── Baseline ───────────────────────────────────────────────────────────')
    t0 = time.time()
    pk_by_d, mean_pk = passkey_eval(model, tokenizer, device)
    baseline_ppl     = ppl_eval(model, val_data, device)
    print(f'  Passkey: {mean_pk:.1%}  |  PPL: {baseline_ppl}  '
          f'({time.time()-t0:.0f}s)')
    results['baseline'] = {
        'passkey': mean_pk,
        'passkey_by_d': {str(d): v for d, v in pk_by_d.items()},
        'ppl': baseline_ppl,
    }

    # ── Section 1: Offset importance ────────────────────────────────────────────
    print('\n─── Section 1: Offset Importance ───────────────────────────────────────')
    offset_results = {}
    for j_idx, delta in enumerate(offsets):
        t0 = time.time()
        j_tensor = torch.tensor([j_idx])
        with zero_offset_rows(model, j_tensor):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        pk_delta  = round(pk - mean_pk, 3)
        ppl_delta = round(ppl - baseline_ppl, 3)
        print(f'  δ={delta:5d}: passkey={pk:.1%} ({pk_delta:+.1%})  '
              f'ppl={ppl} ({ppl_delta:+.2f})  [{time.time()-t0:.0f}s]')
        offset_results[str(delta)] = {
            'j_idx': j_idx, 'passkey': pk, 'passkey_delta': pk_delta,
            'ppl': ppl, 'ppl_delta': ppl_delta,
            'passkey_by_d': {str(d): v for d, v in pk_d.items()},
        }
    results['offset_importance'] = offset_results

    # Group ablations
    print('\n  Group ablations:')
    groups = {}

    # {128, 384} co-dependent pair (if both present)
    pair = [o for o in offsets if o in (128, 384)]
    if len(pair) == 2:
        idxs = torch.tensor([offsets.index(o) for o in pair])
        with zero_offset_rows(model, idxs):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        groups['{128,384}'] = {
            'passkey': pk, 'passkey_delta': round(pk - mean_pk, 3),
            'ppl': ppl, 'ppl_delta': round(ppl - baseline_ppl, 3),
        }
        print(f'  {{128,384}}: passkey={pk:.1%} ({pk-mean_pk:+.1%})  ppl={ppl} ({ppl-baseline_ppl:+.2f})')

    # Dense [33-48] neighborhood (the d41 PPL advantage zone)
    dense_extra = [o for o in offsets if 33 <= o <= 48]
    if dense_extra:
        idxs = torch.tensor([offsets.index(o) for o in dense_extra])
        with zero_offset_rows(model, idxs):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        label = f'dense[33-48] (n={len(dense_extra)})'
        groups[label] = {
            'passkey': pk, 'passkey_delta': round(pk - mean_pk, 3),
            'ppl': ppl, 'ppl_delta': round(ppl - baseline_ppl, 3),
            'offsets_zeroed': dense_extra,
        }
        print(f'  {label}: passkey={pk:.1%} ({pk-mean_pk:+.1%})  ppl={ppl} ({ppl-baseline_ppl:+.2f})')

    # All sparse (>50) zeroed
    sparse = [o for o in offsets if o > 50]
    if sparse:
        idxs = torch.tensor([offsets.index(o) for o in sparse])
        with zero_offset_rows(model, idxs):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        label = f'all_sparse (n={len(sparse)})'
        groups[label] = {
            'passkey': pk, 'passkey_delta': round(pk - mean_pk, 3),
            'ppl': ppl, 'ppl_delta': round(ppl - baseline_ppl, 3),
            'offsets_zeroed': sparse,
        }
        print(f'  {label}: passkey={pk:.1%} ({pk-mean_pk:+.1%})  ppl={ppl} ({ppl-baseline_ppl:+.2f})')

    # Ultra-long only (≥512)
    ultralong = [o for o in offsets if o >= 512]
    if ultralong:
        idxs = torch.tensor([offsets.index(o) for o in ultralong])
        with zero_offset_rows(model, idxs):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        label = f'ultra-long (≥512, n={len(ultralong)})'
        groups[label] = {
            'passkey': pk, 'passkey_delta': round(pk - mean_pk, 3),
            'ppl': ppl, 'ppl_delta': round(ppl - baseline_ppl, 3),
            'offsets_zeroed': ultralong,
        }
        print(f'  {label}: passkey={pk:.1%} ({pk-mean_pk:+.1%})  ppl={ppl} ({ppl-baseline_ppl:+.2f})')

    results['group_ablations'] = groups

    # ── Section 2: Layer residual knockouts ─────────────────────────────────────
    print('\n─── Section 2: Layer Residual Knockouts ────────────────────────────────')
    layer_results = {}
    num_layers = len(model.blocks)
    for li in range(num_layers):
        t0 = time.time()
        with zero_layer_residual(model, li):
            pk_d, pk = passkey_eval(model, tokenizer, device)
            ppl      = ppl_eval(model, val_data, device, n_batches=50)
        pk_delta  = round(pk - mean_pk, 3)
        ppl_delta = round(ppl - baseline_ppl, 3)
        layer_type = 'FULL' if li == model.full_attn_layer else 'DSQG'
        print(f'  L{li} ({layer_type}) skip: passkey={pk:.1%} ({pk_delta:+.1%})  '
              f'ppl={ppl} ({ppl_delta:+.2f})  [{time.time()-t0:.0f}s]')
        layer_results[f'L{li}'] = {
            'type': layer_type,
            'passkey': pk, 'passkey_delta': pk_delta,
            'ppl': ppl, 'ppl_delta': ppl_delta,
            'passkey_by_d': {str(d): v for d, v in pk_d.items()},
        }
    results['layer_knockout'] = layer_results

    # ── Section 3: Head knockouts (penultimate DSQG + full-attn layers) ─────────
    print('\n─── Section 3: Head Knockouts ──────────────────────────────────────────')
    head_results = {}
    fa_layer   = model.full_attn_layer
    # Penultimate DSQG = layer before full-attn
    pen_layer  = fa_layer - 1
    target_layers = [pen_layer, fa_layer]

    H = None
    for block in model.blocks:
        attn = getattr(block, 'attn', None)
        if attn is not None:
            H = getattr(attn, 'num_heads', 8)
            break

    for li in target_layers:
        layer_type = 'FULL' if li == fa_layer else f'penultimate-DSQG'
        for hi in range(H):
            t0 = time.time()
            with zero_head(model, li, hi):
                pk_d, pk = passkey_eval(model, tokenizer, device)
                ppl      = ppl_eval(model, val_data, device, n_batches=30)
            pk_delta  = round(pk - mean_pk, 3)
            ppl_delta = round(ppl - baseline_ppl, 3)
            key = f'L{li}-H{hi}'
            print(f'  {key} ({layer_type}): passkey={pk:.1%} ({pk_delta:+.1%})  '
                  f'ppl={ppl} ({ppl_delta:+.2f})  [{time.time()-t0:.0f}s]')
            head_results[key] = {
                'layer': li, 'head': hi, 'layer_type': layer_type,
                'passkey': pk, 'passkey_delta': pk_delta,
                'ppl': ppl, 'ppl_delta': ppl_delta,
                'passkey_by_d': {str(d): v for d, v in pk_d.items()},
            }
    results['head_knockout'] = head_results

    # ── Section 5: EMA differential (b1 vs b3 isolation) ────────────────────────
    # Only runs on physics models with interference blocks (has_physics flag).
    has_physics = _ARCH_INFO[args.arch].get('has_physics', False) or \
                  _ARCH_INFO[args.arch]['model_cls'] in (
                      'AutoresearchTransformerPhysics', 'CurveTransformer')
    # Detect interference blocks
    if_block_layers = [li for li, blk in enumerate(model.blocks)
                       if hasattr(blk, 'ema_factor')]
    ema_results = {}
    if has_physics and len(if_block_layers) >= 1:
        print('\n─── Section 5: EMA Differential (b1 vs b3 isolation) ───────────────────')
        print(f'  Interference blocks at layers: {if_block_layers}')
        for b_idx, li in enumerate(if_block_layers):
            b_name = f'b{b_idx + 1}'  # b1=first IF block, b3=second (naming matches training logs)
            # 5a: zero only the EMA gate output (kills EMA/KV injection, keeps DSQG path)
            t0 = time.time()
            with zero_ema_block(model, b_idx) as actual_li:
                pk_d, pk = passkey_eval(model, tokenizer, device)
                ppl      = ppl_eval(model, val_data, device, n_batches=50)
            pk_delta  = round(pk - mean_pk, 3)
            ppl_delta = round(ppl - baseline_ppl, 3)
            key = f'{b_name}_ema_zeroed'
            print(f'  {b_name} (L{actual_li}) EMA-gate zeroed: '
                  f'passkey={pk:.1%} ({pk_delta:+.1%})  '
                  f'ppl={ppl} ({ppl_delta:+.2f})  [{time.time()-t0:.0f}s]')
            ema_results[key] = {
                'layer': actual_li, 'block_name': b_name,
                'ablation': 'ema_gate_zeroed',
                'passkey': pk, 'passkey_delta': pk_delta,
                'ppl': ppl, 'ppl_delta': ppl_delta,
                'passkey_by_d': {str(d): v for d, v in pk_d.items()},
            }
            # 5b: zero entire IF block residual (kills everything the block does)
            t0 = time.time()
            with zero_if_layer_output(model, b_idx) as actual_li:
                pk_d2, pk2 = passkey_eval(model, tokenizer, device)
                ppl2       = ppl_eval(model, val_data, device, n_batches=50)
            pk_delta2  = round(pk2 - mean_pk, 3)
            ppl_delta2 = round(ppl2 - baseline_ppl, 3)
            key2 = f'{b_name}_full_zeroed'
            print(f'  {b_name} (L{actual_li}) full block zeroed: '
                  f'passkey={pk2:.1%} ({pk_delta2:+.1%})  '
                  f'ppl={ppl2} ({ppl_delta2:+.2f})  [{time.time()-t0:.0f}s]')
            ema_results[key2] = {
                'layer': actual_li, 'block_name': b_name,
                'ablation': 'full_block_zeroed',
                'passkey': pk2, 'passkey_delta': pk_delta2,
                'ppl': ppl2, 'ppl_delta': ppl_delta2,
                'passkey_by_d': {str(d): v for d, v in pk_d2.items()},
            }
            # Interpretation hint
            ema_contribution = round(pk_delta2 - pk_delta, 3)
            print(f'    → EMA-only contribution to b{b_idx+1}: {ema_contribution:+.1%} passkey '
                  f'(full_zeroed − ema_zeroed)')
    results['ema_differential'] = ema_results

    # ── Section 6: IF layer position (L1-IF vs L3-IF independently) ─────────────
    if_position_results = {}
    if has_physics and len(if_block_layers) >= 1:
        print('\n─── Section 6: Interference Layer Position ─────────────────────────────')
        print(f'  Testing each IF block in isolation (full residual zero)')
        # Already probed individually in Section 5b, but here we report comparison
        # and add: what if we swap which layer has IF? (Too invasive — skip swap.)
        # Instead: report summary comparison across b1 vs b3 full knockouts.
        for b_idx, li in enumerate(if_block_layers):
            b_name = f'b{b_idx + 1}'
            full_key = f'{b_name}_full_zeroed'
            if full_key in ema_results:
                r = ema_results[full_key]
                print(f'  IF at L{li} ({b_name}) full knockout: '
                      f'{r["passkey_delta"]:+.1%} passkey, {r["ppl_delta"]:+.2f} PPL')
                if_position_results[f'L{li}_{b_name}'] = {
                    'layer': li, 'block_name': b_name,
                    'passkey_delta': r['passkey_delta'],
                    'ppl_delta': r['ppl_delta'],
                    'passkey': r['passkey'],
                    'ppl': r['ppl'],
                }
        # Summary: which IF position is more critical?
        if len(if_position_results) >= 2:
            items = sorted(if_position_results.items(),
                           key=lambda x: x[1]['passkey_delta'])
            print(f'  Most critical IF position: {items[0][0]} '
                  f'({items[0][1]["passkey_delta"]:+.1%} passkey)')
            print(f'  Less critical IF position: {items[-1][0]} '
                  f'({items[-1][1]["passkey_delta"]:+.1%} passkey)')
    results['if_position'] = if_position_results

    # ── Section 7: FA Delta Magnitude by Pre-FA Layer Depth ────────────────────
    fa_delta_results = {}
    multi_fa = _ARCH_INFO[args.arch].get('multi_fa', False)
    if not multi_fa and hasattr(model, 'full_attn_layer'):
        print('\n─── Section 7: FA Delta Magnitude by Pre-FA Layer Depth ─────────────────')
        fa_layer = model.full_attn_layer
        print(f'  FA layer: L{fa_layer} (pre-FA depth = {fa_layer} DSQG layers)')

        captured = {}

        def make_hook(name, capture_input=True, capture_output=True):
            def hook(module, inp, out):
                if capture_input:
                    x = inp[0] if isinstance(inp, tuple) else inp
                    captured[f'{name}_input'] = x.detach()
                if capture_output:
                    y = out[0] if isinstance(out, tuple) else out
                    captured[f'{name}_output'] = y.detach()
            return hook

        handles = []
        try:
            handles.append(model.blocks[0].register_forward_hook(
                make_hook('l0', capture_input=True, capture_output=False)))
            handles.append(model.blocks[fa_layer].register_forward_hook(
                make_hook('fa', capture_input=True, capture_output=True)))

            filler_ids = tokenizer.encode(_FILLER_SENTENCE)
            cue_ids = tokenizer.encode(_RETRIEVAL_CUE)
            batch_seqs = []
            for i in range(8):
                target = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
                intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
                d = 256
                filler = []
                while len(filler) < d:
                    filler.extend(filler_ids)
                full_seq = intro_ids + filler[:d] + cue_ids
                if len(full_seq) < MAX_SEQ_LEN:
                    batch_seqs.append(full_seq)

            max_len = max(len(s) for s in batch_seqs)
            padded = [s + [0] * (max_len - len(s)) for s in batch_seqs]
            batch = torch.tensor(padded, dtype=torch.long, device=device)

            with torch.no_grad():
                _ = model(batch)

            l0_input = captured.get('l0_input')
            fa_input = captured.get('fa_input')
            fa_output = captured.get('fa_output')

            if l0_input is not None and fa_input is not None and fa_output is not None:
                l0_input_norm = l0_input.norm(dim=-1).mean().item()
                fa_input_norm = fa_input.norm(dim=-1).mean().item()
                fa_output_norm = fa_output.norm(dim=-1).mean().item()
                fa_delta = fa_output - fa_input
                fa_delta_norm = fa_delta.norm(dim=-1).mean().item()
                fa_delta_ratio = fa_delta_norm / (fa_input_norm + 1e-8)
                preprocessing_ratio = fa_input_norm / (l0_input_norm + 1e-8)

                fa_delta_results = {
                    'fa_layer': fa_layer,
                    'pre_fa_depth': fa_layer,
                    'l0_input_norm': round(l0_input_norm, 4),
                    'fa_input_norm': round(fa_input_norm, 4),
                    'fa_output_norm': round(fa_output_norm, 4),
                    'fa_delta_norm': round(fa_delta_norm, 4),
                    'fa_delta_ratio': round(fa_delta_ratio, 4),
                    'preprocessing_ratio': round(preprocessing_ratio, 4),
                }

                print(f'  L0 input norm:      {l0_input_norm:.4f}')
                print(f'  FA input norm:      {fa_input_norm:.4f} (preprocessing ratio: {preprocessing_ratio:.2f}x)')
                print(f'  FA output norm:     {fa_output_norm:.4f}')
                print(f'  FA delta norm:      {fa_delta_norm:.4f}')
                print(f'  FA delta ratio:     {fa_delta_ratio:.4f} (delta/input)')
            else:
                print('  [WARN] Could not capture FA tensors')
        except Exception as e:
            print(f'  [ERROR] Section 7 failed: {e}')
        finally:
            for h in handles:
                h.remove()
    else:
        print('\n─── Section 7: FA Delta Magnitude (skipped: multi-FA model) ─────────────')
    results['fa_delta'] = fa_delta_results

    # ── Section 8: pos_bias Asymmetry ──────────────────────────────────────────
    print('\n─── Section 8: pos_bias Asymmetry ───────────────────────────────────────')
    pos_bias_results = {'per_layer': [], 'aggregate': {}}
    all_pb_norms = []
    all_se_norms = []

    for li, block in enumerate(model.blocks):
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        pb = getattr(attn, 'pos_bias', None)
        if pb is None:
            continue

        pb_data = pb.data
        pb_norm = pb_data.norm(dim=-1)
        J = pb_norm.shape[0]

        short_mask = torch.zeros(J, dtype=torch.bool, device=device)
        medium_mask = torch.zeros(J, dtype=torch.bool, device=device)
        long_mask = torch.zeros(J, dtype=torch.bool, device=device)

        for j_idx, delta in enumerate(offsets[:J]):
            if delta <= 10:
                short_mask[j_idx] = True
            elif delta <= 100:
                medium_mask[j_idx] = True
            else:
                long_mask[j_idx] = True

        layer_stats = {
            'layer': li,
            'pb_norm_mean': round(pb_norm.mean().item(), 4),
            'pb_norm_std': round(pb_norm.std().item(), 4),
            'pb_norm_max': round(pb_norm.max().item(), 4),
            'pb_norm_min': round(pb_norm.min().item(), 4),
            'specialization_ratio': round(pb_norm.max().item() / (pb_norm.min().item() + 1e-8), 2),
        }

        if short_mask.any():
            layer_stats['short_mean'] = round(pb_norm[short_mask].mean().item(), 4)
        if medium_mask.any():
            layer_stats['medium_mean'] = round(pb_norm[medium_mask].mean().item(), 4)
        if long_mask.any():
            layer_stats['long_mean'] = round(pb_norm[long_mask].mean().item(), 4)

        se = getattr(attn, 'scale_embed', None)
        if se is not None:
            se_data = se.data
            se_norm = se_data.norm(dim=-1)
            layer_stats['se_norm_mean'] = round(se_norm.mean().item(), 4)
            layer_stats['se_specialization'] = round(se_norm.max().item() / (se_norm.min().item() + 1e-8), 2)
            all_se_norms.append(se_norm)

        pos_bias_results['per_layer'].append(layer_stats)
        all_pb_norms.append(pb_norm)

        print(f'  L{li}: pb_norm={layer_stats["pb_norm_mean"]:.4f}±{layer_stats["pb_norm_std"]:.4f}, '
              f'spec_ratio={layer_stats["specialization_ratio"]:.1f}x')

    if all_pb_norms:
        stacked = torch.stack(all_pb_norms, dim=0)
        agg_mean = stacked.mean(dim=0)
        agg_spec = agg_mean.max().item() / (agg_mean.min().item() + 1e-8)
        pos_bias_results['aggregate'] = {
            'mean_norm_per_offset': [round(v.item(), 4) for v in agg_mean],
            'aggregate_specialization': round(agg_spec, 2),
        }
        top_idx = agg_mean.argmax().item()
        bot_idx = agg_mean.argmin().item()
        print(f'  Aggregate: spec_ratio={agg_spec:.1f}x, '
              f'top_offset=δ{offsets[top_idx]} (idx={top_idx}), '
              f'bot_offset=δ{offsets[bot_idx]} (idx={bot_idx})')
    results['pos_bias_asymmetry'] = pos_bias_results

    # ── Section 9: Scale_embed Per-Offset Contribution ─────────────────────────
    print('\n─── Section 9: Scale_embed Per-Offset Contribution ──────────────────────')
    scale_embed_results = {'per_layer': [], 'aggregate': {}, 'top5_by_magnitude': [], 'bottom5_by_magnitude': []}
    all_se_mags = []

    for li, block in enumerate(model.blocks):
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        se = getattr(attn, 'scale_embed', None)
        if se is None:
            continue

        se_data = se.data
        se_magnitude = se_data.abs().mean(dim=-1)
        J = se_magnitude.shape[0]

        layer_data = {
            'layer': li,
            'magnitude_per_offset': [round(v.item(), 4) for v in se_magnitude],
        }
        scale_embed_results['per_layer'].append(layer_data)
        all_se_mags.append(se_magnitude)

    if all_se_mags:
        stacked = torch.stack(all_se_mags, dim=0)
        agg_mag = stacked.mean(dim=0)

        offset_mag_pairs = [(offsets[j], agg_mag[j].item()) for j in range(min(len(offsets), len(agg_mag)))]
        sorted_pairs = sorted(offset_mag_pairs, key=lambda x: -x[1])

        top5 = sorted_pairs[:5]
        bottom5 = sorted_pairs[-5:]

        scale_embed_results['aggregate'] = {
            'mean_magnitude_per_offset': {str(offsets[j]): round(agg_mag[j].item(), 4)
                                           for j in range(min(len(offsets), len(agg_mag)))}
        }
        scale_embed_results['top5_by_magnitude'] = [
            {'delta': d, 'magnitude': round(m, 4)} for d, m in top5
        ]
        scale_embed_results['bottom5_by_magnitude'] = [
            {'delta': d, 'magnitude': round(m, 4)} for d, m in bottom5
        ]

        print(f'  Top scale_embed offsets: ' +
              ', '.join(f'δ={d} (mag={m:.4f})' for d, m in top5))
        print(f'  Bottom scale_embed offsets: ' +
              ', '.join(f'δ={d} (mag={m:.4f})' for d, m in bottom5))

        if 'offset_importance' in results:
            sorted_by_pk = sorted(
                results['offset_importance'].items(),
                key=lambda x: x[1]['passkey_delta']
            )
            critical_offsets = [int(k) for k, _ in sorted_by_pk[:5]]
            print(f'  Section 1 critical offsets for comparison: {critical_offsets}')
            scale_embed_results['section1_critical_offsets'] = critical_offsets
    else:
        print('  [WARN] No scale_embed found in model')
    results['scale_embed_analysis'] = scale_embed_results

    # ── Section 10: IF Block Inference Removability (Self-Erasing Scaffold) ────
    if_removability_results = {}
    if has_physics and len(if_block_layers) >= 1:
        print('\n─── Section 10: IF Block Inference Removability ─────────────────────────')
        print(f'  Testing self-erasing scaffold hypothesis for {len(if_block_layers)} IF blocks')

        for b_idx, li in enumerate(if_block_layers):
            b_name = f'b{b_idx + 1}'
            block = model.blocks[li]
            block_results = {'layer': li, 'block_name': b_name}

            ema_key = f'{b_name}_ema_zeroed'
            if ema_key in ema_results:
                block_results['ema_gate_zero'] = {
                    'passkey': ema_results[ema_key]['passkey'],
                    'ppl': ema_results[ema_key]['ppl'],
                    'passkey_delta': ema_results[ema_key]['passkey_delta'],
                    'ppl_delta': ema_results[ema_key]['ppl_delta'],
                }

            full_key = f'{b_name}_full_zeroed'
            if full_key in ema_results:
                block_results['full_block_zero'] = {
                    'passkey': ema_results[full_key]['passkey'],
                    'ppl': ema_results[full_key]['ppl'],
                    'passkey_delta': ema_results[full_key]['passkey_delta'],
                    'ppl_delta': ema_results[full_key]['ppl_delta'],
                }

            saved_weights = {}
            inter_params = ['inter_norm', 'inter_gate', 'inter_k_proj', 'inter_v_proj']
            try:
                for param_name in inter_params:
                    if hasattr(block, param_name):
                        param = getattr(block, param_name)
                        if hasattr(param, 'weight'):
                            saved_weights[f'{param_name}.weight'] = param.weight.data.clone()
                            param.weight.data.zero_()
                        if hasattr(param, 'bias') and param.bias is not None:
                            saved_weights[f'{param_name}.bias'] = param.bias.data.clone()
                            param.bias.data.zero_()

                if saved_weights:
                    t0 = time.time()
                    pk_d, pk = passkey_eval(model, tokenizer, device)
                    ppl = ppl_eval(model, val_data, device, n_batches=50)
                    pk_delta = round(pk - mean_pk, 3)
                    ppl_delta = round(ppl - baseline_ppl, 3)

                    block_results['weights_zeroed'] = {
                        'passkey': pk,
                        'ppl': ppl,
                        'passkey_delta': pk_delta,
                        'ppl_delta': ppl_delta,
                        'params_zeroed': list(saved_weights.keys()),
                    }
                    print(f'  {b_name} (L{li}) weights zeroed: '
                          f'passkey={pk:.1%} ({pk_delta:+.1%})  '
                          f'ppl={ppl} ({ppl_delta:+.2f})  [{time.time()-t0:.0f}s]')
                else:
                    print(f'  {b_name} (L{li}): no inter_* params found to zero')

            finally:
                for key, val in saved_weights.items():
                    parts = key.split('.')
                    param_name, attr = parts[0], parts[1]
                    if hasattr(block, param_name):
                        param = getattr(block, param_name)
                        if attr == 'weight':
                            param.weight.data.copy_(val)
                        elif attr == 'bias' and param.bias is not None:
                            param.bias.data.copy_(val)

            ema_delta = block_results.get('ema_gate_zero', {}).get('passkey_delta', 0)
            full_delta = block_results.get('full_block_zero', {}).get('passkey_delta', 0)
            weights_delta = block_results.get('weights_zeroed', {}).get('passkey_delta', 0)

            if abs(full_delta) < 0.05 and abs(ema_delta) < 0.05 and abs(weights_delta) < 0.05:
                scaffold_status = 'fully_self_erasing'
            elif abs(full_delta) < 0.10:
                scaffold_status = 'mostly_self_erasing'
            else:
                scaffold_status = 'still_active'
            block_results['scaffold_status'] = scaffold_status
            print(f'    → Scaffold status: {scaffold_status}')

            if_removability_results[b_name] = block_results
    else:
        print('\n─── Section 10: IF Block Removability (skipped: no physics/IF blocks) ──')
    results['if_removability'] = if_removability_results

    # ── Summary ──────────────────────────────────────────────────────────────────
    print('\n─── Summary ────────────────────────────────────────────────────────────')

    # Most critical offsets by passkey delta
    sorted_by_pk = sorted(
        offset_results.items(),
        key=lambda x: x[1]['passkey_delta']
    )
    print('  Most critical offsets (by passkey loss):')
    for delta_str, r in sorted_by_pk[:5]:
        print(f'    δ={delta_str}: {r["passkey_delta"]:+.1%} passkey, '
              f'{r["ppl_delta"]:+.2f} PPL')

    # Most critical layers
    sorted_layers = sorted(
        layer_results.items(),
        key=lambda x: x[1]['passkey_delta']
    )
    print('  Most critical layers (by passkey loss):')
    for lk, lr in sorted_layers[:3]:
        print(f'    {lk} ({lr["type"]}): {lr["passkey_delta"]:+.1%} passkey, '
              f'{lr["ppl_delta"]:+.2f} PPL')

    results['summary'] = {
        'baseline_passkey': mean_pk,
        'baseline_ppl': baseline_ppl,
        'top5_critical_offsets': [
            {'delta': k, **{kk: vv for kk, vv in v.items()
                            if kk in ('passkey_delta', 'ppl_delta', 'passkey')}}
            for k, v in sorted_by_pk[:5]
        ],
        'most_critical_layer': sorted_layers[0][0] if sorted_layers else None,
    }

    # Save
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Champion model ablation probe')
    parser.add_argument('--arch',       required=True, choices=sorted(_ARCH_INFO.keys()),
                        help='Model architecture: d41_35m, condx_v2')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint (relative to DWARF root)')
    parser.add_argument('--out',        required=True,
                        help='Output JSON path')
    args = parser.parse_args()

    os.chdir(ROOT)
    run_probe(args)
