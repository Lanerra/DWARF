"""
⚗️ Checkpoint ablation probe — condU_v5 35M

Systematic ablation studies on the condU_v5 35M hybrid checkpoint to
understand which components drive PPL and passkey retrieval.

Ablation suite:
  1. Offset importance (44 offsets: 0–32 local + 11 dyadic sparse)
  2. Head knockout (per layer, per head)
  3. Layer residual knockout (per layer)
  4. Interference component ablations (inter_gate / inter_k_proj / inter_v_proj)
  5. scale_embed amplitude sweep
  6. pos_bias amplitude sweep
  7. if_gain ablations

Run on 3090:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_condU_v5_35m.py
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE         = "cuda"
CHECKPOINT     = os.path.join(REPO_ROOT, "checkpoints", "condU_v5", "best.pt")
ENCODED_CACHE  = os.path.join(REPO_ROOT, "logs", "fineweb_encoded_2048.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "results", "2048_condI_tokenizer.json")
RESULT_FILE    = os.path.join(REPO_ROOT, "benchmarks", "logs",
                              "probe_condU_v5_35m_results.json")

PASSKEY_TRIALS   = 5
EVAL_BATCH_SIZE  = 4

# ── Import model components from training script ─────────────────────────────

sys.path.insert(0, os.path.join(REPO_ROOT, 'kernels'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'kernels', 'dsqg_cuda'))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    'train_condU_v5',
    os.path.join(REPO_ROOT, 'train', 'train_2048_condU_v5.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CondUV5Transformer = _mod.CondUV5Transformer
BPETokenizerWrapper = _mod.BPETokenizerWrapper
DSQGBlockV5 = _mod.DSQGBlockV5
FullAttentionBlock = _mod.FullAttentionBlock

EMBEDDING_DIM   = _mod.EMBEDDING_DIM
NUM_LAYERS      = _mod.NUM_LAYERS
NUM_HEADS       = _mod.NUM_HEADS
FFN_DIM         = _mod.FFN_DIM
MAX_SEQ_LEN     = _mod.MAX_SEQ_LEN
FULL_ATTN_LAYER = _mod.FULL_ATTN_LAYER
INTERFERENCE    = _mod.INTERFERENCE
ALL_OFFSETS     = _mod._COND_N_OFFSETS

PASSKEY_DISTANCES = _mod.PASSKEY_DISTANCES
_PASSKEY_WORDS    = _mod._PASSKEY_WORDS
_FILLER_SENTENCE  = _mod._FILLER_SENTENCE
_INTRO_TEMPLATE   = _mod._INTRO_TEMPLATE
_RETRIEVAL_CUE    = _mod._RETRIEVAL_CUE

HEAD_DIM = EMBEDDING_DIM // NUM_HEADS

# phase_base / phase_gain are sparse-only: cover indices 33..43 of ALL_OFFSETS
SPARSE_START   = 33
SPARSE_OFFSETS = ALL_OFFSETS[SPARSE_START:]
assert len(SPARSE_OFFSETS) == 11


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ppl(model, data, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size + 1, batch_size):
        x = data[i:i + batch_size, :-1].to(DEVICE)
        y = data[i:i + batch_size, 1:].to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    cross_entropy = total_loss / max(total_tokens, 1)
    return math.exp(min(cross_entropy, 20))


@torch.no_grad()
def evaluate_passkey(model, tokenizer, trials=PASSKEY_TRIALS):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for distance in PASSKEY_DISTANCES:
        correct, n_valid = 0, 0
        for i in range(trials):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if distance > available:
                continue
            filler = []
            while len(filler) < distance:
                filler.extend(filler_ids)
            full_sequence = intro_ids + filler[:distance] + cue_ids
            if len(full_sequence) >= MAX_SEQ_LEN:
                continue
            ids    = torch.tensor([full_sequence], dtype=torch.long, device=DEVICE)
            logits = model(ids)[:, -1, :]
            candidates = [target] + others[:9]
            candidate_ids = [
                (tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                for w in candidates
            ]
            predicted_index = logits[0][candidate_ids].argmax().item()
            correct += int(candidates[predicted_index] == target)
            n_valid += 1
        results[distance] = correct / n_valid if n_valid else 0.0
    return results


def run_eval(model, val_data, tokenizer, label):
    """Run passkey + val PPL, print progress, return result dict."""
    t0       = time.time()
    passkey  = evaluate_passkey(model, tokenizer)
    passkey_mean = sum(passkey.values()) / len(passkey) if passkey else 0.0
    ppl      = evaluate_ppl(model, val_data)
    elapsed  = time.time() - t0
    print(f'  {label:<55s}  pk={passkey_mean * 100:5.1f}%  '
          f'ppl={ppl:7.2f}  [{elapsed:.0f}s]')
    sys.stdout.flush()
    return {
        'label':               label,
        'passkey_mean':        passkey_mean,
        'passkey_by_distance': {str(d): v for d, v in passkey.items()},
        'val_ppl':             ppl,
        'elapsed_s':           elapsed,
    }


# ── State management ──────────────────────────────────────────────────────────

class StateManager:
    """Snapshot and restore model state dict for clean per-test ablation."""

    def __init__(self, model):
        self._snapshot = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self._model = model

    def restore(self):
        self._model.load_state_dict(self._snapshot, strict=True)
        self._model.eval()


# ── Layer introspection ───────────────────────────────────────────────────────

def get_dsqg_layers(model):
    return [(i, block) for i, block in enumerate(model.blocks)
            if isinstance(block, DSQGBlockV5)]


def get_interference_layers(model):
    return [(i, block) for i, block in enumerate(model.blocks)
            if isinstance(block, DSQGBlockV5) and block.interference]


# ── Ablation 1: Offset importance ─────────────────────────────────────────────

def ablation_offset_importance(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ① Offset importance (J=44 offsets)')
    print('=' * 76)
    dsqg = get_dsqg_layers(model)
    results = []

    for j in range(len(ALL_OFFSETS)):
        state.restore()
        for _, block in dsqg:
            block.attn.pos_bias.data[j, :] = 0.0
            block.attn.scale_embed.data[j, :] = 0.0
            if j >= SPARSE_START:
                sparse_index = j - SPARSE_START
                block.attn.phase_base.data[sparse_index, :, :] = 0.0
                block.attn.phase_gain.data[sparse_index, :, :] = 0.0
        row = run_eval(model, val_data, tokenizer,
                       f'zero offset[{j}] = {ALL_OFFSETS[j]}')
        row['offset_index'] = j
        row['offset_value'] = ALL_OFFSETS[j]
        results.append(row)

    sparse_groups = {
        '{128}':          [128],
        '{384}':          [384],
        '{1536}':         [1536],
        '{128,384}':      [128, 384],
        '{128,384,1536}': [128, 384, 1536],
    }
    for group_name, offset_values in sparse_groups.items():
        state.restore()
        indices = [ALL_OFFSETS.index(v) for v in offset_values]
        for _, block in dsqg:
            for j in indices:
                block.attn.pos_bias.data[j, :] = 0.0
                block.attn.scale_embed.data[j, :] = 0.0
                if j >= SPARSE_START:
                    sparse_index = j - SPARSE_START
                    block.attn.phase_base.data[sparse_index, :, :] = 0.0
                    block.attn.phase_gain.data[sparse_index, :, :] = 0.0
        row = run_eval(model, val_data, tokenizer,
                       f'zero sparse group {group_name}')
        row['group'] = group_name
        row['offset_values'] = offset_values
        results.append(row)

    state.restore()
    return results


# ── Ablation 2: Head knockout ────────────────────────────────────────────────

def ablation_head_knockout(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ② Head knockout (per layer, per head)')
    print('=' * 76)
    results = []

    for layer_index in range(NUM_LAYERS):
        block = model.blocks[layer_index]
        for head_index in range(NUM_HEADS):
            state.restore()
            if isinstance(block, DSQGBlockV5):
                block.attn.if_gain.data[head_index] = 0.0
            elif isinstance(block, FullAttentionBlock):
                h_start = head_index * HEAD_DIM
                h_end   = (head_index + 1) * HEAD_DIM
                block.attn.out_proj.weight.data[:, h_start:h_end] = 0.0
            row = run_eval(model, val_data, tokenizer,
                           f'L{layer_index} H{head_index} knockout')
            row['layer'] = layer_index
            row['head']  = head_index
            results.append(row)

    state.restore()
    return results


# ── Ablation 3: Layer residual knockout ───────────────────────────────────────

def ablation_layer_residual_knockout(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ③ Layer residual knockout')
    print('=' * 76)
    results = []

    for layer_index in range(NUM_LAYERS):
        state.restore()

        def _skip_hook(_module, _input, _output):
            return _input[0]

        handle = model.blocks[layer_index].register_forward_hook(_skip_hook)
        row    = run_eval(model, val_data, tokenizer,
                          f'skip layer {layer_index}')
        handle.remove()
        row['layer'] = layer_index
        results.append(row)

    state.restore()
    return results


# ── Ablation 4: Interference component ablations ─────────────────────────────

def ablation_interference_components(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ④ Interference component ablations')
    print('=' * 76)
    interference_layers = get_interference_layers(model)
    if not interference_layers:
        print('  (no interference layers found)')
        return []

    print(f'  interference layers: {[i for i, _ in interference_layers]}')
    results = []

    ablation_specs = [
        ('zero inter_gate (gate→0.5)',   ['inter_gate']),
        ('zero inter_k_proj (no K inj)', ['inter_k_proj']),
        ('zero inter_v_proj (no V inj)', ['inter_v_proj']),
        ('zero all IF',                  ['inter_gate', 'inter_k_proj',
                                          'inter_v_proj']),
    ]

    for layer_index, block in interference_layers:
        for ablation_name, modules_to_zero in ablation_specs:
            state.restore()
            for module_name in modules_to_zero:
                module = getattr(block, module_name)
                module.weight.data.zero_()
                if module.bias is not None:
                    module.bias.data.zero_()
            row = run_eval(model, val_data, tokenizer,
                           f'L{layer_index} {ablation_name}')
            row['layer']    = layer_index
            row['ablation'] = ablation_name
            results.append(row)

    state.restore()
    return results


# ── Ablation 5: scale_embed amplitude sweep ──────────────────────────────────

def ablation_scale_embed_sweep(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ⑤ scale_embed amplitude sweep')
    print('=' * 76)
    dsqg   = get_dsqg_layers(model)
    scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    results = []

    for scale in scales:
        state.restore()
        if scale != 1.0:
            for _, block in dsqg:
                block.attn.scale_embed.data.mul_(scale)
        row = run_eval(model, val_data, tokenizer,
                       f'scale_embed x {scale:.2f}')
        row['scale'] = scale
        results.append(row)

    state.restore()
    return results


# ── Ablation 6: pos_bias amplitude sweep ─────────────────────────────────────

def ablation_pos_bias_sweep(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ⑥ pos_bias amplitude sweep')
    print('=' * 76)
    dsqg   = get_dsqg_layers(model)
    scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    results = []

    for scale in scales:
        state.restore()
        if scale != 1.0:
            for _, block in dsqg:
                block.attn.pos_bias.data.mul_(scale)
        row = run_eval(model, val_data, tokenizer,
                       f'pos_bias x {scale:.2f}')
        row['scale'] = scale
        results.append(row)

    state.restore()
    return results


# ── Ablation 7: if_gain ablations ─────────────────────────────────────────────

def ablation_if_gain(model, state, val_data, tokenizer):
    print('\n' + '=' * 76)
    print('  ⑦ if_gain ablations')
    print('=' * 76)
    dsqg = get_dsqg_layers(model)
    results = []

    state.restore()
    for _, block in dsqg:
        block.attn.if_gain.data.fill_(1.0)
    row = run_eval(model, val_data, tokenizer,
                   'if_gain = 1.0 (uniform, no learned amplification)')
    row['ablation'] = 'uniform_1.0'
    results.append(row)

    state.restore()
    for _, block in dsqg:
        block.attn.if_gain.data.zero_()
    row = run_eval(model, val_data, tokenizer,
                   'if_gain = 0.0 (kill all DSQG head outputs)')
    row['ablation'] = 'zero'
    results.append(row)

    state.restore()
    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(title, rows, baseline):
    if not rows:
        return
    base_pk  = baseline['passkey_mean']
    base_ppl = baseline['val_ppl']
    print(f'\n{"─" * 76}')
    print(f'  {title}')
    print(f'{"─" * 76}')
    print(f'  {"Label":<55s}  {"pk%":>5s}  {"Δpk":>6s}  '
          f'{"ppl":>7s}  {"Δppl":>7s}')
    print(f'  {"─" * 55}  {"─" * 5}  {"─" * 6}  {"─" * 7}  {"─" * 7}')
    for row in rows:
        pk   = row['passkey_mean'] * 100
        dpk  = (row['passkey_mean'] - base_pk) * 100
        ppl  = row['val_ppl']
        dppl = row['val_ppl'] - base_ppl
        print(f'  {row["label"]:<55s}  {pk:5.1f}  {dpk:+5.1f}  '
              f'{ppl:7.2f}  {dppl:+7.2f}')
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=' * 76)
    print('  ⚗️  Checkpoint ablation probe — condU_v5 35M')
    print('=' * 76)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({props.total_memory / 1e9:.1f} GB)')
    print(f'  Checkpoint:      {CHECKPOINT}')
    print(f'  Architecture:    D={EMBEDDING_DIM}, H={NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Offsets:         {len(ALL_OFFSETS)} total '
          f'({SPARSE_START} local + {len(SPARSE_OFFSETS)} sparse)')
    print(f'  Passkey config:  {PASSKEY_TRIALS} trials, '
          f'{len(PASSKEY_DISTANCES)} distances')
    print()

    # ── Load tokenizer ────────────────────────────────────────────────────
    from tokenizers import Tokenizer
    assert os.path.exists(TOKENIZER_PATH), \
        f'tokenizer not found: {TOKENIZER_PATH}'
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))
    print(f'  Tokenizer:  {TOKENIZER_PATH}  (vocab={tokenizer.vocab_size()})')

    # ── Load pre-encoded data ─────────────────────────────────────────────
    assert os.path.exists(ENCODED_CACHE), \
        f'encoded cache not found: {ENCODED_CACHE}'
    cache     = torch.load(ENCODED_CACHE, weights_only=True)
    val_data  = cache['val']
    test_data = cache['test']
    print(f'  Val: {len(val_data):,} seqs  |  Test: {len(test_data):,} seqs')

    # ── Build model and load checkpoint ───────────────────────────────────
    model = CondUV5Transformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
    ).to(DEVICE)

    state_dict = torch.load(CHECKPOINT, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  Model:      {model.param_count():,} params loaded')

    # ── Introspect layer structure ────────────────────────────────────────
    dsqg_layer_indices        = [i for i, _ in get_dsqg_layers(model)]
    interference_layer_indices = [i for i, _ in get_interference_layers(model)]
    layer_types = []
    for i, block in enumerate(model.blocks):
        if isinstance(block, FullAttentionBlock):
            layer_types.append('FULL')
        elif isinstance(block, DSQGBlockV5) and block.interference:
            layer_types.append('DSQG+IF')
        else:
            layer_types.append('DSQG')
    print(f'  Layers:     {layer_types}')
    print(f'  DSQG:       {dsqg_layer_indices}')
    print(f'  IF layers:  {interference_layer_indices}')

    # ── Snapshot state for restoration ────────────────────────────────────
    state = StateManager(model)

    # ── Baseline (unmodified checkpoint) ──────────────────────────────────
    print('\n' + '=' * 76)
    print('  Baseline (unmodified checkpoint)')
    print('=' * 76)
    baseline = run_eval(model, val_data, tokenizer, 'baseline')

    pk_parts = '  '.join(
        f'd={d}:{int(baseline["passkey_by_distance"][str(d)] * 100)}%'
        for d in PASSKEY_DISTANCES)
    print(f'  passkey breakdown: {pk_parts}')

    # ── Run all ablation groups ───────────────────────────────────────────
    abl_1 = ablation_offset_importance(model, state, val_data, tokenizer)
    abl_2 = ablation_head_knockout(model, state, val_data, tokenizer)
    abl_3 = ablation_layer_residual_knockout(model, state, val_data, tokenizer)
    abl_4 = ablation_interference_components(model, state, val_data, tokenizer)
    abl_5 = ablation_scale_embed_sweep(model, state, val_data, tokenizer)
    abl_6 = ablation_pos_bias_sweep(model, state, val_data, tokenizer)
    abl_7 = ablation_if_gain(model, state, val_data, tokenizer)

    # ── Summary tables ────────────────────────────────────────────────────
    print_summary_table('① Offset importance',           abl_1, baseline)
    print_summary_table('② Head knockout',               abl_2, baseline)
    print_summary_table('③ Layer residual knockout',     abl_3, baseline)
    print_summary_table('④ Interference components',     abl_4, baseline)
    print_summary_table('⑤ scale_embed amplitude sweep', abl_5, baseline)
    print_summary_table('⑥ pos_bias amplitude sweep',    abl_6, baseline)
    print_summary_table('⑦ if_gain ablations',           abl_7, baseline)

    # ── Final test PPL (unmodified) ───────────────────────────────────────
    state.restore()
    test_ppl = evaluate_ppl(model, test_data)
    print(f'\n  Test PPL (unmodified): {test_ppl:.3f}')

    total_time = time.time() - t_start
    print(f'  Total runtime: {total_time:.0f}s ({total_time / 3600:.1f}h)')

    # ── Save results JSON ─────────────────────────────────────────────────
    full_results = {
        'experiment': 'probe_condU_v5_35m',
        'checkpoint': CHECKPOINT,
        'architecture': {
            'embedding_dim':   EMBEDDING_DIM,
            'num_layers':      NUM_LAYERS,
            'num_heads':       NUM_HEADS,
            'ffn_dim':         FFN_DIM,
            'head_dim':        HEAD_DIM,
            'full_attn_layer': FULL_ATTN_LAYER,
            'interference':    INTERFERENCE,
            'offsets':         ALL_OFFSETS,
            'layer_types':     layer_types,
        },
        'eval_config': {
            'passkey_trials':    PASSKEY_TRIALS,
            'passkey_distances': PASSKEY_DISTANCES,
            'eval_batch_size':   EVAL_BATCH_SIZE,
        },
        'baseline':                 baseline,
        'test_ppl':                 test_ppl,
        'offset_importance':        abl_1,
        'head_knockout':            abl_2,
        'layer_residual_knockout':  abl_3,
        'interference_components':  abl_4,
        'scale_embed_sweep':        abl_5,
        'pos_bias_sweep':           abl_6,
        'if_gain_ablations':        abl_7,
        'total_runtime_s':          total_time,
    }

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(full_results, fp, indent=2)
    print(f'  Results → {RESULT_FILE}')


if __name__ == '__main__':
    main()
