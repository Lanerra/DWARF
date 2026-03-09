"""
probe_bypass_gate.py — Gate sweep on condX checkpoint (3090)

For each gate value g in the sweep:
    q_input = g * clean_residual + (1 - g) * x

    g=0.00 → pure contaminated Q  (condV-like — weights weren't trained here)
    g=0.10 → condX-v2 converged value
    g=1.00 → condX hard bypass   (weights were trained here)

Monkey-patches FullCausalAttentionBypass.forward at inference time.
Passkey eval: 10 trials/distance (vs 5 in training) for lower noise.

Run on 3090 while condX-v2 bakes on 4090:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/probe_bypass_gate.py
"""

import json, math, os, sys, time
import torch
import torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo, 'kernels'))
sys.path.insert(0, os.path.join(_repo, 'train'))

# ── Gate sweep values ─────────────────────────────────────────────────────────
GATE_VALUES = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20,
               0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

CHECKPOINT   = 'checkpoints/condX_v2/best.pt'
RESULT_FILE  = 'benchmarks/logs/probe_bypass_gate_condX_v2_results.json'
TRIALS       = 10   # per distance (2× training for lower noise)

# ── Passkey eval config (identical to training scripts) ────────────────────────
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
MAX_SEQ_LEN       = 2048

# ── Model hyperparams (condX 13M) ─────────────────────────────────────────────
EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5
VOCAB_SIZE      = 32000


# ══════════════════════════════════════════════════════════════════════════════
# Import condX model (reuses training script classes)
# ══════════════════════════════════════════════════════════════════════════════

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'condX_v2_train', os.path.join(_repo, 'train', 'train_2048_condX_v2.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CondXTransformer         = _mod.CondXTransformer
FullCausalAttentionBypass = _mod.FullCausalAttentionBypass


# ══════════════════════════════════════════════════════════════════════════════
# Gate-aware forward patch
# ══════════════════════════════════════════════════════════════════════════════

# We monkey-patch the forward method on the specific block instance (not the
# class) so other model instances are unaffected.

def _make_gated_forward(orig_q_proj, orig_kv_proj, orig_out_proj,
                        orig_gate_proj, dropout_p, num_heads, head_dim,
                        gate_ref):
    """Return a forward function that blends clean and full residual by gate_ref[0]."""

    def gated_forward(self, x, clean_residual=None):
        B, N, D = x.shape
        H, HD   = num_heads, head_dim
        g       = gate_ref[0]   # read current gate value from shared list

        if clean_residual is not None and g < 1.0:
            q_input = g * clean_residual + (1.0 - g) * x
        elif clean_residual is not None:
            q_input = clean_residual     # gate=1.0 → original condX hard bypass
        else:
            q_input = x                  # no bypass path available

        q = orig_q_proj(q_input)
        k, v = orig_kv_proj(x).split(D, dim=-1)

        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0.0, is_causal=True)   # no dropout at eval

        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(orig_gate_proj(x))
        return orig_out_proj(out_flat * gate)

    return gated_forward


def patch_model_gate(model, gate_ref):
    """Patch the full-attention bypass block to read gate_ref[0] at call time."""
    bypass_block = None
    for block in model.blocks:
        if isinstance(block, _mod.FullAttentionBypassBlock):
            bypass_block = block
            break
    assert bypass_block is not None, 'No FullAttentionBypassBlock found!'

    attn = bypass_block.attn
    assert isinstance(attn, FullCausalAttentionBypass)

    # Bind a gated forward that closes over the module's projection layers
    # and gate_ref (a mutable list so we can update the gate between sweeps)
    gated_fwd = _make_gated_forward(
        attn.q_proj, attn.kv_proj, attn.out_proj, attn.gate_proj,
        attn.dropout_p, attn.num_heads, attn.head_dim, gate_ref)

    import types
    attn.forward = types.MethodType(gated_fwd, attn)
    return attn   # return so we can confirm


# ══════════════════════════════════════════════════════════════════════════════
# Passkey evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_passkey_sweep(model, tokenizer, device, distances, num_trials=TRIALS):
    """
    Evaluate passkey accuracy across distances.
    Exact match to the training script's passkey_accuracy():
      - filler[:d] → d is in TOKENS (not sentences)
      - 10-way forced choice among candidate words (10% chance baseline)
    """
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)

    # Pre-build a long filler buffer to slice from
    filler_buf = filler_ids * (max(distances) // len(filler_ids) + 2)

    results = {}
    for d in distances:
        correct = 0; n_valid = 0
        for i in range(num_trials):
            target   = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others   = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available:
                continue
            full_seq = intro_ids + filler_buf[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(ids)[:, -1, :]          # (1, V)
            # 10-way forced choice — first token of each candidate word
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            pred_word = ([target] + others[:9])[
                logits[0][cand_ids].argmax().item()]
            correct += int(pred_word == target)
            n_valid += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  probe_bypass_gate_v2.py — condX-v2 checkpoint gate sweep')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Checkpoint: {CHECKPOINT}')
    print(f'  Gate values: {GATE_VALUES}')
    print(f'  Trials/distance: {TRIALS}')
    print(f'  Distances: {PASSKEY_DISTANCES}')
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    _tok_candidates = [
        os.path.join(_repo, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_repo, 'benchmarks', 'results', '2048_condI_tokenizer.json'),
    ]
    from train_2048_condX_v2 import BPETokenizerWrapper
    from tokenizers import Tokenizer
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    assert tok_path, 'condI tokenizer not found'
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'  Tokenizer: {tok_path}  (vocab={tokenizer.vocab_size()})')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CondXTransformer(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=INTERFERENCE,
    ).to(device)

    ckpt_path = os.path.join(_repo, CHECKPOINT)
    sd = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f'  Loaded checkpoint: {CHECKPOINT}  ({model.param_count():,} params)')

    # ── Patch gate into model ─────────────────────────────────────────────────
    gate_ref     = [1.0]    # mutable container — closure reads gate_ref[0]
    patched_attn = patch_model_gate(model, gate_ref)
    print(f'  Patched FullCausalAttentionBypass — gate controlled by gate_ref')
    print()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_results = []

    for gate_val in GATE_VALUES:
        gate_ref[0] = gate_val   # update gate for this sweep point

        t0 = time.time()
        by_distance = run_passkey_sweep(model, tokenizer, device,
                                        PASSKEY_DISTANCES, TRIALS)

        mean_acc = sum(by_distance.values()) / len(by_distance)
        above_50 = sum(1 for v in by_distance.values() if v > 0.5)
        elapsed  = time.time() - t0

        row = {
            'gate':        gate_val,
            'mean_passkey': mean_acc,
            'above_50':    above_50,
            'by_distance': by_distance,
            'elapsed_s':   elapsed,
        }
        sweep_results.append(row)

        dist_str = '  '.join(
            f'd={d}:{int(acc*100)}%' for d, acc in sorted(by_distance.items()))
        print(f'gate={gate_val:.2f}  mean={mean_acc*100:.1f}%  '
              f'({above_50}/12 >50%)  [{elapsed:.0f}s]')
        print(f'  {dist_str}')

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        'experiment':   'probe_bypass_gate_condX_v2',
        'checkpoint':   CHECKPOINT,
        'trials':       TRIALS,
        'distances':    PASSKEY_DISTANCES,
        'gate_values':  GATE_VALUES,
        'sweep':        sweep_results,
        'notes': {
            'gate_1.00': 'hard-bypass Q (never seen during condX-v2 training)',
            'gate_0.10': 'condX-v2 converged equilibrium (weights trained here)',
            'gate_0.00': 'pure contaminated Q (condV-like), weights not trained here',
        },
    }
    result_path = os.path.join(_repo, RESULT_FILE)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(out, f, indent=2)

    print()
    print(f'Results → {RESULT_FILE}')
    print()
    print('Summary (gate → mean passkey):')
    for row in sweep_results:
        bar = '█' * int(row['mean_passkey'] * 40)
        print(f'  gate={row["gate"]:.2f}  {row["mean_passkey"]*100:5.1f}%  {bar}')


if __name__ == '__main__':
    main()
