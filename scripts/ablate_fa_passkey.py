"""
Quick ablation: zero out the FA layer at inference time and run passkey eval.
Compares baseline vs FA-zeroed on the same checkpoint.

Usage:
  python3 scripts/ablate_fa_passkey.py \
    --checkpoint autoresearch/checkpoints/d512_l13_triadic_aabbc_mixed_scratch_best.pt \
    --train_script train/train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16.py \
    --device cuda:0 --n_trials 20
"""

import argparse, importlib.util, os, sys, random, torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
for _d in [os.path.join(REPO, 'kernels'), REPO]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(train_script_path, checkpoint_path, device):
    spec = importlib.util.spec_from_file_location('triadic_train', train_script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.TriadicJ96(
        vocab_size=mod.VOCAB_SIZE,
        embedding_dim=mod.EMBEDDING_DIM,
        num_heads=mod.NUM_HEADS,
        ffn_dim=mod.FFN_DIM,
        seq_len=mod.MAX_SEQ_LEN,
        full_attn_layer=mod.FULL_ATTN_LAYER,
        scale_embed_init_val=0.15,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    if any('_orig_mod.' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.endswith('causal_mask')}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, mod

def zero_fa(model, fa_layer_idx):
    """Replace FA block forward with identity (pure skip — no attn, no FFN)."""
    block = model.blocks[fa_layer_idx]
    block.forward = lambda x: x

def build_passkey_seq(tokenizer, n_tokens, key_pos, secret):
    """Build a passkey sequence; return token ids and the answer string."""
    import re
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    filler_ids = tokenizer.encode(filler).ids
    prompt = f"The secret number is {secret}. Remember it. "
    prompt_ids = tokenizer.encode(prompt).ids
    suffix = " What is the secret number? The secret number is"
    suffix_ids = tokenizer.encode(suffix).ids

    # Build sequence: [filler up to key_pos] [passkey] [filler to fill] [suffix]
    pre  = filler_ids[:key_pos]
    post_len = max(0, n_tokens - key_pos - len(prompt_ids) - len(suffix_ids))
    post = filler_ids[:post_len]
    ids  = pre + prompt_ids + post + suffix_ids
    ids  = ids[:n_tokens]
    return ids, str(secret)

@torch.no_grad()
def run_passkey_trial(model, tokenizer, device, n_tokens, key_pos):
    secret = random.randint(10000, 99999)
    ids, answer = build_passkey_seq(tokenizer, n_tokens, key_pos, secret)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(inp)          # [1, T, V]
    # next token after the last input token
    pred_id = logits[0, -1].argmax().item()
    pred_str = tokenizer.decode([pred_id])
    # check if the first digit of the answer appears in the prediction
    first_digit = answer[0]
    return first_digit in pred_str or answer[:3] in tokenizer.decode(
        logits[0, -1].topk(10).indices.tolist())

@torch.no_grad()
def eval_passkey(model, tokenizer, device, n_trials, seq_len=2048):
    results = {}
    for dist in DISTANCES:
        if dist >= seq_len - 50:
            continue
        key_pos = max(1, dist)
        hits = 0
        for _ in range(n_trials):
            hits += run_passkey_trial(model, tokenizer, device, seq_len, key_pos)
        acc = hits / n_trials
        results[dist] = acc
    return results

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train_script', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--seq_len', type=int, default=2048)
    args = parser.parse_args()

    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(REPO, args.checkpoint)
    if not os.path.isabs(args.train_script):
        args.train_script = os.path.join(REPO, args.train_script)

    from tokenizers import Tokenizer
    tok_path = os.path.join(REPO, 'results', 'fineweb_tokenizer_32k.json')
    tokenizer = Tokenizer.from_file(tok_path)

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  FA Ablation: passkey accuracy baseline vs FA-zeroed")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Device: {device}  |  n_trials={args.n_trials}  seq_len={args.seq_len}")
    print(f"{'='*60}\n")

    # ── Baseline ──
    print("Loading model (baseline)...")
    model, mod = load_model(args.train_script, args.checkpoint, device)
    fa_idx = mod.FULL_ATTN_LAYER
    print(f"  FA layer index: L{fa_idx}\n")

    print("Running BASELINE passkey eval...")
    baseline = eval_passkey(model, tokenizer, device, args.n_trials, args.seq_len)

    # ── FA zeroed ──
    print("\nZeroing FA layer (blocks[{}])...".format(fa_idx))
    zero_fa(model, fa_idx)
    print("Running FA-ZEROED passkey eval...")
    zeroed = eval_passkey(model, tokenizer, device, args.n_trials, args.seq_len)

    # ── Results ──
    print(f"\n{'─'*55}")
    print(f"  {'dist':>6}  {'baseline':>10}  {'FA-zeroed':>10}  {'delta':>8}")
    print(f"{'─'*55}")
    for dist in sorted(set(baseline) | set(zeroed)):
        b = baseline.get(dist, float('nan'))
        z = zeroed.get(dist, float('nan'))
        d = z - b
        flag = " ← BETTER" if d > 0.05 else (" ← worse" if d < -0.05 else "")
        print(f"  {dist:>6}  {b:>10.1%}  {z:>10.1%}  {d:>+8.1%}{flag}")
    print(f"{'─'*55}")
    b_mean = np.mean(list(baseline.values()))
    z_mean = np.mean(list(zeroed.values()))
    print(f"  {'mean':>6}  {b_mean:>10.1%}  {z_mean:>10.1%}  {z_mean-b_mean:>+8.1%}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
