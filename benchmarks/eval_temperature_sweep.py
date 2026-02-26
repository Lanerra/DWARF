"""
Temperature Sweep Evaluation — condN (and any DSQG checkpoint)

PURPOSE
-------
Claude Opus 4.6 identified temperature sweep as the simplest diagnostic:
  "condN's mild loops at ep8 may simply reflect the model needing higher
   temperature for diversity, not architecture failure."

This script loads a condN checkpoint and generates at T=0.0, 0.5, 0.7, 1.0
with nucleus sampling (top-p=0.9), allowing comparison across temperatures.

Also runs top-k=50 sampling at T=0.7 as an alternative to nucleus.

Usage:
  # condN best checkpoint (default):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_temperature_sweep.py

  # Different checkpoint or model:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_temperature_sweep.py \
    --checkpoint 2048_condN_checkpoints/best.pt \
    --output benchmarks/2048_condN_temp_sweep.json

Outputs:
  - Printed comparison table (easy to read in logs)
  - JSON file with all generations for record-keeping
  - Repetition rate statistics per temperature
"""

import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ─── Model architecture (must match the checkpoint being loaded) ──────────────
# Defaults match condN. Override if evaluating condP or condM.

VOCAB_SIZE    = 32000
MAX_SEQ_LEN   = 2048
EMBEDDING_DIM = 256
NUM_LAYERS    = 6
NUM_HEADS     = 8
FFN_DIM       = 1024
INTERFERENCE  = 3

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))


# ─── Model (condN, copy of architecture for loading) ─────────────────────────

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        if offsets is None:
            offsets = _COND_N_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(
            -delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))
        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        flat  = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate  = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(flat * gate))


class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionN(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)
        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_pool = nn.Linear(embedding_dim, embedding_dim)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        x = x + torch.utils.checkpoint.checkpoint(
            self._attn_fn, x, use_reentrant=False)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class CondNTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, interference_interval=INTERFERENCE,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DSQGBlock(embedding_dim, num_heads, ffn_dim, seq_len,
                      dropout=dropout,
                      interference=(i % interference_interval == interference_interval - 1))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


# ─── Sampling strategies ──────────────────────────────────────────────────────

def sample_top_p(logits, temperature, top_p=0.9):
    """Nucleus (top-p) sampling."""
    if temperature <= 0.01:
        return logits.argmax()
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    return sorted_idx[torch.multinomial(sorted_probs, 1)]


def sample_top_k(logits, temperature, top_k=50):
    """Top-k sampling."""
    if temperature <= 0.01:
        return logits.argmax()
    top_k_vals, top_k_idx = torch.topk(logits, top_k)
    probs = F.softmax(top_k_vals / temperature, dim=-1)
    return top_k_idx[torch.multinomial(probs, 1)]


def generate(model, tokenizer, prompt, device, max_new=200,
             sampler='top_p', temperature=1.0, top_p=0.9, top_k=50):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids[:, -MAX_SEQ_LEN:])
            last   = logits[0, -1]
            if sampler == 'greedy' or temperature <= 0.01:
                next_id = last.argmax()
            elif sampler == 'top_k':
                next_id = sample_top_k(last, temperature, top_k)
            else:
                next_id = sample_top_p(last, temperature, top_p)
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    gen_ids = ids[0, len(tokenizer.encode(prompt)):].tolist()
    return tokenizer.decode(gen_ids)


# ─── Repetition rate ─────────────────────────────────────────────────────────

def repetition_rate(text, n=4):
    """
    Fraction of tokens that are part of a repeated n-gram.
    Higher = more degenerate repetition.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    from collections import Counter
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(ngrams), 1)


def distinct_n(text, n=2):
    """Ratio of unique n-grams to total n-grams (higher = more diverse)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='2048_condN_checkpoints/best.pt')
    parser.add_argument('--tokenizer',  default='benchmarks/2048_condI_tokenizer.json')
    parser.add_argument('--output',     default='benchmarks/2048_condN_temp_sweep.json')
    parser.add_argument('--max_new',    type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        # Try relative to script dir
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tok_path   = os.path.join(script_dir, '2048_condI_tokenizer.json')
    if not os.path.exists(tok_path):
        print(f'ERROR: tokenizer not found at {args.tokenizer}')
        sys.exit(1)
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}')

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found at {ckpt_path}')
        print('  Run training first, or pass --checkpoint <path>')
        sys.exit(1)

    model = CondNTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'Model: {n_params:,} parameters, condN architecture ({len(_COND_N_OFFSETS)} offsets)')

    # ── Prompts ───────────────────────────────────────────────────────────────
    PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
        # Additional prompts for richer sweep
        'Scientists have recently discovered',
        'The best way to learn programming is',
        'In the year 2045,',
    ]

    # ── Temperature / sampling configurations ─────────────────────────────────
    CONFIGS = [
        {'label': 'greedy (T=0)',  'sampler': 'greedy',  'temperature': 0.0},
        {'label': 'top-p T=0.5',   'sampler': 'top_p',   'temperature': 0.5,  'top_p': 0.9},
        {'label': 'top-p T=0.7',   'sampler': 'top_p',   'temperature': 0.7,  'top_p': 0.9},
        {'label': 'top-p T=1.0',   'sampler': 'top_p',   'temperature': 1.0,  'top_p': 0.9},
        {'label': 'top-k50 T=0.7', 'sampler': 'top_k',   'temperature': 0.7,  'top_k': 50},
    ]

    all_results = {}

    print('\n' + '=' * 70)
    print('  TEMPERATURE SWEEP — condN best checkpoint')
    print('=' * 70)

    for cfg in CONFIGS:
        label = cfg['label']
        print(f'\n── {label} ──')
        config_results = {}

        for prompt in PROMPTS:
            t0  = time.time()
            gen = generate(
                model, tokenizer, prompt, device,
                max_new    = args.max_new,
                sampler    = cfg['sampler'],
                temperature= cfg['temperature'],
                top_p      = cfg.get('top_p', 0.9),
                top_k      = cfg.get('top_k', 50),
            )
            elapsed = time.time() - t0

            rep_rate = repetition_rate(gen)
            dist2    = distinct_n(gen, 2)
            dist4    = distinct_n(gen, 4)

            config_results[prompt] = {
                'generation':     gen,
                'rep_rate_4gram': rep_rate,
                'distinct_2':     dist2,
                'distinct_4':     dist4,
                'time_s':         elapsed,
            }

            print(f'  {repr(prompt[:40])}')
            print(f'    → {repr(gen[:100])}')
            print(f'    rep_rate={rep_rate:.3f}  distinct-2={dist2:.3f}  distinct-4={dist4:.3f}')

        all_results[label] = config_results

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  SUMMARY: Mean repetition rate and diversity by configuration')
    print('=' * 70)
    print(f'  {"Config":<22} {"rep_rate":>10} {"distinct-2":>12} {"distinct-4":>12}')
    print(f'  {"-"*22} {"-"*10} {"-"*12} {"-"*12}')

    for cfg in CONFIGS:
        label   = cfg['label']
        results = all_results[label]
        mean_rep  = sum(r['rep_rate_4gram'] for r in results.values()) / len(results)
        mean_d2   = sum(r['distinct_2']     for r in results.values()) / len(results)
        mean_d4   = sum(r['distinct_4']     for r in results.values()) / len(results)
        print(f'  {label:<22} {mean_rep:>10.3f} {mean_d2:>12.3f} {mean_d4:>12.3f}')

    print()
    print('  Interpretation:')
    print('  - rep_rate: fraction of 4-grams that are repeated (lower = less looping)')
    print('  - distinct-N: unique N-gram ratio (higher = more diverse output)')
    print('  - If greedy has high rep_rate but T=0.7 is much lower:')
    print('    → model quality OK; just needs sampling diversity')
    print('  - If all configs have high rep_rate:')
    print('    → architectural issue; temperature alone cannot fix degenerate attractors')

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        'checkpoint':   args.checkpoint,
        'model_params': n_params,
        'n_offsets':    len(_COND_N_OFFSETS),
        'max_new':      args.max_new,
        'configs':      CONFIGS,
        'results':      all_results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Results → {args.output}')


if __name__ == '__main__':
    main()
