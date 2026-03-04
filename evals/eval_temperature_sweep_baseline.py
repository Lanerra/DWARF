"""
Temperature Sweep Evaluation — Standard Transformer Baseline

Mirrors eval_temperature_sweep.py but loads the standard causal transformer
checkpoint so we can compare generation diversity (distinct-4) against condP.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_temperature_sweep_baseline.py \
    --checkpoint checkpoints/2048_standard_baseline_checkpoints/best.pt \
    --output benchmarks/results/2048_baseline_temp_sweep.json
"""

import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE    = 32000
MAX_SEQ_LEN   = 2048
EMBEDDING_DIM = 256
NUM_LAYERS    = 6
NUM_HEADS     = 8
FFN_DIM       = 1024


# ─── Standard Causal Transformer (must match checkpoint) ─────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=True)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).transpose(1, 2)
        k = k.view(B, N, H, HD).transpose(1, 2)
        v = v.view(B, N, H, HD).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            self.causal_mask[:N, :N].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(embedding_dim, ffn_dim)
        self.fc2     = nn.Linear(ffn_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, seq_len, dropout)
        self.ln2  = nn.LayerNorm(embedding_dim)
        self.ffn  = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardTransformerBaseline(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(seq_len, embedding_dim)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ffn_dim, seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.ln_final(x))


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample_top_p(logits, temperature, top_p=0.9):
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
    if temperature <= 0.01:
        return logits.argmax()
    top_k_vals, top_k_idx = torch.topk(logits, top_k)
    probs = F.softmax(top_k_vals / temperature, dim=-1)
    return top_k_idx[torch.multinomial(probs, 1)]


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, device, max_new=200,
             temperature=1.0, top_p=0.9, top_k=None):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids[:, -MAX_SEQ_LEN:])
            next_logits = logits[0, -1]
            if top_k is not None:
                next_id = sample_top_k(next_logits, temperature, top_k)
            else:
                next_id = sample_top_p(next_logits, temperature, top_p)
            next_id = int(next_id)
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(generated)


def distinct_n(tokens, n=4):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/2048_standard_baseline_checkpoints/best.pt')
    parser.add_argument('--tokenizer',  default='benchmarks/results/2048_condI_tokenizer.json')
    parser.add_argument('--output',     default='benchmarks/results/2048_baseline_temp_sweep.json')
    parser.add_argument('--max_new',    type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tok_path   = os.path.join(script_dir, '..', 'benchmarks', 'results', '2048_condI_tokenizer.json')
    if not os.path.exists(tok_path):
        print(f'ERROR: tokenizer not found at {args.tokenizer}'); sys.exit(1)
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}')

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found at {ckpt_path}'); sys.exit(1)

    model = StandardTransformerBaseline().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'Model: {n_params:,} trainable parameters, standard transformer')

    PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
        'Scientists have recently discovered',
        'The best way to learn programming is',
        'In the year 2045,',
    ]

    CONFIGS = [
        {'label': 'greedy (T=0)',   'temperature': 0.0,  'top_p': 0.9,  'top_k': None},
        {'label': 'top-p T=0.5',   'temperature': 0.5,  'top_p': 0.9,  'top_k': None},
        {'label': 'top-p T=0.7',   'temperature': 0.7,  'top_p': 0.9,  'top_k': None},
        {'label': 'top-p T=1.0',   'temperature': 1.0,  'top_p': 0.9,  'top_k': None},
        {'label': 'top-k50 T=0.7', 'temperature': 0.7,  'top_p': None, 'top_k': 50},
    ]

    all_results = {}
    print('\n' + '=' * 70)
    print('  TEMPERATURE SWEEP — Standard Transformer Baseline')
    print('=' * 70)

    for cfg in CONFIGS:
        label = cfg['label']
        print(f'\n── {label} ──')
        config_results = {}

        for prompt in PROMPTS:
            t0  = time.time()
            gen = generate(
                model, tokenizer, prompt, device,
                max_new=args.max_new,
                temperature=cfg['temperature'],
                top_p=cfg.get('top_p', 0.9),
                top_k=cfg.get('top_k'),
            )
            elapsed = time.time() - t0
            toks    = tokenizer.encode(gen)
            d4      = distinct_n(toks, 4)
            print(f'  [{prompt[:30]:<30}]  distinct-4={d4:.3f}  ({elapsed:.1f}s)')
            print(f'    → {gen[:120]}')
            config_results[prompt] = {
                'generation':  gen,
                'distinct_4':  d4,
                'time_s':      elapsed,
            }

        all_results[label] = config_results

    # Summary
    print('\n' + '=' * 70)
    print('  SUMMARY — avg distinct-4 per config')
    print('=' * 70)
    for label, cfg_results in all_results.items():
        d4s = [r['distinct_4'] for r in cfg_results.values()]
        avg = sum(d4s) / len(d4s)
        mn  = min(d4s)
        mx  = max(d4s)
        print(f'  {label:<25}  avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}')

    output = {
        'checkpoint':   args.checkpoint,
        'arch':         'standard',
        'model_params': n_params,
        'max_new':      args.max_new,
        'configs':      CONFIGS,
        'results':      all_results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
