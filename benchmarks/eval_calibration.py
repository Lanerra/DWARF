"""
Calibration and Logit Entropy Evaluation — condP vs Standard Baseline

PURPOSE
-------
Measure logit entropy and top-1 confidence for condP vs baseline to test
the hypothesis that condP's probability mass is more broadly distributed
(better calibration for sampling despite worse NLL).

DESIGN
------
- Load both models
- For each of 8 diverse prompts, generate 200 tokens
- At EACH generation step, capture the full logit distribution (vocab=32000)
- Compute per-step:
    Entropy: H = -sum(p * log2(p))   [bits]
    Top-1 probability
    Top-5 probability mass
    Effective vocabulary: 2^H
- Aggregate across all prompts and steps:
    Mean entropy, std
    Mean top-1 prob, std
    Percentile distribution of entropies (p10/p25/p50/p75/p90)

Usage:
  cd /home/dlewis3/Desktop/AI/DWARF
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_calibration.py
"""

import json, math, os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ─── Constants ────────────────────────────────────────────────────────────────
VOCAB_SIZE    = 32000
MAX_SEQ_LEN   = 2048
EMBEDDING_DIM = 256
NUM_LAYERS    = 6
NUM_HEADS     = 8
FFN_DIM       = 1024
INTERFERENCE  = 3

_DENSE_LOCAL_W_P     = 64
_DYADIC_LONG_RANGE_P = [96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_P_OFFSETS      = sorted(set(range(0, _DENSE_LOCAL_W_P + 1)) |
                               set(_DYADIC_LONG_RANGE_P))
assert len(_COND_P_OFFSETS) == 74


# ─── condP Architecture ───────────────────────────────────────────────────────

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        if offsets is None:
            offsets = _COND_P_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
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


class FFN_condP(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False, offsets=None):
        super().__init__()
        self.interference = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionN(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout,
                                     offsets=offsets)
        self.ffn   = FFN_condP(embedding_dim, ffn_dim, dropout)
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
                 dropout=0.1, offsets=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DSQGBlock(embedding_dim, num_heads, ffn_dim, seq_len,
                      dropout=dropout,
                      interference=(i % interference_interval == interference_interval - 1),
                      offsets=offsets)
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


# ─── Baseline Architecture ────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
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


class FFN_base(nn.Module):
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
        self.ffn  = FFN_base(embedding_dim, ffn_dim, dropout)

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

def sample_top_p(probs, top_p=0.9):
    """Sample from top-p nucleus given a probability tensor."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs = sorted_probs.clone()
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    return int(sorted_idx[torch.multinomial(sorted_probs, 1)])


# ─── Statistics helpers (stdlib only) ────────────────────────────────────────

def mean_std(values):
    """Compute mean and std from a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return m, var ** 0.5


def percentile(values, p):
    """Compute p-th percentile (0-100) from a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100.0) * (len(sorted_v) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


# ─── Generation with logit capture ───────────────────────────────────────────

def generate_with_logit_stats(model, tokenizer, prompt, device,
                               max_new=200, temperature=1.0, top_p=0.9):
    """
    Generate max_new tokens.
    At each step, compute:
      - entropy in bits
      - top-1 probability
      - top-5 probability mass
      - effective vocabulary (2^entropy)
    Returns list of per-step stat dicts.
    """
    model.eval()
    ids    = tokenizer.encode(prompt)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    stats  = []

    with torch.no_grad():
        for step in range(max_new):
            logits = model(tensor[:, -MAX_SEQ_LEN:])
            last   = logits[0, -1]  # (VOCAB_SIZE,)

            # Convert to probabilities (at temperature=1.0 for calibration measurement;
            # we sample at the requested temperature but measure calibration at T=1.0)
            probs_calib = F.softmax(last, dim=-1)

            # Entropy in bits: H = -sum(p * log2(p))
            # Use log2 = log / log(2)
            log_probs  = torch.log(probs_calib + 1e-10)
            entropy_bits = float(-(probs_calib * (log_probs / math.log(2))).sum())

            # Top-1 probability
            top1_prob = float(probs_calib.max())

            # Top-5 probability mass
            top5_probs, _ = torch.topk(probs_calib, 5)
            top5_mass = float(top5_probs.sum())

            # Effective vocabulary
            effective_vocab = 2 ** entropy_bits

            stats.append({
                'step':            step,
                'entropy_bits':    entropy_bits,
                'top1_prob':       top1_prob,
                'top5_mass':       top5_mass,
                'effective_vocab': effective_vocab,
            })

            # Sample next token using requested temperature
            probs_sample = F.softmax(last / max(temperature, 0.01), dim=-1)
            next_id = sample_top_p(probs_sample, top_p)
            tensor = torch.cat(
                [tensor, torch.tensor([[next_id]], device=device)], dim=1)

    return stats


# ─── Prompts ──────────────────────────────────────────────────────────────────

PROMPTS = [
    'It was a dark and stormy night and the wind howled',
    'The length of the hypotenuse is determined by',
    'The President signed the executive order regarding',
    'Once upon a time there was a princess who lived in',
    'The experimental results clearly indicate that the proposed method',
    'Scientists have recently discovered a new species of',
    'The best way to learn programming is to start with',
    'In the year 2045, artificial intelligence had become',
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condp_checkpoint',
                        default='checkpoints/2048_condP_checkpoints/best.pt')
    parser.add_argument('--baseline_checkpoint',
                        default='checkpoints/2048_standard_baseline_checkpoints/best.pt')
    parser.add_argument('--tokenizer',
                        default='benchmarks/results/2048_condI_tokenizer.json')
    parser.add_argument('--output',
                        default='benchmarks/results/calibration_results.json')
    parser.add_argument('--max_new',     type=int,   default=200)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default 1.0); '
                             'calibration always measured at T=1.0 for fairness')
    parser.add_argument('--top_p',       type=float, default=0.9)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    from tokenizers import Tokenizer
    if not os.path.exists(args.tokenizer):
        print(f'ERROR: tokenizer not found: {args.tokenizer}'); sys.exit(1)
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(args.tokenizer))
    print(f'Loaded tokenizer: {args.tokenizer}')

    # Load condP
    print(f'\nLoading condP from: {args.condp_checkpoint}')
    condp_model = CondNTransformer(offsets=_COND_P_OFFSETS).to(device)
    state = torch.load(args.condp_checkpoint, map_location=device, weights_only=True)
    condp_model.load_state_dict(state)
    condp_model.eval()
    condp_params = sum(p.numel() for p in condp_model.parameters())
    print(f'  condP: {condp_params:,} params')

    # Load baseline
    print(f'Loading baseline from: {args.baseline_checkpoint}')
    base_model = StandardTransformerBaseline().to(device)
    state = torch.load(args.baseline_checkpoint, map_location=device, weights_only=True)
    base_model.load_state_dict(state)
    base_model.eval()
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f'  Baseline: {base_params:,} params')

    # Run evaluation
    print('\n' + '=' * 80)
    print('  CALIBRATION & LOGIT ENTROPY EVALUATION')
    print(f'  Generating {args.max_new} tokens per prompt (T={args.temperature}, '
          f'top_p={args.top_p})')
    print(f'  Calibration measured at T=1.0 (raw model distribution)')
    print('=' * 80)

    all_results = {}

    for model_name, model in [('condP', condp_model), ('baseline', base_model)]:
        print(f'\n── {model_name} ──')
        model_results = {}
        all_entropy    = []
        all_top1       = []
        all_top5       = []
        all_eff_vocab  = []

        for i, prompt in enumerate(PROMPTS):
            t0    = time.time()
            stats = generate_with_logit_stats(
                model, tokenizer, prompt, device,
                max_new=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            elapsed = time.time() - t0

            prompt_entropy   = [s['entropy_bits']    for s in stats]
            prompt_top1      = [s['top1_prob']        for s in stats]
            prompt_top5      = [s['top5_mass']        for s in stats]
            prompt_eff_vocab = [s['effective_vocab']  for s in stats]

            all_entropy   .extend(prompt_entropy)
            all_top1      .extend(prompt_top1)
            all_top5      .extend(prompt_top5)
            all_eff_vocab .extend(prompt_eff_vocab)

            mean_h, _ = mean_std(prompt_entropy)
            mean_t1, _ = mean_std(prompt_top1)
            print(f'  [{i+1}] {repr(prompt[:50])}')
            print(f'       mean_entropy={mean_h:.3f} bits  mean_top1={mean_t1:.4f}  '
                  f'({elapsed:.1f}s)')

            model_results[prompt] = {
                'per_step_stats': stats,
                'summary': {
                    'mean_entropy':   mean_h,
                    'mean_top1':      mean_t1,
                    'mean_top5_mass': sum(prompt_top5) / len(prompt_top5),
                    'mean_eff_vocab': sum(prompt_eff_vocab) / len(prompt_eff_vocab),
                }
            }

        # Aggregate across all prompts and steps
        m_ent, s_ent   = mean_std(all_entropy)
        m_top1, s_top1 = mean_std(all_top1)
        m_top5, _      = mean_std(all_top5)
        m_ev, _        = mean_std(all_eff_vocab)

        pcts = {
            'p10': percentile(all_entropy, 10),
            'p25': percentile(all_entropy, 25),
            'p50': percentile(all_entropy, 50),
            'p75': percentile(all_entropy, 75),
            'p90': percentile(all_entropy, 90),
        }

        print(f'\n  {model_name} aggregated ({len(all_entropy)} steps total):')
        print(f'    Entropy (bits): mean={m_ent:.3f}  std={s_ent:.3f}')
        print(f'    Top-1 prob:     mean={m_top1:.4f} std={s_top1:.4f}')
        print(f'    Top-5 mass:     mean={m_top5:.4f}')
        print(f'    Eff vocab:      mean={m_ev:.1f}')
        print(f'    Entropy percentiles: p10={pcts["p10"]:.2f}  p25={pcts["p25"]:.2f}  '
              f'p50={pcts["p50"]:.2f}  p75={pcts["p75"]:.2f}  p90={pcts["p90"]:.2f}')

        all_results[model_name] = {
            'per_prompt': model_results,
            'aggregate': {
                'n_steps':        len(all_entropy),
                'mean_entropy':   m_ent,
                'std_entropy':    s_ent,
                'mean_top1_prob': m_top1,
                'std_top1_prob':  s_top1,
                'mean_top5_mass': m_top5,
                'mean_eff_vocab': m_ev,
                'entropy_pct':    pcts,
            }
        }

    # Comparison table
    print('\n' + '=' * 80)
    print('  COMPARISON SUMMARY')
    print('=' * 80)
    condp_agg = all_results['condP']['aggregate']
    base_agg  = all_results['baseline']['aggregate']

    print(f'  {"Metric":<30} {"condP":>12} {"baseline":>12} {"Δ (condP-base)":>16}')
    print(f'  {"-"*30} {"-"*12} {"-"*12} {"-"*16}')

    metrics = [
        ('Mean entropy (bits)',  'mean_entropy',   '.3f'),
        ('Std entropy',          'std_entropy',    '.3f'),
        ('Mean top-1 prob',      'mean_top1_prob', '.4f'),
        ('Std top-1 prob',       'std_top1_prob',  '.4f'),
        ('Mean top-5 mass',      'mean_top5_mass', '.4f'),
        ('Mean eff vocab',       'mean_eff_vocab', '.1f'),
    ]

    for label, key, fmt in metrics:
        cv = condp_agg[key]
        bv = base_agg[key]
        delta = cv - bv
        print(f'  {label:<30} {format(cv, fmt):>12} {format(bv, fmt):>12} '
              f'{format(delta, fmt):>16}')

    print(f'\n  Entropy percentiles:')
    print(f'  {"Pct":<10} {"condP":>12} {"baseline":>12} {"Δ":>10}')
    print(f'  {"-"*10} {"-"*12} {"-"*12} {"-"*10}')
    for p in ['p10', 'p25', 'p50', 'p75', 'p90']:
        cv = condp_agg['entropy_pct'][p]
        bv = base_agg['entropy_pct'][p]
        print(f'  {p:<10} {cv:>12.3f} {bv:>12.3f} {cv-bv:>10.3f}')

    print('\n  Interpretation:')
    print('  - Higher entropy → more spread-out distribution (less overconfident)')
    print('  - Lower top-1 prob → less peaked (more options at each step)')
    print('  - If condP has higher entropy → better calibration for sampling')
    print('  - If condP has lower entropy → more confident (possibly overfitted or collapsed)')

    # Save
    output_data = {
        'condP_checkpoint':    args.condp_checkpoint,
        'baseline_checkpoint': args.baseline_checkpoint,
        'condP_params':        condp_params,
        'baseline_params':     base_params,
        'max_new':             args.max_new,
        'temperature':         args.temperature,
        'top_p':               args.top_p,
        'prompts':             PROMPTS,
        'results':             all_results,
    }
    # Remove per-step stats (too verbose) for JSON but keep aggregates
    compact_results = {}
    for model_name in ['condP', 'baseline']:
        compact_results[model_name] = {
            'per_prompt': {
                prompt: data['summary']
                for prompt, data in all_results[model_name]['per_prompt'].items()
            },
            'aggregate': all_results[model_name]['aggregate'],
        }
    output_data['results'] = compact_results

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\n  Results → {args.output}')


if __name__ == '__main__':
    main()
