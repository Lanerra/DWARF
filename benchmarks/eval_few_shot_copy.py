"""
Few-Shot String Copy Evaluation — condP vs Standard Baseline

PURPOSE
-------
Tests whether condP can copy exact strings from few-shot templates.
This is an induction head / in-context learning test.
GemPro raised concern: suppressing the copy attractor might nerf few-shot prompting.

DESIGN
------
- Temperature: T=0.1 (near-greedy, slight stochasticity)
- Top-p: 0.9
- Both models evaluated on identical prompts
- Metrics: exact match, prefix match, Levenshtein edit distance

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_few_shot_copy.py
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
assert len(_COND_P_OFFSETS) == 74, f"Expected 74, got {len(_COND_P_OFFSETS)}"


# ─── condP Architecture (exact copy from eval_temperature_sweep.py) ───────────

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


# ─── Baseline Architecture (exact copy from eval_temperature_sweep_baseline.py) ─

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

def sample_top_p(logits, temperature, top_p=0.9):
    if temperature <= 0.01:
        return int(logits.argmax())
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    return int(sorted_idx[torch.multinomial(sorted_probs, 1)])


def generate(model, tokenizer, prompt, device, max_new=30,
             temperature=0.1, top_p=0.9):
    model.eval()
    ids        = tokenizer.encode(prompt)
    prompt_len = len(ids)
    tensor     = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits  = model(tensor[:, -MAX_SEQ_LEN:])
            last    = logits[0, -1]
            next_id = sample_top_p(last, temperature, top_p)
            tensor  = torch.cat(
                [tensor, torch.tensor([[next_id]], device=device)], dim=1)
    gen_ids = tensor[0, prompt_len:].tolist()
    return tokenizer.decode(gen_ids)


# ─── Levenshtein distance (stdlib only) ───────────────────────────────────────

def levenshtein(a, b):
    """Standard dynamic-programming Levenshtein edit distance."""
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        new_dp = [i] + [0] * lb
        for j in range(1, lb + 1):
            if a[i-1] == b[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    return dp[lb]


# ─── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(generated_text, expected):
    """
    Extract the predicted answer token from generated text.
    Strips leading whitespace, takes first whitespace-delimited token,
    strips trailing punctuation.
    """
    text = generated_text.lstrip(' \n\r\t')
    parts = text.split()
    if not parts:
        return ''
    first = parts[0]
    # Strip trailing punctuation that's not part of expected
    first = first.rstrip('.,;:!?\n\r')
    return first


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_prompt(model, tokenizer, prompt, expected, device,
                    temperature=0.1, top_p=0.9, max_new=30):
    gen = generate(model, tokenizer, prompt, device,
                   max_new=max_new, temperature=temperature, top_p=top_p)

    predicted = extract_answer(gen, expected)

    # Exact match
    exact = (predicted == expected)

    # Prefix match (first len(expected) chars of predicted == expected)
    n = len(expected)
    prefix_match = (predicted[:n] == expected)

    # Edit distance
    edit_dist = levenshtein(predicted, expected)

    return {
        'predicted': predicted,
        'verbatim':  gen[:200],
        'exact_match':   exact,
        'prefix_match':  prefix_match,
        'edit_distance': edit_dist,
        'expected':      expected,
    }


# ─── Test cases ───────────────────────────────────────────────────────────────

TEST_CASES = [
    # 1. Classic alphanumeric code copy (3-shot)
    {
        'name': 'alphanumeric_code_copy_3shot',
        'description': '3-shot alphanumeric code copy',
        'prompt': 'Input: XKCD → Output: XKCD\nInput: 7829 → Output: 7829\nInput: QWRT → Output:',
        'expected': 'QWRT',
        'max_new': 15,
    },
    # 2. Number pattern copy (2-shot)
    {
        'name': 'number_pattern_copy_2shot',
        'description': '2-shot number pattern copy',
        'prompt': '5 → 5\n12 → 12\n847 → ',
        'expected': '847',
        'max_new': 10,
    },
    # 3. Word copy at distance
    {
        'name': 'word_copy_at_distance',
        'description': 'Single keyword recall at context distance',
        'prompt': 'The password is ZEPHYR. What is the password? The password is',
        'expected': 'ZEPHYR',
        'max_new': 10,
    },
    # 4. Template uppercase transformation (3-shot — tests induction head pattern)
    {
        'name': 'template_uppercase_3shot',
        'description': '3-shot uppercase transformation template',
        'prompt': 'alpha → ALPHA\nbanana → BANANA\ncherry → CHERRY\ndragon → ',
        'expected': 'DRAGON',
        'max_new': 15,
    },
    # 5. Symbol-separated copy (2-shot)
    {
        'name': 'pipe_copy_2shot',
        'description': '2-shot pipe-separated identity copy',
        'prompt': 'cat|cat\ndog|dog\nbird|',
        'expected': 'bird',
        'max_new': 10,
    },
    # 6. Mixed alphanumeric pattern (2-shot)
    {
        'name': 'mixed_alphanumeric_copy',
        'description': '2-shot mixed alphanumeric identity copy',
        'prompt': 'X1X → X1X\nY2Y → Y2Y\nZ3Z → ',
        'expected': 'Z3Z',
        'max_new': 10,
    },
    # 7. Repeated keyword recall (multiple occurrences, then recall)
    {
        'name': 'repeated_keyword_recall',
        'description': 'Keyword repeated 3x then recalled',
        'prompt': 'Remember this code: ALPHA7. The code is ALPHA7. Never forget: ALPHA7. What is the code?',
        'expected': 'ALPHA7',
        'max_new': 15,
    },
    # 8. Equals-pattern identity copy (2-shot)
    {
        'name': 'equals_identity_copy',
        'description': '2-shot equals-separated identity copy',
        'prompt': 'red=red, blue=blue, green=',
        'expected': 'green',
        'max_new': 10,
    },
    # 9. Longer few-shot: 4-shot identity with varied codes
    {
        'name': 'code_copy_4shot',
        'description': '4-shot arbitrary code copy',
        'prompt': 'CODE: A3F → A3F\nCODE: B7X → B7X\nCODE: C2P → C2P\nCODE: D9K → D9K\nCODE: E5M → ',
        'expected': 'E5M',
        'max_new': 10,
    },
    # 10. Token at position copy (induction head over positions)
    {
        'name': 'position_indexed_copy',
        'description': 'Position-indexed copy (tests structured induction)',
        'prompt': '[1] apple\n[2] banana\n[3] cherry\n[1]',
        'expected': 'apple',
        'max_new': 15,
    },
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
                        default='benchmarks/results/few_shot_copy_results.json')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p',       type=float, default=0.9)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Tokenizer
    from tokenizers import Tokenizer
    if not os.path.exists(args.tokenizer):
        print(f'ERROR: tokenizer not found at {args.tokenizer}'); sys.exit(1)
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(args.tokenizer))
    print(f'Loaded tokenizer: {args.tokenizer}')

    # Load condP
    print(f'\nLoading condP from: {args.condp_checkpoint}')
    if not os.path.exists(args.condp_checkpoint):
        print('ERROR: condP checkpoint not found'); sys.exit(1)
    condp_model = CondNTransformer(offsets=_COND_P_OFFSETS).to(device)
    state = torch.load(args.condp_checkpoint, map_location=device, weights_only=True)
    condp_model.load_state_dict(state)
    condp_model.eval()
    condp_params = sum(p.numel() for p in condp_model.parameters())
    print(f'  condP: {condp_params:,} params, {len(_COND_P_OFFSETS)} offsets')

    # Load baseline
    print(f'Loading baseline from: {args.baseline_checkpoint}')
    if not os.path.exists(args.baseline_checkpoint):
        print('ERROR: baseline checkpoint not found'); sys.exit(1)
    base_model = StandardTransformerBaseline().to(device)
    state = torch.load(args.baseline_checkpoint, map_location=device, weights_only=True)
    base_model.load_state_dict(state)
    base_model.eval()
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f'  Baseline: {base_params:,} params')

    results = []
    condp_exact_total = 0
    base_exact_total  = 0

    print('\n' + '=' * 80)
    print('  FEW-SHOT STRING COPY EVALUATION')
    print(f'  Temperature={args.temperature}  Top-p={args.top_p}')
    print('=' * 80)

    for tc in TEST_CASES:
        name     = tc['name']
        prompt   = tc['prompt']
        expected = tc['expected']
        max_new  = tc.get('max_new', 30)
        desc     = tc['description']

        print(f'\n[{name}]')
        print(f'  {desc}')
        print(f'  Prompt:   {repr(prompt)}')
        print(f'  Expected: "{expected}"')

        t0 = time.time()
        condp_res = evaluate_prompt(
            condp_model, tokenizer, prompt, expected, device,
            temperature=args.temperature, top_p=args.top_p, max_new=max_new)
        condp_time = time.time() - t0

        t0 = time.time()
        base_res = evaluate_prompt(
            base_model, tokenizer, prompt, expected, device,
            temperature=args.temperature, top_p=args.top_p, max_new=max_new)
        base_time = time.time() - t0

        condp_exact_total += int(condp_res['exact_match'])
        base_exact_total  += int(base_res['exact_match'])

        em_c = '✓' if condp_res['exact_match'] else '✗'
        em_b = '✓' if base_res['exact_match']  else '✗'

        print(f'  condP    → "{condp_res["predicted"]}"  exact={em_c}  '
              f'prefix={condp_res["prefix_match"]}  '
              f'edit_dist={condp_res["edit_distance"]}  ({condp_time:.2f}s)')
        print(f'  baseline → "{base_res["predicted"]}"  exact={em_b}  '
              f'prefix={base_res["prefix_match"]}  '
              f'edit_dist={base_res["edit_distance"]}  ({base_time:.2f}s)')
        print(f'  condP verbatim:    {repr(condp_res["verbatim"][:100])}')
        print(f'  baseline verbatim: {repr(base_res["verbatim"][:100])}')

        results.append({
            'name':        name,
            'description': desc,
            'prompt':      prompt,
            'expected':    expected,
            'condP':       condp_res,
            'baseline':    base_res,
        })

    n = len(TEST_CASES)
    print('\n' + '=' * 80)
    print('  SUMMARY')
    print('=' * 80)
    print(f'  condP    exact match: {condp_exact_total}/{n}  '
          f'({100*condp_exact_total/n:.1f}%)')
    print(f'  baseline exact match: {base_exact_total}/{n}  '
          f'({100*base_exact_total/n:.1f}%)')

    # Per-test comparison table
    print(f'\n  {"Test":<35} {"condP":>7} {"base":>6} {"winner":>8}')
    print(f'  {"-"*35} {"-"*7} {"-"*6} {"-"*8}')
    for r in results:
        cp = '✓' if r['condP']['exact_match'] else '✗'
        bs = '✓' if r['baseline']['exact_match'] else '✗'
        if r['condP']['exact_match'] == r['baseline']['exact_match']:
            winner = 'tie'
        elif r['condP']['exact_match']:
            winner = 'condP'
        else:
            winner = 'base'
        print(f'  {r["name"]:<35} {cp:>7} {bs:>6} {winner:>8}')

    # Edit distance summary
    condp_avg_edit = sum(r['condP']['edit_distance'] for r in results) / n
    base_avg_edit  = sum(r['baseline']['edit_distance'] for r in results) / n
    print(f'\n  Mean edit distance:  condP={condp_avg_edit:.2f}  '
          f'baseline={base_avg_edit:.2f}')

    print('\n  Interpretation:')
    print('  - High exact match → strong induction head / in-context copying')
    print('  - Low edit distance → at least close to the target')
    print('  - If condP lags baseline → copy attractor suppression is hurting ICL')

    # Save
    output_data = {
        'temperature':           args.temperature,
        'top_p':                 args.top_p,
        'condP_checkpoint':      args.condp_checkpoint,
        'baseline_checkpoint':   args.baseline_checkpoint,
        'condP_params':          condp_params,
        'baseline_params':       base_params,
        'summary': {
            'condP_exact_match':      condp_exact_total,
            'baseline_exact_match':   base_exact_total,
            'total_tests':            n,
            'condP_exact_rate':       condp_exact_total / n,
            'baseline_exact_rate':    base_exact_total / n,
            'condP_avg_edit_dist':    condp_avg_edit,
            'baseline_avg_edit_dist': base_avg_edit,
        },
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\n  Results → {args.output}')


if __name__ == '__main__':
    main()
