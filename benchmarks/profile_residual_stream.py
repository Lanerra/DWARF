"""
profile_residual_stream.py — Layer-by-layer residual stream analysis
for condU-v5 38M checkpoint.

Three measurements:

  1. RESIDUAL DELTA NORMS
     For a passkey retrieval task, measure ||h_{l+1} - h_l|| at:
       - passkey insertion token (where the word was planted)
       - passkey query token (last token before answer)
       - control tokens (random filler positions)
     Tells us which layers do the most work for retrieval vs. filler processing.

  2. FULL-ATTENTION WEIGHT PROFILE
     At the passkey query position, extract what the full attention layer
     (layer 5) is attending to. Shows whether it's doing sharp content-addressed
     lookup toward the passkey insertion point.

  3. SURGICAL ABLATION
     For each layer, zero its attention output (keep FFN, keep residual),
     then measure passkey accuracy at all 12 distances. Quantifies each
     layer's causal contribution to retrieval.

Run on 3090:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/profile_residual_stream.py \\
    2>&1 | tee benchmarks/logs/residual_profile.log

Output: benchmarks/logs/residual_profile.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── GPU / device ───────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(SCRIPT_DIR)
CHECKPOINT    = os.path.join(REPO_ROOT, 'checkpoints', 'condU_v5', 'best.pt')
RESULT_FILE   = os.path.join(REPO_ROOT, 'benchmarks', 'logs', 'residual_profile.json')
TOK_PATH      = os.path.join(REPO_ROOT, 'results', '2048_condI_tokenizer.json')

# ── Kernel path ────────────────────────────────────────────────────────────────
_kernel_dir  = os.path.join(REPO_ROOT, 'kernels')
_cuda_ext    = os.path.join(REPO_ROOT, 'kernels', 'dsqg_cuda')
for _d in [_kernel_dir, _cuda_ext, os.path.join(REPO_ROOT, 'train')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Model architecture (38M condU-v5: D=512, H=8, L=6, FFN=2048) ─────────────
VOCAB_SIZE      = 32000
EMBEDDING_DIM   = 512
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 2048
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5
MAX_SEQ_LEN     = 2048

# ── Passkey config ─────────────────────────────────────────────────────────────
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 10
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Offset set ─────────────────────────────────────────────────────────────────
_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))

# ── Load kernel ────────────────────────────────────────────────────────────────
print('[kernel] Loading dsqg_attention_v5...')
import dsqg_attention_v5 as _v5_module
try:
    import dsqg_cuda as _dsqg_cuda_ext  # noqa
    from dsqg_attention_v5_cuda import dsqg_attention_v5_cuda as _cuda_attn_fn
    _v5_module.dsqg_attention_v5 = _cuda_attn_fn
    print('[kernel] CUDA extension loaded')
except ImportError:
    print('[kernel] CUDA extension not found — using Triton')
from dsqg_attention_v5 import DSQGAttentionV5


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (matches condU-v5 38M exactly)
# ══════════════════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV5(nn.Module):
    def __init__(self, d, h, ffn, seq_len, dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads = h
        self.head_dim  = d // h
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn  = DSQGAttentionV5(d, h, seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(d, ffn, dropout)
        if interference:
            self.inter_norm   = nn.LayerNorm(d)
            self.inter_gate   = nn.Linear(d, d)
            self.inter_k_proj = nn.Linear(d, d)
            self.inter_v_proj = nn.Linear(d, d)

    def forward(self, x, *, ablate_attn=False):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            counts  = torch.arange(1, N+1, device=xi.device,
                                   dtype=xi.dtype).view(1, N, 1)
            pool    = xi.cumsum(dim=1) / counts
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0,2,1,3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0,2,1,3).contiguous())
            kv_inject = (k_delta, v_delta)

        attn_out = self.attn(self.norm1(x), kv_inject=kv_inject)
        if ablate_attn:
            attn_out = torch.zeros_like(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FullCausalAttentionProfiled(nn.Module):
    """Full causal attention that optionally returns attention weights."""
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.num_heads = h
        self.head_dim  = d // h
        self.qkv_proj  = nn.Linear(d, 3*d, bias=True)
        self.out_proj  = nn.Linear(d, d, bias=True)
        self.gate_proj = nn.Linear(d, d, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x, return_attn=False, ablate=False):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B,N,H,HD).permute(0,2,1,3)
        k = k.view(B,N,H,HD).permute(0,2,1,3)
        v = v.view(B,N,H,HD).permute(0,2,1,3)

        if ablate:
            out_flat = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
            gate     = torch.sigmoid(self.gate_proj(x))
            return self.out_proj(out_flat * gate), None

        if return_attn:
            # Manual attention to get weights
            scale  = HD ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,N,N]
            # Causal mask
            mask   = torch.triu(torch.full((N,N), float('-inf'),
                                           device=x.device), diagonal=1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_w = torch.softmax(scores, dim=-1)  # [B,H,N,N]
            out    = torch.matmul(attn_w, v)         # [B,H,N,HD]
        else:
            attn_w = None
            out    = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True)

        out_flat = out.permute(0,2,1,3).reshape(B,N,D)
        gate     = torch.sigmoid(self.gate_proj(x))
        result   = F.dropout(self.out_proj(out_flat * gate),
                             p=self.dropout_p, training=self.training)
        return result, attn_w


class FullAttentionBlockProfiled(nn.Module):
    def __init__(self, d, h, ffn, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn  = FullCausalAttentionProfiled(d, h, dropout)
        self.ffn   = FFN(d, ffn, dropout)

    def forward(self, x, return_attn=False, ablate_attn=False):
        attn_out, attn_w = self.attn(self.norm1(x),
                                     return_attn=return_attn,
                                     ablate=ablate_attn)
        if ablate_attn:
            attn_out = torch.zeros_like(x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class CondUV5Profiled(nn.Module):
    def __init__(self):
        super().__init__()
        D, H, L, FFN = EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM
        self.embedding       = nn.Embedding(VOCAB_SIZE, D)
        self.pos_embed       = nn.Embedding(MAX_SEQ_LEN + 2, D)
        self.drop            = nn.Dropout(0.1)
        self.full_attn_layer = FULL_ATTN_LAYER

        blocks = []
        for i in range(L):
            if i == FULL_ATTN_LAYER:
                blocks.append(FullAttentionBlockProfiled(D, H, FFN))
            else:
                blocks.append(DSQGBlockV5(
                    D, H, FFN, MAX_SEQ_LEN,
                    interference=(i % INTERFERENCE == INTERFERENCE - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(D)
        self.out    = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, idx,
                return_residuals=False,
                full_attn_return_weights=False,
                ablate_layer=None):
        """
        Args:
          return_residuals:          if True, return list of residual states per layer
          full_attn_return_weights:  if True, return full-attn attention weights
          ablate_layer:              int or None — zero this layer's attn contribution
        Returns:
          logits [B, N, V]
          residuals: list of [B,N,D] tensors (if return_residuals)
          attn_weights: [B,H,N,N] or None (from full attention layer)
        """
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))

        residuals   = [x.detach()] if return_residuals else None
        attn_weights = None

        for i, block in enumerate(self.blocks):
            ablate = (ablate_layer == i)
            if isinstance(block, FullAttentionBlockProfiled):
                x, aw = block(x,
                               return_attn=full_attn_return_weights,
                               ablate_attn=ablate)
                if aw is not None:
                    attn_weights = aw
            else:
                x = block(x, ablate_attn=ablate)

            if return_residuals:
                residuals.append(x.detach())

        logits = self.out(self.norm(x))
        return logits, residuals, attn_weights

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer + passkey helpers
# ══════════════════════════════════════════════════════════════════════════════

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


def make_passkey_sequence(tokenizer, target_word, distance):
    """Build a passkey sequence. Returns (token_ids, intro_end_pos, cue_start_pos)."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    intro_ids  = tokenizer.encode(_INTRO_TEMPLATE.format(word=target_word))
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)

    filler = []
    while len(filler) < distance:
        filler.extend(filler_ids)
    filler = filler[:distance]

    seq = intro_ids + filler + cue_ids

    # intro_end_pos: index of last intro token (the passkey word token)
    intro_end_pos = len(intro_ids) - 1
    # cue_start_pos: index where retrieval cue starts
    cue_start_pos = len(intro_ids) + len(filler)

    return seq, intro_end_pos, cue_start_pos


# ══════════════════════════════════════════════════════════════════════════════
# Measurement 1: Residual delta norms per layer
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_residual_deltas(model, tokenizer, distances=(1, 64, 512, 1536)):
    """
    For each distance, run a passkey task and measure per-layer residual deltas
    at three position types:
      - 'passkey': the passkey word insertion token
      - 'query': the last token of the retrieval cue
      - 'control': a random filler token in the middle
    """
    model.eval()
    results = {}

    for d in distances:
        if d > MAX_SEQ_LEN - 50:
            continue

        per_trial = []
        for trial_i in range(3):   # 3 trials per distance
            target   = _PASSKEY_WORDS[trial_i % len(_PASSKEY_WORDS)]
            seq, passkey_pos, cue_pos = make_passkey_sequence(tokenizer, target, d)
            if len(seq) >= MAX_SEQ_LEN:
                continue

            ids      = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            logits, residuals, _ = model(ids[:, :-1], return_residuals=True)

            # Positions to probe (in the input sequence, which is seq[:-1])
            n_tokens   = len(seq) - 1
            ctrl_pos   = min(passkey_pos + max(1, d // 2), cue_pos - 1, n_tokens - 1)

            # Clip positions to valid range
            pp = min(passkey_pos, n_tokens - 1)
            qp = min(cue_pos,     n_tokens - 1)
            cp = min(ctrl_pos,    n_tokens - 1)

            # Per-layer delta norms: ||residual[l+1] - residual[l]||
            layer_deltas = {'passkey': [], 'query': [], 'control': []}
            for l in range(len(residuals) - 1):
                h_before = residuals[l]    # [1, N, D]
                h_after  = residuals[l+1]

                delta = (h_after - h_before).squeeze(0)  # [N, D]
                layer_deltas['passkey'].append(delta[pp].norm().item())
                layer_deltas['query'].append(  delta[qp].norm().item())
                layer_deltas['control'].append(delta[cp].norm().item())

            per_trial.append(layer_deltas)

        if not per_trial:
            continue

        # Average across trials
        n_layers = len(per_trial[0]['passkey'])
        avg_deltas = {}
        for pos_type in ('passkey', 'query', 'control'):
            avg_deltas[pos_type] = [
                sum(t[pos_type][l] for t in per_trial) / len(per_trial)
                for l in range(n_layers)
            ]

        results[d] = avg_deltas
        print(f'\n  Residual deltas (d={d}):')
        print(f'  {"Layer":<6}  {"passkey_pos":>12}  {"query_pos":>10}  '
              f'{"control":>10}  {"pk/ctrl ratio":>14}')
        for l in range(n_layers):
            pk  = avg_deltas['passkey'][l]
            qy  = avg_deltas['query'][l]
            ct  = avg_deltas['control'][l]
            ratio = pk/ct if ct > 1e-9 else float('inf')
            layer_label = f'L{l}' + (' [FULL]' if l == FULL_ATTN_LAYER else '')
            print(f'  {layer_label:<6}  {pk:>12.4f}  {qy:>10.4f}  '
                  f'{ct:>10.4f}  {ratio:>14.2f}x')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Measurement 2: Full-attention weight profile
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_full_attn_weights(model, tokenizer, distances=(1, 64, 256, 1536)):
    """
    At the passkey query position, show what the full attention layer attends to.
    Reports: top-5 attended positions, whether passkey insertion position is
    among top-k, and mean attention mass on passkey vs. filler regions.
    """
    model.eval()
    results = {}

    for d in distances:
        if d > MAX_SEQ_LEN - 50:
            continue

        trial_results = []
        for trial_i in range(5):
            target   = _PASSKEY_WORDS[trial_i % len(_PASSKEY_WORDS)]
            seq, passkey_pos, cue_pos = make_passkey_sequence(tokenizer, target, d)
            if len(seq) >= MAX_SEQ_LEN:
                continue

            ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            _, _, attn_w = model(ids[:, :-1],
                                 full_attn_return_weights=True)
            # attn_w: [1, H, N, N]
            if attn_w is None:
                continue

            N = ids.shape[1] - 1
            qp = min(cue_pos, N - 1)
            pp = min(passkey_pos, N - 1)

            # Average across heads: attn from query position to all other positions
            attn_from_query = attn_w[0, :, qp, :qp+1].mean(0)  # [qp+1]

            # Passkey region: intro tokens (0..passkey_pos)
            passkey_mass = attn_from_query[:pp+1].sum().item()
            # Filler region: tokens after intro, before cue
            filler_start = pp + 1
            filler_end   = max(qp, pp + 1)
            filler_mass  = attn_from_query[filler_start:filler_end].sum().item()
            # Cue region: cue tokens themselves
            cue_mass     = attn_from_query[filler_end:qp+1].sum().item()

            # Top-5 attended positions
            top5_vals, top5_idx = attn_from_query.topk(min(5, len(attn_from_query)))
            passkey_in_top5 = pp in top5_idx.tolist()

            # Per-head attention to passkey position
            head_attn_to_pk = attn_w[0, :, qp, pp].tolist()

            trial_results.append({
                'passkey_mass': passkey_mass,
                'filler_mass':  filler_mass,
                'cue_mass':     cue_mass,
                'passkey_in_top5': passkey_in_top5,
                'head_attn_to_pk': head_attn_to_pk,
                'top5_positions': top5_idx.tolist(),
                'top5_values':    top5_vals.tolist(),
                'passkey_pos': pp,
                'query_pos':   qp,
            })

        if not trial_results:
            continue

        n = len(trial_results)
        avg = {
            'passkey_mass':    sum(r['passkey_mass'] for r in trial_results) / n,
            'filler_mass':     sum(r['filler_mass']  for r in trial_results) / n,
            'cue_mass':        sum(r['cue_mass']     for r in trial_results) / n,
            'passkey_in_top5_rate': sum(r['passkey_in_top5'] for r in trial_results) / n,
            'head_attn_to_pk': [
                sum(r['head_attn_to_pk'][h] for r in trial_results) / n
                for h in range(NUM_HEADS)
            ],
        }
        results[d] = {'avg': avg, 'trials': trial_results}

        print(f'\n  Full-attn weight profile (d={d}):')
        print(f'    Attention mass at query position — averaged over {n} trials, {NUM_HEADS} heads:')
        print(f'      → passkey region: {avg["passkey_mass"]:.4f}')
        print(f'      → filler region:  {avg["filler_mass"]:.4f}')
        print(f'      → cue region:     {avg["cue_mass"]:.4f}')
        print(f'    Passkey pos in top-5 attended: {avg["passkey_in_top5_rate"]*100:.0f}% of trials')
        hd_str = '  '.join(f'h{h}:{v:.3f}' for h, v in enumerate(avg['head_attn_to_pk']))
        print(f'    Per-head attn → passkey pos: {hd_str}')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Measurement 3: Surgical ablation — zero each layer's attention output
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_ablation(model, tokenizer):
    """
    For each layer (0..5), zero its attention output and measure passkey accuracy
    at all 12 distances. Ablation of attention only — FFN and residual remain.
    Baseline (no ablation) is measured first.
    Also measures the "identity" ablation (ablate none) as sanity check.
    """
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)

    def passkey_acc_with_ablation(ablate_layer):
        results = {}
        for d in PASSKEY_DISTANCES:
            correct = 0; n_valid = 0
            for trial_i in range(PASSKEY_TRIALS):
                target   = _PASSKEY_WORDS[trial_i % len(_PASSKEY_WORDS)]
                others   = [w for w in _PASSKEY_WORDS if w != target]
                intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
                avail    = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
                if d > avail: continue
                filler = []
                while len(filler) < d: filler.extend(filler_ids)
                seq = intro_ids + filler[:d] + cue_ids
                if len(seq) >= MAX_SEQ_LEN: continue
                ids    = torch.tensor([seq], dtype=torch.long, device=DEVICE)
                logits, _, _ = model(ids, ablate_layer=ablate_layer)
                logits_last  = logits[0, -1]
                cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                            for w in [target] + others[:9]]
                pred     = ([target] + others[:9])[logits_last[cand_ids].argmax().item()]
                correct += int(pred == target)
                n_valid += 1
            results[d] = correct / n_valid if n_valid else 0.0
        return results

    results = {}

    # Baseline: no ablation
    print('\n  Ablation: BASELINE (no ablation)')
    t0  = time.time()
    acc = passkey_acc_with_ablation(ablate_layer=None)
    elapsed = time.time() - t0
    mean_acc = sum(acc.values()) / len(acc)
    above50  = sum(1 for v in acc.values() if v >= 0.5)
    print(f'    mean={mean_acc*100:.1f}%  ({above50}/{len(acc)} >50%)  [{elapsed:.0f}s]')
    parts = [f'd={d}:{int(acc[d]*100)}%' for d in PASSKEY_DISTANCES]
    print('    ' + '  '.join(parts))
    results['baseline'] = {'layer': None, 'acc': acc,
                           'mean': mean_acc, 'above50': above50}

    # Per-layer ablations
    for layer_idx in range(NUM_LAYERS):
        layer_type = 'FULL' if layer_idx == FULL_ATTN_LAYER else 'DSQG'
        int_flag   = ''
        if layer_idx != FULL_ATTN_LAYER:
            is_int = (layer_idx % INTERFERENCE == INTERFERENCE - 1)
            int_flag = '+INT' if is_int else ''
        print(f'\n  Ablation: Layer {layer_idx} [{layer_type}{int_flag}]')
        t0  = time.time()
        acc = passkey_acc_with_ablation(ablate_layer=layer_idx)
        elapsed = time.time() - t0
        mean_acc = sum(acc.values()) / len(acc)
        above50  = sum(1 for v in acc.values() if v >= 0.5)
        baseline_mean = results['baseline']['mean']
        delta = mean_acc - baseline_mean
        print(f'    mean={mean_acc*100:.1f}%  ({above50}/{len(acc)} >50%)  '
              f'[Δ={delta*100:+.1f}pp vs baseline]  [{elapsed:.0f}s]')
        parts = [f'd={d}:{int(acc[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('    ' + '  '.join(parts))
        results[f'layer_{layer_idx}'] = {
            'layer': layer_idx, 'type': layer_type + int_flag,
            'acc': acc, 'mean': mean_acc, 'above50': above50,
            'delta_vs_baseline': delta,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('  Residual Stream Profiler — condU-v5 38M')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Checkpoint: {CHECKPOINT}')

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    from tokenizers import Tokenizer as _HFTok
    tokenizer = BPETokenizerWrapper(_HFTok.from_file(TOK_PATH))
    print(f'  Tokenizer: {TOK_PATH}')

    # ── Build and load model ───────────────────────────────────────────────────
    model = CondUV5Profiled().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f'  Loaded: {model.param_count():,} parameters')

    # Verify layer layout
    layer_types = []
    for i, b in enumerate(model.blocks):
        if isinstance(b, FullAttentionBlockProfiled):
            layer_types.append(f'L{i}:FULL')
        elif isinstance(b, DSQGBlockV5):
            t = 'DSQG+INT' if b.interference else 'DSQG'
            layer_types.append(f'L{i}:{t}')
    print(f'  Layers: {" | ".join(layer_types)}')
    print()

    all_results = {}

    # ── 1. Residual delta norms ────────────────────────────────────────────────
    print('━' * 70)
    print('  MEASUREMENT 1: Per-layer residual delta norms')
    print('  (||h_{l+1} - h_l|| at passkey/query/control positions)')
    print('━' * 70)
    delta_results = measure_residual_deltas(
        model, tokenizer, distances=[1, 32, 256, 1024, 1536])
    all_results['residual_deltas'] = delta_results

    # ── 2. Full-attention weight profile ──────────────────────────────────────
    print()
    print('━' * 70)
    print('  MEASUREMENT 2: Full-attention weight profile')
    print('  (what does layer 5 attend to at the passkey query position?)')
    print('━' * 70)
    attn_results = measure_full_attn_weights(
        model, tokenizer, distances=[1, 32, 256, 1024, 1536])
    all_results['full_attn_weights'] = attn_results

    # ── 3. Surgical ablation ──────────────────────────────────────────────────
    print()
    print('━' * 70)
    print('  MEASUREMENT 3: Surgical ablation (zero attn output per layer)')
    print('  (which layers are causally necessary for passkey retrieval?)')
    print('━' * 70)
    ablation_results = measure_ablation(model, tokenizer)
    all_results['ablation'] = ablation_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('━' * 70)
    print('  ABLATION SUMMARY (mean passkey accuracy when each layer is zeroed)')
    print('━' * 70)
    baseline = ablation_results['baseline']['mean']
    print(f'  {"Layer":<20}  {"Mean passkey":>12}  {"Δ vs baseline":>14}  {"Verdict":}')
    print(f'  {"-"*20}  {"-"*12}  {"-"*14}  {"-"*30}')
    print(f'  {"baseline (none)":<20}  {baseline*100:>11.1f}%  {"—":>14}')
    for layer_idx in range(NUM_LAYERS):
        r = ablation_results[f'layer_{layer_idx}']
        m = r['mean']
        d = r['delta_vs_baseline']
        layer_type = r['type']
        # Verdict: how critical is this layer?
        if d < -0.20:
            verdict = '*** CRITICAL (>20pp drop)'
        elif d < -0.10:
            verdict = '** important (10-20pp drop)'
        elif d < -0.03:
            verdict = '* marginal (3-10pp drop)'
        elif d > 0.05:
            verdict = '↑ helps when ablated (interference?)'
        else:
            verdict = '~ negligible'
        print(f'  L{layer_idx} [{layer_type}]{"":<13}  {m*100:>11.1f}%  '
              f'{d*100:>+13.1f}pp  {verdict}')

    print()
    print(f'  Results → {RESULT_FILE}')
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    # Serialize (convert int keys in dicts to str for JSON)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(RESULT_FILE, 'w') as fp:
        json.dump(to_serializable(all_results), fp, indent=2)
    print('  Done.')


if __name__ == '__main__':
    main()
