"""
DWARF Attention — Condition K
Dyadic Wave And Resonant Field Attention

condK = condJ + four targeted improvements:

  1. RELATIVE POSITION BIAS
     learned scalar per (j, τ) tap-offset per head, added to Q·K score
     before the ELU feature map. 44 offsets × 8 heads × 6 layers = 2,112 params.
     Physics basis: different dyadic offsets have different structural relevance
     independent of content (ALiBi principle, applied at 44 fixed dyadic positions).

  2. ELU-BASED LINEAR ATTENTION NORMALIZER (improvement 3 in plan)
     Feature map: φ(x) = elu(x + bias) + 1  (always positive)
     Normalizer:  Z_i = Σ_{j,τ} gain_j |h_τ| φ(Q·K_{shifted})
     Output:      gathered / (Z + ε)
     Prevents accumulated Q·K magnitude from causing output explosion.
     Standard formulation from Katharopoulos et al. (2020), adapted for dyadic field.

  3. INTERFERENCE POOLING REMOVED (improvement 2 in plan — ablation test)
     condG through condJ all used cumulative-mean pooling blocks every 3rd layer
     but this was never ablated. condK removes them entirely. If condK ≥ condJ PPL,
     the interference blocks were a null result; if condK < condJ, they helped.

  4. RG-INSPIRED LAYER-WISE SCALE INITIALIZATION (improvement 4 in plan)
     Physics basis: Renormalization Group theory says local (high-frequency) operators
     dominate at microscopic scales; global (low-frequency) at macroscopic.
     Layers 0-1: init scale_gain biased toward short-range j0-j3 (+0.5)
     Layers 2-3: flat init (learns freely)
     Layers 4-5: init biased toward long-range j8-j10 (+0.5)
     This mirrors RG flow through depth: local → global.

Architecture name: DWARF — Dyadic Wave And Resonant Field Attention
  Dyadic:   2^j multi-scale dilation structure
  Wave:     D4 Daubechies physics-derived propagation kernel
  And:      (necessary for the acronym)
  Resonant: Q·K inner product selects resonant (matching) past K vectors
  Field:    distributed causal wave field carries information
  Attention: it is an attention mechanism

Condition comparison table:
  Standard 13M (reference)      PPL  64.5
  Wave V4 [A]                   PPL  86.8
  Wave V4D [B] + dispersion     PPL  87.9
  Wave V4 + Morlet [C]          PPL  87.2
  Wave V4 + KdV [D]             PPL  99.6  (failed)
  Causal D4 DWT [G]             PPL  99.4
  G + Q-gate post-gather [H]    PPL 100.0  (null result)
  G + Q-scale pre-gather [I]    PPL  93.3
  OPWF/efficient Q·K [J]        PPL   TBD  (training)
  DWARF condK (this run)        PPL   ???

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condK.py \
    2>&1 | tee benchmarks/logs/condK_run.log

Results → benchmarks/2048_condK_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters (identical to condI/J for comparability) ─────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000
BPE_TRAIN_DOCS  = 50_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
N_SCALES        = 11        # dilations 2^0 … 2^10
N_TAPS          = 4         # D4 filter taps


# ─── DWARF Attention ──────────────────────────────────────────────────────────

class DWARFAttention(nn.Module):
    """
    Dyadic Wave And Resonant Field Attention.

    Efficient OPWF gather (no explicit outer product):
      output_i = Σ_{j,τ} gain_j * φ(Q_i · K_{i-offset} + bias_{j,τ}) * V_{i-offset}
               / Σ_{j,τ} gain_j * |h_τ| * φ(Q_i · K_{i-offset} + bias_{j,τ})

    where φ(x) = elu(x) + 1  (always positive; ELU linear attention feature map)
    and bias_{j,τ} is a learned scalar per offset per head.
    """

    _D4 = [0.4829629131445341,  0.8365163037378079,
           0.2241438680420134, -0.1294095225512604]

    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 n_scales=N_SCALES, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len
        self.n_scales      = n_scales

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Q-conditioned scale selection (retained from condI/J)
        self.scale_gain   = nn.Parameter(torch.zeros(n_scales, num_heads))
        self.q_scale_proj = nn.Linear(self.head_dim, n_scales, bias=False)
        nn.init.normal_(self.q_scale_proj.weight, 0, 0.01)

        # Identity bypass (retained from condJ)
        self.identity_bypass = nn.Parameter(torch.full((num_heads,), -4.0))

        # ── Improvement 1: Relative position bias ─────────────────────────
        # One learned scalar per (j, tau) tap per head.
        # Total: n_scales * n_taps * num_heads = 11 * 4 * 8 = 352 per layer.
        # Initialized to zero (no prior on which offset is more relevant).
        self.pos_bias = nn.Parameter(
            torch.zeros(n_scales * N_TAPS, num_heads))   # [44, H]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H  = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # Q-conditioned scale gains
        q_offset = self.q_scale_proj(q)                         # [B, H, N, S]
        prior    = self.scale_gain.T.unsqueeze(0).unsqueeze(2)  # [1, H, 1, S]
        gains    = F.softmax(q_offset + prior, dim=-1)          # [B, H, N, S]

        # ── DWARF gather with ELU feature map + normalizer ────────────────
        out = torch.zeros(B, H, N, HD, device=x.device, dtype=q.dtype)
        z   = torch.zeros(B, H, N,  1, device=x.device, dtype=q.dtype)

        tap_idx = 0
        for j in range(self.n_scales):
            d   = 1 << j
            g_j = gains[:, :, :, j].unsqueeze(-1)   # [B, H, N, 1]
            for tau, h_coef in enumerate(self._D4):
                offset   = d * tau
                # Improvement 1: per-tap relative position bias
                bias_jt  = self.pos_bias[tap_idx].view(1, H, 1, 1)  # [1,H,1,1]
                tap_idx += 1

                if offset == 0:
                    k_s, v_s = k, v
                elif offset >= N:
                    continue
                else:
                    pad  = k.new_zeros(B, H, offset, HD)
                    k_s  = torch.cat([pad, k[:, :, :N - offset, :]], dim=2)
                    v_s  = torch.cat([pad, v[:, :, :N - offset, :]], dim=2)

                # Raw Q·K + position bias
                qk_raw  = (q * k_s).sum(-1, keepdim=True) + bias_jt  # [B,H,N,1]

                # ELU feature map: φ(x) = elu(x) + 1  (always ≥ 0)
                qk_feat = F.elu(qk_raw) + 1.0

                # Weighted contribution to output and normalizer
                w = g_j * h_coef
                out = out + w * qk_feat * v_s

                # Improvement 2 (normalizer): use |h_coef| so Z stays positive
                z   = z   + g_j * abs(h_coef) * qk_feat

        # Identity bypass at offset=0 (same position)
        bypass   = F.softplus(self.identity_bypass).view(1, H, 1, 1)
        local_qk = (q * k).sum(-1, keepdim=True)
        local_ft = F.elu(local_qk) + 1.0
        out = out + bypass * local_ft * v
        z   = z   + bypass * local_ft

        # Improvement 2 (normalizer): normalize output
        out = out / (z + 1e-6)

        # Gate and output projection
        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def scale_summary(self):
        with torch.no_grad():
            prior_gains = F.softmax(self.scale_gain, dim=0)
        means   = prior_gains.mean(dim=1)
        dom     = int(means.argmax().item())
        bypass  = F.softplus(self.identity_bypass).tolist()
        pb_mean = self.pos_bias.abs().mean().item()
        pb_max  = self.pos_bias.abs().max().item()
        return {
            'gains_mean_per_scale':     means.tolist(),
            'dominant_scale':           dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'identity_bypass_mean':     sum(bypass) / len(bypass),
            'pos_bias_abs_mean':        pb_mean,
            'pos_bias_abs_max':         pb_max,
            'note': 'global prior only; actual gains are Q-conditioned per-position',
        }


# ─── RG-inspired layer-wise scale init ───────────────────────────────────────

def _init_rg_scale_gain(scale_gain: nn.Parameter, layer_idx: int,
                         num_layers: int, n_scales: int):
    """
    Renormalization Group-inspired initialization of scale gains.
    Early layers → biased toward short-range (small j, high frequency).
    Late layers  → biased toward long-range (large j, low frequency).
    This mirrors RG flow through depth: local operators dominate early,
    global operators emerge at coarser (deeper) scales.
    """
    frac = layer_idx / max(num_layers - 1, 1)   # 0.0 (layer 0) → 1.0 (last)
    data = torch.zeros_like(scale_gain)
    for j in range(n_scales):
        if j <= 3:      # short-range (j0–j3, rf=4–32 tokens)
            bias = +0.5 * (1.0 - frac)   # strong early, weak late
        elif j >= 8:    # long-range  (j8–j10, rf=1024–4096 tokens)
            bias = +0.5 * frac            # weak early, strong late
        else:           # mid-range   (j4–j7, rf=64–512 tokens)
            bias = 0.0                    # flat everywhere
        data[j, :] = bias
    with torch.no_grad():
        scale_gain.copy_(data)


# ─── Transformer blocks (no interference pooling) ─────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DWARFBlock(nn.Module):
    """
    DWARF Transformer block.
    Improvement 3: NO interference pooling.
    condG–condJ all included cumulative-mean pooling every 3rd layer.
    condK removes it entirely to test whether it was helping.
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DWARFAttention(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, dropout=dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        if self.use_checkpoint:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_fn, x, use_reentrant=False)
        else:
            x = x + self._attn_fn(x)
        return x + self.ffn(self.norm2(x))


class DWARFTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DWARFBlock(
                embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                dropout=dropout, use_checkpoint=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights(num_layers, n_scales)

    def _init_weights(self, num_layers, n_scales):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # Q-scale proj: small init
        for block in self.blocks:
            nn.init.normal_(block.attn.q_scale_proj.weight, 0, 0.01)
        # Improvement 4: RG-inspired layer-wise scale gain init
        for i, block in enumerate(self.blocks):
            _init_rg_scale_gain(block.attn.scale_gain, i, num_layers, n_scales)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_summary(self):
        summaries = [b.attn.scale_summary() for b in self.blocks]
        n = len(summaries)
        avg_gains   = [sum(s['gains_mean_per_scale'][j] for s in summaries) / n
                       for j in range(N_SCALES)]
        dom         = int(max(range(N_SCALES), key=lambda j: avg_gains[j]))
        bypass_mean = sum(s['identity_bypass_mean'] for s in summaries) / n
        pb_mean     = sum(s['pos_bias_abs_mean']    for s in summaries) / n
        pb_max      = max(s['pos_bias_abs_max']     for s in summaries)
        return {
            'gains_mean_per_scale':     avg_gains,
            'dominant_scale':           dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'identity_bypass_mean':     bypass_mean,
            'pos_bias_abs_mean':        pb_mean,
            'pos_bias_abs_max':         pb_max,
        }


# ─── Data utilities (unchanged) ───────────────────────────────────────────────

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import (Tokenizer, models, trainers,
                            pre_tokenizers, decoders)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>'])
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_data(num_docs=NUM_DOCS):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {num_docs:,} docs)...')
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0:
            print(f'  {i+1:,} docs...')
    n = len(texts)
    print(f'  {n:,} docs | train {int(n*0.95):,} | val 2,500 | test 2,500')
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
    }


def encode_split(split_texts, tokenizer, max_seq_len, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n = (len(tokens) // max_seq_len) * max_seq_len
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


# ─── Evaluation & generation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=150):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                logits  = model(ids[:, -MAX_SEQ_LEN:])
                next_id = logits[0, -1].argmax()
                ids = torch.cat([ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
        gen = tokenizer.decode(
            ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


# ─── Causality check ──────────────────────────────────────────────────────────

def causality_check(model, device):
    print('Running causality check...')
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 32), device=device)
        x2 = x1.clone(); x2[0, 5] = (x2[0, 5] + 1) % VOCAB_SIZE
        out1, out2 = model(x1), model(x2)
        diff = (out1 - out2).abs()
    pre  = diff[0, :5].max().item()
    pos5 = diff[0,  5].max().item()
    post = diff[0, 6:].max().item()
    print(f'  Pre-5:  {pre:.8f}  (expect 0.0)')
    print(f'  Pos-5:  {pos5:.6f}  (expect >0)')
    print(f'  Post-5: {post:.6f}  (expect >0)')
    ok = pre < 1e-6
    print(f'  {"PASS" if ok else "FAIL"} — architecture is {"causal" if ok else "NOT causal"}')
    return ok


# ─── Training loop ────────────────────────────────────────────────────────────

def train(model, train_data, val_data, test_data, tokenizer,
          save_dir='2048_condK_checkpoints', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda')

    GEN_PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
    ]

    best_val_loss, best_val_ppl, best_epoch = float('inf'), float('inf'), 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda'):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1

            if step % 200 == 0:
                loss_val = loss.item() * GRAD_ACCUM
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss_val:.4f}')
                if epoch == 1 and step == 200 and loss_val > 8.5:
                    print(f'  WARNING: ep1 step-200 loss {loss_val:.4f} > 8.5 — check stability')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')

        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print('  Scale prior (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}' for j in top3))
        print(f'  Identity bypass mean: {ss["identity_bypass_mean"]:.4f}')
        print(f'  Pos-bias |mean|={ss["pos_bias_abs_mean"]:.4f}  |max|={ss["pos_bias_abs_max"]:.4f}')

        print('  ── Generation samples ──')
        for prompt, gen in zip(GEN_PROMPTS, generate(model, tokenizer, GEN_PROMPTS, device)):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  DWARF condK TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')
    ss = model.scale_summary()
    print(f'  Dominant prior scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} tokens)')
    print(f'  Full prior gains: '
          + ' '.join(f'j{j}={ss["gains_mean_per_scale"][j]:.3f}'
                     for j in range(N_SCALES)))

    print('\n' + '=' * 70)
    print('  DWARF condK vs. ABLATION TABLE')
    print('=' * 70)
    table = [
        ('Standard transformer 13M (reference)',        64.5),
        ('Wave V4 [A] (G=4096, FFT)',                   86.8),
        ('Wave V4D [B] + dispersion β',                 87.9),
        ('Wave V4 + Morlet [C]',                        87.2),
        ('Wave V4 + KdV [D] (failed)',                  99.6),
        ('Causal D4 DWT [G] (fixed gains)',             99.4),
        ('G + Q-gate post-gather [H] (null)',          100.0),
        ('G + Q-scale pre-gather [I]',                  93.3),
    ]
    print(f'  {"Model":<52} {"PPL":>6}')
    print('  ' + '─' * 60)
    for name, ppl in table:
        print(f'  {name:<52} {ppl:>6.1f}')
    print(f'  {"OPWF efficient Q·K [J]":<52} {"TBD":>6}')
    print(f'  {"DWARF condK (this run)":<52} {test_ppl:>6.1f}')

    return {
        'test_ppl':       test_ppl,
        'test_loss':      test_loss,
        'best_val_ppl':   best_val_ppl,
        'best_epoch':     best_epoch,
        'total_time_s':   time.time() - t0,
        'scale_summary':  ss,
        'architecture_name': 'DWARF — Dyadic Wave And Resonant Field Attention',
        'improvements_over_condJ': {
            '1_relative_pos_bias':     '44 tap offsets × 8 heads = 352 params/layer',
            '2_elu_normalizer':        'φ(x)=elu(x)+1; Z=Σ|h_τ|φ(Q·K); out/=Z',
            '3_no_interference_pool':  'cumulative-mean blocks removed (ablation)',
            '4_rg_scale_init':         'early layers biased j0-j3; late layers j8-j10',
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF ATTENTION — CONDITION K')
    print('  Dyadic Wave And Resonant Field Attention')
    print('  condJ + rel-pos-bias + ELU normalizer + RG init + no interference')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits    = load_data(NUM_DOCS)
    tok_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '2048_condI_tokenizer.json')
    if os.path.exists(tok_path):
        print(f'\nLoading condI BPE tokenizer (shared for comparability)...')
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    else:
        print(f'\nTraining BPE tokenizer (vocab={VOCAB_SIZE})...')
        raw_tok   = train_bpe_tokenizer(splits['train'][:BPE_TRAIN_DOCS])
        raw_tok.save(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  Vocab: {tokenizer.vocab_size()} tokens')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = DWARFTransformer(
        vocab_size     = tokenizer.vocab_size(),
        embedding_dim  = EMBEDDING_DIM,
        num_layers     = NUM_LAYERS,
        num_heads      = NUM_HEADS,
        ffn_dim        = FFN_DIM,
        seq_len        = MAX_SEQ_LEN,
        n_scales       = N_SCALES,
    ).to(device)

    hd       = EMBEDDING_DIM // NUM_HEADS
    n_params = model.param_count()
    print(f'\nDWARF condK: {n_params:,} parameters')
    print(f'  Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads × {EMBEDDING_DIM}d')
    print(f'  Gather: Σ_{{j,τ}} gain_j · φ(Q·K_{{shifted}} + bias_{{j,τ}}) · V_{{shifted}} / Z')
    print(f'  Feature map: φ(x) = elu(x) + 1  (always positive)')
    print(f'  Rel-pos-bias: {N_SCALES * N_TAPS} taps × {NUM_HEADS} heads × {NUM_LAYERS} layers '
          f'= {N_SCALES * N_TAPS * NUM_HEADS * NUM_LAYERS:,} params')
    print(f'  Interference pooling: REMOVED (ablation test)')
    print(f'  Scale init: RG-inspired (early→local, late→global)')
    print(f'  Identity bypass: {NUM_HEADS * NUM_LAYERS} params (softplus per head per layer)')

    if not causality_check(model, device): return

    results = train(
        model, train_data, val_data, test_data, tokenizer,
        save_dir='2048_condK_checkpoints', device=device)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condK_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
