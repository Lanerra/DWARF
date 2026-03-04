"""
DWARF Attention — Condition L (DSQG)
Dyadic Sparse-Query Gated Attention

condL = condK's architecture with D4-propagated c_δ replaced by softmax(Q·K_δ/√d)

WHAT CONDL TESTS
----------------
condK (and condJ before it) uses the OPWF gather:
  output_t = Σ_δ  c_δ · V_{t-δ}
where c_δ comes from D4 convolution (spatial/structural).

condL replaces c_δ with:
  c_δ = softmax({Q_t · K_{t-δ'} / √d  for all δ'})_δ   (semantic)

This is formally: sparse softmax attention with a fixed set of 44 dyadic offset
positions. The same 44 (j, τ) tap positions as condK — just semantic rather than
structural weights.

WHY THIS MATTERS — THE SOFTMAX GAP (CAPACITY THEOREM)
------------------------------------------------------
Ramsauer et al. (NeurIPS 2020) proved that modern Hopfield networks = softmax
attention and have exponential associative memory capacity. Standard linear
attention (and DWARF's OPWF gather) has only linear capacity (Schlag et al.,
ICML 2021 formally proves DWARF = Fast Weight Programmer with linear capacity).

DWARF condK PPL gap vs standard transformer = 20.2 PPL.
Rust rank_bottleneck tests show rank collision explains only ~0.5% of that gap.
The dominant cause is structural: softmax vs. linear capacity.

condL directly tests whether replacing the linear-capacity gather with a
softmax-capacity gather at the same 44 positions closes the gap.

ARCHITECTURE DIFFERENCES vs condK
----------------------------------
  REMOVED:  D4 scale gains (replaced by softmax weights — zero extra params)
  REMOVED:  ELU feature map φ(x) = elu(x)+1 (redundant with softmax normalization)
  REMOVED:  ELU normalizer Z (softmax already sums to 1)
  REMOVED:  Q-conditioned scale projection q_scale_proj
  REMOVED:  identity_bypass (offset=0 is naturally included in 44 softmax offsets)
  KEPT:     Position bias pos_bias[44, H] (biases which offsets are attended to)
  KEPT:     Causal padding for each offset
  KEPT:     Gate proj (output gating)
  KEPT:     No interference pooling
  KEPT:     Same VOCAB, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, SEQ_LEN

POSITION BIAS INITIALIZATION
-----------------------------
Initialized to zero (flat prior). RG-analogous init confirmed irrelevant by
condl_ablation Rust test (washes out in 27 gradient steps).

Run:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 -u benchmarks/train_2048_condL.py \
    2>&1 | tee benchmarks/logs/condL_run.log

Run in parallel with condK_pooling on CUDA_VISIBLE_DEVICES=0 (RTX 4090)

Results → benchmarks/2048_condL_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters (identical to condK for comparability) ──────────────────

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
N_TAPS          = 4         # tap positions per scale (τ=0..3)
N_OFFSETS       = N_SCALES * N_TAPS   # 44 total (j, τ) pairs

# ─── Precompute dyadic offset table ──────────────────────────────────────────

def _build_offset_table(n_scales, n_taps):
    """Build list of (j, tau, offset) for all 44 dyadic tap positions.

    offset = (1 << j) * tau.
    Note: offset=0 appears 11 times (tau=0 for each j), each with its own
    position bias. The model learns to distribute weight across them.
    """
    table = []
    for j in range(n_scales):
        d = 1 << j
        for tau in range(n_taps):
            table.append((j, tau, d * tau))
    return table

OFFSET_TABLE = _build_offset_table(N_SCALES, N_TAPS)   # 44 entries


# ─── DSQG Attention ───────────────────────────────────────────────────────────

class DSQGAttention(nn.Module):
    """
    DSQG — Dyadic Sparse-Query Gated Attention.

    Computes sparse softmax attention over 44 dyadic offset positions:

      score_{t,δ}  = Q_t · K_{t-δ} / √d  +  pos_bias[tap_idx, head]
      attn_{t,δ}   = softmax(score_{t,:})   over 44 offsets
      output_t     = Σ_δ  attn_{t,δ} · V_{t-δ}

    Key properties:
      - c_δ comes from semantic Q·K match (exponential capacity) vs condK's spatial D4
      - No ELU feature map or normalizer (softmax handles normalization)
      - Position biases allow learned structural preference over offsets
      - Causal: all offsets ≥ 0, so only past (or present) tokens contribute
      - Large offsets (≥ N) contribute zero keys/values; model learns to down-weight
    """

    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 n_scales=N_SCALES, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len
        self.n_scales      = n_scales
        self.scale         = self.head_dim ** -0.5

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Position bias: one scalar per (j, tau) tap per head.
        # Initialized to 0 — flat prior, no structural preference at init.
        self.pos_bias = nn.Parameter(
            torch.zeros(N_OFFSETS, num_heads))  # [44, H]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H   = self.num_heads
        HD  = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # ── Pass 1: compute all 44 Q·K scores ─────────────────────────────
        # scores: [B, H, N, 44]
        scores = q.new_zeros(B, H, N, N_OFFSETS)

        for tap_idx, (j, tau, offset) in enumerate(OFFSET_TABLE):
            if offset == 0:
                k_s = k
            elif offset >= N:
                # All-zero keys for out-of-sequence offsets → Q·K = 0
                k_s = k.new_zeros(B, H, N, HD)
            else:
                pad = k.new_zeros(B, H, offset, HD)
                k_s = torch.cat([pad, k[:, :, :N - offset]], dim=2)

            # Q·K score at this offset (scaled)
            # pos_bias[tap_idx]: [H] → broadcast to [1, H, 1]
            qk = (q * k_s).sum(-1)                            # [B, H, N]
            pb = self.pos_bias[tap_idx].view(1, H, 1)         # [1, H, 1]
            scores[:, :, :, tap_idx] = qk * self.scale + pb

        # Causal masking for padding positions: positions that are "in the future"
        # or were zero-padded already have zero keys, so Q·K = pos_bias for them.
        # We do NOT apply an explicit -inf mask beyond padding because offset≥N is
        # already handled by zero keys above, and all other offsets are ≥0 (past).

        # Softmax over 44 offsets → attention weights
        attn_weights = F.softmax(scores, dim=-1)   # [B, H, N, 44]

        # ── Pass 2: weighted sum of V at each offset ───────────────────────
        # out: [B, H, N, HD]
        out = q.new_zeros(B, H, N, HD)

        for tap_idx, (j, tau, offset) in enumerate(OFFSET_TABLE):
            if offset == 0:
                v_s = v
            elif offset >= N:
                continue  # attn_weight is small (from near-zero scores); skip
            else:
                pad = v.new_zeros(B, H, offset, HD)
                v_s = torch.cat([pad, v[:, :, :N - offset]], dim=2)

            # w: [B, H, N, 1] * v_s: [B, H, N, HD] → contribution
            w   = attn_weights[:, :, :, tap_idx].unsqueeze(-1)   # [B, H, N, 1]
            out = out + w * v_s

        # Gate and output projection
        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def scale_summary(self):
        """Summarize learned position biases and inferred scale preferences."""
        with torch.no_grad():
            pb = self.pos_bias  # [44, H]
            pb_mean = pb.abs().mean().item()
            pb_max  = pb.abs().max().item()

            # Average pos bias per scale (mean over 4 taps and all heads)
            scale_biases = []
            for j in range(self.n_scales):
                start = j * N_TAPS
                scale_biases.append(pb[start:start + N_TAPS].mean().item())

            # Which scale has the highest average position bias (structural preference)?
            dom = int(max(range(self.n_scales), key=lambda j: scale_biases[j]))

        return {
            'pos_bias_abs_mean':  pb_mean,
            'pos_bias_abs_max':   pb_max,
            'pos_bias_per_scale': scale_biases,
            'dominant_scale':     dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'attn_type': 'softmax over 44 dyadic offsets (DSQG)',
        }


# ─── FFN ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ─── DWARFBlock (no interference pooling) ────────────────────────────────────

class DWARFBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttention(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        if self.use_checkpoint:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_fn, x, use_reentrant=False)
        else:
            x = x + self._attn_fn(x)
        return x + self.ffn(self.norm2(x))


# ─── Transformer ─────────────────────────────────────────────────────────────

class DSQGTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DWARFBlock(embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                       dropout=dropout, use_checkpoint=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # gate_proj bias initialized to 2.0 in DSQGAttention __init__
        # pos_bias initialized to 0.0 in DSQGAttention __init__ (flat prior)

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
        avg_scale_biases = [
            sum(s['pos_bias_per_scale'][j] for s in summaries) / n
            for j in range(N_SCALES)
        ]
        dom = int(max(range(N_SCALES), key=lambda j: avg_scale_biases[j]))
        return {
            'pos_bias_abs_mean':     sum(s['pos_bias_abs_mean'] for s in summaries) / n,
            'pos_bias_abs_max':      max(s['pos_bias_abs_max']  for s in summaries),
            'pos_bias_per_scale':    avg_scale_biases,
            'dominant_scale':        dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'attn_type':             'DSQG (softmax over 44 dyadic offsets)',
        }


# ─── Data utilities ───────────────────────────────────────────────────────────

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
        if (i + 1) % 25_000 == 0: print(f'  {i+1:,} docs...')
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
    n    = (len(tokens) // max_seq_len) * max_seq_len
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


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
        gen = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


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
          save_dir='2048_condL_checkpoints', device='cuda'):
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
                if epoch == 1 and step == 200 and loss_val > 9.0:
                    print(f'  WARNING: ep1 step-200 loss {loss_val:.4f} > 9.0 — check stability')

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
        biases = ss['pos_bias_per_scale']
        top3   = sorted(range(N_SCALES), key=lambda j: -biases[j])[:3]
        print('  Pos-bias scale preference (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok bias={biases[j]:+.3f}' for j in top3))
        print(f'  Pos-bias |mean|={ss["pos_bias_abs_mean"]:.4f}  |max|={ss["pos_bias_abs_max"]:.4f}')

        print('  ── Generation samples ──')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device)):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')
        sys.stdout.flush()

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  DWARF condL (DSQG) TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    ss = model.scale_summary()
    print(f'\n  Learned scale preference (pos_bias averages):')
    for j in range(N_SCALES):
        b = ss['pos_bias_per_scale'][j]
        bar = '█' * max(0, int((b + 2.0) * 5)) + '░' * max(0, int((2.0 - b) * 5))
        print(f'    j{j:>2} (rf={4*(1<<j):>5} tok): {b:+.4f}  {bar}')

    print('\n' + '=' * 70)
    print('  DSQG vs ABLATION TABLE — SOFTMAX GAP TEST')
    print('=' * 70)
    table = [
        ('Standard transformer 13M (reference)',        64.5),
        ('Wave V4 [A] (OPWF, G=4096, D4-conv)',         86.8),
        ('OPWF efficient Q·K [J]',                      85.1),
        ('DWARF condK (J + pos_bias + RG + no pool)',   84.7),
        ('condK capacity gap to standard:',             20.2),
    ]
    for name, val in table:
        if 'gap' in name.lower():
            print(f'  {name:<52} {val:>+6.1f}')
        else:
            print(f'  {name:<52} {val:>6.1f}')
    print(f'  {"DWARF condL (DSQG — this run)":<52} {test_ppl:>6.1f}')
    delta = test_ppl - 84.7
    gap_closed = 20.2 - (test_ppl - 64.5)
    print(f'  {"  Δ vs condK:":<52} {delta:>+6.1f}')
    print(f'  {"  Gap remaining vs standard:":<52} {test_ppl - 64.5:>6.1f}')
    print(f'  {"  Gap closed by DSQG:":<52} {gap_closed:>+6.1f}')

    if test_ppl < 84.7:
        print(f'\n  → SOFTMAX GAP CONFIRMED: DSQG improves over OPWF by {84.7 - test_ppl:.1f} PPL')
        print(f'    Semantic Q·K weighting provides more than structural D4 weighting')
    elif test_ppl > 85.5:
        print(f'\n  → SURPRISING: DSQG is worse than OPWF by {test_ppl - 84.7:.1f} PPL')
        print(f'    D4 structure may be important beyond just normalization')
    else:
        print(f'\n  → COMPARABLE: DSQG ≈ OPWF (delta = {delta:+.1f} PPL)')
        print(f'    Softmax gap is real but small at this scale; D4 and softmax roughly equivalent')

    return {
        'test_ppl':           test_ppl,
        'test_loss':          test_loss,
        'best_val_ppl':       best_val_ppl,
        'best_epoch':         best_epoch,
        'total_time_s':       time.time() - t0,
        'scale_summary':      ss,
        'condK_baseline_ppl': 84.7,
        'delta_vs_condK':     test_ppl - 84.7,
        'gap_closed_vs_std':  20.2 - (test_ppl - 64.5),
        'architecture_name':  'condL — DSQG (Dyadic Sparse-Query Gated Attention)',
        'key_difference':     'softmax(Q·K_delta/sqrt(d)) over 44 dyadic offsets instead of D4-propagated ELU gather',
        'hypothesis':         'softmax gap (exponential vs linear capacity) is primary PPL bottleneck',
        'n_offsets':          N_OFFSETS,
        'offset_table_unique_count': len(set(offset for _, _, offset in OFFSET_TABLE)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF condL — DSQG (Dyadic Sparse-Query Gated Attention)')
    print('  Test: does replacing D4 c_δ with softmax(Q·K_δ) close the gap?')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    print(f'\n  Architecture comparison (condK → condL):')
    print(f'    c_δ source:      D4 convolution (spatial) → softmax(Q·K_δ) (semantic)')
    print(f'    ELU feature map: φ(x)=elu(x)+1 → REMOVED (softmax normalizes)')
    print(f'    ELU normalizer:  Z = Σ gain·φ   → REMOVED (softmax sums to 1)')
    print(f'    Q-scale gains:   q_scale_proj    → REMOVED (no per-scale gains)')
    print(f'    Identity bypass: separate param  → REMOVED (offset=0 in softmax)')
    print(f'    Position bias:   KEPT (44 taps × 8 heads per layer = 352 params/layer)')
    print(f'    Gate proj:       KEPT')
    print(f'    Pooling:         NONE (same as condK)')
    print(f'\n  Offset table: {N_OFFSETS} (j,τ) pairs, {len(set(o for _,_,o in OFFSET_TABLE))} unique offsets')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits   = load_data(NUM_DOCS)
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '2048_condI_tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'\nLoaded condI BPE tokenizer (shared vocabulary)')
    else:
        print('\nTraining BPE tokenizer...')
        raw_tok   = train_bpe_tokenizer(splits['train'][:BPE_TRAIN_DOCS])
        raw_tok.save(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  Vocab: {tokenizer.vocab_size()} tokens')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = DSQGTransformer(
        vocab_size     = tokenizer.vocab_size(),
        embedding_dim  = EMBEDDING_DIM,
        num_layers     = NUM_LAYERS,
        num_heads      = NUM_HEADS,
        ffn_dim        = FFN_DIM,
        seq_len        = MAX_SEQ_LEN,
        n_scales       = N_SCALES,
    ).to(device)

    n_params = model.param_count()
    print(f'\nDWARF condL (DSQG): {n_params:,} parameters')
    print(f'  Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads × {EMBEDDING_DIM}d')
    print(f'  Attention: softmax over {N_OFFSETS} dyadic offsets (Q·K_δ/√d + pos_bias_δ)')
    print(f'  Position bias: {N_OFFSETS} × {NUM_HEADS} × {NUM_LAYERS} = '
          f'{N_OFFSETS * NUM_HEADS * NUM_LAYERS:,} params')

    # Compare param counts
    cond_k_params = 84_487_456
    print(f'  Δ params vs condK: {n_params - cond_k_params:+,}')
    print(f'  (condK had scale_gain, q_scale_proj, identity_bypass; condL has only pos_bias)')

    if not causality_check(model, device): return

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir='2048_condL_checkpoints', device=device)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condL_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
