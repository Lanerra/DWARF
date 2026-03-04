"""
DWARF Attention — Condition L+RP (Revised DSQG + Interference Pooling)

condL+RP = condL redesigned with three fixes identified from the epoch-1 condL run:

  FIX 1 — DEDUPLICATED OFFSETS (44 → 24 unique)
     Original condL used all 44 (j,τ) pairs, but offset=0 appears 11 times
     (once per scale, τ=0), eating into the softmax budget with 11 slots all
     competing to "attend to current token." Many other offsets are also
     duplicated across scales (e.g., δ=4 from both j=1τ=2 and j=2τ=1).
     condL+RP uses 24 unique offsets: {0,1,2,3,4,6,8,12,...,1536,2048,3072}
     — one softmax slot per distinct past position.

  FIX 2 — D4-WARM-STARTED POSITION BIASES
     Original condL initialized pos_bias=0 (flat prior). The model learned
     by epoch 1 that short-range matters (j=0 dominated at +0.247) but paid
     a convergence tax to discover what D4 gives for free: nearby tokens
     (δ=1 has D4 coeff h1=0.837, largest) carry the most signal.
     condL+RP initializes pos_bias[δ] using the max D4 weight at each offset
     (|D4[τ]|/√2^j), log-scaled and centered. This is a prior, not a constraint
     — the model freely learns to override it.

     Warm-start profile (24 offsets, max |bias|=0.30):
       δ=1   +0.300  (h1=0.837, scale j=0 → highest weight)
       δ=2   +0.261  (h1 at j=1)
       δ=0   +0.238  (h0 at j=0)
       ...
       δ=1536 −0.261  (h1 at j=9, highly attenuated)
       δ=3072 −0.300  (h1 at j=10, most attenuated)

  FIX 3 — INTERFERENCE POOLING RESTORED
     condK+RP showed consistent +3 PPL advantage over condK (which removed
     pooling). Pooling provides global context the DWARF field doesn't cover.
     condL+RP includes it at the same interval (every 3rd layer = blocks 2, 5).

WHAT CONDL+RP TESTS
-------------------
Same core hypothesis as condL: does softmax(Q·K_δ/√d) over dyadic offsets
outperform D4-propagated ELU gather? But now the test is fair:
  - Same structural scale prior (D4 warm-start vs D4 gains in condK+RP)
  - Same global pooling (interference blocks in both)
  - Cleaner softmax budget (24 unique offsets, no duplicate slots)

If condL+RP ≈ condK+RP: softmax capacity and D4 structure are equivalent at 13M
If condL+RP > condK+RP: softmax gap hypothesis confirmed (capacity wins)
If condL+RP < condK+RP: D4 propagation carries information beyond scale prior

Architecture vs condK+RP:
  REMOVED:  D4 convolution propagation (c_δ from D4 → from softmax)
  REMOVED:  Q-conditioned scale gains (q_scale_proj)
  REMOVED:  ELU feature map and normalizer
  REMOVED:  Identity bypass (offset=0 included in 24 unique softmax positions)
  ADDED:    24-offset softmax attention
  KEPT:     Position bias (24 per head per layer, D4-warm-started)
  KEPT:     Interference pooling every 3rd layer
  KEPT:     Gate proj, RG-analogous pos_bias init, no D4 gains in attention

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condLRP.py \
    2>&1 | tee benchmarks/logs/condLRP_run.log

Results → benchmarks/2048_condLRP_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters (identical to all 13M conditions) ───────────────────────

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
INTERFERENCE    = 3       # interference pooling every 3rd layer

# ─── Dyadic offset table (24 unique offsets) ─────────────────────────────────

_D4 = [0.4829629131445341, 0.8365163037378079,
       0.2241438680420134,  0.1294095225512604]   # abs values used for warm-start

def _build_unique_offset_table():
    """
    Build sorted list of unique dyadic offsets from 11 scales × 4 taps.
    Returns (offsets, d4_warmstart_biases).

    Warm-start: for each unique offset δ, find the maximum D4-weighted signal
    across all (j,τ) pairs that generate it:
        weight[δ] = max_{j,τ: (1<<j)*τ=δ} |D4[τ]| / sqrt(2^j)

    Then: pos_bias_init[δ] = log(weight[δ]) centered and scaled to max|bias|=0.30
    """
    offsets_set = set()
    for j in range(11):
        d = 1 << j
        for tau in range(4):
            offsets_set.add(d * tau)
    offsets = sorted(offsets_set)   # 24 unique values

    # Compute D4-inspired weight per unique offset
    weights = []
    for delta in offsets:
        best_w = 0.0
        for j in range(11):
            d = 1 << j
            for tau in range(4):
                if d * tau == delta:
                    w = _D4[tau] / (2.0 ** (j / 2.0))
                    best_w = max(best_w, w)
        weights.append(max(best_w, 1e-6))

    # Log-scale, center, scale to max|bias|=0.30
    log_w     = [math.log(w) for w in weights]
    mean_lw   = sum(log_w) / len(log_w)
    centered  = [lw - mean_lw for lw in log_w]
    max_abs   = max(abs(c) for c in centered)
    scale     = 0.30 / max_abs if max_abs > 0 else 1.0
    biases    = [c * scale for c in centered]

    return offsets, biases

UNIQUE_OFFSETS, D4_WARMSTART_BIASES = _build_unique_offset_table()
N_OFFSETS = len(UNIQUE_OFFSETS)   # 24


# ─── DSQG Attention (revised: 24 unique offsets + D4 warm-start) ─────────────

class DSQGAttention(nn.Module):
    """
    Revised DSQG — Dyadic Sparse-Query Gated Attention.

    Softmax attention over 24 unique dyadic offset positions:
      score_{t,i}  = Q_t · K_{t - δ_i} / √d  +  pos_bias[i, head]
      attn_{t,i}   = softmax(score_{t,:})  over 24 offsets
      output_t     = Σ_i  attn_{t,i} · V_{t - δ_i}

    pos_bias initialized from D4 structure (short-range positive, long-range negative).
    """

    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.scale         = self.head_dim ** -0.5

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Position bias: one scalar per unique offset per head
        # Initialized from D4 warm-start (not zero)
        self.pos_bias = nn.Parameter(torch.zeros(N_OFFSETS, num_heads))

        self.dropout = nn.Dropout(dropout)

    def _init_pos_bias(self):
        """Apply D4 warm-start to pos_bias. Called after __init__."""
        data = torch.zeros_like(self.pos_bias)
        for i, bias_val in enumerate(D4_WARMSTART_BIASES):
            data[i, :] = bias_val
        with torch.no_grad():
            self.pos_bias.copy_(data)

    def forward(self, x):
        B, N, D = x.shape
        H  = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # ── Pass 1: compute scores at all 24 unique offsets ───────────────
        scores = q.new_zeros(B, H, N, N_OFFSETS)

        for tap_idx, delta in enumerate(UNIQUE_OFFSETS):
            if delta == 0:
                k_s = k
            elif delta >= N:
                # Out-of-range: zero keys → Q·K=0; neg init bias handles this
                k_s = k.new_zeros(B, H, N, HD)
            else:
                pad = k.new_zeros(B, H, delta, HD)
                k_s = torch.cat([pad, k[:, :, :N - delta]], dim=2)

            pb = self.pos_bias[tap_idx].view(1, H, 1)   # [1, H, 1]
            scores[:, :, :, tap_idx] = (q * k_s).sum(-1) * self.scale + pb

        attn = F.softmax(scores, dim=-1)   # [B, H, N, 24]

        # ── Pass 2: weighted sum of V ─────────────────────────────────────
        out = q.new_zeros(B, H, N, HD)

        for tap_idx, delta in enumerate(UNIQUE_OFFSETS):
            if delta == 0:
                v_s = v
            elif delta >= N:
                continue
            else:
                pad = v.new_zeros(B, H, delta, HD)
                v_s = torch.cat([pad, v[:, :, :N - delta]], dim=2)

            w   = attn[:, :, :, tap_idx].unsqueeze(-1)   # [B, H, N, 1]
            out = out + w * v_s

        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def scale_summary(self):
        with torch.no_grad():
            pb = self.pos_bias
            top3_idx = pb.mean(dim=-1).topk(3).indices.tolist()
            bot3_idx = pb.mean(dim=-1).topk(3, largest=False).indices.tolist()
        return {
            'pos_bias_abs_mean':  pb.abs().mean().item(),
            'pos_bias_abs_max':   pb.abs().max().item(),
            'top3_offsets':       [UNIQUE_OFFSETS[i] for i in top3_idx],
            'top3_biases':        [pb[i].mean().item() for i in top3_idx],
            'bot3_offsets':       [UNIQUE_OFFSETS[i] for i in bot3_idx],
            'bot3_biases':        [pb[i].mean().item() for i in bot3_idx],
            'n_offsets':          N_OFFSETS,
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


# ─── DWARFBlock with optional interference pooling ───────────────────────────

class DWARFBlock(nn.Module):
    """DSQG transformer block with optional cumulative-mean interference pooling."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttention(embedding_dim, num_heads,
                                   seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_pool = nn.Linear(embedding_dim, embedding_dim)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        if self.use_checkpoint:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_fn, x, use_reentrant=False)
        else:
            x = x + self._attn_fn(x)

        if self.interference:
            xi      = self.inter_norm(x)
            B, N, D = xi.shape
            counts  = torch.arange(1, N + 1, device=xi.device,
                                   dtype=xi.dtype).view(1, N, 1)
            pool    = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)

        return x + self.ffn(self.norm2(x))


# ─── Transformer ─────────────────────────────────────────────────────────────

class DSQGTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, interference_interval=INTERFERENCE,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DWARFBlock(
                embedding_dim, num_heads, ffn_dim, seq_len,
                dropout=dropout, use_checkpoint=True,
                interference=(i % interference_interval == interference_interval - 1))
            for i in range(num_layers)
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
        # gate_proj bias = 2.0 set in DSQGAttention.__init__
        # Apply D4 warm-start to pos_bias in all blocks
        for block in self.blocks:
            block.attn._init_pos_bias()

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
        n  = len(summaries)
        return {
            'pos_bias_abs_mean': sum(s['pos_bias_abs_mean'] for s in summaries) / n,
            'pos_bias_abs_max':  max(s['pos_bias_abs_max']  for s in summaries),
            'top3_offsets':      summaries[0]['top3_offsets'],   # layer 0 as representative
            'top3_biases':       summaries[0]['top3_biases'],
            'bot3_offsets':      summaries[0]['bot3_offsets'],
            'bot3_biases':       summaries[0]['bot3_biases'],
            'n_offsets':         N_OFFSETS,
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
          save_dir='2048_condLRP_checkpoints', device='cuda'):
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
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

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
        print(f'  Top-3 attended offsets: ' +
              ', '.join(f'δ={o} ({b:+.3f})' for o, b in
                        zip(ss['top3_offsets'], ss['top3_biases'])))
        print(f'  Bot-3 attended offsets: ' +
              ', '.join(f'δ={o} ({b:+.3f})' for o, b in
                        zip(ss['bot3_offsets'], ss['bot3_biases'])))
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

    ss = model.scale_summary()
    print(f'\n  condL+RP TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    print('\n' + '=' * 70)
    print('  SOFTMAX GAP TEST — condL+RP vs condK+RP')
    print('=' * 70)
    table = [
        ('Standard transformer 13M',                64.5),
        ('DWARF condK (no pooling)',                 84.7),
        ('DWARF condK+RP (pooling restored)',        '?'),
        ('condL (original, flat init, 44 offsets)', 207.2),
    ]
    for name, ppl in table:
        if isinstance(ppl, str):
            print(f'  {name:<52} {"(training)":>6}')
        else:
            print(f'  {name:<52} {ppl:>6.1f}')
    print(f'  {"condL+RP (this run — revised DSQG + pooling)":<52} {test_ppl:>6.1f}')

    condKRP_ppl = 84.7   # update when condK+RP finishes
    delta = test_ppl - condKRP_ppl
    print(f'\n  vs condK+RP: {delta:+.1f} PPL')
    if test_ppl < condKRP_ppl - 1.0:
        print('  → SOFTMAX GAP CONFIRMED at 13M scale: semantic > spatial weighting')
    elif test_ppl > condKRP_ppl + 1.0:
        print('  → D4 structure still wins: spatial propagation > softmax at 13M')
    else:
        print('  → COMPARABLE: softmax and D4 structure equivalent at 13M scale')

    print(f'\n  Pos-bias (learned), top-3 preferred offsets (layer 0):')
    for o, b in zip(ss['top3_offsets'], ss['top3_biases']):
        print(f'    δ={o:>5} (rf={o+1:>5} tokens): {b:+.4f}')

    return {
        'test_ppl':           test_ppl,
        'test_loss':          test_loss,
        'best_val_ppl':       best_val_ppl,
        'best_epoch':         best_epoch,
        'total_time_s':       time.time() - t0,
        'scale_summary':      ss,
        'architecture':       'condL+RP — revised DSQG with D4 warm-start + interference pooling',
        'fixes_vs_condL':     {
            '1_unique_offsets':    f'{N_OFFSETS} unique vs 44 with duplicates',
            '2_d4_warmstart':      'pos_bias init from D4 weights, max|bias|=0.30',
            '3_pooling_restored':  f'cumulative-mean interference every {INTERFERENCE}rd layer',
        },
        'unique_offsets':     UNIQUE_OFFSETS,
        'd4_warmstart_biases': D4_WARMSTART_BIASES,
        'hypothesis':         'fair softmax gap test vs condK+RP',
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF condL+RP — Revised DSQG + Interference Pooling')
    print('  Fair softmax gap test vs condK+RP')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    print(f'\n  Fixes vs original condL:')
    print(f'    1. {N_OFFSETS} unique offsets (was 44 with 11× duplicate δ=0)')
    print(f'    2. D4 warm-start pos_bias (max|bias|=0.30, short-range positive)')
    print(f'    3. Interference pooling every {INTERFERENCE}rd layer (blocks 2, 5)')

    print(f'\n  D4 warm-start profile ({N_OFFSETS} offsets):')
    for delta, bias in zip(UNIQUE_OFFSETS, D4_WARMSTART_BIASES):
        bar_len = int(abs(bias) * 20)
        bar = ('+' if bias >= 0 else '-') * bar_len
        print(f'    δ={delta:>5}: {bias:+.3f}  {bar}')

    os.makedirs('benchmarks/logs', exist_ok=True)
    splits   = load_data(NUM_DOCS)
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '2048_condI_tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'\nLoaded shared condI BPE tokenizer')
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
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    pooling_blocks = [i for i in range(NUM_LAYERS)
                      if i % INTERFERENCE == INTERFERENCE - 1]
    print(f'\ncondL+RP: {n_params:,} parameters')
    print(f'  {NUM_LAYERS} layers × {NUM_HEADS} heads × {EMBEDDING_DIM}d')
    print(f'  Attention: softmax over {N_OFFSETS} unique dyadic offsets')
    print(f'  Pooling blocks: {pooling_blocks}')
    print(f'  Pos-bias: {N_OFFSETS} offsets × {NUM_HEADS} heads × {NUM_LAYERS} layers '
          f'= {N_OFFSETS * NUM_HEADS * NUM_LAYERS:,} params')

    if not causality_check(model, device): return

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir='2048_condLRP_checkpoints', device=device)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condLRP_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
