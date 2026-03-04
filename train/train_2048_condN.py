"""
DWARF Attention — Condition N: Dense-32 Local + Dyadic Long-Range DSQG

PURPOSE
-------
condK/condK+RP use D4 DWT propagation, a structured linear attention mechanism.
The ~20 PPL gap vs standard transformer has three sources:
  1. Linearity (~12–15 PPL): no softmax competition; linear in Q·K
  2. Sparsity (~3–7 PPL): 24 unique dyadic offsets, ~35% attention mass
  3. K-blind routing (~2–3 PPL): w(n,m) ignores K[m]

condL+RP (DSQG) fixes sources 1 and 3 by replacing the linear gather with
softmax over dyadic offsets. condN fixes all three:

condN ARCHITECTURE: Dense-32 + Dyadic Long-Range DSQG
  - Dense local: offsets {0, 1, 2, ..., 32}   (33 positions — fills all gaps)
  - Dyadic long-range: {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536}
  - Total: 44 unique offsets (same count as condK D4 taps, same compute)
  - Attention: standard softmax over all 44 positions → Q·K competitive selection
  - Position bias: ALiBi-style per-head decay  b[δ,h] = -α_h · ln(1+δ)
    where α_h ∈ [0.2, 2.0] (uniform across 8 heads)
    → head 0 attends globally; head 7 is strongly local-biased
    → structural head diversity from init (no convergence tax)
  - Interference pooling: every 3rd layer (same as condK+RP)

COVERAGE ANALYSIS (1/δ attention prior)
  Dyadic only (condL+RP):    24 offsets → ~35% attention mass
  Dense-32 + dyadic (condN): 44 offsets → ~55% attention mass  (same compute)
  Full attention:            2047 offsets → 100% attention mass

WHAT'S REMOVED vs condK+RP
  - K⊗V outer product field (not needed — DSQG reads K and V directly)
  - D4 DWT propagation (replaced by direct offset indexing)
  - Q-conditioned scale gains (absorbed into softmax Q·K scores)
  - ELU normalizer (softmax provides its own normalization)
  - identity_bypass parameter (offset=0 is part of the 44-offset softmax)

WHAT'S PRESERVED vs condK+RP
  - Output sigmoid gate
  - Interference pooling (every 3rd layer, condK+RP style)
  - Same tokenizer, same data, same hyperparameters → fair comparison

ABLATION QUESTIONS
  condN PPL vs condK+RP (~81–82): How much of the gap does dense local fill?
  condN PPL vs condL+RP: Dense coverage advantage over pure dyadic
  condN PPL vs standard transformer (64.5): Remaining gap after both fixes

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condN.py \
    2>&1 | tee benchmarks/logs/condN_run.log

Results → benchmarks/2048_condN_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters (identical to condK+RP for fair comparison) ──────────────

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
INTERFERENCE    = 3        # interference pooling every 3rd layer

# ─── condN offset set (44 unique offsets) ─────────────────────────────────────
#   Dense local:  {0, 1, ..., 32}  (33 offsets)
#   Dyadic LR:    {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536}  (11)
#   Total: 44 — same count as condK D4 taps

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44, f"Expected 44 offsets, got {len(_COND_N_OFFSETS)}"


# ─── DSQG Attention (condN) ──────────────────────────────────────────────────

class DSQGAttentionN(nn.Module):
    """
    Dense-32 + Dyadic Long-Range DSQG Attention (condN).

    Softmax over 44 unique offsets:
      {0..32} ∪ {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536}

    Position bias init (ALiBi-style, per-head):
      b[δ, h] = -α_h · ln(1 + δ)
      α_h uniformly spaced in [0.2, 2.0] across heads
      → head 0: gentle global decay; head 7: strong local preference
    """

    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len

        if offsets is None:
            offsets = _COND_N_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # ALiBi-style position bias: shape (n_offsets, num_heads)
        # b[δ_idx, h] = -α_h * ln(1 + δ)
        alphas = torch.linspace(0.2, 2.0, num_heads)   # H,
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets], dtype=torch.float32)  # n_offsets,
        # outer product: pos_bias[i, h] = -alphas[h] * log(1 + offsets[i])
        pos_bias_init = -delta_vals.unsqueeze(1) * alphas.unsqueeze(0)  # n_offsets, H
        self.pos_bias = nn.Parameter(pos_bias_init)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # B, H, N, HD
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scale = HD ** -0.5

        # ── Build shifted K and V tensors ─────────────────────────────────────
        # K_list[i], V_list[i]: shifted by offsets[i], zero-padded at start
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k)
                V_list.append(v)
            elif delta >= N:
                # Offset larger than sequence; will be fully masked
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad   = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))

        # Stack: B, H, N, n_offsets, HD
        K_all = torch.stack(K_list, dim=3)   # B, H, N, n_offsets, HD
        V_all = torch.stack(V_list, dim=3)

        # ── Q·K scores: B, H, N, n_offsets ───────────────────────────────────
        # q: B, H, N, HD  →  unsqueeze dim=3  →  B, H, N, 1, HD
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale   # B, H, N, n_offsets

        # Add learnable position bias (broadcast over B, N)
        # pos_bias: n_offsets, H  →  1, H, 1, n_offsets
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)

        # ── Causal mask: position n cannot attend to offset δ if n < δ ────────
        # n_idx: N, 1   d_idx: 1, n_offsets
        n_idx = torch.arange(N, device=x.device).unsqueeze(1)         # N, 1
        d_idx = self.offsets.unsqueeze(0)                              # 1, n_offsets
        causal_invalid = (n_idx < d_idx)                              # N, n_offsets
        scores = scores.masked_fill(
            causal_invalid.unsqueeze(0).unsqueeze(0), float('-inf'))

        # ── Softmax: competitive selection over offsets ───────────────────────
        alpha = F.softmax(scores, dim=-1)   # B, H, N, n_offsets

        # ── Gather V: B, H, N, HD ─────────────────────────────────────────────
        # alpha: B, H, N, n_offsets → unsqueeze → B, H, N, n_offsets, 1
        out = (alpha.unsqueeze(-1) * V_all).sum(dim=3)   # B, H, N, HD

        # ── Output gate ───────────────────────────────────────────────────────
        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def attn_summary(self):
        """Return diagnostics about learned position bias structure."""
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()   # n_offsets, H
        return {
            'pos_bias_abs_mean': pb.abs().mean().item(),
            'pos_bias_abs_max':  pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),    # H
            'pos_bias_delta0':   pb[0].tolist(),              # bias at offset 0 (self)
            'pos_bias_delta1':   pb[1].tolist(),              # bias at offset 1 (nearest)
            'pos_bias_delta32':  pb[32].tolist() if len(pb) > 32 else None,
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


# ─── DWARFBlock (condN) with optional interference pooling ───────────────────

class DSQGBlock(nn.Module):
    """
    condN block: DSQGAttentionN + FFN + optional interference pooling.
    Interference pooling pattern: same as condK+RP (every 3rd layer).
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionN(
            embedding_dim, num_heads, seq_len=seq_len, dropout=dropout)
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
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool   = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)

        x = x + self.ffn(self.norm2(x))
        return x


# ─── condN Transformer ────────────────────────────────────────────────────────

class CondNTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, interference_interval=INTERFERENCE,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DSQGBlock(
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
        # gate_proj bias set to 2.0 in DSQGAttentionN constructor — don't override
        for block in self.blocks:
            nn.init.constant_(block.attn.gate_proj.bias, 2.0)
        # pos_bias already initialized with ALiBi values in DSQGAttentionN
        # (re-init to preserve ALiBi structure — don't overwrite with zeros)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        summaries = [b.attn.attn_summary() for b in self.blocks]
        n = len(summaries)
        pb_mean = sum(s['pos_bias_abs_mean'] for s in summaries) / n
        pb_max  = max(s['pos_bias_abs_max']  for s in summaries)
        # Per-head mean bias across layers (how local vs global each head is)
        head_means = [
            sum(s['pos_bias_mean_per_head'][h] for s in summaries) / n
            for h in range(NUM_HEADS)
        ]
        return {
            'pos_bias_abs_mean':      pb_mean,
            'pos_bias_abs_max':       pb_max,
            'pos_bias_mean_per_head': head_means,
        }


# ─── Data utilities (identical to condK+RP) ──────────────────────────────────

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
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        out1, out2 = model(x1), model(x2)
        diff = (out1 - out2).abs()
    pre  = diff[0, :10].max().item()
    pos  = diff[0,  10].max().item()
    post = diff[0, 11:].max().item()
    print(f'  Pre-10:  {pre:.8f}  (expect 0.0)')
    print(f'  Pos-10:  {pos:.6f}  (expect >0)')
    print(f'  Post-10: {post:.6f}  (expect >0)')
    ok = pre < 1e-6
    print(f'  {"PASS" if ok else "FAIL"} — architecture is {"causal" if ok else "NOT causal"}')
    return ok


# ─── Training loop ────────────────────────────────────────────────────────────

def train(model, train_data, val_data, test_data, tokenizer,
          save_dir='2048_condN_checkpoints', device='cuda'):
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

        ss = model.attn_summary()
        head_means = ss['pos_bias_mean_per_head']
        most_local  = int(max(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        most_global = int(min(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        print(f'  Pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} '
              f'most-local=h{most_local} most-global=h{most_global}')

        print('  ── Generation samples ──')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device)):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')
        sys.stdout.flush()

    # Final test evaluation on best checkpoint
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  condN TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    ss = model.attn_summary()

    print('\n' + '=' * 70)
    print('  condN ABLATION SUMMARY')
    print('=' * 70)
    table = [
        ('Standard transformer 13M (reference)', 64.5),
        ('DWARF condK (D4 linear gather)',        84.7),
        ('DWARF condK+RP (+ interference pool)',  None),  # TBD
        ('DWARF condL+RP (DSQG 24 dyadic)',       None),  # TBD
    ]
    for name, ppl in table:
        ppl_str = f'{ppl:.1f}' if ppl else 'pending'
        print(f'  {name:<52} {ppl_str:>8}')
    print(f'  {"condN (dense-32 + dyadic DSQG)":<52} {test_ppl:>8.1f}')
    print()
    print('  Interpretation:')
    delta_std = test_ppl - 64.5
    delta_k   = test_ppl - 84.7
    print(f'    vs standard transformer: {delta_std:+.1f} PPL')
    print(f'    vs condK (linear):       {delta_k:+.1f} PPL')
    if delta_k < -10:
        print('    → Softmax + dense coverage closes most of the linearity gap')
    elif delta_k < -5:
        print('    → Significant improvement; some gap remains')
    else:
        print('    → Limited improvement at this scale')

    return {
        'test_ppl':              test_ppl,
        'test_loss':             test_loss,
        'best_val_ppl':          best_val_ppl,
        'best_epoch':            best_epoch,
        'total_time_s':          time.time() - t0,
        'condK_baseline_ppl':    84.7,
        'condKRP_baseline_ppl':  None,
        'condLRP_baseline_ppl':  None,
        'delta_vs_condK':        test_ppl - 84.7,
        'n_offsets':             len(_COND_N_OFFSETS),
        'offsets':               _COND_N_OFFSETS,
        'architecture':          'condN: dense-32 + dyadic long-range DSQG + ALiBi bias + interference pooling',
        'attn_summary':          ss,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF condN — Dense-32 Local + Dyadic Long-Range DSQG')
    print('  Softmax over 44 unique offsets with ALiBi per-head bias init')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Offset set: {{0..32}} ∪ {{48,64,...,1536}} = {len(_COND_N_OFFSETS)} offsets')
    print(f'  Coverage improvement: ~35% (dyadic only) → ~55% (dense-32 + dyadic)')
    print(f'  Interference pooling: every {INTERFERENCE}rd layer (condK+RP style)')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits   = load_data(NUM_DOCS)
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '2048_condI_tokenizer.json')
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'Loaded condI BPE tokenizer (shared vocabulary for comparability)')
    else:
        print('Training BPE tokenizer...')
        raw_tok   = train_bpe_tokenizer(splits['train'][:BPE_TRAIN_DOCS])
        raw_tok.save(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  Vocab: {tokenizer.vocab_size()} tokens')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = CondNTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    interference_blocks = [i for i in range(NUM_LAYERS)
                           if i % INTERFERENCE == INTERFERENCE - 1]
    print(f'\ncondN: {n_params:,} parameters')
    print(f'  n_offsets:           {len(_COND_N_OFFSETS)}')
    print(f'  Interference blocks: {interference_blocks}')

    # Sanity: print initial ALiBi bias range
    pb0 = model.blocks[0].attn.pos_bias.detach().cpu()
    print(f'  Initial pos_bias range: [{pb0.min():.3f}, {pb0.max():.3f}]')
    print(f'  (ALiBi: 0.0 at δ=0, negative elsewhere, steeper for local-head)')

    if not causality_check(model, device): return

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir='2048_condN_checkpoints', device=device)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condN_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
