"""
Wave Field 13M — Condition J: Outer-Product Wave Field (OPWF)

Root cause recap:
  condG/H/I all suffer from the same fundamental limitation: the field
  stores V·||K|| per position. This loses the *direction* of K, so the
  gather `output[i] = field[i]` cannot discriminate by content —
  every position receives the same proximity-weighted aggregate.

  condI (Q-scale pre-gather) adds temporal routing — each position can
  select WHICH scale window to prioritize — but still cannot distinguish
  WHICH past tokens are relevant within that window.

  The missing piece: Q·K inner product, which is what enables content-
  based selective retrieval in standard attention.

condJ — efficient OPWF (Outer-Product Wave Field):
  Instead of depositing a scalar-weighted value vector:
      field[i] += V_i * ||K_i||         (condG/H/I)

  The OPWF gather would be:
      output[i] = Q_i @ Σ_t w(i-t) K_t⊗V_t
                = Σ_t w(i-t) (Q_i · K_t) V_t

  Key insight: this NEVER requires forming K⊗V explicitly.
  Expanding w(i-t) into D4 filter taps (h0..h3) at dilations 2^j:

      output[i] = Σ_j gain_j * Σ_τ h_τ * (Q_i · K_{i - 2^j*τ}) * V_{i - 2^j*τ}

  This is 4 taps × 11 scales = 44 scalar inner products per position,
  each producing a weighted V contribution. No K⊗V tensor needed.
  Complexity: O(N × HD × 44) vs O(N × HD² × 44) for explicit outer product.
  Memory: O(N × HD) instead of O(N × HD²).  Speed: ~32× faster.

Mathematical verification (Rust, opwf_analysis.rs):
  Test 1 — Equivalence: Q @ (K⊗V) = (Q·K) * V. Max error 3.55e-14. ✓
  Test 2 — Field rank: accumulated field is near-full-rank (top-4 = 25.3%).
  Test 3 — m/r: m=8,r=4 vs m=4,r=8 output error essentially equal.
  Conclusion: explicit outer product is equivalent to efficient formulation;
              use efficient formulation for 32× speed/memory improvement.

Architecture delta vs. condI:
  - No explicit field tensor (avoided via factored gather)
  - Gather: Σ_{j,τ} gain_j * h_τ * (Q · K_{shifted}) * V_{shifted}
  - Identity bypass: gain_bypass * (Q·K) * V  (48 learned scalars, init ~0)
  - Cross-head coupling: REMOVED
  - Q-scale gains: RETAINED from condI (per-position scale selection)
  - Extra parameters vs condI: +48 (identity bypass)
  - Same O(N × HD) memory as condI

Stability hedge:
  - If ep1 loss > 8.0 at step 200, may need softplus maps + Z normalization.

Comparison table:
  Standard 13M:   PPL  64.5
  A (V4):         PPL  86.8  (G=4096, FFT, floor=0.05)
  B (V4D):        PPL  87.9  (+ dispersion β)
  C (wavelet):    PPL  87.2  (+ causal Morlet)
  D (KdV):        PPL  99.6  (nonlinear — unstable)
  G (causal DWT): PPL  99.4  (fixed scale gains, j0 dominant)
  H (Q-gate):     PPL ~101   (post-gather Q gate — null result)
  I (Q-scale):    PPL  93.3  (pre-gather Q scale selection)
  J (OPWF):       THIS RUN   (outer-product field, Q·K retrieval)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condJ.py \
    2>&1 | tee condJ_run.log

Results → benchmarks/2048_condJ_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ──────────────────────────────────────────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8          # 8 × 4 accum = 32 effective; reduce to 4 if OOM
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000
BPE_TRAIN_DOCS  = 50_000

# 13M model config (matches all prior 13M conditions exactly)
EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3          # every 3rd block gets interference pooling

# Causal DWT config (same as G/H/I)
N_SCALES        = 11         # dilations: 1, 2, 4, ..., 1024 → RF up to 4096 tokens


# ─── Outer-Product Wave Field Attention (Efficient Formulation) ───────────────

class CausalOPWFAttention(nn.Module):
    """
    Efficient OPWF: avoids forming K⊗V explicitly.

    Mathematical equivalence:
      output_i = Q_i @ Σ_{j,τ} gain_{i,j} * h_τ * K_{i-2^j*τ} ⊗ V_{i-2^j*τ}
               = Σ_{j,τ} gain_{i,j} * h_τ * (Q_i · K_{i-2^j*τ}) * V_{i-2^j*τ}

    This is 4 D4 taps × 11 scales = 44 causal inner products per position,
    each producing a scalar (Q·K) weighting of the corresponding V.

    Complexity: O(N × HD × 44) — same as condI, 32× cheaper than explicit OPWF.
    Memory:     O(N × HD)       — no HD×HD tensor needed.

    Additional identity bypass: gain_bypass * (Q_i · K_i) * V_i  (local, no shift).
    Q-scale gains from condI retained: per-position scale selection.
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

        # Learned global scale prior + Q-conditioned offset (from condI)
        self.scale_gain   = nn.Parameter(torch.zeros(n_scales, num_heads))
        self.q_scale_proj = nn.Linear(self.head_dim, n_scales, bias=False)
        nn.init.normal_(self.q_scale_proj.weight, 0, 0.01)

        # Identity bypass: output += softplus(param) * (Q·K) * V  per head
        self.identity_bypass = nn.Parameter(torch.full((num_heads,), -4.0))

        self.dropout = nn.Dropout(dropout)

        # Precompute D4 filter as a tensor (for device placement)
        self.register_buffer('d4', torch.tensor(self._D4, dtype=torch.float32))

    def forward(self, x):
        B, N, D = x.shape
        H  = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # ── Q-conditioned scale gains (from condI) ─────────────────────────
        q_offset = self.q_scale_proj(q)                         # [B, H, N, S]
        prior    = self.scale_gain.T.unsqueeze(0).unsqueeze(2)  # [1, H, 1, S]
        gains    = F.softmax(q_offset + prior, dim=-1)          # [B, H, N, S]

        # ── Efficient OPWF gather ──────────────────────────────────────────
        # output[i] = Σ_{j,τ} gain_j * h_τ * (Q_i · K_{i-offset}) * V_{i-offset}
        # No K⊗V tensor needed; just shifted inner products.
        out = torch.zeros(B, H, N, HD, device=x.device, dtype=q.dtype)

        for j in range(self.n_scales):
            d    = 1 << j        # dilation
            g_j  = gains[:, :, :, j].unsqueeze(-1)  # [B, H, N, 1]
            for tau, h_coef in enumerate(self._D4):
                offset = d * tau
                if offset == 0:
                    k_s, v_s = k, v
                elif offset >= N:
                    continue       # beyond sequence, contribution is zero
                else:
                    # Causal shift: k_s[b,h,i,:] = k[b,h,i-offset,:]
                    pad   = k.new_zeros(B, H, offset, HD)
                    k_s   = torch.cat([pad, k[:, :, :N - offset, :]], dim=2)
                    v_s   = torch.cat([pad, v[:, :, :N - offset, :]], dim=2)
                # Scalar inner product: [B, H, N]
                qk = (q * k_s).sum(-1, keepdim=True)     # [B, H, N, 1]
                out = out + (g_j * h_coef * qk) * v_s

        # ── Identity bypass: gain_bypass * (Q·K) * V at same position ─────
        bypass = F.softplus(self.identity_bypass).view(1, H, 1, 1)  # [1,H,1,1]
        local_qk = (q * k).sum(-1, keepdim=True)                     # [B,H,N,1]
        out = out + bypass * local_qk * v

        # ── Gate and output projection ─────────────────────────────────────
        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def scale_summary(self):
        with torch.no_grad():
            prior_gains = F.softmax(self.scale_gain, dim=0)
        means = prior_gains.mean(dim=1)
        dom   = int(means.argmax().item())
        bypass_vals = F.softplus(self.identity_bypass).tolist()
        return {
            'gains_mean_per_scale':     means.tolist(),
            'dominant_scale':           dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'identity_bypass_mean':     sum(bypass_vals) / len(bypass_vals),
            'note': 'global learned prior; per-position gains are Q-dependent',
        }


# ─── Transformer blocks ───────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class OPWFBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = CausalOPWFAttention(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, dropout=dropout,
        )
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)
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
            xi     = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool   = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class OPWFTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, interference_interval, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            OPWFBlock(
                embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                dropout=dropout, use_checkpoint=True,
                interference=(i % interference_interval == interference_interval - 1),
            )
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.q_scale_proj.weight, 0, 0.01)

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
        avg_gains = [
            sum(s['gains_mean_per_scale'][j] for s in summaries) / n
            for j in range(N_SCALES)
        ]
        dom = int(max(range(N_SCALES), key=lambda j: avg_gains[j]))
        bypass_mean = sum(s['identity_bypass_mean'] for s in summaries) / n
        return {
            'gains_mean_per_scale':     avg_gains,
            'dominant_scale':           dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'identity_bypass_mean':     bypass_mean,
            'note': 'global learned prior (actual gains are position-specific)',
        }


# ─── Data utilities (identical to prior conditions) ───────────────────────────

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import (Tokenizer, models, trainers,
                            pre_tokenizers, decoders)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>'],
    )
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
    print(f'  {len(texts):,} docs | train {int(len(texts)*0.95):,} '
          f'| val 2,500 | test 2,500')
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
        loss = F.cross_entropy(
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
    seq_len = 32
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
        x2 = x1.clone()
        x2[0, 5] = (x2[0, 5] + 1) % VOCAB_SIZE
        out1 = model(x1)
        out2 = model(x2)
        diff = (out1 - out2).abs()
    pre5_max  = diff[0, :5].max().item()
    pos5_max  = diff[0,  5].max().item()
    post5_max = diff[0, 6:].max().item()
    print(f'  Positions 0-4: max |diff| = {pre5_max:.8f}  (expect 0.0)')
    print(f'  Position 5:    max |diff| = {pos5_max:.6f}  (expect >0)')
    print(f'  Positions 6+:  max |diff| = {post5_max:.6f}  (expect >0)')
    if pre5_max < 1e-6:
        print('  PASS — architecture is causal')
        return True
    else:
        print('  FAIL — causality violation detected!')
        return False


# ─── Training loop ────────────────────────────────────────────────────────────

def train_condition_j(model, train_data, val_data, test_data, tokenizer,
                      save_dir='2048_condJ_checkpoints', device='cuda'):
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

    best_val_loss = float('inf')
    best_val_ppl  = float('inf')
    best_epoch    = 0
    t0            = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0
        optimizer.zero_grad()

        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device)
                y = batch[:,  1:].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    ) / GRAD_ACCUM
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                loss_val = loss.item() * GRAD_ACCUM
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss_val:.4f}')
                # Stability check: ep1 loss should be decreasing toward ~5.5 by step 200
                if epoch == 1 and step == 200 and loss_val > 8.5:
                    print(f'\n  WARNING: ep1 step-200 loss {loss_val:.4f} > 8.5')
                    print('  OPWF may be unstable. Consider condJ-v2 with softplus maps.')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        epoch_time = time.time() - t0

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl  = val_ppl
            best_epoch    = epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {epoch_time:.0f}s')

        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print('  Scale prior (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}' for j in top3))
        print(f'  Identity bypass mean: {ss["identity_bypass_mean"]:.4f}')
        print('  (Note: actual gains are position-specific via Q)')

        print('  ── Generation samples (greedy, 150 tokens) ──')
        gens = generate(model, tokenizer, GEN_PROMPTS, device)
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')

    # Final test evaluation
    model.load_state_dict(torch.load(
        os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  CONDITION J TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')
    ss = model.scale_summary()
    print(f'  Final dominant prior scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} tokens)')
    print(f'  Full prior gains: '
          + ' '.join(f'j{j}={ss["gains_mean_per_scale"][j]:.3f}'
                     for j in range(N_SCALES)))
    print(f'  Identity bypass mean: {ss["identity_bypass_mean"]:.4f}')

    print('\n' + '=' * 70)
    print('  CONDITION J RESULTS vs. ABLATION TABLE')
    print('=' * 70)
    rows = [
        ('Standard transformer 13M baseline',        64.5),
        ('Wave V4 [A] (G=4096, FFT, floor=0.05)',    86.8),
        ('Wave V4D [B] (+ dispersion β)',             87.9),
        ('Wave V4 + Morlet [C] (causal wavelet)',     87.2),
        ('Wave V4 + KdV [D] (nonlinear)',             99.6),
        ('Condition G (causal D4 DWT, fixed gains)',  99.4),
        ('Condition H (G + Q-gate post-gather)',     100.0),
        ('Condition I (G + Q-scale pre-gather)',      93.3),
    ]
    print(f'  {"Model":<50} {"PPL":>8}')
    print('  ' + '─' * 60)
    for name, ppl in rows:
        print(f'  {name:<50} {ppl:>8.1f}')
    print(f'  {"Condition J (OPWF: K⊗V field + Q gather)":<50} {test_ppl:>8.1f}')

    return {
        'test_ppl':           test_ppl,
        'test_loss':          test_loss,
        'best_val_ppl':       best_val_ppl,
        'best_epoch':         best_epoch,
        'total_time':         time.time() - t0,
        'scale_summary':      ss,
        'description':        'Outer-product wave field (K⊗V), Q @ F gather, '
                              'causal D4 DWT + Q-scale gains (condI)',
        'architecture': {
            'n_scales':         N_SCALES,
            'filter':           'D4 (fixed)',
            'field_shape':      'none (efficient gather; no K⊗V tensor)',
            'deposit':          'K ⊗ V  (einsum bhni,bhnj->bhnij)',
            'gather':           'Q @ F  (einsum bhni,bhnij->bhnj)',
            'identity_bypass':  'F_prop += softplus(param) * F_0 per head',
            'q_scale_gains':    'per-position softmax (from condI)',
            'cross_head_coupling': 'removed (minimal delta from condI)',
            'extra_params_vs_condI': 48,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  WAVE FIELD 13M — CONDITION J')
    print('  Outer-Product Wave Field (OPWF): K⊗V deposit, Q @ F gather')
    print('  Testing: does content-based Q·K retrieval close the PPL gap?')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  GPU: {gpu_name}  ({total_gb:.1f} GB)')
        print(f'  Note: efficient OPWF — no large field tensor. Memory ~same as condI.')

    # ── Data ──────────────────────────────────────────────────────────────────
    splits = load_data(NUM_DOCS)

    # ── Tokenizer (reuse condI tokenizer — same dataset, comparable results) ─
    tok_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '2048_condI_tokenizer.json')

    if os.path.exists(tok_path):
        print(f'\nLoading condI BPE tokenizer from {tok_path}...')
        from tokenizers import Tokenizer
        raw_tok   = Tokenizer.from_file(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
        print('  (reusing condI tokenizer for direct comparability)')
    else:
        print(f'\nTraining BPE tokenizer (vocab={VOCAB_SIZE})...')
        raw_tok   = train_bpe_tokenizer(
            splits['train'][:BPE_TRAIN_DOCS], vocab_size=VOCAB_SIZE)
        raw_tok.save(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  Vocab: {tokenizer.vocab_size()} tokens')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    # ── Build model ───────────────────────────────────────────────────────────
    model = OPWFTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        n_scales              = N_SCALES,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params     = model.param_count()
    opwf_bypass  = NUM_LAYERS * NUM_HEADS   # identity_bypass params
    hd = EMBEDDING_DIM // NUM_HEADS
    print(f'\nCondition J: {n_params:,} params')
    print(f'  (condI had ~14,117,840; delta = +{n_params - 14_117_840:+,})')
    print(f'  Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads × {EMBEDDING_DIM}d')
    print(f'  Method: efficient OPWF — no explicit K⊗V tensor')
    print(f'  Gather: Σ_j_τ gain_j * h_τ * (Q·K_shifted) * V_shifted')
    print(f'  Taps:   {N_SCALES} scales × 4 D4 = {N_SCALES * 4} inner products/pos')
    print(f'  Memory: O(N × HD) — same as condI, no HD×HD field')
    print(f'  Identity bypass: {opwf_bypass} params (softplus per head per layer)')
    print(f'  Q-scale: per-position scale selection retained from condI')

    # ── Causality check ───────────────────────────────────────────────────────
    if not causality_check(model, device):
        print('\nAborting — causality check failed.')
        return

    # ── Memory sanity check ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f'\n  Free VRAM before training: {free_gb:.1f} GB')
        hd = EMBEDDING_DIM // NUM_HEADS
        field_gb = BATCH_SIZE * NUM_HEADS * MAX_SEQ_LEN * hd * hd * 2 / 1e9
        print(f'  Estimated field tensor (bf16): {field_gb:.2f} GB per layer forward')
        if free_gb < 10.0:
            print(f'  WARNING: Only {free_gb:.1f} GB free. Consider reducing BATCH_SIZE to 4.')

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train_condition_j(
        model, train_data, val_data, test_data, tokenizer,
        save_dir='2048_condJ_checkpoints',
        device=device,
    )

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condJ_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
