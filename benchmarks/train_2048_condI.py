"""
Wave Field 13M — Condition I: Query-Conditioned Scale Selection (Pre-Gather)

Motivation — why Condition H didn't work:
  H added query conditioning to the *gate* (post-gather):
      gate = sigmoid(W_gate(W_q(x))) ⊙ field[i]
  But field[i] is already the undifferentiated mixture of all scales.
  The Q-gate decides how much of that mixture to pass through,
  but cannot select *which* mixture — that decision already happened.
  Result: H tracked G within ~1 PPL (gate expressivity slightly worse,
  convergence dynamics identical, sparse skips decayed to zero).

  Confirmed: Q-gate applied post-gather is neutral-to-slightly-harmful.
  Root cause (query-less gather) is upstream of the gate.

Condition I — fix the gather directly:
  Instead of fixed per-scale gains (learned scalar per scale per head,
  same for every position), use query-conditioned scale selection:

      scale_logits[b,h,n,j] = scale_gain[j,h] + q_scale_proj(q[b,h,n])_j
      gains[b,h,n,:] = softmax(scale_logits[b,h,n,:], dim=-1)

  Now each position dynamically selects its own scale mixture based on
  its query representation. Position i can prefer local context when
  it needs nearby information and long-range scales when it needs
  distant context — the architecture finally has a mechanism to make
  that distinction at gather-time.

Architecture delta vs. Condition G:
  - CausalWaveletFieldAttention:
      + q_scale_proj: Linear(head_dim, n_scales, bias=False)
        (+11×32 = 352 params per block × 6 blocks = 2,112 params total)
      - scale_gain [n_scales, H]: now serves as learned prior/bias
        (still present, still learned — provides global scale preference;
         q_scale_proj provides the position-specific offset)
      - _causal_multiscale now accepts gains [B, H, N, n_scales] instead
        of computing them internally from the static scale_gain
  - gate_proj: unchanged (standard embedding-space gate, same as G)
  - No sparse skip connections (condH showed these decay to zero)
  - Everything else: identical to G

Why this addresses the bottleneck:
  G's gather: output[i] = Σ_j gains[j,h] * conv_j(field)[i]
    gains are the same for every position — local always wins because
    j0 (RF=4 tokens) is most useful on average and dominates.
  I's gather: output[i] = Σ_j gains[b,h,i,j] * conv_j(field)[i]
    gains are position-specific — position i can emphasize j9 (RF=2048)
    when it genuinely needs long-range context.

Causality verification:
  - gains computed from q[b,h,i,:] = linear(x[i]) — no future tokens
  - conv_j is causal (left-pad only, same as G)
  - No causality concern

Expected behavior:
  If position-specific scale selection is the missing mechanism:
    → gains should show meaningful variance across positions (unlike G
       which converged to nearly uniform across positions)
    → PPL should improve meaningfully below G's 99.4 baseline
    → scale_gain prior may diverge from j0-dominant pattern if specific
       positions systematically prefer longer scales
  If query-less gather was already sufficiently addressed by j0 dominance:
    → gains variance low, PPL ≈ G, architecture hits same ceiling

Comparison table:
  Standard 13M:  PPL  64.5
  A (V4):        PPL  86.8  (G=4096, FFT, floor=0.05)
  B (V4D):       PPL  87.9  (+ dispersion β)
  C (wavelet):   PPL  87.2  (+ causal Morlet)
  D (KdV):       PPL  99.6  (nonlinear — destabilising)
  G (causal DWT):PPL  99.4  (fixed scale gains, j0 dominant)
  H (Q-gate):    PPL  ~101  (post-gather Q conditioning — neutral/worse)
  I (Q-scale):   THIS RUN   (pre-gather Q scale selection)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condI.py \
    2>&1 | tee condI_run.log

Results → benchmarks/2048_condI_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ──────────────────────────────────────────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8          # 8 × 4 accum = 32 effective (matches all prior 13M runs)
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000
BPE_TRAIN_DOCS  = 50_000

# 13M model config (matches Condition G exactly)
EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3          # every 3rd block gets interference pooling

# Causal DWT config (same as G)
N_SCALES        = 11         # dilations: 1, 2, 4, ..., 1024 → RF up to 4096 tokens


# ─── Causal Multi-Scale DWT Attention with Query-Conditioned Scale Selection ──

class CausalWaveletFieldAttentionI(nn.Module):
    """
    Condition I: G = seq_len, causal D4 conv, query-conditioned scale selection.

    Key difference from G:
      G: gains[j, h] — one scalar weight per scale per head, shared across
         all positions. Learned global preference. j0 dominates everywhere.
      I: gains[b, h, n, j] — per-position scale weights, computed from Q.
         Each token decides its own scale mixture at gather-time.
         scale_gain[j, h] still provides a learned global prior.
         q_scale_proj(q[b,h,n]) provides the position-specific offset.

    The wave field deposit and propagation are identical to G:
      field[i] += V[i] * ||K[i]||          (deposit)
      field     = causal_dwt(field, gains)  (propagate + select scales)
      output[i] = field[i]                  (gather — still position-i only)

    The gather is still position-local (not attention-global), but now
    each position selects WHICH scale components to emphasize before
    reading from the field. This is the minimum viable fix for query-less
    gather that doesn't increase asymptotic complexity.
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

        # Standard projections (same as G)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # D4 filter — fixed, precomputed depthwise conv weight (same as G)
        C = num_heads * self.head_dim   # = embedding_dim
        d4 = torch.tensor(self._D4, dtype=torch.float32)
        conv_w = d4.view(1, 1, 4).expand(C, 1, 4).contiguous()
        self.register_buffer('conv_weight', conv_w)  # [C, 1, 4]

        # Learned global scale prior — same as G's scale_gain
        # Serves as a per-head, per-scale bias on top of the Q-offset
        # Init: zeros → softmax gives uniform prior (1/n_scales per scale)
        self.scale_gain = nn.Parameter(
            torch.zeros(n_scales, num_heads)
        )

        # Query → scale logit offset (the key addition vs G)
        # Projects each head's query vector to n_scales logits
        # No bias — scale_gain already provides the bias term
        # Params: head_dim × n_scales × num_heads... wait, one proj per head
        # is expensive. Instead: one shared projection [head_dim → n_scales]
        # applied per head independently. Same as sharing W_q across heads.
        # head_dim × n_scales = 32 × 11 = 352 params per block.
        self.q_scale_proj = nn.Linear(self.head_dim, n_scales, bias=False)
        # Init near zero so model starts at the uniform prior (like G ep1)
        nn.init.normal_(self.q_scale_proj.weight, 0, 0.01)

        # Cross-head field coupling (same as G)
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

    def _causal_multiscale(self, field, gains):
        """
        Causal multi-scale D4 convolution with position-specific scale gains.

        field: [B, H, N, HD]
        gains: [B, H, N, n_scales]  (sum to 1.0 across scale dim for each pos)
        returns: [B, H, N, HD]

        For each scale j:
          - compute causal D4 dilated conv output: [B, C, N]
          - weight by per-position gains for scale j: [B, H, N] → [B, C, N]
          - accumulate
        """
        B, H, N, HD = field.shape
        C = H * HD

        # [B, H, N, HD] → [B, C, N]
        x = field.permute(0, 1, 3, 2).reshape(B, C, N)

        out = torch.zeros_like(x)
        for j in range(self.n_scales):
            d   = 1 << j                # dilation = 2^j
            pad = 3 * d                 # left-pad only → causal
            x_pad = F.pad(x, (pad, 0))
            y = F.conv1d(x_pad, self.conv_weight, dilation=d, groups=C)  # [B, C, N]

            # Scale j gain for each position: [B, H, N]
            # Expand to [B, C, N]: head h → HD channels h*HD:(h+1)*HD
            # gains[:,:,:,j] shape: [B, H, N]
            # → unsqueeze(2) → [B, H, 1, N]
            # → expand → [B, H, HD, N]
            # → contiguous().reshape → [B, C, N]
            g_j = (gains[:, :, :, j]
                   .unsqueeze(2)
                   .expand(B, H, HD, N)
                   .contiguous()
                   .reshape(B, C, N))
            out = out + g_j * y

        # [B, C, N] → [B, H, N, HD]
        return out.reshape(B, H, HD, N).permute(0, 1, 3, 2)

    def forward(self, x):
        B, N, D = x.shape
        H  = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # ── Query-conditioned scale selection ──────────────────────────────
        # q_offset: [B, H, N, n_scales]  (position-specific scale preference)
        q_offset = self.q_scale_proj(q)                 # [B, H, N, n_scales]

        # Add learned global prior: scale_gain [n_scales, H] → [1, H, 1, n_scales]
        prior = self.scale_gain.T.unsqueeze(0).unsqueeze(2)  # [1, H, 1, n_scales]

        # Position-specific scale logits and softmax
        scale_logits = q_offset + prior                 # [B, H, N, n_scales]
        gains = F.softmax(scale_logits, dim=-1)         # [B, H, N, n_scales]

        # ── Wave field deposit (same as G) ─────────────────────────────────
        k_mag = k.norm(dim=-1, keepdim=True)            # [B, H, N, 1]
        field = v * k_mag                               # [B, H, N, HD]

        # ── Causal multi-scale convolution (position-specific gains) ───────
        field = self._causal_multiscale(field, gains)   # [B, H, N, HD]

        # ── Cross-head coupling (same as G) ────────────────────────────────
        coupling = F.softmax(self.field_coupling, dim=-1)
        field    = torch.einsum('ij,bjnd->bind', coupling, field)

        # ── Standard embedding-space gate (same as G, not Q-gate from H) ──
        gate     = torch.sigmoid(self.gate_proj(x))     # [B, N, D]

        # ── Gather: identity at stride=1 ───────────────────────────────────
        gathered = field.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(gathered * gate)
        return self.dropout(out)

    def scale_summary(self):
        """
        Report learned scale priors (scale_gain after softmax), averaged over heads.
        Note: actual per-position gains are dynamic (Q-dependent) and vary.
        This reports the global learned preference, not per-token selection.
        """
        with torch.no_grad():
            prior_gains = F.softmax(self.scale_gain, dim=0)  # [n_scales, H]
        means = prior_gains.mean(dim=1)
        dom   = int(means.argmax().item())
        return {
            'gains_mean_per_scale': means.tolist(),
            'dominant_scale':       dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'note': 'global learned prior; per-position gains are Q-dependent',
        }

    def gains_variance_summary(self, q_sample):
        """
        Optional diagnostic: given a sample Q tensor [B, H, N, HD],
        compute the std of gains across position dim to measure how much
        the model differentiates scale selection by position.
        Returns mean std across batch and heads.
        """
        with torch.no_grad():
            q_offset = self.q_scale_proj(q_sample)
            prior    = self.scale_gain.T.unsqueeze(0).unsqueeze(2)
            gains    = F.softmax(q_offset + prior, dim=-1)  # [B, H, N, n_scales]
            std_pos  = gains.std(dim=2)                      # [B, H, n_scales]
        return float(std_pos.mean().item())


# ─── Transformer blocks (identical to G) ─────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class CausalWaveletBlockI(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = CausalWaveletFieldAttentionI(
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
            xi    = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool  = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalWaveletTransformerI(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, interference_interval, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            CausalWaveletBlockI(
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
        # Override q_scale_proj to near-zero init (set after general init)
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
        return {
            'gains_mean_per_scale': avg_gains,
            'dominant_scale':       dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'note': 'global learned prior (actual gains are position-specific)',
        }


# ─── Data utilities (identical to G) ─────────────────────────────────────────

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
    ds     = load_dataset('openwebtext', split='train', streaming=True)
    texts  = []
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


# ─── Evaluation & generation (identical to G) ────────────────────────────────

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


# ─── Training loop ────────────────────────────────────────────────────────────

def train_condition_i(model, train_data, val_data, test_data, tokenizer,
                      save_dir='2048_condI_checkpoints', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    scaler    = torch.amp.GradScaler('cuda')

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
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item() * GRAD_ACCUM:.4f}')

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
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} '
              f'| {epoch_time:.0f}s')

        # Scale prior summary (global learned preference)
        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print(f'  Scale prior (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}'
                        for j in top3))
        print(f'  (Note: actual gains are position-specific via Q)')

        # Generation samples
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

    print(f'\n  CONDITION I TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    ss = model.scale_summary()
    print(f'  Final dominant prior scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} tokens)')
    print(f'  Full prior gains: '
          + ' '.join(f'j{j}={ss["gains_mean_per_scale"][j]:.3f}'
                     for j in range(N_SCALES)))

    return {
        'test_ppl':     test_ppl,
        'test_loss':    test_loss,
        'best_val_ppl': best_val_ppl,
        'best_epoch':   best_epoch,
        'total_time':   time.time() - t0,
        'scale_summary': ss,
    }


# ─── Causality check ─────────────────────────────────────────────────────────

def causality_check(model, device):
    """
    Verify: changing token at position 5 affects positions 5+ but NOT 0-4.
    Same test as Condition G.
    """
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

    print(f'  Positions 0-4: max |diff| = {pre5_max:.8f} (expect 0.0)')
    print(f'  Position 5:    max |diff| = {pos5_max:.6f} (expect >0)')
    print(f'  Positions 6+:  max |diff| = {post5_max:.6f} (expect >0)')

    if pre5_max < 1e-6:
        print('  PASS — architecture is causal')
        return True
    else:
        print('  FAIL — causality violation detected!')
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  WAVE FIELD 13M — CONDITION I')
    print('  Causal D4 DWT + Query-Conditioned Scale Selection (Pre-Gather)')
    print('  Testing: does per-position scale selection fix query-less gather?')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ──────────────────────────────────────────────────────────────────
    splits = load_data(NUM_DOCS)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '2048_condI_tokenizer.json')

    if os.path.exists(tok_path):
        print(f'\nLoading existing BPE tokenizer from {tok_path}...')
        from tokenizers import Tokenizer
        raw_tok   = Tokenizer.from_file(tok_path)
        tokenizer = BPETokenizerWrapper(raw_tok)
    else:
        print(f'\nTraining BPE tokenizer (vocab={VOCAB_SIZE})...')
        raw_tok   = train_bpe_tokenizer(
            splits['train'][:BPE_TRAIN_DOCS], vocab_size=VOCAB_SIZE)
        raw_tok.save(tok_path)
        print(f'  Saved → {tok_path}')
        tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  Vocab: {tokenizer.vocab_size()} tokens')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    # ── Build model ───────────────────────────────────────────────────────────
    model = CausalWaveletTransformerI(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        n_scales              = N_SCALES,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    q_scale_params = NUM_LAYERS * (EMBEDDING_DIM // NUM_HEADS) * N_SCALES
    print(f'\nCondition I: {n_params:,} params')
    print(f'  (G had 14,115,728; delta = +{n_params - 14_115_728:,} from q_scale_proj)')
    print(f'  Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads '
          f'× {EMBEDDING_DIM}d | FFN {FFN_DIM}')
    print(f'  q_scale_proj: head_dim({EMBEDDING_DIM//NUM_HEADS}) × '
          f'n_scales({N_SCALES}) × {NUM_LAYERS} layers = {q_scale_params:,} params')
    print(f'  Wave field: G=seq_len={MAX_SEQ_LEN}, stride=1')
    print(f'  Gains: position-specific via Q (not fixed per scale/head)')

    print(f'\n  Scale receptive fields:')
    for j in range(N_SCALES):
        d  = 1 << j
        rf = 4 * d
        pct = 100 * rf / MAX_SEQ_LEN
        print(f'    j={j:2d}: dilation={d:4d}, RF={rf:5d} tokens '
              f'({pct:5.1f}% of context)')

    # ── Causality check ───────────────────────────────────────────────────────
    if not causality_check(model, device):
        print('\nAborting — causality check failed.')
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train_condition_i(
        model, train_data, val_data, test_data, tokenizer,
        save_dir='2048_condI_checkpoints',
        device=device,
    )
    results['description']  = 'Causal D4 DWT + query-conditioned scale selection (pre-gather)'
    results['architecture'] = {
        'n_scales':         N_SCALES,
        'filter':           'D4 (fixed)',
        'dilations':        [1 << j for j in range(N_SCALES)],
        'max_rf_tokens':    4 * (1 << (N_SCALES - 1)),
        'G':                MAX_SEQ_LEN,
        'stride':           1,
        'scale_selection':  'query-conditioned (per-position softmax)',
        'q_scale_proj':     f'Linear({EMBEDDING_DIM//NUM_HEADS}, {N_SCALES}, bias=False)',
        'gate':             'standard embedding-space gate (same as G)',
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  CONDITION I RESULTS vs. ABLATION TABLE')
    print('=' * 70)
    print(f'  {"Model":<50} {"PPL":>8}')
    print('  ' + '─' * 60)
    print(f'  {"Standard transformer 13M baseline":<50} {"64.5":>8}')
    print(f'  {"Wave V4 [A] (G=4096, FFT, floor=0.05)":<50} {"86.8":>8}')
    print(f'  {"Wave V4D [B] (+ dispersion β)":<50} {"87.9":>8}')
    print(f'  {"Wave V4 + Morlet [C] (causal wavelet)":<50} {"87.2":>8}')
    print(f'  {"Wave V4 + KdV [D] (nonlinear)":<50} {"99.6":>8}')
    print(f'  {"Condition G (causal D4 DWT, fixed gains)":<50} {"99.4":>8}')
    print(f'  {"Condition H (G + Q-gate post-gather)":<50} {"~101":>8}')
    print(f'  {"Condition I (G + Q-scale pre-gather)":<50} '
          f'{results["test_ppl"]:>8.1f}')

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condI_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
