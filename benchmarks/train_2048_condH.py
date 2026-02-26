"""
Wave Field 13M — Condition H: Causal DWT + Query Gating + Sparse Long-Distance Residual

Hypothesis being tested:
  Condition G proved the architecture is causally sound and can access full-context range,
  but converged to RF=4 token dominance (j0 gain=0.440). Diagnosis: query-less gathering.
  The mechanism deposits V·‖K‖ into a field and reads it back undifferentiated — every
  position gets the same aggregated mixture regardless of what it needs. Without Q·K
  selectivity, local always wins because nearby tokens are almost always relevant, while
  distant tokens are only relevant *sometimes*, and identifying *when* requires query
  conditioning that doesn't exist.

  Condition H adds query conditioning at two levels, motivated directly by the empirical
  findings from G:

  Change 1 — Query-multiplicative gate (Hyena-style):
    G:  gate = sigmoid(gate_proj(x))           [gate from raw input]
    H:  gate = sigmoid(q_gate_proj(q_flat))    [gate from Q projection]
    Why: the gate now operates in query-space — the subspace shaped by W_q to represent
    "what am I looking for at this position?" This gives per-dimension query conditioning
    of the gathered field. Not cross-position selection, but dimension-level: Q_i controls
    which features of the aggregated field pass through to position i.
    Cost: zero extra params (q_gate_proj same size as gate_proj).
    Inductive bias: gate semantics aligned with query content.

  Change 2 — Sparse long-distance residual:
    Motivated by G's j10 uptick: gain at j10 (0.060) > j7 (0.028), j8 (0.024), j9 (0.027).
    j10 at dilation=1024 with seq_len=2048 is geometrically a 2-tap sparse connection
    (positions i and i-1024 only; taps i-2048, i-3072 zero-padded). The model independently
    found this sparse long-distance "copy" more useful than dense mid-range convolution.
    H makes this explicit as a standalone mechanism:
      out += sigmoid(w_d) * field[..., i-d, :]  for d in {512, 1024}
    - Simple 1-tap direct copies (not D4-filtered) — clean "past field memory" semantics
    - sigmoid(w_d) with w_d initialized to -4 → weight ≈ 0.018 at start (near-zero,
      grows toward 1.0 if useful, bounded [0,1])
    - Not part of the softmax-normalized scale budget — additive, independent
    Cost: 2 scalar params per attention block × 6 layers = 12 total params.

  Change 3 (implicit): the 11-scale D4 DWT and all other machinery from G is unchanged.
  Only the gate source and the sparse skip addition are new.

What success looks like:
  - PPL < 99.4 (beat G) — confirms query conditioning helps
  - PPL < 86.8 (beat V4 baseline A) — confirms causal DWT + query gate is competitive
  - PPL → 64.5 (match standard 13M) — confirms the architecture has the right pieces
  Scale gain pattern: if H's j0 gain drops below G's 0.440, the query gate is allowing
  the model to use longer-range context more effectively.

Prior conditions for context:
  A (baseline):  PPL 86.8  (V4, G=4096, FFT, E95=2 tokens)
  B (V4D):       PPL 87.9  (+ dispersion β per head)
  C (wavelet):   PPL 87.2  (+ causal Morlet kernel)
  D (KdV):       PPL 99.6  (failed — nonlinear term destabilizes)
  G (causal DWT): PPL 99.4 (causally sound, but query-less gather → j0 dominates)
  Standard 13M:  PPL 64.5

Run:
  /path/to/.venv/bin/python3 benchmarks/train_2048_condH.py

Results → 2048_condH_results.json
Tokenizer → 2048_condH_tokenizer.json (saved, reused on restart)
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ─────────────────────────────────────────────────────────

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
INTERFERENCE    = 3
N_SCALES        = 11

# Sparse long-distance skip dilations (motivated by G's j10 uptick finding)
SPARSE_DILATIONS = (512, 1024)


# ─── Condition H Attention ────────────────────────────────────────────────────

class CausalWaveletFieldAttentionH(nn.Module):
    """
    Condition H: Causal D4 DWT + query-multiplicative gate + sparse long-distance residual.

    Three changes from G:
      1. gate = sigmoid(q_gate_proj(Q))  [was: sigmoid(gate_proj(x))]
      2. sparse_skip adds field[i-512] and field[i-1024] as a residual after DWT
      3. (implicit) n_scales, D4 filter, field_coupling all unchanged from G

    Causality: preserved.
      - q_gate_proj(Q): Q is derived from x at position i only → causal
      - sparse_skip at d=512, 1024: left-pad → output[i] sees only input[i-d] → causal
      - All D4 dilated convs: left-pad only (same as G) → causal
    """

    _D4 = [0.4829629131445341,  0.8365163037378079,
           0.2241438680420134, -0.1294095225512604]

    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 n_scales=N_SCALES, sparse_dilations=SPARSE_DILATIONS,
                 dropout=0.1):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_heads       = num_heads
        self.head_dim        = embedding_dim // num_heads
        self.seq_len         = seq_len
        self.n_scales        = n_scales
        self.sparse_dilations = sparse_dilations

        # Standard projections
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Change 1: query-multiplicative gate (replaces gate_proj(x))
        # Same size, same init — but fed Q (query-projected space) instead of raw x.
        # Inductive bias: gate is in the space of "what am I looking for?"
        self.q_gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.q_gate_proj.bias, 2.0)   # same init as prior gate_proj

        # D4 conv weight: precomputed to avoid per-step reallocation (see G post-mortem)
        C   = embedding_dim                               # H * HD = embedding_dim
        d4  = torch.tensor(self._D4, dtype=torch.float32)
        conv_w = d4.view(1, 1, 4).expand(C, 1, 4).contiguous()
        self.register_buffer('conv_weight', conv_w)       # [C, 1, 4]

        # Per-scale, per-head softmax-normalised gain (same as G)
        self.scale_gain = nn.Parameter(torch.zeros(n_scales, num_heads))

        # Change 2: sparse long-distance residual
        # sigmoid(w) initialized to sigmoid(-4) ≈ 0.018 → near-zero start
        # bounded [0,1], can grow if useful, architecture-interpretable
        self.sparse_skip_weight = nn.Parameter(
            torch.full((len(sparse_dilations),), -4.0)
        )

        # Cross-head field coupling (unchanged from G)
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

    # ── Causal multi-scale D4 DWT (identical to G) ───────────────────────────

    def _causal_multiscale(self, field):
        """
        11-scale causal D4 dilated convolution.
        field: [B, H, N, HD] → [B, H, N, HD]
        """
        B, H, N, HD = field.shape
        C = H * HD
        x   = field.permute(0, 1, 3, 2).reshape(B, C, N)
        gains = F.softmax(self.scale_gain, dim=0)        # [n_scales, H]
        gains_expanded = gains.unsqueeze(-1).expand(
            self.n_scales, H, HD).reshape(self.n_scales, C)

        out = torch.zeros_like(x)
        for j in range(self.n_scales):
            d     = 1 << j
            pad   = 3 * d
            x_pad = F.pad(x, (pad, 0))
            y     = F.conv1d(x_pad, self.conv_weight, dilation=d, groups=C)
            g     = gains_expanded[j].unsqueeze(0).unsqueeze(-1)
            out   = out + g * y

        return out.reshape(B, H, HD, N).permute(0, 1, 3, 2)

    # ── Sparse long-distance residual (new in H) ─────────────────────────────

    def _sparse_skip(self, field):
        """
        Causal sparse skip connections at specified dilations.
        Adds a weighted copy of field[..., i-d, :] to field[..., i, :].

        Uses simple 1-tap direct copy (not D4-filtered) — clean "field memory"
        semantics: position i receives a bounded fraction of what position i-d
        contained after DWT propagation.

        field: [B, H, N, HD] → residual [B, H, N, HD]

        Causality: F.pad(x, (d, 0))[:, :, :N] shifts the sequence right by d,
        so output[i] = input[i-d] (causal, uses only past positions).
        """
        B, H, N, HD = field.shape
        C = H * HD
        x   = field.permute(0, 1, 3, 2).reshape(B, C, N)   # [B, C, N]
        weights = torch.sigmoid(self.sparse_skip_weight)     # [n_dilations], bounded [0,1]

        out = torch.zeros_like(x)
        for w, d in zip(weights.unbind(), self.sparse_dilations):
            x_shifted = F.pad(x, (d, 0))[:, :, :N]          # causal d-step delay
            out = out + w * x_shifted

        return out.reshape(B, H, HD, N).permute(0, 1, 3, 2)  # [B, H, N, HD]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        B, N, D = x.shape
        H  = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)    # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # Deposit: v weighted by k magnitude (same as G)
        k_mag = k.norm(dim=-1, keepdim=True)
        field = v * k_mag                                 # [B, H, N, HD]

        # === Core field processing ===

        # 1. Causal multi-scale D4 DWT (11 scales, long-range mixing)
        field = self._causal_multiscale(field)

        # 2. Sparse long-distance residual (explicit memory at i-512, i-1024)
        field = field + self._sparse_skip(field)

        # 3. Cross-head coupling (unchanged from G)
        coupling = F.softmax(self.field_coupling, dim=-1)
        field    = torch.einsum('ij,bjnd->bind', coupling, field)

        # === Query-conditioned gather (Change 1) ===

        # Flatten Q to [B, N, D] for gate projection
        q_flat  = q.permute(0, 2, 1, 3).reshape(B, N, D)
        # Gate is now in query-space: which dimensions of the gathered field
        # are relevant to what position i is looking for?
        gate    = torch.sigmoid(self.q_gate_proj(q_flat))  # [B, N, D]

        gathered = field.permute(0, 2, 1, 3).reshape(B, N, D)
        out      = self.out_proj(gathered * gate)
        return self.dropout(out)

    def scale_summary(self):
        with torch.no_grad():
            gains = F.softmax(self.scale_gain, dim=0)     # [n_scales, H]
        skip_w = torch.sigmoid(self.sparse_skip_weight).detach().tolist()
        avg_gains = gains.mean(dim=1).tolist()
        dom = int(max(range(self.n_scales), key=lambda j: avg_gains[j]))
        return {
            'gains_mean_per_scale': avg_gains,
            'dominant_scale':       dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'sparse_skip_weights':  {
                f'd={d}': f'{w:.4f}'
                for d, w in zip(self.sparse_dilations, skip_w)
            },
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


class CondHBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 sparse_dilations, dropout=0.1, use_checkpoint=True,
                 interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = CausalWaveletFieldAttentionH(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, sparse_dilations=sparse_dilations,
            dropout=dropout,
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


class CondHTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, sparse_dilations,
                 interference_interval, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            CondHBlock(
                embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                sparse_dilations, dropout=dropout, use_checkpoint=True,
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
        # Restore q_gate bias override (zeroed by init loop above)
        for block in self.blocks:
            nn.init.constant_(block.attn.q_gate_proj.bias, 2.0)
        # Restore sparse skip weight init (also zeroed)
        for block in self.blocks:
            nn.init.constant_(block.attn.sparse_skip_weight, -4.0)

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
        avg_skip = {}
        for d in SPARSE_DILATIONS:
            key = f'd={d}'
            avg_skip[key] = sum(
                float(s['sparse_skip_weights'][key]) for s in summaries
            ) / n
        return {
            'gains_mean_per_scale':   avg_gains,
            'dominant_scale':         dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
            'sparse_skip_weights_avg': avg_skip,
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
    n    = (len(tokens) // max_seq_len) * max_seq_len
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
                ids     = torch.cat(
                    [ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
        gen = tokenizer.decode(
            ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


# ─── Training ─────────────────────────────────────────────────────────────────

def train_condition_h(model, train_data, val_data, test_data, tokenizer,
                      save_dir='2048_condH_checkpoints', device='cuda'):
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

        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print(f'  Scale gains (top-3): '
              + ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}'
                          for j in top3))
        skip_info = ', '.join(
            f'{k}→{v}' for k, v in ss['sparse_skip_weights_avg'].items())
        print(f'  Sparse skip weights: {skip_info}')

        print('  ── Generation samples (greedy, 150 tokens) ──')
        gens = generate(model, tokenizer, GEN_PROMPTS, device)
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')

    # Final test eval
    model.load_state_dict(torch.load(
        os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  CONDITION H TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')
    ss = model.scale_summary()
    print(f'  Dominant scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} tokens)')
    print(f'  Full gains: '
          + ' '.join(f'j{j}={ss["gains_mean_per_scale"][j]:.3f}'
                     for j in range(N_SCALES)))
    print(f'  Final sparse skip weights: '
          + ', '.join(f'{k}={v}' for k, v in
                      ss["sparse_skip_weights_avg"].items()))

    return {
        'test_ppl':      test_ppl,
        'test_loss':     test_loss,
        'best_val_ppl':  best_val_ppl,
        'best_epoch':    best_epoch,
        'total_time':    time.time() - t0,
        'scale_summary': ss,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  WAVE FIELD 13M — CONDITION H')
    print('  Causal D4 DWT + Query Gate + Sparse Long-Distance Residual')
    print('  Testing hypothesis: query-less gather is the binding constraint')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ─────────────────────────────────────────────────────────────────
    splits = load_data(NUM_DOCS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path   = os.path.join(script_dir, '2048_condH_tokenizer.json')

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
    model = CondHTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        n_scales              = N_SCALES,
        sparse_dilations      = SPARSE_DILATIONS,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    print(f'\nCondition H: {n_params:,} params (G was 14,115,728)')
    print(f'  Changes from G:')
    print(f'    gate source: x → Q (query-multiplicative gating)')
    print(f'    sparse skip: +{2*len(SPARSE_DILATIONS)} params '
          f'(dilations {SPARSE_DILATIONS}, sigmoid init=-4 → {torch.sigmoid(torch.tensor(-4.0)):.3f})')

    # Verify init
    sample_block = model.blocks[0].attn
    init_gate_val = torch.sigmoid(sample_block.sparse_skip_weight[0]).item()
    print(f'    sparse_skip_weight[0] init: sigmoid(-4.0) = {init_gate_val:.4f}')

    # ── Causality check ───────────────────────────────────────────────────────
    print('\nRunning causality check...')
    model.eval()
    x1 = torch.randint(0, tokenizer.vocab_size(), (1, 16)).to(device)
    x2 = x1.clone()
    x2[0, 5] = (x2[0, 5] + 1) % tokenizer.vocab_size()
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)
    diff = (o1 - o2).abs()
    pre_change  = diff[0, :5].max().item()
    at_change   = diff[0, 5].max().item()
    post_change = diff[0, 6:].max().item()
    print(f'  Positions 0-4: max |diff| = {pre_change:.8f}  (expect 0.0)')
    print(f'  Position 5:    max |diff| = {at_change:.6f}  (expect >0)')
    print(f'  Positions 6+:  max |diff| = {post_change:.6f}  (expect >0)')
    if pre_change > 1e-6:
        print('  *** CAUSALITY VIOLATION DETECTED — ABORTING ***')
        sys.exit(1)
    else:
        print('  PASS — architecture is causal')
    model.train()

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train_condition_h(
        model, train_data, val_data, test_data, tokenizer,
        save_dir='2048_condH_checkpoints', device=device,
    )
    results['description']  = 'Causal D4 DWT + query-multiplicative gate + sparse skip'
    results['hypothesis']   = 'query-less gather is the binding constraint in G'
    results['changes_from_g'] = {
        'gate':        'q_gate_proj(Q) instead of gate_proj(x)',
        'sparse_skip': f'dilations {SPARSE_DILATIONS}, sigmoid weight init -4.0',
        'dtw':         'unchanged from G (11 scales, D4, softmax gains)',
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  CONDITION H RESULTS vs. ABLATION TABLE')
    print('=' * 70)
    print(f'  {"Model":<48} {"PPL":>6}')
    print('  ' + '─' * 56)
    print(f'  {"Standard transformer 13M baseline":<48} {"64.5":>6}')
    print(f'  {"Wave V4 [A]":<48} {"86.8":>6}')
    print(f'  {"Wave V4D [B] (+ dispersion β)":<48} {"87.9":>6}')
    print(f'  {"Wave V4 + Morlet [C]":<48} {"87.2":>6}')
    print(f'  {"Wave V4 + KdV [D]":<48} {"99.6":>6}')
    print(f'  {"Condition G (causal DWT)":<48} {"99.4":>6}')
    print(f'  {"Condition H (+ Q gate + sparse skip)":<48} '
          f'{results["test_ppl"]:>6.1f}')

    results_path = os.path.join(script_dir, '2048_condH_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
