"""
Wave Field 13M — Condition G: Causal Multi-Scale DWT Field @ 2048 tokens

Motivation (from Conditions E/F post-mortem):
  - FFT convolution is circular: with G=seq_len, last token wraps distance-1
    adjacent to first token, creating a global information highway. Loss
    collapses to ~0.92 within the first epoch — same class of bug as the
    Haar DWT C_gaussian incident (PPL 7.3 at ep2).
  - Root cause: causality was *enforced* via masking/padding (secondary
    constraint), not built into the mechanism itself.

Condition G fixes this fundamentally:
  Replace the FFT circular convolution with **causal multi-scale temporal
  convolution** using Daubechies D4 filter coefficients. Causality is
  structural — the filter has no future-facing component by design.

Architecture delta vs. V4 baseline:
  - G = seq_len = 2048 (stride = 1, no scatter/gather interpolation at all)
  - Wave propagation: FFT circular conv → J=11 causal dilated D4 convs
    Dilations: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    Effective RFs: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 tokens
    → Scale 10 covers >seq_len; architecture can learn full-context dependencies
  - Per-scale, per-head gain (48 params vs. 3×8=24 wave params in V4)
    These replace α (damping), ω (frequency), φ (phase) — more interpretable

Causality proof:
  For scale j, dilation = 2^j, kernel_size = 4 (D4):
    left_pad = (kernel_size - 1) * dilation = 3 * 2^j
    output[i] = sum_{l=0}^{3} d4[l] * input[i - l * 2^j]
    All indices ≤ i → strictly causal.
  No masking required. Works identically in training and inference.

Why D4 and not Haar?
  - Haar (D2): single difference/sum — coarse, no overlap between scales
  - D4: 4-tap, orthogonal, smooth, 2 vanishing moments — better frequency
    selectivity per scale without the acausal Haar DWT butterfly

Expected behavior:
  - No causality violation (loss should NOT collapse in epoch 1)
  - Multi-scale coverage: model can learn to use scales from 4→4096 tokens
  - PPL target: approach or beat 64.5 (standard 13M baseline) if architecture
    can exploit long-range; confirm architecture has a viable path forward

Comparison table:
  A (baseline):  PPL 86.8  (V4, G=4096, FFT, E95=2 tokens)
  B (V4D):       PPL 87.9  (+ dispersion β per head)
  C (wavelet):   PPL 87.2  (+ causal Morlet kernel)
  D (KdV):       PPL 99.6  (failed — nonlinear term destabilizes)
  E (G=2048):    INVALID    (circular FFT + G=seq_len → causality leak)
  F (G=2048+α):  INVALID    (same leak, different init)
  G (causal DWT): THIS RUN
  Standard 13M:  PPL 64.5

Run:
  /path/to/.venv/bin/python3 train_2048_condG.py

Results → 2048_condG_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ─────────────────────────────────────────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8          # 8 × 4 accum = 32 effective (matches all prior 13M runs)
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000
BPE_TRAIN_DOCS  = 50_000

# 13M model config (matches Condition A baseline exactly)
EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3          # every 3rd block gets interference pooling

# Causal DWT config
N_SCALES        = 11         # dilations: 1, 2, 4, ..., 1024 → RF up to 4096 tokens


# ─── Causal Multi-Scale DWT Attention (Condition G) ──────────────────────────

class CausalWaveletFieldAttention(nn.Module):
    """
    Condition G attention: G = seq_len, causal dilated D4 conv instead of FFT.

    The "wave field" is still conceptually present (scatter v*k into field at
    stride-1 positions, propagate, gather) but with G=seq_len and stride=1,
    scatter and gather are identity operations — no interpolation overhead.

    The wave propagation kernel is replaced by a sum of causal dilated 1D
    convolutions at exponentially spaced scales, initialized from D4 filter
    coefficients. Per-scale, per-head gain parameters (softmax-normalized)
    replace the damping/frequency/phase parameters of V4.
    """

    # Daubechies D4 analysis filter — 4 taps, orthogonal, 2 vanishing moments
    # Source: standard wavelet tables (Mallat 1989)
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

        # Standard projections (same as V4)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # D4 filter — fixed (not learned); precomputed as depthwise conv weight
        # Precomputed in __init__ to avoid repeated .expand().contiguous() in
        # the forward loop (132 allocs/step with 11 scales × 6 layers × 2 checkpoint).
        C = num_heads * (embedding_dim // num_heads)             # = embedding_dim
        d4 = torch.tensor(self._D4, dtype=torch.float32)
        # Shape [C, 1, 4] — groups=C depthwise conv (same filter for all channels)
        conv_w = d4.view(1, 1, 4).expand(C, 1, 4).contiguous()
        self.register_buffer('conv_weight', conv_w)              # [C, 1, 4]

        # Per-scale, per-head gain — replaces α/ω/φ
        # Initialized to uniform (1/n_scales); softmax-normalized in forward
        # so gains sum to 1.0 per head across scales
        self.scale_gain = nn.Parameter(
            torch.zeros(n_scales, num_heads)    # softmax(0) = uniform = 1/n_scales
        )

        # Cross-head field coupling (same as V4)
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

    def _causal_multiscale(self, field):
        """
        Causal multi-scale D4 convolution.

        field: [B, H, N, HD]
        returns: [B, H, N, HD]

        For each scale j (dilation = 2^j):
          - left-pad = 3 * 2^j  (so output[i] sees only inputs[0..i])
          - apply D4 filter as depthwise conv along the N dimension
          - weight output by per-scale, per-head gain
        All operations are causal by construction.
        """
        B, H, N, HD = field.shape
        C = H * HD                              # total channels for grouped conv

        # [B, H, N, HD] → [B, H*HD, N]
        x = field.permute(0, 1, 3, 2).reshape(B, C, N)

        # Per-scale gains: softmax over scales → sum to 1.0 per head
        gains = F.softmax(self.scale_gain, dim=0)   # [n_scales, H]

        # Expand gain to [n_scales, H*HD]: each head_dim element in a head
        # shares the same scale gain
        gains_expanded = gains.unsqueeze(-1).expand(
            self.n_scales, H, HD
        ).reshape(self.n_scales, C)                  # [n_scales, H*HD]

        out = torch.zeros_like(x)
        for j in range(self.n_scales):
            d     = 1 << j                          # dilation = 2^j
            pad   = 3 * d                           # left-pad only → causal
            x_pad = F.pad(x, (pad, 0))              # [B, C, N + pad]
            y     = F.conv1d(x_pad, self.conv_weight,
                             dilation=d, groups=C)  # [B, C, N]

            # Scale's gain: [C] → [1, C, 1] for broadcast
            g = gains_expanded[j].unsqueeze(0).unsqueeze(-1)   # [1, C, 1]
            out = out + g * y

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

        # Deposit: v weighted by k magnitude (same as V4)
        k_mag  = k.norm(dim=-1, keepdim=True)           # [B, H, N, 1]
        field  = v * k_mag                              # [B, H, N, HD]

        # With G=seq_len and stride=1, scatter is an identity — field IS the
        # token sequence weighted by k_mag.  No scatter/gather interpolation.

        # Causal multi-scale convolution (replaces FFT wave propagation)
        field = self._causal_multiscale(field)           # [B, H, N, HD]

        # Cross-head coupling (same as V4)
        coupling = F.softmax(self.field_coupling, dim=-1)  # [H, H]
        field    = torch.einsum('ij,bjnd->bind', coupling, field)

        # Content-dependent gate (same as V4)
        gate    = torch.sigmoid(self.gate_proj(x))       # [B, N, D]

        # Gather: identity (stride=1, G=N) — just reshape back
        gathered = field.permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.out_proj(gathered * gate)
        return self.dropout(out)

    def scale_summary(self):
        """Report per-scale, per-head gains after softmax normalisation."""
        with torch.no_grad():
            gains = F.softmax(self.scale_gain, dim=0)  # [n_scales, H]
        return {
            'gains_mean_per_scale': gains.mean(dim=1).tolist(),  # [n_scales]
            'dominant_scale':       gains.mean(dim=1).argmax().item(),
            'dominant_scale_rf_tokens': 4 * (1 << int(gains.mean(dim=1).argmax().item())),
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


class CausalWaveletBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = CausalWaveletFieldAttention(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, dropout=dropout,
        )
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        # Causal interference pooling (same as longrange ablation — cumulative mean)
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
            pool  = xi.cumsum(dim=1) / counts              # causal mean
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalWaveletTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, n_scales, interference_interval, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            CausalWaveletBlock(
                embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                dropout=dropout, use_checkpoint=True,
                interference=(i % interference_interval == interference_interval - 1),
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight      # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

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
        # Average gains per scale across all blocks
        avg_gains = [
            sum(s['gains_mean_per_scale'][j] for s in summaries) / n
            for j in range(N_SCALES)
        ]
        dom = int(max(range(N_SCALES), key=lambda j: avg_gains[j]))
        return {
            'gains_mean_per_scale': avg_gains,
            'dominant_scale':       dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
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
        tokens.append(3)    # </s>
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


# ─── Training loop ────────────────────────────────────────────────────────────

def train_condition_g(model, train_data, val_data, test_data, tokenizer,
                      save_dir='2048_condG_checkpoints', device='cuda'):
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

        # Scale gain summary
        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print(f'  Scale gains (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}'
                        for j in top3))

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
    test_acc  = 0.0  # omitted for brevity

    print(f'\n  CONDITION G TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    ss = model.scale_summary()
    print(f'  Final dominant scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} tokens)')
    print(f'  Full gains: '
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  WAVE FIELD 13M — CONDITION G: CAUSAL MULTI-SCALE DWT FIELD')
    print(f'  D4 filter | {N_SCALES} scales | dilations 1..{1<<(N_SCALES-1)} '
          f'| RF up to {4*(1<<(N_SCALES-1))} tokens')
    print('  Causality: structural (left-pad only, no circular wrap)')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ─────────────────────────────────────────────────────────────────
    splits = load_data(NUM_DOCS)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Save tokenizer alongside checkpoints so model can be reloaded later.
    # Without this, the checkpoint is useless after the process exits.
    tok_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '2048_condG_tokenizer.json')

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
    model = CausalWaveletTransformer(
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
    print(f'\nCondition G: {n_params:,} params')
    print(f'  Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads '
          f'× {EMBEDDING_DIM}d | FFN {FFN_DIM}')
    print(f'  Wave field: G=seq_len={MAX_SEQ_LEN}, stride=1 (identity scatter/gather)')
    print(f'  Convolution: {N_SCALES} causal D4 dilated convs, '
          f'dilation ∈ {{1..{1<<(N_SCALES-1)}}}')

    # Sanity check: first scale RF diagnostics
    print(f'\n  Scale receptive fields:')
    for j in range(N_SCALES):
        d  = 1 << j
        rf = 4 * d
        pct = 100 * rf / MAX_SEQ_LEN
        print(f'    j={j:2d}: dilation={d:4d}, RF={rf:5d} tokens '
              f'({pct:5.1f}% of context)')

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train_condition_g(
        model, train_data, val_data, test_data, tokenizer,
        save_dir='2048_condG_checkpoints',
        device=device,
    )
    results['description']  = 'Causal multi-scale D4 DWT field, G=seq_len'
    results['architecture'] = {
        'n_scales':   N_SCALES,
        'filter':     'D4 (fixed)',
        'dilations':  [1 << j for j in range(N_SCALES)],
        'max_rf_tokens': 4 * (1 << (N_SCALES - 1)),
        'G':          MAX_SEQ_LEN,
        'stride':     1,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  CONDITION G RESULTS vs. ABLATION TABLE')
    print('=' * 70)
    print(f'  {"Model":<46} {"PPL":>8}')
    print('  ' + '─' * 56)
    print(f'  {"Standard transformer 13M baseline":<46} {"64.5":>8}')
    print(f'  {"Wave V4 [A] (G=4096, FFT, floor=0.05)":<46} {"86.8":>8}')
    print(f'  {"Wave V4D [B] (+ dispersion β)":<46} {"87.9":>8}')
    print(f'  {"Wave V4 + Morlet [C] (causal wavelet)":<46} {"87.2":>8}')
    print(f'  {"Wave V4 + KdV [D] (nonlinear)":<46} {"99.6":>8}')
    print(f'  {"Condition G (causal D4 DWT, G=seq_len)":<46} '
          f'{results["test_ppl"]:>8.1f}')

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condG_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
