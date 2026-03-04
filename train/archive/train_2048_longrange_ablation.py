"""
Wave Field 13M — Long-Range Alpha Ablation @ 2048 tokens

Based on freq_decomp_analysis findings (2026-02-24):
  - Current V4 architecture has 0.05 damping floor → max E95 ≈ 30 field positions ≈ 15 tokens
  - Only 0.8% of the 2048-token context window is visible to any head
  - At G=4096 with seq_len=2048: stride=2.0005 (not integer) → 36% scatter/gather RMS loss
  - Heads 5-7 have 91% spectral overlap — redundant

Two new conditions vs Condition A baseline (PPL 86.8):

  Condition E — G=seq_len fix only
    field_size = 2048 (stride = 1.0 exactly, eliminates scatter/gather loss)
    damping    = linspace(-2.0, 0.0, 8)   ← same as V4
    floor      = 0.05                      ← same as V4
    Expected: isolates the geometric fix; still local (~30 positions, ~30 tokens)

  Condition F — G=seq_len + long-range α init
    field_size = 2048 (stride = 1.0 exactly)
    damping    = linspace(-8.0, -1.0, 8)  ← much more negative → smaller α
    floor      = 0.001                     ← lowered from 0.05
    α range:   [0.001, 0.269]             ← E95 range: [5.6, ~1122] positions
    Expected: heads can now learn to cover 3–560 tokens; tests whether
              the architecture benefits when initialized for long-range

Both conditions use V4 architecture (no dispersion) for clean comparison to A.

Run with:
  /path/to/.venv/bin/python3 train_2048_longrange_ablation.py [--start_condition {e,f}]

Results saved to: 2048_longrange_ablation_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ─────────────────────────────────────────────────────────

VOCAB_SIZE     = 32000
NUM_EPOCHS     = 10
BATCH_SIZE     = 8        # matches existing 13M training (8×4=32 effective)
GRAD_ACCUM     = 4
LR             = 3e-4
MAX_SEQ_LEN    = 2048
NUM_DOCS       = 100_000
BPE_TRAIN_DOCS = 50_000

# 13M model config (matches Condition A baseline)
EMBEDDING_DIM  = 256
NUM_LAYERS     = 6
NUM_HEADS      = 8
FFN_DIM        = 1024
INTERFERENCE   = 3

# ─── Wave Field Attention V4 (parametrised) ───────────────────────────────────

class WaveFieldAttentionV4(nn.Module):
    """
    V4 Wave Field Attention with configurable field_size and damping floor.

    Identical to V4 from train_2048_dispersion_ablation.py except:
      - field_size is now a constructor argument (default 4096 for V4 parity)
      - damping_init and damping_floor are constructor arguments
      - no dispersion (use_dispersion always False — cleaner baseline)
    """
    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, dropout=0.1,
                 damping_init=None, damping_floor=0.05):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_heads      = num_heads
        self.head_dim       = embedding_dim // num_heads
        self.field_size     = field_size
        self.damping_floor  = damping_floor

        # Projections
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # V4: log-scale frequency init
        self.wave_frequency = nn.Parameter(
            torch.logspace(math.log10(0.1), math.log10(10.0), num_heads)
        )
        # Damping init: configurable (default V4: linspace(-2, 0))
        if damping_init is None:
            damping_init = torch.linspace(-2.0, 0.0, num_heads)
        self.wave_damping = nn.Parameter(damping_init.clone())

        # V4: full 2π phase coverage
        self.wave_phase = nn.Parameter(
            torch.linspace(0.0, 2.0 * math.pi * (num_heads - 1) / num_heads, num_heads)
        )

        self.field_coupling = nn.Parameter(torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads))
        self.dropout = nn.Dropout(dropout)

    def _build_kernels(self, device):
        """Build wave kernels with autograd — called each forward so wave params train."""
        G     = self.field_size
        t     = torch.arange(G, dtype=torch.float32, device=device)
        alpha = (F.softplus(self.wave_damping) + self.damping_floor).unsqueeze(1)  # [H,1]
        omega = self.wave_frequency.abs().unsqueeze(1)                              # [H,1]
        phi   = self.wave_phase.unsqueeze(1)                                        # [H,1]

        k = torch.exp(-alpha * t.unsqueeze(0)) \
          * torch.cos(omega * t.unsqueeze(0) + phi)         # [H, G]
        k = k / k.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.fft.rfft(k, n=2 * G)                   # [H, G+1] complex

    def _convolve(self, field, kernel_fft):
        """field: [B, H, G, D] → convolved field same shape. Matches original (B*D, H, G) layout."""
        B, H, G, D = field.shape
        pad = 2 * G
        # Permute to (B*D, H, G) — same batching as original for memory efficiency
        field_t   = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft = torch.fft.rfft(field_t, n=pad)          # [B*D, H, G+1]
        conv_fft  = field_fft * kernel_fft.unsqueeze(0)     # broadcast [H, G+1]
        convolved = torch.fft.irfft(conv_fft, n=pad)[:, :, :G]  # [B*D, H, G]
        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)  # [B, H, G, D]

    def forward(self, x):
        B, N, D = x.shape
        G = self.field_size
        H = self.num_heads
        HD = self.head_dim

        qkv = self.qkv_proj(x)                  # [B, N, 3D]
        q, k, v = qkv.split(D, dim=-1)          # [B, N, D] each
        q = q.view(B, N, H, HD).transpose(1, 2) # [B, H, N, HD]
        k = k.view(B, N, H, HD).transpose(1, 2)
        v = v.view(B, N, H, HD).transpose(1, 2)

        # Dynamic field position: stride from actual seq_len
        stride = (G - 1) / max(N - 1, 1)
        pos    = torch.arange(N, dtype=x.dtype, device=x.device) * stride  # [N]
        pos    = pos.clamp(0, G - 2)
        lo     = pos.long()
        frac   = (pos - lo.float()).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,N,1]

        # Bilinear scatter: deposit v * k_mag onto field
        k_mag = k.norm(dim=-1, keepdim=True)     # [B, H, N, 1]
        deposit = v * k_mag                       # [B, H, N, HD]

        field = torch.zeros(B, H, G, HD, device=x.device, dtype=x.dtype)
        # Scatter lower and upper neighbours
        lo_idx = lo.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, N, HD)
        hi_idx = (lo + 1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, N, HD)
        w_lo = (1 - frac).expand(B, H, N, HD)
        w_hi = frac.expand(B, H, N, HD)
        field.scatter_add_(2, lo_idx, deposit * w_lo)
        field.scatter_add_(2, hi_idx, deposit * w_hi)

        # Wave convolution (build kernels with autograd so wave params get gradients)
        kernel_fft = self._build_kernels(x.device)
        field = self._convolve(field, kernel_fft) # [B, H, G, D_head]

        # Field coupling (static cross-head mixing)
        coupling = F.softmax(self.field_coupling, dim=-1)  # [H, H]
        field = torch.einsum('ij,bjgd->bigd', coupling, field)

        # Content-dependent gate
        gate = torch.sigmoid(self.gate_proj(x))   # [B, N, D]

        # Bilinear gather
        lo_g = lo.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, N, HD)
        hi_g = (lo + 1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, N, HD)
        w_lo_g = (1 - frac).expand(B, H, N, HD)
        w_hi_g = frac.expand(B, H, N, HD)

        gathered = field.gather(2, lo_g) * w_lo_g + field.gather(2, hi_g) * w_hi_g
        gathered = gathered.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]

        out = self.out_proj(gathered * gate)
        return self.dropout(out)

    def alpha_summary(self):
        alpha = (F.softplus(self.wave_damping) + self.damping_floor).detach()
        e95   = 1.498 / alpha.clamp(min=1e-8)
        stride = (self.field_size - 1) / max(MAX_SEQ_LEN - 1, 1)
        tokens = e95 / stride
        return {
            'alpha_mean': alpha.mean().item(),
            'alpha_min':  alpha.min().item(),
            'alpha_max':  alpha.max().item(),
            'e95_tokens_mean': tokens.mean().item(),
            'e95_tokens_max':  tokens.max().item(),
        }


# ─── Transformer block + full model ──────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class WaveBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, field_size,
                 max_seq_len, dropout, damping_init, damping_floor,
                 use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = WaveFieldAttentionV4(
            embedding_dim, num_heads, field_size=field_size,
            max_seq_len=max_seq_len, dropout=dropout,
            damping_init=damping_init, damping_floor=damping_floor,
        )
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)
        # NOTE: interference pooling must be CAUSAL (cumulative mean over past only).
        # Non-causal global mean leaks future tokens — causing catastrophic data leakage.
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
            # Causal cumulative mean: position t sees mean of positions 0..t only
            B, N, D = xi.shape
            cumsum = xi.cumsum(dim=1)                                          # [B, N, D]
            counts = torch.arange(1, N + 1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
            pool   = cumsum / counts                                           # [B, N, D]
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class WaveFieldTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, field_size, max_seq_len, interference_interval,
                 dropout=0.1, damping_init=None, damping_floor=0.05):
        super().__init__()
        if damping_init is None:
            damping_init = torch.linspace(-2.0, 0.0, num_heads)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            WaveBlock(
                embedding_dim, num_heads, ffn_dim, field_size,
                max_seq_len, dropout, damping_init, damping_floor,
                use_checkpoint=True,
                interference=(i % interference_interval == interference_interval - 1),
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        # Weight-tied output projection
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

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def alpha_summary(self):
        # Average across all attention layers
        summaries = [b.attn.alpha_summary() for b in self.blocks]
        return {k: sum(s[k] for s in summaries) / len(summaries) for k in summaries[0]}


# ─── Data utilities ───────────────────────────────────────────────────────────

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
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
    ds = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 25000 == 0:
            print(f'  {i+1:,} docs...')
    print(f'  {len(texts):,} docs | train {int(len(texts)*0.95):,} | val 2,500 | test 2,500')
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
        tokens.append(3)  # </s>
    n = (len(tokens) // max_seq_len) * max_seq_len
    tokens = tokens[:n]
    data   = torch.tensor(tokens, dtype=torch.long)
    seqs   = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


# ─── Training loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size, 1:].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=150):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(ids[:, -MAX_SEQ_LEN:])
                next_id = logits[0, -1].argmax()
                ids = torch.cat([ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
        generated = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(generated[:120])
    return results


def train_condition(model, train_data, val_data, test_data, tokenizer,
                    condition_name, peak_lr=LR, save_dir='checkpoints',
                    device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                   weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = torch.amp.GradScaler('cuda')
    use_amp     = True

    best_val_loss = float('inf')
    best_val_ppl  = float('inf')
    best_epoch    = 0
    t0 = time.time()

    GEN_PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
    ]

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0
        optimizer.zero_grad()

        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx >= len(train_data): continue
                batch = train_data[indices[idx: idx + BATCH_SIZE]]
                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
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
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        val_acc    = 0.0  # skipping for brevity

        epoch_time = time.time() - t0
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl  = val_ppl
            best_epoch    = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} '
              f'PPL {val_ppl:.1f}{marker} | {epoch_time:.0f}s')

        # Alpha summary
        asumm = model.alpha_summary()
        print(f'  α mean={asumm["alpha_mean"]:.4f} min={asumm["alpha_min"]:.4f} '
              f'max={asumm["alpha_max"]:.4f} | '
              f'E95 max={asumm["e95_tokens_max"]:.1f} tokens')

        # Generation samples
        print('  ── Generation samples (greedy, 150 tokens) ──')
        gens = generate(model, tokenizer, GEN_PROMPTS, device)
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')

    # Final test eval
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  {condition_name} TEST: PPL {test_ppl:.1f} | Loss {test_loss:.4f}')

    # Final alpha summary
    asumm = model.alpha_summary()
    print(f'  Final α: mean={asumm["alpha_mean"]:.4f} '
          f'min={asumm["alpha_min"]:.4f} '
          f'max={asumm["alpha_max"]:.4f}')
    print(f'  Final E95 coverage: max={asumm["e95_tokens_max"]:.1f} tokens '
          f'/ {MAX_SEQ_LEN} ctx ({100*asumm["e95_tokens_max"]/MAX_SEQ_LEN:.1f}%)')

    return {
        'test_ppl':       test_ppl,
        'test_loss':      test_loss,
        'best_val_ppl':   best_val_ppl,
        'best_epoch':     best_epoch,
        'total_time':     time.time() - t0,
        'alpha_final':    asumm,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_condition', choices=['e', 'f'], default='e',
                        help='Skip conditions before this (e=run both, f=skip E)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  WAVE FIELD 13M — LONG-RANGE ALPHA ABLATION @ 2048 TOKENS')
    print('  Condition E: V4 + G=seq_len (geometric fix only)')
    print('  Condition F: V4 + G=seq_len + long-range α init (both fixes)')
    print('  Baseline: Condition A PPL 86.8 (V4, G=4096, original init)')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ─────────────────────────────────────────────────────────────────
    splits = load_data(NUM_DOCS)

    print(f'\nTraining BPE tokenizer (vocab={VOCAB_SIZE})...')
    raw_tok = train_bpe_tokenizer(splits['train'][:BPE_TRAIN_DOCS], vocab_size=VOCAB_SIZE)
    tokenizer = BPETokenizerWrapper(raw_tok)
    print(f'  BPE vocab: {tokenizer.vocab_size()} tokens')

    print('Encoding data (max_seq_len={})...'.format(MAX_SEQ_LEN))
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    results = {}

    # ── Condition E: G=seq_len fix only ──────────────────────────────────────
    if args.start_condition == 'e':
        print('\n' + '=' * 70)
        print('  CONDITION E — V4 + G=seq_len (geometric fix only)')
        print('  field_size=2048 (stride=1.0), damping=linspace(-2,0), floor=0.05')
        print('=' * 70)

        model_e = WaveFieldTransformer(
            vocab_size          = tokenizer.vocab_size(),
            embedding_dim       = EMBEDDING_DIM,
            num_layers          = NUM_LAYERS,
            num_heads           = NUM_HEADS,
            ffn_dim             = FFN_DIM,
            field_size          = MAX_SEQ_LEN,        # ← G=2048 (stride=1 at train len)
            max_seq_len         = MAX_SEQ_LEN,
            interference_interval = INTERFERENCE,
            damping_init        = torch.linspace(-2.0, 0.0, NUM_HEADS),
            damping_floor       = 0.05,               # ← same as V4
        ).to(device)

        print(f'  Wave V4 + G=seq_len: {model_e.param_count():,} params | '
              f'field_size={MAX_SEQ_LEN} | damping_floor=0.05')

        # Initial alpha check
        asumm = model_e.alpha_summary()
        print(f'  Init α: mean={asumm["alpha_mean"]:.4f} '
              f'max_E95={asumm["e95_tokens_max"]:.1f} tokens '
              f'({100*asumm["e95_tokens_max"]/MAX_SEQ_LEN:.1f}% of ctx)')

        res_e = train_condition(
            model_e, train_data, val_data, test_data, tokenizer,
            condition_name='Condition E',
            save_dir='2048_longrange_E_checkpoints',
            device=device,
        )
        results['E_g_seq_len_only'] = {
            **res_e,
            'description': 'V4 + G=seq_len (geometric fix only)',
            'field_size':  MAX_SEQ_LEN,
            'damping_init': 'linspace(-2.0, 0.0)',
            'damping_floor': 0.05,
        }

    # ── Condition F: G=seq_len + long-range α ─────────────────────────────────
    print('\n' + '=' * 70)
    print('  CONDITION F — V4 + G=seq_len + long-range α init')
    print('  field_size=2048 (stride=1.0), damping=linspace(-8,-1), floor=0.001')
    print('=' * 70)

    model_f = WaveFieldTransformer(
        vocab_size          = tokenizer.vocab_size(),
        embedding_dim       = EMBEDDING_DIM,
        num_layers          = NUM_LAYERS,
        num_heads           = NUM_HEADS,
        ffn_dim             = FFN_DIM,
        field_size          = MAX_SEQ_LEN,            # ← G=2048
        max_seq_len         = MAX_SEQ_LEN,
        interference_interval = INTERFERENCE,
        damping_init        = torch.linspace(-8.0, -1.0, NUM_HEADS),  # ← long-range
        damping_floor       = 0.001,                  # ← lowered from 0.05
    ).to(device)

    print(f'  Wave V4 + G=seq_len + long-range α: {model_f.param_count():,} params | '
          f'field_size={MAX_SEQ_LEN} | damping_floor=0.001')

    asumm = model_f.alpha_summary()
    print(f'  Init α: mean={asumm["alpha_mean"]:.4f} '
          f'max_E95={asumm["e95_tokens_max"]:.1f} tokens '
          f'({100*asumm["e95_tokens_max"]/MAX_SEQ_LEN:.1f}% of ctx)')

    res_f = train_condition(
        model_f, train_data, val_data, test_data, tokenizer,
        condition_name='Condition F',
        save_dir='2048_longrange_F_checkpoints',
        device=device,
    )
    results['F_g_seq_len_longrange'] = {
        **res_f,
        'description': 'V4 + G=seq_len + long-range α init',
        'field_size':  MAX_SEQ_LEN,
        'damping_init': 'linspace(-8.0, -1.0)',
        'damping_floor': 0.001,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  LONG-RANGE ABLATION RESULTS')
    print('=' * 70)
    print(f'  {"Model":<45} {"Test PPL":>9} {"vs A":>8}')
    print('  ' + '─' * 65)
    print(f'  {"Wave V4 baseline [A] (G=4096, orig init)":<45} {"86.8":>9} {"  +0.0":>8}')
    if 'E_g_seq_len_only' in results:
        e_ppl = results["E_g_seq_len_only"]["test_ppl"]
        print(f'  {"Wave V4 + G=seq_len only [E]":<45} {e_ppl:>9.1f} {e_ppl-86.8:>+8.1f}')
    f_ppl = results["F_g_seq_len_longrange"]["test_ppl"]
    print(f'  {"Wave V4 + G=seq_len + long-range α [F]":<45} {f_ppl:>9.1f} {f_ppl-86.8:>+8.1f}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_longrange_ablation_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
