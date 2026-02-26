"""
Wave Field 13M — Condition Byte: SpaceByte-style word-boundary patching.

Hypothesis:
  BPE tokenization creates spatially inconsistent positions — "the" is one
  BPE token (~3 chars) while "uncharacteristically" might be 4-5 tokens
  (~20 chars). The Wave Field's convolution treats all positions equally,
  so the concept of "local" (j0, RF=4 tokens) means very different amounts
  of text depending on which words landed in those 4 positions.

  Word-boundary byte patches make spatial structure consistent: each wave
  field position = one word, regardless of word length. The patch encoder
  derives the word embedding from its raw bytes, preserving sub-word
  morphological information without BPE's variable-length inconsistency.

Architecture:
  - BytePatchEncoder: replaces nn.Embedding(vocab_size, D)
      Embedding(257, 32) → Conv1d(32→32, kernel=3) → mean-pool → Linear(32→D)
      Params: ~20k (vs BPE embedding: 32k × 256 = 8.2M)
  - Wave field blocks: identical to Condition G (causal D4 DWT)
  - BytePatchDecoder: replaces nn.Linear(D, vocab_size)
      Linear(D, MAX_PATCH_BYTES × 257)
      Predicts all bytes of next word simultaneously
      Loss: mean cross-entropy over non-pad bytes (= bits-per-byte × log2)

Prediction task:
  Input:  byte patches of words 0..N-1
  Target: byte patches of words 1..N
  Loss:   average CE over all non-pad byte positions in target patches
  Metric: bits-per-byte (bpb) = loss / log(2)
          estimated word PPL = 2^(bpb * avg_bytes_per_word)
  Note: not directly comparable to BPE word-PPL; compare within-condition.

Why not fix query-less gather:
  This experiment tests the *tokenization* hypothesis, not the *gather*
  hypothesis. Condition I (query-conditioned scale selection) tests the
  gather fix. These are orthogonal experiments.

Prerequisite:
  Run prep_byte_patches.py first (CPU, ~20-30 min).
  Expects: byte_patch_data/byte_patches_{train,val,test}.pt

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condByte.py \
    2>&1 | tee condByte_run.log

Results → benchmarks/2048_condByte_results.json
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ──────────────────────────────────────────────────────────

NUM_EPOCHS      = 10
BATCH_SIZE      = 8          # 8 × 4 accum = 32 effective
GRAD_ACCUM      = 4
LR              = 3e-4
NUM_PATCHES     = 2048       # sequence length (patches = words)
MAX_PATCH_BYTES = 16         # bytes per patch (set in prep_byte_patches.py)
BYTE_VOCAB      = 257        # 0=pad, 1-256=byte+1

# Model config — same as Condition G for fair architecture comparison
EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
N_SCALES        = 11

BYTE_EMBED_DIM  = 32         # internal byte embedding dim in patch encoder
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(SCRIPT_DIR, '..', 'byte_patch_data')


# ─── Byte Patch Encoder ───────────────────────────────────────────────────────

class BytePatchEncoder(nn.Module):
    """
    Converts a word's byte sequence into a patch embedding.

    Input:  [B, N, P]  (uint8; 0=pad, 1-256=byte+1)
    Output: [B, N, D]  (float32 patch embeddings)

    Architecture:
      1. Byte embedding: [B, N, P] → [B, N, P, byte_embed_dim]
         padding_idx=0 so pad bytes contribute zero gradient
      2. 1D conv over byte positions (kernel=3, captures byte n-grams)
      3. Masked mean-pool over non-pad bytes
      4. Linear projection to EMBEDDING_DIM
    """
    def __init__(self, byte_vocab=BYTE_VOCAB, byte_embed_dim=BYTE_EMBED_DIM,
                 max_patch_bytes=MAX_PATCH_BYTES, output_dim=EMBEDDING_DIM):
        super().__init__()
        self.max_patch_bytes = max_patch_bytes
        self.byte_embed = nn.Embedding(byte_vocab, byte_embed_dim, padding_idx=0)
        # Conv over byte positions — captures morphological patterns
        self.conv = nn.Conv1d(byte_embed_dim, byte_embed_dim, kernel_size=3,
                              padding=1, bias=True)
        self.proj = nn.Linear(byte_embed_dim, output_dim)
        self._init()

    def _init(self):
        nn.init.normal_(self.byte_embed.weight, 0, 0.02)
        nn.init.zeros_(self.byte_embed.weight[0])   # pad embedding = zero
        nn.init.normal_(self.conv.weight, 0, 0.02)
        nn.init.zeros_(self.conv.bias)
        nn.init.normal_(self.proj.weight, 0, 0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: [B, N, P]
        B, N, P = x.shape
        x_long = x.long()

        # [B, N, P, E]
        emb = self.byte_embed(x_long.reshape(B * N, P))  # [B*N, P, E]

        # 1D conv over byte positions
        emb_t = emb.permute(0, 2, 1)        # [B*N, E, P]
        emb_t = F.gelu(self.conv(emb_t))     # [B*N, E, P]
        emb   = emb_t.permute(0, 2, 1)       # [B*N, P, E]

        # Masked mean-pool: average over non-pad byte positions only
        mask = (x.reshape(B * N, P) > 0).float().unsqueeze(-1)  # [B*N, P, 1]
        patch_emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B*N, E]

        patch_emb = patch_emb.reshape(B, N, -1)   # [B, N, E]
        return self.proj(patch_emb)                # [B, N, D]

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ─── Byte Patch Decoder ───────────────────────────────────────────────────────

class BytePatchDecoder(nn.Module):
    """
    Predicts the bytes of the next word from the current position's embedding.

    Input:  [B, N, D]  model output embeddings
    Output: [B, N, MAX_PATCH_BYTES, BYTE_VOCAB]  byte logits per position
    """
    def __init__(self, input_dim=EMBEDDING_DIM, max_patch_bytes=MAX_PATCH_BYTES,
                 byte_vocab=BYTE_VOCAB):
        super().__init__()
        self.max_patch_bytes = max_patch_bytes
        self.byte_vocab      = byte_vocab
        self.linear = nn.Linear(input_dim, max_patch_bytes * byte_vocab)
        nn.init.normal_(self.linear.weight, 0, 0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        out = self.linear(x)   # [B, N, P*V]
        return out.reshape(B, N, self.max_patch_bytes, self.byte_vocab)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ─── Wave Field Architecture (identical to Condition G) ───────────────────────

class CausalWaveletFieldAttentionByte(nn.Module):
    """Condition G attention — unchanged from G."""

    _D4 = [0.4829629131445341,  0.8365163037378079,
           0.2241438680420134, -0.1294095225512604]

    def __init__(self, embedding_dim, num_heads, seq_len=NUM_PATCHES,
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

        C  = num_heads * (embedding_dim // num_heads)
        d4 = torch.tensor(self._D4, dtype=torch.float32)
        conv_w = d4.view(1, 1, 4).expand(C, 1, 4).contiguous()
        self.register_buffer('conv_weight', conv_w)

        self.scale_gain = nn.Parameter(torch.zeros(n_scales, num_heads))

        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads))

        self.dropout = nn.Dropout(dropout)

    def _causal_multiscale(self, field):
        B, H, N, HD = field.shape
        C = H * HD
        x = field.permute(0, 1, 3, 2).reshape(B, C, N)
        gains = F.softmax(self.scale_gain, dim=0)
        gains_exp = gains.unsqueeze(-1).expand(self.n_scales, H, HD).reshape(self.n_scales, C)
        out = torch.zeros_like(x)
        for j in range(self.n_scales):
            d   = 1 << j
            pad = 3 * d
            x_pad = F.pad(x, (pad, 0))
            y = F.conv1d(x_pad, self.conv_weight, dilation=d, groups=C)
            g = gains_exp[j].unsqueeze(0).unsqueeze(-1)
            out = out + g * y
        return out.reshape(B, H, HD, N).permute(0, 1, 3, 2)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        k_mag = k.norm(dim=-1, keepdim=True)
        field = v * k_mag
        field = self._causal_multiscale(field)
        coupling = F.softmax(self.field_coupling, dim=-1)
        field    = torch.einsum('ij,bjnd->bind', coupling, field)
        gate     = torch.sigmoid(self.gate_proj(x))
        gathered = field.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.dropout(self.out_proj(gathered * gate))

    def scale_summary(self):
        with torch.no_grad():
            gains = F.softmax(self.scale_gain, dim=0)
        return {
            'gains_mean_per_scale': gains.mean(dim=1).tolist(),
            'dominant_scale':       int(gains.mean(dim=1).argmax().item()),
            'dominant_scale_rf_tokens': 4 * (1 << int(gains.mean(dim=1).argmax().item())),
        }


class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class CausalWaveletBlockByte(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = CausalWaveletFieldAttentionByte(
            embedding_dim, num_heads, seq_len=seq_len,
            n_scales=n_scales, dropout=dropout)
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


class ByteWaveletTransformer(nn.Module):
    """
    Wave Field transformer with byte-patch encoder/decoder.
    Wave field blocks are identical to Condition G.
    """
    def __init__(self, embedding_dim, num_layers, num_heads, ffn_dim,
                 seq_len, n_scales, interference_interval,
                 byte_embed_dim=BYTE_EMBED_DIM, dropout=0.1):
        super().__init__()
        # Byte patch encoder (replaces nn.Embedding)
        self.patch_encoder = BytePatchEncoder(
            byte_vocab=BYTE_VOCAB, byte_embed_dim=byte_embed_dim,
            max_patch_bytes=MAX_PATCH_BYTES, output_dim=embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CausalWaveletBlockByte(
                embedding_dim, num_heads, ffn_dim, seq_len, n_scales,
                dropout=dropout, use_checkpoint=True,
                interference=(i % interference_interval == interference_interval - 1))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

        # Byte patch decoder (replaces nn.Linear(D, vocab_size))
        # No weight tying possible with byte encoder (different shapes)
        self.patch_decoder = BytePatchDecoder(
            input_dim=embedding_dim,
            max_patch_bytes=MAX_PATCH_BYTES,
            byte_vocab=BYTE_VOCAB)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # Preserve special inits
        for block in self.blocks:
            nn.init.constant_(block.attn.gate_proj.bias, 2.0)
        nn.init.zeros_(self.patch_encoder.byte_embed.weight[0])

    def forward(self, patches):
        # patches: [B, N, P] uint8
        B, N, P = patches.shape
        pos = torch.arange(N, device=patches.device).unsqueeze(0)  # [1, N]

        x = self.patch_encoder(patches.float().long())  # [B, N, D]
        x = self.drop(x + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.patch_decoder(x)   # [B, N, P, V]

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
            'dominant_scale': dom,
            'dominant_scale_rf_tokens': 4 * (1 << dom),
        }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_byte_patches():
    print('Loading byte-patch data...')
    splits = {}
    for split in ('train', 'val', 'test'):
        path = os.path.join(DATA_DIR, f'byte_patches_{split}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Missing: {path}\n'
                f'Run prep_byte_patches.py first.')
        data = torch.load(path, weights_only=True)  # [N, 2048, 12] uint8
        splits[split] = data
        print(f'  {split}: {len(data):,} sequences '
              f'[{data.shape[1]} patches × {data.shape[2]} bytes]')
    return splits['train'], splits['val'], splits['test']


# ─── Loss and evaluation ──────────────────────────────────────────────────────

def byte_loss(logits, patches):
    """
    Compute masked byte prediction loss.

    logits:  [B, N-1, P, V]  — predictions for positions 0..N-2
    patches: [B, N, P]       — targets (bytes of positions 1..N)

    Loss: mean CE over non-pad bytes of target patches.
    Returns (loss, n_bytes_predicted) for accurate BPC computation.
    """
    targets = patches[:, 1:, :].long()           # [B, N-1, P]
    B, Nm1, P = targets.shape
    V = logits.size(-1)

    mask = (targets > 0).float()                  # [B, N-1, P]
    n_bytes = mask.sum()

    # Cross-entropy with ignore_index=0 (pad)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=0,
        reduction='sum',
    )
    return loss / (n_bytes + 1e-8), n_bytes


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss   = 0.0
    total_bytes  = 0
    for i in range(0, len(data) - batch_size, batch_size):
        patches = data[i:i + batch_size].to(device)          # [B, N, P]
        logits  = model(patches[:, :-1, :])                  # [B, N-1, P, V]
        loss, nb = byte_loss(logits, patches)
        total_loss  += loss.item() * nb.item()
        total_bytes += nb.item()
    avg_loss = total_loss / max(total_bytes, 1)
    bpc      = avg_loss / math.log(2)     # bits per byte
    return avg_loss, bpc


def generate_bytes(model, prompts_bytes, device, max_new_patches=50):
    """Generate text by predicting next word bytes greedily."""
    model.eval()
    results = []
    for prompt_patches in prompts_bytes:
        # prompt_patches: list of [MAX_PATCH_BYTES] uint8 arrays
        ids = torch.zeros(1, len(prompt_patches), MAX_PATCH_BYTES,
                          dtype=torch.uint8, device=device)
        for i, p in enumerate(prompt_patches):
            ids[0, i, :len(p)] = torch.tensor(
                [b + 1 for b in p], dtype=torch.uint8)

        generated_words = []
        with torch.no_grad():
            for _ in range(max_new_patches):
                ctx = ids[:, -NUM_PATCHES:, :]
                logits = model(ctx)                    # [1, N, P, V]
                last_logits = logits[0, -1, :, :]     # [P, V]
                # Greedy: pick most likely byte at each position
                pred_bytes = last_logits.argmax(dim=-1)  # [P]
                # Decode: value 0 = pad (stop), 1-256 = byte
                word_bytes = []
                for b in pred_bytes.tolist():
                    if b == 0:
                        break
                    word_bytes.append(b - 1)
                try:
                    word = bytes(word_bytes).decode('utf-8', errors='replace')
                except:
                    word = '?'
                generated_words.append(word)

                # Append predicted patch to context
                new_patch = torch.zeros(1, 1, MAX_PATCH_BYTES,
                                        dtype=torch.uint8, device=device)
                for pi, bv in enumerate(pred_bytes[:MAX_PATCH_BYTES].tolist()):
                    new_patch[0, 0, pi] = bv
                ids = torch.cat([ids, new_patch], dim=1)

        results.append(''.join(generated_words))
    return results


def text_to_patches(text, max_patches=20):
    """Convert a prompt string to a list of byte patches."""
    import re
    words = re.findall(r'[ \t]*\S+', text)
    patches = []
    for w in words[:max_patches]:
        b = w.encode('utf-8')[:MAX_PATCH_BYTES]
        patches.append(b)
    return patches


# ─── Training loop ────────────────────────────────────────────────────────────

def train_condition_byte(model, train_data, val_data, test_data,
                         save_dir='2048_condByte_checkpoints', device='cuda'):
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
    GEN_PATCHES = [text_to_patches(p) for p in GEN_PROMPTS]

    best_val_loss = float('inf')
    best_bpc      = float('inf')
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
                patches = batch.to(device)                        # [B, N, P]
                with torch.amp.autocast('cuda'):
                    logits = model(patches[:, :-1, :])            # [B, N-1, P, V]
                    loss, _ = byte_loss(logits, patches)
                    loss    = loss / GRAD_ACCUM
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                bpc_step = loss.item() * GRAD_ACCUM / math.log(2)
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item()*GRAD_ACCUM:.4f} '
                      f'| BPC {bpc_step:.4f}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss, val_bpc = evaluate(model, val_data, BATCH_SIZE, device)

        # Approximate word PPL: BPC × avg_bytes_per_word
        # English avg word length (with leading space) ≈ 5.5 bytes
        AVG_WORD_BYTES = 5.5
        word_ppl_approx = 2 ** (val_bpc * AVG_WORD_BYTES)

        epoch_time = time.time() - t0
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bpc      = val_bpc
            best_epoch    = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} | BPC {val_bpc:.4f} '
              f'| ~WordPPL {word_ppl_approx:.1f}{marker} | {epoch_time:.0f}s')

        ss = model.scale_summary()
        gains = ss['gains_mean_per_scale']
        top3  = sorted(range(N_SCALES), key=lambda j: -gains[j])[:3]
        print(f'  Scale gains (top-3): ' +
              ', '.join(f'j={j} rf={4*(1<<j)}tok gain={gains[j]:.3f}'
                        for j in top3))

        print('  ── Generation samples (greedy, 50 words) ──')
        gens = generate_bytes(model, GEN_PATCHES, device, max_new_patches=50)
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')

    # Final test evaluation
    model.load_state_dict(torch.load(
        os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss, test_bpc = evaluate(model, test_data, BATCH_SIZE, device)
    word_ppl_final = 2 ** (test_bpc * 5.5)

    print(f'\n  CONDITION BYTE TEST:')
    print(f'    BPC        : {test_bpc:.4f} bits/byte')
    print(f'    ~Word PPL  : {word_ppl_final:.1f}  (BPC × 5.5 bytes/word approx)')
    print(f'    Note: not directly comparable to BPE word-PPL')
    print(f'    Baseline context: Condition A (BPE) = 86.8 word-PPL')

    ss = model.scale_summary()
    print(f'  Final dominant scale: j={ss["dominant_scale"]} '
          f'(RF={ss["dominant_scale_rf_tokens"]} words)')
    print(f'  Full gains: '
          + ' '.join(f'j{j}={ss["gains_mean_per_scale"][j]:.3f}'
                     for j in range(N_SCALES)))

    return {
        'test_bpc':             test_bpc,
        'test_loss':            test_loss,
        'word_ppl_approx':      word_ppl_final,
        'best_val_bpc':         best_bpc,
        'best_epoch':           best_epoch,
        'total_time':           time.time() - t0,
        'scale_summary':        ss,
        'avg_word_bytes_assumed': 5.5,
        'note': ('BPC is bits-per-byte; word_ppl_approx = 2^(BPC * avg_word_bytes). '
                 'Not directly comparable to BPE word PPL. Compare within condByte '
                 'vs. further condByte variants, or use BPC as standalone metric.'),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  WAVE FIELD — CONDITION BYTE: WORD-BOUNDARY BYTE PATCHING')
    print('  SpaceByte-style: each position = one word (bytes → embedding)')
    print('  Hypothesis: spatial consistency improves Wave Field performance')
    print('=' * 70)
    print(f'\n  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Load preprocessed data ────────────────────────────────────────────────
    train_data, val_data, test_data = load_byte_patches()

    # ── Build model ───────────────────────────────────────────────────────────
    model = ByteWaveletTransformer(
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = NUM_PATCHES,
        n_scales              = N_SCALES,
        interference_interval = INTERFERENCE,
        byte_embed_dim        = BYTE_EMBED_DIM,
    ).to(device)

    n_total = model.param_count()
    n_enc   = model.patch_encoder.param_count()
    n_dec   = model.patch_decoder.param_count()
    n_wave  = n_total - n_enc - n_dec
    print(f'\nCondition Byte: {n_total:,} params total')
    print(f'  Patch encoder:  {n_enc:,} params (vs BPE embedding: 8,192,000)')
    print(f'  Wave field:     {n_wave:,} params (identical to G)')
    print(f'  Patch decoder:  {n_dec:,} params (vs BPE output: 8,192,000 shared)')
    print(f'  (G total: 14,115,728 — {n_total - 14_115_728:+,} params vs G)')
    print(f'\n  Config: {NUM_PATCHES} patches × {MAX_PATCH_BYTES} bytes/patch')
    print(f'  Metric: bits-per-byte (BPC); lower is better')
    print(f'  Reference: good byte LMs achieve ~1.0-1.5 BPC on English')

    print(f'\n  Scale receptive fields (in words/patches):')
    for j in range(N_SCALES):
        d  = 1 << j
        rf = 4 * d
        print(f'    j={j:2d}: dilation={d:4d}, RF={rf:5d} words'
              f'  ({100*rf/NUM_PATCHES:.1f}% of context)')

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train_condition_byte(
        model, train_data, val_data, test_data,
        save_dir='2048_condByte_checkpoints',
        device=device,
    )
    results['description'] = (
        'Wave Field G with SpaceByte-style word-boundary byte patching; '
        'BytePatchEncoder (Emb+Conv+pool) replaces BPE embedding; '
        'BytePatchDecoder predicts next word bytes'
    )
    results['architecture'] = {
        'patch_encoder': 'BytePatchEncoder(257, 32) → Conv1d(3) → mean-pool → Linear(32,256)',
        'wave_field':    'identical to Condition G (causal D4 DWT)',
        'patch_decoder': f'Linear(256, {MAX_PATCH_BYTES}×{BYTE_VOCAB})',
        'MAX_PATCH_BYTES': MAX_PATCH_BYTES,
        'NUM_PATCHES': NUM_PATCHES,
    }

    print('\n' + '=' * 70)
    print('  CONDITION BYTE RESULTS')
    print('=' * 70)
    print(f'  {"Condition Byte — BPC":<50} {results["test_bpc"]:>8.4f}')
    print(f'  {"Condition Byte — ~Word PPL (BPC×5.5 bytes/word)":<50} '
          f'{results["word_ppl_approx"]:>8.1f}')
    print(f'  (Condition G BPE word PPL for scale reference: 99.4)')

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_condByte_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
