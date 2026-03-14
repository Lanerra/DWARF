"""
🔬 DWARF Autoresearch — J16D 12M Flagship

12M champion architecture with:
  - V=8K vocabulary (better gradient flow through tied LM head)
  - condV physics: Kalman-EMA + KdV soliton + AGC
  - J16D relay-optimal offsets

Architecture: D=256, H=8, FFN=2048, L=6 (~12M params)
Physics: EMA α₀=0.003 (window≈333t), KdV zero-init, AGC normalization

Run (from /tmp/dwarf-j17d):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_j16d_12m_flagship.py \
    > autoresearch/run_j16d_12m_flagship.log 2>&1 &

Resume from checkpoint:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_j16d_12m_flagship.py --resume \
    > autoresearch/run_j16d_12m_flagship_resume.log 2>&1 &
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT KNOBS
# ═══════════════════════════════════════════════════════════════════════════════

OFFSETS = [1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM   = 256
NUM_HEADS       = 8
FFN_DIM         = 2048
NUM_LAYERS      = 6
VOCAB_SIZE      = 8192
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5

SCALE_EMBED_INIT = 0.1
LR               = 3e-4
LR_MULT          = 15.0
EMA_INIT         = 0.003
EMA_FLOOR        = 0.0003

MAX_TRAIN_SEQS = 58_600
SCREEN_EPOCHS  = 3
GRAD_ACCUM     = 4
BATCH_SIZE     = 8
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

# ═══════════════════════════════════════════════════════════════════════════════

FW_CACHE_FILE  = '/home/dlewis3/Desktop/AI/DWARF/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_PATH = '/tmp/dwarf-j17d/results/fineweb_v8k_tokenizer.json'
ENCODED_CACHE  = '/tmp/dwarf-j17d/logs/fineweb_encoded_2048_v8k.pt'

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 20
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

CHECKPOINT_DIR  = '/tmp/dwarf-j17d/autoresearch/checkpoints'
CHECKPOINT_NAME = 'j16d_12m_flagship_best.pt'

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_attention_v6j16d import DSQGAttentionV6J16D as DSQGAttentionV6

assert len(OFFSETS) == 16, f"OFFSETS has {len(OFFSETS)} entries but V6 kernel requires exactly 16."
assert FULL_ATTN_LAYER == NUM_LAYERS - 1
assert EMBEDDING_DIM // NUM_HEADS == 32, f"head_dim must be 32 (power of 2), got {EMBEDDING_DIM // NUM_HEADS}"

# ═══════════════════════════════════════════════════════════════════════════════
# condV PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

_EMA_KERNEL_LEN = 256


def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor,
                floor: float = 0.0003) -> torch.Tensor:
    """Causal EMA via depthwise conv with exponential kernel."""
    B, N, D = xi.shape
    alpha   = ema_factor.clamp(floor, 0.5)
    k_len   = min(_EMA_KERNEL_LEN, N)
    t       = torch.arange(k_len, device=xi.device, dtype=torch.float32)
    kernel  = alpha.float() * (1.0 - alpha.float()).pow(t)
    kernel  = (kernel / kernel.sum()).flip(0)
    xi_f    = xi.float()
    xi_bd   = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p    = F.pad(xi_bd, (k_len - 1, 0))
    pool    = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(xi.dtype)


def _kdv_correction(pool: torch.Tensor,
                    kdv_alpha: torch.Tensor) -> torch.Tensor:
    """KdV soliton: pool += α * pool * Δpool. Zero-init → identity at start."""
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    return pool + alpha * pool * (pool - pool_prev)


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AGC: normalise to unit RMS per token. No learnable params."""
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV6Physics(nn.Module):
    """V6 DSQG attention + condV interference (EMA + KdV + AGC)."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionV6(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))
            self.kdv_alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim

            pool = _causal_ema(xi, self.ema_factor, floor=EMA_FLOOR)
            pool = _kdv_correction(pool, self.kdv_alpha)
            pool = _agc_normalize(pool)

            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            kv_inject = (k_delta, v_delta)

        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AutoresearchTransformerPhysics(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer, interference_interval,
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed       = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop            = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                has_if = (interference_interval is not None and
                          i % interference_interval == interference_interval - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights(scale_embed_init_val)

    def _init_weights(self, scale_embed_init_val):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def physics_summary(self):
        """Log EMA and KdV state for all interference blocks."""
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = block.ema_factor.item()
                kdv   = block.kdv_alpha.item()
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t) kdv={kdv:.4f}')
        return '  '.join(entries)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA / TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def load_or_train_tokenizer():
    from tokenizers import Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        print(f'Loading V=8K tokenizer from {TOKENIZER_PATH}')
        return BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))

    print(f'Training V=8192 BPE tokenizer from {FW_CACHE_FILE}...')
    from tokenizers.models import BPE
    from tokenizers import trainers
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    with open(FW_CACHE_FILE) as f:
        texts = json.load(f)
    print(f'  {len(texts):,} docs for training')

    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=['<unk>', '<pad>', '<s>', '</s>'],
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(
        (t if isinstance(t, str) else t.get('text', '') for t in texts),
        trainer=trainer)
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tok.save(TOKENIZER_PATH)
    print(f'  Saved → {TOKENIZER_PATH}  (vocab={tok.get_vocab_size()})')
    return BPETokenizerWrapper(tok)


def load_data():
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = json.load(fp)
        print(f'  {len(texts):,} docs')
    else:
        raise FileNotFoundError(f'Doc cache not found: {FW_CACHE_FILE}')
    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95):int(n * 0.95) + 2500],
    }


def encode_split(split_texts, tokenizer, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text if isinstance(text, str) else text.get('text', '')))
        tokens.append(3)
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device)
        y = data[i:i+BATCH_SIZE,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for d in PASSKEY_DISTANCES:
        correct, n_valid = 0, 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available:
                continue
            filler = []
            while len(filler) < d:
                filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target] + others[:9])[
                            logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(resume: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    print('=' * 70)
    print('  🔬 DWARF — J16D 12M Flagship')
    print('  V=8K + condV Physics (EMA + KdV + AGC)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  head_dim={EMBEDDING_DIM // NUM_HEADS}')
    print(f'  IF interval={INTERFERENCE}, Full attn layer={FULL_ATTN_LAYER}')
    print(f'  scale_embed init={SCALE_EMBED_INIT}, LR mult={LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')

    # ── Tokenizer & data ──────────────────────────────────────────────────
    tokenizer = load_or_train_tokenizer()
    actual_vocab = tokenizer.vocab_size()
    print(f'Tokenizer vocab: {actual_vocab}')

    splits = load_data()

    if os.path.exists(ENCODED_CACHE):
        print(f'Loading pre-encoded dataset from {ENCODED_CACHE}')
        _cache     = torch.load(ENCODED_CACHE, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
    else:
        print(f'Encoding data with V={actual_vocab} tokenizer...')
        train_data = encode_split(splits['train'], tokenizer, 'Train')
        val_data   = encode_split(splits['val'],   tokenizer, 'Val')
        os.makedirs(os.path.dirname(ENCODED_CACHE) or '.', exist_ok=True)
        torch.save({'train': train_data, 'val': val_data}, ENCODED_CACHE)
        print(f'Saved encoded cache → {ENCODED_CACHE}')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoresearchTransformerPhysics(
        vocab_size            = actual_vocab,
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
        scale_embed_init_val  = SCALE_EMBED_INIT,
    ).to(device)

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    start_epoch = 1
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if resume and os.path.exists(checkpoint_path):
        print(f'Resuming from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        start_epoch = SCREEN_EPOCHS + 1
        print(f'  Loaded checkpoint, continuing from epoch {start_epoch}')

    # ── Optimizer ─────────────────────────────────────────────────────────
    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = torch.optim.AdamW([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params,     'lr': LR * LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + SCREEN_EPOCHS):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, actual_vocab),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = OrderedDict((k, v.cpu()) for k, v in model.state_dict().items())
            torch.save(state_dict, checkpoint_path)
            marker = ' *'

        print(f'\nEp {epoch}/{start_epoch + SCREEN_EPOCHS - 1} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionV6):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_s     = time.time() - t_start
    memory_mb     = torch.cuda.max_memory_allocated() / 1e6
    final_epoch   = start_epoch + SCREEN_EPOCHS - 1
    passkey_final = passkey_results.get(final_epoch, 0.0)
    ppl_final     = ppl_results.get(final_epoch, 999.0)
    PPL_BASELINE     = 61.75
    PASSKEY_BASELINE = 18.3
    ar_score = (passkey_final - PASSKEY_BASELINE) - max(0, ppl_final - PPL_BASELINE) * 0.5

    print('\n---')
    for ep in sorted(passkey_results.keys()):
        print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
    for ep in sorted(ppl_results.keys()):
        print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score:       {ar_score:.2f}')
    print(f'memory_mb:      {memory_mb:.1f}')
    print(f'elapsed_s:      {elapsed_s:.1f}')
    print(f'num_params_M:   {n_params / 1e6:.1f}')
    print(f'vocab_size:     {actual_vocab}')
    print(f'head_dim:       {EMBEDDING_DIM // NUM_HEADS}')
    print(f'scale_embed_lr_mult: {LR_MULT}')
    print(f'scale_embed_init: {SCALE_EMBED_INIT}')
    print(f'ema_init:       {EMA_INIT}')
    print(f'description:    J16D 12M flagship — D={EMBEDDING_DIM}, V={actual_vocab}, '
          f'FFN={FFN_DIM}, L={NUM_LAYERS}, condV physics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint for additional epochs')
    args = parser.parse_args()
    train(resume=args.resume)
