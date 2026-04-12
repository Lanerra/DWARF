"""
🧪 DWARF D=512 L=13 Selective RoPE — position-dependent Q/K rotation for distal offsets

Architecture: D=512, H=8 (hd=64), L=13, FFN=1024, TIED lm_head
  Same triadic [A,A,B,B,C,C] layout as d512_l13_triadic_aabbc.
  DSQGAttentionGroupedSelectiveRoPE replaces DSQGAttentionGrouped:
    Groups B and C (j_small=0, all J_LARGE) get absolute-position RoPE on Q/K.
    Group A (j_small=17, j_large=15): two-pass merge (small unrotated, large rotated).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_selective_rope_4090_bf16.py
"""

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available, using standard AdamW")

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

USE_LIGER_CE = False

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_dynamic_j import ALL_OFFSETS as _ALL_96
from dsqg_attention_v8_selective_rope import (
    DSQGAttentionGroupedSelectiveRoPE as DSQGAttentionGrouped,
    npci_rotate, R_PLANES,
)
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

# =============================================================================
# OFFSET GROUPS
# =============================================================================

GROUP_A = _ALL_96[0:32]
GROUP_B = _ALL_96[32:64]
GROUP_C = _ALL_96[64:96]
GROUP_T = sorted([1, 2, 3] + _ALL_96[48:], key=lambda d: (0 if d <= 28 else 1, d))

def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets), (
        f"J_SMALL({j_small}) + J_LARGE({j_large}) != J({len(offsets)})")
    return j_small, j_large

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)
J_SMALL_T, J_LARGE_T = _count_small_large(GROUP_T)

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 1024
NUM_LAYERS       = 13
FULL_ATTN_LAYER  = 3
VOCAB_SIZE       = 32768

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 20.0
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
LR        = 3e-4
DROPOUT   = 0.1

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '16'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '8'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '200000'))
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582
CE_CHUNK       = 512
SCREEN_EPOCHS  = 3

TOKENIZER_CANDIDATES = ['results/fineweb_tokenizer_32k.json']
PASSKEY_DISTANCES    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS       = 50
PASSKEY_BATCH_SIZE   = 8
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'
CKPT_BASE_NAME    = 'd512_l13_selective_rope'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# LAYER LAYOUT: L=13, FA@L3
# =============================================================================

LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L0: cycle 1
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L1
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, True),    # L2 + preIF
    ('FA', None, 0, 0, False),                       # L3: FullAttention
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L4: cycle 1 cont.
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L5
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L6
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L7: cycle 2
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L8
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L9
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L10
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L11
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L12
]

assert len(LAYER_LAYOUT) == NUM_LAYERS

# =============================================================================
# MODEL BLOCKS
# =============================================================================

def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _unwrap_compiled_module(module):
    return getattr(module, '_orig_mod', module)

def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)

def _agc_normalize(pool, eps=1e-6):
    D = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockTriadic(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 offsets, j_small, j_large, group_label,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.group_label = group_label
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = DSQGAttentionGrouped(
            embedding_dim, num_heads, offsets, j_small, j_large,
            seq_len=seq_len, dropout=dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)
            pool = _agc_normalize(pool)
            inter = torch.sigmoid(self.inter_gate(xi)) * pool
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
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
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
        self.attn = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MODEL
# =============================================================================

class TriadicJ96(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_dim, seq_len,
                 full_attn_layer, scale_embed_init_val=0.15, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i, (label, offsets, js, jl, has_if) in enumerate(LAYER_LAYOUT):
            if label == 'FA':
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                blocks.append(DSQGBlockTriadic(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    offsets, js, jl, group_label=label,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
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
            if isinstance(m, DSQGAttentionGrouped):
                nn.init.normal_(m.phase_base, 0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes, 0.0, 0.01)
                nn.init.normal_(m.phase_gain, 0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx):
        if block_idx == self.full_attn_layer:
            return True
        if block_idx == self.full_attn_layer - 1:
            return True
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.full_attn_layer
        return False

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def forward_hidden(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionGrouped):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}[{block.group_label}]: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)

    def layer_summary(self):
        parts = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic):
                label = block.group_label
                j = block.attn.j_val
                iflag = '+IF' if block.interference else ''
                rope = '+RoPE' if (block.attn.j_small == 0 and block.attn.j_large > 0) else ''
                if block.attn.j_small > 0 and block.attn.j_large > 0:
                    rope = '+RoPE(mixed)'
                parts.append(f'L{i}:DSQG-{label}(J={j}){iflag}{rope}')
            else:
                parts.append(f'L{i}:FA')
        return '  '.join(parts)


# =============================================================================
# DATA UTILITIES
# =============================================================================

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    bs = max(1, BATCH_SIZE // 2)
    for i in range(0, len(data) - bs + 1, bs):
        x = data[i:i+bs, :-1].to(device, non_blocking=True)
        y = data[i:i+bs, 1:].to(device, non_blocking=True)
        with _amp_context(device):
            logits = model(x)
        T, V = logits.size(1), logits.size(2)
        batch_loss = 0.0
        for c in range(0, T, CE_CHUNK):
            lc = logits[:, c:c+CE_CHUNK, :].reshape(-1, V).float()
            yc = y[:, c:c+CE_CHUNK].reshape(-1)
            batch_loss += F.cross_entropy(lc, yc, reduction='sum').item()
        total_loss += batch_loss
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)
    pad_id = 0
    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        if not encoded:
            raise ValueError(f'Could not encode passkey word: {word}')
        word_token_ids[word] = encoded[0]

    results = {}
    for d in PASSKEY_DISTANCES:
        seqs, last_pos, cand_rows = [], [], []
        for i in range(PASSKEY_TRIALS):
            target = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others = [w for w in _PASSKEY_WORDS if w != target]
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
            seqs.append(full_seq + [pad_id] * (MAX_SEQ_LEN - len(full_seq)))
            last_pos.append(len(full_seq) - 1)
            cand_words = [target] + others[:9]
            cand_rows.append([word_token_ids[w] for w in cand_words])

        if not seqs:
            results[d] = 0.0
            continue

        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        pos = torch.tensor(last_pos, dtype=torch.long, device=device)
        cand = torch.tensor(cand_rows, dtype=torch.long, device=device)

        correct = 0
        total = ids.size(0)
        for start in range(0, total, PASSKEY_BATCH_SIZE):
            ids_b = ids[start:start + PASSKEY_BATCH_SIZE]
            pos_b = pos[start:start + PASSKEY_BATCH_SIZE]
            cand_b = cand[start:start + PASSKEY_BATCH_SIZE]
            with _amp_context(device):
                logits = model(ids_b)
            row = torch.arange(ids_b.size(0), device=device)
            next_logits = logits[row, pos_b, :]
            cand_logits = torch.gather(next_logits, 1, cand_b)
            correct += (cand_logits.argmax(dim=1) == 0).sum().item()
        results[d] = correct / total
    return results


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir):
    full_attn_block = _unwrap_compiled_module(model.blocks[model.full_attn_layer])
    state_dict = {}
    for name, param in full_attn_block.named_parameters():
        state_dict[f"blocks.{model.full_attn_layer}.{name}"] = param.data.clone()
    payload = {
        "full_attn_block": state_dict,
        "config": {
            "embedding_dim": EMBEDDING_DIM, "num_heads": NUM_HEADS,
            "ffn_dim": FFN_DIM, "seq_len": MAX_SEQ_LEN,
            "source_script": "train/train_d512_l13_selective_rope_4090_bf16.py",
            "source_layer": FULL_ATTN_LAYER, "num_layers": NUM_LAYERS,
            "epoch": epoch, "git_hash": git_hash,
            "note": (f"Selective RoPE L=13: D={EMBEDDING_DIM} H={NUM_HEADS} "
                     f"FA@L{FULL_ATTN_LAYER}. Epoch {epoch}/{SCREEN_EPOCHS}."),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"{CKPT_BASE_NAME}_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF D512-L13 Selective RoPE')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) '
          f'C(J={len(GROUP_C)}) T(J={len(GROUP_T)})')
    print(f'  Bandwidth ratios: A/B/C={(len(GROUP_A)*64)/EMBEDDING_DIM:.1f}x  '
          f'T={(len(GROUP_T)*64)/EMBEDDING_DIM:.1f}x')
    print(f'  Selective RoPE: J_LARGE offsets get position-based Q/K rotation')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print('  Using chunked CE (Liger disabled)')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    encoded_path = 'logs/fineweb_edu_encoded_2048_v2.pt'
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f'Dataset not found: {encoded_path}')
    _cache = torch.load(encoded_path, weights_only=True)
    train_data = _cache['train'].long()
    val_data = _cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs')

    model = TriadicJ96(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
    ).to(device)

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    scale_embed_params = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)

    def _lr_lambda(step, group_idx):
        if group_idx == 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda s: _lr_lambda(s, 0), lambda s: _lr_lambda(s, 1)])

    freeze_se = os.getenv('DWARF_FREEZE_SE', '0') == '1'
    if freeze_se:
        for p in scale_embed_params:
            p.requires_grad_(False)
        optimizer.param_groups[1]['lr'] = 0.0
        print('  [FREEZE] scale_embed frozen (DWARF_FREEZE_SE=1)')

    resume_path = os.getenv('DWARF_RESUME', '')
    start_epoch = int(os.getenv('DWARF_START_EPOCH', '1'))
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f'  Resumed from {resume_path} (starting epoch {start_epoch})')

    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}

    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)

    for epoch in range(start_epoch, SCREEN_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        step_times = deque(maxlen=20)

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)

                with _amp_context(device):
                    logits = model(x)
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                T = logits_flat.size(0)
                grad_logits = torch.empty_like(logits_flat)
                total_loss = 0.0
                for chunk_start in range(0, T, CE_CHUNK):
                    chunk_end = min(chunk_start + CE_CHUNK, T)
                    chunk = logits_flat[chunk_start:chunk_end].detach().requires_grad_(True)
                    chunk_loss = F.cross_entropy(
                        chunk, y_flat[chunk_start:chunk_end], reduction='sum')
                    chunk_loss.backward()
                    grad_logits[chunk_start:chunk_end] = chunk.grad
                    total_loss += chunk_loss.item()
                logits_flat.backward(grad_logits / (T * GRAD_ACCUM))
                loss_val = total_loss / T
                del logits, logits_flat, y_flat, grad_logits

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            step_ms = (time.time() - t0) * 1000
            step_times.append(step_ms)

            if step % 100 == 0:
                se_vals = [m.scale_embed.detach().abs()
                           for m in model.modules() if isinstance(m, DSQGAttentionGrouped)]
                se_max = torch.cat(se_vals).max().item() if se_vals else 0.0
                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000.0)
                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} | SE|max|={se_max:.4f} '
                      f'| {tok_s:.0f} tok/s', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt'))
            marker = ' *'

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules() if isinstance(m, DSQGAttentionGrouped)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

        resume_state = {
            'model_state_dict': {k.replace('._orig_mod', ''): v
                                 for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(resume_state,
                   os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_ep{epoch}_resume.pt'))

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s = time.time() - t_start
    memory_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final = ppl_results.get(SCREEN_EPOCHS, 999.0)
    PPL_BASELINE = 35.04
    PASSKEY_BASELINE = 99.2
    ar_score = (passkey_final - PASSKEY_BASELINE) + (PPL_BASELINE - ppl_final) * 0.5

    print('\n' + '=' * 70)
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score: {ar_score:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_params / 1e6:.1f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init: {EMA_INIT}')
    print(f'description: d512_l13_selective_rope L=13 D={EMBEDDING_DIM} H={NUM_HEADS} '
          f'FFN={FFN_DIM} FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1} '
          f'groups=A({len(GROUP_A)})+B({len(GROUP_B)})+C({len(GROUP_C)})+T({len(GROUP_T)})')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
