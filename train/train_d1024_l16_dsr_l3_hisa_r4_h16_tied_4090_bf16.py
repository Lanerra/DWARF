"""
DWARF D=1024 L=16 Triadic J=96 — DSR/HISA@L3 + R_PLANES=4, TIED LM_HEAD

Architecture: D=1024, H=16 (hd=64), L=16, FFN=2048, TIED lm_head
  Triadic partitioning: 96 offsets split into 3 pure groups of 32 (no tail)
  DSR at L3 (replacing the C-slot after the first triad):
    HierarchicalSparseAttentionV15HISA(C=32, top_k=4, HISA_m=32)
All DSQG blocks use V19 R_PLANES=4 Triton kernel (sequential Givens, grouped sparse, SE gates).

  L00: DSQGBlock(GROUP_A)  -- triad 1
  L01: DSQGBlock(GROUP_B)
  L02: DSQGBlock(GROUP_C)  + preIF
  L03: DSRBlock            -- HierarchicalSparseAttentionV15HISA (C=32, top_k=4, HISA_m=32)
  L04: DSQGBlock(GROUP_A)  -- triad 2
  L05: DSQGBlock(GROUP_B)
  L06: DSQGBlock(GROUP_C)
  L07: DSQGBlock(GROUP_A)  -- triad 3
  L08: DSQGBlock(GROUP_B)
  L09: DSQGBlock(GROUP_C)
  L10: DSQGBlock(GROUP_A)  -- triad 4
  L11: DSQGBlock(GROUP_B)
  L12: DSQGBlock(GROUP_C)
  L13: DSQGBlock(GROUP_A)  -- triad 5
  L14: DSQGBlock(GROUP_B)
  L15: DSQGBlock(GROUP_C)

  pre-DSR: 1 triad (L0-2), post-DSR: 4 triads (L4-15).
  Full J=96 topological coverage within each 3-layer triad.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d1024_l16_dsr_l3_hisa_r4_h16_tied_4090_bf16.py
"""

import contextlib, math, os, subprocess, sys, time
from collections import deque
from functools import partial
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
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "0") != "0"

try:
    from liger_kernel.transformers import LigerLayerNorm
    _LayerNorm = LigerLayerNorm
    _LIGER_LN = True
except ImportError:
    _LayerNorm = torch.nn.LayerNorm
    _LIGER_LN = False

# Selective Activation Checkpointing (SAC) — requires PyTorch 2.4+
try:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    _SAC_AVAILABLE = True
    _sac_intensive_ops = frozenset([
        torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm,
    ])
    def _sac_policy_fn(ctx, op, *args, **kwargs):
        return (CheckpointPolicy.MUST_SAVE if op in _sac_intensive_ops
                else CheckpointPolicy.PREFER_RECOMPUTE)
except ImportError:
    _SAC_AVAILABLE = False

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v19_bf16_se import (
    DSQGAttentionV19,
    dsqg_attention_v18_grouped,
    npci_rotate, R_PLANES, _next_pow2,
)
from dsqg_attention_v8_dynamic_j_r4 import ALL_OFFSETS as _ALL_96
assert R_PLANES == 4, f"Expected R_PLANES=4, got {R_PLANES}"
_DSQG_TYPES = (DSQGAttentionV19,)
print('  Kernel: V19 (sequential Givens, grouped sparse, SE gates)')
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

# =============================================================================
# DSR IMPORT
# =============================================================================

from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

# =============================================================================
# OFFSET GROUPS
# =============================================================================

GROUP_A = _ALL_96[0:32]
GROUP_B = _ALL_96[32:64]
GROUP_C = _ALL_96[64:96]

def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets), (
        f"J_SMALL({j_small}) + J_LARGE({j_large}) != J({len(offsets)})")
    return j_small, j_large

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)  # 17, 15
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)  # 0, 32
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)  # 0, 32

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 1024
NUM_HEADS        = 16
FFN_DIM          = 2048
NUM_LAYERS       = 16
DSR_LAYER        = 3
VOCAB_SIZE       = 32768

NUM_CHUNKS       = 32
TOP_K_CHUNKS     = 4
HISA_TOP_M_TOKENS = 32

SCALE_EMBED_INIT_VAL = 0.15
# Requested override: LR_MULT=24 for scale_embed
SCALE_EMBED_LR_MULT  = 24.0
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
LR        = 3e-4
DROPOUT   = 0.1

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '8'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '16'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '500000'))
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582
CE_CHUNK       = 1024
SCREEN_EPOCHS  = 3
TRAIN_LOG_INTERVAL = int(os.environ.get('DWARF_LOG_INTERVAL', '200'))
MAX_ACC_STEPS = int(os.environ.get('DWARF_MAX_ACC_STEPS', '0'))
BENCH_ONLY = os.getenv('DWARF_BENCH_ONLY', '0') == '1'
_compile_env = os.getenv('DWARF_TORCH_COMPILE')
if _compile_env is None:
    TORCH_COMPILE_ENABLED = os.getenv('DWARF_COMPILE', '1') != '0'
else:
    TORCH_COMPILE_ENABLED = _compile_env != '0'
TORCH_COMPILE_MODE = os.getenv('DWARF_TORCH_COMPILE_MODE', 'default')
COMPILE_CAPTURE_SCALARS = os.getenv('DWARF_COMPILE_CAPTURE_SCALARS', '1') == '1'
COMPILE_CAPTURE_DYNAMIC = os.getenv('DWARF_COMPILE_CAPTURE_DYNAMIC', '1') == '1'
TORCH_COMPILE_DYNAMIC = os.getenv('DWARF_TORCH_COMPILE_DYNAMIC', '1') == '1'
TORCH_COMPILE_FULLGRAPH = os.getenv('DWARF_TORCH_COMPILE_FULLGRAPH', '0') == '1'
COMPILE_SUPPRESS_ERRORS = os.getenv('DWARF_COMPILE_SUPPRESS_ERRORS', '1') == '1'
_compile_budget_env = os.getenv('DWARF_COMPILE_ACTIVATION_BUDGET', '0.3').strip().lower()
COMPILE_ACTIVATION_BUDGET = None if _compile_budget_env in ('', 'none') else float(_compile_budget_env)

TOKENIZER_CANDIDATES = ['results/mixed_tokenizer_32k.json']
PASSKEY_DISTANCES    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS       = int(os.environ.get('DWARF_PASSKEY_TRIALS', '20'))
PASSKEY_BATCH_SIZE   = 8
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'
CKPT_BASE_NAME    = 'd1024_l16_dsr_l3_hisa_r4_h16_tied'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# LAYER LAYOUT: L=16, DSR@L3
# Pre-DSR: 1 triad (L0-2), Post-DSR: 4 triads (L4-15)
# =============================================================================

LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00: triad 1
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L01
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L02 + preIF
    ('DSR', None, 0, 0, False),                     # L03: HierarchicalSparseAttentionV15HISA
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L04: triad 2
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L05
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L06
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07: triad 3
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L08
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L09
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L10: triad 4
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L11
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L12
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L13: triad 5
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L14
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L15
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
        self.norm1 = _LayerNorm(embedding_dim)
        self.norm2 = _LayerNorm(embedding_dim)
        self.attn = DSQGAttentionV19(
            embedding_dim, num_heads, offsets, j_small, j_large,
            seq_len=seq_len, dropout=dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = _LayerNorm(embedding_dim)
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


class DSRBlock(nn.Module):
    """DSR block: HierarchicalSparseAttentionV15HISA + FFN + SDPA output gate."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, head_dim,
                 num_chunks, top_k_chunks, dropout=0.1, hisa_top_m_tokens=32):
        super().__init__()
        self.norm1 = _LayerNorm(embedding_dim)
        self.norm2 = _LayerNorm(embedding_dim)
        self.attn = HierarchicalSparseAttentionV15HISA(
            D=embedding_dim, H=num_heads, hd=head_dim,
            num_chunks=num_chunks, top_k_chunks=top_k_chunks,
            hisa_top_m_tokens=hisa_top_m_tokens,
        )
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        gate = torch.sigmoid(self.gate_proj(x))
        x = x + self.drop(attn_out * gate)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MODEL
# =============================================================================

class TriadicJ96Dsr(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_dim, seq_len,
                 dsr_layer, scale_embed_init_val=0.15, dropout=0.1,
                 num_chunks=32, top_k_chunks=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.dsr_layer = dsr_layer
        self.head_dim = embedding_dim // num_heads

        blocks = []
        for i, (label, offsets, js, jl, has_if) in enumerate(LAYER_LAYOUT):
            if label == 'DSR':
                blocks.append(DSRBlock(
                    embedding_dim, num_heads, ffn_dim,
                    self.head_dim, num_chunks, top_k_chunks, dropout,
                    hisa_top_m_tokens=HISA_TOP_M_TOKENS))
            else:
                blocks.append(DSQGBlockTriadic(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    offsets, js, jl, group_label=label,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm = _LayerNorm(embedding_dim)
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
        global _DSQG_TYPES
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                nn.init.normal_(m.phase_base, 0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes, 0.0, 0.01)
                nn.init.normal_(m.phase_gain, 0.0, 0.001)
                nn.init.zeros_(m.phase_gate)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx):
        if CHECKPOINT_STRATEGY == 'none':
            return False
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.dsr_layer
        return False

    def _ckpt(self, block, x):
        if _SAC_AVAILABLE:
            return grad_ckpt(block, x, use_reentrant=False,
                             context_fn=partial(create_selective_checkpoint_contexts,
                                                _sac_policy_fn))
        return grad_ckpt(block, x, use_reentrant=False)

    def forward(self, idx):
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = self._ckpt(block, x)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def forward_hidden(self, idx):
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = self._ckpt(block, x)
            else:
                x = block(x)
        return self.norm(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        exclude_ids = {id(p) for p in self.scale_embed_parameters()}
        exclude_ids.update(id(p) for p in self.phase_parameters())
        for p in self.parameters():
            if id(p) not in exclude_ids:
                yield p

    def phase_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.phase_gain
                yield m.phase_gate
                yield m.query_probes
                yield m.key_probes

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}[{block.group_label}]: alpha={alpha:.4f}(w~{win}t)')
        return '  '.join(entries)

    def layer_summary(self):
        parts = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic):
                label = block.group_label
                j = block.attn.j_val
                iflag = '+IF' if block.interference else ''
                parts.append(f'L{i}:DSQG-{label}(J={j}){iflag}')
            elif isinstance(block, DSRBlock):
                parts.append(f'L{i}:DSR(C={block.attn.num_chunks},k={block.attn.top_k_chunks},HISA_m={block.attn.hisa_top_m_tokens})')
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
    print('  DWARF D1024-L16 Triadic J=96 — DSR/HISA@L3 + R_PLANES=4, TIED LM_HEAD')
    print(f'  DSR@L{DSR_LAYER}: HierarchicalSparseAttentionV15HISA(C={NUM_CHUNKS}, top_k={TOP_K_CHUNKS}, HISA_m={HISA_TOP_M_TOKENS})')
    print('  PURE LM loss only. No distillation. No teacher. Random init.')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) C(J={len(GROUP_C)})')
    print(f'  Per-layer bandwidth ratio: {(len(GROUP_A)*64)/EMBEDDING_DIM:.2f}x  (safe <= 3.0x)')
    print(f'  R_PLANES={R_PLANES}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA alpha0={EMA_INIT} (window~{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} x GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print('  DSQG: V19 (sequential Givens, grouped sparse, SE gates)')
    print('  DSR:  V15HISA (hierarchical sparse attention + token refinement)')
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Using chunked CE')
    print(f'  LayerNorm: {"LigerLayerNorm (fused)" if _LIGER_LN else "nn.LayerNorm"}')
    print(f'  SAC: {"enabled (PyTorch 2.4+)" if _SAC_AVAILABLE else "unavailable (requires PyTorch 2.4+)"}')
    print(f'  PASSKEY_TRIALS={PASSKEY_TRIALS}  CE_CHUNK={CE_CHUNK}  LOG_INTERVAL={TRAIN_LOG_INTERVAL}')
    if MAX_ACC_STEPS:
        print(f'  MAX_ACC_STEPS={MAX_ACC_STEPS}  BENCH_ONLY={BENCH_ONLY}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    encoded_path = 'logs/mixed_encoded_2048_v6_modular.pt'
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f'Dataset not found: {encoded_path}')
    _cache = torch.load(encoded_path, weights_only=True)
    train_data = _cache['train'].long()
    val_data = _cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    if device == 'cuda':
        train_data = train_data.pin_memory()
        val_data = val_data.pin_memory()
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs')

    model = TriadicJ96Dsr(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        dsr_layer=DSR_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
        num_chunks=NUM_CHUNKS,
        top_k_chunks=TOP_K_CHUNKS,
    ).to(device)

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')

    if TORCH_COMPILE_ENABLED:
        if COMPILE_CAPTURE_SCALARS:
            torch._dynamo.config.capture_scalar_outputs = True
        if COMPILE_CAPTURE_DYNAMIC:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.suppress_errors = COMPILE_SUPPRESS_ERRORS
        if COMPILE_ACTIVATION_BUDGET is not None and hasattr(torch._dynamo.config, 'activation_memory_budget'):
            torch._dynamo.config.activation_memory_budget = COMPILE_ACTIVATION_BUDGET
        compile_wrap_t0 = time.time()
        model = torch.compile(
            model,
            mode=TORCH_COMPILE_MODE,
            dynamic=TORCH_COMPILE_DYNAMIC,
            fullgraph=TORCH_COMPILE_FULLGRAPH)
        compile_wrap_ms = (time.time() - compile_wrap_t0) * 1000.0
        print(
            f'  torch.compile=ON mode={TORCH_COMPILE_MODE} '
            f'dynamic={TORCH_COMPILE_DYNAMIC} fullgraph={TORCH_COMPILE_FULLGRAPH} '
            f'capture_scalars={COMPILE_CAPTURE_SCALARS} '
            f'capture_dynamic={COMPILE_CAPTURE_DYNAMIC} '
            f'suppress_errors={COMPILE_SUPPRESS_ERRORS} '
            f'budget={COMPILE_ACTIVATION_BUDGET} wrap_ms={compile_wrap_ms:.1f}')
    else:
        print('  torch.compile=OFF')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_ref = _unwrap_compiled_module(model)

    scale_embed_params = list(model_ref.scale_embed_parameters())
    phase_params = list(model_ref.phase_parameters())
    other_params = list(model_ref.non_scale_embed_parameters())
    _opt_cls = torch.optim.AdamW
    if _BNB_AVAILABLE:
        _opt_cls = getattr(bnb.optim, 'PagedAdamW8bit', bnb.optim.AdamW8bit)
    optimizer = _opt_cls([
        {'params': other_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
        {'params': phase_params, 'lr': LR * 50, 'name': 'phase'},
    ], weight_decay=0.1, betas=(0.9, 0.95))
    print(f'  Optimizer: {_opt_cls.__name__}')
    print(f'  phase params LR: {LR * 50:.2e} (50x base)')

    steps_per_epoch_nominal = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    if MAX_ACC_STEPS:
        steps_per_epoch_nominal = min(steps_per_epoch_nominal, MAX_ACC_STEPS)
    total_steps = SCREEN_EPOCHS * max(steps_per_epoch_nominal, 1)

    def _lr_lambda(step, group_idx):
        if group_idx == 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda s: _lr_lambda(s, 0), lambda s: _lr_lambda(s, 1),
                   lambda s: _lr_lambda(s, 2)])

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
            skip_opt = os.getenv('DWARF_SKIP_OPT', '0') == '1'
            if not skip_opt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except (ValueError, RuntimeError) as _oe:
                    print(f'  [resume] optimizer state mismatch ({_oe}); starting fresh optimizer')
            else:
                print('  [resume] skipping optimizer state (DWARF_SKIP_OPT=1)')
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f'  Resumed from {resume_path} (starting epoch {start_epoch})')

    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}

    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)

    model_ref = _unwrap_compiled_module(model)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)

    for epoch in range(start_epoch, SCREEN_EPOCHS + 1):
        if device == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        indices = torch.randperm(len(train_data))
        step = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        if MAX_ACC_STEPS:
            steps_per_epoch = min(steps_per_epoch, MAX_ACC_STEPS)
        step_times = deque(maxlen=20)
        first_step_ms = None

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            loss_val = 0.0
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                    # LigerFusedLinearCrossEntropyLoss.forward(lin_weight, _input, target)
                    # lin_weight first, hidden states second
                    loss = liger_ce_fn(
                        model.out.weight,
                        hidden.contiguous().reshape(-1, hidden.size(-1)),
                        y.view(-1))
                    # Liger requires grad_output==1.0; scale grads manually after loop
                    loss.backward()
                    loss_val = loss.item()
                    del hidden, loss
                else:
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

            if USE_LIGER_CE:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(GRAD_ACCUM)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            step_ms = (time.time() - t0) * 1000
            if first_step_ms is None:
                first_step_ms = step_ms
            step_times.append(step_ms)

            should_log = ((acc_step + 1) % TRAIN_LOG_INTERVAL == 0) or ((acc_step + 1) == steps_per_epoch)
            if should_log:
                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000.0)
                se_max = max(
                    (p.abs().max().item() for p in model_ref.scale_embed_parameters()),
                    default=0.0)
                total_norm = math.sqrt(sum(
                    p.grad.square().sum().item()
                    for p in model.parameters() if p.grad is not None))
                lr_now = scheduler.get_last_lr()[0]
                routing_entropy = getattr(
                    model_ref.blocks[DSR_LAYER].attn,
                    '_routing_entropy', None)
                entropy_str = ''
                if routing_entropy is not None:
                    entropy_str = f' routing_ent={routing_entropy:.3f}'
                print(f'  [ep{epoch} step {acc_step+1}/{steps_per_epoch}] '
                      f'ce={loss_val:.4f} se_max={se_max:.3f} '
                      f'grad_norm={total_norm:.4f} lr={lr_now:.2e} '
                      f'{tok_s:.0f} tok/s{entropy_str}', flush=True)

        if BENCH_ONLY:
            avg_ms = sum(step_times) / len(step_times)
            tok_s = tokens_per_step / (avg_ms / 1000.0)
            compile_overhead_ms = max(first_step_ms - avg_ms, 0.0) if first_step_ms is not None else 0.0
            memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            print(f'\n[BENCH] first_step_ms={first_step_ms:.1f} trailing_avg_ms={avg_ms:.1f} '
                  f'steady_tok_s={tok_s:.0f} approx_compile_overhead_ms={compile_overhead_ms:.1f}')
            print(f'[BENCH] peak_vram={memory_mb:.0f}MB compile={TORCH_COMPILE_ENABLED} '
                  f'mode={TORCH_COMPILE_MODE if TORCH_COMPILE_ENABLED else "eager"} '
                  f'window={len(step_times)} steps={step}')
            return

        val_loss = evaluate(model, val_data, device)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        _is_best = val_loss < best_val_loss
        if _is_best:
            best_val_loss = val_loss
            clean_state = model_ref.state_dict()
            torch.save({
                'model_state_dict': clean_state,
                'config': {
                    'embedding_dim': EMBEDDING_DIM, 'num_heads': NUM_HEADS,
                    'ffn_dim': FFN_DIM, 'seq_len': MAX_SEQ_LEN,
                    'dsr_layer': DSR_LAYER, 'num_layers': NUM_LAYERS,
                    'num_chunks': NUM_CHUNKS, 'top_k_chunks': TOP_K_CHUNKS,
                    'hisa_top_m_tokens': HISA_TOP_M_TOKENS,
                    'r_planes': R_PLANES,
                    'tied_lm_head': True,
                },
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt'),
               pickle_protocol=5)
            marker = ' *'

        _save_full = (_is_best or epoch % 3 == 0 or epoch == SCREEN_EPOCHS)
        _ep_state = model_ref.state_dict()
        _ep_ckpt = {'model_state_dict': _ep_state, 'epoch': epoch,
                    'global_step': step}
        if _save_full:
            _ep_ckpt['optimizer_state_dict'] = optimizer.state_dict()
            _ep_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(_ep_ckpt,
                   os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_ep{epoch}.pt'),
                   pickle_protocol=5)

        se_vals = [m.scale_embed.detach().abs()
                   for m in model_ref.modules() if isinstance(m, _DSQG_TYPES)]
        if se_vals:
            se_all = torch.cat(se_vals)
            se_mean = se_all.mean().item()
            se_max = se_all.max().item()
            total_se = se_all.numel()
            print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')
            print(f'  scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}')

            for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                count = (se_all >= threshold).sum().item()
                pct = count / total_se * 100
                print(f'  SE>={threshold}: {pct:.1f}% ({count}/{total_se})')

            if se_max >= 1.8:
                if se_max >= 2.0:
                    print(f'  PERCOLATION ZONE: |scale_embed|max={se_max:.4f} >= 1.8 (near threshold ~2.0)')
                    print(f'  PHASE TRANSITION: |scale_embed|max={se_max:.4f} CROSSED 2.0!')
                else:
                    print(f'  PERCOLATION ZONE: |scale_embed|max={se_max:.4f} >= 1.8 (near threshold ~2.0)')

        for i, block in enumerate(model_ref.blocks):
            if isinstance(block, DSQGBlockTriadic):
                attn = _unwrap_compiled_module(block.attn)
                phase_base = attn.phase_base.detach().abs()
                phase_gain = attn.phase_gain.detach().abs()

                for plane in range(R_PLANES):
                    pb_plane = phase_base[:, :, plane]
                    pg_plane = phase_gain[:, :, plane]
                    combined = torch.cat([pb_plane.flatten(), pg_plane.flatten()])
                    print(f'  MOVT[L{i}] plane={plane}: |mean|={combined.mean():.4f} |max|={combined.max():.4f} |min|={combined.min():.4f}')

                head_vals = torch.cat([torch.cat([phase_base[:, :, p], phase_gain[:, :, p]]) for p in range(R_PLANES)])
                head_means = head_vals.mean(dim=0)
                print(f'  MOVT[L{i}] head means: min={head_means.min():.4f} max={head_means.max():.4f}')

                all_phase = torch.cat([phase_base.flatten(), phase_gain.flatten()])
                print(f'  MOVT[L{i}]: |mean|={all_phase.mean():.4f} |max|={all_phase.max():.4f}')

        routing_entropy = getattr(
            model_ref.blocks[DSR_LAYER].attn, '_routing_entropy', None)
        if routing_entropy is not None:
            print(f'  DSR routing entropy: {routing_entropy:.4f} '
                  f'(max={math.log(NUM_CHUNKS):.2f}, min=0.00)')

        print(f'  Physics: {model_ref.physics_summary()}')

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)

    print('\n' + '=' * 70)
    print(f'  DSR + R_PLANES={R_PLANES} Summary (D{EMBEDDING_DIM}-L{NUM_LAYERS}, tied lm_head)')
    print('=' * 70)
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'  ep{ep}: ppl={ppl_results.get(ep, 999.0):.2f}  '
              f'passkey={passkey_results.get(ep, 0.0):.1f}%')
    print(f'  peak_vram={memory_mb:.0f}MB  elapsed={elapsed_s:.0f}s')
    print(f'  params={n_params / 1e6:.1f}M  R_PLANES={R_PLANES}')
    print(f'  num_chunks={NUM_CHUNKS}  top_k_chunks={TOP_K_CHUNKS}  HISA_m={HISA_TOP_M_TOKENS}')

    if passkey_final >= 80:
        print('\n  CONTENT-ADDRESSED ROUTING ACHIEVED — passkey >= 80%')
    elif passkey_final >= 60:
        print('\n  PARTIAL — routing emerging but not fully content-addressed')
    else:
        print('\n  BELOW THRESHOLD — FA signal may be required for routing bootstrap')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
