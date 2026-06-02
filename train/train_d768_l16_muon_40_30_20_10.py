"""
DWARF D=768 L=16 Triadic J=96 — HISA/DSQG hybrid v2 L3, Bible-Muon scratch 40/30/20/10 variant.

Scratch target: train the scaled Bible-Muon D768 architecture from random init on the
mixed-tokenizer 40% Dolma3 Mix-6T / 30% Cosmopedia / 20% FineWeb-Edu / 10% Longmino cache.
This is the ~120M-class capability step from the D512/L10 40/30/20/10 control.

Architecture: D=768, H=12 (hd=64), L=16, FFN=2048, tied lm_head
  Triadic partitioning: 96 offsets split into 3 pure groups of 32
  HISA at L3 (after one pre-DSR triad, followed by four post-DSR cleanup triads)
  HISA block: HierarchicalSparseAttentionV15HISA(C=32, top_k=4, HISA_m=64 default)
  All DSQG blocks use the V20-compatible R_PLANES=4 Triton kernel with scale_embed + sequential MOVT.

Layout:
  L00: DSQGBlock(GROUP_A)
  L01: DSQGBlock(GROUP_B)
  L02: DSQGBlock(GROUP_C) + preIF
  L03: DSRBlock / HISA
  L04: DSQGBlock(GROUP_A)
  L05: DSQGBlock(GROUP_B)
  L06: DSQGBlock(GROUP_C)
  L07: DSQGBlock(GROUP_A)
  L08: DSQGBlock(GROUP_B)
  L09: DSQGBlock(GROUP_C)
  L10: DSQGBlock(GROUP_A)
  L11: DSQGBlock(GROUP_B)
  L12: DSQGBlock(GROUP_C)
  L13: DSQGBlock(GROUP_A)
  L14: DSQGBlock(GROUP_B)
  L15: DSQGBlock(GROUP_C)
"""

import contextlib, hashlib, json, math, os, random, subprocess, sys, time, types
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
# Triton 3.5+ compatibility for module-scope constants referenced by JIT kernels.
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"

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
_script_dir = str(_pl.Path(__file__).resolve().parent)
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
_tools_dir = os.path.join(_project_root, 'tools')
for _d in [_script_dir, _kernel_dir, _tools_dir, _project_root]:
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)
# Hermes also has a top-level `tools` package. Prefer DWARF/tools when it
# exists, but allow a local passkey_eval.py when running this uploaded bundle.
if os.path.isdir(_tools_dir):
    _tools_pkg = types.ModuleType('tools')
    _tools_pkg.__path__ = [_tools_dir]
    _tools_pkg.__file__ = os.path.join(_tools_dir, '__init__.py')
    sys.modules['tools'] = _tools_pkg

try:
    from tools.passkey_eval import PasskeyConfig, format_passkey_results, passkey_prefix_consistency_audit
except Exception:
    from passkey_eval import PasskeyConfig, format_passkey_results, passkey_prefix_consistency_audit


from dsqg_attention_v20_bf16_se import (
    DSQGAttentionV19,
    dsqg_attention_v18_grouped,
    npci_rotate, R_PLANES, _next_pow2,
    NPCI_THETA_MAX, NPCI_THETA_INIT,
    ALL_OFFSETS,
)
assert R_PLANES == 4, f"Expected R_PLANES=4, got {R_PLANES}"
_DSQG_TYPES = (DSQGAttentionV19,)
print('  Kernel: V20-compatible DSQG (R=4 sequential Givens, grouped sparse, SE gates)')
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

# =============================================================================
# DSR IMPORT
# =============================================================================

try:
    from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA
except Exception:
    from hierarchical_sparse_attn_v15_hisa_triton import HierarchicalSparseAttentionV15HISA

# =============================================================================
# OFFSET GROUPS
# =============================================================================

def _canonicalize_all_offsets(offsets):
    vals = [int(d) for d in offsets]
    if len(vals) != 96:
        raise ValueError(f'Expected 96 offsets, got {len(vals)}')
    if len(set(vals)) != len(vals):
        raise ValueError('Duplicate offsets are not supported')
    middle = [d for d in vals if not (d <= 28 or d >= 48)]
    if middle:
        raise ValueError(f'Unsupported offsets in gap 29..47: {middle}')
    ordered = sorted(vals)
    if ordered != vals:
        print('  [offsets] ALL_OFFSETS was not sorted; using sorted canonical order')
    return ordered


def _canonicalize_offset_group(offsets):
    vals = [int(d) for d in offsets]
    small = sorted(d for d in vals if d <= 28)
    large = sorted(d for d in vals if d >= 48)
    if len(small) + len(large) != len(vals):
        raise ValueError(f'Offset group contains unsupported middle offsets: {vals}')
    return small + large


def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets), (
        f"J_SMALL({j_small}) + J_LARGE({j_large}) != J({len(offsets)})")
    if offsets[:j_small] != sorted(offsets[:j_small]) or any(d > 28 for d in offsets[:j_small]):
        raise ValueError(f'Offset group is not small-first sorted: {offsets}')
    if offsets[j_small:] != sorted(offsets[j_small:]) or any(d < 48 for d in offsets[j_small:]):
        raise ValueError(f'Offset group is not large-second sorted: {offsets}')
    return j_small, j_large

_ALL_96_ORDERED = _canonicalize_all_offsets(_ALL_96)
GROUP_A = _canonicalize_offset_group(_ALL_96_ORDERED[0:32])
GROUP_B = _canonicalize_offset_group(_ALL_96_ORDERED[32:64])
GROUP_C = _canonicalize_offset_group(_ALL_96_ORDERED[64:96])

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)  # 17, 15
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)  # 0, 32
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)  # 0, 32

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 768
NUM_HEADS        = 12
FFN_DIM          = 2048
NUM_LAYERS       = 16
DSR_LAYER        = 3
VOCAB_SIZE       = int(os.environ.get('DWARF_VOCAB_SIZE', '32000'))

NUM_CHUNKS       = 32
TOP_K_CHUNKS     = int(os.environ.get('DWARF_HISA_TOP_K', '4'))
HISA_TOP_M_TOKENS = int(os.environ.get('DWARF_HISA_TOP_M', '64'))

SCALE_EMBED_INIT_VAL = 0.15
# Conservative audited default; override with DWARF_SCALE_EMBED_LR_MULT.
SCALE_EMBED_LR_MULT  = float(os.environ.get('DWARF_SCALE_EMBED_LR_MULT', '8.0'))
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
LR        = float(os.environ.get('DWARF_LR', '3e-4'))
WEIGHT_DECAY = float(os.environ.get('DWARF_WEIGHT_DECAY', '0.1'))
PHASE_LR_MULT = float(os.environ.get('DWARF_PHASE_LR_MULT', '10.0'))
NPCI_THETA_LR_MULT = float(os.environ.get('DWARF_NPCI_THETA_LR_MULT', '8.0'))
GRAD_CLIP_NORM = float(os.environ.get('DWARF_GRAD_CLIP_NORM', '1.0'))
SKIP_NONFINITE_STEP = os.getenv('DWARF_SKIP_NONFINITE_STEP', '1') == '1'
SE_MAX_ABORT = float(os.environ.get('DWARF_SE_MAX_ABORT', '0.0'))
LR_WARMUP_STEPS = int(os.environ.get('DWARF_LR_WARMUP_STEPS', '500'))
MIN_LR_RATIO = float(os.environ.get('DWARF_MIN_LR_RATIO', '0.1'))
SCALE_EMBED_CONSTANT_LR = os.getenv('DWARF_SCALE_EMBED_CONSTANT_LR', '0') == '1'
DROPOUT   = 0.1

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '32'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '8'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '800000'))
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582
CE_CHUNK       = int(os.environ.get('DWARF_CE_ROWS', '2048'))  # rows per streamed final-projection CE chunk
PIN_DATASET    = os.getenv('DWARF_PIN_DATASET', '0') == '1'
SEED           = int(os.environ.get('DWARF_SEED', '42'))
REQUIRE_PREFIX_CLEAN = os.getenv('DWARF_REQUIRE_PREFIX_CLEAN', '0') == '1'
SCREEN_EPOCHS  = int(os.environ.get('DWARF_EPOCHS', '1'))

TRAIN_LOG_INTERVAL = int(os.environ.get('DWARF_LOG_INTERVAL', '100'))
MAX_ACC_STEPS = int(os.environ.get('DWARF_MAX_ACC_STEPS', '0'))
BENCH_ONLY = os.getenv('DWARF_BENCH_ONLY', '0') == '1'
_compile_env = os.getenv('DWARF_TORCH_COMPILE')
if _compile_env is None:
    # Custom Triton/autograd + checkpointing paths are benchmark axes, not a safe default.
    TORCH_COMPILE_ENABLED = os.getenv('DWARF_COMPILE', '0') != '0'
else:
    TORCH_COMPILE_ENABLED = _compile_env != '0'
TORCH_COMPILE_MODE = os.getenv('DWARF_TORCH_COMPILE_MODE', 'default')
COMPILE_CAPTURE_SCALARS = os.getenv('DWARF_COMPILE_CAPTURE_SCALARS', '1') == '1'
COMPILE_CAPTURE_DYNAMIC = os.getenv('DWARF_COMPILE_CAPTURE_DYNAMIC', '1') == '1'
TORCH_COMPILE_DYNAMIC = os.getenv('DWARF_TORCH_COMPILE_DYNAMIC', '1') == '1'
TORCH_COMPILE_FULLGRAPH = os.getenv('DWARF_TORCH_COMPILE_FULLGRAPH', '0') == '1'
COMPILE_SUPPRESS_ERRORS = os.getenv('DWARF_COMPILE_SUPPRESS_ERRORS', '0') == '1'
_compile_budget_env = os.getenv('DWARF_COMPILE_ACTIVATION_BUDGET', '0.3').strip().lower()
COMPILE_ACTIVATION_BUDGET = None if _compile_budget_env in ('', 'none') else float(_compile_budget_env)

TOKENIZER_CANDIDATES = [os.environ.get('DWARF_TOKENIZER', 'tokenizers/mixed_tokenizer_32k.json')]
PASSKEY_DISTANCES    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS       = int(os.environ.get('DWARF_PASSKEY_TRIALS', '20'))
PASSKEY_BATCH_SIZE   = 8
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = os.environ.get('DWARF_CHECKPOINT_DIR', 'autoresearch/checkpoints')
_CKPT_BASE_DEFAULT = 'd768_l16_muon_40_30_20_10_staggered_movt' if os.getenv('DWARF_STAGGER_MOVT_PLANES', '1') == '1' else 'd768_l16_muon_40_30_20_10'
CKPT_BASE_NAME    = os.environ.get('DWARF_CKPT_BASE_NAME', _CKPT_BASE_DEFAULT)

# A/B switch for optimizer experiments. AdamW keeps the original PagedAdamW8bit baseline path.
# Muon uses torch.optim.Muon on 2D hidden matrices only and AdamW on topology/embedding/norm/etc.
OPTIMIZER_KIND = os.getenv('DWARF_OPT', 'muon').strip().lower()
if OPTIMIZER_KIND not in {'adamw', 'muon'}:
    raise ValueError(f"Unsupported DWARF_OPT={OPTIMIZER_KIND!r}; expected 'adamw' or 'muon'")
MUON_ADJUST_LR_FN = os.getenv('DWARF_MUON_ADJUST_LR_FN', 'match_rms_adamw')
MUON_MOMENTUM = float(os.getenv('DWARF_MUON_MOMENTUM', '0.95'))
MUON_NS_STEPS = int(os.getenv('DWARF_MUON_NS_STEPS', '5'))

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'every_other').lower()
DATASET_PATH = os.environ.get('DWARF_DATASET', 'datasets/continuation_40mix6t_20fw_30cosmo_10longmino_mixedtok_2048_800k.pt')
STAGGER_MOVT_PLANES = os.getenv('DWARF_STAGGER_MOVT_PLANES', '1') == '1'


def _movt_plane_shift_for_dsqg_index(dsqg_index: int, head_dim: int, r_planes: int = R_PLANES) -> int:
    """Return the per-DSQG-layer MOVT plane shift; DSR/HISA layers do not consume slots."""
    segment = max(2, int(head_dim) // int(r_planes))
    slots = max(1, segment // 2)
    return 2 * (int(dsqg_index) % slots)

# =============================================================================
# LAYER LAYOUT: L=16, DSR/HISA@L3
# Pre-DSR: 1 triad (L0-2), Post-DSR: 4 full triads (L4-15)
# =============================================================================

LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L01
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L02
    ('DSR', None, 0, 0, False),                    # L03: HierarchicalSparseAttentionV15HISA
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L04
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L05
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L06
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L08
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L09
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L10
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L11
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L12
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L13
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
                 dropout=0.1, interference=False, plane_shift=0):
        super().__init__()
        self.interference = interference
        self.group_label = group_label
        self.plane_shift = int(plane_shift)
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.norm1 = _LayerNorm(embedding_dim)
        self.norm2 = _LayerNorm(embedding_dim)
        self.attn = DSQGAttentionV19(
            embedding_dim, num_heads, offsets, j_small, j_large,
            seq_len=seq_len, dropout=dropout, plane_shift=self.plane_shift)
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
    """DSR block: HierarchicalSparseAttentionV15HISA + FFN + output gate."""
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
        dsqg_idx = 0
        for i, (label, offsets, js, jl, has_if) in enumerate(LAYER_LAYOUT):
            if label == 'DSR':
                blocks.append(DSRBlock(
                    embedding_dim, num_heads, ffn_dim,
                    self.head_dim, num_chunks, top_k_chunks, dropout,
                    hisa_top_m_tokens=HISA_TOP_M_TOKENS))
            else:
                plane_shift = (_movt_plane_shift_for_dsqg_index(dsqg_idx, self.head_dim, R_PLANES)
                               if STAGGER_MOVT_PLANES else 0)
                blocks.append(DSQGBlockTriadic(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    offsets, js, jl, group_label=label,
                    dropout=dropout, interference=has_if,
                    plane_shift=plane_shift))
                dsqg_idx += 1
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
                if hasattr(m, 'reset_phase_probes_'):
                    m.reset_phase_probes_()
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
        exclude_ids.update(id(p) for p in self.npci_theta_parameters())
        for p in self.parameters():
            if id(p) not in exclude_ids:
                yield p

    def phase_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.phase_base
                yield m.phase_gain
                yield m.phase_gate
                yield m.query_probes
                yield m.key_probes

    def npci_theta_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.npci_theta_k
                yield m.npci_theta_v

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
                shift = getattr(block, 'plane_shift', 0)
                parts.append(f'L{i}:DSQG-{label}(J={j},shift={shift}){iflag}')
            elif isinstance(block, DSRBlock):
                parts.append(f'L{i}:DSR-V15HISA(C={block.attn.num_chunks},k={block.attn.top_k_chunks},HISA_m={block.attn.hisa_top_m_tokens})')
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


def _sha256_file(path, *, max_bytes=None):
    h = hashlib.sha256()
    remaining = max_bytes
    with open(path, 'rb') as f:
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _file_fingerprint(path, *, hash_file=False):
    if not path or not os.path.exists(path):
        return {'path': path, 'exists': False}
    st = os.stat(path)
    info = {
        'path': path,
        'exists': True,
        'size_bytes': st.st_size,
        'mtime_ns': st.st_mtime_ns,
    }
    if hash_file:
        info['sha256'] = _sha256_file(path)
    return info


def _git_status_short():
    try:
        return subprocess.check_output(
            ['git', 'status', '--short'], stderr=subprocess.DEVNULL
        ).decode().splitlines()
    except Exception:
        return []


def _env_snapshot():
    prefixes = ('DWARF_', 'HISA_', 'PYTORCH_', 'CUDA_VISIBLE_DEVICES')
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def _base_checkpoint_config(*, git_hash, tok_path, encoded_path, n_params):
    return {
        'script': os.path.relpath(__file__, _project_root),
        'script_fingerprint': _file_fingerprint(__file__, hash_file=True),
        'git_hash': git_hash,
        'git_status_short': _git_status_short(),
        'seed': SEED,
        'tokenizer': _file_fingerprint(tok_path, hash_file=True),
        'dataset': _file_fingerprint(encoded_path, hash_file=os.getenv('DWARF_HASH_DATASET', '0') == '1'),
        'env': _env_snapshot(),
        'model': {
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'head_dim': EMBEDDING_DIM // NUM_HEADS,
            'ffn_dim': FFN_DIM,
            'seq_len': MAX_SEQ_LEN,
            'dsr_layer': DSR_LAYER,
            'num_layers': NUM_LAYERS,
            'num_chunks': NUM_CHUNKS,
            'top_k_chunks': TOP_K_CHUNKS,
            'hisa_top_m_tokens': HISA_TOP_M_TOKENS,
            'r_planes': R_PLANES,
            'stagger_movt_planes': STAGGER_MOVT_PLANES,
            'movt_plane_shifts': [
                _movt_plane_shift_for_dsqg_index(i, EMBEDDING_DIM // NUM_HEADS, R_PLANES)
                for i in range(sum(1 for label, *_ in LAYER_LAYOUT if label != 'DSR'))
            ] if STAGGER_MOVT_PLANES else [],
            'tied_lm_head': True,
            'params': n_params,
            'layer_layout': [(label, len(offsets) if offsets is not None else 0, has_if)
                             for label, offsets, _, _, has_if in LAYER_LAYOUT],
        },
        'training': {
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'scale_embed_lr_mult': SCALE_EMBED_LR_MULT,
            'phase_lr_mult': PHASE_LR_MULT,
            'npci_theta_lr_mult': NPCI_THETA_LR_MULT,
            'grad_clip_norm': GRAD_CLIP_NORM,
            'skip_nonfinite_step': SKIP_NONFINITE_STEP,
            'se_max_abort': SE_MAX_ABORT,
            'npci_theta_max': NPCI_THETA_MAX,
            'npci_theta_init': NPCI_THETA_INIT,
            'lr_warmup_steps': LR_WARMUP_STEPS,
            'min_lr_ratio': MIN_LR_RATIO,
            'scale_embed_constant_lr': SCALE_EMBED_CONSTANT_LR,
            'batch_size': BATCH_SIZE,
            'grad_accum': GRAD_ACCUM,
            'max_train_seqs': MAX_TRAIN_SEQS,
            'screen_epochs': SCREEN_EPOCHS,
            'checkpoint_strategy': CHECKPOINT_STRATEGY,
            'torch_compile_enabled': TORCH_COMPILE_ENABLED,
            'torch_compile_mode': TORCH_COMPILE_MODE,
            'compile_suppress_errors': COMPILE_SUPPRESS_ERRORS,
            'use_liger_ce': USE_LIGER_CE,
            'ce_chunk': CE_CHUNK,
            'optimizer_kind': OPTIMIZER_KIND,
            'muon_adjust_lr_fn': MUON_ADJUST_LR_FN,
            'muon_momentum': MUON_MOMENTUM,
            'muon_ns_steps': MUON_NS_STEPS,
        },
        'eval': {
            'passkey_distances': list(PASSKEY_DISTANCES),
            'passkey_trials': PASSKEY_TRIALS,
            'passkey_batch_size': PASSKEY_BATCH_SIZE,
            'require_prefix_clean': REQUIRE_PREFIX_CLEAN,
            'passkey_words': list(_PASSKEY_WORDS),
            'retrieval_cue': _RETRIEVAL_CUE,
        },
    }


def _adamw_cls():
    if _BNB_AVAILABLE:
        return getattr(bnb.optim, 'PagedAdamW8bit', bnb.optim.AdamW8bit)
    return torch.optim.AdamW


def _make_optimizer_param_groups(model_ref):
    scale_embed_params = list(model_ref.scale_embed_parameters())
    phase_params = list(model_ref.phase_parameters())
    npci_theta_params = list(model_ref.npci_theta_parameters())
    special_ids = {id(p) for p in scale_embed_params}
    special_ids.update(id(p) for p in phase_params)
    special_ids.update(id(p) for p in npci_theta_params)

    decay_params, no_decay_params = [], []
    for name, p in model_ref.named_parameters():
        if not p.requires_grad or id(p) in special_ids:
            continue
        lname = name.lower()
        if p.ndim < 2 or name.endswith('.bias') or 'norm' in lname or name in ('embedding.weight', 'out.weight'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {'params': decay_params, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'decay'},
        {'params': no_decay_params, 'lr': LR, 'weight_decay': 0.0, 'name': 'no_decay'},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT, 'weight_decay': 0.0, 'name': 'scale_embed'},
        {'params': phase_params, 'lr': LR * PHASE_LR_MULT, 'weight_decay': 0.0, 'name': 'phase'},
        {'params': npci_theta_params, 'lr': LR * NPCI_THETA_LR_MULT, 'weight_decay': 0.0, 'name': 'npci_theta'},
    ]


def _is_muon_hidden_param(name, p, special_ids):
    if id(p) in special_ids or p.ndim != 2:
        return False
    lname = name.lower()
    if name in ('embedding.weight', 'out.weight'):
        return False
    if name.endswith('.bias') or 'norm' in lname:
        return False
    return True


def _make_hybrid_muon_param_groups(model_ref):
    scale_embed_params = list(model_ref.scale_embed_parameters())
    phase_params = list(model_ref.phase_parameters())
    npci_theta_params = list(model_ref.npci_theta_parameters())
    special_ids = {id(p) for p in scale_embed_params}
    special_ids.update(id(p) for p in phase_params)
    special_ids.update(id(p) for p in npci_theta_params)

    muon_hidden, adamw_decay, adamw_no_decay = [], [], []
    for name, p in model_ref.named_parameters():
        if not p.requires_grad or id(p) in special_ids:
            continue
        lname = name.lower()
        if _is_muon_hidden_param(name, p, special_ids):
            muon_hidden.append(p)
        elif p.ndim >= 2 and not name.endswith('.bias') and 'norm' not in lname and name not in ('embedding.weight', 'out.weight'):
            adamw_decay.append(p)
        else:
            adamw_no_decay.append(p)

    if not muon_hidden:
        raise RuntimeError('DWARF_OPT=muon selected but no 2D hidden parameters were assigned to Muon')

    return {
        'muon': [
            {'params': muon_hidden, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'muon_hidden'},
        ],
        'adamw': [
            {'params': adamw_decay, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'adamw_decay'},
            {'params': adamw_no_decay, 'lr': LR, 'weight_decay': 0.0, 'name': 'adamw_no_decay'},
            {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_scale_embed'},
            {'params': phase_params, 'lr': LR * PHASE_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_phase'},
            {'params': npci_theta_params, 'lr': LR * NPCI_THETA_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_npci_theta'},
        ],
        'scale_embed_params': scale_embed_params,
        'phase_params': phase_params,
    }


class _MultiOptimizer:
    """Small wrapper so the trainer can step Muon(hidden) + AdamW(rest) together."""

    def __init__(self, named_optimizers):
        self.named_optimizers = list(named_optimizers)
        self.param_groups = []
        for opt_name, opt in self.named_optimizers:
            for group in opt.param_groups:
                group.setdefault('optimizer', opt_name)
                self.param_groups.append(group)

    def zero_grad(self, set_to_none=True):
        for _, opt in self.named_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for _, opt in self.named_optimizers:
            opt.step()

    def state_dict(self):
        return {
            'kind': 'multi',
            'optimizers': [
                {'name': name, 'state_dict': opt.state_dict()}
                for name, opt in self.named_optimizers
            ],
        }

    def load_state_dict(self, state_dict):
        if state_dict.get('kind') != 'multi':
            raise ValueError('cannot load non-multi optimizer state into hybrid Muon optimizer')
        saved = {entry['name']: entry['state_dict'] for entry in state_dict['optimizers']}
        for name, opt in self.named_optimizers:
            if name not in saved:
                raise ValueError(f'missing optimizer state for {name}')
            opt.load_state_dict(saved[name])


class _LambdaLRScheduler:
    """LambdaLR-compatible scheduler for both real optimizers and _MultiOptimizer."""

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambdas = list(lr_lambda)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
        self._last_lr = list(self.base_lrs)
        self._apply()

    def _apply(self):
        self._last_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            factor = self.lr_lambdas[i](self.last_epoch)
            lr = self.base_lrs[i] * factor
            group['lr'] = lr
            self._last_lr.append(lr)

    def step(self):
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            'base_lrs': list(self.base_lrs),
            'last_epoch': self.last_epoch,
            '_last_lr': list(self._last_lr),
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = list(state_dict['base_lrs'])
        self.last_epoch = int(state_dict.get('last_epoch', 0))
        self._apply()


def _build_optimizer(model_ref):
    if OPTIMIZER_KIND == 'adamw':
        optimizer_groups = _make_optimizer_param_groups(model_ref)
        opt_cls = _adamw_cls()
        optimizer = opt_cls(optimizer_groups, betas=(0.9, 0.95), eps=1e-8)
        return optimizer, optimizer_groups[2]['params'], optimizer_groups[3]['params'], opt_cls.__name__

    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('DWARF_OPT=muon requested, but torch.optim.Muon is not available')

    grouped = _make_hybrid_muon_param_groups(model_ref)
    muon = torch.optim.Muon(
        grouped['muon'],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MUON_MOMENTUM,
        nesterov=True,
        ns_steps=MUON_NS_STEPS,
        adjust_lr_fn=MUON_ADJUST_LR_FN,
    )
    adamw_cls = _adamw_cls()
    adamw = adamw_cls(grouped['adamw'], betas=(0.9, 0.95), eps=1e-8)
    optimizer = _MultiOptimizer([('muon', muon), ('adamw', adamw)])
    label = f"HybridMuon({MUON_ADJUST_LR_FN}, ns={MUON_NS_STEPS})+{adamw_cls.__name__}"
    return optimizer, grouped['scale_embed_params'], grouped['phase_params'], label


def _streamed_linear_ce_loss(hidden: torch.Tensor,
                             targets: torch.Tensor,
                             weight: torch.Tensor,
                             *,
                             chunk_rows: int,
                             grad_denom: float | None = None) -> tuple[torch.Tensor, int]:
    """Compute tied final-projection CE without materializing [B,T,V] logits.

    In training mode (grad_denom is not None), this streams CE chunks through a
    detached hidden leaf, accumulates the tied output-weight gradient chunk by
    chunk, stores only d_hidden [B*T,D], and then calls hidden.backward(d_hidden)
    once. That avoids both the full logits/grad_logits tensors and the severe
    checkpoint recompute overhead caused by calling backward through the full
    model once per CE chunk.
    """
    hidden_c = hidden.contiguous()
    h = hidden_c.view(-1, hidden_c.size(-1))
    y = targets.reshape(-1)
    n_rows = h.size(0)
    total_loss = torch.zeros((), device=h.device, dtype=torch.float32)
    chunk_rows = max(1, int(chunk_rows))
    grad_h = torch.empty_like(h) if grad_denom is not None else None

    for s in range(0, n_rows, chunk_rows):
        e = min(s + chunk_rows, n_rows)
        h_chunk = h[s:e]
        if grad_denom is not None:
            h_chunk = h_chunk.detach().requires_grad_(True)
        with _amp_context(h.device.type):
            logits = F.linear(h_chunk, weight)
        loss_sum = F.cross_entropy(logits.float(), y[s:e], reduction='sum')
        total_loss = total_loss + loss_sum.detach()
        if grad_denom is not None:
            (loss_sum / float(grad_denom)).backward()
            grad_h[s:e].copy_(h_chunk.grad)
        del logits, loss_sum, h_chunk

    if grad_h is not None:
        hidden_c.backward(grad_h.view_as(hidden_c))

    return total_loss, n_rows


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    model_ref = _unwrap_compiled_module(model)
    total_loss, total_tokens = 0.0, 0
    bs = max(1, BATCH_SIZE // 2)
    for i in range(0, len(data) - bs + 1, bs):
        x = data[i:i+bs, :-1].to(device, non_blocking=True)
        if x.dtype not in (torch.int32, torch.int64):
            x = x.long()
        y = data[i:i+bs, 1:].to(device, non_blocking=True).long()
        with _amp_context(device):
            hidden = model.forward_hidden(x)
            loss_sum, n_rows = _streamed_linear_ce_loss(
                hidden, y, model_ref.out.weight,
                chunk_rows=CE_CHUNK,
                grad_denom=None,
            )
        total_loss += float(loss_sum.item())
        total_tokens += n_rows
        del hidden, x, y
    return total_loss / max(total_tokens, 1)


def _passkey_config():
    return PasskeyConfig(
        max_seq_len=MAX_SEQ_LEN,
        distances=list(PASSKEY_DISTANCES),
        trials=PASSKEY_TRIALS,
        batch_size=PASSKEY_BATCH_SIZE,
        words=list(_PASSKEY_WORDS),
        filler_sentence=_FILLER_SENTENCE,
        intro_template=_INTRO_TEMPLATE,
        retrieval_cue=_RETRIEVAL_CUE,
        pad_id=0,
    )


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    audit = passkey_prefix_consistency_audit(model, tokenizer, device, _passkey_config())
    print(
        f"  [passkey audit] clean={audit['prefix_consistent']} "
        f"max_pad_delta={audit['max_pad_logit_delta']:.3e} "
        f"max_suffix_delta={audit['max_suffix_logit_delta']:.3e}",
        flush=True,
    )
    if not audit['prefix_consistent']:
        print(
            '  [passkey audit] WARNING: prefix-only score is reported; legacy padded passkey is contaminated.',
            flush=True,
        )
        if REQUIRE_PREFIX_CLEAN:
            raise RuntimeError(
                'Passkey prefix-consistency gate failed: '
                f"max_pad_delta={audit['max_pad_logit_delta']:.3e}, "
                f"max_suffix_delta={audit['max_suffix_logit_delta']:.3e}"
            )
    return audit['prefix_accuracy']


# =============================================================================
# TRAINING
# =============================================================================

def train():
    if not torch.cuda.is_available():
        raise RuntimeError('DWARF DSQG/HISA kernels require CUDA + Triton; CPU execution is not supported.')
    device = 'cuda'
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    print('=' * 70)
    print(f'  DWARF D768-L16 Triadic J=96 — V15 HISA@L{DSR_LAYER} + R_PLANES=4, TIED LM_HEAD')
    print(f'  DSR@L{DSR_LAYER}: HierarchicalSparseAttentionV15HISA(C={NUM_CHUNKS}, top_k={TOP_K_CHUNKS}, HISA_m={HISA_TOP_M_TOKENS})')
    print('  SCRATCH PRETRAINING: 40% Dolma3 Mix-6T / 30% Cosmopedia / 20% FineWeb-Edu / 10% Longmino. Bible-Muon defaults.')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) C(J={len(GROUP_C)})')
    print(f'  Per-layer bandwidth ratio: {(len(GROUP_A)*64)/EMBEDDING_DIM:.2f}x  (safe <= 3.0x)')
    print(f'  R_PLANES={R_PLANES}')
    if STAGGER_MOVT_PLANES:
        _shift_schedule = [
            _movt_plane_shift_for_dsqg_index(i, EMBEDDING_DIM // NUM_HEADS, R_PLANES)
            for i in range(sum(1 for label, *_ in LAYER_LAYOUT if label != 'DSR'))
        ]
        print(f'  MOVT plane staggering: enabled, DSQG-index shifts={_shift_schedule}')
    else:
        print('  MOVT plane staggering: disabled, all DSQG shifts=0')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  phase LR mult={PHASE_LR_MULT}  grad_clip={GRAD_CLIP_NORM}  skip_nonfinite={SKIP_NONFINITE_STEP}  se_abort={SE_MAX_ABORT}')
    print(f'  EMA alpha0={EMA_INIT} (window~{round(1/EMA_INIT)}t)')
    print(f'  NPCI theta init={NPCI_THETA_INIT} max={NPCI_THETA_MAX} LR mult={NPCI_THETA_LR_MULT}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}, Seed={SEED}')
    print(f'  Batch: BS={BATCH_SIZE} x GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print('  DSQG: V20-compatible (R=4 sequential Givens, grouped sparse, SE gates)')
    print('  DSR:  V15HISA')
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Using streamed final-projection CE')
    print(f'  LayerNorm: {"LigerLayerNorm (fused)" if _LIGER_LN else "nn.LayerNorm"}')
    print(f'  SAC: {"enabled (PyTorch 2.4+)" if _SAC_AVAILABLE else "unavailable (requires PyTorch 2.4+)"}')
    print(f'  PASSKEY_TRIALS={PASSKEY_TRIALS}  CE_ROWS={CE_CHUNK}  LOG_INTERVAL={TRAIN_LOG_INTERVAL}  PIN_DATASET={PIN_DATASET}  REQUIRE_PREFIX_CLEAN={REQUIRE_PREFIX_CLEAN}')
    if MAX_ACC_STEPS:
        print(f'  MAX_ACC_STEPS={MAX_ACC_STEPS}  BENCH_ONLY={BENCH_ONLY}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    tok_vocab_size = tokenizer.vocab_size()
    if tok_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f'Tokenizer vocab_size={tok_vocab_size} does not match model VOCAB_SIZE={VOCAB_SIZE}. '
            'Set DWARF_VOCAB_SIZE to match DWARF_TOKENIZER/DWARF_DATASET.'
        )
    print(f'Loaded tokenizer from {tok_path} (vocab={tok_vocab_size:,})')

    encoded_path = DATASET_PATH
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f'Dataset not found: {encoded_path}')
    _cache = torch.load(encoded_path, weights_only=True)
    cache_vocab_size = _cache.get('vocab_size') if isinstance(_cache, dict) else None
    if cache_vocab_size is not None and int(cache_vocab_size) != VOCAB_SIZE:
        raise ValueError(
            f'Dataset vocab_size={cache_vocab_size} does not match model VOCAB_SIZE={VOCAB_SIZE}. '
            'Set DWARF_VOCAB_SIZE to match the encoded dataset.'
        )
    # Keep cached sequences compact on host; cast targets to int64 per batch for CE.
    train_data = _cache['train'].to(dtype=torch.int32).contiguous()
    val_data = _cache['val'].to(dtype=torch.int32).contiguous()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    if PIN_DATASET:
        train_data = train_data.pin_memory()
        val_data = val_data.pin_memory()
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs  host_dtype={train_data.dtype}')

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

    optimizer, scale_embed_params, phase_params, optimizer_label = _build_optimizer(model_ref)
    print(f'  Optimizer: {optimizer_label} (DWARF_OPT={OPTIMIZER_KIND})')
    for group in optimizer.param_groups:
        n_group = sum(p.numel() for p in group['params'])
        print(f"  opt_group[{group.get('name', '?')} via {group.get('optimizer', OPTIMIZER_KIND)}]: "
              f"params={n_group:,} lr={group['lr']:.2e} wd={group.get('weight_decay', 0.0):.2g}")

    steps_per_epoch_nominal = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    if MAX_ACC_STEPS:
        steps_per_epoch_nominal = min(steps_per_epoch_nominal, MAX_ACC_STEPS)
    total_steps = SCREEN_EPOCHS * max(steps_per_epoch_nominal, 1)

    warmup_steps = min(max(LR_WARMUP_STEPS, 0), max(total_steps - 1, 0))

    def _lr_lambda(step, group_idx):
        group_name = optimizer.param_groups[group_idx].get('name', '')
        if SCALE_EMBED_CONSTANT_LR and group_name.endswith('scale_embed'):
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    scheduler = _LambdaLRScheduler(
        optimizer,
        lr_lambda=[lambda s, gi=gi: _lr_lambda(s, gi) for gi in range(len(optimizer.param_groups))])

    freeze_se = os.getenv('DWARF_FREEZE_SE', '0') == '1'
    if freeze_se:
        for p in scale_embed_params:
            p.requires_grad_(False)
        for gi, group in enumerate(optimizer.param_groups):
            if group.get('name', '').endswith('scale_embed'):
                group['lr'] = 0.0
                scheduler.base_lrs[gi] = 0.0
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
    checkpoint_config = _base_checkpoint_config(
        git_hash=git_hash, tok_path=tok_path, encoded_path=encoded_path, n_params=n_params
    )

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

            loss_accum = 0.0
            micro_starts = []
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start < len(train_data):
                    micro_starts.append(idx_start)
            total_rows_this_accum = 0
            for idx_start in micro_starts:
                mb = min(BATCH_SIZE, len(train_data) - idx_start)
                if mb > 0:
                    total_rows_this_accum += mb * (MAX_SEQ_LEN - 1)
            total_rows_this_accum = max(total_rows_this_accum, 1)

            for idx_start in micro_starts:
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device, non_blocking=True)
                if x.dtype not in (torch.int32, torch.int64):
                    x = x.long()
                y = batch[:, 1:].to(device, non_blocking=True).long()

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                    # LigerFusedLinearCrossEntropyLoss.forward(lin_weight, _input, target)
                    loss = liger_ce_fn(
                        model_ref.out.weight,
                        hidden.contiguous().reshape(-1, hidden.size(-1)),
                        y.reshape(-1))
                    # Liger returns a mean loss; scale by token rows so the
                    # accumulation window is weighted by total tokens, not microbatch count.
                    n_rows = y.numel()
                    (loss * (float(n_rows) / float(total_rows_this_accum))).backward()
                    loss_accum += float(loss.detach().item()) * n_rows
                    del hidden, loss
                else:
                    n_rows = y.numel()
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                    total_loss, _ = _streamed_linear_ce_loss(
                        hidden, y, model_ref.out.weight,
                        chunk_rows=CE_CHUNK,
                        grad_denom=total_rows_this_accum,
                    )
                    loss_accum += float(total_loss.item())
                    del hidden, total_loss

            loss_val = loss_accum / float(total_rows_this_accum)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            total_norm_for_guard = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            if (not math.isfinite(loss_val)) or (not math.isfinite(total_norm_for_guard)):
                msg = (f'  [guard] non-finite step at ep{epoch} step {acc_step+1}: '
                       f'loss={loss_val} grad_norm={total_norm_for_guard}')
                if SKIP_NONFINITE_STEP:
                    print(msg + ' — skipping optimizer step', flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError(msg)

            if SE_MAX_ABORT > 0.0:
                se_guard = max((p.detach().abs().max().item() for p in model_ref.scale_embed_parameters()), default=0.0)
                if (not math.isfinite(se_guard)) or se_guard >= SE_MAX_ABORT:
                    raise FloatingPointError(
                        f'scale_embed guard tripped at ep{epoch} step {acc_step+1}: '
                        f'se_max={se_guard:.6f} >= {SE_MAX_ABORT:.6f}'
                    )

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
                total_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                lr_now = scheduler.get_last_lr()[0]
                routing_entropy = getattr(
                    model_ref.blocks[DSR_LAYER].attn,
                    '_routing_entropy', None)
                entropy_str = ''
                if routing_entropy is not None:
                    if torch.is_tensor(routing_entropy):
                        routing_entropy = float(routing_entropy.detach().item())
                    if isinstance(routing_entropy, float) and math.isfinite(routing_entropy):
                        entropy_str = f' routing_ent={routing_entropy:.3f}'
                stage2_frac = getattr(
                    model_ref.blocks[DSR_LAYER].attn,
                    '_stage2_selected_fraction', None)
                if stage2_frac is not None:
                    if torch.is_tensor(stage2_frac):
                        stage2_frac = float(stage2_frac.detach().item())
                    if isinstance(stage2_frac, float) and math.isfinite(stage2_frac):
                        entropy_str += f' stage2_frac={stage2_frac:.3f}'
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
                'config': checkpoint_config,
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt'),
               pickle_protocol=5)
            marker = ' *'

        _save_full = (_is_best or epoch % 3 == 0 or epoch == SCREEN_EPOCHS)
        _ep_state = model_ref.state_dict()
        _ep_ckpt = {'model_state_dict': _ep_state, 'epoch': epoch,
                    'global_step': step, 'config': checkpoint_config}
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
            if torch.is_tensor(routing_entropy):
                routing_entropy = float(routing_entropy.detach().item())
            if isinstance(routing_entropy, float) and math.isfinite(routing_entropy):
                print(f'  DSR routing entropy: {routing_entropy:.4f} '
                      f'(max={math.log(NUM_CHUNKS):.2f}, min=0.00)')

        print(f'  Physics: {model_ref.physics_summary()}')

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        print('  ' + format_passkey_results(pk))

    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)

    print('\n' + '=' * 70)
    print(f'  DSR + R_PLANES={R_PLANES} Bible-Muon Scratch 40/30/20/10 Summary (D{EMBEDDING_DIM}-L{NUM_LAYERS}, tied lm_head)')
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
