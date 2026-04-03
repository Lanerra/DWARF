"""
🚀 DWARF Moonshot-58M — Tensor-Parallel Version (head-parallel TP)

Architecture: D=512, H=8 (hd=64), L=8, FFN=2048, J=24 (se015 offsets)
  L0:  DSQGBlockV6Physics  IF=False  ← pure DSQG relay
  L1:  DSQGBlockV6Physics  IF=True   ← preIF (single layer before FA)
  L2:  FullAttentionBlock            ← FA@L2 (empirically optimal placement)
  L3-7: DSQGBlockV6Physics IF=False  ← post-FA relay layers

TP Design (head-parallel):
  - All ranks see the SAME input batch (NOT data-parallel, no DistributedSampler)
  - Each rank owns H_local = NUM_HEADS // TP_SIZE heads
  - Communication: dist.all_reduce at output projections ONLY (~2 allreduces per layer)
  - No DDP, no weight gradient allreduce (all ranks compute identical loss → identical weight grads)
  - Only rank 0 logs, saves checkpoints, runs passkey eval

Config:
  - Tokenizer: fineweb_tokenizer_32k.json  (32K BPE, FineWeb proper)
               EOS id = 0  (<|endoftext|>)
  - Dataset:   fineweb_edu_encoded_2048_v2.pt (~2.01M seqs, 4.13B tokens)
               pre-encoded with fineweb_tokenizer_32k
  - EMA_INIT = 0.0208 (= 1/48 = 1/δ_relay_min; empirically validated for J24D)
  - SCALE_EMBED_INIT = 0.1, LR_MULT = 15.0
  - Batch: BS=64 × GRAD_ACCUM=2 → eff_batch=128
  - Warm-start from moonshot_58m_best.pt (ep2, 35.04 PPL / 99.2% passkey)
  - MAX_TRAIN_SEQS=50K, 1 epoch — TP validation run
  - ~45.6M parameters (total; split across TP ranks)

Run (from repo root):
  # Single GPU (TP disabled):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_moonshot_58m_tp.py \\
    > logs/run_moonshot_58m_tp.log 2>&1 &

  # 2× GPU tensor parallel (NVLink required):
  torchrun --nproc_per_node=2 train/train_moonshot_58m_tp.py

  # 4× GPU tensor parallel:
  torchrun --nproc_per_node=4 train/train_moonshot_58m_tp.py
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

import os

# Auto-detect from torchrun (sets WORLD_SIZE automatically).
# Falls back to explicit TP_SIZE env var, then 1 (single-GPU).
# This fixes the "both ranks mapped to GPU 0" bug: when TP_SIZE=1,
# the TP branch is never entered and both ranks share cuda:0.
_world_size_env = os.environ.get("WORLD_SIZE")
TP_SIZE = int(_world_size_env) if _world_size_env is not None else int(os.environ.get("TP_SIZE", "1"))

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 2048
NUM_LAYERS       = 8
FULL_ATTN_LAYER  = 2

MAX_TRAIN_SEQS      = 50_000   # TP validation run — just enough to confirm TP works
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 1             # 1 epoch: validate TP init, training steps, passkey eval

# Warm-start from moonshot-58M ep2 checkpoint (full model, not FA-only).
# On RunPod, the checkpoint should be at autoresearch/checkpoints/moonshot_58m_best.pt
# (uploaded or copied from network volume).
WARM_START_CKPT = 'autoresearch/checkpoints/moonshot_58m_best.pt'

EXTRACTED_CKPT = None

# =============================================================================

import json, math, subprocess, sys, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

VOCAB_SIZE     = 32000
# H100/H200 141GB/rank: chunked CE eliminates the fp32 logit-grad spike, so limiting factor
# is just the BF16 logits: BS=64 × 2047 × 32K × 2B ≈ 8.4GB/rank — comfortable.
# BS=64 GRAD_ACCUM=2 gives eff_batch=128, much better GPU utilisation than BS=16 GRAD_ACCUM=8.
BATCH_SIZE     = 64
GRAD_ACCUM     = 2    # eff_batch = 128
CE_CHUNK       = 512  # chunked CE token stride — avoids materialising full fp32 grad tensor
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',   # RunPod network volume alt name
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_h100 import (
    DSQGAttentionV8_H100 as DSQGAttentionV6,
    dsqg_attention_v8,
    npci_rotate,
    ALL_OFFSETS as KERNEL_OFFSETS,
    J, J_SMALL, J_LARGE, R_PLANES,
)

assert len(OFFSETS) == 24
assert OFFSETS == KERNEL_OFFSETS

# ── condV physics helpers ─────────────────────────────────────────────────────

from causal_ema_scan import causal_ema_scan as _causal_ema_scan


def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor,
                floor: float = EMA_FLOOR) -> torch.Tensor:
    """Causal EMA — Triton scan (O(B·N·D) memory vs O(B·D·N·K) conv)."""
    return _causal_ema_scan(xi, ema_factor, floor=floor)


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


# =============================================================================
# Tensor Parallel Primitives
# =============================================================================

class _AllReduceFunc(torch.autograd.Function):
    """Differentiable allreduce: forward allreduces, backward passes through unchanged."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous(), None


def _all_reduce(x, group):
    """Allreduce with identity fallback for TP_SIZE=1 or uninitialized dist."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return x
    return _AllReduceFunc.apply(x, group)


# =============================================================================
# TP-aware Attention Modules
# =============================================================================

class TPDSQGAttentionV6(nn.Module):
    """
    Head-parallel DSQG V8 attention for tensor parallelism.

    Each rank owns H_local = num_heads_global // world_size heads.
    ColParallel: qkv_proj, gate_proj (input dim = D, output dim = 3*D_local or D_local)
    RowParallel: out_proj (input dim = D_local, output dim = D) + allreduce
    """
    def __init__(self, embedding_dim, num_heads_global, seq_len, dropout, rank, world_size, tp_group):
        super().__init__()
        assert num_heads_global % world_size == 0, \
            f"num_heads_global ({num_heads_global}) must be divisible by world_size ({world_size})"

        H_local = num_heads_global // world_size
        HD = embedding_dim // num_heads_global
        D_local = H_local * HD

        self.num_heads = H_local
        self.head_dim = HD
        self.rank = rank
        self.world_size = world_size
        self.tp_group = tp_group
        self.embedding_dim = embedding_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * D_local, bias=True)
        self.out_proj  = nn.Linear(D_local, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, D_local, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas_global = torch.linspace(0.2, 2.0, num_heads_global)
        alphas_local  = alphas_global[rank * H_local : (rank + 1) * H_local]
        delta_vals    = torch.tensor([math.log(1.0 + d) for d in OFFSETS], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas_local.unsqueeze(0))

        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain     = nn.Parameter(torch.ones(H_local))

        self.phase_base  = nn.Parameter(torch.zeros(J_LARGE, H_local, R_PLANES))
        self.phase_gain  = nn.Parameter(torch.zeros(J_LARGE, H_local, R_PLANES))

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes   = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(H_local))
        self.npci_theta_v = nn.Parameter(torch.zeros(H_local))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        D_local = H * HD

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D_local, dim=-1)

        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        sc = HD ** -0.5
        y_pre = torch.einsum('bhnd,rd->bhnr', q.float(), self.query_probes.float()).mul(sc).contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr', k.float(), self.key_probes.float()).mul(sc).contiguous()

        out = dsqg_attention_v8(
            q, k, v,
            self.pos_bias, self.scale_embed,
            self.phase_base, self.phase_gain,
            y_pre, z_pre,
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D_local)

        gate = torch.sigmoid(self.gate_proj(x))
        partial = self.out_proj(out_flat * gate)
        return self.dropout(_all_reduce(partial, self.tp_group))


class TPFullCausalAttention(nn.Module):
    """
    Head-parallel full causal attention for tensor parallelism.

    Same structure as DSQG: ColParallel qkv/gate, RowParallel out + allreduce.
    """
    def __init__(self, embedding_dim, num_heads_global, dropout, rank, world_size, tp_group):
        super().__init__()
        assert num_heads_global % world_size == 0, \
            f"num_heads_global ({num_heads_global}) must be divisible by world_size ({world_size})"

        H_local = num_heads_global // world_size
        HD = embedding_dim // num_heads_global
        D_local = H_local * HD

        self.num_heads = H_local
        self.head_dim = HD
        self.rank = rank
        self.world_size = world_size
        self.tp_group = tp_group
        self.embedding_dim = embedding_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * D_local, bias=True)
        self.out_proj  = nn.Linear(D_local, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, D_local, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        D_local = H * HD

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D_local, dim=-1)

        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D_local)

        gate = torch.sigmoid(self.gate_proj(x))
        partial = self.out_proj(out_flat * gate)
        return F.dropout(_all_reduce(partial, self.tp_group), p=self.dropout_p, training=self.training)


class TPFFN(nn.Module):
    """
    Column+Row parallel FFN for tensor parallelism.

    ColParallel fc1: Linear(D, FFN_DIM // world_size)
    RowParallel fc2: Linear(FFN_DIM // world_size, D) + allreduce
    """
    def __init__(self, embedding_dim, ffn_dim, dropout, world_size, tp_group):
        super().__init__()
        assert ffn_dim % world_size == 0, \
            f"ffn_dim ({ffn_dim}) must be divisible by world_size ({world_size})"

        ffn_local = ffn_dim // world_size
        self.world_size = world_size
        self.tp_group = tp_group

        self.fc1  = nn.Linear(embedding_dim, ffn_local)
        self.fc2  = nn.Linear(ffn_local, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        partial = self.fc2(self.drop(F.gelu(self.fc1(x))))
        return _all_reduce(partial, self.tp_group)


# =============================================================================
# Standard (non-TP) Modules
# =============================================================================

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


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


# =============================================================================
# Blocks (TP-aware)
# =============================================================================

class DSQGBlockV6Physics(nn.Module):
    """V8 DSQG attention + condV interference (EMA + KdV + AGC). TP-aware."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False, tp_config=None):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        if tp_config is not None and tp_config["world_size"] > 1:
            H_local = num_heads // tp_config["world_size"]
            D_local = H_local * self.head_dim
            self.attn = TPDSQGAttentionV6(
                embedding_dim, num_heads, seq_len, dropout,
                tp_config["rank"], tp_config["world_size"], tp_config["group"],
            )
            self.ffn = TPFFN(embedding_dim, ffn_dim, dropout, tp_config["world_size"], tp_config["group"])
            self._tp_active = True
            self._tp_config = tp_config
            self._H_local = H_local
            self._D_local = D_local
        else:
            self.attn = DSQGAttentionV6(embedding_dim, num_heads,
                                        seq_len=seq_len, dropout=dropout)
            self.ffn  = FFN(embedding_dim, ffn_dim, dropout)
            self._tp_active = False
            self._tp_config = None

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            if self._tp_active:
                D_local = self._D_local
                self.inter_k_proj = nn.Linear(embedding_dim, D_local)
                self.inter_v_proj = nn.Linear(embedding_dim, D_local)
            else:
                self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
                self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))
            self.kdv_alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape

            if self._tp_active:
                H, HD = self._H_local, self.head_dim
            else:
                H, HD = self.num_heads, self.head_dim

            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)
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


class FullAttentionBlock(nn.Module):
    """Full causal attention block. TP-aware."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1, tp_config=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        if tp_config is not None and tp_config["world_size"] > 1:
            self.attn = TPFullCausalAttention(
                embedding_dim, num_heads, dropout,
                tp_config["rank"], tp_config["world_size"], tp_config["group"],
            )
            self.ffn = TPFFN(embedding_dim, ffn_dim, dropout, tp_config["world_size"], tp_config["group"])
        else:
            self.attn = FullCausalAttention(embedding_dim, num_heads, dropout)
            self.ffn  = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Model (TP-aware)
# =============================================================================

class AutoresearchTransformerPhysics(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer, interference_interval,
                 scale_embed_init_val=0.0, dropout=0.1, tp_config=None):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.drop            = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer
        self._tp_config      = tp_config

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout, tp_config=tp_config))
            else:
                has_if = (i == full_attn_layer - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if, tp_config=tp_config))
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
            if isinstance(m, (DSQGAttentionV6, TPDSQGAttentionV6)):
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def forward(self, idx):
        B, N = idx.shape
        x    = self.drop(self.embedding(idx))
        for block in self.blocks:
            if self.training:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, (DSQGAttentionV6, TPDSQGAttentionV6)):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def full_attn_parameters(self):
        for p in self.blocks[self.full_attn_layer].parameters():
            yield p

    def non_full_attn_parameters(self):
        fa_ids = {id(p) for p in self.full_attn_parameters()}
        for p in self.parameters():
            if id(p) not in fa_ids:
                yield p

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                kdv   = block.kdv_alpha.item()
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t) kdv={kdv:.4f}')
        return '  '.join(entries)


# =============================================================================
# Data utilities
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


def load_data():
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
    else:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True)
        texts = []
        for item in ds:
            if len(item['text']) < 5_000:
                continue
            texts.append(item['text'])
            if len(texts) >= 100_000:
                break
        os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
        with open(FW_CACHE_FILE, 'w') as fp:
            json.dump(texts, fp)
    n = len(texts)
    return {'train': texts[:int(n * 0.95)],
            'val':   texts[int(n * 0.95):int(n * 0.95) + 2500]}


def encode_split(split_texts, tokenizer, split_name):
    from tokenizers import Tokenizer as _Tokenizer
    _tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    _raw_tok = _Tokenizer.from_file(_tok_path)
    eos_id = _raw_tok.token_to_id('<|endoftext|>')
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(eos_id)
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
            filler   = []
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


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir):
    full_attn_block = model.blocks[model.full_attn_layer]
    state_dict = {}
    for name, param in full_attn_block.named_parameters():
        state_dict[f"blocks.{model.full_attn_layer}.{name}"] = param.data.clone()

    payload = {
        "full_attn_block": state_dict,
        "config": {
            "embedding_dim":     EMBEDDING_DIM,
            "num_heads":         NUM_HEADS,
            "ffn_dim":           FFN_DIM,
            "seq_len":           MAX_SEQ_LEN,
            "source_script":     "train/train_moonshot_58m_tp.py",
            "source_layer":      FULL_ATTN_LAYER,
            "num_layers":        NUM_LAYERS,
            "num_offsets":       len(OFFSETS),
            "epoch":             epoch,
            "git_hash":          git_hash,
            "tp_size":           TP_SIZE,
            "note": (
                f"Moonshot-58M TP: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} "
                f"J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1}. "
                f"Epoch {epoch}/{SCREEN_EPOCHS}. fineweb_tokenizer_32k. Warm-start ep2. TP_SIZE={TP_SIZE}."
            ),
        },
    }

    out_path = os.path.join(checkpoint_dir, f"moonshot_58m_tp{TP_SIZE}_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# =============================================================================
# Training
# =============================================================================

def train():
    if TP_SIZE > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = dist.get_world_size()
        tp_group = dist.new_group(list(range(world_size)))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        tp_group = None
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    if rank == 0:
        print('=' * 70)
        print(f'  🚀 DWARF Moonshot-58M TP — D=512 H=8 hd=64 L=8 J=24, cold start')
        print(f'  TP_SIZE={TP_SIZE}, world_size={world_size}')
        print('  FA@L2, preIF@L1, fineweb_tokenizer_32k, EMA_INIT=1/δ_relay_min=0.0208')
        print('=' * 70)
        if torch.cuda.is_available():
            print(f'  GPU: {torch.cuda.get_device_name(0)}')
            _cc = torch.cuda.get_device_capability()
            _path = 'sm_90 (H100)' if ((_cc[0]==9 and _cc[1]==0) or _cc[0]>9) else \
                    'sm_89 (4090 Ada — tuned)' if (_cc[0]==8 and _cc[1]==9) else \
                    f'sm_{_cc[0]}{_cc[1]} (generic)'
            print(f'  Kernel path: {_path}')
        print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
        print(f'  Pre-FA IF only: IF on layers < {FULL_ATTN_LAYER}, pure DSQG on layers >= {FULL_ATTN_LAYER}')
        print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
        print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
        print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
        print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
        print(f'  git={git_hash}')

    splits = load_data()
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    if rank == 0:
        print(f'Loaded tokenizer from {tok_path}')

    _encoded_cache = next(
        (p for p in ['logs/fineweb_edu_encoded_2048_v2.pt',
                     'logs/fineweb_encoded_2048.pt']          # RunPod fallback (old)
         if os.path.exists(p)), None)
    if _encoded_cache is not None:
        if rank == 0:
            print(f'Loading pre-encoded dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train'].long()
        val_data   = _cache['val'].long()
    else:
        train_data = encode_split(splits['train'], tokenizer, 'Train')
        val_data   = encode_split(splits['val'],   tokenizer, 'Val')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    if rank == 0:
        print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    tp_cfg = {"rank": rank, "world_size": world_size, "group": tp_group} if TP_SIZE > 1 else None
    model = AutoresearchTransformerPhysics(
        vocab_size=tokenizer.vocab_size(), embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=None,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        tp_config=tp_cfg,
    ).to(device)

    best_ckpt_name = f'moonshot_58m_tp{TP_SIZE}_best.pt'

    # Warm-start: load full model weights from ep2 checkpoint, slicing for TP
    if WARM_START_CKPT and os.path.exists(WARM_START_CKPT):
        if rank == 0:
            print(f'\n  Warm-start from {WARM_START_CKPT} (TP_SIZE={TP_SIZE})')
        raw = torch.load(WARM_START_CKPT, map_location='cpu', weights_only=True)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            full_state = raw['model_state_dict']
        else:
            full_state = raw
        # Strip torch.compile _orig_mod prefix if present
        full_state = {k.replace('_orig_mod.', ''): v for k, v in full_state.items()}

        if TP_SIZE == 1:
            # No slicing needed — load directly
            missing, unexpected = model.load_state_dict(full_state, strict=False)
        else:
            # TP-aware load: slice head-parallel tensors for this rank
            H_global = NUM_HEADS
            H_local  = H_global // TP_SIZE
            HD       = EMBEDDING_DIM // H_global
            D_local  = H_local * HD
            FFN_local = FFN_DIM // TP_SIZE
            h_start  = rank * H_local
            h_end    = h_start + H_local
            f_start  = rank * FFN_local
            f_end    = f_start + FFN_local

            sliced = {}
            for k, v in full_state.items():
                # Per-head scalars/vectors: [H, ...] → [H_local, ...]
                if any(k.endswith(s) for s in [
                    'attn.pos_bias',       # [J, H]  → [J, H_local]
                    'attn.scale_embed',    # [J, HD]  (not split — shared across heads of same offset... actually per-head in TP)
                    'attn.if_gain',        # [H]      → [H_local]
                    'attn.npci_theta_k',   # [H]      → [H_local]
                    'attn.npci_theta_v',   # [H]      → [H_local]
                ]):
                    # pos_bias: [J, H], if_gain/npci: [H], need head slice
                    if k.endswith('attn.pos_bias'):
                        sliced[k] = v[:, h_start:h_end]              # [J, H_local]
                    elif k.endswith('attn.scale_embed'):
                        sliced[k] = v                                 # [J, HD] — not head-split, same for all ranks
                    else:
                        sliced[k] = v[h_start:h_end]                 # [H_local]
                elif k.endswith('attn.phase_base') or k.endswith('attn.phase_gain'):
                    sliced[k] = v[:, h_start:h_end, :]               # [J_large, H_local, R]
                # ColParallel: qkv_proj weight [3*H*HD, D] → [3*H_local*HD, D]
                elif k.endswith('attn.qkv_proj.weight'):
                    # rows are [q_heads | k_heads | v_heads], each H*HD
                    q = v[h_start*HD : h_end*HD]
                    k_ = v[H_global*HD + h_start*HD : H_global*HD + h_end*HD]
                    vv = v[2*H_global*HD + h_start*HD : 2*H_global*HD + h_end*HD]
                    sliced[k] = torch.cat([q, k_, vv], dim=0)
                elif k.endswith('attn.qkv_proj.bias'):
                    q = v[h_start*HD : h_end*HD]
                    k_ = v[H_global*HD + h_start*HD : H_global*HD + h_end*HD]
                    vv = v[2*H_global*HD + h_start*HD : 2*H_global*HD + h_end*HD]
                    sliced[k] = torch.cat([q, k_, vv], dim=0)
                # ColParallel: gate_proj weight [H*HD, D] → [H_local*HD, D]
                elif k.endswith('attn.gate_proj.weight'):
                    sliced[k] = v[h_start*HD : h_end*HD]
                elif k.endswith('attn.gate_proj.bias'):
                    sliced[k] = v[h_start*HD : h_end*HD]
                # preIF interference projections: inter_k_proj/inter_v_proj [D_local, D] → slice rows
                elif k.endswith('inter_k_proj.weight') or k.endswith('inter_v_proj.weight'):
                    sliced[k] = v[h_start*HD : h_end*HD]
                elif k.endswith('inter_k_proj.bias') or k.endswith('inter_v_proj.bias'):
                    sliced[k] = v[h_start*HD : h_end*HD]
                # RowParallel: out_proj weight [D, H*HD] → [D, H_local*HD]  (col slice)
                elif k.endswith('attn.out_proj.weight'):
                    sliced[k] = v[:, h_start*HD : h_end*HD]
                # FFN ColParallel: fc1 [FFN, D] → [FFN_local, D]
                elif k.endswith('ffn.fc1.weight'):
                    sliced[k] = v[f_start:f_end]
                elif k.endswith('ffn.fc1.bias'):
                    sliced[k] = v[f_start:f_end]
                # FFN RowParallel: fc2 [D, FFN] → [D, FFN_local]
                elif k.endswith('ffn.fc2.weight'):
                    sliced[k] = v[:, f_start:f_end]
                else:
                    # Embeddings, lm_head, layer norms, FA block — replicated on all ranks
                    sliced[k] = v

            missing, unexpected = model.load_state_dict(sliced, strict=False)

        if rank == 0:
            if missing:
                print(f'  Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing)>5 else ""}')
            if unexpected:
                print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
            print(f'  Warm-start loaded OK')
    else:
        if rank == 0:
            print(f'\n  Cold start — random init (WARM_START_CKPT not found or not set)')

    if TP_SIZE == 1:
        try:
            for i, block in enumerate(model.blocks):
                if type(block).__name__ == "FullAttentionBlock":
                    model.blocks[i] = torch.compile(block, fullgraph=False)
                    if rank == 0:
                        print(f"  torch.compile applied to FullAttentionBlock at layer {i}")
        except Exception as e:
            if rank == 0:
                print(f"  torch.compile skipped: {e}")

    n_params = model.param_count()
    if rank == 0:
        print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = torch.optim.AdamW([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params,     'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, SCREEN_EPOCHS + 1):
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
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat      = y.reshape(-1)
                T           = logits_flat.size(0)
                grad_logits = torch.empty_like(logits_flat)
                total_loss  = 0.0
                for chunk_start in range(0, T, CE_CHUNK):
                    chunk_end = min(chunk_start + CE_CHUNK, T)
                    chunk     = logits_flat[chunk_start:chunk_end].detach().requires_grad_(True)
                    chunk_loss = F.cross_entropy(
                        chunk, y_flat[chunk_start:chunk_end], reduction='sum')
                    chunk_loss.backward()
                    grad_logits[chunk_start:chunk_end] = chunk.grad
                    total_loss += chunk_loss.item()
                logits_flat.backward(grad_logits / (T * GRAD_ACCUM))
                loss = torch.tensor(total_loss / T, device=device)
                del logits, logits_flat, y_flat, grad_logits
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 200 == 0 and rank == 0:
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item():.4f}', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, best_ckpt_name))
                marker = ' *'

            print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

            se_vals = []
            for m in model.modules():
                if isinstance(m, (DSQGAttentionV6, TPDSQGAttentionV6)):
                    se_vals.append(m.scale_embed.detach().abs())
            if se_vals:
                se_all = torch.cat(se_vals)
                print(f'  scale_embed |mean|={se_all.mean():.4f} '
                      f'|max|={se_all.max():.4f}')

            print(f'  Physics: {model.physics_summary()}')
            save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

            pk      = passkey_accuracy(model, tokenizer, device)
            pk_mean = sum(pk.values()) / len(pk)
            passkey_results[epoch] = pk_mean * 100
            print(f'  Passkey mean={pk_mean * 100:.1f}%')
            parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
            print('  ' + '  '.join(parts))
            sys.stdout.flush()

    if rank == 0:
        elapsed_s     = time.time() - t_start
        memory_mb     = torch.cuda.max_memory_allocated() / 1e6
        passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
        ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
        PPL_BASELINE     = 61.75
        PASSKEY_BASELINE = 18.3
        ar_score = (passkey_final - PASSKEY_BASELINE) - max(0, ppl_final - PPL_BASELINE) * 0.5

        print('\n---')
        for ep in range(1, SCREEN_EPOCHS + 1):
            print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
        for ep in range(1, SCREEN_EPOCHS + 1):
            print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
        print(f'ar_score:       {ar_score:.2f}')
        print(f'memory_mb:      {memory_mb:.1f}')
        print(f'elapsed_s:      {elapsed_s:.1f}')
        print(f'num_params_M:   {n_params / 1e6:.1f}')
        print(f'num_layers:     {NUM_LAYERS}')
        print(f'num_offsets:    {len(OFFSETS)}')
        print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
        print(f'ema_init:       {EMA_INIT}')
        print(f'tp_size:        {TP_SIZE}')
        print(f'description:    Moonshot 58M TP — D={EMBEDDING_DIM} H={NUM_HEADS} hd=64 L={NUM_LAYERS} '
              f'FFN={FFN_DIM} J=24 se015, cold start, fineweb_tokenizer_32k, TP_SIZE={TP_SIZE}')

    if TP_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
