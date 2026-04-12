"""
🔬 ABLATION: Pure DSQG — D=768 L=25 Triadic J=96 [A,A,B,B,C,C], NO FA LAYER

Tests: does relay form without full-attention layer?
  - If passkey >50% by ep2: relay is genuinely load-bearing, FA not required
  - If passkey stays random: FA is the operative mechanism (retrieval heads)
  - Reference: Wu et al. ICLR 2025 — retrieval heads alternative hypothesis

Architecture: identical to hybrid model EXCEPT:
  - L06 is now DSQGBlock(GROUP_A) instead of FullAttentionBlock
  - No full-attention anywhere in the stack
  - preIF stays at L05 (ablating FA specifically, not preIF)
  - 25 pure DSQG layers, full J=96 triadic [A,A,B,B,C,C] coverage

Layer layout (pure DSQG):
  L00: DSQGBlock(GROUP_A)  ─┐
  L01: DSQGBlock(GROUP_A)   │ triad 1
  L02: DSQGBlock(GROUP_B)   │
  L03: DSQGBlock(GROUP_B)   │
  L04: DSQGBlock(GROUP_C)   │
  L05: DSQGBlock(GROUP_C)  ─┘ + preIF (C retains EMA injection)
  L06: DSQGBlock(GROUP_A)  ─┐  ← was FA@24%; now DSQG-A
  L07: DSQGBlock(GROUP_A)   │ triad 2
  L08: DSQGBlock(GROUP_B)   │
  L09: DSQGBlock(GROUP_B)   │
  L10: DSQGBlock(GROUP_C)   │
  L11: DSQGBlock(GROUP_C)  ─┘
  L12: DSQGBlock(GROUP_A)  ─┐ triad 3
  L13: DSQGBlock(GROUP_A)   │
  L14: DSQGBlock(GROUP_B)   │
  L15: DSQGBlock(GROUP_B)   │
  L16: DSQGBlock(GROUP_C)   │
  L17: DSQGBlock(GROUP_C)  ─┘
  L18: DSQGBlock(GROUP_A)  ─┐ triad 4
  L19: DSQGBlock(GROUP_A)   │
  L20: DSQGBlock(GROUP_B)   │
  L21: DSQGBlock(GROUP_B)   │
  L22: DSQGBlock(GROUP_C)   │
  L23: DSQGBlock(GROUP_C)  ─┘
  L24: DSQGBlock(GROUP_A)    triad 5 partial (tail)

No FA → no O(N²) attention map → ~3.5 GB VRAM freed → no checkpointing needed.
Expected: ~200–250K tok/s on 4090 (vs ~93K for hybrid with FA).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d768_l25_pure_dsqg_4090_bf16.py

Recovery (if relay stalls): set DWARF_PHASE_GAIN_LR_MULT=10 to give phase_gain
  a dedicated higher learning rate (Selective RoPE fix for undertrained rotations).

Corpus: mixed (60% FineWeb-Edu / 25% PG19 / 15% Stack)
If relay stalls by ep2, switch to FineWeb-Edu only:
  DWARF_DATASET=fineweb .venv/bin/python3 -u ...
"""

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available, falling back to standard AdamW")

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Liger disabled — known BF16 autocast instability with gradient accumulation
USE_LIGER_CE = False

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import triton
from dsqg_attention_v8_dynamic_j import (
    ALL_OFFSETS as _ALL_96,
    _fwd_v8, _compute_D_v8, _bwd_dq_v8, _bwd_dkdv_v8,
    npci_rotate, R_PLANES, _next_pow2,
)
from dsqg_persistent_v1 import DSQGPersistentBlock

# V13 + gradient checkpointing = known memory aliasing issues on 4090.
# Default to V8 here. Override with DWARF_USE_V13=1 only on H200+ (no checkpointing).
_USE_V13 = os.environ.get('DWARF_USE_V13', '0') == '1'
_V13Cls = None
if _USE_V13:
    try:
        from dsqg_attention_v13_dynamic_j import DSQGAttentionV13Dynamic as _V13Cls
        print('  Kernel: V13-dynamic-J (two-pass, ~3× faster)')
    except Exception as _e:
        print(f'  V13 import failed ({_e}), falling back to V8')
        _USE_V13 = False
_DSQG_TYPES = None

from causal_ema_scan import causal_ema_scan as _causal_ema_scan

# =============================================================================
# OFFSET GROUPS  (same as hybrid model)
# =============================================================================

GROUP_A = _ALL_96[0:32]
GROUP_B = _ALL_96[32:64]
GROUP_C = _ALL_96[64:96]

def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets)
    return j_small, j_large

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

EMBEDDING_DIM   = 768
NUM_HEADS       = 12
FFN_DIM         = 1536
NUM_LAYERS      = 25
VOCAB_SIZE      = 32768

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 24.0   # mixed-domain: higher multiplier for gradient dilution
PHASE_GAIN_LR_MULT   = float(os.environ.get('DWARF_PHASE_GAIN_LR_MULT', '1.0'))
# Set DWARF_PHASE_GAIN_LR_MULT=10 to apply Selective RoPE fix for undertrained rotations

EMA_INIT  = 0.010     # mixed-domain: longer window (learned α → 690–822t on mixed)
EMA_FLOOR = 0.00001
LR        = 3e-4
DROPOUT   = 0.0

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '64'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '2'))     # eff_batch = 128
MAX_SEQ_LEN    = 2048
CE_CHUNK       = 512
SCREEN_EPOCHS  = 3

# Dataset selection: mixed (default) or fineweb-only for fallback
_DATASET_KEY = os.environ.get('DWARF_DATASET', 'mixed').lower()
if _DATASET_KEY == 'fineweb':
    DATASET_CANDIDATES = [
        'fineweb_wiki_80_20_encoded_2048.pt',
        'logs/fineweb_wiki_80_20_encoded_2048.pt',
        'logs/fineweb_edu_encoded_2048_v2.pt',
    ]
    MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '366000'))
    CKPT_BASE_NAME = 'd768_l25_pure_dsqg_fineweb_scratch'
    print("  Dataset: FineWeb-Edu/Wiki (80/20) — fallback corpus")
else:
    DATASET_CANDIDATES = [
        'mixed_encoded_2048_fineweb_tok.pt',
        'logs/mixed_encoded_2048_fineweb_tok.pt',
        'data/mixed_encoded_2048_fineweb_tok.pt',
    ]
    MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '234000'))
    CKPT_BASE_NAME = 'd768_l25_pure_dsqg_mixed_scratch'
    print("  Dataset: Mixed corpus (60% FineWeb / 25% PG19 / 15% Stack)")

MAX_VAL_SEQS = 5_582
TOKENIZER_CANDIDATES = ['results/fineweb_tokenizer_32k.json']

PASSKEY_DISTANCES   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS      = 50
PASSKEY_BATCH_SIZE  = 8
_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape',
                    'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'

CHECKPOINT_DIR      = 'autoresearch/checkpoints'
# NOTE: PyTorch SDPA uses Flash Attention (O(N) memory) — removing FA saves ~0 VRAM.
# DSQG custom autograd saves q,k,v,out,lse per layer (~100MB × 25 = ~2.5GB).
# every_other checkpointing required, same as hybrid model. Use DWARF_CKPT=none only on H200+.
CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# LAYER LAYOUT — pure DSQG, L06 is GROUP_A not FA
# =============================================================================

LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00 ┐ triad 1
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L01 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L02 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L03 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L04 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L05 ┘ + preIF (EMA injection)
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L06 ┐ triad 2  ← WAS FA
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L08 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L09 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L10 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L11 ┘
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L12 ┐ triad 3
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L13 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L14 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L15 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L16 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L17 ┘
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L18 ┐ triad 4
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L19 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L20 │
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L21 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L22 │
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L23 ┘
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L24   tail
]

assert len(LAYER_LAYOUT) == NUM_LAYERS
assert not any(label == 'FA' for label, *_ in LAYER_LAYOUT), "Pure DSQG: no FA entries allowed"

# =============================================================================
# GROUPED ATTENTION (copy from hybrid — identical)
# =============================================================================

class _DSQGFnGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                offsets_dev, j_val, j_small, j_large):
        B, H, N, HD = q.shape
        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)
        if HD <= 64:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64, 8, 2
            else:       BLOCK_N, _num_warps, _num_stages = 64, 4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 32, 4, 2
        else:
            BLOCK_N, _num_warps, _num_stages = 16, 4, 2
        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BLOCK_N))
        _fwd_v8[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse, offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=_num_warps, num_stages=_num_stages,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev)
        ctx.BLOCK_N = BLOCK_N; ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _num_warps; ctx.num_stages = _num_stages
        ctx.j_val = j_val; ctx.j_small = j_small; ctx.j_large = j_large
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse, offsets_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD = ctx.BLOCK_N, ctx.BLOCK_HD
        NW, NS = ctx.num_warps, ctx.num_stages
        j_val, j_small, j_large = ctx.j_val, ctx.j_small, ctx.j_large
        dout = dout.contiguous()
        D_buf = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BN))
        _compute_D_v8[g](
            dout, out, D_buf,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )
        blocks_n = (N + BN - 1) // BN
        dq = torch.zeros_like(q)
        dy_pre = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B*H, blocks_n, j_val, device=q.device, dtype=torch.float32)
        dse_buf = torch.empty(B*H, blocks_n, j_val*HD, device=q.device, dtype=torch.float32)
        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D_buf, dq, dpb_buf, dse_buf, dy_pre, offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dpb_buf.stride(0), dpb_buf.stride(1),
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            dse_buf.stride(0), dse_buf.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        # dpb_buf: [B*H, blocks_n, j_val] — pos_bias is [j_val, H], must preserve H
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0)  # [j_val, H]
        # dse_buf: [B*H, blocks_n, j_val*HD] — scale_embed is [j_val, HD], shared across heads
        dse = dse_buf.view(B, H, blocks_n, j_val * HD).sum(dim=(0, 1, 2)).view(j_val, HD)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)
        # phase_base/gain use reduction buffers: [B*H, blocks_n, J_LARGE*2]
        phase_base_buf = torch.empty(B*H, blocks_n, j_large * 2,
                                     device=q.device, dtype=torch.float32)
        phase_gain_buf = torch.empty(B*H, blocks_n, j_large * 2,
                                     device=q.device, dtype=torch.float32)
        stride_buf_bh  = blocks_n * j_large * 2
        stride_buf_blk = j_large * 2
        _bwd_dkdv_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D_buf, dk, dv, phase_base_buf, phase_gain_buf, dz_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            stride_buf_bh, stride_buf_blk,
            dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        # Reduce phase buffers: [B*H, blocks_n, J_LARGE*2] -> [J_LARGE, H, 2]
        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, j_large, 2).sum(dim=(0, 2))
            return r.permute(1, 0, 2).contiguous()
        dphase_base = _reduce_phase_buf(phase_base_buf)
        dphase_gain = _reduce_phase_buf(phase_gain_buf)
        return (dq, dk, dv, dpb, dse,
                dphase_base, dphase_gain, dy_pre, dz_pre,
                None, None, None, None)


def dsqg_attention_grouped(q, k, v, pos_bias, scale_embed,
                            phase_base, phase_gain, y_pre, z_pre,
                            offsets_dev, j_val, j_small, j_large):
    q = q.contiguous(); k = k.contiguous()
    v = v.contiguous(); y_pre = y_pre.contiguous(); z_pre = z_pre.contiguous()
    return _DSQGFnGrouped.apply(q, k, v, pos_bias, scale_embed,
                                phase_base, phase_gain, y_pre, z_pre,
                                offsets_dev, j_val, j_small, j_large)


class DSQGAttentionGrouped(nn.Module):
    def __init__(self, embedding_dim, num_heads, offsets, j_small, j_large,
                 seq_len=2048, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.j_val = len(offsets)
        self.j_small = j_small
        self.j_large = j_large
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.pos_bias    = nn.Parameter(torch.zeros(self.j_val, num_heads))
        self.scale_embed = nn.Parameter(torch.zeros(self.j_val, self.head_dim))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))
        if j_large > 0:
            self.phase_base  = nn.Parameter(torch.zeros(j_large, num_heads, R_PLANES))
            # SO(2) Lie group parameterization: (x,y) on unit circle, no 2pi wrap-arounds.
            # init=(1,0) -> identity rotation (numerically same as zero-init scalar).
            _pgv = torch.zeros(j_large, num_heads, R_PLANES, 2)
            _pgv[..., 0] = 1.0
            self.phase_gain_vec = nn.Parameter(_pgv)
            self.phase_gate  = nn.Parameter(torch.zeros(j_large))
            self.query_probes = nn.Parameter(torch.zeros(num_heads, self.head_dim, R_PLANES))
            self.key_probes   = nn.Parameter(torch.zeros(num_heads, self.head_dim, R_PLANES))
        else:
            _z = torch.zeros(0, num_heads, R_PLANES)
            self.phase_base = nn.Parameter(_z.clone())
            # SO(2) empty-case: shape (0, H, R_PLANES, 2)
            self.phase_gain_vec = nn.Parameter(torch.zeros(0, num_heads, R_PLANES, 2))
            self.phase_gate = nn.Parameter(torch.zeros(max(j_large, 1)))
            self.query_probes = nn.Parameter(torch.zeros(num_heads, self.head_dim, R_PLANES))
            self.key_probes   = nn.Parameter(torch.zeros(num_heads, self.head_dim, R_PLANES))
        _offs = torch.tensor(offsets, dtype=torch.int32)
        self.register_buffer('offsets_dev', _offs)
        self.dropout_p = dropout

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta
        q_norm = F.normalize(q.float(), dim=-1)
        v_norm = F.normalize(v.float(), dim=-1)
        qp_norm = F.normalize(self.query_probes.float(), dim=1)
        kp_norm = F.normalize(self.key_probes.float(), dim=1)
        y_pre = (q_norm @ qp_norm).contiguous()
        z_pre = (v_norm @ kp_norm).contiguous()

        # SO(2) MOVT: derive scalar phase_gain via unit-circle normalization
        _pgv_unit = self.phase_gain_vec / self.phase_gain_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        phase_gain = _pgv_unit[..., 0].contiguous()
        gate = torch.sigmoid(self.phase_gate)[:, None, None]
        gated_phase_base = self.phase_base * gate
        gated_phase_gain = phase_gain * gate

        out = dsqg_attention_grouped(
            q, k, v, self.pos_bias, self.scale_embed,
            gated_phase_base, gated_phase_gain, y_pre, z_pre,
            self.offsets_dev, self.j_val, self.j_small, self.j_large,
        )
        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out_flat * gate)


# =============================================================================
# MODEL BLOCKS
# =============================================================================

def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)

def _agc_normalize(pool, eps=1e-6):
    rms = pool.norm(dim=-1, keepdim=True) / (pool.shape[-1] ** 0.5)
    return pool / (rms + eps)


class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockTriadic(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 offsets, j_small, j_large, group_label,
                 dropout=0.0, interference=False):
        super().__init__()
        self.interference = interference
        self.group_label = group_label
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        _AttnCls = _V13Cls if (_USE_V13 and _V13Cls is not None) else DSQGAttentionGrouped
        self.attn = _AttnCls(
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


# =============================================================================
# PURE DSQG MODEL  (uses DSQGPersistentBlock — fused multi-layer + SO(2) MOVT)
# =============================================================================

class PureDSQGModel(nn.Module):
    """
    D=768 L=25 triadic J=96 [A,A,B,B,C,C] — no full-attention layer.
    Uses DSQGPersistentBlock for fused parameter management + SO(2) phase_gain.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_dim, seq_len,
                 scale_embed_init_val=0.15, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        self.drop = nn.Dropout(dropout)

        # Map CHECKPOINT_STRATEGY -> checkpoint_every int for DSQGPersistentBlock
        _ckpt_every = {'none': 0, 'all': 1, 'every_other': 2}.get(
            CHECKPOINT_STRATEGY, 0)

        layer_configs = [
            {'offsets': offsets, 'j_small': js, 'j_large': jl, 'has_if': has_if}
            for (label, offsets, js, jl, has_if) in LAYER_LAYOUT
        ]

        self.blocks = DSQGPersistentBlock(
            num_layers=NUM_LAYERS,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            layer_configs=layer_configs,
            dropout=dropout,
            scale_embed_init_val=scale_embed_init_val,
            checkpoint_every=_ckpt_every,
            ema_init=EMA_INIT,
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight  # weight tying

    def forward(self, idx):
        x = self.drop(self.embedding(idx))
        x = self.blocks(x)
        return self.norm(x)

    def forward_hidden(self, idx):
        return self.forward(idx)

    def logits(self, hidden):
        return self.out(hidden)

    @property
    def scale_embed_max(self):
        vals = [p.abs().max().item() for p in self.blocks.scale_embed_list]
        return max(vals) if vals else 0.0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# OPTIMIZER — with optional phase_gain higher LR
# =============================================================================

def build_optimizer(model):
    PHASE_LR_MULT = 50
    scale_embed_params = []
    phase_params = []
    other_params = []

    phase_keywords = ('phase_gain', 'phase_gate', 'query_probes', 'key_probes')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale_embed' in name:
            scale_embed_params.append(param)
        elif any(kw in name for kw in phase_keywords):
            phase_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params,       'lr': LR, 'name': 'main'},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT, 'name': 'scale_embed'},
        {'params': phase_params,       'lr': LR * PHASE_LR_MULT, 'name': 'phase'},
    ]
    print(f'  phase params LR: {LR * PHASE_LR_MULT:.2e} (50× base)')

    if _BNB_AVAILABLE:
        opt = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        print(f"  Optimizer: AdamW8bit (bitsandbytes)")
    else:
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        print(f"  Optimizer: AdamW (standard)")
    return opt


def build_scheduler(optimizer, total_steps, warmup_steps=200):
    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# =============================================================================
# DATASET
# =============================================================================

def find_file(candidates, root=None):
    root = root or _project_root
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def load_dataset(path, max_train, max_val, seq_len):
    print(f"  Loading dataset: {path}")
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        train_data = data.get('train', data.get('input_ids'))
        val_data   = data.get('val',   data.get('val_ids'))
    else:
        n_val = min(max_val, len(data) // 20)
        train_data, val_data = data[:-n_val], data[-n_val:]

    train_data = train_data[:max_train]
    val_data   = val_data[:max_val]
    print(f"  Train: {len(train_data):,} seqs | Val: {len(val_data):,} seqs | "
          f"seq_len={seq_len}")
    return train_data, val_data


# =============================================================================
# CHUNKED CROSS-ENTROPY (Liger disabled — BF16 instability)
# =============================================================================

def chunked_ce_loss(model, hidden, targets, chunk_size=CE_CHUNK):
    B, N, D = hidden.shape
    hidden_2d = hidden.contiguous().reshape(B * N, D)
    targets_1d = targets.contiguous().reshape(B * N).long()  # dataset stores int32; CE needs int64
    total_loss = torch.tensor(0.0, device=hidden.device, requires_grad=True)
    n_chunks = math.ceil(B * N / chunk_size)
    for i in range(n_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, B * N)
        logits_chunk = model.logits(hidden_2d[s:e])
        loss_chunk = F.cross_entropy(
            logits_chunk.float(),
            targets_1d[s:e],
            ignore_index=-1,
            reduction='mean'
        ) * (e - s) / (B * N)
        total_loss = total_loss + loss_chunk
    return total_loss


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, val_data, device, batch_size=8):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i+batch_size].to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with _amp_context(device):
            hidden = model.forward_hidden(inputs.long())
        loss = chunked_ce_loss(model, hidden, targets)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def passkey_eval(model, tokenizer, device, distances, n_trials, batch_size):
    """Standard passkey retrieval evaluation."""
    model.eval()
    results = {}
    for dist in distances:
        correct = 0
        total = 0
        for batch_start in range(0, n_trials, batch_size):
            b = min(batch_size, n_trials - batch_start)
            word_idxs = [(batch_start + j) % len(_PASSKEY_WORDS) for j in range(b)]
            words = [_PASSKEY_WORDS[idx] for idx in word_idxs]
            prompts = []
            for w in words:
                intro = _INTRO_TEMPLATE.format(word=w)
                filler_tokens = tokenizer.encode(_FILLER_SENTENCE, add_special_tokens=False)
                intro_tokens  = tokenizer.encode(intro, add_special_tokens=False)
                cue_tokens    = tokenizer.encode(_RETRIEVAL_CUE, add_special_tokens=False)
                n_filler_needed = dist - len(intro_tokens)
                if n_filler_needed < 0:
                    prompts = None; break
                reps = math.ceil(n_filler_needed / max(1, len(filler_tokens)))
                filler = (filler_tokens * (reps + 1))[:n_filler_needed]
                seq = intro_tokens + filler + cue_tokens
                prompts.append(seq)
            if prompts is None:
                break
            max_len = max(len(p) for p in prompts)
            pad_id  = tokenizer.pad_token_id or 0
            padded  = [p + [pad_id] * (max_len - len(p)) for p in prompts]
            inp     = torch.tensor(padded, dtype=torch.long, device=device)
            with _amp_context(device):
                hidden = model.forward_hidden(inp)
            logits_last = model.logits(hidden[:, -1, :])
            preds = logits_last.argmax(dim=-1)
            for j, (pred_id, w) in enumerate(zip(preds.tolist(), words)):
                pred_str = tokenizer.decode([pred_id]).strip().lower()
                if pred_str == w.lower():
                    correct += 1
                total += 1
        results[dist] = correct / max(1, total)
    mean_acc = sum(results.values()) / len(results)
    return results, mean_acc


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  ABLATION: Pure DSQG — D={EMBEDDING_DIM} L={NUM_LAYERS}")
    print(f"  NO FULL-ATTENTION LAYER (testing relay without FA)")
    print(f"  phase_gain LR mult: {PHASE_GAIN_LR_MULT}×")
    print(f"  Checkpoint strategy: {CHECKPOINT_STRATEGY}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Dataset
    dataset_path = find_file(DATASET_CANDIDATES)
    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset not found. Tried: {DATASET_CANDIDATES}\n"
            f"For FineWeb-only: DWARF_DATASET=fineweb python3 {__file__}")
    train_data, val_data = load_dataset(
        dataset_path, MAX_TRAIN_SEQS, MAX_VAL_SEQS, MAX_SEQ_LEN)

    # Tokenizer
    tok_path = find_file(TOKENIZER_CANDIDATES)
    tokenizer = None
    if tok_path:
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        except Exception as e:
            print(f"  WARNING: tokenizer load failed ({e}), passkey eval disabled")

    # Model
    model = PureDSQGModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
    ).to(device)
    n_params = model.n_params()
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  scale_embed LR mult: {SCALE_EMBED_LR_MULT}×  "
          f"(effective LR: {LR * SCALE_EMBED_LR_MULT:.2e})")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    steps_per_epoch = math.ceil(len(train_data) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * SCREEN_EPOCHS
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | "
          f"eff_batch: {BATCH_SIZE * GRAD_ACCUM}")

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, total_steps)

    best_val_loss = float('inf')
    global_step   = 0

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        step_losses = deque(maxlen=50)
        epoch_start = time.time()
        optimizer.zero_grad()

        for step_in_epoch in range(steps_per_epoch):
            step_start = time.time()

            # Accumulate gradient over GRAD_ACCUM micro-batches
            for ga in range(GRAD_ACCUM):
                idx = ((step_in_epoch * GRAD_ACCUM + ga) * BATCH_SIZE)
                if idx >= len(train_data):
                    break
                batch = train_data[idx:idx+BATCH_SIZE].to(device)
                if batch.shape[0] == 0:
                    break
                inputs, targets = batch[:, :-1], batch[:, 1:]

                with _amp_context(device):
                    hidden = model.forward_hidden(inputs.long())  # int32 → int64 for embedding
                loss = chunked_ce_loss(model, hidden, targets) / GRAD_ACCUM
                loss.backward()

            # Gradient clip + step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            step_losses.append(loss.item() * GRAD_ACCUM)
            se_max = model.scale_embed_max
            elapsed = time.time() - step_start
            tokens_per_sec = (BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)) / elapsed

            if global_step % 100 == 0:
                print(f"  Ep{epoch} Step {step_in_epoch+1}/{steps_per_epoch} | "
                      f"Loss {sum(step_losses)/len(step_losses):.4f} | "
                      f"SE|max|={se_max:.4f} | "
                      f"{tokens_per_sec/1000:.0f}K tok/s")
                sys.stdout.flush()

        # Epoch-end evaluation
        print(f"\n  --- Ep{epoch} validation ---")
        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(val_loss)
        epoch_time = time.time() - epoch_start
        print(f"  Val loss: {val_loss:.4f}  PPL: {val_ppl:.2f}  "
              f"({epoch_time/60:.1f} min)")

        # Passkey eval
        if tokenizer is not None:
            pk_results, pk_mean = passkey_eval(
                model, tokenizer, device,
                PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE)
            dist_str = ' '.join(f'd={d}:{v:.0%}' for d, v in sorted(pk_results.items()))
            print(f"  Passkey: {pk_mean:.1%} mean | {dist_str}")

            # Early abort: if ep2 passkey is still random, log and note
            if epoch == 2 and pk_mean < 0.15:
                print(f"\n  ⚠️  ABLATION RESULT: Passkey random at ep2 ({pk_mean:.1%})")
                print(f"  → FA is operative mechanism; DSQG relay insufficient alone")
                print(f"  → Consider: DWARF_DATASET=fineweb for FineWeb-only retry")
                print(f"  → Or:       DWARF_PHASE_GAIN_LR_MULT=10 for phase_gain fix")
        else:
            pk_mean = None
            print("  Passkey: skipped (no tokenizer)")

        # Checkpoint
        se_max = model.scale_embed_max
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'passkey_mean': pk_mean,
            'scale_embed_max': se_max,
            'config': {
                'D': EMBEDDING_DIM, 'H': NUM_HEADS, 'L': NUM_LAYERS,
                'FFN': FFN_DIM, 'n_params': n_params,
                'pure_dsqg': True, 'has_fa': False,
                'dataset': _DATASET_KEY,
                'phase_gain_lr_mult': PHASE_GAIN_LR_MULT,
            },
        }
        resume_path = os.path.join(CHECKPOINT_DIR,
                                   f'{CKPT_BASE_NAME}_ep{epoch}_resume.pt')
        torch.save(ckpt, resume_path)
        print(f"  Saved: {resume_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt')
            eval_ckpt = {'model_state_dict': model.state_dict(),
                         'val_loss': val_loss, 'val_ppl': val_ppl,
                         'passkey_mean': pk_mean, 'epoch': epoch,
                         'config': ckpt['config']}
            torch.save(eval_ckpt, best_path)
            print(f"  ★ New best: {best_path}  (PPL {val_ppl:.2f})")

        print()

    print(f"\n{'='*60}")
    print(f"  Pure DSQG ablation complete")
    print(f"  Best val PPL: {math.exp(best_val_loss):.2f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    train()
