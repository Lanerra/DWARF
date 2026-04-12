"""
🧪 DWARF D=512 L=13 Triadic J=96 — phase_gain gradient balance smoke test

Architecture: D=512, H=8 (hd=64), L=13, FFN=1024, TIED lm_head, ~50M params
  Triadic [A,A,B,B,C,C] layout with 96 offsets split into 3 groups of 32
  FA@L4 (preIF@L3, 25% depth)

  L00: DSQGBlock(GROUP_A)  -- pair 1a
  L01: DSQGBlock(GROUP_A)  -- pair 1b
  L02: DSQGBlock(GROUP_B)  -- pair 2a
  L03: DSQGBlock(GROUP_B)  -- pair 2b + preIF
  L04: FullAttentionBlock  -- FA at 25% depth
  L05: DSQGBlock(GROUP_C)  -- pair 3a
  L06: DSQGBlock(GROUP_C)  -- pair 3b
  L07: DSQGBlock(GROUP_A)  -- pair 4a
  L08: DSQGBlock(GROUP_A)  -- pair 4b
  L09: DSQGBlock(GROUP_B)  -- pair 5a
  L10: DSQGBlock(GROUP_B)  -- pair 5b
  L11: DSQGBlock(GROUP_C)  -- pair 6a
  L12: DSQGBlock(GROUP_C)  -- pair 6b

Purpose: validate that the 50× LR optimizer group keeps the
phase_gain / phase_base gradient ratio near 1.0× (not 2400×).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_triadic_j96_phasegain_test_4090_bf16.py
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

import triton
from dsqg_attention_v8_dynamic_j import (
    ALL_OFFSETS as _ALL_96,
    _fwd_v8, _compute_D_v8, _bwd_dq_v8, _bwd_dkdv_v8,
    npci_rotate, R_PLANES, _next_pow2,
)
_USE_V13 = os.environ.get('DWARF_USE_V13', '1') == '1'
_V13Cls = None
if _USE_V13:
    try:
        from dsqg_attention_v13_dynamic_j import DSQGAttentionV13Dynamic as _V13Cls
        print('  Kernel: V13-dynamic-J (two-pass, faster)')
    except Exception as _e:
        print(f'  V13 import failed ({_e}), falling back to V8')
        _USE_V13 = False
_DSQG_TYPES = None
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

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

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 1024
NUM_LAYERS       = 13
FULL_ATTN_LAYER  = 4
VOCAB_SIZE       = 32768

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 20.0
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
LR        = 3e-4
DROPOUT   = 0.1

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '16'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '8'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '30000'))
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582
CE_CHUNK       = 512
SCREEN_EPOCHS  = int(os.environ.get('DWARF_EPOCHS', '3'))

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
CKPT_BASE_NAME    = 'd512_l13_triadic_j96_phasegain_test'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# LAYER LAYOUT: L=13, FA@L4, [A,A,B,B,C,C] pairs
# =============================================================================

LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00: pair 1a
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L01: pair 1b
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L02: pair 2a
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, True),    # L03: pair 2b + preIF
    ('FA', None, 0, 0, False),                       # L04: FullAttention @ 25% depth
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L05: pair 3a
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L06: pair 3b
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07: pair 4a
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L08: pair 4b
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L09: pair 5a
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L10: pair 5b
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L11: pair 6a
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L12: pair 6b
]

assert len(LAYER_LAYOUT) == NUM_LAYERS

# =============================================================================
# GROUPED AUTOGRAD FUNCTION (V8 fallback)
# =============================================================================

class _DSQGFnGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                offsets_dev, j_val, j_small, j_large):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape == (j_val, H)
        assert scale_embed.shape == (j_val, HD)
        assert phase_base.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert phase_gain.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert y_pre.shape == (B, H, N, R_PLANES)
        assert z_pre.shape == (B, H, N, R_PLANES)

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
        elif HD <= 256:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 32, 4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 32, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 16, 4, 2
        else:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 16, 4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 16, 4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 8, 4, 2

        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_v8[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
            offsets_dev,
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
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _num_warps
        ctx.num_stages = _num_stages
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
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

        D = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g = (B * H, triton.cdiv(N, BN))

        _compute_D_v8[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = (N + BN - 1) // BN
        _dev = q.device

        dq = torch.zeros_like(q)
        dy_pre = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B * H, blocks_n, j_val, device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, j_val * HD, device=_dev, dtype=torch.float32)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            blocks_n * j_val, j_val,
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            blocks_n * j_val * HD, j_val * HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(dim=(0, 1, 2)).contiguous()

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)

        if j_large > 0:
            phase_base_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                         device=_dev, dtype=torch.float32)
            phase_gain_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                         device=_dev, dtype=torch.float32)
        else:
            phase_base_buf = torch.empty(B * H, blocks_n, 1,
                                         device=_dev, dtype=torch.float32)
            phase_gain_buf = torch.empty(B * H, blocks_n, 1,
                                         device=_dev, dtype=torch.float32)

        stride_buf_bh = blocks_n * max(j_large, 1) * 2
        stride_buf_blk = max(j_large, 1) * 2

        _bwd_dkdv_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
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

        if j_large > 0:
            def _reduce_phase_buf(buf):
                r = buf.view(B, H, blocks_n, j_large, 2).sum(dim=(0, 2))
                return r.permute(1, 0, 2).contiguous()
            d_phase_base = _reduce_phase_buf(phase_base_buf)
            d_phase_gain = _reduce_phase_buf(phase_gain_buf)
        else:
            d_phase_base = torch.zeros_like(phase_base)
            d_phase_gain = torch.zeros_like(phase_gain)

        return (dq, dk, dv, dpb, dse, d_phase_base, d_phase_gain,
                dy_pre, dz_pre, None, None, None, None)


def dsqg_attention_grouped(q, k, v, pos_bias, scale_embed,
                           phase_base, phase_gain, y_pre, z_pre,
                           offsets_dev, j_val, j_small, j_large):
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnGrouped.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
        offsets_dev, j_val, j_small, j_large,
    )
    return out if orig == torch.bfloat16 else out.to(orig)


# =============================================================================
# GROUPED ATTENTION MODULE
# =============================================================================

class DSQGAttentionGrouped(nn.Module):
    def __init__(self, embedding_dim, num_heads, offsets, j_small, j_large,
                 seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        HD = self.head_dim
        j_val = len(offsets)
        self.j_val = j_val
        self.j_small = j_small
        self.j_large = j_large
        assert HD >= 4

        self.register_buffer(
            'offsets_dev',
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets],
                                  dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(j_val, HD))
        self.if_gain = nn.Parameter(torch.ones(num_heads))

        self.phase_base = nn.Parameter(torch.zeros(max(j_large, 1), num_heads, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(max(j_large, 1), num_heads, R_PLANES))
        self.phase_gate = nn.Parameter(torch.zeros(max(j_large, 1)))
        self.content_layer_norm = nn.LayerNorm(R_PLANES, elementwise_affine=False)

        # Non-zero init required: LayerNorm(zeros)=zeros, starving phase_gain from step 0
        self.query_probes = nn.Parameter(torch.randn(R_PLANES, HD) * 0.02)
        self.key_probes = nn.Parameter(torch.randn(R_PLANES, HD) * 0.02)

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        q_norm = F.normalize(q.float(), dim=-1)
        k_norm = F.normalize(k.float(), dim=-1)
        qp_norm = F.normalize(self.query_probes.float(), dim=-1)
        kp_norm = F.normalize(self.key_probes.float(), dim=-1)
        y_pre_raw = torch.einsum('bhnd,rd->bhnr', q_norm, qp_norm)
        z_pre_raw = torch.einsum('bhnd,rd->bhnr', k_norm, kp_norm)
        # V14 LayerNorm fix: normalize the product to cure phase_gain gradient starvation
        y_pre = self.content_layer_norm(y_pre_raw * z_pre_raw).contiguous()
        z_pre = torch.ones_like(y_pre)  # kernel computes phase_gain * y_pre * z_pre; z_pre=1 avoids double-scaling

        gate = torch.sigmoid(self.phase_gate)[:, None, None]
        gated_phase_base = self.phase_base * gate
        gated_phase_gain = self.phase_gain * gate

        out = dsqg_attention_grouped(
            q, k, v,
            self.pos_bias, self.scale_embed,
            gated_phase_base, gated_phase_gain,
            y_pre, z_pre,
            self.offsets_dev, self.j_val, self.j_small, self.j_large,
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))


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
        global _DSQG_TYPES
        if _DSQG_TYPES is None:
            _DSQG_TYPES = (DSQGAttentionGrouped,) + ((_V13Cls,) if _USE_V13 else ())
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
            if isinstance(m, _DSQG_TYPES):
                yield m.scale_embed

    def phase_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.phase_gain
                yield m.phase_gate
                yield m.query_probes
                yield m.key_probes

    def non_scale_embed_parameters(self):
        exclude_ids = {id(p) for p in self.scale_embed_parameters()}
        exclude_ids.update(id(p) for p in self.phase_parameters())
        for p in self.parameters():
            if id(p) not in exclude_ids:
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
                parts.append(f'L{i}:DSQG-{label}(J={j}){iflag}')
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
    print('  🧪 DWARF D512-L13 Triadic J=96 [A,A,B,B,C,C] — phase_gain gradient balance test')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) C(J={len(GROUP_C)})')
    print(f'  Bandwidth ratio: {(len(GROUP_A)*64)/EMBEDDING_DIM:.1f}x  (safe ≤ 3.0x)')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print(f'  Using chunked CE (Liger disabled)')
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

    # Resume support: DWARF_RESUME=path loads model+optimizer+epoch state
    _resume_path = os.environ.get('DWARF_RESUME', '')
    _start_epoch = 1
    if _resume_path and os.path.exists(_resume_path):
        print(f'  Resuming from {_resume_path}')
        _ckpt = torch.load(_resume_path, map_location=device)
        model.load_state_dict(_ckpt['model_state_dict'])
        _start_epoch = _ckpt.get('epoch', 0) + 1
        # content_active is a persistent buffer — already correct from state_dict.
        # Read it back to sync the training-loop flag.
        _ca = any(getattr(_m, 'content_active', torch.tensor(False)).item()
                  for _m in model.modules() if hasattr(_m, 'content_active'))
        print(f'  Resuming from epoch {_start_epoch}, content_active={_ca}')
        del _ckpt

    scale_embed_params = list(model.scale_embed_parameters())
    phase_params = list(model.phase_parameters())
    other_params = list(model.non_scale_embed_parameters())
    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': other_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
        {'params': phase_params, 'lr': LR, 'name': 'phase'},
    ], weight_decay=0.1, betas=(0.9, 0.95))
    print(f'  phase params LR: {LR:.2e} (1× base — LayerNorm fix active)')

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)

    def _lr_lambda(step, group_idx):
        if group_idx == 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda s: _lr_lambda(s, 0), lambda s: _lr_lambda(s, 1),
                   lambda s: _lr_lambda(s, 2)])

    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}
    content_signal_active = _ca if _resume_path else False  # restored from checkpoint or starts False

    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)

    for epoch in range(_start_epoch, SCREEN_EPOCHS + _start_epoch):
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

            # Log phase_gain/phase_base gradient ratio BEFORE zero_grad clears them
            if step % 100 == 0:
                _pg, _pb = [], []
                for _n, _p in model.named_parameters():
                    if _p.grad is not None:
                        if 'phase_gain' in _n: _pg.append(_p.grad.norm().item())
                        elif 'phase_base' in _n: _pb.append(_p.grad.norm().item())
                if _pg and _pb:
                    _ratio = (sum(_pg)/len(_pg)) / (sum(_pb)/len(_pb) + 1e-10)
                    print(f'    phase_gain/phase_base grad ratio: {_ratio:.3f}x (target: ~1.0x)', flush=True)
                else:
                    print(f'    phase_gain/phase_base: no grads found (pg={len(_pg)} pb={len(_pb)})', flush=True)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            step_ms = (time.time() - t0) * 1000
            step_times.append(step_ms)

            if step % 100 == 0:
                se_vals = [m.scale_embed.detach().abs()
                           for m in model.modules() if isinstance(m, _DSQG_TYPES)]
                se_max = torch.cat(se_vals).max().item() if se_vals else 0.0

                # Percolation trigger: activate content signal once SE|max| crosses 2.0
                if se_max >= 2.0 and not content_signal_active:
                    content_signal_active = True
                    for m in model.modules():
                        if hasattr(m, 'content_active'):
                            m.content_active = torch.tensor(True)
                    print(f'  🎯 Ep{epoch} Step {step}: SE|max|={se_max:.4f} — PERCOLATION CROSSED, MOVT content signal ACTIVATED', flush=True)

                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000.0)
                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} | SE|max|={se_max:.4f} '
                      f'| {tok_s:.0f} tok/s', flush=True)

                phase_gain_norms = []
                phase_base_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if "phase_gain" in name:
                            phase_gain_norms.append(param.grad.norm().item())
                        elif "phase_base" in name:
                            phase_base_norms.append(param.grad.norm().item())
                if phase_gain_norms and phase_base_norms:
                    pg_mean = sum(phase_gain_norms) / len(phase_gain_norms)
                    pb_mean = sum(phase_base_norms) / len(phase_base_norms)
                    ratio = pg_mean / (pb_mean + 1e-10)
                    print(f'    phase_gain grad norm (mean): {pg_mean:.6f}')
                    print(f'    phase_base grad norm (mean): {pb_mean:.6f}')
                    print(f'    phase_gain/phase_base grad ratio: {ratio:.3f}x (target: ~1.0x)')

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
                   for m in model.modules() if isinstance(m, _DSQG_TYPES)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        resume_state = {
            'model_state_dict': {k.replace('._orig_mod', ''): v
                                 for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'content_signal_active': content_signal_active,
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

    print('\n' + '=' * 70)
    print(f'passkey_ep1: {passkey_results.get(1, 0.0):.1f}')
    print(f'ppl_ep1: {ppl_results.get(1, 999.0):.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_params / 1e6:.1f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init: {EMA_INIT}')
    print(f'description: phase_gain smoke test — Triadic J=96 [A,A,B,B,C,C] L=13 '
          f'D={EMBEDDING_DIM} H={NUM_HEADS} FFN={FFN_DIM} FA@L{FULL_ATTN_LAYER} '
          f'preIF@L{FULL_ATTN_LAYER-1} 50x phase LR')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
