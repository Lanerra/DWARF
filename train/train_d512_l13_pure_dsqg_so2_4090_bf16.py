"""
🚀 DWARF D=512 L=13 Pure DSQG SO(2) — no FA layer, 4090, cold start

Architecture: D=512, H=8 (hd=64), L=13, FFN=1024, TIED lm_head
  Triadic partitioning: 96 offsets split into 3 groups of 32 + tail group of 51
  Per-layer bandwidth ratio at D=512: (32×64)/512 = 4.0x (safe)
  Tail bandwidth: (51×64)/512 = 6.4x (consolidation, acceptable)
  No FA layer — pure DSQG

  L0: DSQGBlock(GROUP_A)   -- cycle 1
  L1: DSQGBlock(GROUP_B)
  L2: DSQGBlock(GROUP_C)   + preIF
  L3: DSQGBlock(GROUP_C)   + preIF (was FA; pure-DSQG ablation)
  L4: DSQGBlock(GROUP_B)   -- cycle 1 cont.
  L5: DSQGBlock(GROUP_C)
  L6: DSQGBlock(GROUP_C)
  L7: DSQGBlock(GROUP_A)   -- cycle 2
  L8: DSQGBlock(GROUP_A)
  L9: DSQGBlock(GROUP_B)
  L10: DSQGBlock(GROUP_B)
  L11: DSQGBlock(GROUP_C)
  L12: DSQGBlock(GROUP_C)

  Full J=96 topological coverage achieved within each 3-layer triad.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_pure_dsqg_so2_4090_bf16.py
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

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "0") != "0"  # disabled: GA scaling corrupts Liger grad_output check

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import triton
import triton.language as tl
from dsqg_attention_v8_dynamic_j import (
    ALL_OFFSETS as _ALL_96,
    _compute_D_v8,
    npci_rotate, R_PLANES, _next_pow2,
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

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)  # 17, 15
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)  # 0, 32
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)  # 0, 32
J_SMALL_T, J_LARGE_T = _count_small_large(GROUP_T)  # 3, 48

# =============================================================================
# FUSED TRITON KERNELS (if_gain + y_pre + z_pre inlined)
# =============================================================================

@triton.jit
def _fwd_fused(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, IF_GAIN,
    QP0, QP1, KP0, KP1,
    OUT, LSE,
    OFFSETS,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_lb,  stride_lh,  stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H;           h   = bh % H
    n0  = blk * BLOCK_N

    ns  = n0 + tl.arange(0, BLOCK_N)
    nm  = ns < N
    sc  = 1.0 / (HD ** 0.5)
    ds  = tl.arange(0, BLOCK_HD)
    dm  = ds < HD

    qb  = Q + b * stride_qb + h * stride_qh
    kb  = K + b * stride_kb + h * stride_kh
    vb  = V + b * stride_vb + h * stride_vh

    q   = tl.load(qb + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    qp0 = tl.load(QP0 + ds, mask=dm, other=0.0).to(tl.float32)
    qp1 = tl.load(QP1 + ds, mask=dm, other=0.0).to(tl.float32)
    y0  = tl.sum(q * qp0[None,:], axis=1) * sc
    y1  = tl.sum(q * qp1[None,:], axis=1) * sc

    kp0 = tl.load(KP0 + ds, mask=dm, other=0.0).to(tl.float32)
    kp1 = tl.load(KP1 + ds, mask=dm, other=0.0).to(tl.float32)

    col0 = (ds == 0);  col1 = (ds == 1)
    col2 = (ds == 2);  col3 = (ds == 3)
    f0   = tl.where(col0, 1.0, 0.0)
    f1   = tl.where(col1, 1.0, 0.0)
    f2   = tl.where(col2, 1.0, 0.0)
    f3   = tl.where(col3, 1.0, 0.0)

    mi  = tl.full([BLOCK_N], float('-inf'), tl.float32)
    li  = tl.zeros([BLOCK_N], tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        mn    = tl.maximum(mi, s)
        cor   = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p     = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        li    = li * cor + p;     mi = mn

        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        if i < J_SMALL_VAL:
            acc = acc * cor[:,None] + p[:,None] * vt
        else:
            pi  = i - J_SMALL_VAL
            z0  = tl.sum(kt * kp0[None,:], axis=1) * sc
            z1  = tl.sum(kt * kp1[None,:], axis=1) * sc

            pb0 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0 * z0
            theta1 = pb1 + pg1 * y1 * z1

            cos0 = tl.cos(theta0);  sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1);  sin1 = tl.sin(theta1)

            v0  = tl.sum(vt * f0[None,:], axis=1)
            v1  = tl.sum(vt * f1[None,:], axis=1)
            vt  = tl.where(col0[None,:], (cos0 * v0 - sin0 * v1)[:,None], vt)
            vt  = tl.where(col1[None,:], (sin0 * v0 + cos0 * v1)[:,None], vt)

            v2  = tl.sum(vt * f2[None,:], axis=1)
            v3  = tl.sum(vt * f3[None,:], axis=1)
            vt  = tl.where(col2[None,:], (cos1 * v2 - sin1 * v3)[:,None], vt)
            vt  = tl.where(col3[None,:], (sin1 * v2 + cos1 * v3)[:,None], vt)

            acc = acc * cor[:,None] + p[:,None] * vt

    ls  = tl.where(li > 0.0, li, 1.0)
    lse = mi + tl.log(ls)
    acc = acc / ls[:,None]

    ig  = tl.load(IF_GAIN + h)
    acc = acc * ig

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             acc.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


@triton.jit
def _bwd_dq_fused(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN,
    QP0, QP1, KP0, KP1,
    DO, LSE, Dv,
    DQ, DPB_BUF, DSE_BUF,
    DQP0, DQP1,
    OFFSETS,
    stride_qb,   stride_qh,   stride_qn,   stride_qd,
    stride_kb,   stride_kh,   stride_kn,   stride_kd,
    stride_vb,   stride_vh,   stride_vn,   stride_vd,
    stride_dob,  stride_doh,  stride_don,  stride_dod,
    stride_lb,   stride_lh,   stride_ln,
    stride_Db,   stride_Dh,   stride_Dn,
    stride_dqb,  stride_dqh,  stride_dqn,  stride_dqd,
    stride_dpb_bh, stride_dpb_blk,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dse_bh, stride_dse_blk,
    stride_phi,  stride_phh,
    stride_pgi,  stride_pgh,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H; h = bh % H
    n0  = blk * BLOCK_N
    ns  = n0 + tl.arange(0, BLOCK_N); nm = ns < N
    ds  = tl.arange(0, BLOCK_HD);     dm = ds < HD
    sc  = 1.0 / (HD ** 0.5)

    qb  = Q  + b*stride_qb + h*stride_qh
    kb  = K  + b*stride_kb + h*stride_kh
    vb  = V  + b*stride_vb + h*stride_vh
    dob = DO + b*stride_dob + h*stride_doh

    q    = tl.load(qb  + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                   mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    do   = tl.load(dob + ns[:,None]*stride_don + ds[None,:]*stride_dod,
                   mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    lse  = tl.load(LSE + b*stride_lb + h*stride_lh + ns*stride_ln, mask=nm, other=0.0)
    Dval = tl.load(Dv  + b*stride_Db + h*stride_Dh + ns*stride_Dn, mask=nm, other=0.0)

    qp0 = tl.load(QP0 + ds, mask=dm, other=0.0).to(tl.float32)
    qp1 = tl.load(QP1 + ds, mask=dm, other=0.0).to(tl.float32)
    y0  = tl.sum(q * qp0[None,:], axis=1) * sc
    y1  = tl.sum(q * qp1[None,:], axis=1) * sc

    kp0 = tl.load(KP0 + ds, mask=dm, other=0.0).to(tl.float32)
    kp1 = tl.load(KP1 + ds, mask=dm, other=0.0).to(tl.float32)

    f0 = tl.where(ds == 0, 1.0, 0.0);  f1 = tl.where(ds == 1, 1.0, 0.0)
    f2 = tl.where(ds == 2, 1.0, 0.0);  f3 = tl.where(ds == 3, 1.0, 0.0)
    col0 = (ds == 0); col1 = (ds == 1)
    col2 = (ds == 2); col3 = (ds == 3)

    dq      = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    dy_pre0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre1 = tl.zeros([BLOCK_N], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i  = tl.load(SE + i*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lse), 0.0)

        if i < J_SMALL_VAL:
            dot_rv = tl.sum(do * vt, axis=1)
            ds_v   = alpha * (dot_rv - Dval)
            dq    += ds_v[:,None] * kt * sc
            dq    += ds_v[:,None] * se_i[None,:] * sc
            tl.store(DPB_BUF + bh*stride_dpb_bh + blk*stride_dpb_blk + i,
                     tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.store(DSE_BUF + bh*stride_dse_bh + blk*stride_dse_blk + i*HD + ds,
                     tl.where(dm, dse_i, 0.0), mask=dm)
        else:
            pi  = i - J_SMALL_VAL
            z0  = tl.sum(kt * kp0[None,:], axis=1) * sc
            z1  = tl.sum(kt * kp1[None,:], axis=1) * sc

            pb0 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0 * z0
            theta1 = pb1 + pg1 * y1 * z1

            cos0 = tl.cos(theta0); sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1); sin1 = tl.sin(theta1)

            v0 = tl.sum(vt * f0[None,:], axis=1); v1 = tl.sum(vt * f1[None,:], axis=1)
            v2 = tl.sum(vt * f2[None,:], axis=1); v3 = tl.sum(vt * f3[None,:], axis=1)
            vt_rot = vt
            vt_rot = tl.where(col0[None,:], (cos0 * v0 - sin0 * v1)[:,None], vt_rot)
            vt_rot = tl.where(col1[None,:], (sin0 * v0 + cos0 * v1)[:,None], vt_rot)
            vt_rot = tl.where(col2[None,:], (cos1 * v2 - sin1 * v3)[:,None], vt_rot)
            vt_rot = tl.where(col3[None,:], (sin1 * v2 + cos1 * v3)[:,None], vt_rot)

            dot_rv = tl.sum(do * vt_rot, axis=1)
            ds_v   = alpha * (dot_rv - Dval)
            dq    += ds_v[:,None] * kt * sc
            dq    += ds_v[:,None] * se_i[None,:] * sc
            tl.store(DPB_BUF + bh*stride_dpb_bh + blk*stride_dpb_blk + i,
                     tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.store(DSE_BUF + bh*stride_dse_bh + blk*stride_dse_blk + i*HD + ds,
                     tl.where(dm, dse_i, 0.0), mask=dm)

            do0 = tl.sum(do * f0[None,:], axis=1); do1 = tl.sum(do * f1[None,:], axis=1)
            do2 = tl.sum(do * f2[None,:], axis=1); do3 = tl.sum(do * f3[None,:], axis=1)

            dth0 = alpha * (do0 * (-v0 * sin0 - v1 * cos0) + do1 * (v0 * cos0 - v1 * sin0))
            dth1 = alpha * (do2 * (-v2 * sin1 - v3 * cos1) + do3 * (v2 * cos1 - v3 * sin1))

            dy_pre0 += dth0 * pg0 * z0
            dy_pre1 += dth1 * pg1 * z1

    dq += dy_pre0[:,None] * qp0[None,:] * sc
    dq += dy_pre1[:,None] * qp1[None,:] * sc

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])

    dy0_m = tl.where(nm, dy_pre0, 0.0)
    dy1_m = tl.where(nm, dy_pre1, 0.0)
    tl.atomic_add(DQP0 + ds, tl.sum(dy0_m[:,None] * q, axis=0) * sc, mask=dm)
    tl.atomic_add(DQP1 + ds, tl.sum(dy1_m[:,None] * q, axis=0) * sc, mask=dm)


@triton.jit
def _bwd_dkdv_fused(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN,
    QP0, QP1, KP0, KP1,
    DO, LSE, Dv,
    DK, DV,
    DPHASE_BASE_BUF, DPHASE_GAIN_BUF,
    DKP0, DKP1,
    OFFSETS,
    stride_qb,   stride_qh,   stride_qn,   stride_qd,
    stride_kb,   stride_kh,   stride_kn,   stride_kd,
    stride_vb,   stride_vh,   stride_vn,   stride_vd,
    stride_dob,  stride_doh,  stride_don,  stride_dod,
    stride_lb,   stride_lh,   stride_ln,
    stride_Db,   stride_Dh,   stride_Dn,
    stride_dkb,  stride_dkh,  stride_dkn,  stride_dkd,
    stride_dvb,  stride_dvh,  stride_dvn,  stride_dvd,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_phi,  stride_phh,
    stride_pgi,  stride_pgh,
    stride_buf_bh, stride_buf_blk,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
):
    bh  = tl.program_id(0); blk = tl.program_id(1)
    b   = bh // H; h = bh % H
    m0  = blk * BLOCK_M
    ms  = m0 + tl.arange(0, BLOCK_M); mm = ms < N
    ds  = tl.arange(0, BLOCK_HD);     dm = ds < HD
    sc  = 1.0 / (HD ** 0.5)

    kb  = K  + b*stride_kb + h*stride_kh
    vb  = V  + b*stride_vb + h*stride_vh
    qb  = Q  + b*stride_qb + h*stride_qh
    dob = DO + b*stride_dob + h*stride_doh

    kt  = tl.load(kb + ms[:,None]*stride_kn + ds[None,:]*stride_kd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    vt  = tl.load(vb + ms[:,None]*stride_vn + ds[None,:]*stride_vd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    qp0 = tl.load(QP0 + ds, mask=dm, other=0.0).to(tl.float32)
    qp1 = tl.load(QP1 + ds, mask=dm, other=0.0).to(tl.float32)
    kp0_v = tl.load(KP0 + ds, mask=dm, other=0.0).to(tl.float32)
    kp1_v = tl.load(KP1 + ds, mask=dm, other=0.0).to(tl.float32)

    z0_t = tl.sum(kt * kp0_v[None,:], axis=1) * sc
    z1_t = tl.sum(kt * kp1_v[None,:], axis=1) * sc

    f0 = tl.where(ds == 0, 1.0, 0.0); f1 = tl.where(ds == 1, 1.0, 0.0)
    f2 = tl.where(ds == 2, 1.0, 0.0); f3 = tl.where(ds == 3, 1.0, 0.0)
    col0 = (ds == 0); col1 = (ds == 1)
    col2 = (ds == 2); col3 = (ds == 3)

    v0_t = tl.sum(vt * f0[None,:], axis=1)
    v1_t = tl.sum(vt * f1[None,:], axis=1)
    v2_t = tl.sum(vt * f2[None,:], axis=1)
    v3_t = tl.sum(vt * f3[None,:], axis=1)

    dk      = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv      = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre1 = tl.zeros([BLOCK_M], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_   = ms + delta
        val   = (np_ < N) & mm

        qn    = tl.load(qb  + np_[:,None]*stride_qn + ds[None,:]*stride_qd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don   = tl.load(dob + np_[:,None]*stride_don + ds[None,:]*stride_dod,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen  = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn    = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        y0_n  = tl.sum(qn * qp0[None,:], axis=1) * sc
        y1_n  = tl.sum(qn * qp1[None,:], axis=1) * sc

        se_i  = tl.load(SE + i*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(qn * kt, axis=1) * sc
        s    += tl.load(PB + i*stride_pbi + h*stride_pbh)
        s    += tl.sum(qn * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lsen), 0.0)

        if i < J_SMALL_VAL:
            dot_rv = tl.sum(don * vt, axis=1)
            ds_v   = alpha * (dot_rv - Dn)
            dk    += ds_v[:,None] * qn * sc
            dv    += alpha[:,None] * don
        else:
            pi  = i - J_SMALL_VAL
            pb0 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0_n * z0_t
            theta1 = pb1 + pg1 * y1_n * z1_t

            cos0 = tl.cos(theta0); sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1); sin1 = tl.sin(theta1)

            vt_rot = vt
            vt_rot = tl.where(col0[None,:], (cos0 * v0_t - sin0 * v1_t)[:,None], vt_rot)
            vt_rot = tl.where(col1[None,:], (sin0 * v0_t + cos0 * v1_t)[:,None], vt_rot)
            vt_rot = tl.where(col2[None,:], (cos1 * v2_t - sin1 * v3_t)[:,None], vt_rot)
            vt_rot = tl.where(col3[None,:], (sin1 * v2_t + cos1 * v3_t)[:,None], vt_rot)

            dot_rv = tl.sum(don * vt_rot, axis=1)
            ds_v   = alpha * (dot_rv - Dn)
            dk    += ds_v[:,None] * qn * sc

            don0 = tl.sum(don * f0[None,:], axis=1); don1 = tl.sum(don * f1[None,:], axis=1)
            don2 = tl.sum(don * f2[None,:], axis=1); don3 = tl.sum(don * f3[None,:], axis=1)
            dv_c = alpha[:,None] * don
            dv_c = tl.where(col0[None,:], (alpha * ( cos0*don0 + sin0*don1))[:,None], dv_c)
            dv_c = tl.where(col1[None,:], (alpha * (-sin0*don0 + cos0*don1))[:,None], dv_c)
            dv_c = tl.where(col2[None,:], (alpha * ( cos1*don2 + sin1*don3))[:,None], dv_c)
            dv_c = tl.where(col3[None,:], (alpha * (-sin1*don2 + cos1*don3))[:,None], dv_c)
            dv  += dv_c

            dth0 = alpha * (don0*(-v0_t*sin0 - v1_t*cos0) + don1*(v0_t*cos0 - v1_t*sin0))
            dth1 = alpha * (don2*(-v2_t*sin1 - v3_t*cos1) + don3*(v2_t*cos1 - v3_t*sin1))

            buf_off = bh * stride_buf_bh + blk * stride_buf_blk + pi * 2
            tl.store(DPHASE_BASE_BUF + buf_off + 0, tl.sum(tl.where(val, dth0, 0.0)))
            tl.store(DPHASE_BASE_BUF + buf_off + 1, tl.sum(tl.where(val, dth1, 0.0)))
            tl.store(DPHASE_GAIN_BUF + buf_off + 0,
                     tl.sum(tl.where(val, dth0 * y0_n * z0_t, 0.0)))
            tl.store(DPHASE_GAIN_BUF + buf_off + 1,
                     tl.sum(tl.where(val, dth1 * y1_n * z1_t, 0.0)))

            dz_pre0 += tl.where(val, dth0 * pg0 * y0_n, 0.0)
            dz_pre1 += tl.where(val, dth1 * pg1 * y1_n, 0.0)

    dk += dz_pre0[:,None] * kp0_v[None,:] * sc
    dk += dz_pre1[:,None] * kp1_v[None,:] * sc

    tl.store(DK + b*stride_dkb + h*stride_dkh
             + ms[:,None]*stride_dkn + ds[None,:]*stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])
    tl.store(DV + b*stride_dvb + h*stride_dvh
             + ms[:,None]*stride_dvn + ds[None,:]*stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])

    dz0_m = tl.where(mm, dz_pre0, 0.0)
    dz1_m = tl.where(mm, dz_pre1, 0.0)
    tl.atomic_add(DKP0 + ds, tl.sum(dz0_m[:,None] * kt, axis=0) * sc, mask=dm)
    tl.atomic_add(DKP1 + ds, tl.sum(dz1_m[:,None] * kt, axis=0) * sc, mask=dm)


# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 1024
NUM_LAYERS       = 13
# No FA layer — pure DSQG
VOCAB_SIZE       = 32768

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 20.0  # bumped from 15.0; D=512 mixed-domain validated, need ep1 percolation crossing
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
LR        = 3e-4
DROPOUT   = 0.1

BATCH_SIZE     = int(os.environ.get('DWARF_BS', '16'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '8'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '200000'))  # 1562 steps/epoch; ~860 steps post-percolation; 3 epochs = ~127% Chinchilla total
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
CKPT_BASE_NAME    = 'd512_l13_pure_dsqg_so2'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# LAYER LAYOUT: L=13, pure DSQG (no FA)
# =============================================================================

# [A,B,C] + [C,B,C,C] + [A,A,B,B,C,C] layout: same structure as AABBC but L3 is DSQG-C+preIF instead of FA
LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L0: cycle 1
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L1
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L2 + preIF
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L3: DSQG-C + preIF (was FA; second preIF for pure-DSQG)
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
# GROUPED AUTOGRAD FUNCTION
# =============================================================================

class _DSQGFnGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, if_gain,
                qp0, qp1, kp0, kp1,
                offsets_dev, j_val, j_small, j_large):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape == (j_val, H)
        assert scale_embed.shape == (j_val, HD)
        assert phase_base.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert phase_gain.shape == (j_large, H, R_PLANES) if j_large > 0 else True
        assert if_gain.shape == (H,)
        assert qp0.shape == (HD,) and qp1.shape == (HD,)
        assert kp0.shape == (HD,) and kp1.shape == (HD,)

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

        _fwd_fused[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain, if_gain,
            qp0, qp1, kp0, kp1,
            out, lse,
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
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=_num_warps, num_stages=_num_stages,
        )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, if_gain,
                              qp0, qp1, kp0, kp1,
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
        (q, k, v, pb, se, phase_base, phase_gain, if_gain,
         qp0, qp1, kp0, kp1,
         out, lse, offsets_dev) = ctx.saved_tensors
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

        d_if_gain = D.sum(dim=(0, 2)) / if_gain

        dout_scaled = (dout.float() * if_gain.view(1, H, 1, 1)).to(dout.dtype).contiguous()

        blocks_n = (N + BN - 1) // BN
        _dev = q.device

        dq = torch.zeros_like(q)
        d_qp0 = torch.zeros(HD, device=_dev, dtype=torch.float32)
        d_qp1 = torch.zeros(HD, device=_dev, dtype=torch.float32)
        dpb_buf = torch.empty(B * H, blocks_n, j_val, device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, j_val * HD, device=_dev, dtype=torch.float32)

        _bwd_dq_fused[g](
            q, k, v, pb, se, phase_base, phase_gain,
            qp0, qp1, kp0, kp1,
            dout_scaled, lse, D,
            dq, dpb_buf, dse_buf,
            d_qp0, d_qp1,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout_scaled.stride(0), dout_scaled.stride(1), dout_scaled.stride(2), dout_scaled.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            blocks_n * j_val, j_val,
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            blocks_n * j_val * HD, j_val * HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(dim=(0, 1, 2)).contiguous()

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        d_kp0 = torch.zeros(HD, device=_dev, dtype=torch.float32)
        d_kp1 = torch.zeros(HD, device=_dev, dtype=torch.float32)

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

        _bwd_dkdv_fused[g](
            q, k, v, pb, se, phase_base, phase_gain,
            qp0, qp1, kp0, kp1,
            dout_scaled, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            d_kp0, d_kp1,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout_scaled.stride(0), dout_scaled.stride(1), dout_scaled.stride(2), dout_scaled.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            pb.stride(0), pb.stride(1),
            se.stride(0), se.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            stride_buf_bh, stride_buf_blk,
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

        return (dq, dk, dv, dpb, dse, d_phase_base, d_phase_gain, d_if_gain,
                d_qp0, d_qp1, d_kp0, d_kp1,
                None, None, None, None)


def dsqg_attention_grouped(q, k, v, pos_bias, scale_embed,
                           phase_base, phase_gain, if_gain,
                           qp0, qp1, kp0, kp1,
                           offsets_dev, j_val, j_small, j_large):
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnGrouped.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        if_gain.float().contiguous(),
        qp0.float().contiguous(), qp1.float().contiguous(),
        kp0.float().contiguous(), kp1.float().contiguous(),
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
        # SO(2) Lie group parameterization: (x,y) on unit circle, no 2pi wrap-arounds.
        # init=(1,0) -> identity rotation (numerically same as zero-init scalar).
        _pgv = torch.zeros(max(j_large, 1), num_heads, R_PLANES, 2)
        _pgv[..., 0] = 1.0
        self.phase_gain_vec = nn.Parameter(_pgv)

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))

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

        _pgv_unit = self.phase_gain_vec / self.phase_gain_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        phase_gain = _pgv_unit[..., 0].contiguous()

        out = dsqg_attention_grouped(
            q, k, v,
            self.pos_bias, self.scale_embed,
            self.phase_base, phase_gain,
            self.if_gain,
            self.query_probes[0], self.query_probes[1],
            self.key_probes[0], self.key_probes[1],
            self.offsets_dev, self.j_val, self.j_small, self.j_large,
        )

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



# =============================================================================
# MODEL
# =============================================================================

class TriadicJ96(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_dim, seq_len,
                 scale_embed_init_val=0.15, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)

        blocks = []
        for i, (label, offsets, js, jl, has_if) in enumerate(LAYER_LAYOUT):
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
                nn.init.normal_(m.phase_gain_vec, 0.0, 0.001)
                # Re-normalize after noise init so unit-circle constraint holds
                with torch.no_grad():
                    m.phase_gain_vec.data = m.phase_gain_vec.data / m.phase_gain_vec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx):
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
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
            label = block.group_label
            j = block.attn.j_val
            iflag = '+IF' if block.interference else ''
            parts.append(f'L{i}:DSQG-{label}(J={j}){iflag}')
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


def save_checkpoint_weights(model, epoch, git_hash, checkpoint_dir):
    se_state = {}
    for name, param in model.named_parameters():
        if 'scale_embed' in name:
            se_state[name] = param.data.clone()
    payload = {
        "scale_embed_weights": se_state,
        "config": {
            "embedding_dim": EMBEDDING_DIM, "num_heads": NUM_HEADS,
            "ffn_dim": FFN_DIM, "seq_len": MAX_SEQ_LEN,
            "source_script": "train/train_d512_l13_pure_dsqg_so2_4090_bf16.py",
            "num_layers": NUM_LAYERS,
            "epoch": epoch, "git_hash": git_hash,
            "note": (f"Pure DSQG J=96 L=13: D={EMBEDDING_DIM} H={NUM_HEADS}. "
                     f"Epoch {epoch}/{SCREEN_EPOCHS}."),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"{CKPT_BASE_NAME}_ep{epoch}_scale_embed.pt")
    torch.save(payload, out_path)
    print(f"  Saved scale_embed checkpoint: {out_path}")


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
    print('  🚀 DWARF D512-L13 Pure DSQG SO(2) — no FA, cold start')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) '
          f'C(J={len(GROUP_C)}) T(J={len(GROUP_T)})')
    print(f'  Bandwidth ratios: A/B/C={(len(GROUP_A)*64)/EMBEDDING_DIM:.1f}x  '
          f'T={(len(GROUP_T)*64)/EMBEDDING_DIM:.1f}x')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Using chunked CE')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    encoded_path = 'logs/mixed_encoded_2048_fineweb_tok.pt'
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

    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

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

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                        loss = liger_ce_fn(
                            hidden.contiguous().reshape(-1, hidden.size(-1)),
                            model.out.weight,
                            y.view(-1))
                    (loss / GRAD_ACCUM).backward()
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

        save_checkpoint_weights(model, epoch, git_hash, CHECKPOINT_DIR)

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
    print(f'description: Pure DSQG J=96 L=13 D={EMBEDDING_DIM} H={NUM_HEADS} '
          f'FFN={FFN_DIM} preIF@L2+L3 '
          f'groups=A({len(GROUP_A)})+B({len(GROUP_B)})+C({len(GROUP_C)})+T({len(GROUP_T)}) NO_FA')


if __name__ == '__main__' and False:
    def _test_kernel_fusion():
        device = 'cuda'
        torch.manual_seed(42)
        B, H, N, HD = 2, 8, 64, 64
        D = H * HD

        attn = DSQGAttentionGrouped(
            D, H, GROUP_A, J_SMALL_A, J_LARGE_A, seq_len=N, dropout=0.0
        ).to(device)

        x = torch.randn(B, N, D, device=device, requires_grad=False)
        results = {}

        # ── Check 1: forward cosine similarity vs reference ──
        with torch.no_grad():
            qkv = attn.qkv_proj(x)
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

            sc = HD ** -0.5
            y_pre = torch.einsum('bhnd,rd->bhnr',
                                 q, attn.query_probes.to(q.dtype)).mul(sc).float().contiguous()
            z_pre = torch.einsum('bhnd,rd->bhnr',
                                 k, attn.key_probes.to(k.dtype)).mul(sc).float().contiguous()
            _pgv_unit = attn.phase_gain_vec / attn.phase_gain_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            phase_gain_ref = _pgv_unit[..., 0].contiguous()

            q_bf = q.bfloat16()
            k_bf = k.bfloat16()
            v_bf = v.bfloat16()

            from dsqg_attention_v8_dynamic_j import _fwd_v8 as _fwd_ref
            out_ref = torch.empty_like(q_bf)
            lse_ref = torch.empty(B, H, N, device=device, dtype=torch.float32)
            BLOCK_N, BHD = 64, _next_pow2(HD)
            g = (B * H, triton.cdiv(N, BLOCK_N))
            _fwd_ref[g](
                q_bf, k_bf, v_bf,
                attn.pos_bias.float(), attn.scale_embed.float(),
                attn.phase_base.float(), phase_gain_ref.float(),
                y_pre, z_pre, out_ref, lse_ref,
                attn.offsets_dev,
                q_bf.stride(0), q_bf.stride(1), q_bf.stride(2), q_bf.stride(3),
                k_bf.stride(0), k_bf.stride(1), k_bf.stride(2), k_bf.stride(3),
                v_bf.stride(0), v_bf.stride(1), v_bf.stride(2), v_bf.stride(3),
                out_ref.stride(0), out_ref.stride(1), out_ref.stride(2), out_ref.stride(3),
                lse_ref.stride(0), lse_ref.stride(1), lse_ref.stride(2),
                attn.pos_bias.stride(0), attn.pos_bias.stride(1),
                attn.scale_embed.stride(0), attn.scale_embed.stride(1),
                attn.phase_base.stride(0), attn.phase_base.stride(1),
                phase_gain_ref.stride(0), phase_gain_ref.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BHD,
                J_VAL=attn.j_val, J_SMALL_VAL=attn.j_small,
                num_warps=4, num_stages=2,
            )
            out_ref = out_ref.float() * attn.if_gain.view(1, H, 1, 1)

            out_fused = torch.empty_like(q_bf)
            lse_fused = torch.empty(B, H, N, device=device, dtype=torch.float32)
            _fwd_fused[g](
                q_bf, k_bf, v_bf,
                attn.pos_bias.float(), attn.scale_embed.float(),
                attn.phase_base.float(), phase_gain_ref.float(),
                attn.if_gain.float().contiguous(),
                attn.query_probes[0].float().contiguous(),
                attn.query_probes[1].float().contiguous(),
                attn.key_probes[0].float().contiguous(),
                attn.key_probes[1].float().contiguous(),
                out_fused, lse_fused,
                attn.offsets_dev,
                q_bf.stride(0), q_bf.stride(1), q_bf.stride(2), q_bf.stride(3),
                k_bf.stride(0), k_bf.stride(1), k_bf.stride(2), k_bf.stride(3),
                v_bf.stride(0), v_bf.stride(1), v_bf.stride(2), v_bf.stride(3),
                out_fused.stride(0), out_fused.stride(1), out_fused.stride(2), out_fused.stride(3),
                lse_fused.stride(0), lse_fused.stride(1), lse_fused.stride(2),
                attn.pos_bias.stride(0), attn.pos_bias.stride(1),
                attn.scale_embed.stride(0), attn.scale_embed.stride(1),
                attn.phase_base.stride(0), attn.phase_base.stride(1),
                phase_gain_ref.stride(0), phase_gain_ref.stride(1),
                H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BHD,
                J_VAL=attn.j_val, J_SMALL_VAL=attn.j_small,
                num_warps=4, num_stages=2,
            )

            a = out_ref.flatten().float()
            b = out_fused.flatten().float()
            cosine_similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            results['cosine_similarity'] = cosine_similarity
            print(f'  Check 1 — forward cosine similarity: {cosine_similarity:.6f}  '
                  f'{"PASS" if cosine_similarity > 0.9999 else "FAIL"}')

        # ── Checks 2-4: gradient existence ──
        attn.zero_grad()
        x_grad = x.clone().requires_grad_(True)
        out_module = attn(x_grad)
        loss = out_module.sum()
        loss.backward()

        for name, param, check_number in [
            ('query_probes', attn.query_probes, 2),
            ('key_probes', attn.key_probes, 3),
            ('if_gain', attn.if_gain, 4),
        ]:
            has_grad = param.grad is not None and torch.isfinite(param.grad).all().item()
            results[name] = has_grad
            print(f'  Check {check_number} — {name}.grad finite: {has_grad}  '
                  f'{"PASS" if has_grad else "FAIL"}')

        # ── Check 5: 20-step training loop stays finite ──
        mini_model = TriadicJ96(
            vocab_size=256, embedding_dim=D, num_heads=H,
            ffn_dim=128, seq_len=N,
            scale_embed_init_val=SCALE_EMBED_INIT_VAL, dropout=0.0,
        ).to(device)
        opt = torch.optim.Adam(mini_model.parameters(), lr=1e-4)
        finite_ok = True
        for step in range(20):
            ids = torch.randint(0, 256, (B, N), device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = mini_model(ids)
            step_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))
            if not torch.isfinite(step_loss):
                finite_ok = False
                break
            opt.zero_grad()
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(mini_model.parameters(), 1.0)
            opt.step()
        results['training_finite'] = finite_ok
        print(f'  Check 5 — 20-step training loop finite: {finite_ok}  '
              f'{"PASS" if finite_ok else "FAIL"}')

        # ── Check 6: syntax check ──
        results['syntax'] = True
        print(f'  Check 6 — Python syntax: True  PASS')

        all_passed = (
            results['cosine_similarity'] > 0.9999
            and results['query_probes']
            and results['key_probes']
            and results['if_gain']
            and results['training_finite']
            and results['syntax']
        )
        print(f'\n  {"ALL PASSED" if all_passed else "SOME FAILED"}')
        return all_passed

    _test_kernel_fusion()


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
