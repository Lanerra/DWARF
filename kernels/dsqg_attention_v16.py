"""
DSQG Attention V16 — Two-Pass Forward for Triadic J=32
======================================================

Derived from V8-dynamic-J. Replaces the serial online-softmax forward
with a two-pass approach:

  Pass 1: compute global max score per position (no V loads, no exp)
  Pass 2: compute softmax weights with known max, accumulate weighted V

The serial dependency chain in V8 (mi/li/acc correction per iteration)
is replaced by a simple running-max in Pass 1 and a correction-free
accumulation in Pass 2. K data is loaded twice (global then L2 cache).

Backward kernels are unchanged from V8.

Usage:
  from dsqg_attention_v16 import (
      ALL_OFFSETS, _fwd_v16, _compute_D_v16, _bwd_dq_v16, _bwd_dkdv_v16,
      npci_rotate, R_PLANES, _next_pow2, dsqg_attention_v16,
  )

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/test_v16.py
"""

import warnings
import torch
import triton
import triton.language as tl

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

from dsqg_attention_v8_dynamic_j import (
    ALL_OFFSETS, J, J_SMALL, J_LARGE, MAX_DELTA,
    npci_rotate, R_PLANES, _next_pow2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel V16 — Two-Pass: max-score then weighted-V accumulation
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_v16(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
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
    stride_yb,  stride_yh,  stride_yn,
    stride_zb,  stride_zh,  stride_zn,
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
    yb  = Y_PRE + b * stride_yb + h * stride_yh
    zb  = Z_PRE + b * stride_zb + h * stride_zh

    q   = tl.load(qb + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                  mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    y0  = tl.load(yb + ns*stride_yn + 0, mask=nm, other=0.0)
    y1  = tl.load(yb + ns*stride_yn + 1, mask=nm, other=0.0)

    col0 = (ds == 0);  col1 = (ds == 1)
    col2 = (ds == 2);  col3 = (ds == 3)
    f0   = tl.where(col0, 1.0, 0.0)
    f1   = tl.where(col1, 1.0, 0.0)
    f2   = tl.where(col2, 1.0, 0.0)
    f3   = tl.where(col3, 1.0, 0.0)

    # ═══ Pass 1: global max score per position (no V loads, no exp) ═══
    max_s = tl.full([BLOCK_N], float('-inf'), tl.float32)
    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        max_s = tl.maximum(max_s, s)

    # ═══ Pass 2: softmax weights + weighted V sum (K from L2 cache) ═══
    sum_exp = tl.zeros([BLOCK_N], tl.float32)
    acc     = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + i * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        e_i   = tl.where(val, tl.exp(s - max_s), 0.0)
        sum_exp += e_i

        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        if i < J_SMALL_VAL:
            acc += e_i[:,None] * vt
        else:
            pi  = i - J_SMALL_VAL
            z0  = tl.load(zb + kp*stride_zn + 0, mask=val, other=0.0)
            z1  = tl.load(zb + kp*stride_zn + 1, mask=val, other=0.0)

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

            acc += e_i[:,None] * vt

    ls  = tl.where(sum_exp > 0.0, sum_exp, 1.0)
    lse = max_s + tl.log(ls)
    acc = acc / ls[:,None]

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             acc.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# D computation — identical to V8
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_D_v16(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_db,  stride_dh,  stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh = tl.program_id(0); blk = tl.program_id(1)
    b  = bh // H; h = bh % H
    n0 = blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N); nm = ns < N
    ds = tl.arange(0, BLOCK_HD);     dm = ds < HD
    do = tl.load(DO + b*stride_dob + h*stride_doh
                 + ns[:,None]*stride_don + ds[None,:]*stride_dod,
                 mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    o  = tl.load(O  + b*stride_ob  + h*stride_oh
                 + ns[:,None]*stride_on  + ds[None,:]*stride_od,
                 mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    tl.store(D + b*stride_db + h*stride_dh + ns*stride_dn,
             tl.sum(do * o, axis=1), mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# Backward: dQ + dPOS_BIAS + dSCALE_EMBED + dY_PRE
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_v16(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, O, LSE, Dv,
    DQ, DPB_BUF, DSE_BUF, DY_PRE,
    OFFSETS,
    stride_qb,   stride_qh,   stride_qn,   stride_qd,
    stride_kb,   stride_kh,   stride_kn,   stride_kd,
    stride_vb,   stride_vh,   stride_vn,   stride_vd,
    stride_dob,  stride_doh,  stride_don,  stride_dod,
    stride_ob,   stride_oh,   stride_on,   stride_od,
    stride_lb,   stride_lh,   stride_ln,
    stride_Db,   stride_Dh,   stride_Dn,
    stride_dqb,  stride_dqh,  stride_dqn,  stride_dqd,
    stride_dpb_bh, stride_dpb_blk,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dse_bh, stride_dse_blk,
    stride_phi,  stride_phh,
    stride_pgi,  stride_pgh,
    stride_yb,   stride_yh,   stride_yn,
    stride_zb,   stride_zh,   stride_zn,
    stride_dyb,  stride_dyh,  stride_dyn,
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
    yb  = Y_PRE + b*stride_yb + h*stride_yh
    zb  = Z_PRE + b*stride_zb + h*stride_zh

    q    = tl.load(qb  + ns[:,None]*stride_qn + ds[None,:]*stride_qd,
                   mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    do   = tl.load(dob + ns[:,None]*stride_don + ds[None,:]*stride_dod,
                   mask=nm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    lse  = tl.load(LSE + b*stride_lb + h*stride_lh + ns*stride_ln, mask=nm, other=0.0)
    Dval = tl.load(Dv  + b*stride_Db + h*stride_Dh + ns*stride_Dn, mask=nm, other=0.0)

    y0   = tl.load(yb + ns*stride_yn + 0, mask=nm, other=0.0)
    y1   = tl.load(yb + ns*stride_yn + 1, mask=nm, other=0.0)

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
            z0  = tl.load(zb + kp*stride_zn + 0, mask=val, other=0.0)
            z1  = tl.load(zb + kp*stride_zn + 1, mask=val, other=0.0)

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

    tl.store(DQ + b*stride_dqb + h*stride_dqh
             + ns[:,None]*stride_dqn + ds[None,:]*stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])

    dyb = DY_PRE + b*stride_dyb + h*stride_dyh
    tl.store(dyb + ns*stride_dyn + 0, tl.where(nm, dy_pre0, 0.0), mask=nm)
    tl.store(dyb + ns*stride_dyn + 1, tl.where(nm, dy_pre1, 0.0), mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# Backward: dK + dV + d_phase_base + d_phase_gain + dZ_PRE
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_v16(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, Dv,
    DK, DV,
    DPHASE_BASE_BUF, DPHASE_GAIN_BUF,
    DZ_PRE,
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
    stride_yb,   stride_yh,   stride_yn,
    stride_zb,   stride_zh,   stride_zn,
    stride_buf_bh, stride_buf_blk,
    stride_dzb,  stride_dzh,  stride_dzn,
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
    yb  = Y_PRE + b*stride_yb + h*stride_yh
    zb  = Z_PRE + b*stride_zb + h*stride_zh

    kt  = tl.load(kb + ms[:,None]*stride_kn + ds[None,:]*stride_kd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)
    vt  = tl.load(vb + ms[:,None]*stride_vn + ds[None,:]*stride_vd,
                  mask=mm[:,None] & dm[None,:], other=0.0).to(tl.float32)

    z0_t = tl.load(zb + ms*stride_zn + 0, mask=mm, other=0.0)
    z1_t = tl.load(zb + ms*stride_zn + 1, mask=mm, other=0.0)

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

        y0_n  = tl.load(yb + np_*stride_yn + 0, mask=val, other=0.0)
        y1_n  = tl.load(yb + np_*stride_yn + 1, mask=val, other=0.0)

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

    tl.store(DK + b*stride_dkb + h*stride_dkh
             + ms[:,None]*stride_dkn + ds[None,:]*stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])
    tl.store(DV + b*stride_dvb + h*stride_dvh
             + ms[:,None]*stride_dvn + ds[None,:]*stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:,None] & dm[None,:])

    dzb = DZ_PRE + b*stride_dzb + h*stride_dzh
    tl.store(dzb + ms*stride_dzn + 0, tl.where(mm, dz_pre0, 0.0), mask=mm)
    tl.store(dzb + ms*stride_dzn + 1, tl.where(mm, dz_pre1, 0.0), mask=mm)


# ─────────────────────────────────────────────────────────────────────────────
# Autograd wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnV16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape[1] == H
        assert scale_embed.shape[1] == HD
        assert y_pre.shape == (B, H, N, R_PLANES)
        assert z_pre.shape == (B, H, N, R_PLANES)

        j_val   = pos_bias.shape[0]
        j_large = phase_base.shape[0]
        j_small = j_val - j_large

        q = q.contiguous().clone()
        k = k.contiguous().clone()
        v = v.contiguous().clone()

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64,  8, 4
            else:       BLOCK_N, _num_warps, _num_stages = 64,  4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64,  8, 4
            else:       BLOCK_N, _num_warps, _num_stages = 64,  4, 2
        else:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 32,  4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 32,  4, 4
            else:       BLOCK_N, _num_warps, _num_stages = 16,  4, 2

        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        offsets_dev = torch.tensor(ALL_OFFSETS[:j_val], device=q.device, dtype=torch.int32)

        _fwd_v16[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
            offsets_dev,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            pos_bias.stride(0),    pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            phase_base.stride(0),  phase_base.stride(1),
            phase_gain.stride(0),  phase_gain.stride(1),
            y_pre.stride(0),       y_pre.stride(1),       y_pre.stride(2),
            z_pre.stride(0),       z_pre.stride(1),       z_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=_num_warps, num_stages=_num_stages,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre, out, lse,
                              offsets_dev)
        ctx.BLOCK_N    = BLOCK_N
        ctx.BLOCK_HD   = BLOCK_HD
        ctx.num_warps  = _num_warps
        ctx.num_stages = _num_stages
        ctx.j_val      = j_val
        ctx.j_small    = j_small
        ctx.j_large    = j_large
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse, offsets_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD, NW, NS = ctx.BLOCK_N, ctx.BLOCK_HD, ctx.num_warps, ctx.num_stages
        j_val   = ctx.j_val
        j_small = ctx.j_small
        j_large = ctx.j_large
        dout = dout.contiguous()

        D  = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g  = (B * H, triton.cdiv(N, BN))

        _compute_D_v16[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0),    D.stride(1),    D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = (N + BN - 1) // BN
        _dev     = q.device

        dq      = torch.zeros_like(q)
        dy_pre  = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B * H, blocks_n, j_val,       device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, j_val * HD,  device=_dev, dtype=torch.float32)

        _bwd_dq_v16[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            offsets_dev,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            blocks_n * j_val,   j_val,
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            blocks_n*j_val*HD,  j_val*HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(dim=(0, 1, 2)).contiguous()

        dk     = torch.zeros_like(k)
        dv     = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)
        phase_base_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                     device=_dev, dtype=torch.float32)
        phase_gain_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                     device=_dev, dtype=torch.float32)
        stride_buf_bh  = blocks_n * j_large * 2
        stride_buf_blk = j_large * 2

        _bwd_dkdv_v16[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
            offsets_dev,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dk.stride(0),   dk.stride(1),   dk.stride(2),   dk.stride(3),
            dv.stride(0),   dv.stride(1),   dv.stride(2),   dv.stride(3),
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            stride_buf_bh, stride_buf_blk,
            dz_pre.stride(0),     dz_pre.stride(1),     dz_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )

        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, j_large, 2).sum(dim=(0, 2))
            return r.permute(1, 0, 2).contiguous()
        d_phase_base = _reduce_phase_buf(phase_base_buf)
        d_phase_gain = _reduce_phase_buf(phase_gain_buf)

        return (dq, dk, dv,
                dpb, dse, d_phase_base, d_phase_gain, dy_pre, dz_pre)


def dsqg_attention_v16(q, k, v, pos_bias, scale_embed,
                       phase_base, phase_gain, y_pre, z_pre):
    """
    q, k, v:       [B, H, N, HD]   bfloat16
    pos_bias:      [J, H]           float32
    scale_embed:   [J, HD]          float32
    phase_base:    [J_LARGE, H, 2]  float32
    phase_gain:    [J_LARGE, H, 2]  float32
    y_pre:         [B, H, N, 2]     float32
    z_pre:         [B, H, N, 2]     float32
    Returns:       [B, H, N, HD]    bfloat16
    """
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV16.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
    )
    return out if orig == torch.bfloat16 else out.to(orig)
