"""
DSQG Attention V13-dynamic-J — Two-pass streaming kernel with runtime-configurable J
=====================================================================================

Merges V13's two-pass streaming Triton kernel with V8-dynamic-J's runtime-configurable
J support. J, J_SMALL, J_LARGE are passed as tl.constexpr kernel parameters (not
module-level constants), and offsets are loaded from a device pointer (not a hardcoded
tuple).

V13 mechanisms preserved:
  1. Two-pass forward decomposition (pass 1: scores, pass 2: softmax + weighted V sum)
  2. MOVT in pass 2 for large offsets
  3. BF16 fixes (.contiguous().clone() for pos_bias/scale_embed)
  4. No @triton.autotune
  5. Per-arch SM dispatch for BLOCK_N/num_warps/num_stages

Usage:
  from dsqg_attention_v13_dynamic_j import DSQGAttentionV13Dynamic
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

R_PLANES = 2


def _detect_gpu_capability():
    if not torch.cuda.is_available():
        return None, None
    cc = torch.cuda.get_device_capability()
    sm_90 = (cc[0] == 9 and cc[1] == 0) or cc[0] > 9
    sm_89 = (cc[0] == 8 and cc[1] == 9)
    return sm_90, sm_89


def _next_pow2(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ---------------------------------------------------------------------------
# Forward Kernel — Two-pass decomposition with MOVT, dynamic J
# ---------------------------------------------------------------------------

@triton.jit
def _two_pass_fwd_v13_dj(
    Q, K, V, POS_BIAS, SCALE_EMBED,
    PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    OUT, LSE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_PAD: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    sc = 1.0 / tl.sqrt(HD * 1.0)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh

    q = tl.load(
        qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=nm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)

    # -- Pass 1: Compute all J scores independently (no serial mi/li chain) --
    scores = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)
    js = tl.arange(0, J_PAD)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & nm

        kt = tl.load(
            kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se_i = tl.load(
            SCALE_EMBED + i * stride_sei + ds * stride_sed,
            mask=dm, other=0.0
        ).to(tl.float32)
        s += tl.sum(q * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        col_mask = js == i
        scores = tl.where(col_mask[None, :], s[:, None], scores)

    # -- Softmax over J dimension (columns >= J_VAL are -inf -> contribute 0) --
    mi = tl.max(scores, axis=1)
    all_invalid = mi == float('-inf')
    safe_mi = tl.where(all_invalid, 0.0, mi)
    probs = tl.exp(scores - safe_mi[:, None])
    li = tl.sum(probs, axis=1)
    ls = tl.where(li > 0.0, li, 1.0)
    lse_val = tl.where(all_invalid, tl.zeros_like(mi), mi + tl.log(ls))
    probs = probs / ls[:, None]

    # -- Pass 2: Weighted V sum with MOVT rotation for large offsets --
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh
    y0 = tl.load(yb + ns * stride_yn + 0, mask=nm, other=0.0)
    y1 = tl.load(yb + ns * stride_yn + 1, mask=nm, other=0.0)

    col0 = (ds == 0); col1 = (ds == 1)
    col2 = (ds == 2); col3 = (ds == 3)
    f0 = tl.where(col0, 1.0, 0.0)
    f1 = tl.where(col1, 1.0, 0.0)
    f2 = tl.where(col2, 1.0, 0.0)
    f3 = tl.where(col3, 1.0, 0.0)

    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & nm

        vt = tl.load(
            vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)

        col_mask = js == i
        pi = tl.sum(probs * col_mask[None, :].to(tl.float32), axis=1)

        if i < J_SMALL_VAL:
            acc += pi[:, None] * vt
        else:
            pi_idx = i - J_SMALL_VAL
            z0 = tl.load(zb + kp * stride_zn + 0, mask=val, other=0.0)
            z1 = tl.load(zb + kp * stride_zn + 1, mask=val, other=0.0)

            pb0 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0 * z0
            theta1 = pb1 + pg1 * y1 * z1

            cos0 = tl.cos(theta0); sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1); sin1 = tl.sin(theta1)

            v0 = tl.sum(vt * f0[None, :], axis=1)
            v1 = tl.sum(vt * f1[None, :], axis=1)
            vt = tl.where(col0[None, :], (cos0 * v0 - sin0 * v1)[:, None], vt)
            vt = tl.where(col1[None, :], (sin0 * v0 + cos0 * v1)[:, None], vt)

            v2 = tl.sum(vt * f2[None, :], axis=1)
            v3 = tl.sum(vt * f3[None, :], axis=1)
            vt = tl.where(col2[None, :], (cos1 * v2 - sin1 * v3)[:, None], vt)
            vt = tl.where(col3[None, :], (sin1 * v2 + cos1 * v3)[:, None], vt)

            acc += pi[:, None] * vt

    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(
        ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=nm[:, None] & dm[None, :]
    )
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


# ---------------------------------------------------------------------------
# D computation kernel — sum(dO * O, dim=-1)
# ---------------------------------------------------------------------------

@triton.jit
def _compute_D_v13_dj(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_db, stride_dh, stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD

    do = tl.load(
        DO + b * stride_dob + h * stride_doh + ns[:, None] * stride_don + ds[None, :] * stride_dod,
        mask=nm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)
    o = tl.load(
        O + b * stride_ob + h * stride_oh + ns[:, None] * stride_on + ds[None, :] * stride_od,
        mask=nm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)
    tl.store(
        D + b * stride_db + h * stride_dh + ns * stride_dn,
        tl.sum(do * o, axis=1),
        mask=nm
    )


# ---------------------------------------------------------------------------
# Backward: dQ + dpos_bias + dscale_embed + dY_PRE
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_dq_v13_dj(
    Q, K, V, POS_BIAS, SCALE_EMBED,
    PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, Dv,
    DQ, DPB_BUF, DSE_BUF, DY_PRE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_Db, stride_Dh, stride_Dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_dpb_bh, stride_dpb_blk,
    stride_dse_bh, stride_dse_blk,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dyb, stride_dyh, stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    sc = 1.0 / tl.sqrt(HD * 1.0)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(
        qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=nm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)
    do = tl.load(
        dob + ns[:, None] * stride_don + ds[None, :] * stride_dod,
        mask=nm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + ns * stride_ln, mask=nm, other=0.0)
    Dval = tl.load(Dv + b * stride_Db + h * stride_Dh + ns * stride_Dn, mask=nm, other=0.0)

    y0 = tl.load(yb + ns * stride_yn + 0, mask=nm, other=0.0)
    y1 = tl.load(yb + ns * stride_yn + 1, mask=nm, other=0.0)

    f0 = tl.where(ds == 0, 1.0, 0.0); f1 = tl.where(ds == 1, 1.0, 0.0)
    f2 = tl.where(ds == 2, 1.0, 0.0); f3 = tl.where(ds == 3, 1.0, 0.0)
    col0 = (ds == 0); col1 = (ds == 1)
    col2 = (ds == 2); col3 = (ds == 3)

    dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    dy_pre0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre1 = tl.zeros([BLOCK_N], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & nm

        kt = tl.load(
            kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)
        vt = tl.load(
            vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)

        se_i = tl.load(
            SCALE_EMBED + i * stride_sei + ds * stride_sed,
            mask=dm, other=0.0
        ).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s += tl.sum(q * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        diff = tl.minimum(s - lse, 0.0)
        alpha = tl.where(val & (lse > float('-inf')), tl.exp(diff), 0.0)

        if i < J_SMALL_VAL:
            dot_rv = tl.sum(do * vt, axis=1)
            ds_v = alpha * (dot_rv - Dval)
            dq += ds_v[:, None] * kt * sc
            dq += ds_v[:, None] * se_i[None, :] * sc

            tl.store(
                DPB_BUF + bh * stride_dpb_bh + blk * stride_dpb_blk + i,
                tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0))
            )
            dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
            tl.store(
                DSE_BUF + bh * stride_dse_bh + blk * stride_dse_blk + i * HD + ds,
                tl.where(dm, dse_i, 0.0),
                mask=dm
            )
        else:
            pi_idx = i - J_SMALL_VAL
            z0 = tl.load(zb + kp * stride_zn + 0, mask=val, other=0.0)
            z1 = tl.load(zb + kp * stride_zn + 1, mask=val, other=0.0)

            pb0 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0 * z0
            theta1 = pb1 + pg1 * y1 * z1

            cos0 = tl.cos(theta0); sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1); sin1 = tl.sin(theta1)

            v0 = tl.sum(vt * f0[None, :], axis=1); v1 = tl.sum(vt * f1[None, :], axis=1)
            v2 = tl.sum(vt * f2[None, :], axis=1); v3 = tl.sum(vt * f3[None, :], axis=1)
            vt_rot = vt
            vt_rot = tl.where(col0[None, :], (cos0 * v0 - sin0 * v1)[:, None], vt_rot)
            vt_rot = tl.where(col1[None, :], (sin0 * v0 + cos0 * v1)[:, None], vt_rot)
            vt_rot = tl.where(col2[None, :], (cos1 * v2 - sin1 * v3)[:, None], vt_rot)
            vt_rot = tl.where(col3[None, :], (sin1 * v2 + cos1 * v3)[:, None], vt_rot)

            dot_rv = tl.sum(do * vt_rot, axis=1)
            ds_v = alpha * (dot_rv - Dval)
            dq += ds_v[:, None] * kt * sc
            dq += ds_v[:, None] * se_i[None, :] * sc

            tl.store(
                DPB_BUF + bh * stride_dpb_bh + blk * stride_dpb_blk + i,
                tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0))
            )
            dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
            tl.store(
                DSE_BUF + bh * stride_dse_bh + blk * stride_dse_blk + i * HD + ds,
                tl.where(dm, dse_i, 0.0),
                mask=dm
            )

            do0 = tl.sum(do * f0[None, :], axis=1); do1 = tl.sum(do * f1[None, :], axis=1)
            do2 = tl.sum(do * f2[None, :], axis=1); do3 = tl.sum(do * f3[None, :], axis=1)

            dth0 = alpha * (do0 * (-v0 * sin0 - v1 * cos0) + do1 * (v0 * cos0 - v1 * sin0))
            dth1 = alpha * (do2 * (-v2 * sin1 - v3 * cos1) + do3 * (v2 * cos1 - v3 * sin1))

            dy_pre0 += dth0 * pg0 * z0
            dy_pre1 += dth1 * pg1 * z1

    tl.store(
        DQ + b * stride_dqb + h * stride_dqh + ns[:, None] * stride_dqn + ds[None, :] * stride_dqd,
        dq.to(tl.bfloat16),
        mask=nm[:, None] & dm[None, :]
    )

    dyb = DY_PRE + b * stride_dyb + h * stride_dyh
    tl.atomic_add(dyb + ns * stride_dyn + 0, tl.where(nm, dy_pre0, 0.0))
    tl.atomic_add(dyb + ns * stride_dyn + 1, tl.where(nm, dy_pre1, 0.0))


# ---------------------------------------------------------------------------
# Backward: dK + dV + d_phase_base + d_phase_gain + dZ_PRE
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_dkdv_v13_dj(
    Q, K, V, POS_BIAS, SCALE_EMBED,
    PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, Dv,
    DK, DV,
    DPHASE_BASE_BUF, DPHASE_GAIN_BUF,
    DZ_PRE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_Db, stride_Dh, stride_Dn,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_buf_bh, stride_buf_blk,
    stride_dzb, stride_dzh, stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    m0 = blk * BLOCK_M

    ms = m0 + tl.arange(0, BLOCK_M)
    mm = ms < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    sc = 1.0 / tl.sqrt(HD * 1.0)

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    kt = tl.load(
        kb + ms[:, None] * stride_kn + ds[None, :] * stride_kd,
        mask=mm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)
    vt = tl.load(
        vb + ms[:, None] * stride_vn + ds[None, :] * stride_vd,
        mask=mm[:, None] & dm[None, :],
        other=0.0
    ).to(tl.float32)

    z0_t = tl.load(zb + ms * stride_zn + 0, mask=mm, other=0.0)
    z1_t = tl.load(zb + ms * stride_zn + 1, mask=mm, other=0.0)

    f0 = tl.where(ds == 0, 1.0, 0.0); f1 = tl.where(ds == 1, 1.0, 0.0)
    f2 = tl.where(ds == 2, 1.0, 0.0); f3 = tl.where(ds == 3, 1.0, 0.0)
    col0 = (ds == 0); col1 = (ds == 1)
    col2 = (ds == 2); col3 = (ds == 3)

    v0_t = tl.sum(vt * f0[None, :], axis=1)
    v1_t = tl.sum(vt * f1[None, :], axis=1)
    v2_t = tl.sum(vt * f2[None, :], axis=1)
    v3_t = tl.sum(vt * f3[None, :], axis=1)

    dk = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre1 = tl.zeros([BLOCK_M], tl.float32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_ = ms + delta
        val = (np_ < N) & mm

        qn = tl.load(
            qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)
        don = tl.load(
            dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)
        lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln, mask=val, other=0.0)
        Dn = tl.load(Dv + b * stride_Db + h * stride_Dh + np_ * stride_Dn, mask=val, other=0.0)

        se_i = tl.load(
            SCALE_EMBED + i * stride_sei + ds * stride_sed,
            mask=dm, other=0.0
        ).to(tl.float32)

        s = tl.sum(qn * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s += tl.sum(qn * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        diff = tl.minimum(s - lsen, 0.0)
        alpha = tl.where(val & (lsen > float('-inf')), tl.exp(diff), 0.0)

        if i < J_SMALL_VAL:
            dot_rv = tl.sum(don * vt, axis=1)
            ds_v = alpha * (dot_rv - Dn)
            dk += ds_v[:, None] * qn * sc
            dv += alpha[:, None] * don
        else:
            pi_idx = i - J_SMALL_VAL

            y0_n = tl.load(yb + np_ * stride_yn + 0, mask=val, other=0.0)
            y1_n = tl.load(yb + np_ * stride_yn + 1, mask=val, other=0.0)

            pb0 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 0)
            pb1 = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + 1)
            pg0 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 0)
            pg1 = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + 1)
            theta0 = pb0 + pg0 * y0_n * z0_t
            theta1 = pb1 + pg1 * y1_n * z1_t

            cos0 = tl.cos(theta0); sin0 = tl.sin(theta0)
            cos1 = tl.cos(theta1); sin1 = tl.sin(theta1)

            vt_rot = vt
            vt_rot = tl.where(col0[None, :], (cos0 * v0_t - sin0 * v1_t)[:, None], vt_rot)
            vt_rot = tl.where(col1[None, :], (sin0 * v0_t + cos0 * v1_t)[:, None], vt_rot)
            vt_rot = tl.where(col2[None, :], (cos1 * v2_t - sin1 * v3_t)[:, None], vt_rot)
            vt_rot = tl.where(col3[None, :], (sin1 * v2_t + cos1 * v3_t)[:, None], vt_rot)

            dot_rv = tl.sum(don * vt_rot, axis=1)
            ds_v = alpha * (dot_rv - Dn)
            dk += ds_v[:, None] * qn * sc

            don0 = tl.sum(don * f0[None, :], axis=1); don1 = tl.sum(don * f1[None, :], axis=1)
            don2 = tl.sum(don * f2[None, :], axis=1); don3 = tl.sum(don * f3[None, :], axis=1)
            dv_c = alpha[:, None] * don
            dv_c = tl.where(col0[None, :], (alpha * ( cos0 * don0 + sin0 * don1))[:, None], dv_c)
            dv_c = tl.where(col1[None, :], (alpha * (-sin0 * don0 + cos0 * don1))[:, None], dv_c)
            dv_c = tl.where(col2[None, :], (alpha * ( cos1 * don2 + sin1 * don3))[:, None], dv_c)
            dv_c = tl.where(col3[None, :], (alpha * (-sin1 * don2 + cos1 * don3))[:, None], dv_c)
            dv += dv_c

            dth0 = alpha * (don0 * (-v0_t * sin0 - v1_t * cos0) + don1 * (v0_t * cos0 - v1_t * sin0))
            dth1 = alpha * (don2 * (-v2_t * sin1 - v3_t * cos1) + don3 * (v2_t * cos1 - v3_t * sin1))

            buf_off = bh * stride_buf_bh + blk * stride_buf_blk + pi_idx * 2
            tl.store(DPHASE_BASE_BUF + buf_off + 0, tl.sum(tl.where(val.to(tl.int1), dth0, 0.0)))
            tl.store(DPHASE_BASE_BUF + buf_off + 1, tl.sum(tl.where(val.to(tl.int1), dth1, 0.0)))
            tl.store(DPHASE_GAIN_BUF + buf_off + 0,
                     tl.sum(tl.where(val.to(tl.int1), dth0 * y0_n * z0_t, 0.0)))
            tl.store(DPHASE_GAIN_BUF + buf_off + 1,
                     tl.sum(tl.where(val.to(tl.int1), dth1 * y1_n * z1_t, 0.0)))

            dz_pre0 += tl.where(val.to(tl.int1), dth0 * pg0 * y0_n, 0.0)
            dz_pre1 += tl.where(val.to(tl.int1), dth1 * pg1 * y1_n, 0.0)

    tl.store(
        DK + b * stride_dkb + h * stride_dkh + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
        dk.to(tl.bfloat16),
        mask=mm[:, None] & dm[None, :]
    )
    tl.store(
        DV + b * stride_dvb + h * stride_dvh + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
        dv.to(tl.bfloat16),
        mask=mm[:, None] & dm[None, :]
    )

    dzb = DZ_PRE + b * stride_dzb + h * stride_dzh
    tl.store(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre0, 0.0), mask=mm)
    tl.store(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre1, 0.0), mask=mm)


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _DSQGV13DJFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                j_val, j_small, j_large, offsets_dev):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16

        # Clone all inputs that will be saved for backward or passed to the Triton
        # kernel. Under gradient checkpointing, autocast re-runs this forward in a
        # new context that may rebind BF16 tensor storage in-place. Without owned
        # copies, the saved_tensors retrieved in backward point at stale memory →
        # illegal memory access.
        q = q.contiguous().clone()
        k = k.contiguous().clone()
        v = v.contiguous().clone()
        y_pre = y_pre.contiguous().clone()
        z_pre = z_pre.contiguous().clone()
        pos_bias = pos_bias.contiguous().clone()
        scale_embed = scale_embed.contiguous().clone()

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 16, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64,  8, 4
            else:       BLOCK_N, _nw, _ns = 64,  8, 4
        elif HD <= 128:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 16, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64,  8, 4
            else:       BLOCK_N, _nw, _ns = 64,  8, 4
        else:
            if _sm90:   BLOCK_N, _nw, _ns = 32,  8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 32,  8, 4
            else:       BLOCK_N, _nw, _ns = 16,  8, 4

        BLOCK_HD = _next_pow2(HD)
        J_PAD = _next_pow2(j_val)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        grid = (B * H, triton.cdiv(N, BLOCK_N))

        _two_pass_fwd_v13_dj[grid](
            q, k, v, pos_bias, scale_embed,
            phase_base, phase_gain, y_pre, z_pre,
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
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            H=H, N=N, HD=HD,
            BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
            J_VAL=j_val, J_SMALL_VAL=j_small, J_PAD=J_PAD,
            num_warps=_nw, num_stages=_ns,
        )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev)
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _nw
        ctx.num_stages = _ns
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pos_bias, scale_embed,
         phase_base, phase_gain, y_pre, z_pre,
         out, lse, offsets_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN = ctx.BLOCK_N
        BHD = ctx.BLOCK_HD
        NW = ctx.num_warps
        NS = ctx.num_stages
        j_val = ctx.j_val
        j_small = ctx.j_small
        j_large = ctx.j_large

        dout = dout.contiguous()

        D = torch.zeros(B, H, N, device=q.device, dtype=torch.float32)
        grid = (B * H, triton.cdiv(N, BN))

        _compute_D_v13_dj[grid](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD,
            BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = triton.cdiv(N, BN)
        dq = torch.zeros_like(q)
        dpb_buf = torch.zeros(B * H, blocks_n, j_val, device=q.device, dtype=torch.float32)
        dse_buf = torch.zeros(B * H, blocks_n, j_val * HD, device=q.device, dtype=torch.float32)
        dy_pre = torch.zeros_like(y_pre)

        _bwd_dq_v13_dj[grid](
            q, k, v, pos_bias, scale_embed,
            phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            offsets_dev,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            blocks_n * j_val, j_val,
            blocks_n * j_val * HD, j_val * HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
            H=H, N=N, HD=HD,
            BLOCK_N=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )

        dpb = dpb_buf.view(B, H, blocks_n, j_val).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, j_val, HD).sum(dim=(0, 1, 2)).contiguous()

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)

        _dev = q.device
        phase_base_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                     device=_dev, dtype=torch.float32)
        phase_gain_buf = torch.empty(B * H, blocks_n, j_large * 2,
                                     device=_dev, dtype=torch.float32)
        stride_buf_bh = blocks_n * j_large * 2
        stride_buf_blk = j_large * 2

        _bwd_dkdv_v13_dj[grid](
            q, k, v, pos_bias, scale_embed,
            phase_base, phase_gain, y_pre, z_pre,
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
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            stride_buf_bh, stride_buf_blk,
            dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
            H=H, N=N, HD=HD,
            BLOCK_M=BN, BLOCK_HD=BHD,
            J_VAL=j_val, J_SMALL_VAL=j_small,
            num_warps=NW, num_stages=NS,
        )

        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, j_large, 2).sum(dim=(0, 2))
            return r.permute(1, 0, 2).contiguous()
        d_phase_base = _reduce_phase_buf(phase_base_buf)
        d_phase_gain = _reduce_phase_buf(phase_gain_buf)

        dq_total = dq.float()
        dk_total = dk.float()

        return (dq_total.bfloat16(), dk_total.bfloat16(), dv,
                dpb, dse, d_phase_base, d_phase_gain, dy_pre, dz_pre,
                None, None, None, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dsqg_attention_v13_dj(q, k, v, pos_bias, scale_embed,
                          phase_base, phase_gain, y_pre, z_pre,
                          j_val, j_small, j_large, offsets_dev):
    """
    DSQG V13-dynamic-J attention — two-pass streaming kernel with configurable J.

    Args:
        q, k, v:       [B, H, N, HD] bfloat16
        pos_bias:      [J, H]        float32
        scale_embed:   [J, HD]       float32
        phase_base:    [J_LARGE, H, 2] float32
        phase_gain:    [J_LARGE, H, 2] float32
        y_pre:         [B, H, N, 2]  float32
        z_pre:         [B, H, N, 2]  float32
        j_val:         int — total number of offsets
        j_small:       int — number of small offsets (no MOVT)
        j_large:       int — number of large offsets (MOVT applied)
        offsets_dev:   [J] int32 tensor on device

    Returns:
        out: [B, H, N, HD] bfloat16
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    out = _DSQGV13DJFn.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
        j_val, j_small, j_large, offsets_dev,
    )

    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class DSQGAttentionV13Dynamic(nn.Module):
    """
    DSQG V13-dynamic-J: Two-pass streaming attention with runtime-configurable J.

    Constructor takes offsets list, j_small count, j_large count.
    MOVT (2-plane Givens rotation) applied to V for large offsets (indices j_small..j-1).
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets,
        j_small,
        j_large,
        scale_embed_init=0.15,
        ema_init=0.0208,
        seq_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        D = embedding_dim
        H = num_heads
        self.num_heads = H
        self.head_dim = D // H
        HD = self.head_dim
        self.seq_len = seq_len

        assert HD >= 4, "HD must be >= 4 for 2-plane Givens rotation on channels (0,1),(2,3)"

        j_val = len(offsets)
        assert j_small + j_large == j_val, (
            f"j_small({j_small}) + j_large({j_large}) != len(offsets)({j_val})"
        )
        self.j_val = j_val
        self.j_small = j_small
        self.j_large = j_large

        self.register_buffer(
            'offsets_dev',
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

        self.qkv_proj = nn.Linear(D, 3 * D, bias=True)
        self.out_proj = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, H)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets],
            dtype=torch.float32
        )
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.full((j_val, HD), scale_embed_init))
        self.if_gain = nn.Parameter(torch.ones(H))

        self.phase_base = nn.Parameter(torch.zeros(j_large, H, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(j_large, H, R_PLANES))
        self.phase_gate = nn.Parameter(torch.zeros(j_large))
        self.content_layer_norm = nn.LayerNorm(R_PLANES, elementwise_affine=False)
        # Non-zero probe init: LayerNorm(zeros)=zeros, which starves phase_gain from step 0
        self.query_probes = nn.Parameter(torch.randn(R_PLANES, HD) * 0.02)
        self.key_probes = nn.Parameter(torch.randn(R_PLANES, HD) * 0.02)
        # content_active: False until percolation (SE|max|>=2.0); training loop flips this
        self.register_buffer('content_active', torch.tensor(False), persistent=True)
        self.npci_theta_k = nn.Parameter(torch.zeros(H))
        self.npci_theta_v = nn.Parameter(torch.zeros(H))

        self.register_buffer('ema_alpha', torch.tensor(ema_init), persistent=False)

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
            k = (k + k_delta).contiguous().clone()
            v = (v + v_delta).contiguous().clone()
        else:
            k = k.clone()
            v = v.clone()
        q = q.clone()

        q_norm = F.normalize(q.float(), dim=-1)
        k_norm = F.normalize(k.float(), dim=-1)
        qp_norm = F.normalize(self.query_probes.float(), dim=-1)
        kp_norm = F.normalize(self.key_probes.float(), dim=-1)
        y_raw = torch.einsum('bhnd,rd->bhnr', q_norm, qp_norm)
        z_raw = torch.einsum('bhnd,rd->bhnr', k_norm, kp_norm)
        if self.content_active:
            # Post-percolation: LayerNorm content signal active, phase_gain learns
            y_pre = self.content_layer_norm(y_raw * z_raw).contiguous()
        else:
            # Pre-percolation: content signal off, relay forms on pure position geometry
            y_pre = torch.zeros_like(y_raw)
        z_pre = torch.ones_like(y_pre)

        gate = torch.sigmoid(self.phase_gate)[:, None, None]
        gated_phase_base = self.phase_base * gate
        gated_phase_gain = self.phase_gain * gate

        out = dsqg_attention_v13_dj(q, k, v,
                                    self.pos_bias, self.scale_embed,
                                    gated_phase_base, gated_phase_gain,
                                    y_pre, z_pre,
                                    self.j_val, self.j_small, self.j_large,
                                    self.offsets_dev)

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D).contiguous()
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def get_autotune_config(self):
        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)
        arch = "sm_90" if _sm90 else ("sm_89" if _sm89 else "sm_86")
        return {
            "arch": arch,
            "kernel": "v13_two_pass_movt_dynamic_j",
            "autotune": False,
            "j_val": self.j_val,
            "j_small": self.j_small,
            "j_large": self.j_large,
        }

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
            se = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            phb = self.phase_base.detach().cpu()
            phg = self.phase_gain.detach().cpu()
            phgate = torch.sigmoid(self.phase_gate).detach().cpu()
            qp = self.query_probes.detach().cpu()
            kp = self.key_probes.detach().cpu()
            thk = self.npci_theta_k.detach().cpu()
            thv = self.npci_theta_v.detach().cpu()

        plane_diff = (phb[:, :, 0] - phb[:, :, 1]).abs().mean().item() if self.j_large > 0 else 0.0

        return {
            'pos_bias_abs_mean': pb.abs().mean().item(),
            'pos_bias_abs_max': pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean': se.abs().mean().item(),
            'scale_embed_abs_max': se.abs().max().item(),
            'if_gain': gain.tolist(),
            'phase_base_abs_mean': phb.abs().mean().item() if self.j_large > 0 else 0.0,
            'phase_base_abs_max': phb.abs().max().item() if self.j_large > 0 else 0.0,
            'phase_base_local_mean': 0.0,
            'phase_base_sparse_mean': phb.abs().mean().item() if self.j_large > 0 else 0.0,
            'phase_base_by_head': phb.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            'phase_base_sparse_by_head': phb.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            'phase_base_plane_diff': plane_diff,
            'phase_gain_abs_mean': phg.abs().mean().item() if self.j_large > 0 else 0.0,
            'phase_gain_abs_max': phg.abs().max().item() if self.j_large > 0 else 0.0,
            'phase_gain_sparse_mean': phg.abs().mean().item() if self.j_large > 0 else 0.0,
            'phase_gain_by_head': phg.abs().mean(dim=(0, 2)).tolist() if self.j_large > 0 else [],
            'phase_gate_mean': phgate.mean().item() if self.j_large > 0 else 0.5,
            'phase_gate_min': phgate.min().item() if self.j_large > 0 else 0.5,
            'phase_gate_max': phgate.max().item() if self.j_large > 0 else 0.5,
            'query_probe_norm': qp.norm(dim=1).tolist(),
            'key_probe_norm': kp.norm(dim=1).tolist(),
            'npci_theta_k': thk.tolist(),
            'npci_theta_v': thv.tolist(),
            'v13_dynamic_j_kernel': 'two_pass_streaming_movt',
            'j_val': self.j_val,
            'j_small': self.j_small,
            'j_large': self.j_large,
        }
