"""
DSQG Attention V8 J=12 — Relay-optimal minimal offset set with 4090-specific tuning
===================================================================================

J=12 subset of V8 (J=24), keeping only the confirmed-necessary offsets for relay:

    ALL_OFFSETS = [1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024]

Local coverage (J_SMALL=4): δ=1,2,4,8 — immediate neighbors, no MOVT
Distal coverage (J_LARGE=8): δ=16+ — relay hops, MOVT applied

This is a science experiment to test whether the relay mechanism works with
a sparser offset set (12 vs 24). If successful, this reduces compute cost
significantly while maintaining relay effectiveness.

Key differences from V8:
- ALL_OFFSETS: J=24 → J=12 (drops δ=3,5,6,7,9,10,13,15,21,23,28,48)
- J_SMALL=4 (was 14): only δ≤8 are local
- J_LARGE=8 (was 10): δ≥16 are distal with MOVT
- Forward/backward: static_range(12) with 12-element delta tuple
- pos_bias shape: [12, H]  (was [24, H])
- scale_embed shape: [12, HD]  (was [24, HD])
- phase_base/gain shape: [8, H, 2]  (was [10, H, 2])

4090-specific tuning (sm_89 = Ada Lovelace):
- Optimized block sizes for 4090's L2 cache and SM architecture
- Falls between H100 (sm90) and 3090 (sm86) performance profiles

Usage:
  from dsqg_attention_v8_j12 import DSQGAttentionV8J12, npci_rotate

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v8_j12.py
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

warnings.filterwarnings("ignore", message=".*tl.advance.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not being used.*", category=UserWarning)

ALL_OFFSETS = [1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024]
J         = len(ALL_OFFSETS)   # 12
J_SMALL   = 4                  # indices 0-3: δ≤8 — local, no MOVT
J_LARGE   = 8                  # indices 4-11: δ≥16 — distal, MOVT applied
assert J == 12 and J_SMALL + J_LARGE == J

R_PLANES  = 2   # number of Givens rotation planes; constexpr throughout

MAX_DELTA = max(ALL_OFFSETS)   # 1024

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# GPU capability detection
_cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9   # H100 NVL
_sm89 = (_cc[0] == 8 and _cc[1] == 9)                  # RTX 4090 Ada


# ─────────────────────────────────────────────────────────────────────────────
# NPCI — Norm-Preserving Coupled Injection (pure PyTorch, unchanged from V5)
# ─────────────────────────────────────────────────────────────────────────────

def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta_h: torch.Tensor) -> torch.Tensor:
    """
    Apply norm-preserving coupled injection.

    x, x_delta: [B, H, N, HD]  (same shape)
    theta_h:    [H]             per-head rotation angle (radians)

    Returns x' with ||x'[b,h,n,:]|| = ||x[b,h,n,:]|| exactly.
    """
    theta  = theta_h.view(1, -1, 1, 1)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    x_hat  = x / x_norm

    parallel  = (x_delta * x_hat).sum(dim=-1, keepdim=True) * x_hat
    perp      = x_delta - parallel
    perp_norm = perp.norm(dim=-1, keepdim=True)

    mask  = perp_norm > x_norm * 1e-10
    u_hat = torch.where(mask, perp / perp_norm.clamp(min=1e-30), torch.zeros_like(perp))

    return torch.cos(theta) * x + torch.sin(theta) * x_norm * u_hat


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kernel V8 J=12 — single static_range(J) loop, direct loads for all offsets
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_v8(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    stride_qb,  stride_qh,  stride_qn,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_ob,  stride_oh,  stride_on,  stride_od,
    stride_lb,  stride_lh,  stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,   # PHASE_BASE strides (last dim stride=1 implicit)
    stride_pgi, stride_pgh,   # PHASE_GAIN strides
    stride_yb,  stride_yh,  stride_yn,
    stride_zb,  stride_zh,  stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
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

    mi  = tl.full([BLOCK_N], float('-inf'), tl.float32)
    li  = tl.zeros([BLOCK_N], tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in tl.static_range(12):
        delta = (1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024)[i]
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

        if i < 4:
            acc = acc * cor[:,None] + p[:,None] * vt
        else:
            pi  = i - 4
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

            acc = acc * cor[:,None] + p[:,None] * vt

    ls  = tl.where(li > 0.0, li, 1.0)
    lse = mi + tl.log(ls)
    acc = acc / ls[:,None]

    ob  = OUT + b*stride_ob + h*stride_oh
    lb  = LSE + b*stride_lb + h*stride_lh
    tl.store(ob + ns[:,None]*stride_on + ds[None,:]*stride_od,
             acc.to(tl.bfloat16), mask=nm[:,None] & dm[None,:])
    tl.store(lb + ns*stride_ln, lse, mask=nm)


# ─────────────────────────────────────────────────────────────────────────────
# D computation — unchanged from V7
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_D_v8(
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
def _bwd_dq_v8(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, O, LSE, Dv,
    DQ, DPB, DSE, DY_PRE,
    stride_qb,   stride_qh,   stride_qn,   stride_qd,
    stride_kb,   stride_kh,   stride_kn,   stride_kd,
    stride_vb,   stride_vh,   stride_vn,   stride_vd,
    stride_dob,  stride_doh,  stride_don,  stride_dod,
    stride_ob,   stride_oh,   stride_on,   stride_od,
    stride_lb,   stride_lh,   stride_ln,
    stride_Db,   stride_Dh,   stride_Dn,
    stride_dqb,  stride_dqh,  stride_dqn,  stride_dqd,
    stride_dpbi, stride_dpbh,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dsei, stride_dsed,
    stride_phi,  stride_phh,
    stride_pgi,  stride_pgh,
    stride_yb,   stride_yh,   stride_yn,
    stride_zb,   stride_zh,   stride_zn,
    stride_dyb,  stride_dyh,  stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
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

    for i in tl.static_range(12):
        delta = (1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024)[i]
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

        if i < 4:
            dot_rv = tl.sum(do * vt, axis=1)
            ds_v   = alpha * (dot_rv - Dval)
            dq    += ds_v[:,None] * kt * sc
            dq    += ds_v[:,None] * se_i[None,:] * sc
            tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                          tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.atomic_add(DSE + i*stride_dsei + ds*stride_dsed, tl.where(dm, dse_i, 0.0))
        else:
            pi  = i - 4
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
            tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                          tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.atomic_add(DSE + i*stride_dsei + ds*stride_dsed, tl.where(dm, dse_i, 0.0))

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
    tl.atomic_add(dyb + ns*stride_dyn + 0, tl.where(nm, dy_pre0, 0.0))
    tl.atomic_add(dyb + ns*stride_dyn + 1, tl.where(nm, dy_pre1, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Backward: dK + dV + d_phase_base + d_phase_gain + dZ_PRE
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_v8(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, Dv,
    DK, DV,
    DPHASE_BASE_BUF, DPHASE_GAIN_BUF,  # [B*H, BLOCKS_N, J_LARGE*2] — no atomics
    DZ_PRE,
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
    stride_buf_bh, stride_buf_blk,   # buffer strides: buf[bh, blk, pi*2+m]
    stride_dzb,  stride_dzh,  stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
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

    for i in tl.static_range(12):
        delta = (1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024)[i]
        np_   = ms + delta          # query positions n = t + delta
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

        if i < 4:
            dot_rv = tl.sum(don * vt, axis=1)
            ds_v   = alpha * (dot_rv - Dn)
            dk    += ds_v[:,None] * qn * sc
            dv    += alpha[:,None] * don
        else:
            pi  = i - 4
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

class _DSQGFnV8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (J, H),          f"pos_bias: {pos_bias.shape}"
        assert scale_embed.shape == (J, HD),          f"scale_embed: {scale_embed.shape}"
        assert phase_base.shape  == (J_LARGE, H, R_PLANES), f"phase_base: {phase_base.shape}"
        assert phase_gain.shape  == (J_LARGE, H, R_PLANES), f"phase_gain: {phase_gain.shape}"
        assert y_pre.shape       == (B, H, N, R_PLANES)
        assert z_pre.shape       == (B, H, N, R_PLANES)

        # Block size tuning for SRAM budget
        # Three-way selection: H100 (sm90) / 4090 (sm89) / generic
        _cc = torch.cuda.get_device_capability()
        _is_sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _is_sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 32:
            if _is_sm90:
                BLOCK_N, _num_warps = (128, 8)
            elif _is_sm89:
                BLOCK_N, _num_warps = (64, 4)
            else:
                BLOCK_N, _num_warps = (64, 4)
        elif HD <= 64:
            if _is_sm90:
                BLOCK_N, _num_warps = (128, 8)
            elif _is_sm89:
                BLOCK_N, _num_warps = (64, 4)
            else:
                BLOCK_N, _num_warps = (64, 4)
        elif HD <= 128:
            if _is_sm90:
                BLOCK_N, _num_warps = (64, 4)
            elif _is_sm89:
                BLOCK_N, _num_warps = (64, 4)
            else:
                BLOCK_N, _num_warps = (32, 4)
        elif HD <= 256:
            BLOCK_N, _num_warps = (32, 4) if _is_sm90 else (16, 4)
        else:
            BLOCK_N, _num_warps = (16, 4) if _is_sm90 else (8, 4)
        BLOCK_HD = _next_pow2(HD)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_v8[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
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
            num_warps=_num_warps,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre, out, lse)
        ctx.BLOCK_N   = BLOCK_N
        ctx.BLOCK_HD  = BLOCK_HD
        ctx.num_warps = _num_warps
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD, NW = ctx.BLOCK_N, ctx.BLOCK_HD, ctx.num_warps
        dout = dout.contiguous()

        D  = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        g  = (B * H, triton.cdiv(N, BN))

        _compute_D_v8[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0),    D.stride(1),    D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW,
        )

        dq     = torch.empty_like(q)
        dpb    = torch.zeros_like(pb)
        dse    = torch.zeros_like(se)
        dy_pre = torch.zeros_like(y_pre)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb, dse, dy_pre,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            dpb.stride(0),  dpb.stride(1),
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            dse.stride(0),  dse.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW,
        )

        dk     = torch.empty_like(k)
        dv     = torch.empty_like(v)
        dz_pre = torch.zeros_like(z_pre)

        blocks_n = (N + BN - 1) // BN
        _dev     = q.device
        phase_base_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        phase_gain_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        stride_buf_bh  = blocks_n * J_LARGE * 2
        stride_buf_blk = J_LARGE * 2

        _bwd_dkdv_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
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
            num_warps=NW,
        )

        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, J_LARGE, 2).sum(dim=(0, 2))  # [H, J_LARGE, 2]
            return r.permute(1, 0, 2).contiguous()                     # [J_LARGE, H, 2]
        d_phase_base = _reduce_phase_buf(phase_base_buf)
        d_phase_gain = _reduce_phase_buf(phase_gain_buf)

        dq_total = dq.float()
        dk_total = dk.float()

        return (dq_total.bfloat16(), dk_total.bfloat16(), dv,
                dpb, dse, d_phase_base, d_phase_gain, dy_pre, dz_pre)


def dsqg_attention_v8(q, k, v, pos_bias, scale_embed,
                      phase_base, phase_gain, y_pre, z_pre):
    """
    q, k, v:       [B, H, N, HD]   bfloat16
    pos_bias:      [12, H]          float32
    scale_embed:   [12, HD]         float32
    phase_base:    [8, H, 2]        float32  MOVT angles (large offsets only, zero=identity)
    phase_gain:    [8, H, 2]        float32  QK-OVT gains (zero=pure MOVT)
    y_pre:         [B, H, N, 2]    float32  Q @ query_probes.T / sqrt(HD)
    z_pre:         [B, H, N, 2]    float32  K @ key_probes.T / sqrt(HD)
    Returns:       [B, H, N, HD]   bfloat16
    """
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV8.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
    )
    return out if orig == torch.bfloat16 else out.to(orig)


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionV8J12(nn.Module):
    """
    DSQG V8 J=12: Minimal relay-optimal offsets + MOVT(r=2) + QK-OVT + NPCI.

    J=12 subset: [1,2,4,8,16,64,96,192,384,512,768,1024]
    J_SMALL=4 (δ≤8, no MOVT), J_LARGE=8 (δ≥16, MOVT applied).

    Parameters owned:
      pos_bias      [12, H]    global frequency prior
      scale_embed   [12, HD]   Q-matched-filter (zero-init)
      if_gain       [H]        IF amplifier gain (1.0-init)
      phase_base    [8, H, 2]  MOVT angles for large offsets (zero-init)
      phase_gain    [8, H, 2]  QK-OVT gains for large offsets (zero-init)
      query_probes  [2, HD]    y precomputation probes (zero-init)
      key_probes    [2, HD]    z precomputation probes (zero-init)
      npci_theta_k  [H]        NPCI K rotation angle (zero-init)
      npci_theta_v  [H]        NPCI V rotation angle (zero-init)
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim
        assert HD >= 4, "HD must be >= 4 for 2-plane Givens rotation on channels (0,1),(2,3)"

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

        self.phase_base   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))
        self.phase_gain   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes   = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        sc    = HD ** -0.5
        y_pre = torch.einsum('bhnd,rd->bhnr',
                             q.float(), self.query_probes.float()).mul(sc).contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr',
                             k.float(), self.key_probes.float()).mul(sc).contiguous()

        out = dsqg_attention_v8(q, k, v,
                                self.pos_bias, self.scale_embed,
                                self.phase_base, self.phase_gain,
                                y_pre, z_pre)

        out     = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()       # [12, H]
            se   = self.scale_embed.detach().cpu()    # [12, HD]
            gain = self.if_gain.detach().cpu()        # [H]
            phb  = self.phase_base.detach().cpu()     # [8, H, 2]
            phg  = self.phase_gain.detach().cpu()     # [8, H, 2]
            qp   = self.query_probes.detach().cpu()
            kp   = self.key_probes.detach().cpu()
            thk  = self.npci_theta_k.detach().cpu()
            thv  = self.npci_theta_v.detach().cpu()

        plane_diff = (phb[:, :, 0] - phb[:, :, 1]).abs().mean().item()

        return {
            'pos_bias_abs_mean':         pb.abs().mean().item(),
            'pos_bias_abs_max':          pb.abs().max().item(),
            'pos_bias_mean_per_head':    pb.mean(0).tolist(),
            'scale_embed_abs_mean':      se.abs().mean().item(),
            'scale_embed_abs_max':       se.abs().max().item(),
            'if_gain':                   gain.tolist(),
            'phase_base_abs_mean':       phb.abs().mean().item(),
            'phase_base_abs_max':        phb.abs().max().item(),
            'phase_base_local_mean':     0.0,
            'phase_base_sparse_mean':    phb.abs().mean().item(),
            'phase_base_by_head':        phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_sparse_by_head': phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_plane_diff':     plane_diff,
            'phase_base_p0_sparse':      phb[:, :, 0].mean(0).tolist(),
            'phase_base_p1_sparse':      phb[:, :, 1].mean(0).tolist(),
            'phase_gain_abs_mean':       phg.abs().mean().item(),
            'phase_gain_abs_max':        phg.abs().max().item(),
            'phase_gain_sparse_mean':    phg.abs().mean().item(),
            'phase_gain_by_head':        phg.abs().mean(dim=(0, 2)).tolist(),
            'query_probe_norm':          qp.norm(dim=1).tolist(),
            'key_probe_norm':            kp.norm(dim=1).tolist(),
            'npci_theta_k':              thk.tolist(),
            'npci_theta_v':              thv.tolist(),
        }


DSQGAttentionV8 = DSQGAttentionV8J12


# ─────────────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch — for correctness testing)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_v8(q, k, v, pos_bias, scale_embed,
                  phase_base, phase_gain, y_pre, z_pre):
    """Pure PyTorch reference. Slow — for testing only.
    phase_base/gain: [J_LARGE=8, H, 2] — large offsets only (indices 4-11).
    """
    B, H, N, HD = q.shape
    sc   = HD ** -0.5
    off  = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)
    kp   = F.pad(k.float(), (0, 0, MAX_DELTA, 0))
    vp   = F.pad(v.float(), (0, 0, MAX_DELTA, 0))
    ni   = torch.arange(N, device=q.device)
    gi   = MAX_DELTA - off[None, :] + ni[:, None]   # [N, 12]
    Ka   = kp[:, :, gi, :]                           # [B, H, N, 12, HD]
    Va   = vp[:, :, gi, :]                           # [B, H, N, 12, HD]

    s = (q.float().unsqueeze(3) * Ka).sum(-1) * sc   # [B, H, N, 12]
    s += pos_bias.T[None, :, None, :]
    s += (q.float().unsqueeze(3) * scale_embed[None, None, :, :]).sum(-1) * sc
    s  = s.masked_fill(
        (ni[:, None] < off[None, :]).unsqueeze(0).unsqueeze(0), float('-inf'))
    a  = F.softmax(s, dim=-1)                        # [B, H, N, 12]
    a  = torch.nan_to_num(a, nan=0.0)

    z_pad  = F.pad(z_pre, (0, 0, MAX_DELTA, 0))
    gi_lg  = gi[:, J_SMALL:]                         # [N, 8]
    za_lg  = z_pad[:, :, gi_lg, :]                  # [B, H, N, 8, 2]
    ya_lg  = y_pre.unsqueeze(3).expand(-1, -1, -1, J_LARGE, -1)  # [B, H, N, 8, 2]

    pb_exp = phase_base.permute(1, 0, 2)[None, :, None, :, :]    # [1, H, 1, 8, 2]
    pg_exp = phase_gain.permute(1, 0, 2)[None, :, None, :, :]    # [1, H, 1, 8, 2]

    theta  = pb_exp + pg_exp * ya_lg * za_lg         # [B, H, N, 8, 2]
    theta0 = theta[..., 0];  theta1 = theta[..., 1]

    cos0 = torch.cos(theta0); sin0 = torch.sin(theta0)
    cos1 = torch.cos(theta1); sin1 = torch.sin(theta1)

    Va_rot = Va.clone()
    v0 = Va[:, :, :, J_SMALL:, 0]; v1 = Va[:, :, :, J_SMALL:, 1]
    Va_rot[:, :, :, J_SMALL:, 0] = cos0 * v0 - sin0 * v1
    Va_rot[:, :, :, J_SMALL:, 1] = sin0 * v0 + cos0 * v1
    v2 = Va[:, :, :, J_SMALL:, 2]; v3 = Va[:, :, :, J_SMALL:, 3]
    Va_rot[:, :, :, J_SMALL:, 2] = cos1 * v2 - sin1 * v3
    Va_rot[:, :, :, J_SMALL:, 3] = sin1 * v2 + cos1 * v3

    out = (a.unsqueeze(-1) * Va_rot).sum(3)
    return out.to(q.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    print("=" * 70)
    print("DSQG V8 J=12 — Correctness Tests (J=12 minimal + MOVT + QK-OVT + NPCI)")
    print("=" * 70)

    _cc = torch.cuda.get_device_capability() if device == 'cuda' else (0, 0)
    _is_sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
    _is_sm89 = (_cc[0] == 8 and _cc[1] == 9)
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    kernel_path = "H100 (sm90)" if _is_sm90 else ("4090 (sm89)" if _is_sm89 else "generic")
    print(f"  GPU: {gpu_name}")
    print(f"  Kernel path: {kernel_path}")
    print()

    cfgs = [
        (1, 8,   64, 32, "tiny"),
        (2, 8,  512, 32, "mid"),
        (2, 8, 2047, 32, "13M shape"),
    ]
    ok_all = True

    for B, H, N, HD, lbl in cfgs:
        torch.manual_seed(42)
        q   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        k   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        v   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
        pb  = torch.randn(J,  H,       device=device, dtype=torch.float32) * 0.5
        se  = torch.randn(J,  HD,      device=device, dtype=torch.float32) * 0.05
        phb = torch.randn(J_LARGE, H, 2, device=device, dtype=torch.float32) * 0.3
        phg = torch.randn(J_LARGE, H, 2, device=device, dtype=torch.float32) * 0.1
        qpr = torch.randn(2, HD,       device=device, dtype=torch.float32) * 0.1
        kpr = torch.randn(2, HD,       device=device, dtype=torch.float32) * 0.1
        sc  = HD ** -0.5
        y   = torch.einsum('bhnd,rd->bhnr', q.float(), qpr).mul(sc).contiguous()
        z   = torch.einsum('bhnd,rd->bhnr', k.float(), kpr).mul(sc).contiguous()

        ref = _reference_v8(q, k, v, pb, se, phb, phg, y, z)
        out = dsqg_attention_v8(q.clone(), k.clone(), v.clone(), pb, se, phb, phg, y, z)
        fe  = (ref.float() - out.float()).abs().max().item()
        ok  = fe < 0.05
        if not ok: ok_all = False
        print(f"  {lbl:22s}  fwd_err={fe:.4f}  {'PASS' if ok else 'FAIL'}")

    print()
    print("  Zero-init phase -> pure pos_bias+scale_embed attention:")
    B, H, N, HD = 2, 8, 128, 32
    torch.manual_seed(7)
    q   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    k   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    v   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    pb  = torch.randn(J, H,       device=device, dtype=torch.float32) * 0.5
    se  = torch.zeros(J, HD,      device=device, dtype=torch.float32)
    phb_z = torch.zeros(J_LARGE, H, 2, device=device, dtype=torch.float32)
    phg_z = torch.zeros(J_LARGE, H, 2, device=device, dtype=torch.float32)
    y_z   = torch.zeros(B, H, N, 2,    device=device, dtype=torch.float32)
    z_z   = torch.zeros(B, H, N, 2,    device=device, dtype=torch.float32)

    out1  = dsqg_attention_v8(q.clone(), k.clone(), v.clone(), pb, se, phb_z, phg_z, y_z, z_z)
    ref1  = _reference_v8(q, k, v, pb, se, phb_z, phg_z, y_z, z_z)
    diff  = (out1.float() - ref1.float()).abs().max().item()
    ok_z  = diff < 0.05
    if not ok_z: ok_all = False
    print(f"  {'triton vs ref (zero)':22s}  max_diff={diff:.6f}  {'PASS' if ok_z else 'FAIL'}")

    print()
    print("  NPCI norm preservation:")
    x     = torch.randn(2, 8, 64, 32, device=device, dtype=torch.float32)
    xd    = torch.randn_like(x)
    theta = torch.randn(8, device=device) * 0.5
    x_out = npci_rotate(x, xd, theta)
    norm_err = (x_out.norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    ok_n = norm_err < 1e-4
    if not ok_n: ok_all = False
    print(f"  {'NPCI norm err':22s}  max_err={norm_err:.2e}  {'PASS' if ok_n else 'FAIL'}")

    print()
    print("  J=12 vs J=24 timing comparison:")
    B, H, N, HD = 2, 8, 2048, 64
    q   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    k   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    v   = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16) * 0.1
    pb  = torch.randn(J, H, device=device, dtype=torch.float32) * 0.5
    se  = torch.randn(J, HD, device=device, dtype=torch.float32) * 0.05
    phb = torch.randn(J_LARGE, H, 2, device=device, dtype=torch.float32) * 0.3
    phg = torch.randn(J_LARGE, H, 2, device=device, dtype=torch.float32) * 0.1
    sc  = HD ** -0.5
    qpr = torch.randn(2, HD, device=device, dtype=torch.float32) * 0.1
    kpr = torch.randn(2, HD, device=device, dtype=torch.float32) * 0.1
    y   = torch.einsum('bhnd,rd->bhnr', q.float(), qpr).mul(sc).contiguous()
    z   = torch.einsum('bhnd,rd->bhnr', k.float(), kpr).mul(sc).contiguous()

    import time
    torch.cuda.synchronize()
    for _ in range(3):
        dsqg_attention_v8(q, k, v, pb, se, phb, phg, y, z)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        dsqg_attention_v8(q, k, v, pb, se, phb, phg, y, z)
    torch.cuda.synchronize()
    t_j12 = (time.perf_counter() - t0) / 20 * 1000
    print(f"  J=12 kernel: {t_j12:.2f} ms/fwd (B={B}, N={N}, HD={HD})")

    print("=" * 70)
    print(f"{'ALL PASSED' if ok_all else 'SOME FAILED'}")
    return ok_all


if __name__ == "__main__":
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ok = run_tests(device)
    sys.exit(0 if ok else 1)
