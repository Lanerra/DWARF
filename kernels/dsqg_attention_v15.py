"""
DSQG Attention V15 — K-Tile Coalesced, Two-Segment Kernels
==========================================================

Derived from V8-dynamic-J. Groups the 96 offsets by K-tile (delta // BLOCK_N)
and splits the inner loop into two segments (GROUP_A without MOVT, GROUP_B+C
with MOVT). Pre-sorted offset arrays and original-index maps are cached once
per device — zero per-call tensor allocation overhead.

Benefits over V8-dynamic-J:
  - L1 cache locality: offsets within a tile are processed consecutively
  - Reduced register pressure: GROUP_A segment carries no MOVT variables
  - Tile map metadata available for introspection and future extensions

Same mechanisms as V8 (MOVT + QK-OVT + NPCI), same J=96 offset set.

Usage:
  from dsqg_attention_v15 import DSQGAttentionV15, npci_rotate

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v15.py
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

ALL_OFFSETS = [
    1,2,3,4,5,6,7,8,9,10,13,15,16,19,21,23,28,
    48,64,96,121,161,192,212,245,273,295,342,375,384,
    413,441,473,512,549,579,593,631,653,694,716,768,
    826,846,900,936,970,1000,1024,1074,1108,1144,1166,
    1190,1218,1244,1288,1322,1385,1423,1451,1497,1522,
    1550,1581,1603,1617,1634,1651,1661,1710,1743,1780,
    1810,1820,1852,1860,1876,1886,1897,1903,1916,1926,
    1929,1941,1965,1983,2006,2011,2029,2037,2044,2068,
    2097,2113,2199,
]
J         = len(ALL_OFFSETS)   # 96
J_SMALL   = 17                 # delta<=28, no MOVT
J_LARGE   = 79                 # delta>=48, MOVT applied
MAX_DELTA = max(ALL_OFFSETS)   # 2199

assert J_SMALL + J_LARGE == J

R_PLANES  = 2

def _next_pow2(n):
    if n <= 0: return 1
    n -= 1; n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; return n+1


# ─────────────────────────────────────────────────────────────────────────────
# Tile Map — groups offsets by K-tile for cache-friendly access
# ─────────────────────────────────────────────────────────────────────────────

def _build_tile_map(offsets, block_n):
    """Group offsets by which K-tile they fall into (delta // block_n).

    Returns list of (tile_idx, [(orig_index, delta)]) sorted by tile_idx
    descending (furthest-back tile first, matching causal order).
    """
    tile_map = {}
    for i, delta in enumerate(offsets):
        tile_idx = delta // block_n
        tile_map.setdefault(tile_idx, []).append((i, delta))
    return sorted(tile_map.items(), reverse=True)


def _build_reorder(offsets, block_n):
    """Build a tile-sorted reorder permutation.

    Returns reorder list where reorder[j] = original offset index,
    with offsets grouped by tile_idx (ascending for forward scan).
    """
    tile_map = {}
    for i, delta in enumerate(offsets):
        tile_idx = delta // block_n
        tile_map.setdefault(tile_idx, []).append(i)
    reorder = []
    for tile_idx in sorted(tile_map.keys()):
        reorder.extend(tile_map[tile_idx])
    return reorder


# ─────────────────────────────────────────────────────────────────────────────
# Cached constant tensors — built once per (device, BLOCK_N)
# ─────────────────────────────────────────────────────────────────────────────

_TILE_CACHE = {}

def _get_tile_tensors(device, block_n):
    key = (str(device), block_n)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]

    reorder = _build_reorder(ALL_OFFSETS, block_n)
    reorder_small = [r for r in reorder if r < J_SMALL]
    reorder_large = [r for r in reorder if r >= J_SMALL]

    cache = {
        'offsets_small': torch.tensor(
            [ALL_OFFSETS[r] for r in reorder_small], device=device, dtype=torch.int32),
        'offsets_large': torch.tensor(
            [ALL_OFFSETS[r] for r in reorder_large], device=device, dtype=torch.int32),
        'orig_idx_small': torch.tensor(
            reorder_small, device=device, dtype=torch.int32),
        'orig_idx_large': torch.tensor(
            reorder_large, device=device, dtype=torch.int32),
        'phase_map_large': torch.tensor(
            [r - J_SMALL for r in reorder_large], device=device, dtype=torch.int32),
        'j_small_n': len(reorder_small),
        'j_large_n': len(reorder_large),
        'reorder_small': reorder_small,
        'reorder_large': reorder_large,
    }
    _TILE_CACHE[key] = cache
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# NPCI — Norm-Preserving Coupled Injection (unchanged from V8)
# ─────────────────────────────────────────────────────────────────────────────

def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta_h: torch.Tensor) -> torch.Tensor:
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
# Forward Kernel V15 — two-segment loop with orig-index indirection
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_v15(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    OFFSETS_SMALL, OFFSETS_LARGE, ORIG_IDX_SMALL, ORIG_IDX_LARGE, PHASE_MAP,
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
    J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
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

    # ── Segment 1: GROUP_A (small offsets, no MOVT) ─────────────────────
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS_SMALL + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_SMALL + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + oi * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + oi * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        mn    = tl.maximum(mi, s)
        cor   = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p     = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        li    = li * cor + p;     mi = mn

        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        acc   = acc * cor[:,None] + p[:,None] * vt

    # ── Segment 2: GROUP_B+C (large offsets, with MOVT) ─────────────────
    for i in range(J_LARGE_VAL):
        delta = tl.load(OFFSETS_LARGE + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_LARGE + i).to(tl.int32)
        pi    = tl.load(PHASE_MAP + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(POS_BIAS + oi * stride_pbi + h * stride_pbh)
        se_i  = tl.load(SE + oi * stride_sei + ds * stride_sed,
                        mask=dm, other=0.0).to(tl.float32)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))

        mn    = tl.maximum(mi, s)
        cor   = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p     = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        li    = li * cor + p;     mi = mn

        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

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
# D computation — unchanged from V8
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_D_v15(
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
# Backward: dQ + dPOS_BIAS + dSCALE_EMBED + dY_PRE (two-segment)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_v15(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, O, LSE, Dv,
    DQ, DPB_BUF, DSE_BUF, DY_PRE,
    OFFSETS_SMALL, OFFSETS_LARGE,
    ORIG_IDX_SMALL, ORIG_IDX_LARGE, PHASE_MAP,
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
    J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
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

    # ── Segment 1: GROUP_A (no MOVT) ────────────────────────────────────
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS_SMALL + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_SMALL + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i  = tl.load(SE + oi*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + oi*stride_pbi + h*stride_pbh)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lse), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v   = alpha * (dot_rv - Dval)
        dq    += ds_v[:,None] * kt * sc
        dq    += ds_v[:,None] * se_i[None,:] * sc
        tl.store(DPB_BUF + bh*stride_dpb_bh + blk*stride_dpb_blk + oi,
                 tl.sum(tl.where(val, ds_v, 0.0)))
        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
        tl.store(DSE_BUF + bh*stride_dse_bh + blk*stride_dse_blk + oi*HD + ds,
                 tl.where(dm, dse_i, 0.0), mask=dm)

    # ── Segment 2: GROUP_B+C (with MOVT) ────────────────────────────────
    for i in range(J_LARGE_VAL):
        delta = tl.load(OFFSETS_LARGE + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_LARGE + i).to(tl.int32)
        pi    = tl.load(PHASE_MAP + i).to(tl.int32)
        kp    = ns - delta
        val   = (kp >= 0) & nm

        kt    = tl.load(kb + kp[:,None]*stride_kn + ds[None,:]*stride_kd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        vt    = tl.load(vb + kp[:,None]*stride_vn + ds[None,:]*stride_vd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)

        se_i  = tl.load(SE + oi*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(q * kt, axis=1) * sc
        s    += tl.load(PB + oi*stride_pbi + h*stride_pbh)
        s    += tl.sum(q * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lse), 0.0)

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
        tl.store(DPB_BUF + bh*stride_dpb_bh + blk*stride_dpb_blk + oi,
                 tl.sum(tl.where(val, ds_v, 0.0)))
        dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
        tl.store(DSE_BUF + bh*stride_dse_bh + blk*stride_dse_blk + oi*HD + ds,
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
# Backward: dK + dV + d_phase_base + d_phase_gain + dZ_PRE (two-segment)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkdv_v15(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, Dv,
    DK, DV,
    DPHASE_BASE_BUF, DPHASE_GAIN_BUF,
    DZ_PRE,
    OFFSETS_SMALL, OFFSETS_LARGE,
    ORIG_IDX_SMALL, ORIG_IDX_LARGE, PHASE_MAP,
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
    J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
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

    # ── Segment 1: GROUP_A (no MOVT) ────────────────────────────────────
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS_SMALL + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_SMALL + i).to(tl.int32)
        np_   = ms + delta
        val   = (np_ < N) & mm

        qn    = tl.load(qb  + np_[:,None]*stride_qn + ds[None,:]*stride_qd,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        don   = tl.load(dob + np_[:,None]*stride_don + ds[None,:]*stride_dod,
                        mask=val[:,None] & dm[None,:], other=0.0).to(tl.float32)
        lsen  = tl.load(LSE + b*stride_lb + h*stride_lh + np_*stride_ln, mask=val, other=0.0)
        Dn    = tl.load(Dv  + b*stride_Db + h*stride_Dh + np_*stride_Dn, mask=val, other=0.0)

        se_i  = tl.load(SE + oi*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(qn * kt, axis=1) * sc
        s    += tl.load(PB + oi*stride_pbi + h*stride_pbh)
        s    += tl.sum(qn * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lsen), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v   = alpha * (dot_rv - Dn)
        dk    += ds_v[:,None] * qn * sc
        dv    += alpha[:,None] * don

    # ── Segment 2: GROUP_B+C (with MOVT) ────────────────────────────────
    for i in range(J_LARGE_VAL):
        delta = tl.load(OFFSETS_LARGE + i).to(tl.int32)
        oi    = tl.load(ORIG_IDX_LARGE + i).to(tl.int32)
        pi    = tl.load(PHASE_MAP + i).to(tl.int32)
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

        se_i  = tl.load(SE + oi*stride_sei + ds*stride_sed, mask=dm, other=0.0).to(tl.float32)
        s     = tl.sum(qn * kt, axis=1) * sc
        s    += tl.load(PB + oi*stride_pbi + h*stride_pbh)
        s    += tl.sum(qn * se_i[None,:], axis=1) * sc
        s     = tl.where(val, s, float('-inf'))
        alpha = tl.where(val, tl.exp(s - lsen), 0.0)

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
# Autograd wrapper — zero per-call tensor allocation
# ─────────────────────────────────────────────────────────────────────────────

class _DSQGFnV15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert pos_bias.shape    == (J, H),                f"pos_bias: {pos_bias.shape}"
        assert scale_embed.shape == (J, HD),                f"scale_embed: {scale_embed.shape}"
        assert phase_base.shape  == (J_LARGE, H, R_PLANES), f"phase_base: {phase_base.shape}"
        assert phase_gain.shape  == (J_LARGE, H, R_PLANES), f"phase_gain: {phase_gain.shape}"
        assert y_pre.shape       == (B, H, N, R_PLANES)
        assert z_pre.shape       == (B, H, N, R_PLANES)

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64,  8, 2
            else:       BLOCK_N, _num_warps, _num_stages = 64,  4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 128, 8, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 64,  4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 32,  4, 2
        elif HD <= 256:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 32,  4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 32,  4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 16,  4, 2
        else:
            if _sm90:   BLOCK_N, _num_warps, _num_stages = 16,  4, 3
            elif _sm89: BLOCK_N, _num_warps, _num_stages = 16,  4, 2
            else:       BLOCK_N, _num_warps, _num_stages = 8,   4, 2

        BLOCK_HD = _next_pow2(HD)
        _dev = q.device

        tc = _get_tile_tensors(_dev, BLOCK_N)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=_dev, dtype=torch.float32)
        g   = (B * H, triton.cdiv(N, BLOCK_N))

        _fwd_v15[g](
            q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
            y_pre, z_pre, out, lse,
            tc['offsets_small'], tc['offsets_large'],
            tc['orig_idx_small'], tc['orig_idx_large'], tc['phase_map_large'],
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
            J_SMALL_VAL=tc['j_small_n'], J_LARGE_VAL=tc['j_large_n'],
            num_warps=_num_warps, num_stages=_num_stages,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre, out, lse)
        ctx.BLOCK_N    = BLOCK_N
        ctx.BLOCK_HD   = BLOCK_HD
        ctx.num_warps  = _num_warps
        ctx.num_stages = _num_stages
        ctx.tile_cache = tc
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pb, se, phase_base, phase_gain,
         y_pre, z_pre, out, lse) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN, BHD, NW, NS = ctx.BLOCK_N, ctx.BLOCK_HD, ctx.num_warps, ctx.num_stages
        tc = ctx.tile_cache
        dout = dout.contiguous()
        _dev = q.device

        D  = torch.empty(B, H, N, device=_dev, dtype=torch.float32)
        g  = (B * H, triton.cdiv(N, BN))

        _compute_D_v15[g](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            D.stride(0),    D.stride(1),    D.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        blocks_n = (N + BN - 1) // BN

        dq      = torch.zeros_like(q)
        dy_pre  = torch.zeros_like(y_pre)
        dpb_buf = torch.empty(B * H, blocks_n, J,       device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, J * HD,  device=_dev, dtype=torch.float32)

        _bwd_dq_v15[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            tc['offsets_small'], tc['offsets_large'],
            tc['orig_idx_small'], tc['orig_idx_large'], tc['phase_map_large'],
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            blocks_n * J,   J,
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            blocks_n*J*HD,  J*HD,
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            J_SMALL_VAL=tc['j_small_n'], J_LARGE_VAL=tc['j_large_n'],
            num_warps=NW, num_stages=NS,
        )
        dpb = dpb_buf.view(B, H, blocks_n, J).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, J, HD).sum(dim=(0, 1, 2)).contiguous()

        dk     = torch.zeros_like(k)
        dv     = torch.zeros_like(v)
        dz_pre = torch.zeros_like(z_pre)
        phase_base_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        phase_gain_buf = torch.empty(B * H, blocks_n, J_LARGE * 2,
                                     device=_dev, dtype=torch.float32)
        stride_buf_bh  = blocks_n * J_LARGE * 2
        stride_buf_blk = J_LARGE * 2

        _bwd_dkdv_v15[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, lse, D,
            dk, dv,
            phase_base_buf, phase_gain_buf,
            dz_pre,
            tc['offsets_small'], tc['offsets_large'],
            tc['orig_idx_small'], tc['orig_idx_large'], tc['phase_map_large'],
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
            J_SMALL_VAL=tc['j_small_n'], J_LARGE_VAL=tc['j_large_n'],
            num_warps=NW, num_stages=NS,
        )

        def _reduce_phase_buf(buf):
            r = buf.view(B, H, blocks_n, J_LARGE, 2).sum(dim=(0, 2))
            return r.permute(1, 0, 2).contiguous()
        d_phase_base = _reduce_phase_buf(phase_base_buf)
        d_phase_gain = _reduce_phase_buf(phase_gain_buf)

        return (dq, dk, dv,
                dpb, dse, d_phase_base, d_phase_gain, dy_pre, dz_pre)


def dsqg_attention_v15(q, k, v, pos_bias, scale_embed,
                       phase_base, phase_gain, y_pre, z_pre):
    """
    q, k, v:       [B, H, N, HD]   bfloat16
    pos_bias:      [96, H]          float32
    scale_embed:   [96, HD]         float32
    phase_base:    [79, H, 2]       float32
    phase_gain:    [79, H, 2]       float32
    y_pre:         [B, H, N, 2]    float32
    z_pre:         [B, H, N, 2]    float32
    Returns:       [B, H, N, HD]   bfloat16
    """
    orig = q.dtype
    if orig != torch.bfloat16:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = _DSQGFnV15.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
    )
    return out if orig == torch.bfloat16 else out.to(orig)


# ─────────────────────────────────────────────────────────────────────────────
# Module — drop-in replacement for DSQGAttentionV8_H100
# ─────────────────────────────────────────────────────────────────────────────

class DSQGAttentionV15(nn.Module):
    """
    DSQG V15: K-Tile Coalesced, Two-Segment Kernels.

    Drop-in replacement for DSQGAttentionV8_H100.
    J=96 offsets + MOVT(r=2) + QK-OVT + NPCI.
    J_SMALL=17 (delta<=28, no MOVT), J_LARGE=79 (delta>=48, MOVT applied).
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

        self._tile_map = _build_tile_map(ALL_OFFSETS, 64)

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
                             q, self.query_probes.to(q.dtype)).mul(sc).float().contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr',
                             k, self.key_probes.to(k.dtype)).mul(sc).float().contiguous()

        out = dsqg_attention_v15(q, k, v,
                                 self.pos_bias, self.scale_embed,
                                 self.phase_base, self.phase_gain,
                                 y_pre, z_pre)

        out      = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def tile_map_summary(self):
        lines = []
        for tile_idx, entries in self._tile_map:
            offsets = [delta for _, delta in entries]
            lines.append(f"  tile {tile_idx:3d}: {len(entries)} offsets, "
                         f"delta=[{min(offsets)}..{max(offsets)}]")
        return "\n".join(lines)

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()
            se   = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            phb  = self.phase_base.detach().cpu()
            phg  = self.phase_gain.detach().cpu()
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


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(device='cuda'):
    from dsqg_attention_v8_dynamic_j import dsqg_attention_v8

    print("=" * 70)
    print("DSQG V15 — K-Tile Coalesced Correctness Tests")
    print("=" * 70)

    cfgs = [
        (1, 4,   64, 64, "tiny D=256 H=4"),
        (1, 4,  128, 64, "small D=256 H=4"),
        (2, 4,  512, 64, "mid D=256 H=4"),
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
        qpr = torch.randn(2, HD, device=device, dtype=torch.float32) * 0.1
        kpr = torch.randn(2, HD, device=device, dtype=torch.float32) * 0.1
        sc  = HD ** -0.5
        y   = torch.einsum('bhnd,rd->bhnr', q.float(), qpr).mul(sc).contiguous()
        z   = torch.einsum('bhnd,rd->bhnr', k.float(), kpr).mul(sc).contiguous()

        ref = dsqg_attention_v8(q.clone(), k.clone(), v.clone(), pb, se, phb, phg, y, z)
        out = dsqg_attention_v15(q.clone(), k.clone(), v.clone(), pb, se, phb, phg, y, z)
        fe  = (ref.float() - out.float()).abs().max().item()
        ok  = fe < 0.001
        if not ok: ok_all = False
        print(f"  {lbl:28s}  fwd_err={fe:.6f}  {'PASS' if ok else 'FAIL'}")

    print()
    print("  Module forward+backward (D=256, H=4, N=128):")
    torch.manual_seed(99)
    model = DSQGAttentionV15(256, 4, seq_len=128, dropout=0.0).to(device)
    x = torch.randn(1, 128, 256, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grad_ok = model.qkv_proj.weight.grad is not None
    print(f"  {'module fwd+bwd':28s}  out_shape={tuple(out.shape)}  grad_ok={grad_ok}  "
          f"{'PASS' if grad_ok else 'FAIL'}")
    if not grad_ok: ok_all = False

    print()
    print("  Tile map:")
    print(model.tile_map_summary())

    print("=" * 70)
    print(f"{'ALL PASSED' if ok_all else 'SOME FAILED'}")
    return ok_all


if __name__ == "__main__":
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ok = run_tests(device)
    sys.exit(0 if ok else 1)
