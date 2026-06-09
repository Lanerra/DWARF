"""
DSQG Attention V20-bf16-SE — grouped sparse DSQG with R=4 MOVT.

This corrected source keeps the working V19/V18 grouped Triton kernels and
updates the versioning contract to match the implementation: the sparse MOVT
path uses sequential per-plane Givens rotations for R_PLANES=4, with
plane channel pairs spread across the head dimension. The previously
advertised dense matmul MOVT path is intentionally not claimed here, because for
R=4 the sequential path is usually cheaper than constructing and applying a
composite rotation matrix.

Public classes:
  DSQGAttentionV19: original class name kept for checkpoint/script compatibility.
  DSQGAttentionV20: alias to DSQGAttentionV19, exposing the corrected V20 module
                    name without changing parameter names or state_dict layout.
"""


import math
import os

# Triton 3.5+ compatibility for module-scope constants referenced by @jit kernels.
# Force this on; allowing parent env to turn it off just creates brittle launch failures.
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

R_PLANES = 4
_LOG2E = tl.constexpr(1.4426950408889634)
NPCI_THETA_MAX = 0.25
NPCI_THETA_INIT = 0.01

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


def _normalize_plane_shift(r_planes: int, hd: int, plane_shift: int = 0) -> int:
    """Validate the within-segment MOVT channel-pair shift used for staggering."""
    if isinstance(plane_shift, bool):
        raise TypeError("plane_shift must be an integer, got bool")
    try:
        shift = int(plane_shift.__index__())
    except AttributeError as exc:
        raise TypeError(f"plane_shift must be an integer, got {type(plane_shift).__name__}") from exc
    except TypeError as exc:
        raise TypeError(f"plane_shift must be an integer, got {type(plane_shift).__name__}") from exc
    if hd < 2 * r_planes:
        raise ValueError(f"head_dim={hd} must be >= 2*r_planes={2 * r_planes}")
    segment = max(2, hd // r_planes)
    if shift < 0 or shift > segment - 2:
        raise ValueError(
            f"plane_shift={shift} must keep each MOVT pair inside its hd/r segment; "
            f"allowed range is [0, {segment - 2}] for head_dim={hd}, r_planes={r_planes}"
        )
    return shift


def _phase_plane_channels(r_planes: int, hd: int, plane_shift: int = 0) -> list[tuple[int, int]]:
    """Spread MOVT plane pairs across the head dimension, with optional layer stagger."""
    shift = _normalize_plane_shift(r_planes, hd, plane_shift)
    stride = max(2, hd // r_planes)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for r in range(r_planes):
        a = min(r * stride + shift, hd - 2)
        b = a + 1
        if a in used or b in used:
            # Extremely small/odd head sizes can collide with the simple spread.
            # Fall back to the next free adjacent pair while preserving determinism.
            found = False
            for cand in range(0, hd - 1):
                if cand not in used and cand + 1 not in used:
                    a, b = cand, cand + 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not assign {r_planes} disjoint MOVT planes in head_dim={hd}")
        used.add(a); used.add(b)
        pairs.append((a, b))
    return pairs


def _canonicalize_offsets(offsets: list[int], j_small: int, j_large: int) -> tuple[list[int], int, int]:
    """Return offsets sorted into the kernel-required [small | large] partition."""
    vals = []
    for d in offsets:
        if isinstance(d, bool):
            raise TypeError(f"DSQG offsets must be integer positive causal deltas, got bool: {d!r}")
        try:
            ivalue = d.__index__()
        except AttributeError as exc:
            raise TypeError(
                f"DSQG offsets must be integer positive causal deltas, got {type(d).__name__}: {d!r}"
            ) from exc
        except TypeError as exc:
            raise TypeError(
                f"DSQG offsets must be integer positive causal deltas, got {type(d).__name__}: {d!r}"
            ) from exc
        if ivalue <= 0:
            raise ValueError(f"DSQG offsets must be positive causal deltas; got {ivalue}")
        vals.append(int(ivalue))
    if len(set(vals)) != len(vals):
        raise ValueError(f"Duplicate DSQG offsets are not supported: {vals}")
    middle = [d for d in vals if not (d <= 28 or d >= 48)]
    if middle:
        raise ValueError(f"Offsets must be <=28 or >=48 for the DSQG split; got middle offsets {middle}")
    small = sorted(d for d in vals if d <= 28)
    large = sorted(d for d in vals if d >= 48)
    if len(small) != int(j_small) or len(large) != int(j_large):
        raise ValueError(
            f"j_small/j_large mismatch after canonicalization: "
            f"declared ({j_small}, {j_large}) vs actual ({len(small)}, {len(large)})"
        )
    return small + large, len(small), len(large)


def _rms_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize along the last dimension; cheaper than full L2 normalize."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta_h: torch.Tensor) -> torch.Tensor:
    """NPCI rotation: project x_delta onto perpendicular of x, rotate by theta."""
    theta = theta_h.view(1, -1, 1, 1)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    x_hat = x / x_norm
    parallel = (x_delta * x_hat).sum(dim=-1, keepdim=True) * x_hat
    perp = x_delta - parallel
    perp_norm = perp.norm(dim=-1, keepdim=True)
    mask = perp_norm > x_norm * 1e-10
    u_hat = torch.where(mask, perp / perp_norm.clamp(min=1e-30), torch.zeros_like(perp))
    return torch.cos(theta) * x + torch.sin(theta) * x_norm * u_hat


def _orthogonal_phase_probes(r_planes: int, hd: int, plane_shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic orthogonal q/k probes with a pi/2 phase offset per plane."""
    assert hd >= 2 * r_planes
    q = torch.zeros(r_planes, hd, dtype=torch.float32)
    k = torch.zeros(r_planes, hd, dtype=torch.float32)
    for r, (a, b) in enumerate(_phase_plane_channels(r_planes, hd, plane_shift=plane_shift)):
        angle = 2.0 * math.pi * r / max(r_planes, 1)
        q[r, a] = math.cos(angle)
        q[r, b] = math.sin(angle)
        k[r, a] = math.cos(angle + math.pi / 2.0)
        k[r, b] = math.sin(angle + math.pi / 2.0)
    return q, k


def _raw_npci_theta_from_effective(theta: float) -> float:
    theta = float(theta)
    limit = float(NPCI_THETA_MAX)
    if not (0.0 <= abs(theta) < limit):
        raise ValueError(f"NPCI_THETA_INIT={theta} must satisfy abs(theta) < {limit}")
    return math.atanh(theta / limit)


GROUPED_MODE_IDS = {
    'baseline': 0,
    'overlap_slab': 1,
    'overlap_slab_bwd': 3,
}
_UNIMPLEMENTED_GROUPED_MODES = {'packed_kv'}


def _resolve_grouped_mode(mode: str) -> int:
    if not isinstance(mode, str):
        raise TypeError(f"grouped_mode must be a string, got {type(mode).__name__}")
    if mode in _UNIMPLEMENTED_GROUPED_MODES:
        raise NotImplementedError(
            f"grouped_mode='{mode}' is advertised by older configs but is not implemented in this kernel"
        )
    if mode not in GROUPED_MODE_IDS:
        allowed = ', '.join(sorted(GROUPED_MODE_IDS.keys()))
        raise ValueError(f"Unknown grouped_mode='{mode}'. Allowed: {allowed}")
    return GROUPED_MODE_IDS[mode]


def _normalize_int_arg(name: str, value, *, minimum: int) -> int:
    """Normalize integer configuration args without accepting bool/float coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= {minimum}, got bool")
    try:
        ivalue = value.__index__()
    except AttributeError as exc:
        raise TypeError(f"{name} must be an integer >= {minimum}, got {type(value).__name__}") from exc
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer >= {minimum}, got {type(value).__name__}") from exc
    if ivalue < minimum:
        raise ValueError(f"{name}={ivalue} must be >= {minimum}")
    return int(ivalue)


def _normalize_grouping_args(
    k_group_size,
    max_group_size,
    max_group_spread,
    grouped_mode_id: int | None = None,
) -> tuple[int, int, int | None]:
    """Validate sparse grouping knobs and return canonical integer values.

    k_group_size is the compiled kernel capacity: kernels iterate exactly this
    many slots per sparse group. max_group_size is only a planner cap: it may be
    smaller than k_group_size to reduce slab waste, but it can never be larger.
    """
    k_group_size = _normalize_int_arg('k_group_size', k_group_size, minimum=1)

    if max_group_size is None:
        max_group_size = k_group_size
    else:
        max_group_size = _normalize_int_arg('max_group_size', max_group_size, minimum=1)
        if max_group_size > k_group_size:
            raise ValueError(
                f"max_group_size={max_group_size} cannot exceed k_group_size={k_group_size}; "
                "overlap-slab kernels iterate only K_GROUP_SIZE slots per group"
            )

    if grouped_mode_id in (
        GROUPED_MODE_IDS['overlap_slab'],
        GROUPED_MODE_IDS['overlap_slab_bwd'],
    ) and max_group_spread is None:
        # Keep slab tiles bounded for the first streaming/tiling implementation.
        # Larger spreads can be enabled explicitly after profiling/tuning.
        max_group_spread = 64

    if max_group_spread is not None:
        max_group_spread = _normalize_int_arg('max_group_spread', max_group_spread, minimum=0)

    return k_group_size, max_group_size, max_group_spread


def _plan_sparse_groups(
    offsets: list[int],
    j_small: int,
    k_group_size: int,
    max_group_size: int | None = None,
    max_group_spread: int | None = None,
) -> dict:
    """Plan sparse grouping for overlap-slab scheduling.

    This is schedule metadata only; logical index semantics remain unchanged.
    k_group_size is kernel capacity; max_group_size is a planner cap that may
    be smaller than, but never larger than, k_group_size.
    The planner minimizes a row-cost proxy: sum(BN + spread_g), approximated
    by minimizing (group_count * k_group_size + sum(spread_g)) for fixed knobs.
    """
    k_group_size, max_group_size, max_group_spread = _normalize_grouping_args(
        k_group_size=k_group_size,
        max_group_size=max_group_size,
        max_group_spread=max_group_spread,
    )

    sparse = list(range(j_small, len(offsets)))
    sparse.sort(key=lambda i: offsets[i])
    n_sparse = len(sparse)

    if n_sparse == 0:
        return {
            'sparse_order': [],
            'group_start': [],
            'group_len': [],
            'group_dmin': [],
            'group_dmax': [],
            'group_i': [],
            'group_rel': [],
            'max_group_len': 0,
            'max_group_spread': 0,
        }

    gmax = max_group_size

    # DP over sorted sparse offsets.
    inf = 10**18
    dp = [inf] * (n_sparse + 1)
    prev = [None] * (n_sparse + 1)
    dp[0] = 0
    for i0 in range(n_sparse):
        if dp[i0] >= inf:
            continue
        dmin = offsets[sparse[i0]]
        dmax = dmin
        for glen in range(1, gmax + 1):
            i1 = i0 + glen
            if i1 > n_sparse:
                break
            d = offsets[sparse[i1 - 1]]
            dmin = d if d < dmin else dmin
            dmax = d if d > dmax else dmax
            spread = dmax - dmin
            if max_group_spread is not None and spread > max_group_spread:
                break

            # Cost proxy with mild preference toward larger groups.
            cost = spread + (k_group_size / max(glen, 1))
            cand = dp[i0] + cost
            if cand < dp[i1]:
                dp[i1] = cand
                prev[i1] = (i0, i1, dmin, dmax)

    if prev[n_sparse] is None:
        # Fallback: fixed-size chunking of sorted sparse indices.
        sparse_order = sparse
        groups = [(i0, min(i0 + gmax, n_sparse))
                  for i0 in range(0, n_sparse, gmax)]
    else:
        groups_rev = []
        cur = n_sparse
        while cur > 0:
            step = prev[cur]
            if step is None:
                raise RuntimeError('Sparse grouping DP reconstruction failed')
            i0, i1, _dmin, _dmax = step
            groups_rev.append((i0, i1))
            cur = i0
        groups = list(reversed(groups_rev))
        sparse_order = []
        for i0, i1 in groups:
            sparse_order.extend(sparse[i0:i1])

    group_start = []
    group_len = []
    group_dmin = []
    group_dmax = []
    group_i = []
    group_rel = []
    max_len = 0
    max_spread_obs = 0

    cursor = 0
    for i0, i1 in groups:
        idxs = sparse[i0:i1]
        deltas = [offsets[i] for i in idxs]
        dmin = min(deltas)
        dmax = max(deltas)
        rel = [dmax - d for d in deltas]
        spread = dmax - dmin

        group_start.append(cursor)
        group_len.append(len(idxs))
        group_dmin.append(dmin)
        group_dmax.append(dmax)
        group_i.append(idxs)
        group_rel.append(rel)
        max_len = max(max_len, len(idxs))
        max_spread_obs = max(max_spread_obs, spread)
        cursor += len(idxs)

    return {
        'sparse_order': sparse_order,
        'group_start': group_start,
        'group_len': group_len,
        'group_dmin': group_dmin,
        'group_dmax': group_dmax,
        'group_i': group_i,
        'group_rel': group_rel,
        'max_group_len': max_len,
        'max_group_spread': max_spread_obs,
    }


def _verify_sparse_plan(offsets: list[int], j_small: int, plan: dict) -> None:
    sparse_expected = list(range(j_small, len(offsets)))
    sparse_order = list(plan['sparse_order'])

    if sorted(sparse_order) != sparse_expected:
        raise AssertionError('Sparse plan does not include each sparse logical index exactly once')

    for gi, start in enumerate(plan['group_start']):
        glen = plan['group_len'][gi]
        dmax = plan['group_dmax'][gi]
        for t in range(glen):
            idx = sparse_order[start + t]
            rel = plan['group_rel'][gi][t]
            delta = offsets[idx]
            if rel != dmax - delta:
                raise AssertionError(
                    f'Invalid rel mapping at group={gi}, t={t}: rel={rel}, dmax={dmax}, delta={delta}'
                )


# ===========================================================================
# Forward Kernel — with K block grouping on sparse cluster
# ===========================================================================

@triton.jit
def _fwd_v18_grouped(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
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
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,  # K block group size (e.g., 3)
):
    """
    V18 forward kernel with explicit K block grouping.
    
    K block grouping strategy:
    - Offsets J_SMALL_VAL to J_VAL are grouped into K_GROUP_SIZE-sized groups
    - K blocks within each group are stored contiguously in memory
    - Coalesced loads for grouped K blocks
    
    Dense path (J_SMALL = 0-16):
    - 2 K block GEMMs, diagonal extraction into scores[:, 0:21]
    
    Sparse path (J_SMALL to J_VAL = 17-95):
    - K block loads grouped by K_GROUP_SIZE
    - Sequential dots from warm K
    - Loaded into scores[:, 21:96]
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    sc = 1.0 / (HD ** 0.5)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    # ── Tensor Core: batch scale_embed scoring (BF16 for tensor core util) ──
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # Keep in BF16 (native type from memory)
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # ═══ Pass 1: compute all J scores with K block grouping ═══
    scores = tl.where(js[None, :] < J_VAL, se_all, float('-inf'))

    # Dense cluster (unchanged traversal). These are one-key-per-query diagonal
    # row reductions; a tiled tl.dot would compute off-diagonal pairs we discard.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        scores = tl.where((js == i)[None, :], s[:, None] + scores, scores)

    # Sparse cluster — explicit K-block grouped traversal via SPARSE_ORDER.
    # Baseline grouping still has one key per query row, so rowwise reductions
    # are cheaper than materializing a mostly-unused QK tile.
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                s = tl.sum(q * kt, axis=1) * sc
                s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                scores = tl.where((js == i)[None, :], s[:, None] + scores, scores)

    # ── Offline softmax over J dimension ──
    mi = tl.max(scores, axis=1)
    all_invalid = mi == float('-inf')
    safe_mi = tl.where(all_invalid, 0.0, mi)
    exp_s = tl.exp2((scores - safe_mi[:, None]) * _LOG2E)
    exp_s = tl.where(js[None, :] < J_VAL, exp_s, 0.0)
    li = tl.sum(exp_s, axis=1)
    ls = tl.where(li > 0.0, li, 1.0)
    probs = exp_s / ls[:, None]
    lse_val = tl.where(all_invalid, 0.0, safe_mi + tl.log2(ls) * (1.0 / _LOG2E))

    # ═══ Pass 2: weighted V sum with MOVT rotation ═══
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # Dense cluster — no rotation (passthrough)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt

    # Sparse cluster — MOVT rotation (R_PLANES Givens)
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)

                # MOVT: general R_PLANES Givens rotation
                pi_idx = i - J_SMALL_VAL
                vt_rot = vt
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + kp * stride_vn + ch_a * stride_vd, mask=val, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + kp * stride_vn + ch_b * stride_vd, mask=val, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                acc += p_i[:, None] * vt_rot

    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


@triton.jit
def _fwd_v18_overlap_slab(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
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
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,  # maximum sparse offsets per slab group
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
):
    """
    V18 forward kernel with explicit K block grouping.
    
    K block grouping strategy:
    - Offsets J_SMALL_VAL to J_VAL are grouped into K_GROUP_SIZE-sized groups
    - K blocks within each group are stored contiguously in memory
    - Coalesced loads for grouped K blocks
    
    Dense path (J_SMALL = 0-16):
    - 2 K block GEMMs, diagonal extraction into scores[:, 0:21]
    
    Sparse path (J_SMALL to J_VAL = 17-95):
    - K block loads grouped by K_GROUP_SIZE
    - Sequential dots from warm K
    - Loaded into scores[:, 21:96]
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    sc = 1.0 / (HD ** 0.5)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    # ── Tensor Core: batch scale_embed scoring (BF16 for tensor core util) ──
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # Keep in BF16 (native type from memory)
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # ═══ Pass 1: compute all J scores with K block grouping ═══
    scores = tl.where(js[None, :] < J_VAL, se_all, float('-inf'))

    # Dense cluster (unchanged traversal)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        scores = tl.where((js == i)[None, :], s[:, None] + scores, scores)

    # Sparse cluster — real overlap-slab scoring for K.
    # For group G: slab_base0 = n0 - dmax, col(row, i) = row + dmax - delta.
    # Therefore slab_base0 + col = n0 + row - delta = ns - delta exactly.
    # We tile slab columns so qk_tile is [BLOCK_N, SLAB_TILE_N], avoiding a huge
    # [BLOCK_N, BLOCK_SLAB_N] accumulator while still loading each K slab tile
    # once per group and serving every offset whose diagonal lies in that tile.
    rows = tl.arange(0, BLOCK_N)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = n0 - dmax
        real_slab_len = BLOCK_N + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            kpos = slab_base0 + cols
            k_tile_t = tl.load(
                kb + kpos[None, :] * stride_kn + ds[:, None] * stride_kd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (kpos[None, :] >= 0) & (kpos[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(q, k_tile_t, input_precision="tf32") * sc

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = dmax - delta
                diag_col = rows + rel
                kp = ns - delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (kp >= 0) & (kp < N) & nm & valid_slot & in_tile

                s = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                scores = tl.where(((js == i) & valid_slot)[None, :] & in_tile[:, None], s[:, None] + scores, scores)

    # ── Offline softmax over J dimension ──
    mi = tl.max(scores, axis=1)
    all_invalid = mi == float('-inf')
    safe_mi = tl.where(all_invalid, 0.0, mi)
    exp_s = tl.exp2((scores - safe_mi[:, None]) * _LOG2E)
    exp_s = tl.where(js[None, :] < J_VAL, exp_s, 0.0)
    li = tl.sum(exp_s, axis=1)
    ls = tl.where(li > 0.0, li, 1.0)
    probs = exp_s / ls[:, None]
    lse_val = tl.where(all_invalid, 0.0, safe_mi + tl.log2(ls) * (1.0 / _LOG2E))

    # ═══ Pass 2: weighted V sum with MOVT rotation ═══
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # Dense cluster — no rotation (passthrough)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt

    # Sparse cluster — MOVT rotation (R_PLANES Givens)
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = tl.load(SPARSE_ORDER + slot).to(tl.int32)
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)

                # MOVT: general R_PLANES Givens rotation
                pi_idx = i - J_SMALL_VAL
                vt_rot = vt
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + kp * stride_vn + ch_a * stride_vd, mask=val, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + kp * stride_vn + ch_b * stride_vd, mask=val, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                acc += p_i[:, None] * vt_rot

    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


# ===========================================================================
# D computation — sum(dO * O, dim=-1)
# ===========================================================================

@triton.jit
def _compute_D_v18_grouped(
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
    do = tl.load(DO + b * stride_dob + h * stride_doh
                 + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    o = tl.load(O + b * stride_ob + h * stride_oh
                + ns[:, None] * stride_on + ds[None, :] * stride_od,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    tl.store(D + b * stride_db + h * stride_dh + ns * stride_dn,
             tl.sum(do * o, axis=1), mask=nm)


# ===========================================================================
# Backward: dQ + atomic dPOS_BIAS + Tensor Core dSCALE_EMBED + dY_PRE
# ===========================================================================

@triton.jit
def _bwd_dq_v18_grouped(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, O,
    DQ, DPB, DSE, DY_PRE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dyb, stride_dyh, stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
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
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    dob = DO + b * stride_dob + h * stride_doh
    ob = O + b * stride_ob + h * stride_oh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    do = tl.load(dob + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    o = tl.load(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + ns * stride_ln, mask=nm, other=0.0)
    Dval = tl.sum(do * o, axis=1)

    # Pre-compute scale_embed scores via Tensor Core (BF16 for tensor core util)
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # Keep in BF16 (native type from memory)
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # dy_pre accumulators
    dy_pre_0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_1 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_2 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_3 = tl.zeros([BLOCK_N], tl.float32)

    # Accumulator for batched dscale_embed via Tensor Core
    DSV_all = tl.zeros([J_PAD, BLOCK_N], tl.float32)

    # Dense cluster. Rowwise reductions here are diagonal attention terms; using
    # tl.dot would add off-diagonal work instead of reusing useful K/V data.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
        s += se_score_i
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lse > float('-inf')),
                         tl.exp2((s - lse) * _LOG2E), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v = alpha * (dot_rv - Dval)
        dq += ds_v[:, None] * kt * sc
        se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        dq += ds_v[:, None] * se_i_vec[None, :] * sc

        tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                      tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

        row_mask = js == i
        DSV_all = tl.where(row_mask[:, None], ds_v[None, :], DSV_all)

    # Sparse cluster — grouped traversal
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                # Recompute score
                s = tl.sum(q * kt, axis=1) * sc
                s += tl.load(PB + i * stride_pbi + h * stride_pbh)
                se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
                s += se_score_i
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lse > float('-inf')),
                                 tl.exp2((s - lse) * _LOG2E), 0.0)

                # MOVT rotation path
                pi_idx = i - J_SMALL_VAL
                vt_rot = vt
                first_rot = vt

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + kp * stride_vn + ch_a * stride_vd, mask=val, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + kp * stride_vn + ch_b * stride_vd, mask=val, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                    don_a = tl.load(dob + ns * stride_don + ch_a * stride_dod, mask=nm, other=0.0).to(tl.float32)
                    don_b = tl.load(dob + ns * stride_don + ch_b * stride_dod, mask=nm, other=0.0).to(tl.float32)
                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    contrib = dth_r * pg_r * z_p
                    dy_pre_0 = tl.where(r == 0, dy_pre_0 + contrib, dy_pre_0)
                    dy_pre_1 = tl.where(r == 1, dy_pre_1 + contrib, dy_pre_1)
                    dy_pre_2 = tl.where(r == 2, dy_pre_2 + contrib, dy_pre_2)
                    dy_pre_3 = tl.where(r == 3, dy_pre_3 + contrib, dy_pre_3)

                first_rot = vt_rot
                dot_rv = tl.sum(do * first_rot, axis=1)
                ds_v = alpha * (dot_rv - Dval)
                dq += ds_v[:, None] * kt * sc
                se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
                dq += ds_v[:, None] * se_i_vec[None, :] * sc

                tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                              tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

                row_mask = js == i
                DSV_all = tl.where(row_mask[:, None], ds_v[None, :], DSV_all)

    # ── Tensor Core: batched dscale_embed (BF16 for tensor core util) ──
    dse_all = tl.dot(DSV_all.to(tl.bfloat16), q.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # Store dQ
    tl.store(DQ + b * stride_dqb + h * stride_dqh
             + ns[:, None] * stride_dqn + ds[None, :] * stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])

    # Store dSE — atomic. Use a real mask, not just zeroed values: for HD values
    # that are not powers of two (e.g. D768/H16 -> HD=48, BLOCK_HD=64), padded
    # lanes would otherwise address past the end of each scale_embed row and can
    # trigger CUDA illegal memory access.
    tl.atomic_add(DSE + js[:, None] * stride_sei + ds[None, :] * stride_sed,
                  dse_all, mask=(js[:, None] < J_VAL) & dm[None, :], sem="relaxed")

    # Store dy_pre
    dyb = DY_PRE + b * stride_dyb + h * stride_dyh
    tl.store(dyb + ns * stride_dyn + 0, tl.where(nm, dy_pre_0, 0.0), mask=nm)
    tl.store(dyb + ns * stride_dyn + 1, tl.where(nm, dy_pre_1, 0.0), mask=nm)
    if R_PLANES_VAL >= 3:
        tl.store(dyb + ns * stride_dyn + 2, tl.where(nm, dy_pre_2, 0.0), mask=nm)
    if R_PLANES_VAL >= 4:
        tl.store(dyb + ns * stride_dyn + 3, tl.where(nm, dy_pre_3, 0.0), mask=nm)


@triton.jit
def _bwd_dq_v18_overlap_slab(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, O,
    DQ, DPB, DSE, DY_PRE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dyb, stride_dyh, stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
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
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    dob = DO + b * stride_dob + h * stride_doh
    ob = O + b * stride_ob + h * stride_oh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    do = tl.load(dob + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    o = tl.load(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + ns * stride_ln, mask=nm, other=0.0)
    Dval = tl.sum(do * o, axis=1)

    # Pre-compute scale_embed scores via Tensor Core (BF16 for tensor core util)
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # Keep in BF16 (native type from memory)
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # dy_pre accumulators
    dy_pre_0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_1 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_2 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_3 = tl.zeros([BLOCK_N], tl.float32)

    # Accumulator for batched dscale_embed via Tensor Core
    DSV_all = tl.zeros([J_PAD, BLOCK_N], tl.float32)

    # Dense cluster
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
        s += se_score_i
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lse > float('-inf')),
                         tl.exp2((s - lse) * _LOG2E), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v = alpha * (dot_rv - Dval)
        dq += ds_v[:, None] * kt * sc
        se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        dq += ds_v[:, None] * se_i_vec[None, :] * sc

        tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                      tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

        row_mask = js == i
        DSV_all = tl.where(row_mask[:, None], ds_v[None, :], DSV_all)

    # Sparse cluster — real overlap-slab K reuse for dQ.
    # For group G: slab_base0 = n0 - dmax, col(row, i) = row + dmax - delta.
    # qk_tile scores each Q row against a contiguous K slab tile loaded once.
    # Per-offset ds_v values are scattered into dsv_tile on the same diagonals;
    # a single dot(dsv_tile, K_tile) then accumulates the K-score dQ term.
    rows = tl.arange(0, BLOCK_N)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = n0 - dmax
        real_slab_len = BLOCK_N + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            kpos = slab_base0 + cols
            k_tile_t = tl.load(
                kb + kpos[None, :] * stride_kn + ds[:, None] * stride_kd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (kpos[None, :] >= 0) & (kpos[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(q, k_tile_t, input_precision="tf32") * sc
            dsv_tile = tl.zeros([BLOCK_N, SLAB_TILE_N], tl.float32)

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = dmax - delta
                diag_col = rows + rel
                kp = ns - delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (kp >= 0) & (kp < N) & nm & valid_slot & in_tile

                s = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                s += tl.load(PB + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
                s += se_score_i
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lse > float('-inf')),
                                 tl.exp2((s - lse) * _LOG2E), 0.0)

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                # MOVT rotation path
                pi_idx = i - J_SMALL_VAL
                vt_rot = vt
                first_rot = vt

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r, mask=valid_slot, other=0.0)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r, mask=valid_slot, other=0.0)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + kp * stride_vn + ch_a * stride_vd, mask=val, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + kp * stride_vn + ch_b * stride_vd, mask=val, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                    don_a = tl.load(dob + ns * stride_don + ch_a * stride_dod, mask=nm, other=0.0).to(tl.float32)
                    don_b = tl.load(dob + ns * stride_don + ch_b * stride_dod, mask=nm, other=0.0).to(tl.float32)
                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    contrib = dth_r * pg_r * z_p
                    dy_pre_0 = tl.where(r == 0, dy_pre_0 + contrib, dy_pre_0)
                    dy_pre_1 = tl.where(r == 1, dy_pre_1 + contrib, dy_pre_1)
                    dy_pre_2 = tl.where(r == 2, dy_pre_2 + contrib, dy_pre_2)
                    dy_pre_3 = tl.where(r == 3, dy_pre_3 + contrib, dy_pre_3)

                first_rot = vt_rot
                dot_rv = tl.sum(do * first_rot, axis=1)
                ds_v = alpha * (dot_rv - Dval)

                # dQ contribution through scale_embed is per-offset and unchanged.
                se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
                dq += ds_v[:, None] * se_i_vec[None, :] * sc

                tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                              tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

                # Sparse dscale_embed is accumulated immediately because this tiled
                # path only has the true per-row ds_v inside the tile containing
                # each diagonal. Dense offsets still use DSV_all + final batched dot.
                dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
                tl.atomic_add(DSE + i * stride_sei + ds * stride_sed,
                              dse_i, mask=valid_slot & dm, sem="relaxed")
                dsv_tile += ds_v[:, None] * (cols[None, :] == diag_col[:, None]).to(tl.float32)

            dq += tl.dot(dsv_tile, tl.trans(k_tile_t), input_precision="tf32") * sc

    # ── Tensor Core: batched dscale_embed (BF16 for tensor core util) ──
    dse_all = tl.dot(DSV_all.to(tl.bfloat16), q.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # Store dQ
    tl.store(DQ + b * stride_dqb + h * stride_dqh
             + ns[:, None] * stride_dqn + ds[None, :] * stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])

    # Store dSE — atomic. Use a real mask, not just zeroed values: for HD values
    # that are not powers of two (e.g. D768/H16 -> HD=48, BLOCK_HD=64), padded
    # lanes would otherwise address past the end of each scale_embed row and can
    # trigger CUDA illegal memory access.
    tl.atomic_add(DSE + js[:, None] * stride_sei + ds[None, :] * stride_sed,
                  dse_all, mask=(js[:, None] < J_VAL) & dm[None, :], sem="relaxed")

    # Store dy_pre
    dyb = DY_PRE + b * stride_dyb + h * stride_dyh
    tl.store(dyb + ns * stride_dyn + 0, tl.where(nm, dy_pre_0, 0.0), mask=nm)
    tl.store(dyb + ns * stride_dyn + 1, tl.where(nm, dy_pre_1, 0.0), mask=nm)
    if R_PLANES_VAL >= 3:
        tl.store(dyb + ns * stride_dyn + 2, tl.where(nm, dy_pre_2, 0.0), mask=nm)
    if R_PLANES_VAL >= 4:
        tl.store(dyb + ns * stride_dyn + 3, tl.where(nm, dy_pre_3, 0.0), mask=nm)


# ===========================================================================
# Backward: dK + dV + atomic d_phase_base + atomic d_phase_gain + dZ_PRE
# ===========================================================================

@triton.jit
def _bwd_dkdv_v18_grouped(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, O,
    DK, DV,
    DPHASE_BASE, DPHASE_GAIN,
    DZ_PRE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_dphi, stride_dphh,
    stride_dpgi, stride_dpgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dzb, stride_dzh, stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
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
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh
    ob = O + b * stride_ob + h * stride_oh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    kt = tl.load(kb + ms[:, None] * stride_kn + ds[None, :] * stride_kd,
                  mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vt = tl.load(vb + ms[:, None] * stride_vn + ds[None, :] * stride_vd,
                  mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre_0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_1 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_2 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_3 = tl.zeros([BLOCK_M], tl.float32)

    # Dense cluster. Each key row maps to sparse query diagonals, so row
    # reductions avoid useless off-diagonal tile work.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_ = ms + delta
        val = (np_ >= 0) & (np_ < N) & mm

        qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        on = tl.load(ob + np_[:, None] * stride_on + ds[None, :] * stride_od,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                      mask=val, other=0.0)
        Dn = tl.sum(don * on, axis=1)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.float32)
        s = tl.sum(qn * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        s += tl.sum(qn * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lsen > float('-inf')),
                        tl.exp2((s - lsen) * _LOG2E), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v = alpha * (dot_rv - Dn)
        dk += ds_v[:, None] * qn * sc
        dv += alpha[:, None] * don

    # Sparse cluster — grouped traversal
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                np_ = ms + delta
                val = (np_ >= 0) & (np_ < N) & mm

                qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                              mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                on = tl.load(ob + np_[:, None] * stride_on + ds[None, :] * stride_od,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                               mask=val, other=0.0)
                Dn = tl.sum(don * on, axis=1)

                se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                                mask=dm, other=0.0).to(tl.float32)
                s = tl.sum(qn * kt, axis=1) * sc
                s += tl.load(PB + i * stride_pbi + h * stride_pbh)
                s += tl.sum(qn * se_i[None, :], axis=1) * sc
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lsen > float('-inf')),
                                 tl.exp2((s - lsen) * _LOG2E), 0.0)

                pi_idx = i - J_SMALL_VAL

                dz_pre_0_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_1_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_2_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_3_local = tl.zeros([BLOCK_M], tl.float32)

                dv_c = alpha[:, None] * don
                vt_rot = vt

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_r = tl.load(yb + np_ * stride_yn + r, mask=val, other=0.0)
                    z_r = tl.load(zb + ms * stride_zn + r, mask=mm, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_r * z_r
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + ms * stride_vn + ch_a * stride_vd, mask=mm, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + ms * stride_vn + ch_b * stride_vd, mask=mm, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                    don_a = tl.load(dob + np_ * stride_don + ch_a * stride_dod, mask=val, other=0.0).to(tl.float32)
                    don_b = tl.load(dob + np_ * stride_don + ch_b * stride_dod, mask=val, other=0.0).to(tl.float32)
                    dv_c = tl.where(mask_a[None, :],
                                    (alpha * (cos_r * don_a + sin_r * don_b))[:, None], dv_c)
                    dv_c = tl.where(mask_b[None, :],
                                    (alpha * (-sin_r * don_a + cos_r * don_b))[:, None], dv_c)

                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    tl.atomic_add(DPHASE_BASE + pi_idx * stride_dphi + h * stride_dphh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r, 0.0)), sem="relaxed")
                    tl.atomic_add(DPHASE_GAIN + pi_idx * stride_dpgi + h * stride_dpgh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r * y_r * z_r, 0.0)), sem="relaxed")

                    contrib = tl.where(val.to(tl.int1), dth_r * pg_r * y_r, 0.0)
                    dz_pre_0_local = tl.where(r == 0, dz_pre_0_local + contrib, dz_pre_0_local)
                    dz_pre_1_local = tl.where(r == 1, dz_pre_1_local + contrib, dz_pre_1_local)
                    dz_pre_2_local = tl.where(r == 2, dz_pre_2_local + contrib, dz_pre_2_local)
                    dz_pre_3_local = tl.where(r == 3, dz_pre_3_local + contrib, dz_pre_3_local)

                dot_rv = tl.sum(don * vt_rot, axis=1)
                ds_v = alpha * (dot_rv - Dn)
                dk += ds_v[:, None] * qn * sc
                dv += dv_c

                dz_pre_0 += dz_pre_0_local
                dz_pre_1 += dz_pre_1_local
                dz_pre_2 += dz_pre_2_local
                dz_pre_3 += dz_pre_3_local

    tl.store(DK + b * stride_dkb + h * stride_dkh
             + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])
    tl.store(DV + b * stride_dvb + h * stride_dvh
             + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])

    dzb = DZ_PRE + b * stride_dzb + h * stride_dzh
    tl.store(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre_0, 0.0), mask=mm)
    tl.store(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre_1, 0.0), mask=mm)
    if R_PLANES_VAL >= 3:
        tl.store(dzb + ms * stride_dzn + 2, tl.where(mm, dz_pre_2, 0.0), mask=mm)
    if R_PLANES_VAL >= 4:
        tl.store(dzb + ms * stride_dzn + 3, tl.where(mm, dz_pre_3, 0.0), mask=mm)


@triton.jit
def _bwd_dkdv_v18_overlap_slab(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, O,
    DK, DV,
    DPHASE_BASE, DPHASE_GAIN,
    DZ_PRE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_dphi, stride_dphh,
    stride_dpgi, stride_dpgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dzb, stride_dzh, stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_SMALL_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
):
    """dK/dV backward with sparse query-slab tiling for overlap_slab_bwd mode.

    The dense offsets keep the direct write-once path.  Sparse offsets are
    grouped by offset spread; each group loads contiguous Q slabs and reuses the
    QK tile plus a dscore slab to accumulate the K-score dK term with one
    dot(dscore_slab, Q_slab) per tile.
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    m0 = blk * BLOCK_M
    ms = m0 + tl.arange(0, BLOCK_M)
    mm = ms < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    sc = 1.0 / (HD ** 0.5)

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh
    ob = O + b * stride_ob + h * stride_oh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    kt = tl.load(kb + ms[:, None] * stride_kn + ds[None, :] * stride_kd,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vt = tl.load(vb + ms[:, None] * stride_vn + ds[None, :] * stride_vd,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre_0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_1 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_2 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_3 = tl.zeros([BLOCK_M], tl.float32)

    # Dense cluster: direct write-once accumulation.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_ = ms + delta
        val = (np_ >= 0) & (np_ < N) & mm

        qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                      mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        on = tl.load(ob + np_[:, None] * stride_on + ds[None, :] * stride_od,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                       mask=val, other=0.0)
        Dn = tl.sum(don * on, axis=1)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.float32)
        s = tl.sum(qn * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        s += tl.sum(qn * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))
        alpha = tl.where(val & (lsen > float('-inf')),
                         tl.exp2((s - lsen) * _LOG2E), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v = alpha * (dot_rv - Dn)
        dk += ds_v[:, None] * qn * sc
        dv += alpha[:, None] * don

    rows = tl.arange(0, BLOCK_M)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)

    # Sparse cluster: query-slab reuse for qk and dK score path.
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = m0 + dmin
        real_slab_len = BLOCK_M + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            qpos_tile = slab_base0 + cols
            q_tile_t = tl.load(
                qb + qpos_tile[None, :] * stride_qn + ds[:, None] * stride_qd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (qpos_tile[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(kt, q_tile_t, input_precision="tf32") * sc
            dsv_tile = tl.zeros([BLOCK_M, SLAB_TILE_N], tl.float32)

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = delta - dmin
                diag_col = rows + rel
                np_ = ms + delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (np_ >= 0) & (np_ < N) & mm & valid_slot & in_tile

                s_qk = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                              mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                on = tl.load(ob + np_[:, None] * stride_on + ds[None, :] * stride_od,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                               mask=val, other=0.0)
                Dn = tl.sum(don * on, axis=1)

                se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                               mask=dm, other=0.0).to(tl.float32)
                s = s_qk
                s += tl.load(PB + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                s += tl.sum(qn * se_i[None, :], axis=1) * sc
                s = tl.where(val.to(tl.int1), s, float('-inf'))
                alpha = tl.where(val & (lsen > float('-inf')),
                                 tl.exp2((s - lsen) * _LOG2E), 0.0)

                pi_idx = i - J_SMALL_VAL
                dz_pre_0_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_1_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_2_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_3_local = tl.zeros([BLOCK_M], tl.float32)

                dv_c = alpha[:, None] * don
                vt_rot = vt
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_r = tl.load(yb + np_ * stride_yn + r, mask=val, other=0.0)
                    z_r = tl.load(zb + ms * stride_zn + r, mask=mm, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r, mask=valid_slot, other=0.0)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r, mask=valid_slot, other=0.0)
                    theta_r = pb_r + pg_r * y_r * z_r
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.load(vb + ms * stride_vn + ch_a * stride_vd, mask=mm, other=0.0).to(tl.float32)
                    v_b = tl.load(vb + ms * stride_vn + ch_b * stride_vd, mask=mm, other=0.0).to(tl.float32)
                    vt_rot = tl.where(mask_a[None, :], (cos_r * v_a - sin_r * v_b)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_r * v_a + cos_r * v_b)[:, None], vt_rot)

                    don_a = tl.load(dob + np_ * stride_don + ch_a * stride_dod, mask=val, other=0.0).to(tl.float32)
                    don_b = tl.load(dob + np_ * stride_don + ch_b * stride_dod, mask=val, other=0.0).to(tl.float32)
                    dv_c = tl.where(mask_a[None, :],
                                    (alpha * (cos_r * don_a + sin_r * don_b))[:, None], dv_c)
                    dv_c = tl.where(mask_b[None, :],
                                    (alpha * (-sin_r * don_a + cos_r * don_b))[:, None], dv_c)

                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    tl.atomic_add(DPHASE_BASE + pi_idx * stride_dphi + h * stride_dphh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r, 0.0)), mask=valid_slot, sem="relaxed")
                    tl.atomic_add(DPHASE_GAIN + pi_idx * stride_dpgi + h * stride_dpgh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r * y_r * z_r, 0.0)), mask=valid_slot, sem="relaxed")

                    contrib = tl.where(val.to(tl.int1), dth_r * pg_r * y_r, 0.0)
                    dz_pre_0_local = tl.where(r == 0, dz_pre_0_local + contrib, dz_pre_0_local)
                    dz_pre_1_local = tl.where(r == 1, dz_pre_1_local + contrib, dz_pre_1_local)
                    dz_pre_2_local = tl.where(r == 2, dz_pre_2_local + contrib, dz_pre_2_local)
                    dz_pre_3_local = tl.where(r == 3, dz_pre_3_local + contrib, dz_pre_3_local)

                dot_rv = tl.sum(don * vt_rot, axis=1)
                ds_v = alpha * (dot_rv - Dn)
                dsv_tile += ds_v[:, None] * (cols[None, :] == diag_col[:, None]).to(tl.float32)
                dv += dv_c
                dz_pre_0 += dz_pre_0_local
                dz_pre_1 += dz_pre_1_local
                dz_pre_2 += dz_pre_2_local
                dz_pre_3 += dz_pre_3_local

            dk += tl.dot(dsv_tile, tl.trans(q_tile_t), input_precision="tf32") * sc

    tl.store(DK + b * stride_dkb + h * stride_dkh
             + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])
    tl.store(DV + b * stride_dvb + h * stride_dvh
             + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])

    dzb = DZ_PRE + b * stride_dzb + h * stride_dzh
    tl.store(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre_0, 0.0), mask=mm)
    tl.store(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre_1, 0.0), mask=mm)
    if R_PLANES_VAL >= 3:
        tl.store(dzb + ms * stride_dzn + 2, tl.where(mm, dz_pre_2, 0.0), mask=mm)
    if R_PLANES_VAL >= 4:
        tl.store(dzb + ms * stride_dzn + 3, tl.where(mm, dz_pre_3, 0.0), mask=mm)


# ===========================================================================
# Autograd Function
# ===========================================================================

class _DSQGV18GroupedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                j_val, j_small, j_large,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                k_group_size, grouped_mode_id, verify_mode, max_group_spread_observed,
                plane_shift):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        if not q.is_cuda:
            raise RuntimeError("DSQGAttentionV19/20 requires CUDA/Triton tensors")

        # Kernels consume explicit strides, so Q/K/V can stay as qkv_proj views.
        y_pre = y_pre.contiguous()
        z_pre = z_pre.contiguous()
        pos_bias = pos_bias.contiguous()
        scale_embed = scale_embed.contiguous()

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 8, 2
            else:       BLOCK_N, _nw, _ns = 64, 4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 4, 2
            else:       BLOCK_N, _nw, _ns = 32, 4, 2
        elif HD <= 256:
            if _sm90:   BLOCK_N, _nw, _ns = 32, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 32, 4, 2
            else:       BLOCK_N, _nw, _ns = 16, 4, 2
        else:
            if _sm90:   BLOCK_N, _nw, _ns = 16, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 16, 4, 2
            else:       BLOCK_N, _nw, _ns = 8, 4, 2

        BLOCK_HD = _next_pow2(HD)
        J_PAD = _next_pow2(j_val)
        if BLOCK_N < 16 or BLOCK_HD < 16 or J_PAD < 16:
            raise RuntimeError(
                f"Unsupported DSQG v20 launch shape for tl.dot: BLOCK_N={BLOCK_N}, "
                f"BLOCK_HD={BLOCK_HD}, J_PAD={J_PAD}; each must be >= 16"
            )

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        grid = (B * H, triton.cdiv(N, BLOCK_N))

        if grouped_mode_id in (
            GROUPED_MODE_IDS['overlap_slab'],
            GROUPED_MODE_IDS['overlap_slab_bwd'],
        ):
            _fwd_v18_overlap_slab[grid](
                q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
                y_pre, z_pre, out, lse,
                offsets_dev, sparse_order_dev, group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
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
                J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BLOCK_N + int(max_group_spread_observed)),
                SLAB_TILE_N=32,
                num_warps=_nw, num_stages=_ns,
            )
        else:
            _fwd_v18_grouped[grid](
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
                J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                num_warps=_nw, num_stages=_ns,
            )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev, sparse_order_dev,
                              group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev)
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _nw
        ctx.num_stages = _ns
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
        ctx.J_PAD = J_PAD
        ctx.k_group_size = k_group_size
        ctx.grouped_mode_id = grouped_mode_id
        ctx.verify_mode = verify_mode
        ctx.max_group_spread_observed = max_group_spread_observed
        ctx.plane_shift = plane_shift
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pos_bias, scale_embed,
         phase_base, phase_gain, y_pre, z_pre,
         out, lse, offsets_dev, sparse_order_dev,
         group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        BN = ctx.BLOCK_N
        BHD = ctx.BLOCK_HD
        NW = ctx.num_warps
        NS = ctx.num_stages
        j_val = ctx.j_val
        j_small = ctx.j_small
        j_large = ctx.j_large
        J_PAD = ctx.J_PAD
        k_group_size = ctx.k_group_size
        plane_shift = ctx.plane_shift

        dout = dout.contiguous()
        _dev = q.device
        grid = (B * H, triton.cdiv(N, BN))

        # dQ + atomic dpos_bias + atomic dscale_embed + dy_pre
        dq = torch.empty_like(q)
        dy_pre = torch.empty_like(y_pre)
        dpb = torch.zeros(j_val, H, device=_dev, dtype=torch.float32)
        dse = torch.zeros(j_val, HD, device=_dev, dtype=torch.float32)

        if ctx.grouped_mode_id == GROUPED_MODE_IDS['overlap_slab_bwd']:
            _bwd_dq_v18_overlap_slab[grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, out,
                dq, dpb, dse, dy_pre,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
                J_VAL=j_val, J_SMALL_VAL=j_small,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BN + int(ctx.max_group_spread_observed)),
                SLAB_TILE_N=32,
                num_warps=NW, num_stages=NS,
            )
        else:
            _bwd_dq_v18_grouped[grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, out,
                dq, dpb, dse, dy_pre,
                offsets_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
                J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                num_warps=NW, num_stages=NS,
            )

        # dK + dV + atomic dphase_base + atomic dphase_gain + dz_pre
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dz_pre = torch.empty_like(z_pre)
        d_phase_base_buf = torch.zeros_like(phase_base)
        d_phase_gain_buf = torch.zeros_like(phase_gain)

        if ctx.grouped_mode_id == GROUPED_MODE_IDS['overlap_slab_bwd']:
            _bwd_dkdv_v18_overlap_slab[grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, out,
                dk, dv,
                d_phase_base_buf, d_phase_gain_buf,
                dz_pre,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                d_phase_base_buf.stride(0), d_phase_base_buf.stride(1),
                d_phase_gain_buf.stride(0), d_phase_gain_buf.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
                J_SMALL_VAL=j_small,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BN + int(ctx.max_group_spread_observed)),
                SLAB_TILE_N=32,
                num_warps=NW, num_stages=NS,
            )
        else:
            _bwd_dkdv_v18_grouped[grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, out,
                dk, dv,
                d_phase_base_buf, d_phase_gain_buf,
                dz_pre,
                offsets_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                d_phase_base_buf.stride(0), d_phase_base_buf.stride(1),
                d_phase_gain_buf.stride(0), d_phase_gain_buf.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
                J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                num_warps=NW, num_stages=NS,
        )

        return (dq, dk, dv,
                dpb, dse, d_phase_base_buf, d_phase_gain_buf, dy_pre, dz_pre,
                None, None, None,
                None, None,
                None, None, None, None,
                None, None, None,
                None, None)


# ===========================================================================
# Public API
# ===========================================================================

def dsqg_attention_v18_grouped(q, k, v, pos_bias, scale_embed,
                               phase_base, phase_gain, y_pre, z_pre,
                               j_val, j_small, j_large,
                               offsets_dev, sparse_order_dev,
                               group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                               k_group_size, grouped_mode_id, verify_mode, max_group_spread_observed,
                               plane_shift=0):
    orig_dtype = q.dtype
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    if k.dtype != torch.bfloat16:
        k = k.to(torch.bfloat16)
    if v.dtype != torch.bfloat16:
        v = v.to(torch.bfloat16)

    out = _DSQGV18GroupedFn.apply(
        q, k, v,
        pos_bias.float(), scale_embed.float(),
        phase_base.float(), phase_gain.float(),
        y_pre.float(), z_pre.float(),
        j_val, j_small, j_large,
        offsets_dev, sparse_order_dev,
        group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
        k_group_size, grouped_mode_id, verify_mode, max_group_spread_observed,
        plane_shift,
    )
    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


# ===========================================================================
# Module — DSQGAttentionV19/V20-compatible wrapper
# ===========================================================================

class DSQGAttentionV19(nn.Module):

    def __init__(self, embedding_dim, num_heads, offsets, j_small, j_large,
                 seq_len=2048, dropout=0.1, k_group_size=3,
                 # overlap_slab_bwd is retained for experiments, but on RTX 4090
                 # D512/L10 J=32 triads it is 4-6x slower than the direct sparse
                 # traversal because the planner mostly emits singleton groups.
                 grouped_mode='overlap_slab', verify_mode=False,
                 max_group_size=None, max_group_spread=None,
                 plane_shift=0):
        super().__init__()
        D = embedding_dim
        H = num_heads
        self.num_heads = H
        if D % H != 0:
            raise ValueError(f"embedding_dim={D} must be divisible by num_heads={H}")
        self.head_dim = D // H
        HD = self.head_dim
        if HD < 16:
            raise ValueError(f"head_dim={HD} is too small for Triton tl.dot paths; require head_dim >= 16")
        self.seq_len = seq_len
        self.grouped_mode = grouped_mode
        self.grouped_mode_id = _resolve_grouped_mode(grouped_mode)
        self.verify_mode = bool(verify_mode)
        k_group_size, max_group_size, max_group_spread = _normalize_grouping_args(
            k_group_size=k_group_size,
            max_group_size=max_group_size,
            max_group_spread=max_group_spread,
            grouped_mode_id=self.grouped_mode_id,
        )
        self.k_group_size = k_group_size
        self.max_group_size = max_group_size
        self.max_group_spread = max_group_spread

        assert HD >= 2 * R_PLANES
        self.plane_shift = _normalize_plane_shift(R_PLANES, HD, plane_shift)

        offsets, j_small, j_large = _canonicalize_offsets(list(offsets), j_small, j_large)
        j_val = len(offsets)
        assert j_small + j_large == j_val
        if _next_pow2(j_val) < 16:
            raise ValueError(f"j_val={j_val} is too small for Triton tl.dot paths; require padded J >= 16")
        self.j_val = j_val
        self.j_small = j_small
        self.j_large = j_large

        self.register_buffer(
            'offsets_dev',
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )
        plan = _plan_sparse_groups(
            offsets=offsets,
            j_small=j_small,
            k_group_size=k_group_size,
            max_group_size=max_group_size,
            max_group_spread=max_group_spread,
        )
        if int(plan['max_group_len']) > int(max_group_size):
            raise ValueError(
                f"sparse planner produced max_group_len={plan['max_group_len']} > max_group_size={max_group_size}"
            )
        if int(plan['max_group_len']) > int(k_group_size):
            raise ValueError(
                f"sparse planner produced max_group_len={plan['max_group_len']} > k_group_size={k_group_size}"
            )
        if self.verify_mode:
            _verify_sparse_plan(offsets, j_small, plan)

        sparse_order = plan['sparse_order']
        self.register_buffer(
            'sparse_order_dev',
            torch.tensor(sparse_order, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'group_start_dev',
            torch.tensor(plan['group_start'], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'group_len_dev',
            torch.tensor(plan['group_len'], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'group_dmin_dev',
            torch.tensor(plan['group_dmin'], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            'group_dmax_dev',
            torch.tensor(plan['group_dmax'], dtype=torch.int32),
            persistent=False,
        )
        self.max_group_len = int(plan['max_group_len'])
        self.max_group_spread_observed = int(plan['max_group_spread'])
        self.num_sparse_groups = int(len(plan['group_len']))

        self.qkv_proj = nn.Linear(D, 3 * D, bias=True)
        self.out_proj = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, H)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(j_val, HD))
        self.if_gain = nn.Parameter(torch.ones(H))

        self.phase_base = nn.Parameter(
            torch.randn(max(j_large, 1), H, R_PLANES) * 0.1)
        self.phase_gain = nn.Parameter(
            torch.randn(max(j_large, 1), H, R_PLANES) * 0.001)
        self.phase_gate = nn.Parameter(torch.zeros(max(j_large, 1)))

        query_probes, key_probes = _orthogonal_phase_probes(R_PLANES, HD, plane_shift=self.plane_shift)
        self.query_probes = nn.Parameter(query_probes)
        self.key_probes = nn.Parameter(key_probes)

        raw_theta_init = _raw_npci_theta_from_effective(NPCI_THETA_INIT)
        self.npci_theta_k = nn.Parameter(torch.full((H,), raw_theta_init))
        self.npci_theta_v = nn.Parameter(torch.full((H,), raw_theta_init))

        self.dropout = nn.Dropout(dropout)

    def reset_phase_probes_(self) -> None:
        """Restore deterministic orthogonal phase probes after external init passes."""
        query_probes, key_probes = _orthogonal_phase_probes(R_PLANES, self.head_dim, plane_shift=self.plane_shift)
        with torch.no_grad():
            self.query_probes.copy_(query_probes.to(device=self.query_probes.device, dtype=self.query_probes.dtype))
            self.key_probes.copy_(key_probes.to(device=self.key_probes.device, dtype=self.key_probes.dtype))

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        # Kernels consume explicit strides, so Q/K/V can stay as qkv_proj views.
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            theta_k = NPCI_THETA_MAX * torch.tanh(self.npci_theta_k)
            theta_v = NPCI_THETA_MAX * torch.tanh(self.npci_theta_v)
            k = npci_rotate(k, k_delta, theta_k).to(dtype=q.dtype)
            v = npci_rotate(v, v_delta, theta_v).to(dtype=q.dtype)

        # RMS-normalized dot probes avoid two full L2-normalization passes over
        # [B,H,N,HD] while keeping phase inputs scale controlled.  Probe vectors
        # remain cheap L2-normalized trainable parameters.
        q_norm = _rms_normalize_last(q)
        k_norm = _rms_normalize_last(k)
        qp_norm = F.normalize(self.query_probes.float(), dim=-1)
        kp_norm = F.normalize(self.key_probes.float(), dim=-1)
        probe_scale = 1.0 / math.sqrt(float(HD))
        y_pre = (torch.einsum('bhnd,rd->bhnr', q_norm, qp_norm) * probe_scale).contiguous()
        z_pre = (torch.einsum('bhnd,rd->bhnr', k_norm, kp_norm) * probe_scale).contiguous()

        gate = torch.sigmoid(self.phase_gate)[:, None, None]
        gated_phase_base = self.phase_base * gate
        gated_phase_gain = self.phase_gain * gate

        out = dsqg_attention_v18_grouped(
            q, k, v,
            self.pos_bias, self.scale_embed,
            gated_phase_base, gated_phase_gain,
            y_pre, z_pre,
            self.j_val, self.j_small, self.j_large,
            self.offsets_dev, self.sparse_order_dev,
            self.group_start_dev, self.group_len_dev, self.group_dmin_dev, self.group_dmax_dev,
            self.k_group_size, self.grouped_mode_id, self.verify_mode,
            self.max_group_spread_observed,
            self.plane_shift,
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))


# Backward-compatible alias for scripts that prefer the V20 name.
DSQGAttentionV20 = DSQGAttentionV19
