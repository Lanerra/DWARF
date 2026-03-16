"""
DSQG Attention V10 — Tiled, coalesced, ground-up rewrite
==========================================================

Ground-up rewrite for the March 15 2026 performance investigation.
V8/V9 profiling: DSQG is 60% of step time, 4.5× slower than FlashAttention.
Root cause: J sequential non-coalesced HBM loads per output position.

V10 fix: process BLOCK_M output positions per program, loading KV in
contiguous tiles of size [BLOCK_M × HD] per offset. Each tile load is
coalesced and reused across BLOCK_M queries simultaneously.

V8 access pattern:
  for each query q:
    for j in range(J):         ← J=20 scattered loads, non-coalesced
      load K[q - delta_j]      ← single row, random address

V10 access pattern:
  for each query tile [n0, n0+BM):
    for j in range(J):         ← J=20 coalesced tile loads
      load K[n0-delta_j : n0-delta_j+BM]  ← contiguous block, 1 transaction
    compute J scores for all BM queries simultaneously via tl.dot

Expected speedup: 2-4× from coalescing alone on 4090 (HBM bandwidth-bound).

What's removed vs V8:
  - MOVT / QK-OVT / NPCI: complex per-offset phase rotations
    Rationale: ablation showed pos_bias + scale_embed do 95% of routing.
    NPCI/MOVT added <1 PPL at large model size. Not worth 300 lines of kernel.
  - Separate complex backward kernel (700 lines in V8)
    Replaced by: torch.autograd for pos_bias and scale_embed gradients.
    KV gradients computed in a clean backward pass (see _DSQGTiled backward).

What's kept:
  - pos_bias [J, H]: learnable per-head routing prior (the relay routing table)
  - scale_embed [J, HD]: Q-matched content filter (zero-init)
  - Online softmax over J offsets (numerically stable)
  - Causal masking via position check (kp >= 0)
  - NPCI available as a Python-level wrapper (npci_rotate) if needed

Architecture (unchanged from V9):
  J=20, OFFSETS = [1,2,3,4,5,6,7,8,9,11,13,15,16,23,32,64,128,256,512,1024]

Usage:
  from dsqg_attention_v10 import DSQGAttentionV10, npci_rotate

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v10.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ── Offset set (J=20, Frobenius-optimal from co-opt search Mar 15 2026) ───────
ALL_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 23, 32, 64, 128, 256, 512, 1024]
J = len(ALL_OFFSETS)   # 20
assert J == 20
MAX_DELTA = max(ALL_OFFSETS)  # 1024

# ── NPCI (pure PyTorch, unchanged — norm-preserving coupled injection) ────────
def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta_h: torch.Tensor) -> torch.Tensor:
    """Norm-preserving rotation: ||output|| = ||x|| exactly.
    x, x_delta: [B, H, N, HD]; theta_h: [H]."""
    theta    = theta_h.view(1, -1, 1, 1)
    x_norm   = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    x_hat    = x / x_norm
    parallel = (x_delta * x_hat).sum(dim=-1, keepdim=True) * x_hat
    perp     = x_delta - parallel
    perp_n   = perp.norm(dim=-1, keepdim=True)
    mask     = perp_n > x_norm * 1e-10
    u_hat    = torch.where(mask, perp / perp_n.clamp(min=1e-30), torch.zeros_like(perp))
    return torch.cos(theta) * x + torch.sin(theta) * x_norm * u_hat


# ── Forward kernel ────────────────────────────────────────────────────────────
#
# Grid: (B*H, ceil(N/BLOCK_M))   — one program per (batch*head, query_tile)
#
# Each program:
#   1. Load Q tile  [BLOCK_M, HD]  — stays in registers throughout
#   2. For each offset j (tl.static_range(J)):
#        kp = tile_positions - delta_j           ← relative positions
#        Load K[kp] tile [BLOCK_M, HD]           ← coalesced if kp contiguous
#        Load V[kp] tile [BLOCK_M, HD]           ← coalesced
#        score[m] = Q[m] · K[m] * scale          ← dot over HD (one per query in tile)
#                 + pos_bias[j, h]               ← scalar bias
#                 + Q[m] · se[j] * scale         ← content filter
#        Online softmax update: mi, li, acc
#   3. Normalize and write output

@triton.jit
def _dsqg_fwd(
    Q, K, V, POS_BIAS, SCALE_EMBED, OUT, LSE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bh    = tl.program_id(0)
    tile  = tl.program_id(1)
    b     = bh // H
    h     = bh %  H
    n0    = tile * BLOCK_M

    ms    = n0 + tl.arange(0, BLOCK_M)    # query positions in this tile
    mm    = ms < N
    ds    = tl.arange(0, HD)
    dm    = ds < HD
    sc    = 1.0 / (HD ** 0.5)

    # Load Q tile [BLOCK_M, HD]
    qb = Q + b * stride_qb + h * stride_qh
    q  = tl.load(qb + ms[:, None] * stride_qn + ds[None, :] * stride_qd,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    # Preload scale_embed for all J offsets into a 2D buffer isn't possible in
    # static_range, so we load se_j inside the loop (J=20, HD=64 → 1280 floats,
    # fits in registers).

    # Online softmax accumulators
    mi  = tl.full([BLOCK_M], float('-inf'), tl.float32)   # running max
    li  = tl.zeros([BLOCK_M], tl.float32)                 # running sum of exp
    acc = tl.zeros([BLOCK_M, HD], tl.float32)             # weighted value sum

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh

    for i in tl.static_range(20):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 23, 32, 64, 128, 256, 512, 1024)[i]

        # kp[m] = ms[m] - delta: positions of KVs for each query in tile
        # These are n0-delta, n0-delta+1, ..., n0-delta+BLOCK_M-1  → CONTIGUOUS
        # tl.load with contiguous addresses = coalesced HBM access
        kp  = ms - delta                         # [BLOCK_M] positions
        val = (kp >= 0) & mm                     # causal + bounds mask

        # Load K tile [BLOCK_M, HD] — coalesced
        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        # Load V tile [BLOCK_M, HD] — coalesced
        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        # Score: Q·K per query in tile  [BLOCK_M]
        # q: [BLOCK_M, HD], kt: [BLOCK_M, HD] → elementwise product, sum over HD
        s  = tl.sum(q * kt, axis=1) * sc                          # [BLOCK_M]

        # pos_bias[j, h]: scalar offset for this offset×head
        pb = tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)  # scalar
        s  = s + pb

        # scale_embed[j, :HD]: content filter projected onto Q
        se = tl.load(SCALE_EMBED + i * stride_sei + ds * stride_sed,
                     mask=dm, other=0.0).to(tl.float32)            # [HD]
        s  = s + tl.sum(q * se[None, :], axis=1) * sc             # [BLOCK_M]

        # Mask out invalid (causal or out-of-bounds) positions
        s  = tl.where(val, s, float('-inf'))

        # Online softmax update
        mn  = tl.maximum(mi, s)
        cor = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p   = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        acc = acc * cor[:, None] + p[:, None] * vt
        li  = li  * cor + p
        mi  = mn

    # Normalize
    ls  = tl.where(li > 0.0, li, 1.0)
    lse = mi + tl.log(ls)
    out = acc / ls[:, None]

    # Write output
    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ms[:, None] * stride_on + ds[None, :] * stride_od,
             out.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])
    tl.store(lb + ms * stride_ln, lse, mask=mm)


# ── Backward kernel — dK + dV ─────────────────────────────────────────────────
#
# Standard attention backward: for each (key position kn, offset j):
#   dV[kn] += Σ_q  p[q,j] * dO[q]    where q = kn + delta_j
#   dK[kn] += Σ_q  (p[q,j]*(dO[q]·O[q] - D[q])) * Q[q] * sc
#
# We iterate over KEY tiles; for each key position kn and offset j,
# the query that attends to it is q = kn + delta_j.
# Grid: (B*H, ceil(N/BLOCK_M))  — one program per (bh, key_tile)

@triton.jit
def _dsqg_bwd_dkdv(
    Q, K, V, POS_BIAS, SCALE_EMBED, DO, O, LSE, D,
    DK, DV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bh   = tl.program_id(0)
    tile = tl.program_id(1)
    b    = bh // H
    h    = bh %  H
    n0   = tile * BLOCK_M

    ks   = n0 + tl.arange(0, BLOCK_M)   # key positions
    km   = ks < N
    ds   = tl.arange(0, HD)
    dm   = ds < HD
    sc   = 1.0 / (HD ** 0.5)

    kb  = K  + b * stride_kb + h * stride_kh
    vb  = V  + b * stride_vb + h * stride_vh
    qb  = Q  + b * stride_qb + h * stride_qh
    dob = DO + b * stride_ob + h * stride_oh
    ob  = O  + b * stride_ob + h * stride_oh
    lb  = LSE + b * stride_lb + h * stride_lh
    db  = D  + b * stride_db + h * stride_dn

    kt  = tl.load(kb + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
                  mask=km[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vt  = tl.load(vb + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
                  mask=km[:, None] & dm[None, :], other=0.0).to(tl.float32)

    dk  = tl.zeros([BLOCK_M, HD], tl.float32)
    dv  = tl.zeros([BLOCK_M, HD], tl.float32)

    for i in tl.static_range(20):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 23, 32, 64, 128, 256, 512, 1024)[i]

        qs  = ks + delta      # query positions that attend to keys in this tile
        qm  = (qs < N) & km

        qt  = tl.load(qb + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
                      mask=qm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        do  = tl.load(dob + qs[:, None] * stride_on + ds[None, :] * stride_od,
                      mask=qm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        ot  = tl.load(ob  + qs[:, None] * stride_on + ds[None, :] * stride_od,
                      mask=qm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        lse = tl.load(lb + qs * stride_ln, mask=qm, other=0.0)
        Dv  = tl.load(db + qs * stride_dn, mask=qm, other=0.0)

        # Recompute score  (note: key position ks attended by query qs = ks+delta)
        s   = tl.sum(qt * kt, axis=1) * sc
        pb  = tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        se  = tl.load(SCALE_EMBED + i * stride_sei + ds * stride_sed,
                      mask=dm, other=0.0).to(tl.float32)
        s   = s + pb + tl.sum(qt * se[None, :], axis=1) * sc
        s   = tl.where(qm, s, float('-inf'))

        # p[q, j] = softmax weight = exp(s - lse)
        p   = tl.where(qm, tl.exp(s - lse), tl.zeros_like(s))  # [BLOCK_M]

        # dV accumulate: dV[kn] += p[q,j] * dO[q]
        dv  = dv + p[:, None] * do

        # dK accumulate: delta_q = dO·O per query
        delta_q = tl.sum(do * ot, axis=1)                       # [BLOCK_M]
        dkterm  = tl.where(qm, p * (delta_q - Dv), tl.zeros_like(p))
        dk      = dk + dkterm[:, None] * qt * sc

    dkb = DK + b * stride_kb + h * stride_kh
    dvb = DV + b * stride_vb + h * stride_vh
    tl.store(dkb + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
             dk.to(tl.bfloat16), mask=km[:, None] & dm[None, :])
    tl.store(dvb + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
             dv.to(tl.bfloat16), mask=km[:, None] & dm[None, :])


# ── D vector (for backward dK computation) ───────────────────────────────────
@triton.jit
def _compute_D(
    DO, O, D,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_db, stride_dh, stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bh   = tl.program_id(0)
    tile = tl.program_id(1)
    b    = bh // H;  h = bh % H
    n0   = tile * BLOCK_M
    ms   = n0 + tl.arange(0, BLOCK_M);  mm = ms < N
    ds   = tl.arange(0, HD);             dm = ds < HD
    do   = tl.load(DO + b*stride_ob + h*stride_oh + ms[:,None]*stride_on + ds[None,:]*stride_od,
                   mask=mm[:,None]&dm[None,:], other=0.0).to(tl.float32)
    o    = tl.load(O  + b*stride_ob + h*stride_oh + ms[:,None]*stride_on + ds[None,:]*stride_od,
                   mask=mm[:,None]&dm[None,:], other=0.0).to(tl.float32)
    tl.store(D + b*stride_db + h*stride_dh + ms*stride_dn,
             tl.sum(do * o, axis=1), mask=mm)


# ── Autograd Function ─────────────────────────────────────────────────────────
class _DSQGTiled(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed, BLOCK_M=16):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16
        assert HD in (32, 64, 128), f"HD={HD} not supported (must be 32/64/128)"

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        tiles = triton.cdiv(N, BLOCK_M)
        grid = (B * H, tiles)

        _dsqg_fwd[grid](
            q, k, v, pos_bias, scale_embed, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD, BLOCK_M=BLOCK_M,
            num_stages=3, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        ctx.BLOCK_M = BLOCK_M
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, pos_bias, scale_embed, o, lse = ctx.saved_tensors
        B, H, N, HD = q.shape
        BLOCK_M = ctx.BLOCK_M
        do = do.contiguous()
        sc = 1.0 / (HD ** 0.5)

        # ── D vector: D[b,h,n] = sum_d(dO * O) ─────────────────────────────
        D = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        tiles = triton.cdiv(N, BLOCK_M)
        grid  = (B * H, tiles)
        _compute_D[grid](
            do, o, D,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD, BLOCK_M=BLOCK_M,
        )

        # ── dK, dV via Triton kernel ─────────────────────────────────────────
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        _dsqg_bwd_dkdv[grid](
            q, k, v, pos_bias, scale_embed, do, o, lse, D, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD, BLOCK_M=BLOCK_M,
        )

        # ── Shared: recompute softmax weights p[B,H,N,J] ────────────────────
        # Used for both dQ and param grads.
        qf = q.float(); kf = k.float(); vf = v.float()
        dof = do.float(); Df = D.float()
        arange_N = torch.arange(N, device=q.device)

        scores_all = torch.full((B, H, N, J), float('-inf'), device=q.device, dtype=torch.float32)
        for idx, delta in enumerate(ALL_OFFSETS):
            kp    = arange_N - delta
            valid = kp >= 0
            kp_c  = kp.clamp(0)
            s = (qf * kf[:, :, kp_c]).sum(-1) * sc
            s = s + pos_bias[idx].float().view(1, H, 1)
            s = s + (qf * scale_embed[idx].float()).sum(-1) * sc
            s = torch.where(valid[None, None, :].expand(B, H, N), s,
                            torch.full_like(s, float('-inf')))
            scores_all[:, :, :, idx] = s

        # Guard against all-masked positions (n < min_delta) to prevent NaN
        all_inf = (scores_all == float('-inf')).all(-1, keepdim=True)
        scores_safe = scores_all.masked_fill(all_inf.expand_as(scores_all), 0.0)
        p = torch.softmax(scores_safe, dim=-1)                   # [B,H,N,J]
        p = p.masked_fill(all_inf.expand_as(p), 0.0)            # zero out masked positions

        # ── dQ (PyTorch, avoids another Triton kernel) ───────────────────────
        dq = torch.zeros_like(qf)
        for idx, delta in enumerate(ALL_OFFSETS):
            kp   = arange_N - delta
            kp_c = kp.clamp(0)
            pj   = p[:, :, :, idx]                               # [B,H,N]
            # dL/dQ contribution from score: p[j] * delta_term * K[kp] * sc
            delta_term = (dof * vf[:, :, kp_c]).sum(-1) - Df    # [B,H,N]
            dq += (pj * delta_term).unsqueeze(-1) * kf[:, :, kp_c] * sc
            # dL/dQ contribution from scale_embed term: p[j] * se[j] * sc
            dq += pj.unsqueeze(-1) * scale_embed[idx].float() * sc
        dq = dq.to(q.dtype)

        # ── param grads: pos_bias and scale_embed ───────────────────────────
        dpos_bias    = None
        dscale_embed = None
        if pos_bias.requires_grad or scale_embed.requires_grad:
            # dL/dw[b,h,n,j] = dO[b,h,n] · V[b,h,kp_j]
            dloss_dw = torch.zeros_like(p)
            for idx, delta in enumerate(ALL_OFFSETS):
                kp_c = (arange_N - delta).clamp(0)
                dloss_dw[:, :, :, idx] = (dof * vf[:, :, kp_c]).sum(-1)

            # Softmax backward: ds[b,h,n,j] = p[j] * (dw[j] - sum_j(p[j]*dw[j]))
            dw_sum = (dloss_dw * p).sum(-1, keepdim=True)
            ds_grad = p * (dloss_dw - dw_sum)                   # [B,H,N,J]

            if pos_bias.requires_grad:
                # dpb[j,h] = sum_{b,n} ds[b,h,n,j]
                dpos_bias = ds_grad.sum(0).sum(1).t()            # [J,H]
                dpos_bias = dpos_bias.to(pos_bias.dtype)

            if scale_embed.requires_grad:
                # dse[j,d] = sum_{b,h,n} ds[b,h,n,j] * Q[b,h,n,d] * sc
                dscale_embed = torch.einsum('bhnj,bhnd->jd', ds_grad, qf) * sc
                dscale_embed = dscale_embed.to(scale_embed.dtype)

        return dq, dk, dv, dpos_bias, dscale_embed, None


def dsqg_tiled_attention(q, k, v, pos_bias, scale_embed, block_m=16):
    """Main entry point. q,k,v: [B,H,N,HD] bfloat16."""
    return _DSQGTiled.apply(q, k, v, pos_bias, scale_embed, block_m)


# ── Module ────────────────────────────────────────────────────────────────────
class DSQGAttentionV10(nn.Module):
    """
    DSQG V10: J=20 Frobenius-optimal offsets, tiled coalesced kernel.

    Mechanisms:
      pos_bias     [J, H]   relay routing prior (learnable)
      scale_embed  [J, HD]  Q-matched content filter (zero-init)
      if_gain      [H]      IF amplifier gain (1.0-init, PyTorch-level)

    NPCI available via npci_rotate() if needed at the block level.

    Parameters: significantly fewer than V8 (no phase_base/gain, no query/key probes).
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        HD = embedding_dim // num_heads
        self.head_dim  = HD

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS], dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))
        self.dropout_p   = dropout

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()  # [B,H,N,HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        # NPCI on K/V if theta params are non-zero (optional; skip if theta=0)
        # Intentionally not included in base V10 — add if ablation confirms benefit

        # IF gain: per-head amplitude scaling of Q
        q = q * self.if_gain.view(1, H, 1, 1)

        # Core attention — tiled coalesced Triton kernel
        out = dsqg_tiled_attention(q, k, v, self.pos_bias, self.scale_embed)  # [B,H,N,HD]

        # Reshape and gate
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(
            self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
            p=self.dropout_p, training=self.training)


# ── Correctness test ──────────────────────────────────────────────────────────
def _reference_forward(q, k, v, pos_bias, scale_embed):
    """Pure PyTorch reference: O(N×J) but exact."""
    B, H, N, HD = q.shape
    sc = 1.0 / (HD ** 0.5)
    out = torch.zeros_like(q, dtype=torch.float32)
    for n in range(N):
        scores = []
        vals   = []
        for idx, delta in enumerate(ALL_OFFSETS):
            kp = n - delta
            if kp < 0:
                scores.append(torch.full((B, H), float('-inf'), device=q.device))
                vals.append(torch.zeros(B, H, HD, device=q.device))
                continue
            s = (q[:, :, n, :].float() * k[:, :, kp, :].float()).sum(-1) * sc  # [B,H]
            s = s + pos_bias[idx, :]  # [H] broadcast
            s = s + (q[:, :, n, :].float() * scale_embed[idx].float()).sum(-1) * sc
            scores.append(s)
            vals.append(v[:, :, kp, :].float())
        scores = torch.stack(scores, dim=-1)  # [B,H,J]
        # If all offsets are masked (n < min_delta), output is zero
        all_masked = (scores == float('-inf')).all(-1)  # [B,H]
        scores_safe = scores.clone()
        scores_safe[all_masked] = 0.0   # avoid softmax(all -inf) = NaN
        w      = torch.softmax(scores_safe, dim=-1)
        w      = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(w), w)
        vals   = torch.stack(vals, dim=-2)    # [B,H,J,HD]
        out[:, :, n, :] = (w.unsqueeze(-1) * vals).sum(-2)
    return out.to(q.dtype)


if __name__ == '__main__':
    import time
    torch.manual_seed(42)
    device = 'cuda'

    print('=' * 70)
    print('DSQG V10 — Correctness & Performance Tests')
    print('=' * 70)

    # ── Correctness ───────────────────────────────────────────────────────────
    for name, B, N, H, HD in [
        ('tiny',     1,  16, 4, 32),
        ('mid',      2,  64, 8, 64),
        ('training', 2, 128, 8, 64),
    ]:
        q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
        k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
        v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
        pb = torch.randn(J, H, device=device, dtype=torch.float32) * 0.5
        se = torch.randn(J, HD, device=device, dtype=torch.float32) * 0.1

        ref = _reference_forward(q, k, v, pb, se)
        out = dsqg_tiled_attention(q.clone(), k.clone(), v.clone(), pb, se)

        err = (ref.float() - out.float()).abs()
        status = 'PASS' if err.max() < 0.05 else 'FAIL'
        print(f'  {name:12s}  fwd_err={err.max():.4f}  {status}')
        assert status == 'PASS', f'Forward failed: {name}'

    # ── Backward smoke test ───────────────────────────────────────────────────
    B, H, N, HD = 2, 8, 64, 64
    q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16, requires_grad=True)
    k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16, requires_grad=True)
    v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16, requires_grad=True)
    pb = torch.randn(J, H,  device=device, dtype=torch.float32, requires_grad=True)
    se = torch.randn(J, HD, device=device, dtype=torch.float32, requires_grad=True)
    out = dsqg_tiled_attention(q, k, v, pb, se)
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert pb.grad is not None and se.grad is not None
    print(f'  backward         PASS  (dk,dv,dpb,dse all non-None)')

    # ── Module smoke test ─────────────────────────────────────────────────────
    B, N, D = 2, 128, 512
    m = DSQGAttentionV10(D, 8, seq_len=N).to(device).to(torch.bfloat16)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == (B, N, D)
    print(f'  module fwd       PASS  shape={y.shape}')

    print()
    print('ALL PASSED')
    print()

    # ── Performance vs V8 ─────────────────────────────────────────────────────
    print('=' * 70)
    print('Performance: V10 vs V8 (B=2, N=2048, D=512, H=8)')
    print('=' * 70)
    B, H, N, HD = 2, 8, 2048, 64
    q  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
    k  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
    v  = torch.randn(B, H, N, HD, device=device, dtype=torch.bfloat16)
    pb = torch.randn(J, H,  device=device, dtype=torch.float32)
    se = torch.randn(J, HD, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = dsqg_tiled_attention(q, k, v, pb, se)
    torch.cuda.synchronize()

    REPS = 50
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = dsqg_tiled_attention(q, k, v, pb, se)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    v10_ms = (t1 - t0) / REPS * 1000

    print(f'  V10 fwd:   {v10_ms:.2f}ms')
    print(f'  V8  fwd:   ~4.30ms (measured)')
    print(f'  Speedup:   {4.30/v10_ms:.2f}×')
    print()

    # Extrapolate to training step
    step_v8  = 1740.0   # measured on 4090
    overhead = step_v8 - (4.30 * 3 * 5 * 4)   # non-DSQG part
    step_v10 = v10_ms * 3 * 5 * 4 + overhead
    epoch_steps = 121232 // 8 // 4
    print(f'  Predicted step time (BS=8): {step_v10:.0f}ms (V8: {step_v8:.0f}ms)')
    print(f'  Predicted epoch time:       {step_v10 * epoch_steps / 3600:.1f}h '
          f'(V8: {step_v8 * epoch_steps / 3600:.1f}h)')
