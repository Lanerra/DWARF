"""
Hierarchical Sparse Attention V17-HISA-STRICT candidate.

Strict-causal semantics match V16:
  - exact causal attention within the active chunk
  - sparse retrieval only over strictly earlier chunks
  - per-token Stage-1 chunk routing
  - per-token Stage-2 token selection
  - no training-only routing-score bias in final attention

This version replaces the hot strict-attention body with a real Triton forward
kernel on CUDA while keeping the outer module-level custom autograd boundary.
Backward still recomputes through a known-correct eager reference path so the
trainer only has to save x + weights, not Q/K/V plus sparse metadata.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import environments
    triton = None
    tl = None

from hierarchical_sparse_attn_v16_hisa_strict import (
    _build_stage2_token_indices_per_token as _build_stage2_token_indices_per_token_eager,
    _compute_chunk_representatives,
    _compute_past_only_routing,
)


def _build_stage1_top_k_per_token_prev_mandatory(
    routing_weights: torch.Tensor,
    *,
    seq_len: int,
    chunk_size: int,
    num_chunks: int,
    top_k_chunks: int,
) -> torch.Tensor:
    """
    Strict-causal Stage-1 chunk selection with a mandatory previous-chunk slot.

    For tokens in query chunk c_q > 0:
      - slot 0 is forced to chunk c_q - 1
      - remaining slots are routed over strictly older chunks [0, ..., c_q - 2]

    This keeps the total sparse chunk budget fixed (no free extra slot), preserves
    strict causality, and guarantees a direct cross-boundary bridge without
    allowing the current or future chunks into the sparse path.
    """
    B, H, N, _ = routing_weights.shape
    device = routing_weights.device

    out = torch.full((B, H, N, top_k_chunks), -1, dtype=torch.long, device=device)
    if top_k_chunks <= 0:
        return out

    positions = torch.arange(seq_len, device=device)
    query_chunks = torch.div(positions, chunk_size, rounding_mode='floor')
    has_prev = query_chunks > 0
    prev_chunk = (query_chunks - 1).clamp(min=0)

    out[..., 0] = torch.where(
        has_prev.view(1, 1, N),
        prev_chunk.view(1, 1, N).expand(B, H, N),
        out[..., 0],
    )

    if top_k_chunks == 1:
        return out

    chunk_ids = torch.arange(num_chunks, device=device)
    remote_mask = chunk_ids.unsqueeze(0) < (query_chunks.unsqueeze(1) - 1)
    masked = routing_weights.masked_fill(~remote_mask[None, None], float('-inf'))

    k_eff = min(top_k_chunks - 1, num_chunks)
    if k_eff <= 0:
        return out

    values, indices = masked.topk(k_eff, dim=-1)
    indices = indices.masked_fill(torch.isneginf(values), -1)
    out[..., 1:1 + k_eff] = indices[..., :top_k_chunks - 1]
    return out


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _to_compute_dtype(t: torch.Tensor) -> torch.Tensor:
    return t


def _to_heads(t: torch.Tensor, B: int, N: int, H: int, hd: int) -> torch.Tensor:
    return t.reshape(B, N, H, hd).transpose(1, 2)


def _summarize_local_block(
    q_chunk: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    q_abs: torch.Tensor,
    k_abs: torch.Tensor,
    seq_len: int,
    hd: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(hd)
    qf = _to_compute_dtype(q_chunk)
    kf = _to_compute_dtype(k_local)
    vf = _to_compute_dtype(v_local)

    scores = torch.einsum('bhqd,bhkd->bhqk', qf, kf) * scale
    local_mask = (k_abs.view(1, 1, 1, -1) <= q_abs.view(1, 1, -1, 1)) & (k_abs.view(1, 1, 1, -1) < seq_len)
    scores = scores.masked_fill(~local_mask, float('-inf'))

    m = scores.amax(dim=-1)
    probs = torch.exp(scores - m.unsqueeze(-1))
    probs = torch.nan_to_num(probs, nan=0.0)
    l = probs.sum(dim=-1)
    acc = torch.einsum('bhqk,bhkd->bhqd', probs, vf)
    return m, l, acc


def _merge_sparse_blocks(
    m: torch.Tensor,
    l: torch.Tensor,
    acc: torch.Tensor,
    q_chunk: torch.Tensor,
    k_pad: torch.Tensor,
    v_pad: torch.Tensor,
    sparse_idx: torch.Tensor,
    seq_len: int,
    hd: int,
    sparse_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, _q_len, total_sparse = sparse_idx.shape
    device = sparse_idx.device
    scale = 1.0 / math.sqrt(hd)
    qf = _to_compute_dtype(q_chunk)
    kf = _to_compute_dtype(k_pad)
    vf = _to_compute_dtype(v_pad)
    acc = _to_compute_dtype(acc)
    m = _to_compute_dtype(m)
    l = _to_compute_dtype(l)

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1)

    for start in range(0, total_sparse, sparse_block_size):
        stop = min(start + sparse_block_size, total_sparse)
        idx_block = sparse_idx[..., start:stop]
        valid_block = (idx_block >= 0) & (idx_block < seq_len)
        if not valid_block.any():
            continue

        safe_idx = idx_block.clamp(min=0)
        k_block = kf[b_idx, h_idx, safe_idx]
        v_block = vf[b_idx, h_idx, safe_idx]

        scores = (qf.unsqueeze(-2) * k_block).sum(dim=-1) * scale
        scores = scores.masked_fill(~valid_block, float('-inf'))

        block_m = scores.amax(dim=-1)
        block_has = valid_block.any(dim=-1)
        safe_block_m = torch.where(block_has, block_m, torch.zeros_like(block_m))

        block_probs = torch.exp(scores - safe_block_m.unsqueeze(-1))
        block_probs = torch.where(valid_block, block_probs, torch.zeros_like(block_probs))
        block_l = block_probs.sum(dim=-1)
        block_acc = (block_probs.unsqueeze(-1) * v_block).sum(dim=-2)

        new_m = torch.where(block_has, torch.maximum(m, block_m), m)
        alpha = torch.exp(m - new_m)
        beta = torch.where(block_has, torch.exp(safe_block_m - new_m), torch.zeros_like(new_m))
        acc = acc * alpha.unsqueeze(-1) + block_acc * beta.unsqueeze(-1)
        l = l * alpha + block_l * beta
        m = new_m

    return m, l, acc


def _strict_hisa_core_eager_with_lse(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    sparse_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, hd = Q.shape
    device = Q.device
    pad_len = chunk_size * num_chunks - N
    K_pad = F.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
    V_pad = F.pad(V, (0, 0, 0, pad_len)) if pad_len > 0 else V
    out = torch.zeros_like(Q)
    lse = torch.full((B, H, N), float('-inf'), device=device, dtype=torch.float32)
    local_k_abs = torch.arange(chunk_size, device=device, dtype=torch.long)

    for c_q in range(num_chunks):
        q_start = c_q * chunk_size
        if q_start >= N:
            break
        q_end = min(q_start + chunk_size, N)

        q_chunk = Q[:, :, q_start:q_end, :]
        k_local = K_pad[:, :, q_start:q_start + chunk_size, :]
        v_local = V_pad[:, :, q_start:q_start + chunk_size, :]
        q_abs = torch.arange(q_start, q_end, device=device, dtype=torch.long)
        k_abs = local_k_abs + q_start

        m, l, acc = _summarize_local_block(q_chunk, k_local, v_local, q_abs, k_abs, N, hd)
        sparse_idx = stage2_token_indices[:, :, q_start:q_end, :]
        m, l, acc = _merge_sparse_blocks(
            m,
            l,
            acc,
            q_chunk,
            K_pad,
            V_pad,
            sparse_idx,
            N,
            hd,
            sparse_block_size,
        )
        safe_l = torch.where(l > 0.0, l, torch.ones_like(l))
        out[:, :, q_start:q_end, :] = (acc / safe_l.unsqueeze(-1)).to(dtype=Q.dtype)
        safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        lse[:, :, q_start:q_end] = torch.where(
            torch.isfinite(m),
            safe_m + torch.log(safe_l),
            torch.full_like(safe_m, float('-inf')),
        )

    return out, lse


def _strict_hisa_core_eager(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    sparse_block_size: int,
) -> torch.Tensor:
    out, _lse = _strict_hisa_core_eager_with_lse(
        Q,
        K,
        V,
        stage2_token_indices,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        sparse_block_size=sparse_block_size,
    )
    return out


if triton is not None:
    @triton.jit
    def _stage2_top_tokens_fwd(
        Q, K, TOPK_IDX, OUT,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_tb, stride_th, stride_tn, stride_ts,
        stride_ob, stride_oh, stride_on, stride_os,
        N, NUM_CHUNKS, H: tl.constexpr, HD: tl.constexpr,
        TOP_K: tl.constexpr,
        TOP_M: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_TOK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        q_idx = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_valid = q_idx < N
        safe_q = tl.where(q_valid, q_idx, 0)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        tok_offsets = tl.arange(0, BLOCK_TOK)
        tok_mask = tok_offsets < CHUNK_SIZE
        sc = 1.0 / tl.sqrt(HD * 1.0)

        q_base = Q + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        t_base = TOPK_IDX + b * stride_tb + h * stride_th
        o_base = OUT + b * stride_ob + h * stride_oh

        q = tl.load(
            q_base + safe_q * stride_qn + ds * stride_qd,
            mask=q_valid & dm,
            other=0.0,
        ).to(tl.float32)

        for slot in range(TOP_K):
            chunk_idx = tl.load(t_base + safe_q * stride_tn + slot * stride_ts, mask=q_valid, other=-1).to(tl.int32)
            valid_chunk = q_valid & (chunk_idx >= 0) & (chunk_idx < NUM_CHUNKS)
            safe_chunk_idx = tl.where(valid_chunk, chunk_idx, 0)
            abs_idx = safe_chunk_idx * CHUNK_SIZE + tok_offsets

            k_block = tl.load(
                k_base + abs_idx[:, None] * stride_kn + ds[None, :] * stride_kd,
                mask=(valid_chunk & tok_mask)[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k_block * q[None, :], axis=1) * sc
            scores = tl.where(valid_chunk & tok_mask, scores, float('-inf'))

            selected = tl.full([BLOCK_TOK], False, tl.int1)
            for mi in range(TOP_M):
                cand = tl.where((~selected) & tok_mask & valid_chunk, scores, float('-inf'))
                best_val, best_idx = tl.max(cand, axis=0, return_indices=True)
                has_any = valid_chunk & (best_val > -3.0e38)
                out_idx = slot * TOP_M + mi
                abs_pos = safe_chunk_idx * CHUNK_SIZE + best_idx
                out_val = tl.where(has_any, abs_pos, -1).to(tl.int64)
                tl.store(o_base + safe_q * stride_on + out_idx * stride_os, out_val, mask=q_valid)
                selected = selected | ((tok_offsets == best_idx) & has_any)

    @triton.jit
    def _strict_hisa_fwd(
        Q, K, V, SPARSE_IDX, OUT, LSE_OUT,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_sb, stride_sh, stride_sn, stride_ss,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_lseb, stride_lseh, stride_lsen,
        N, H: tl.constexpr, HD: tl.constexpr,
        TOTAL_SPARSE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        bh = tl.program_id(0)
        c_q = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_start = c_q * CHUNK_SIZE
        q_offsets = tl.arange(0, BLOCK_Q)
        qs = q_start + q_offsets
        qm = (q_offsets < CHUNK_SIZE) & (qs < N)
        safe_qs = tl.where(qm, qs, 0)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        sc = 1.0 / tl.sqrt(HD * 1.0)

        q_base = Q + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        s_base = SPARSE_IDX + b * stride_sb + h * stride_sh
        o_base = OUT + b * stride_ob + h * stride_oh

        q_c = tl.load(
            q_base + safe_qs[:, None] * stride_qn + ds[None, :] * stride_qd,
            mask=qm[:, None] & dm[None, :],
            other=0.0,
        )
        q_f = q_c.to(tl.float32)

        mi = tl.full([BLOCK_Q], float('-inf'), tl.float32)
        li = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, BLOCK_HD], tl.float32)

        # Exact local causal attention within the active chunk, including self.
        k_offsets = tl.arange(0, BLOCK_Q)
        ks_local = q_start + k_offsets
        km_local = (k_offsets < CHUNK_SIZE) & (ks_local < N)
        safe_ks_local = tl.where(km_local, ks_local, 0)
        k_local = tl.load(
            k_base + safe_ks_local[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km_local[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        v_local = tl.load(
            v_base + safe_ks_local[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km_local[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)

        s_local = tl.dot(q_f, tl.trans(k_local), input_precision="ieee") * sc
        local_causal = (ks_local[None, :] <= qs[:, None]) & qm[:, None] & km_local[None, :]
        s_local = tl.where(local_causal, s_local, float('-inf'))

        m_new = tl.max(s_local, axis=1)
        has_prev = mi > float('-inf')
        has_curr = m_new > float('-inf')
        has_any = has_prev | has_curr
        mn_raw = tl.maximum(mi, m_new)
        mn = tl.where(has_any, mn_raw, tl.zeros_like(mn_raw))
        cor = tl.where(has_prev, tl.exp(mi - mn), tl.zeros_like(mi))
        p_local = tl.exp(s_local - mn[:, None])
        p_local = tl.where(local_causal, p_local, 0.0)
        li = tl.where(has_any, li * cor + tl.sum(p_local, axis=1), li)
        mi = tl.where(has_any, mn_raw, mi)

        if BLOCK_Q >= 16 and BLOCK_HD >= 16:
            acc = acc * cor[:, None] + tl.dot(p_local.to(tl.float32), v_local, input_precision="ieee")
        else:
            acc = acc * cor[:, None] + tl.sum(p_local[:, :, None] * v_local[None, :, :], axis=1)

        # Sparse retrieval over earlier chunks using per-token Stage-2 indices.
        for si in range(TOTAL_SPARSE):
            idx = tl.load(s_base + safe_qs * stride_sn + si * stride_ss, mask=qm, other=-1).to(tl.int32)
            valid = (idx >= 0) & (idx < N) & qm
            safe_idx = tl.where(valid, idx, 0)

            k_sel = tl.load(
                k_base + safe_idx[:, None] * stride_kn + ds[None, :] * stride_kd,
                mask=valid[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            v_sel = tl.load(
                v_base + safe_idx[:, None] * stride_vn + ds[None, :] * stride_vd,
                mask=valid[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)

            s_vec = tl.sum(q_f * k_sel, axis=1) * sc
            s_vec = tl.where(valid, s_vec, float('-inf'))

            m_new = s_vec
            has_prev = mi > float('-inf')
            has_curr = valid
            has_any = has_prev | has_curr
            mn_raw = tl.maximum(mi, m_new)
            mn = tl.where(has_any, mn_raw, tl.zeros_like(mn_raw))
            cor = tl.where(has_prev, tl.exp(mi - mn), tl.zeros_like(mi))
            p_vec = tl.where(valid, tl.exp(s_vec - mn), 0.0)

            li = tl.where(has_any, li * cor + p_vec, li)
            mi = tl.where(has_any, mn_raw, mi)
            acc = acc * cor[:, None] + p_vec[:, None] * v_sel

        ls = tl.where(li > 0.0, li, 1.0)
        out = acc / ls[:, None]
        safe_mi = tl.where(mi > float('-inf'), mi, tl.zeros_like(mi))
        lse = tl.where(mi > float('-inf'), safe_mi + tl.log(ls), float('-inf'))
        tl.store(
            o_base + safe_qs[:, None] * stride_on + ds[None, :] * stride_od,
            out,
            mask=qm[:, None] & dm[None, :],
        )
        tl.store(
            LSE_OUT + b * stride_lseb + h * stride_lseh + safe_qs * stride_lsen,
            lse,
            mask=qm,
        )


    @triton.jit
    def _strict_hisa_bwd(
        Q, K, V, O, DO, LSE, SPARSE_IDX, DQ, DK, DV,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_lseb, stride_lseh, stride_lsen,
        stride_sb, stride_sh, stride_sn, stride_ss,
        stride_dqb, stride_dqh, stride_dqn, stride_dqd,
        stride_dkb, stride_dkh, stride_dkn, stride_dkd,
        stride_dvb, stride_dvh, stride_dvn, stride_dvd,
        N, H: tl.constexpr, HD: tl.constexpr,
        TOTAL_SPARSE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        bh = tl.program_id(0)
        c_q = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_start = c_q * CHUNK_SIZE
        q_offsets = tl.arange(0, BLOCK_Q)
        qs = q_start + q_offsets
        qm = (q_offsets < CHUNK_SIZE) & (qs < N)
        safe_qs = tl.where(qm, qs, 0)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        sc = 1.0 / tl.sqrt(HD * 1.0)

        q_base = Q + b * stride_qb + h * stride_qh
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        o_base = O + b * stride_ob + h * stride_oh
        do_base = DO + b * stride_dob + h * stride_doh
        s_base = SPARSE_IDX + b * stride_sb + h * stride_sh
        dq_base = DQ + b * stride_dqb + h * stride_dqh

        q = tl.load(
            q_base + safe_qs[:, None] * stride_qn + ds[None, :] * stride_qd,
            mask=qm[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        o = tl.load(
            o_base + safe_qs[:, None] * stride_on + ds[None, :] * stride_od,
            mask=qm[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_base + safe_qs[:, None] * stride_don + ds[None, :] * stride_dod,
            mask=qm[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + b * stride_lseb + h * stride_lseh + safe_qs * stride_lsen,
            mask=qm,
            other=0.0,
        )
        d_val = tl.sum(do * o, axis=1)
        lse_finite = lse > float('-inf')

        dq = tl.zeros([BLOCK_Q, BLOCK_HD], tl.float32)

        k_offsets = tl.arange(0, BLOCK_Q)
        ks_local = q_start + k_offsets
        km_local = (k_offsets < CHUNK_SIZE) & (ks_local < N)
        safe_ks_local = tl.where(km_local, ks_local, 0)
        k_local = tl.load(
            k_base + safe_ks_local[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km_local[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        v_local = tl.load(
            v_base + safe_ks_local[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km_local[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)

        s_local = tl.dot(q, tl.trans(k_local), input_precision="ieee") * sc
        local_causal = (ks_local[None, :] <= qs[:, None]) & qm[:, None] & km_local[None, :]
        local_alpha = tl.where(
            local_causal & lse_finite[:, None],
            tl.exp(tl.minimum(s_local - lse[:, None], 0.0)),
            0.0,
        )

        if BLOCK_Q >= 16 and BLOCK_HD >= 16:
            dot_rv_local = tl.dot(do, tl.trans(v_local), input_precision="ieee")
            ds_local = local_alpha * (dot_rv_local - d_val[:, None])
            dq += tl.dot(ds_local, k_local, input_precision="ieee") * sc
            dk_local = tl.dot(tl.trans(ds_local), q, input_precision="ieee") * sc
            dv_local = tl.dot(tl.trans(local_alpha), do, input_precision="ieee")
        else:
            dot_rv_local = tl.sum(do[:, None, :] * v_local[None, :, :], axis=2)
            ds_local = local_alpha * (dot_rv_local - d_val[:, None])
            dq += tl.sum(ds_local[:, :, None] * k_local[None, :, :], axis=1) * sc
            dk_local = tl.sum(ds_local[:, :, None] * q[:, None, :], axis=0) * sc
            dv_local = tl.sum(local_alpha[:, :, None] * do[:, None, :], axis=0)

        tl.atomic_add(
            DK + b * stride_dkb + h * stride_dkh + safe_ks_local[:, None] * stride_dkn + ds[None, :] * stride_dkd,
            tl.where(km_local[:, None] & dm[None, :], dk_local, 0.0),
            mask=km_local[:, None] & dm[None, :],
        )
        tl.atomic_add(
            DV + b * stride_dvb + h * stride_dvh + safe_ks_local[:, None] * stride_dvn + ds[None, :] * stride_dvd,
            tl.where(km_local[:, None] & dm[None, :], dv_local, 0.0),
            mask=km_local[:, None] & dm[None, :],
        )

        for si in range(TOTAL_SPARSE):
            idx = tl.load(s_base + safe_qs * stride_sn + si * stride_ss, mask=qm, other=-1).to(tl.int32)
            valid = (idx >= 0) & (idx < N) & qm
            safe_idx = tl.where(valid, idx, 0)

            k_sel = tl.load(
                k_base + safe_idx[:, None] * stride_kn + ds[None, :] * stride_kd,
                mask=valid[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            v_sel = tl.load(
                v_base + safe_idx[:, None] * stride_vn + ds[None, :] * stride_vd,
                mask=valid[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)

            s_vec = tl.sum(q * k_sel, axis=1) * sc
            alpha_vec = tl.where(
                valid & lse_finite,
                tl.exp(tl.minimum(s_vec - lse, 0.0)),
                0.0,
            )
            dot_rv_vec = tl.sum(do * v_sel, axis=1)
            ds_vec = alpha_vec * (dot_rv_vec - d_val)

            dq += ds_vec[:, None] * k_sel * sc

            tl.atomic_add(
                DK + b * stride_dkb + h * stride_dkh + safe_idx[:, None] * stride_dkn + ds[None, :] * stride_dkd,
                tl.where(valid[:, None] & dm[None, :], ds_vec[:, None] * q * sc, 0.0),
                mask=valid[:, None] & dm[None, :],
            )
            tl.atomic_add(
                DV + b * stride_dvb + h * stride_dvh + safe_idx[:, None] * stride_dvn + ds[None, :] * stride_dvd,
                tl.where(valid[:, None] & dm[None, :], alpha_vec[:, None] * do, 0.0),
                mask=valid[:, None] & dm[None, :],
            )

        tl.store(
            dq_base + safe_qs[:, None] * stride_dqn + ds[None, :] * stride_dqd,
            dq,
            mask=qm[:, None] & dm[None, :],
        )


def _strict_hisa_core_triton_with_lse(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or Q.device.type != 'cuda':
        raise RuntimeError('Triton strict HISA core requested without CUDA Triton support')

    B, H, N, hd = Q.shape
    out = torch.empty_like(Q)
    lse = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
    block_q = max(16, _next_pow2(chunk_size))
    block_hd = max(16, _next_pow2(hd))
    total_sparse = int(stage2_token_indices.shape[-1])
    num_warps = 4 if max(block_q, block_hd) <= 64 else 8

    grid = (B * H, num_chunks)
    _strict_hisa_fwd[grid](
        Q, K, V, stage2_token_indices, out, lse,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        stage2_token_indices.stride(0), stage2_token_indices.stride(1), stage2_token_indices.stride(2), stage2_token_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        N=Q.shape[2],
        H=H,
        HD=hd,
        TOTAL_SPARSE=total_sparse,
        CHUNK_SIZE=chunk_size,
        BLOCK_Q=block_q,
        BLOCK_HD=block_hd,
        num_warps=num_warps,
        num_stages=2,
    )
    return out, lse


def _strict_hisa_core_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
) -> torch.Tensor:
    out, _lse = _strict_hisa_core_triton_with_lse(
        Q,
        K,
        V,
        stage2_token_indices,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
    )
    return out


def _strict_hisa_core_with_lse(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    sparse_block_size: int,
    use_fused: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fused and Q.device.type == 'cuda' and triton is not None:
        return _strict_hisa_core_triton_with_lse(
            Q,
            K,
            V,
            stage2_token_indices,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
    return _strict_hisa_core_eager_with_lse(
        Q,
        K,
        V,
        stage2_token_indices,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        sparse_block_size=sparse_block_size,
    )


def _strict_hisa_core(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    sparse_block_size: int,
    use_fused: bool,
) -> torch.Tensor:
    out, _lse = _strict_hisa_core_with_lse(
        Q,
        K,
        V,
        stage2_token_indices,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        sparse_block_size=sparse_block_size,
        use_fused=use_fused,
    )
    return out


def _strict_hisa_core_backward_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or Q.device.type != 'cuda':
        raise RuntimeError('Triton strict HISA backward requested without CUDA Triton support')

    B, H, _N, hd = Q.shape
    grad_out = grad_out.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()
    dQ = torch.empty_like(Q, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)
    block_q = max(16, _next_pow2(chunk_size))
    block_hd = max(16, _next_pow2(hd))
    total_sparse = int(stage2_token_indices.shape[-1])
    num_warps = 4 if max(block_q, block_hd) <= 64 else 8

    grid = (B * H, num_chunks)
    _strict_hisa_bwd[grid](
        Q, K, V, out, grad_out, lse, stage2_token_indices, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        stage2_token_indices.stride(0), stage2_token_indices.stride(1), stage2_token_indices.stride(2), stage2_token_indices.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        N=Q.shape[2],
        H=H,
        HD=hd,
        TOTAL_SPARSE=total_sparse,
        CHUNK_SIZE=chunk_size,
        BLOCK_Q=block_q,
        BLOCK_HD=block_hd,
        num_warps=num_warps,
        num_stages=2,
    )
    return dQ.to(dtype=Q.dtype), dK.to(dtype=K.dtype), dV.to(dtype=V.dtype)


def _strict_hisa_core_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    stage2_token_indices: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    sparse_block_size: int,
    backward_impl: str,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backward_impl == 'eager':
        with torch.enable_grad():
            Q_ = Q.detach().requires_grad_(True)
            K_ = K.detach().requires_grad_(True)
            V_ = V.detach().requires_grad_(True)
            core_out, _core_lse = _strict_hisa_core_with_lse(
                Q_,
                K_,
                V_,
                stage2_token_indices,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                sparse_block_size=sparse_block_size,
                use_fused=False,
            )
        dQ, dK, dV = torch.autograd.grad(
            core_out,
            (Q_, K_, V_),
            grad_out,
            retain_graph=False,
            allow_unused=False,
        )
        return dQ, dK, dV
    if backward_impl == 'triton':
        if out is None or lse is None:
            raise ValueError('strict HISA Triton backward requires fused forward outputs: out and lse')
        return _strict_hisa_core_backward_triton(
            Q,
            K,
            V,
            stage2_token_indices,
            grad_out,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            out=out,
            lse=lse,
        )
    raise ValueError(f'Unsupported strict HISA backward_impl={backward_impl!r}')


def _build_stage2_token_indices_per_token_triton(
    Q: torch.Tensor,
    K_pad: torch.Tensor,
    top_k_indices: torch.Tensor,
    *,
    seq_len: int,
    num_chunks: int,
    chunk_size: int,
    hisa_top_m_tokens: int,
) -> torch.Tensor:
    if triton is None or Q.device.type != 'cuda':
        raise RuntimeError('Triton Stage-2 builder requested without CUDA Triton support')

    B, H, N, hd = Q.shape
    top_k = int(top_k_indices.shape[-1])
    m_actual = min(hisa_top_m_tokens, chunk_size)
    total_selected = top_k * m_actual
    if total_selected == 0:
        return torch.empty((B, H, N, 0), device=Q.device, dtype=torch.long)

    out = torch.empty((B, H, N, total_selected), device=Q.device, dtype=torch.long)
    top_k_indices_i32 = top_k_indices.to(dtype=torch.int32) if top_k_indices.dtype != torch.int32 else top_k_indices
    block_hd = max(16, _next_pow2(hd))
    block_tok = max(16, _next_pow2(chunk_size))
    num_warps = 4 if max(block_hd, block_tok) <= 64 else 8

    grid = (B * H, seq_len)
    _stage2_top_tokens_fwd[grid](
        Q,
        K_pad,
        top_k_indices_i32,
        out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_pad.stride(0), K_pad.stride(1), K_pad.stride(2), K_pad.stride(3),
        top_k_indices_i32.stride(0), top_k_indices_i32.stride(1), top_k_indices_i32.stride(2), top_k_indices_i32.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N=seq_len,
        NUM_CHUNKS=num_chunks,
        H=H,
        HD=hd,
        TOP_K=top_k,
        TOP_M=m_actual,
        CHUNK_SIZE=chunk_size,
        BLOCK_HD=block_hd,
        BLOCK_TOK=block_tok,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def _strict_hisa_module_project_qkv(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    *,
    num_heads: int,
    hd: int,
    num_chunks_base: int,
    top_k_chunks: int,
    hisa_top_m_tokens: int,
    chunk_size: int,
    kv_inject=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    B, N, _ = x.shape
    C = max(num_chunks_base, math.ceil(max(N, 1) / chunk_size))

    Q = _to_heads(F.linear(x, W_q), B, N, num_heads, hd)
    K = _to_heads(F.linear(x, W_k), B, N, num_heads, hd)
    V = _to_heads(F.linear(x, W_v), B, N, num_heads, hd)

    if kv_inject is not None:
        k_delta, v_delta = kv_inject
        K = K + k_delta
        V = V + v_delta

    return Q, K, V, C


def _strict_hisa_build_stage2_token_indices(
    Q: torch.Tensor,
    K: torch.Tensor,
    *,
    num_chunks: int,
    top_k_chunks: int,
    hisa_top_m_tokens: int,
    chunk_size: int,
) -> torch.Tensor:
    N = Q.shape[2]
    hd = Q.shape[-1]

    with torch.no_grad():
        pad_len = chunk_size * num_chunks - N
        K_pad = F.pad(K.detach(), (0, 0, 0, pad_len)) if pad_len > 0 else K.detach()
        chunk_reps = _compute_chunk_representatives(K_pad, num_chunks)
        routing_weights = _compute_past_only_routing(
            Q.detach(),
            chunk_reps,
            seq_len=N,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            hd=hd,
        )
        top_k_indices = _build_stage1_top_k_per_token_prev_mandatory(
            routing_weights,
            seq_len=N,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            top_k_chunks=top_k_chunks,
        )
        if Q.device.type == 'cuda' and triton is not None:
            return _build_stage2_token_indices_per_token_triton(
                Q.detach(),
                K_pad,
                top_k_indices,
                seq_len=N,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                hisa_top_m_tokens=hisa_top_m_tokens,
            )
        return _build_stage2_token_indices_per_token_eager(
            Q.detach(),
            K_pad,
            top_k_indices,
            seq_len=N,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            hisa_top_m_tokens=hisa_top_m_tokens,
        )


def _pack_stage2_token_indices(stage2_token_indices: torch.Tensor) -> torch.Tensor:
    if stage2_token_indices.numel() == 0:
        return stage2_token_indices.to(dtype=torch.int16)
    min_idx = int(stage2_token_indices.min().item())
    max_idx = int(stage2_token_indices.max().item())
    if min_idx >= torch.iinfo(torch.int16).min and max_idx <= torch.iinfo(torch.int16).max:
        return stage2_token_indices.to(dtype=torch.int16)
    return stage2_token_indices.to(dtype=torch.int32)


def _unpack_stage2_token_indices(packed_stage2_token_indices: torch.Tensor) -> torch.Tensor:
    return packed_stage2_token_indices.to(dtype=torch.long)


def _strict_hisa_module_recompute_state(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    *,
    num_heads: int,
    hd: int,
    num_chunks_base: int,
    top_k_chunks: int,
    hisa_top_m_tokens: int,
    chunk_size: int,
    kv_inject=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    Q, K, V, C = _strict_hisa_module_project_qkv(
        x,
        W_q,
        W_k,
        W_v,
        num_heads=num_heads,
        hd=hd,
        num_chunks_base=num_chunks_base,
        top_k_chunks=top_k_chunks,
        hisa_top_m_tokens=hisa_top_m_tokens,
        chunk_size=chunk_size,
        kv_inject=kv_inject,
    )
    stage2_token_indices = _strict_hisa_build_stage2_token_indices(
        Q,
        K,
        num_chunks=C,
        top_k_chunks=top_k_chunks,
        hisa_top_m_tokens=hisa_top_m_tokens,
        chunk_size=chunk_size,
    )
    return Q, K, V, stage2_token_indices, C


def _strict_hisa_module_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    *,
    num_heads: int,
    hd: int,
    num_chunks_base: int,
    top_k_chunks: int,
    hisa_top_m_tokens: int,
    chunk_size: int,
    sparse_block_size: int,
    kv_inject=None,
    use_fused: bool,
) -> torch.Tensor:
    B, N, _ = x.shape
    Q, K, V, stage2_token_indices, C = _strict_hisa_module_recompute_state(
        x,
        W_q,
        W_k,
        W_v,
        num_heads=num_heads,
        hd=hd,
        num_chunks_base=num_chunks_base,
        top_k_chunks=top_k_chunks,
        hisa_top_m_tokens=hisa_top_m_tokens,
        chunk_size=chunk_size,
        kv_inject=kv_inject,
    )
    out = _strict_hisa_core(
        Q,
        K,
        V,
        stage2_token_indices,
        num_chunks=C,
        chunk_size=chunk_size,
        sparse_block_size=sparse_block_size,
        use_fused=use_fused,
    )
    out_flat = out.transpose(1, 2).reshape(B, N, num_heads * hd)
    return F.linear(out_flat, W_o)


class _StrictHISAModuleFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        W_q,
        W_k,
        W_v,
        W_o,
        num_heads,
        hd,
        num_chunks_base,
        top_k_chunks,
        hisa_top_m_tokens,
        chunk_size,
        sparse_block_size,
        backward_impl,
    ):
        ctx.num_heads = int(num_heads)
        ctx.hd = int(hd)
        ctx.num_chunks_base = int(num_chunks_base)
        ctx.top_k_chunks = int(top_k_chunks)
        ctx.hisa_top_m_tokens = int(hisa_top_m_tokens)
        ctx.chunk_size = int(chunk_size)
        ctx.sparse_block_size = int(sparse_block_size)
        ctx.backward_impl = str(backward_impl)
        ctx.device_type = x.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled()
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)

        Q, K, V, C = _strict_hisa_module_project_qkv(
            x,
            W_q,
            W_k,
            W_v,
            num_heads=ctx.num_heads,
            hd=ctx.hd,
            num_chunks_base=ctx.num_chunks_base,
            top_k_chunks=ctx.top_k_chunks,
            hisa_top_m_tokens=ctx.hisa_top_m_tokens,
            chunk_size=ctx.chunk_size,
        )
        stage2_token_indices = _strict_hisa_build_stage2_token_indices(
            Q,
            K,
            num_chunks=C,
            top_k_chunks=ctx.top_k_chunks,
            hisa_top_m_tokens=ctx.hisa_top_m_tokens,
            chunk_size=ctx.chunk_size,
        )
        packed_stage2_token_indices = torch.empty(0, device=x.device, dtype=torch.int16)
        if ctx.backward_impl in {'eager', 'triton'}:
            packed_stage2_token_indices = _pack_stage2_token_indices(stage2_token_indices)
        ctx.save_for_backward(x, W_q, W_k, W_v, W_o, packed_stage2_token_indices)

        out = _strict_hisa_core(
            Q,
            K,
            V,
            stage2_token_indices,
            num_chunks=C,
            chunk_size=ctx.chunk_size,
            sparse_block_size=ctx.sparse_block_size,
            use_fused=(x.device.type == 'cuda' and triton is not None),
        )
        out_flat = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], ctx.num_heads * ctx.hd)
        return F.linear(out_flat, W_o)

    @staticmethod
    def backward(ctx, grad_out):
        x, W_q, W_k, W_v, W_o, packed_stage2_token_indices = ctx.saved_tensors
        cached_stage2_token_indices = None
        if packed_stage2_token_indices.numel() > 0:
            cached_stage2_token_indices = _unpack_stage2_token_indices(packed_stage2_token_indices)
        if ctx.backward_impl == 'eager':
            with torch.enable_grad():
                x_ = x.detach().requires_grad_(True)
                W_q_ = W_q.detach().requires_grad_(True)
                W_k_ = W_k.detach().requires_grad_(True)
                W_v_ = W_v.detach().requires_grad_(True)
                W_o_ = W_o.detach().requires_grad_(True)
                with torch.amp.autocast(
                    device_type=ctx.device_type,
                    dtype=ctx.autocast_dtype,
                    enabled=ctx.autocast_enabled,
                ):
                    Q, K, V, C = _strict_hisa_module_project_qkv(
                        x_,
                        W_q_,
                        W_k_,
                        W_v_,
                        num_heads=ctx.num_heads,
                        hd=ctx.hd,
                        num_chunks_base=ctx.num_chunks_base,
                        top_k_chunks=ctx.top_k_chunks,
                        hisa_top_m_tokens=ctx.hisa_top_m_tokens,
                        chunk_size=ctx.chunk_size,
                    )
                    stage2_token_indices = cached_stage2_token_indices
                    if stage2_token_indices is None:
                        stage2_token_indices = _strict_hisa_build_stage2_token_indices(
                            Q,
                            K,
                            num_chunks=C,
                            top_k_chunks=ctx.top_k_chunks,
                            hisa_top_m_tokens=ctx.hisa_top_m_tokens,
                            chunk_size=ctx.chunk_size,
                        )
                    out_core = _strict_hisa_core(
                        Q,
                        K,
                        V,
                        stage2_token_indices,
                        num_chunks=C,
                        chunk_size=ctx.chunk_size,
                        sparse_block_size=ctx.sparse_block_size,
                        use_fused=False,
                    )
                    out_flat = out_core.transpose(1, 2).reshape(x.shape[0], x.shape[1], ctx.num_heads * ctx.hd)
                    out = F.linear(out_flat, W_o_)
            dx, dW_q, dW_k, dW_v, dW_o = torch.autograd.grad(
                out,
                (x_, W_q_, W_k_, W_v_, W_o_),
                grad_out,
                retain_graph=False,
                allow_unused=False,
            )
            return dx, dW_q, dW_k, dW_v, dW_o, None, None, None, None, None, None, None, None

        if ctx.backward_impl != 'triton':
            raise ValueError(f"Unsupported strict HISA backward_impl={ctx.backward_impl!r}")

        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            W_q_ = W_q.detach().requires_grad_(True)
            W_k_ = W_k.detach().requires_grad_(True)
            W_v_ = W_v.detach().requires_grad_(True)
            W_o_ = W_o.detach().requires_grad_(True)
            with torch.amp.autocast(
                device_type=ctx.device_type,
                dtype=ctx.autocast_dtype,
                enabled=ctx.autocast_enabled,
            ):
                Q, K, V, C = _strict_hisa_module_project_qkv(
                    x_,
                    W_q_,
                    W_k_,
                    W_v_,
                    num_heads=ctx.num_heads,
                    hd=ctx.hd,
                    num_chunks_base=ctx.num_chunks_base,
                    top_k_chunks=ctx.top_k_chunks,
                    hisa_top_m_tokens=ctx.hisa_top_m_tokens,
                    chunk_size=ctx.chunk_size,
                )
                stage2_token_indices = cached_stage2_token_indices
                if stage2_token_indices is None:
                    stage2_token_indices = _strict_hisa_build_stage2_token_indices(
                        Q,
                        K,
                        num_chunks=C,
                        top_k_chunks=ctx.top_k_chunks,
                        hisa_top_m_tokens=ctx.hisa_top_m_tokens,
                        chunk_size=ctx.chunk_size,
                    )
                out_core, lse = _strict_hisa_core_with_lse(
                    Q,
                    K,
                    V,
                    stage2_token_indices,
                    num_chunks=C,
                    chunk_size=ctx.chunk_size,
                    sparse_block_size=ctx.sparse_block_size,
                    use_fused=True,
                )
                out_flat = out_core.transpose(1, 2).reshape(x.shape[0], x.shape[1], ctx.num_heads * ctx.hd).detach().requires_grad_(True)
                out = F.linear(out_flat, W_o_)

        d_out_flat, dW_o = torch.autograd.grad(
            out,
            (out_flat, W_o_),
            grad_out,
            retain_graph=False,
            allow_unused=False,
        )
        d_out = d_out_flat.reshape(x.shape[0], x.shape[1], ctx.num_heads, ctx.hd).transpose(1, 2).contiguous()
        dQ, dK, dV = _strict_hisa_core_backward(
            Q,
            K,
            V,
            stage2_token_indices,
            d_out,
            num_chunks=C,
            chunk_size=ctx.chunk_size,
            sparse_block_size=ctx.sparse_block_size,
            backward_impl='triton',
            out=out_core,
            lse=lse,
        )
        dx, dW_q, dW_k, dW_v = torch.autograd.grad(
            (Q, K, V),
            (x_, W_q_, W_k_, W_v_),
            (dQ, dK, dV),
            retain_graph=False,
            allow_unused=False,
        )
        return dx, dW_q, dW_k, dW_v, dW_o, None, None, None, None, None, None, None, None


class HierarchicalSparseAttentionV17HISAStrictTriton(nn.Module):
    """
    Optimized strict HISA candidate.

    CUDA forward uses a fused Triton strict-attention body. The default CUDA
    path now also uses Triton backward inside the module-level custom autograd
    boundary, while keeping eager backward available as an explicit fallback.
    CPU and other eager-only paths still bypass the custom Triton autograd path.
    """

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int = 32,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int = 32,
        chunk_size: int = 64,
        sparse_block_size: int = 64,
        backward_impl: str = 'triton',
    ):
        super().__init__()
        if backward_impl not in {'eager', 'triton'}:
            raise ValueError(
                f"Unsupported strict HISA backward_impl={backward_impl!r}; expected 'eager' or 'triton'"
            )
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.num_chunks = num_chunks
        self.top_k_chunks = top_k_chunks
        self.hisa_top_m_tokens = hisa_top_m_tokens
        self.chunk_size = chunk_size
        self.sparse_block_size = sparse_block_size
        self.backward_impl = backward_impl
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)
        self._routing_entropy: torch.Tensor | float = float('nan')

    def extra_repr(self) -> str:
        return (
            f'H={self.H}, hd={self.hd}, num_chunks={self.num_chunks}, '
            f'top_k_chunks={self.top_k_chunks}, hisa_top_m_tokens={self.hisa_top_m_tokens}, '
            f'chunk_size={self.chunk_size}, sparse_block_size={self.sparse_block_size}, '
            f'backward_impl={self.backward_impl}'
        )

    def forward(self, x: torch.Tensor, kv_inject=None) -> torch.Tensor:
        B, N, _ = x.shape
        H, hd = self.H, self.hd
        C = max(self.num_chunks, math.ceil(max(N, 1) / self.chunk_size))
        chunk_size = self.chunk_size

        if kv_inject is not None:
            return _strict_hisa_module_forward(
                x,
                self.W_q.weight,
                self.W_k.weight,
                self.W_v.weight,
                self.W_o.weight,
                num_heads=H,
                hd=hd,
                num_chunks_base=self.num_chunks,
                top_k_chunks=self.top_k_chunks,
                hisa_top_m_tokens=self.hisa_top_m_tokens,
                chunk_size=chunk_size,
                sparse_block_size=self.sparse_block_size,
                kv_inject=kv_inject,
                use_fused=False,
            )

        with torch.no_grad():
            Q = _to_heads(self.W_q(x), B, N, H, hd)
            K = _to_heads(self.W_k(x), B, N, H, hd)
            pad_len = chunk_size * C - N
            K_pad = F.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
            chunk_reps = _compute_chunk_representatives(K_pad, C)
            routing_weights = _compute_past_only_routing(
                Q,
                chunk_reps,
                seq_len=N,
                num_chunks=C,
                chunk_size=chunk_size,
                hd=hd,
            )
            w = routing_weights.clamp(min=1e-8)
            self._routing_entropy = (-(w * w.log()).sum(dim=-1).mean()).detach()

        if x.device.type != 'cuda' or triton is None:
            return _strict_hisa_module_forward(
                x,
                self.W_q.weight,
                self.W_k.weight,
                self.W_v.weight,
                self.W_o.weight,
                num_heads=H,
                hd=hd,
                num_chunks_base=self.num_chunks,
                top_k_chunks=self.top_k_chunks,
                hisa_top_m_tokens=self.hisa_top_m_tokens,
                chunk_size=chunk_size,
                sparse_block_size=self.sparse_block_size,
                use_fused=False,
            )

        return _StrictHISAModuleFn.apply(
            x,
            self.W_q.weight,
            self.W_k.weight,
            self.W_v.weight,
            self.W_o.weight,
            H,
            hd,
            self.num_chunks,
            self.top_k_chunks,
            self.hisa_top_m_tokens,
            chunk_size,
            self.sparse_block_size,
            self.backward_impl,
        )


__all__ = ["HierarchicalSparseAttentionV17HISAStrictTriton"]
