"""
Hierarchical Sparse Attention V15-HISA — corrected compact token refinement.

This version fixes the main V15 Stage-2 bug: token refinement is now scoped by
query chunk and selected chunk instead of using one global per-position mask.
The Triton kernels consume compact token indices [B, H, C_query, K, M] and only
load/dot the selected M tokens for each selected chunk, so the top-m refinement
also reduces QK/V work instead of computing a full key chunk and masking it
afterward.

Stage 1: select top-k chunks per query chunk, with the self chunk guaranteed.
Stage 2: within each selected chunk, select top-m key tokens for that query
         chunk. Selection metadata is built under no_grad; the train-time
         routing-gradient path is preserved by adding log(routing_weight) in
         the attention kernel.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl



def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _compute_chunk_representatives(K_pad: torch.Tensor, num_chunks: int) -> torch.Tensor:
    """
    Build Stage-1 chunk representatives without mean-pooling away a standout key.

    We select the highest-L2 token within each chunk and use that exact token
    vector as the chunk representative. This preserves a real token direction
    instead of constructing a synthetic per-dimension max vector, and it avoids
    the 1/chunk_size signal dilution of mean pooling.
    """
    B, H, N_padded, hd = K_pad.shape
    assert N_padded % num_chunks == 0, "K_pad must be padded to chunk boundaries"

    chunk_size = N_padded // num_chunks
    K_chunks = K_pad.reshape(B, H, num_chunks, chunk_size, hd)

    token_energy = K_chunks.float().square().sum(dim=-1)
    best_idx = token_energy.argmax(dim=3, keepdim=True)
    gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, -1, 1, hd)
    return torch.gather(K_chunks, dim=3, index=gather_idx).squeeze(3)


def _compute_routing(
    Q: torch.Tensor,
    chunk_reps: torch.Tensor,
    *,
    seq_len: int,
    num_chunks: int,
    chunk_size: int,
    hd: int,
    temperature: float,
    training: bool,
) -> tuple[torch.Tensor, float]:
    """
    Compute routing logits and weights with causal masking and temperature.

    Returns (routing_weights, effective_temperature).
    routing_weights shape: (B, H, N, num_chunks)
    """
    device = Q.device

    routing_logits = torch.matmul(Q, chunk_reps.transpose(-2, -1)) / math.sqrt(hd)

    positions = torch.arange(seq_len, device=device)
    chunk_starts = torch.arange(num_chunks, device=device) * chunk_size
    causal_ok = chunk_starts.unsqueeze(0) < positions.unsqueeze(1)
    routing_logits = routing_logits.masked_fill(~causal_ok[None, None], float("-inf"))

    temp = temperature if training else 1.0
    routing_weights = F.softmax(routing_logits / temp, dim=-1)
    routing_weights = torch.nan_to_num(routing_weights, nan=0.0)

    return routing_weights, temp


def _prepare_stage2_selected_chunks(
    top_k_packed: torch.Tensor,
    c_q: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Consume the full packed Stage-1 selection for query chunk c_q.

    Padding remains represented as -1 and is handled by the returned valid mask.
    """
    selected = top_k_packed[:, :, c_q, :]
    valid = selected >= 0
    ci = selected.clamp(0, num_chunks - 1)
    return selected, valid, ci


def _build_stage1_top_k_packed(
    routing_weights: torch.Tensor,
    *,
    seq_len: int,
    chunk_size: int,
    num_chunks: int,
    top_k_chunks: int,
) -> torch.Tensor:
    """
    Build the packed Stage-1 chunk selection for every query chunk, including
    chunk 0. The self chunk is guaranteed in the packed selection.
    """
    B, H, _, _ = routing_weights.shape
    device = routing_weights.device

    top_k_list = []
    for c_q in range(num_chunks):
        q_start = c_q * chunk_size
        q_end = min(q_start + chunk_size, seq_len)
        if q_start >= seq_len:
            top_k_list.append(torch.full((B, H, top_k_chunks), -1, dtype=torch.long, device=device))
            continue

        n_valid_with_self = c_q + 1
        w_c_full = routing_weights[:, :, q_start:q_end, :n_valid_with_self]
        w_mean_full = w_c_full.mean(dim=2)

        n_others = c_q
        if n_others > 0 and top_k_chunks > 1:
            w_others = w_mean_full.clone()
            w_others[:, :, c_q] = float("-inf")
            topk_others = min(top_k_chunks - 1, n_others)
            _, idx_others = w_others.topk(topk_others, dim=-1)
            idx = torch.cat([
                idx_others,
                torch.full((B, H, 1), c_q, dtype=torch.long, device=device),
            ], dim=-1)
        else:
            idx = torch.full((B, H, 1), c_q, dtype=torch.long, device=device)

        if idx.shape[-1] < top_k_chunks:
            pad = torch.full((B, H, top_k_chunks - idx.shape[-1]), -1, dtype=torch.long, device=device)
            idx = torch.cat([idx, pad], dim=-1)

        top_k_list.append(idx)

    return torch.stack(top_k_list, dim=2)


def _build_stage2_token_indices(
    Q_pad: torch.Tensor,
    K_pad: torch.Tensor,
    top_k_packed: torch.Tensor,
    *,
    B: int,
    H: int,
    N: int,
    num_chunks: int,
    chunk_size: int,
    hisa_top_m_tokens: int,
    collect_stats: bool = False,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Build query-scoped compact Stage-2 token indices.

    Returns:
      token_idx_packed: int32 [B, H, C_query, K, M], where invalid slots are -1.
      m_actual:         min(max(hisa_top_m_tokens, 1), chunk_size)
      selected_fraction: scalar tensor with fraction of valid compact slots, or NaN when telemetry is disabled.

    Unlike the old global [B,H,N] token mask, this metadata is scoped to the
    query chunk and the selected chunk. It therefore cannot be polluted by token
    choices made by later query chunks, and the Triton kernel can gather only M
    keys/values instead of computing a full key chunk.
    """
    device = Q_pad.device
    C = num_chunks
    K_sel = top_k_packed.shape[-1]
    hd = Q_pad.shape[-1]
    m_actual = min(max(int(hisa_top_m_tokens), 1), chunk_size)

    token_idx = torch.full(
        (B, H, C, K_sel, m_actual),
        -1,
        dtype=torch.int32,
        device=device,
    )

    K_reshaped = K_pad.view(B, H, C, chunk_size, hd)
    Q_reshaped = Q_pad.view(B, H, C, chunk_size, hd)

    b_idx_3d = torch.arange(B, device=device).view(B, 1, 1)
    h_idx_3d = torch.arange(H, device=device).view(1, H, 1)
    q_offsets = torch.arange(chunk_size, device=device)
    k_offsets = torch.arange(chunk_size, device=device)

    for c_q in range(C):
        q_start = c_q * chunk_size
        if q_start >= N:
            break

        selected, valid_chunks, ci = _prepare_stage2_selected_chunks(
            top_k_packed, c_q=c_q, num_chunks=C
        )
        del selected

        # [B,H,K,chunk,hd]
        k_slices = K_reshaped[b_idx_3d, h_idx_3d, ci]
        # [B,H,chunk,hd]
        q_slice = Q_reshaped[:, :, c_q, :]

        # [B,H,K,Q,Kc]
        scores = torch.matmul(
            q_slice.unsqueeze(2),
            k_slices.transpose(-2, -1),
        ) / math.sqrt(hd)

        q_abs = q_start + q_offsets
        k_abs = ci[..., None] * chunk_size + k_offsets.view(1, 1, 1, chunk_size)
        q_valid = q_abs < N
        k_valid = k_abs < N
        causal = (
            k_valid.unsqueeze(-2)
            & q_valid.view(1, 1, 1, chunk_size, 1)
            & (k_abs.unsqueeze(-2) < q_abs.view(1, 1, 1, chunk_size, 1))
            & valid_chunks[..., None, None]
        )

        scores = scores.masked_fill(~causal, float("-inf"))

        # One compact token set per selected chunk for this query chunk. Max
        # over query rows preserves tokens that are important to any row in the
        # query chunk while keeping the kernel metadata compact.
        token_scores = scores.max(dim=-2).values  # [B,H,K,chunk]
        top_vals, top_m_idx = token_scores.topk(m_actual, dim=-1)
        flat_pos = ci[..., None] * chunk_size + top_m_idx
        finite = torch.isfinite(top_vals)
        valid_flat = finite & valid_chunks[..., None] & (flat_pos < N)
        flat_pos = torch.where(valid_flat, flat_pos, torch.full_like(flat_pos, -1))
        token_idx[:, :, c_q, :, :] = flat_pos.to(torch.int32)

    if collect_stats:
        selected_fraction = (token_idx >= 0).float().mean().detach()
    else:
        selected_fraction = torch.full((), float('nan'), device=device)
    return token_idx, m_actual, selected_fraction


# ---------------------------------------------------------------------------
# Triton forward: compact HISA token refinement within selected chunks
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_fwd_hisa(
    Q, K, V, ROUTING_W, TOP_K_IDX, TOKEN_IDX, OUT, LSE_OUT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_ib, stride_ih, stride_ip,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lseb, stride_lseh, stride_lsen,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_CHUNK: tl.constexpr, BLOCK_HD: tl.constexpr,
    M_VAL: tl.constexpr, M_PAD: tl.constexpr,
    TRAINING: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    q_offsets = tl.arange(0, BLOCK_CHUNK)
    qs = q_start + q_offsets
    qm = (qs < N) & (q_offsets < CHUNK_SIZE)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    ms = tl.arange(0, M_PAD)
    mm = ms < M_VAL

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    q_c = tl.load(
        q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )

    mi = tl.full([BLOCK_CHUNK], float("-inf"), tl.float32)
    li = tl.zeros([BLOCK_CHUNK], tl.float32)
    acc = tl.zeros([BLOCK_CHUNK, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_idx_base = TOKEN_IDX + b * stride_ib + h * stride_ih

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + (c_q * K_VAL + ki) * stride_tk).to(tl.int32)
        chunk_valid = chunk_idx >= 0
        safe_chunk_idx = tl.maximum(chunk_idx, 0)
        idx_off = ((c_q * K_VAL + ki) * M_VAL + ms) * stride_ip
        ks = tl.load(token_idx_base + idx_off, mask=mm, other=-1).to(tl.int32)
        km = (ks >= 0) & (ks < N) & mm & chunk_valid

        k_block = tl.load(
            k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )

        q_f = q_c.to(tl.float32)
        k_f = k_block.to(tl.float32)
        s = tl.dot(q_f, tl.trans(k_f)) * sc

        selected = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :] & chunk_valid
        s = tl.where(selected, s, float("-inf"))

        if TRAINING:
            rw = tl.load(
                ROUTING_W + b * stride_rb + h * stride_rh + qs * stride_rn + safe_chunk_idx * stride_rc,
                mask=qm & chunk_valid,
                other=1e-8,
            ).to(tl.float32)
            log_rw = tl.log(tl.maximum(rw, 1e-8))
            s = tl.where(selected, s + log_rw[:, None], float("-inf"))

        m_new = tl.max(s, axis=1)
        has_prev = mi > float("-inf")
        has_curr = m_new > float("-inf")
        has_any = has_prev | has_curr
        mn_raw = tl.maximum(mi, m_new)
        mn = tl.where(has_any, mn_raw, tl.zeros_like(mn_raw))
        cor = tl.where(
            has_prev,
            tl.math.exp2((mi - mn) * 1.4426950408889634),
            tl.zeros_like(mi),
        )
        p_raw = tl.math.exp2((s - mn[:, None]) * 1.4426950408889634)
        p = tl.where(selected, p_raw, 0.0)

        li = tl.where(has_any, li * cor + tl.sum(p, axis=1), li)
        mi = tl.where(has_any, mn_raw, mi)

        v_block = tl.load(
            v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )
        acc = acc * cor[:, None] + tl.dot(p.to(tl.float32), v_block.to(tl.float32))

    ls = tl.where(li > 0.0, li, 1.0)
    acc = acc / ls[:, None]
    safe_mi = tl.where(mi > float("-inf"), mi, tl.zeros_like(mi))
    lse = tl.where(mi > float("-inf"), safe_mi + tl.log(ls), float("-inf"))

    o_base = OUT + b * stride_ob + h * stride_oh
    tl.store(
        o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=qm[:, None] & dm[None, :],
    )
    lse_base = LSE_OUT + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + qs * stride_lsen, lse, mask=qm)


# ---------------------------------------------------------------------------
# Triton backward: dQ direct, dK/dV atomic, dRouting direct
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_bwd_hisa(
    Q, K, V, O, DO, LSE_OUT, ROUTING_W, TOP_K_IDX, TOKEN_IDX,
    DQ, DK, DV, DRW,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lseb, stride_lseh, stride_lsen,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_ib, stride_ih, stride_ip,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_drb, stride_drh, stride_drn, stride_drc,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_CHUNK: tl.constexpr, BLOCK_HD: tl.constexpr,
    M_VAL: tl.constexpr, M_PAD: tl.constexpr,
    TRAINING: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    q_offsets = tl.arange(0, BLOCK_CHUNK)
    qs = q_start + q_offsets
    qm = (qs < N) & (q_offsets < CHUNK_SIZE)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    ms = tl.arange(0, M_PAD)
    mm = ms < M_VAL

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh

    q_c = tl.load(
        q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    do_c = tl.load(
        do_base + qs[:, None] * stride_don + ds[None, :] * stride_dod,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    o_c = tl.load(
        o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    lse_c = tl.load(
        LSE_OUT + b * stride_lseb + h * stride_lseh + qs * stride_lsen,
        mask=qm,
        other=0.0,
    )

    D_c = tl.sum(do_c.to(tl.float32) * o_c.to(tl.float32), axis=1)
    dq_c = tl.zeros([BLOCK_CHUNK, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_idx_base = TOKEN_IDX + b * stride_ib + h * stride_ih

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + (c_q * K_VAL + ki) * stride_tk).to(tl.int32)
        chunk_valid = chunk_idx >= 0
        safe_chunk_idx = tl.maximum(chunk_idx, 0)
        idx_off = ((c_q * K_VAL + ki) * M_VAL + ms) * stride_ip
        ks = tl.load(token_idx_base + idx_off, mask=mm, other=-1).to(tl.int32)
        km = (ks >= 0) & (ks < N) & mm & chunk_valid

        k_block = tl.load(
            k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )
        v_block = tl.load(
            v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )

        q_f = q_c.to(tl.float32)
        k_f = k_block.to(tl.float32)
        s = tl.dot(q_f, tl.trans(k_f)) * sc

        selected = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :] & chunk_valid

        if TRAINING:
            rw = tl.load(
                ROUTING_W + b * stride_rb + h * stride_rh + qs * stride_rn + safe_chunk_idx * stride_rc,
                mask=qm & chunk_valid,
                other=1e-8,
            ).to(tl.float32)
            rw_safe = tl.maximum(rw, 1e-8)
            log_rw = tl.log(rw_safe)
            s = tl.where(selected, s + log_rw[:, None], float("-inf"))
        else:
            rw_safe = tl.full([BLOCK_CHUNK], 1.0, tl.float32)
            s = tl.where(selected, s, float("-inf"))

        has_lse = lse_c > float("-inf")
        safe_lse = tl.where(has_lse, lse_c, tl.zeros_like(lse_c))
        alpha = tl.where(selected & has_lse[:, None], tl.exp(s - safe_lse[:, None]), 0.0)

        do_f = do_c.to(tl.float32)
        v_f = v_block.to(tl.float32)
        dot_rv = tl.dot(do_f, tl.trans(v_f))
        ds_matrix = alpha * (dot_rv - D_c[:, None])

        dq_c += tl.dot(ds_matrix, k_f) * sc

        dk_block = tl.dot(tl.trans(ds_matrix), q_f) * sc
        tl.atomic_add(
            DK + b * stride_dkb + h * stride_dkh + ks[:, None] * stride_dkn + ds[None, :] * stride_dkd,
            tl.where(km[:, None] & dm[None, :], dk_block, 0.0),
            mask=km[:, None] & dm[None, :],
        )

        dv_block = tl.dot(tl.trans(alpha), do_f)
        tl.atomic_add(
            DV + b * stride_dvb + h * stride_dvh + ks[:, None] * stride_dvn + ds[None, :] * stride_dvd,
            tl.where(km[:, None] & dm[None, :], dv_block, 0.0),
            mask=km[:, None] & dm[None, :],
        )

        if TRAINING:
            drw_accum = tl.sum(tl.where(selected, ds_matrix, 0.0), axis=1)
            drw_chunk = drw_accum / rw_safe
            tl.store(
                DRW + b * stride_drb + h * stride_drh + qs * stride_drn + safe_chunk_idx * stride_drc,
                tl.where(qm & chunk_valid, drw_chunk, 0.0),
                mask=qm & chunk_valid,
            )

    dq_base = DQ + b * stride_dqb + h * stride_dqh
    tl.store(
        dq_base + qs[:, None] * stride_dqn + ds[None, :] * stride_dqd,
        dq_c.to(tl.bfloat16),
        mask=qm[:, None] & dm[None, :],
    )


# ---------------------------------------------------------------------------
# Autograd function: Triton forward + Triton backward
# ---------------------------------------------------------------------------

class _DSRHISAAttendFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        routing_weights,
        top_k_packed,
        token_idx_packed,
        chunk_size,
        training_flag,
        N_orig=None,
        pad_len=None,
        hisa_top_m_tokens=None,
        effective_temp=None,
    ):
        B, H, N, hd = Q.shape
        C = routing_weights.shape[-1]
        k_val = top_k_packed.shape[-1]
        m_val = token_idx_packed.shape[-1]
        device = Q.device

        if N_orig is None:
            N_orig = N
        if pad_len is None:
            pad_len = 0
        if hisa_top_m_tokens is None:
            hisa_top_m_tokens = m_val
        if effective_temp is None:
            effective_temp = 1.0

        BLOCK_HD = _next_pow2(hd)
        BLOCK_CHUNK = _next_pow2(chunk_size)
        M_PAD = max(16, _next_pow2(m_val))

        out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
        lse_out = torch.full((B, H, N), float("-inf"), dtype=torch.float32, device=device)

        top_k_flat = top_k_packed.contiguous().reshape(B, H, -1).to(torch.int32)
        token_idx_flat = token_idx_packed.contiguous().reshape(B, H, -1).to(torch.int32)

        if C > 1:
            _nw = 4
            _ns = 2
            grid = (B * H, C)

            _dsr_fwd_hisa[grid](
                Q, K, V, routing_weights, top_k_flat, token_idx_flat, out, lse_out,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                routing_weights.stride(0), routing_weights.stride(1),
                routing_weights.stride(2), routing_weights.stride(3),
                top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_CHUNK=BLOCK_CHUNK, BLOCK_HD=BLOCK_HD,
                M_VAL=m_val, M_PAD=M_PAD,
                TRAINING=1 if training_flag else 0,
                num_warps=_nw, num_stages=_ns,
            )

        replay_mode = os.environ.get("HISA_RECOMPUTE", "none").lower()

        if replay_mode in ("out_lse", "all"):
            ctx.save_for_backward(Q, K, V, routing_weights, top_k_flat, token_idx_flat)
        else:
            ctx.save_for_backward(Q, K, V, routing_weights, out, lse_out, top_k_flat, token_idx_flat)

        ctx.chunk_size = chunk_size
        ctx.training_flag = training_flag
        ctx.C = C
        ctx.k_val = k_val
        ctx.m_val = m_val
        ctx.M_PAD = M_PAD
        ctx.BLOCK_CHUNK = BLOCK_CHUNK
        ctx.BLOCK_HD = BLOCK_HD
        ctx.replay_mode = replay_mode
        ctx.N_orig = N_orig
        ctx.pad_len = pad_len
        ctx.hisa_top_m_tokens = hisa_top_m_tokens
        ctx.effective_temp = effective_temp
        return out

    @staticmethod
    def backward(ctx, grad_output):
        replay_mode = ctx.replay_mode
        saved = ctx.saved_tensors

        if replay_mode in ("out_lse", "all"):
            Q, K, V, routing_weights, top_k_flat, token_idx_flat = saved
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
            m_val = ctx.m_val
            M_PAD = ctx.M_PAD
            BLOCK_CHUNK = ctx.BLOCK_CHUNK
            BLOCK_HD = ctx.BLOCK_HD
            device = Q.device

            out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
            lse_out = torch.full((B, H, N), float("-inf"), dtype=torch.float32, device=device)

            if C > 1:
                _nw = 4
                _ns = 2
                grid = (B * H, C)
                _dsr_fwd_hisa[grid](
                    Q, K, V, routing_weights, top_k_flat, token_idx_flat, out, lse_out,
                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    routing_weights.stride(0), routing_weights.stride(1),
                    routing_weights.stride(2), routing_weights.stride(3),
                    top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                    token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                    N=N, H=H, HD=hd,
                    C=C, K_VAL=k_val,
                    CHUNK_SIZE=chunk_size, BLOCK_CHUNK=BLOCK_CHUNK, BLOCK_HD=BLOCK_HD,
                    M_VAL=m_val, M_PAD=M_PAD,
                    TRAINING=1 if ctx.training_flag else 0,
                    num_warps=_nw, num_stages=_ns,
                )
        else:
            Q, K, V, routing_weights, out, lse_out, top_k_flat, token_idx_flat = saved
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
            m_val = ctx.m_val
            M_PAD = ctx.M_PAD
            BLOCK_CHUNK = ctx.BLOCK_CHUNK
            BLOCK_HD = ctx.BLOCK_HD
            device = Q.device

        grad_output = grad_output.contiguous()

        dQ = torch.zeros_like(Q)
        dK = torch.zeros(B, H, N, hd, device=device, dtype=torch.float32)
        dV = torch.zeros(B, H, N, hd, device=device, dtype=torch.float32)
        dRW = torch.zeros_like(routing_weights)

        if C > 1:
            _nw = 4
            _ns = 2
            grid = (B * H, C)

            _dsr_bwd_hisa[grid](
                Q, K, V, out, grad_output, lse_out, routing_weights, top_k_flat,
                token_idx_flat, dQ, dK, dV, dRW,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                grad_output.stride(0), grad_output.stride(1),
                grad_output.stride(2), grad_output.stride(3),
                lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                routing_weights.stride(0), routing_weights.stride(1),
                routing_weights.stride(2), routing_weights.stride(3),
                top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                dRW.stride(0), dRW.stride(1), dRW.stride(2), dRW.stride(3),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_CHUNK=BLOCK_CHUNK, BLOCK_HD=BLOCK_HD,
                M_VAL=m_val, M_PAD=M_PAD,
                TRAINING=1 if ctx.training_flag else 0,
                num_warps=_nw, num_stages=_ns,
            )

        return (
            dQ,
            dK.to(K.dtype),
            dV.to(V.dtype),
            dRW,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Module — HISA token-level refinement
# ---------------------------------------------------------------------------

class HierarchicalSparseAttentionV15HISA(nn.Module):

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int = 32,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int = 32,
    ):
        super().__init__()
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.num_chunks = num_chunks
        self.top_k_chunks = top_k_chunks
        self.hisa_top_m_tokens = hisa_top_m_tokens
        self.temperature = 1.0
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)
        self._routing_entropy: torch.Tensor | float = float("nan")
        self._stage2_selected_fraction: torch.Tensor | float = float("nan")
        self.collect_telemetry = os.getenv("HISA_TELEMETRY", "0") == "1"

    def forward(self, x: torch.Tensor, kv_inject=None) -> torch.Tensor:
        # kv_inject is accepted for API compatibility with DSQG blocks, but HISA
        # intentionally does not consume NPCI K/V deltas in this architecture.
        del kv_inject

        B, N, _ = x.shape
        H, hd = self.H, self.hd
        C = self.num_chunks
        k = self.top_k_chunks
        m = self.hisa_top_m_tokens
        chunk_size = math.ceil(N / C)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, H, hd).transpose(1, 2)

        Q = to_heads(self.W_q(x))
        K = to_heads(self.W_k(x))
        V = to_heads(self.W_v(x))

        pad_len = chunk_size * C - N
        V_pad = F.pad(V, (0, 0, 0, pad_len)) if pad_len > 0 else V
        K_pad = F.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
        Q_pad = F.pad(Q, (0, 0, 0, pad_len)) if pad_len > 0 else Q
        chunk_reps = _compute_chunk_representatives(K_pad, num_chunks=C)

        routing_weights, effective_temp = _compute_routing(
            Q,
            chunk_reps,
            seq_len=N,
            num_chunks=C,
            chunk_size=chunk_size,
            hd=hd,
            temperature=self.temperature,
            training=self.training,
        )
        routing_weights_pad = F.pad(routing_weights, (0, 0, 0, pad_len)) if pad_len > 0 else routing_weights

        with torch.no_grad():
            w = routing_weights.clamp(min=1e-8)
            self._routing_entropy = (-(w * w.log()).sum(dim=-1).mean()).detach()

            top_k_packed = _build_stage1_top_k_packed(
                routing_weights,
                seq_len=N,
                chunk_size=chunk_size,
                num_chunks=C,
                top_k_chunks=k,
            )

            token_idx_packed, m_actual, selected_fraction = _build_stage2_token_indices(
                Q_pad,
                K_pad,
                top_k_packed,
                B=B,
                H=H,
                N=N,
                num_chunks=C,
                chunk_size=chunk_size,
                hisa_top_m_tokens=m,
                collect_stats=self.collect_telemetry,
            )
            self._stage2_selected_fraction = selected_fraction

        out = _DSRHISAAttendFn.apply(
            Q_pad,
            K_pad,
            V_pad,
            routing_weights_pad,
            top_k_packed,
            token_idx_packed,
            chunk_size,
            self.training,
            N,
            pad_len,
            m_actual,
            effective_temp,
        )

        out_flat = out[:, :, :N, :].transpose(1, 2).reshape(B, N, H * hd)
        return self.W_o(out_flat)
