"""
Hierarchical Sparse Attention V15-HISA — HISA token-level refinement.

Extends V14 with HISA's two-stage token filtering:
  Stage 1: Select top-k chunks (V14)
  Stage 2: Within selected chunks, select top-m tokens (HISA)

This keeps V14's train-time log-routing gradient path while adding HISA's
token-level sparsity mask.

Key design: Token-level refinement is done in PyTorch (cheap, only 256 tokens max),
then passed to the Triton kernel via a boolean mask.

Usage: Replace HierarchicalSparseAttentionV14 in the training script.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


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
    routing_logits = routing_logits.masked_fill(~causal_ok[None, None], float('-inf'))

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

    This preserves the guaranteed self-chunk entry instead of re-truncating the
    packed selection based on c_q. Padding remains represented as -1 and is
    handled by the returned valid mask.
    """
    selected = top_k_packed[:, :, c_q, :]
    valid = selected >= 0
    ci = selected.clamp(0, num_chunks - 1)
    return selected, valid, ci


def _build_stage2_token_mask(
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
) -> tuple[torch.Tensor, int]:
    """
    Build Stage-2 token-level refinement mask within selected chunks.

    Returns (token_mask_packed, m_actual).
    token_mask_packed shape: (B, H, N_padded) — float 0/1 mask.
    m_actual: the actual top-m used (clamped to chunk_size).
    """
    device = Q_pad.device
    m = hisa_top_m_tokens
    C = num_chunks
    m_actual = min(m, chunk_size)

    token_hits = torch.zeros(B, H, N, dtype=torch.int32, device=device)

    K_reshaped = K_pad.view(B, H, C, chunk_size, Q_pad.shape[-1])
    Q_reshaped = Q_pad.view(B, H, C, chunk_size, Q_pad.shape[-1])

    b_idx_3d = torch.arange(B, device=device).view(B, 1, 1)
    h_idx_3d = torch.arange(H, device=device).view(1, H, 1)

    for c_q in range(C):
        q_start = c_q * chunk_size
        if q_start >= N:
            break

        selected, valid, ci = _prepare_stage2_selected_chunks(
            top_k_packed, c_q=c_q, num_chunks=C
        )

        k_slices = K_reshaped[b_idx_3d, h_idx_3d, ci]

        q_slice = Q_reshaped[:, :, c_q, :].unsqueeze(2)
        scores = torch.matmul(q_slice, k_slices.transpose(-2, -1)) / math.sqrt(Q_pad.shape[-1])

        _, top_m_idx = scores.topk(m_actual, dim=-1)

        ci_expanded = ci.unsqueeze(-1).unsqueeze(-1)
        flat_pos = (ci_expanded * chunk_size + top_m_idx).clamp(max=N - 1)
        updates = valid.unsqueeze(-1).unsqueeze(-1).expand_as(flat_pos).to(torch.int32)
        token_hits.scatter_add_(2, flat_pos.reshape(B, H, -1), updates.reshape(B, H, -1))

    token_mask = token_hits > 0
    N_padded = chunk_size * C
    token_mask_padded = F.pad(token_mask, (0, N_padded - N)) if N_padded > N else token_mask
    token_mask_packed = token_mask_padded.float()

    return token_mask_packed, m_actual



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
    chunk 0.

    The previous implementation only packed rows for query chunks 1..C-1, which
    left the first query chunk without any Stage-2 metadata. That exactly matches
    the passkey cliff where d<=32 (same first chunk) fails but d>=64 succeeds.
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
        if n_others > 0:
            w_others = w_mean_full.clone()
            w_others[:, :, c_q] = float('-inf')
            topk_others = min(top_k_chunks - 1, n_others)
            _, idx_others = w_others.topk(topk_others, dim=-1)
            idx = torch.cat([
                idx_others,
                torch.full((B, H, 1), c_q, dtype=torch.long, device=device)
            ], dim=-1)
        else:
            idx = torch.full((B, H, 1), 0, dtype=torch.long, device=device)

        if idx.shape[-1] < top_k_chunks:
            pad = torch.full((B, H, top_k_chunks - idx.shape[-1]), -1, dtype=torch.long, device=device)
            idx = torch.cat([idx, pad], dim=-1)

        top_k_list.append(idx)

    return torch.stack(top_k_list, dim=2)


# ---------------------------------------------------------------------------
# Triton forward: HISA token-level refinement within selected chunks
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_fwd_hisa(
    Q, K, V, ROUTING_W, TOP_K_IDX, TOKEN_MASK, OUT, LSE_OUT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_tm, stride_tmh, stride_tmp,  # token mask strides: batch, head, position
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lseb, stride_lseh, stride_lsen,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_HD: tl.constexpr,
    TRAINING: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    qs = q_start + tl.arange(0, CHUNK_SIZE)
    qm = qs < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    q_c = tl.load(q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
                   mask=qm[:, None] & dm[None, :], other=0.0)

    mi = tl.full([CHUNK_SIZE], float('-inf'), tl.float32)
    li = tl.zeros([CHUNK_SIZE], tl.float32)
    acc = tl.zeros([CHUNK_SIZE, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_mask_base = TOKEN_MASK + b * stride_tm + h * stride_tmh

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + c_q * K_VAL * stride_tk + ki * stride_tk).to(tl.int32)

        still_valid = chunk_idx >= 0
        if still_valid:
            k_start = chunk_idx * CHUNK_SIZE
            ks = k_start + tl.arange(0, CHUNK_SIZE)
            km = ks < N

            # HISA: Load token mask to select only top-m tokens within chunk (1D)
            token_sel = tl.load(token_mask_base + ks * stride_tmp,
                                mask=km, other=0.0)

            k_block = tl.load(k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
                              mask=km[:, None] & dm[None, :], other=0.0)

            q_f = q_c.to(tl.float32)
            k_f = k_block.to(tl.float32)
            s = tl.dot(q_f, tl.trans(k_f)) * sc

            causal = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :]
            s = tl.where(causal, s, float('-inf'))

            # HISA: Mask out non-selected tokens within chunk
            token_mask_2d = token_sel[None, :]
            s = tl.where(token_mask_2d > 0, s, float('-inf'))

            if TRAINING:
                rw = tl.load(ROUTING_W + b * stride_rb + h * stride_rh
                             + qs * stride_rn + chunk_idx * stride_rc,
                             mask=qm, other=1e-8).to(tl.float32)
                log_rw = tl.log(tl.maximum(rw, 1e-8))
                s = tl.where(causal, s + log_rw[:, None], float('-inf'))

            m_new = tl.max(s, axis=1)
            has_prev = mi > float('-inf')
            has_curr = m_new > float('-inf')
            has_any = has_prev | has_curr
            mn_raw = tl.maximum(mi, m_new)
            mn = tl.where(has_any, mn_raw, tl.zeros_like(mn_raw))
            cor = tl.where(has_prev,
                           tl.math.exp2((mi - mn) * 1.4426950408889634),
                           tl.zeros_like(mi))
            p_raw = tl.math.exp2((s - mn[:, None]) * 1.4426950408889634)
            p = tl.where(causal, p_raw, 0.0)

            li = tl.where(has_any, li * cor + tl.sum(p, axis=1), li)
            mi = tl.where(has_any, mn_raw, mi)

            v_block = tl.load(v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
                              mask=km[:, None] & dm[None, :], other=0.0)
            p_f = p.to(tl.float32)
            v_f = v_block.to(tl.float32)
            if CHUNK_SIZE >= 16:
                acc = acc * cor[:, None] + tl.dot(p_f, v_f)
            else:
                acc = acc * cor[:, None] + tl.sum(p_f[:, :, None] * v_f[None, :, :], axis=1)

    ls = tl.where(li > 0.0, li, 1.0)
    acc = acc / ls[:, None]
    safe_mi = tl.where(mi > float('-inf'), mi, tl.zeros_like(mi))
    lse = tl.where(mi > float('-inf'), safe_mi + tl.log(ls), float('-inf'))

    o_base = OUT + b * stride_ob + h * stride_oh
    tl.store(o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=qm[:, None] & dm[None, :])
    lse_base = LSE_OUT + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + qs * stride_lsen, lse, mask=qm)


# ---------------------------------------------------------------------------
# Triton backward: dQ (direct), dK/dV (atomic), DRW (direct)
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_bwd_hisa(
    Q, K, V, O, DO, LSE_OUT, ROUTING_W, TOP_K_IDX, TOKEN_MASK,
    DQ, DK, DV, DRW,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lseb, stride_lseh, stride_lsen,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_tm, stride_tmh, stride_tmp,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_drb, stride_drh, stride_drn, stride_drc,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_HD: tl.constexpr,
    TRAINING: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    qs = q_start + tl.arange(0, CHUNK_SIZE)
    qm = qs < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh

    q_c = tl.load(q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
                   mask=qm[:, None] & dm[None, :], other=0.0)
    do_c = tl.load(do_base + qs[:, None] * stride_don + ds[None, :] * stride_dod,
                    mask=qm[:, None] & dm[None, :], other=0.0)
    o_c = tl.load(o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
                  mask=qm[:, None] & dm[None, :], other=0.0)
    lse_c = tl.load(LSE_OUT + b * stride_lseb + h * stride_lseh + qs * stride_lsen,
                    mask=qm, other=0.0)

    D_c = tl.sum(do_c.to(tl.float32) * o_c.to(tl.float32), axis=1)
    dq_c = tl.zeros([CHUNK_SIZE, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_mask_base = TOKEN_MASK + b * stride_tm + h * stride_tmh

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + c_q * K_VAL * stride_tk + ki * stride_tk).to(tl.int32)

        still_valid = chunk_idx >= 0
        if still_valid:
            k_start = chunk_idx * CHUNK_SIZE
            ks = k_start + tl.arange(0, CHUNK_SIZE)
            km = ks < N

            token_sel = tl.load(token_mask_base + ks * stride_tmp,
                                mask=km, other=0.0)

            k_block = tl.load(k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
                              mask=km[:, None] & dm[None, :], other=0.0)
            v_block = tl.load(v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
                              mask=km[:, None] & dm[None, :], other=0.0)

            q_f = q_c.to(tl.float32)
            k_f = k_block.to(tl.float32)
            s = tl.dot(q_f, tl.trans(k_f)) * sc

            causal = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :]
            selected = causal & (token_sel[None, :] > 0)

            if TRAINING:
                rw = tl.load(ROUTING_W + b * stride_rb + h * stride_rh
                             + qs * stride_rn + chunk_idx * stride_rc,
                             mask=qm, other=1e-8).to(tl.float32)
                rw_safe = tl.maximum(rw, 1e-8)
                log_rw = tl.log(rw_safe)
                s = tl.where(selected, s + log_rw[:, None], float('-inf'))
            else:
                rw_safe = tl.full([CHUNK_SIZE], 1.0, tl.float32)
                s = tl.where(selected, s, float('-inf'))

            has_lse = lse_c > float('-inf')
            safe_lse = tl.where(has_lse, lse_c, tl.zeros_like(lse_c))
            alpha = tl.where(selected & has_lse[:, None],
                             tl.exp(s - safe_lse[:, None]), 0.0)

            do_f = do_c.to(tl.float32)
            v_f = v_block.to(tl.float32)
            dot_rv = tl.dot(do_f, tl.trans(v_f))
            ds_matrix = alpha * (dot_rv - D_c[:, None])

            dq_c += tl.dot(ds_matrix, k_f) * sc

            dk_block = tl.dot(tl.trans(ds_matrix), q_f) * sc
            tl.atomic_add(DK + b * stride_dkb + h * stride_dkh
                          + ks[:, None] * stride_dkn + ds[None, :] * stride_dkd,
                          tl.where(km[:, None] & dm[None, :], dk_block, 0.0),
                          mask=km[:, None] & dm[None, :])

            if CHUNK_SIZE >= 16:
                dv_block = tl.dot(tl.trans(alpha), do_f)
            else:
                dv_block = tl.sum(tl.trans(alpha)[:, :, None] * do_f[None, :, :], axis=1)
            tl.atomic_add(DV + b * stride_dvb + h * stride_dvh
                          + ks[:, None] * stride_dvn + ds[None, :] * stride_dvd,
                          tl.where(km[:, None] & dm[None, :], dv_block, 0.0),
                          mask=km[:, None] & dm[None, :])

            if TRAINING:
                drw_accum = tl.sum(tl.where(selected, ds_matrix, 0.0), axis=1)
                drw_chunk = drw_accum / rw_safe
                tl.store(DRW + b * stride_drb + h * stride_drh
                         + qs * stride_drn + chunk_idx * stride_drc,
                         tl.where(qm, drw_chunk, 0.0), mask=qm)

    dq_base = DQ + b * stride_dqb + h * stride_dqh
    tl.store(dq_base + qs[:, None] * stride_dqn + ds[None, :] * stride_dqd,
             dq_c.to(tl.bfloat16), mask=qm[:, None] & dm[None, :])


# ---------------------------------------------------------------------------
# Autograd function: Triton forward + Triton backward
# ---------------------------------------------------------------------------

class _DSRHISAAttendFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, routing_weights, top_k_packed, token_mask_packed,
                chunk_size, training_flag,
                N_orig=None, pad_len=None, hisa_top_m_tokens=None, effective_temp=None):
        B, H, N, hd = Q.shape
        C = routing_weights.shape[-1]
        k_val = top_k_packed.shape[-1]
        device = Q.device

        # Infer missing metadata for backward replay
        if N_orig is None:
            N_orig = N
        if pad_len is None:
            pad_len = 0
        if hisa_top_m_tokens is None:
            hisa_top_m_tokens = 32  # default
        if effective_temp is None:
            effective_temp = 1.0

        BLOCK_HD = _next_pow2(hd)

        out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
        lse_out = torch.full((B, H, N), float('-inf'), dtype=torch.float32, device=device)

        if C > 1:
            top_k_flat = top_k_packed.reshape(B, H, -1).contiguous().to(torch.int32)
            token_mask_flat = token_mask_packed.reshape(B, H, N).contiguous()

            _nw = 4
            _ns = 2
            grid = (B * H, C)

            _dsr_fwd_hisa[grid](
                Q, K, V, routing_weights, top_k_flat, token_mask_flat, out, lse_out,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                routing_weights.stride(0), routing_weights.stride(1),
                routing_weights.stride(2), routing_weights.stride(3),
                top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                token_mask_flat.stride(0), token_mask_flat.stride(1), token_mask_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_HD=BLOCK_HD,
                TRAINING=1 if training_flag else 0,
                num_warps=_nw, num_stages=_ns,
            )
        else:
            top_k_flat = top_k_packed.reshape(B, H, -1).contiguous().to(torch.int32)
            token_mask_flat = token_mask_packed.reshape(B, H, N).contiguous()

        # Check if replay mode is enabled
        replay_mode = os.environ.get('HISA_RECOMPUTE', 'none').lower()

        if replay_mode in ('out_lse', 'all'):
            # Replay V1/V2: do NOT save out/lse_out — recompute in backward
            ctx.save_for_backward(Q, K, V, routing_weights,
                                  top_k_flat, token_mask_flat)
        else:
            # Baseline: save everything
            ctx.save_for_backward(Q, K, V, routing_weights, out, lse_out,
                                  top_k_flat, token_mask_flat)

        ctx.chunk_size = chunk_size
        ctx.training_flag = training_flag
        ctx.C = C
        ctx.k_val = k_val
        ctx.BLOCK_HD = BLOCK_HD
        ctx.replay_mode = replay_mode
        # Replay metadata
        ctx.N_orig = N_orig
        ctx.pad_len = pad_len
        ctx.hisa_top_m_tokens = hisa_top_m_tokens
        ctx.effective_temp = effective_temp
        return out

    @staticmethod
    def backward(ctx, grad_output):
        replay_mode = ctx.replay_mode
        saved = ctx.saved_tensors

        if replay_mode in ('out_lse', 'all'):
            # Replay mode: 6 saved tensors (no out/lse_out)
            Q, K, V, routing_weights, top_k_flat, token_mask_flat = saved
            # Recompute out and lse_out by re-running forward kernel
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
            BLOCK_HD = ctx.BLOCK_HD
            device = Q.device

            out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
            lse_out = torch.full((B, H, N), float('-inf'), dtype=torch.float32, device=device)

            if C > 1:
                _nw = 4
                _ns = 2
                grid = (B * H, C)
                _dsr_fwd_hisa[grid](
                    Q, K, V, routing_weights, top_k_flat, token_mask_flat, out, lse_out,
                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    routing_weights.stride(0), routing_weights.stride(1),
                    routing_weights.stride(2), routing_weights.stride(3),
                    top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                    token_mask_flat.stride(0), token_mask_flat.stride(1), token_mask_flat.stride(2),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                    N=N, H=H, HD=hd,
                    C=C, K_VAL=k_val,
                    CHUNK_SIZE=chunk_size, BLOCK_HD=BLOCK_HD,
                    TRAINING=1 if ctx.training_flag else 0,
                    num_warps=_nw, num_stages=_ns,
                )
        else:
            # Baseline: 8 saved tensors (includes out/lse_out)
            Q, K, V, routing_weights, out, lse_out, top_k_flat, token_mask_flat = saved
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
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
                token_mask_flat, dQ, dK, dV, dRW,
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
                token_mask_flat.stride(0), token_mask_flat.stride(1), token_mask_flat.stride(2),
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                dRW.stride(0), dRW.stride(1), dRW.stride(2), dRW.stride(3),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_HD=BLOCK_HD,
                TRAINING=1 if ctx.training_flag else 0,
                num_warps=_nw, num_stages=_ns,
            )

        return (dQ, dK.to(K.dtype), dV.to(V.dtype), dRW,
                None, None, None, None,  # top_k_packed, token_mask_packed, chunk_size, training_flag
                None, None, None, None)   # N_orig, pad_len, hisa_top_m_tokens, effective_temp


# ---------------------------------------------------------------------------
# Module — HISA token-level refinement
# ---------------------------------------------------------------------------

class HierarchicalSparseAttentionV15HISA(nn.Module):

    def __init__(self, D: int, H: int, hd: int,
                 num_chunks: int = 32, top_k_chunks: int = 4,
                 hisa_top_m_tokens: int = 32):
        super().__init__()
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.num_chunks = num_chunks
        self.top_k_chunks = top_k_chunks
        self.hisa_top_m_tokens = hisa_top_m_tokens  # HISA: tokens per chunk
        self.temperature = 1.0
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)
        self._routing_entropy: torch.Tensor | float = float("nan")

    def forward(self, x: torch.Tensor, kv_inject=None) -> torch.Tensor:
        B, N, _ = x.shape
        H, hd = self.H, self.hd
        C = self.num_chunks
        k = self.top_k_chunks
        m = self.hisa_top_m_tokens  # HISA: tokens per chunk
        chunk_size = math.ceil(N / C)
        device = x.device

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

        # Use replayable routing helper
        routing_weights, effective_temp = _compute_routing(
            Q, chunk_reps,
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

        # Stage 1: select top-k chunks with explicit coverage for query chunk 0.
        top_k_packed = _build_stage1_top_k_packed(
            routing_weights,
            seq_len=N,
            chunk_size=chunk_size,
            num_chunks=C,
            top_k_chunks=k,
        )

        # HISA Stage 2: Token-level refinement within selected chunks
        token_mask_packed, m_actual = _build_stage2_token_mask(
            Q_pad, K_pad, top_k_packed,
            B=B, H=H, N=N,
            num_chunks=C,
            chunk_size=chunk_size,
            hisa_top_m_tokens=m,
        )

        out = _DSRHISAAttendFn.apply(
            Q_pad, K_pad, V_pad, routing_weights_pad,
            top_k_packed, token_mask_packed,
            chunk_size, self.training,
            N, pad_len, m, effective_temp,
        )

        out_flat = out[:, :, :N, :].transpose(1, 2).reshape(B, N, H * hd)
        return self.W_o(out_flat)
