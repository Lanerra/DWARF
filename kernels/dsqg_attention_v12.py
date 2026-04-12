"""
DSQG Attention V12 — Dual-path implementation with GPU auto-detection
======================================================================

Two-path kernel that auto-detects GPU architecture at init time:
  - sm_90 (H100/H200): gather K/V into contiguous buffers, then torch.einsum/matmul
  - sm_89 (4090): streaming fused Triton kernel with proper tiling and @triton.autotune

NPCI (phase rotation) DROPPED:
  Across 31 trained layers (d768_l32_fa8_best.pt checkpoint):
    phase_gain RMS = 0.0010
    phase_base RMS = 0.0224
    ratio = 4.4%
  The phase_gain signal is effectively zero — NPCI rotation has negligible effect.

  Kept as UNUSED parameters for checkpoint compatibility:
    - phase_base    [J_LARGE=10, H, 2]
    - phase_gain    [J_LARGE=10, H, 2]
    - query_probes  [2, HD]
    - key_probes    [2, HD]

Offset set: J=24
  ALL_OFFSETS = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]

Usage:
  from dsqg_attention_v12 import DSQGAttentionV12

Testing:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/test_v12.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

ALL_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
J = len(ALL_OFFSETS)
J_SMALL = 14
J_LARGE = 10
R_PLANES = 2
MAX_DELTA = max(ALL_OFFSETS)


def _detect_gpu_capability():
    """Detect GPU compute capability at runtime."""
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


@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 16, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 32, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 64, "TILE_HD": 64}, num_warps=4, num_stages=2),
    ],
    key=["N", "HD"],
)
@triton.jit
def _streaming_fwd_v12(
    Q, K, V, POS_BIAS, SCALE_EMBED, OUT, LSE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    TILE_N: tl.constexpr, TILE_HD: tl.constexpr,
):
    """Streaming fused forward kernel for sm_89 (4090) — online softmax across J offsets."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * TILE_N

    ns = n0 + tl.arange(0, TILE_N)
    nm = ns < N
    ds = tl.arange(0, TILE_HD)
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

    mi = tl.full([TILE_N], float('-inf'), tl.float32)
    li = tl.zeros([TILE_N], tl.float32)
    acc = tl.zeros([TILE_N, TILE_HD], tl.float32)

    for i in tl.static_range(24):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
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
            mask=dm,
            other=0.0
        ).to(tl.float32)
        s += tl.sum(q * se_i[None, :], axis=1) * sc
        s = tl.where(val, s, float('-inf'))

        mn = tl.maximum(mi, s)
        cor = tl.where(mi > float('-inf'), tl.exp(mi - mn), tl.zeros_like(mi))
        p = tl.where(val, tl.exp(s - mn), tl.zeros_like(s))
        li = li * cor + p
        mi = mn

        vt = tl.load(
            vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=val[:, None] & dm[None, :],
            other=0.0
        ).to(tl.float32)

        acc = acc * cor[:, None] + p[:, None] * vt

    # Guard: positions where ALL offsets were invalid (e.g. position 0)
    # have mi==-inf and li==0. Without this, lse_val=-inf and backward
    # computes exp(s - (-inf)) = inf -> NaN.
    all_invalid = mi == float('-inf')
    ls = tl.where(li > 0.0, li, 1.0)
    lse_val = tl.where(all_invalid, tl.zeros_like(mi), mi + tl.log(ls))
    acc = tl.where(all_invalid[:, None], tl.zeros_like(acc), acc / ls[:, None])

    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(
        ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=nm[:, None] & dm[None, :]
    )
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 16, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 32, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 64, "TILE_HD": 64}, num_warps=4, num_stages=2),
    ],
    key=["N", "HD"],
)
@triton.jit
def _compute_D_v12(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_db, stride_dh, stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    TILE_N: tl.constexpr, TILE_HD: tl.constexpr,
):
    """Compute D = sum(do * out, dim=-1) for backward pass."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * TILE_N
    ns = n0 + tl.arange(0, TILE_N)
    nm = ns < N
    ds = tl.arange(0, TILE_HD)
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


@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 16, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 32, "TILE_HD": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 64, "TILE_HD": 64}, num_warps=4, num_stages=2),
    ],
    key=["N", "HD"],
)
@triton.jit
def _streaming_bwd_dq_v12(
    Q, K, V, POS_BIAS, SCALE_EMBED,
    DO, LSE, Dv,
    DQ, DPB_BUF, DSE_BUF,
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
    H: tl.constexpr, N, HD: tl.constexpr,
    TILE_N: tl.constexpr, TILE_HD: tl.constexpr,
):
    """Backward pass for dQ, dpos_bias, dscale_embed."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = blk * TILE_N

    ns = n0 + tl.arange(0, TILE_N)
    nm = ns < N
    ds = tl.arange(0, TILE_HD)
    dm = ds < HD
    sc = 1.0 / tl.sqrt(HD * 1.0)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    dob = DO + b * stride_dob + h * stride_doh

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

    dq = tl.zeros([TILE_N, TILE_HD], tl.float32)

    for i in tl.static_range(24):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
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
            mask=dm,
            other=0.0
        ).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s += tl.sum(q * se_i[None, :], axis=1) * sc
        s = tl.where(val, s, float('-inf'))
        # Guard: if lse==-inf (all-invalid position), exp(s-lse) would be inf.
        # Clamp s - lse to <= 0 to prevent exp overflow from numerical precision issues.
        diff = tl.minimum(s - lse, 0.0)
        alpha = tl.where(val & (lse > float('-inf')), tl.exp(diff), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v = alpha * (dot_rv - Dval)

        dq += ds_v[:, None] * kt * sc
        dq += ds_v[:, None] * se_i[None, :] * sc

        tl.store(
            DPB_BUF + bh * stride_dpb_bh + blk * stride_dpb_blk + i,
            tl.sum(tl.where(val, ds_v, 0.0))
        )
        dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
        tl.store(
            DSE_BUF + bh * stride_dse_bh + blk * stride_dse_blk + i * HD + ds,
            tl.where(dm, dse_i, 0.0),
            mask=dm
        )

    tl.store(
        DQ + b * stride_dqb + h * stride_dqh + ns[:, None] * stride_dqn + ds[None, :] * stride_dqd,
        dq.to(tl.bfloat16),
        mask=nm[:, None] & dm[None, :]
    )


@triton.jit
def _streaming_bwd_dkdv_v12(
    Q, K, V, POS_BIAS, SCALE_EMBED,
    DO, LSE, Dv,
    DK, DV,
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
    H: tl.constexpr, N, HD: tl.constexpr,
    TILE_M: tl.constexpr, TILE_HD: tl.constexpr,
):
    """Backward pass for dK and dV."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    m0 = blk * TILE_M

    ms = m0 + tl.arange(0, TILE_M)
    mm = ms < N
    ds = tl.arange(0, TILE_HD)
    dm = ds < HD
    sc = 1.0 / tl.sqrt(HD * 1.0)

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh

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

    dk = tl.zeros([TILE_M, TILE_HD], tl.float32)
    dv = tl.zeros([TILE_M, TILE_HD], tl.float32)

    for i in tl.static_range(24):
        delta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024)[i]
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
            mask=dm,
            other=0.0
        ).to(tl.float32)

        s = tl.sum(qn * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s += tl.sum(qn * se_i[None, :], axis=1) * sc
        s = tl.where(val, s, float('-inf'))
        # Guard: if lse==-inf (all-invalid position), exp(s-lse) would be inf.
        # Clamp s - lsen to <= 0 to prevent exp overflow from numerical precision issues.
        # Softmax weights should always be <= 1, so exp(s - lsen) should be <= 1.
        diff = tl.minimum(s - lsen, 0.0)
        alpha = tl.where(val & (lsen > float('-inf')), tl.exp(diff), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v = alpha * (dot_rv - Dn)

        dk += ds_v[:, None] * qn * sc
        dv += alpha[:, None] * don

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


class _DSQGV12StreamingFn(torch.autograd.Function):
    """Streaming (sm_89) autograd function for V12."""

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        assert q.dtype == torch.bfloat16

        TILE_N = 32
        TILE_HD = _next_pow2(HD)

        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        grid = (B * H, triton.cdiv(N, TILE_N))

        _streaming_fwd_v12[grid](
            q, k, v, pos_bias, scale_embed, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            H=H, N=N, HD=HD,
        )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pos_bias, scale_embed, out, lse = ctx.saved_tensors
        B, H, N, HD = q.shape

        TILE_N = 32
        TILE_HD = _next_pow2(HD)

        dout = dout.contiguous()

        D = torch.zeros(B, H, N, device=q.device, dtype=torch.float32)
        grid = (B * H, triton.cdiv(N, TILE_N))

        _compute_D_v12[grid](
            dout, out, D,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            H=H, N=N, HD=HD,
        )

        blocks_n = triton.cdiv(N, TILE_N)
        dq = torch.zeros_like(q)
        dpb_buf = torch.zeros(B * H, blocks_n, J, device=q.device, dtype=torch.float32)
        dse_buf = torch.zeros(B * H, blocks_n, J * HD, device=q.device, dtype=torch.float32)

        _streaming_bwd_dq_v12[grid](
            q, k, v, pos_bias, scale_embed,
            dout, lse, D,
            dq, dpb_buf, dse_buf,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            blocks_n * J, J,
            blocks_n * J * HD, J * HD,
            H=H, N=N, HD=HD,
        )

        dpb = dpb_buf.view(B, H, blocks_n, J).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, J, HD).sum(dim=(0, 1, 2)).contiguous()

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        _streaming_bwd_dkdv_v12[grid](
            q, k, v, pos_bias, scale_embed,
            dout, lse, D,
            dk, dv,
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
            H=H, N=N, HD=HD, TILE_M=32, TILE_HD=64,
        )

        return dq, dk, dv, dpb, dse


def _sm90_forward(q, k, v, pos_bias, scale_embed):
    """
    sm_90 (H100/H200) forward path using gather + torch.einsum.

    Gather K/V into contiguous [B,H,N,J,HD] buffers, then use tensor-core-friendly
    torch.einsum for the attention computation.
    """
    B, H, N, HD = q.shape
    device = q.device
    dtype = q.dtype

    offsets = torch.tensor(ALL_OFFSETS, device=device, dtype=torch.long)

    positions = torch.arange(N, device=device).unsqueeze(1)
    gather_indices = positions - offsets.unsqueeze(0)
    valid_mask = gather_indices >= 0
    gather_indices_clamped = gather_indices.clamp(min=0)

    idx = gather_indices_clamped.view(1, 1, N, J, 1).expand(B, H, N, J, HD)

    K_g = k.unsqueeze(3).expand(B, H, N, J, HD).gather(2, idx)
    V_g = v.unsqueeze(3).expand(B, H, N, J, HD).gather(2, idx)

    K_g = K_g * valid_mask.view(1, 1, N, J, 1).to(dtype)
    V_g = V_g * valid_mask.view(1, 1, N, J, 1).to(dtype)

    scale = HD ** -0.5

    q_se = torch.einsum("bhnd,jd->bhnj", q.float(), scale_embed.float()) * scale

    scores = torch.einsum("bhnd,bhnjd->bhnj", q.float(), K_g.float()) * scale

    scores = scores + pos_bias.T[None, :, None, :].float()

    scores = scores + q_se

    causal_mask = ~valid_mask.view(1, 1, N, J)
    scores = scores.masked_fill(causal_mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    out = torch.einsum("bhnj,bhnjd->bhnd", weights, V_g.float())

    return out.to(dtype)


class _DSQGV12SM90Fn(torch.autograd.Function):
    """sm_90 (H100/H200) autograd function using gather + torch.einsum."""

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed):
        B, H, N, HD = q.shape
        device = q.device
        dtype = q.dtype

        offsets = torch.tensor(ALL_OFFSETS, device=device, dtype=torch.long)

        positions = torch.arange(N, device=device).unsqueeze(1)
        gather_indices = positions - offsets.unsqueeze(0)
        valid_mask = gather_indices >= 0
        gather_indices_clamped = gather_indices.clamp(min=0)

        idx = gather_indices_clamped.view(1, 1, N, J, 1).expand(B, H, N, J, HD)

        K_g = k.unsqueeze(3).expand(B, H, N, J, HD).gather(2, idx)
        V_g = v.unsqueeze(3).expand(B, H, N, J, HD).gather(2, idx)

        K_g = K_g * valid_mask.view(1, 1, N, J, 1).to(dtype)
        V_g = V_g * valid_mask.view(1, 1, N, J, 1).to(dtype)

        scale = HD ** -0.5

        q_se = torch.einsum("bhnd,jd->bhnj", q.float(), scale_embed.float()) * scale

        scores = torch.einsum("bhnd,bhnjd->bhnj", q.float(), K_g.float()) * scale
        scores = scores + pos_bias.T[None, :, None, :].float()
        scores = scores + q_se

        causal_mask = ~valid_mask.view(1, 1, N, J)
        scores = scores.masked_fill(causal_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        out = torch.einsum("bhnj,bhnjd->bhnd", weights, V_g.float())

        ctx.save_for_backward(
            q, k, v, pos_bias, scale_embed,
            K_g, V_g, weights, valid_mask, offsets
        )
        ctx.scale = scale

        return out.to(dtype)

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pos_bias, scale_embed,
         K_g, V_g, weights, valid_mask, offsets) = ctx.saved_tensors
        scale = ctx.scale
        B, H, N, HD = q.shape
        device = q.device
        dtype = q.dtype

        dout = dout.float()
        q = q.float()
        K_g = K_g.float()
        V_g = V_g.float()

        dV_g = torch.einsum("bhnj,bhnd->bhnjd", weights, dout)

        dweights = torch.einsum("bhnd,bhnjd->bhnj", dout, V_g)

        dsoftmax = weights * (dweights - (dweights * weights).sum(dim=-1, keepdim=True))

        dscores = dsoftmax

        dpb = dscores.sum(dim=(0, 2)).T.contiguous()

        dq_se = dscores
        dse = torch.einsum("bhnj,bhnd->jd", dq_se, q) * scale

        dq_from_scores = torch.einsum("bhnj,bhnjd->bhnd", dscores, K_g) * scale
        dq_from_se = torch.einsum("bhnj,jd->bhnd", dscores, scale_embed.float()) * scale
        dq = dq_from_scores + dq_from_se

        dK_g = torch.einsum("bhnj,bhnd->bhnjd", dscores, q) * scale

        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        for j in range(J):
            delta = offsets[j].item()
            for n in range(delta, N):
                kp = n - delta
                dk[:, :, kp, :] += dK_g[:, :, n, j, :]
                dv[:, :, kp, :] += dV_g[:, :, n, j, :]

        return (
            dq.to(dtype),
            dk.to(dtype),
            dv.to(dtype),
            dpb,
            dse,
        )


def dsqg_attention_v12(q, k, v, pos_bias, scale_embed, use_sm90_path):
    """
    DSQG V12 attention with dual-path selection.

    Args:
        q, k, v: [B, H, N, HD] bfloat16
        pos_bias: [J, H] float32
        scale_embed: [J, HD] float32
        use_sm90_path: bool, True for H100/H200 gather+matmul, False for 4090 streaming

    Returns:
        out: [B, H, N, HD] bfloat16
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    if use_sm90_path:
        out = _DSQGV12SM90Fn.apply(
            q, k, v,
            pos_bias.float(), scale_embed.float(),
        )
    else:
        out = _DSQGV12StreamingFn.apply(
            q, k, v,
            pos_bias.float(), scale_embed.float(),
        )

    return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)


class DSQGAttentionV12(nn.Module):
    """
    DSQG V12: Dual-path attention with GPU auto-detection.

    - sm_90 (H100/H200): gather K/V into contiguous buffers, then torch.einsum
    - sm_89 (4090): streaming fused Triton kernel with proper tiling

    J=24 relay-optimal offsets, same score computation as V8.
    NPCI rotation dropped (phase_gain effectively zero in trained checkpoints).

    Parameters (checkpoint-compatible with V8/V11):
      pos_bias      [24, H]    global frequency prior
      scale_embed   [24, HD]   Q-matched-filter (zero-init)
      if_gain       [H]        IF amplifier gain (1.0-init)
      phase_base    [10, H, 2] MOVT angles — UNUSED, kept for checkpoint compat
      phase_gain    [10, H, 2] QK-OVT gains — UNUSED, kept for checkpoint compat
      query_probes  [2, HD]    y precomputation probes — UNUSED, kept for checkpoint compat
      key_probes    [2, HD]    z precomputation probes — UNUSED, kept for checkpoint compat
      npci_theta_k  [H]        NPCI K rotation angle — UNUSED, kept for checkpoint compat
      npci_theta_v  [H]        NPCI V rotation angle — UNUSED, kept for checkpoint compat
      ema_alpha     scalar     EMA decay — UNUSED, kept for checkpoint compat
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets=None,
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

        assert HD >= 4, "HD must be >= 4 for checkpoint compatibility"

        offsets_list = offsets if offsets is not None else ALL_OFFSETS
        assert len(offsets_list) == J, f"Expected {J} offsets, got {len(offsets_list)}"

        self.register_buffer(
            'offset_tensor',
            torch.tensor(offsets_list, dtype=torch.int32),
            persistent=False
        )

        self.qkv_proj = nn.Linear(D, 3 * D, bias=True)
        self.out_proj = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas = torch.linspace(0.2, 2.0, H)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets_list],
            dtype=torch.float32
        )
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.full((J, HD), scale_embed_init))
        self.if_gain = nn.Parameter(torch.ones(H))

        self.phase_base = nn.Parameter(torch.zeros(J_LARGE, H, R_PLANES))
        self.phase_gain = nn.Parameter(torch.zeros(J_LARGE, H, R_PLANES))
        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.npci_theta_k = nn.Parameter(torch.zeros(H))
        self.npci_theta_v = nn.Parameter(torch.zeros(H))

        self.ema_alpha = nn.Parameter(torch.tensor(ema_init))

        self.dropout = nn.Dropout(dropout)

        sm_90, sm_89 = _detect_gpu_capability()
        self._use_sm90_path = sm_90 if sm_90 is not None else False
        self._detected_arch = "sm_90" if sm_90 else ("sm_89" if sm_89 else "unknown")

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        # kv_inject: inter-layer relay signal (k_delta, v_delta) from preIF block.
        # V12 dropped NPCI rotation (phase_gain≈0), so inject additively.
        # Critical: must apply BEFORE dsqg_attention_v12 or gradients are disconnected.
        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        out = dsqg_attention_v12(
            q, k, v,
            self.pos_bias,
            self.scale_embed,
            self._use_sm90_path
        )

        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def get_autotune_config(self):
        """Return the autotune configuration that was selected."""
        return {
            "arch": self._detected_arch,
            "use_sm90_path": self._use_sm90_path,
        }

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
            se = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            phb = self.phase_base.detach().cpu()
            phg = self.phase_gain.detach().cpu()
            qp = self.query_probes.detach().cpu()
            kp = self.key_probes.detach().cpu()
            thk = self.npci_theta_k.detach().cpu()
            thv = self.npci_theta_v.detach().cpu()

        plane_diff = (phb[:, :, 0] - phb[:, :, 1]).abs().mean().item()

        return {
            'pos_bias_abs_mean': pb.abs().mean().item(),
            'pos_bias_abs_max': pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean': se.abs().mean().item(),
            'scale_embed_abs_max': se.abs().max().item(),
            'if_gain': gain.tolist(),
            'phase_base_abs_mean': phb.abs().mean().item(),
            'phase_base_abs_max': phb.abs().max().item(),
            'phase_base_local_mean': 0.0,
            'phase_base_sparse_mean': phb.abs().mean().item(),
            'phase_base_by_head': phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_sparse_by_head': phb.abs().mean(dim=(0, 2)).tolist(),
            'phase_base_plane_diff': plane_diff,
            'phase_base_p0_sparse': phb[:, :, 0].mean(0).tolist(),
            'phase_base_p1_sparse': phb[:, :, 1].mean(0).tolist(),
            'phase_gain_abs_mean': phg.abs().mean().item(),
            'phase_gain_abs_max': phg.abs().max().item(),
            'phase_gain_sparse_mean': phg.abs().mean().item(),
            'phase_gain_by_head': phg.abs().mean(dim=(0, 2)).tolist(),
            'query_probe_norm': qp.norm(dim=1).tolist(),
            'key_probe_norm': kp.norm(dim=1).tolist(),
            'npci_theta_k': thk.tolist(),
            'npci_theta_v': thv.tolist(),
            'v12_arch': self._detected_arch,
            'v12_use_sm90': self._use_sm90_path,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("DSQG Attention V12 — Dual-path implementation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    sm_90, sm_89 = _detect_gpu_capability()
    print(f"GPU: sm_90={sm_90}, sm_89={sm_89}")

    B, H, N, HD = 2, 12, 512, 64
    D = H * HD

    model = DSQGAttentionV12(D, H, seq_len=N).to(device)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
    print(f"Autotune config: {model.get_autotune_config()}")

    print("\nBasic forward pass: PASS")
