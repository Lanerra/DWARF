"""
causal_ema_scan.py — memory-efficient causal EMA via Triton scans.

Forward recurrence:
    y[t] = α·x[t] + (1−α)·y[t−1],  y[−1] = 0

Backward:
    dx[t] via reverse scan
        s[t] = dy[t] + (1−α)·s[t+1]
        dx[t] = α·s[t]

    dα via forward sensitivity scan
        sens[t] = (x[t] − y[t−1]) + (1−α)·sens[t−1]
        dα = Σ_t dy[t]·sens[t]

This version replaces the original single-program serial sequence loops with a
chunked associative scan:
  1. independent per-time-block local scans;
  2. a short carry propagation over blocks;
  3. parallel per-block correction.

The dα path accumulates into one scalar with a Triton atomic, avoiding a
separate per-program partial tensor and PyTorch reduction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_D = 64  # D-dims per program; tuned for DWARF D/head multiples
BLOCK_T = 64  # sequence tokens per local scan block
DA_AUX_FIELDS = 7


def _ema_num_stages() -> int:
    if not torch.cuda.is_available():
        return 2
    cc = torch.cuda.get_device_capability()
    sm90_or_newer = (cc[0] == 9 and cc[1] == 0) or cc[0] > 9
    return 4 if sm90_or_newer else 2


@triton.jit
def _fwd_local(
    X, Y, ALPHA, BLOCK_END,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    sXb, sXn, sXd,
    sYb, sYn, sYd,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    tb = tl.program_id(1)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    state = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    n0 = tb * BLOCK_T_VAL

    Xb = X + b * sXb
    Yb = Y + b * sYb
    for t in range(BLOCK_T_VAL):
        n = n0 + t
        valid = (n < N) & d_mask
        x = tl.load(Xb + n * sXn + d_off * sXd, mask=valid, other=0.0).to(tl.float32)
        new_state = a * x + decay * state
        state = tl.where(n < N, new_state, state)
        tl.store(Yb + n * sYn + d_off * sYd, state, mask=valid)

    off = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    tl.store(BLOCK_END + off, state, mask=d_mask)


@triton.jit
def _fwd_carry(
    BLOCK_END, CARRY, ALPHA,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    carry = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)

    for tb in range(NUM_T_BLOCKS):
        base = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
        tl.store(CARRY + base, carry, mask=d_mask)
        local_end = tl.load(BLOCK_END + base, mask=d_mask, other=0.0).to(tl.float32)

        n0 = tb * BLOCK_T_VAL
        valid_len = tl.minimum(BLOCK_T_VAL, N - n0)
        pow_decay = tl.full((), 1.0, tl.float32)
        for t in range(BLOCK_T_VAL):
            pow_decay = tl.where(t < valid_len, pow_decay * decay, pow_decay)
        carry = local_end + pow_decay * carry


@triton.jit
def _fwd_apply_carry(
    Y, CARRY, ALPHA,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    sYb, sYn, sYd,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    tb = tl.program_id(1)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    base = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    carry = tl.load(CARRY + base, mask=d_mask, other=0.0).to(tl.float32)

    Yb = Y + b * sYb
    n0 = tb * BLOCK_T_VAL
    pow_decay = decay
    for t in range(BLOCK_T_VAL):
        n = n0 + t
        valid = (n < N) & d_mask
        y = tl.load(Yb + n * sYn + d_off * sYd, mask=valid, other=0.0).to(tl.float32)
        y = y + pow_decay * carry
        tl.store(Yb + n * sYn + d_off * sYd, y, mask=valid)
        pow_decay *= decay


@triton.jit
def _bwd_dx_local(
    DY, DX, ALPHA, BLOCK_START,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    sDYb, sDYn, sDYd,
    sDXb, sDXn, sDXd,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    tb = tl.program_id(1)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    state = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    n0 = tb * BLOCK_T_VAL

    DYb = DY + b * sDYb
    DXb = DX + b * sDXb
    for it in range(BLOCK_T_VAL):
        t = BLOCK_T_VAL - 1 - it
        n = n0 + t
        valid = (n < N) & d_mask
        dy = tl.load(DYb + n * sDYn + d_off * sDYd, mask=valid, other=0.0).to(tl.float32)
        new_state = dy + decay * state
        state = tl.where(n < N, new_state, state)
        tl.store(DXb + n * sDXn + d_off * sDXd, a * state, mask=valid)

    off = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    tl.store(BLOCK_START + off, state, mask=d_mask)


@triton.jit
def _bwd_dx_carry(
    BLOCK_START, AFTER, ALPHA,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    state_after = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)

    for ib in range(NUM_T_BLOCKS):
        tb = NUM_T_BLOCKS - 1 - ib
        base = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
        tl.store(AFTER + base, state_after, mask=d_mask)
        local_start = tl.load(BLOCK_START + base, mask=d_mask, other=0.0).to(tl.float32)

        n0 = tb * BLOCK_T_VAL
        valid_len = tl.minimum(BLOCK_T_VAL, N - n0)
        pow_decay = tl.full((), 1.0, tl.float32)
        for t in range(BLOCK_T_VAL):
            pow_decay = tl.where(t < valid_len, pow_decay * decay, pow_decay)
        state_after = local_start + pow_decay * state_after


@triton.jit
def _bwd_dx_apply_carry(
    DX, AFTER, ALPHA,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    sDXb, sDXn, sDXd,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    tb = tl.program_id(1)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    base = (pid * NUM_T_BLOCKS + tb) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    future = tl.load(AFTER + base, mask=d_mask, other=0.0).to(tl.float32)

    n0 = tb * BLOCK_T_VAL
    valid_len = tl.minimum(BLOCK_T_VAL, N - n0)
    DXb = DX + b * sDXb
    pow_tail = decay
    for it in range(BLOCK_T_VAL):
        t = BLOCK_T_VAL - 1 - it
        n = n0 + t
        active_t = t < valid_len
        valid = active_t & d_mask
        dx = tl.load(DXb + n * sDXn + d_off * sDXd, mask=valid, other=0.0).to(tl.float32)
        dx = dx + a * pow_tail * future
        tl.store(DXb + n * sDXn + d_off * sDXd, dx, mask=valid)
        pow_tail = tl.where(active_t, pow_tail * decay, pow_tail)


@triton.jit
def _bwd_da_local(
    X, DY, ALPHA, AUX,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    sXb, sXn, sXd,
    sDYb, sDYn, sDYd,
    BLOCK_T_VAL: tl.constexpr, BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    tb = tl.program_id(1)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    n0 = tb * BLOCK_T_VAL

    y_prev = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    sens = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    local_c = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    coef_sens_in = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)  # Σ dy * d^(r+1)
    coef_y_in = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)     # Σ dy * (r+1)d^r
    pow_r = tl.full((), 1.0, tl.float32)
    sens_y_coef = tl.full((), 0.0, tl.float32)

    Xb = X + b * sXb
    DYb = DY + b * sDYb
    for t in range(BLOCK_T_VAL):
        n = n0 + t
        valid = (n < N) & d_mask
        x = tl.load(Xb + n * sXn + d_off * sXd, mask=valid, other=0.0).to(tl.float32)
        dy = tl.load(DYb + n * sDYn + d_off * sDYd, mask=valid, other=0.0).to(tl.float32)

        new_sens = (x - y_prev) + decay * sens
        new_y = a * x + decay * y_prev
        sens = tl.where(n < N, new_sens, sens)
        y_prev = tl.where(n < N, new_y, y_prev)

        local_c += tl.where(valid, dy * sens, 0.0)
        coef_sens_in += tl.where(valid, dy * (pow_r * decay), 0.0)
        coef_y_in += tl.where(valid, dy * ((t + 1.0) * pow_r), 0.0)

        new_sens_y_coef = -pow_r + decay * sens_y_coef
        sens_y_coef = tl.where(n < N, new_sens_y_coef, sens_y_coef)
        pow_r = tl.where(n < N, pow_r * decay, pow_r)

    base = ((pid * NUM_T_BLOCKS + tb) * DA_AUX_FIELDS) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    tl.store(AUX + base + 0 * BLOCK_D_VAL, y_prev, mask=d_mask)
    tl.store(AUX + base + 1 * BLOCK_D_VAL, sens, mask=d_mask)
    tl.store(AUX + base + 2 * BLOCK_D_VAL, local_c, mask=d_mask)
    tl.store(AUX + base + 3 * BLOCK_D_VAL, coef_sens_in, mask=d_mask)
    tl.store(AUX + base + 4 * BLOCK_D_VAL, coef_y_in, mask=d_mask)
    tl.store(AUX + base + 5 * BLOCK_D_VAL, pow_r, mask=d_mask)
    tl.store(AUX + base + 6 * BLOCK_D_VAL, sens_y_coef, mask=d_mask)


@triton.jit
def _bwd_da_carry(
    AUX, DA,
    N: tl.constexpr, D: tl.constexpr, NUM_T_BLOCKS: tl.constexpr,
    BLOCK_D_VAL: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D_VAL)
    db = pid % n_dblk
    d_off = db * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
    d_mask = d_off < D

    y_in = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    sens_in = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)
    da_vec = tl.zeros([BLOCK_D_VAL], dtype=tl.float32)

    for tb in range(NUM_T_BLOCKS):
        base = ((pid * NUM_T_BLOCKS + tb) * DA_AUX_FIELDS) * BLOCK_D_VAL + tl.arange(0, BLOCK_D_VAL)
        y_end = tl.load(AUX + base + 0 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)
        sens_end = tl.load(AUX + base + 1 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)
        local_c = tl.load(AUX + base + 2 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)
        coef_sens_in = tl.load(AUX + base + 3 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)
        coef_y_in = tl.load(AUX + base + 4 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)
        decay_len = tl.load(AUX + base + 5 * BLOCK_D_VAL, mask=d_mask, other=1.0).to(tl.float32)
        sens_y_coef = tl.load(AUX + base + 6 * BLOCK_D_VAL, mask=d_mask, other=0.0).to(tl.float32)

        da_vec += local_c + sens_in * coef_sens_in - y_in * coef_y_in
        sens_in = sens_end + decay_len * sens_in + sens_y_coef * y_in
        y_in = y_end + decay_len * y_in

    tl.atomic_add(DA, tl.sum(tl.where(d_mask, da_vec, 0.0), axis=0), sem="relaxed")


class _Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, ema_factor: torch.Tensor, floor: float) -> torch.Tensor:
        x = x.contiguous()
        B, N, D = x.shape
        alpha = ema_factor.detach().clamp(float(floor), 0.5).to(
            device=x.device, dtype=torch.float32
        ).reshape(1)

        y = torch.empty_like(x)
        n_dblk = triton.cdiv(D, BLOCK_D)
        num_programs = B * n_dblk
        num_t_blocks = triton.cdiv(N, BLOCK_T)
        states = torch.empty((num_programs, num_t_blocks, BLOCK_D), device=x.device, dtype=torch.float32)
        carries = torch.empty_like(states)
        num_stages = _ema_num_stages()

        _fwd_local[(num_programs, num_t_blocks)](
            x, y, alpha, states, N, D, num_t_blocks,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=num_stages,
        )
        _fwd_carry[(num_programs,)](
            states, carries, alpha, N, D, num_t_blocks,
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=1,
        )
        _fwd_apply_carry[(num_programs, num_t_blocks)](
            y, carries, alpha, N, D, num_t_blocks,
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=num_stages,
        )

        ctx.save_for_backward(x, ema_factor, alpha)
        ctx.floor = float(floor)
        ctx.num_t_blocks = int(num_t_blocks)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, ema_factor, alpha = ctx.saved_tensors
        dy = dy.contiguous()
        B, N, D = dy.shape
        n_dblk = triton.cdiv(D, BLOCK_D)
        num_programs = B * n_dblk
        num_t_blocks = int(ctx.num_t_blocks)
        num_stages = _ema_num_stages()

        dx = torch.empty_like(dy)
        states = torch.empty((num_programs, num_t_blocks, BLOCK_D), device=dy.device, dtype=torch.float32)
        carries = torch.empty_like(states)

        _bwd_dx_local[(num_programs, num_t_blocks)](
            dy, dx, alpha, states, N, D, num_t_blocks,
            dy.stride(0), dy.stride(1), dy.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=num_stages,
        )
        _bwd_dx_carry[(num_programs,)](
            states, carries, alpha, N, D, num_t_blocks,
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=1,
        )
        _bwd_dx_apply_carry[(num_programs, num_t_blocks)](
            dx, carries, alpha, N, D, num_t_blocks,
            dx.stride(0), dx.stride(1), dx.stride(2),
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=num_stages,
        )

        aux = torch.empty(
            (num_programs, num_t_blocks, DA_AUX_FIELDS, BLOCK_D),
            device=dy.device,
            dtype=torch.float32,
        )
        da = torch.zeros_like(alpha)
        _bwd_da_local[(num_programs, num_t_blocks)](
            x, dy, alpha, aux, N, D, num_t_blocks,
            x.stride(0), x.stride(1), x.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            BLOCK_T_VAL=BLOCK_T, BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=num_stages,
        )
        _bwd_da_carry[(num_programs,)](
            aux, da, N, D, num_t_blocks,
            BLOCK_D_VAL=BLOCK_D, num_warps=4, num_stages=1,
        )
        da = da.reshape_as(ema_factor).to(dtype=ema_factor.dtype)

        boundary = (ema_factor <= ctx.floor) | (ema_factor >= 0.5)
        d_ema = torch.where(boundary, torch.zeros_like(ema_factor), da)
        return dx, d_ema, None


def _reference_ema_autograd(x: torch.Tensor, ema_factor: torch.Tensor, floor: float) -> torch.Tensor:
    """Small CPU/non-CUDA fallback; intentionally simple and differentiable."""
    a = ema_factor.clamp(float(floor), 0.5).to(dtype=torch.float32)
    decay = 1.0 - a
    state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32)
    ys = []
    xf = x.float()
    for t in range(x.shape[1]):
        state = a * xf[:, t, :] + decay * state
        ys.append(state)
    return torch.stack(ys, dim=1).to(dtype=x.dtype)


def causal_ema_scan(x: torch.Tensor, ema_factor: torch.Tensor, floor: float = 1e-5) -> torch.Tensor:
    """
    Drop-in replacement for _causal_ema() in DWARF training scripts.

    x:          [B, N, D] typically bf16
    ema_factor: scalar tensor / nn.Parameter (raw or transformed before call)
    floor:      minimum alpha value
    """
    if not x.is_cuda:
        return _reference_ema_autograd(x, ema_factor, floor)
    return _Fn.apply(x, ema_factor, floor)


if __name__ == "__main__":
    def _reference_ema(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        a = alpha.to(dtype=torch.float32)
        decay = 1.0 - a
        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=torch.float32)
        ys = []
        xf = x.float()
        for t in range(x.shape[1]):
            state = a * xf[:, t, :] + decay * state
            ys.append(state)
        return torch.stack(ys, dim=1).to(dtype=x.dtype)

    if not torch.cuda.is_available():
        print("CUDA is required for the Triton self-test.")
        raise SystemExit(0)

    torch.manual_seed(42)
    dev = "cuda"
    B, N, D = 4, 257, 130
    alpha_v = 0.05

    x = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    ef = torch.tensor(alpha_v, device=dev, dtype=torch.float32, requires_grad=True)

    ref = _reference_ema(x.detach(), ef.detach())
    out = causal_ema_scan(x, ef)
    err = (ref.float() - out.float()).abs()
    print(f"Forward  max_err={err.max():.4e}  mean_err={err.mean():.4e}")
    assert err.max() < 5e-3, "Forward mismatch too large"

    gout = torch.randn_like(out, dtype=torch.bfloat16)
    out.backward(gout)
    dx_triton = x.grad.detach().clone()
    da_triton = ef.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    ef_ref = ef.detach().clone().requires_grad_(True)
    ref_out = _reference_ema(x_ref, ef_ref)
    ref_out.backward(gout)
    dx_ref = x_ref.grad.detach()
    da_ref = ef_ref.grad.detach()

    dx_err = (dx_triton.float() - dx_ref.float()).abs()
    da_err = (da_triton.float() - da_ref.float()).abs()
    print(f"Backward dx max_err={dx_err.max():.4e} mean_err={dx_err.mean():.4e}")
    print(f"Backward dα max_err={da_err.max():.4e} value={da_triton.item():.4e}")
    assert dx_err.max() < 5e-3, "dx mismatch too large"
    assert da_err.max() < 5e-2, "dα mismatch too large"
    print("✓ forward + backward correct")
