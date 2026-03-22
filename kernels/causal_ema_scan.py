"""
causal_ema_scan.py — Memory-efficient causal EMA via Triton parallel scan.

Forward recurrence:
    y[t] = α·x[t] + (1−α)·y[t−1],  y[−1] = 0

Backward:
    dx[t]  via reverse scan
        s[t] = dy[t] + (1−α)·s[t+1]
        dx[t] = α·s[t]

    dα     via forward sensitivity scan, fully on GPU in Triton
        sens[t] = (x[t] − y[t−1]) + (1−α)·sens[t−1]
        dα = Σ_t dy[t]·sens[t]

This version avoids the old Python-side O(N) loop in backward and also avoids
scalar `.item()` syncs for alpha during forward/backward. The only host-side
work left in backward is summing one float32 partial per Triton program.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_D = 64   # D-dims per program; tuned for HD/D multiples common in DWARF


# ── forward kernel ────────────────────────────────────────────────────────────
@triton.jit
def _fwd(
    X, Y, ALPHA,
    N: tl.constexpr,
    D: tl.constexpr,
    sXb, sXn, sXd,
    sYb, sYn, sYd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    state = tl.zeros([BLOCK_D], dtype=tl.float32)

    Xb = X + b * sXb
    Yb = Y + b * sYb
    for n in range(N):
        x = tl.load(Xb + n * sXn + d_off * sXd, mask=mask, other=0.0).to(tl.float32)
        state = a * x + decay * state
        tl.store(Yb + n * sYn + d_off * sYd, state, mask=mask)


# ── backward kernel — reverse scan for dx ─────────────────────────────────────
@triton.jit
def _bwd_dx(
    DY, DX, ALPHA,
    N: tl.constexpr,
    D: tl.constexpr,
    sDYb, sDYn, sDYd,
    sDXb, sDXn, sDXd,
    BLOCK_D: tl.constexpr,
):
    """s[t] = dy[t] + (1−α)·s[t+1];  dx[t] = α·s[t]"""
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a
    state = tl.zeros([BLOCK_D], dtype=tl.float32)

    DYb = DY + b * sDYb
    DXb = DX + b * sDXb
    for i in range(N):
        n = N - 1 - i
        dy = tl.load(DYb + n * sDYn + d_off * sDYd, mask=mask, other=0.0).to(tl.float32)
        state = dy + decay * state
        tl.store(DXb + n * sDXn + d_off * sDXd, a * state, mask=mask)


# ── backward kernel — forward sensitivity scan for dα partials ─────────────────
@triton.jit
def _bwd_da_partial(
    X, Y, DY, ALPHA, DA_PARTIAL,
    N: tl.constexpr,
    D: tl.constexpr,
    sXb, sXn, sXd,
    sYb, sYn, sYd,
    sDYb, sDYn, sDYd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D)
    b = pid // n_dblk
    db = pid % n_dblk
    d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    a = tl.load(ALPHA).to(tl.float32)
    decay = 1.0 - a

    sens = tl.zeros([BLOCK_D], dtype=tl.float32)
    y_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    Xb = X + b * sXb
    Yb = Y + b * sYb
    DYb = DY + b * sDYb

    for n in range(N):
        x = tl.load(Xb + n * sXn + d_off * sXd, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Yb + n * sYn + d_off * sYd, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DYb + n * sDYn + d_off * sDYd, mask=mask, other=0.0).to(tl.float32)
        sens = (x - y_prev) + decay * sens
        acc += dy * sens
        y_prev = y

    tl.store(DA_PARTIAL + pid, tl.sum(acc, axis=0))


# ── autograd Function ─────────────────────────────────────────────────────────
class _Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, ema_factor: torch.Tensor,
                floor: float) -> torch.Tensor:
        x = x.contiguous()
        B, N, D = x.shape
        alpha = ema_factor.detach().clamp(float(floor), 0.5).to(device=x.device, dtype=torch.float32).reshape(1)

        y = torch.empty_like(x)
        grid = (B * triton.cdiv(D, BLOCK_D),)
        _fwd[grid](
            x, y, alpha, N, D,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x, y, ema_factor, alpha)
        ctx.floor = float(floor)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, y, ema_factor, alpha = ctx.saved_tensors
        dy = dy.contiguous()
        B, N, D = dy.shape
        num_programs = B * triton.cdiv(D, BLOCK_D)

        dx = torch.empty_like(dy)
        grid = (num_programs,)
        _bwd_dx[grid](
            dy, dx, alpha, N, D,
            dy.stride(0), dy.stride(1), dy.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            BLOCK_D=BLOCK_D,
        )

        da_partial = torch.empty(num_programs, device=x.device, dtype=torch.float32)
        _bwd_da_partial[grid](
            x, y, dy, alpha, da_partial, N, D,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            BLOCK_D=BLOCK_D,
        )
        da = da_partial.sum().reshape_as(ema_factor).to(dtype=ema_factor.dtype)

        boundary = (ema_factor <= ctx.floor) | (ema_factor >= 0.5)
        d_ema = torch.where(boundary, torch.zeros_like(ema_factor), da)
        return dx, d_ema, None


def causal_ema_scan(x: torch.Tensor, ema_factor: torch.Tensor,
                    floor: float = 1e-5) -> torch.Tensor:
    """
    Drop-in replacement for _causal_ema() in DWARF training scripts.

    x:          [B, N, D]  typically bf16
    ema_factor: scalar tensor / nn.Parameter (raw or transformed before call)
    floor:      minimum alpha value
    """
    return _Fn.apply(x, ema_factor, floor)


# ── exact reference + basic correctness / memory test ─────────────────────────
if __name__ == "__main__":
    def _reference_ema(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        a = alpha.to(dtype=torch.float32)
        decay = 1.0 - a
        state = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        ys = []
        xf = x.float()
        for t in range(N):
            state = a * xf[:, t, :] + decay * state
            ys.append(state)
        return torch.stack(ys, dim=1).to(dtype=x.dtype)

    torch.manual_seed(42)
    dev = "cuda"
    B, N, D = 4, 256, 128
    alpha_v = 0.05

    x = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    ef = torch.tensor(alpha_v, device=dev, dtype=torch.float32, requires_grad=True)

    # Forward
    ref = _reference_ema(x.detach(), ef.detach())
    out = causal_ema_scan(x, ef)
    err = (ref.float() - out.float()).abs()
    print(f"Forward  max_err={err.max():.4e}  mean_err={err.mean():.4e}")
    assert err.max() < 5e-3, "Forward mismatch too large"

    # Backward against exact reference implementation
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

    # Memory comparison at training scale
    B2, N2, D2 = 8, 2048, 512
    x2 = torch.randn(B2, N2, D2, device=dev, dtype=torch.bfloat16)
    ef2 = torch.tensor(alpha_v, device=dev, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    _ = causal_ema_scan(x2, ef2)
    torch.cuda.synchronize()
    scan_mb = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    _ = _reference_ema(x2, ef2)
    torch.cuda.synchronize()
    ref_mb = torch.cuda.max_memory_allocated() / 1e6

    print("\nMemory @ B=8, N=2048, D=512:")
    print(f"  scan:   {scan_mb:.1f} MB")
    print(f"  ref:    {ref_mb:.1f} MB")
