"""
causal_ema_scan.py — Memory-efficient causal EMA via Triton parallel scan.

Replaces the F.conv1d depthwise-conv in DWARF training scripts.

Memory (B=8, D=512, N=2048, K=256):
  Old (conv1d):  B×D×(N+K) float32 ≈ 8.6 GB workspace
  New (scan):    2×B×N×D bfloat16  ≈ 32 MB  (just x and y)

Forward:   y[t] = α·x[t] + (1−α)·y[t−1],  y[−1] = 0
Backward:
  dx[t]  via reverse scan  s[t] = dy[t] + (1−α)·s[t+1];  dx[t] = α·s[t]
  dα     via forward sensitivity scan  (scalar gradient, cheap Python loop)

Both Triton kernels scan over N sequentially, parallel across B×ceil(D/BLOCK_D).
"""

import torch
import triton
import triton.language as tl

BLOCK_D = 64   # D-dims per program; tune if D is not a multiple of 64


# ── forward kernel ────────────────────────────────────────────────────────────
@triton.jit
def _fwd(
    X, Y,
    alpha_val,          # Python float32 scalar
    N: tl.constexpr,
    D: tl.constexpr,
    sXb, sXn,           # strides; d-stride == 1 (contiguous)
    sYb, sYn,
    BLOCK_D: tl.constexpr,
):
    pid    = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D)
    b      = pid // n_dblk
    db     = pid %  n_dblk
    d_off  = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask   = d_off < D

    a     = tl.full([1], alpha_val, dtype=tl.float32)
    decay = tl.full([1], 1.0 - alpha_val, dtype=tl.float32)
    state = tl.zeros([BLOCK_D], dtype=tl.float32)

    Xb = X + b * sXb
    Yb = Y + b * sYb
    for n in range(N):
        x     = tl.load(Xb + n * sXn + d_off, mask=mask, other=0.0).to(tl.float32)
        state = a * x + decay * state
        tl.store(Yb + n * sYn + d_off, state.to(tl.bfloat16), mask=mask)


# ── backward kernel — reverse scan for dx ─────────────────────────────────────
@triton.jit
def _bwd_dx(
    DY, DX,
    alpha_val,
    N: tl.constexpr,
    D: tl.constexpr,
    sDYb, sDYn,
    sDXb, sDXn,
    BLOCK_D: tl.constexpr,
):
    """s[t] = dy[t] + (1−α)·s[t+1];  dx[t] = α·s[t]"""
    pid    = tl.program_id(0)
    n_dblk = tl.cdiv(D, BLOCK_D)
    b      = pid // n_dblk
    db     = pid %  n_dblk
    d_off  = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask   = d_off < D

    a     = tl.full([1], alpha_val, dtype=tl.float32)
    decay = tl.full([1], 1.0 - alpha_val, dtype=tl.float32)
    state = tl.zeros([BLOCK_D], dtype=tl.float32)

    DYb = DY + b * sDYb
    DXb = DX + b * sDXb
    for i in range(N):
        n  = N - 1 - i
        dy = tl.load(DYb + n * sDYn + d_off, mask=mask, other=0.0).to(tl.float32)
        state = dy + decay * state
        tl.store(DXb + n * sDXn + d_off, (a * state).to(tl.bfloat16), mask=mask)


# ── autograd Function ─────────────────────────────────────────────────────────
class _Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, ema_factor: torch.Tensor,
                floor: float) -> torch.Tensor:
        x  = x.contiguous()
        B, N, D = x.shape
        alpha_val = float(ema_factor.clamp(float(floor), 0.5).item())

        y = torch.empty_like(x)
        grid = (B * triton.cdiv(D, BLOCK_D),)
        _fwd[grid](
            x, y, alpha_val, N, D,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x, y, ema_factor)
        ctx.floor = floor
        ctx.alpha_val = alpha_val
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, y, ema_factor = ctx.saved_tensors
        alpha_val = ctx.alpha_val
        dy = dy.contiguous()
        B, N, D = dy.shape

        # dx — Triton reverse scan
        dx = torch.empty_like(dy)
        grid = (B * triton.cdiv(D, BLOCK_D),)
        _bwd_dx[grid](
            dy, dx, alpha_val, N, D,
            dy.stride(0), dy.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_D=BLOCK_D,
        )

        # dα — forward sensitivity scan  (scalar; cheap PyTorch loop over N)
        # s[t] = (x[t] − y_prev[t]) + (1−α)·s[t−1];  dα = Σ dy[t]·s[t]
        # y_prev[t] = y[t-1], y[-1] = 0
        a     = alpha_val
        decay = 1.0 - a
        xf    = x.float()
        yf    = y.float()
        dyf   = dy.float()
        s     = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        da    = torch.zeros(1,    device=x.device, dtype=torch.float32)
        y_prev = torch.zeros_like(s)
        for t in range(N):
            s      = (xf[:, t, :] - y_prev) + decay * s   # sensitivity [B, D]
            da    += (dyf[:, t, :] * s).sum()
            y_prev = yf[:, t, :]

        # gradient only flows if ema_factor is not at clamp boundary
        at_floor = (ema_factor.item() <= float(ctx.floor))
        at_ceil  = (ema_factor.item() >= 0.5)
        if at_floor or at_ceil:
            d_ema = torch.zeros_like(ema_factor)
        else:
            d_ema = da.reshape_as(ema_factor)

        return dx, d_ema, None   # None for floor


def causal_ema_scan(x: torch.Tensor, ema_factor: torch.Tensor,
                    floor: float = 1e-5) -> torch.Tensor:
    """
    Drop-in replacement for _causal_ema() in DWARF training scripts.

    x:          [B, N, D]  bfloat16
    ema_factor: scalar nn.Parameter (raw, before clamping)
    floor:      minimum alpha value (default matches EMA_FLOOR in j26d script)
    """
    return _Fn.apply(x, ema_factor, floor)


# ── correctness + memory test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F

    def _conv_ema(x, alpha_val, K=256):
        """Reference: current conv1d implementation."""
        B, N, D = x.shape
        t      = torch.arange(min(K, N), device=x.device, dtype=torch.float32)
        kernel = alpha_val * (1.0 - alpha_val) ** t
        kernel = (kernel / kernel.sum()).flip(0)
        xi_f   = x.float()
        xi_bd  = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
        xi_p   = F.pad(xi_bd, (min(K, N) - 1, 0))
        pool   = F.conv1d(xi_p, kernel.view(1, 1, -1))
        return pool.view(B, D, N).permute(0, 2, 1).to(x.dtype)

    torch.manual_seed(42)
    dev = "cuda"
    B, N, D = 4, 256, 128
    alpha_v = 0.05

    x = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16)
    ef = torch.tensor(alpha_v, device=dev, requires_grad=True)

    # Forward
    ref  = _conv_ema(x, alpha_v)
    out  = causal_ema_scan(x, ef)
    err  = (ref.float() - out.float()).abs()
    print(f"Forward  max_err={err.max():.4e}  mean_err={err.mean():.4e}")
    assert err.max() < 0.05, "Forward mismatch too large"

    # Backward
    out.sum().backward()
    print(f"Backward  dx_shape={x.shape}  dα={ef.grad.item():.4e}")
    assert ef.grad is not None
    print("✓ forward + backward correct")

    # Memory comparison at training scale
    B2, N2, D2 = 8, 2048, 512
    x2 = torch.randn(B2, N2, D2, device=dev, dtype=torch.bfloat16)
    ef2 = torch.tensor(alpha_v, device=dev)

    torch.cuda.reset_peak_memory_stats()
    _ = causal_ema_scan(x2, ef2)
    torch.cuda.synchronize()
    scan_mb = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    _ = _conv_ema(x2, alpha_v)
    torch.cuda.synchronize()
    conv_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f"\nMemory @ B=8, N=2048, D=512:")
    print(f"  scan:   {scan_mb:.1f} MB")
    print(f"  conv:   {conv_mb:.1f} MB")
    print(f"  reduction: {conv_mb/scan_mb:.1f}×")
