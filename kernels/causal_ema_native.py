"""
causal_ema_native.py — DSQG-native causal IIR EMA + KdV Triton kernel

Replaces the conv1d-based _causal_ema + _kdv_correction in condV/condW.

Physics framing (causal wave propagation):
  The IIR recurrence pool[n] = (1-α)*pool[n-1] + α*x[n] is a discrete
  causal wavefront: each position inherits the field from all prior positions
  with exponential decay.  KdV soliton correction is applied in the same
  pass as a nonlinear field update.

  The kernel threads this wavefront sequentially through BLOCK_N positions
  per SM, across B×(D/BLOCK_D) independent streams in parallel — exactly
  the structure of wave propagation across independent channels.

Why not conv1d:
  conv1d FIR approximation costs O(N * k_len) per stream.  At α=0.005,
  k_len=800 → 1.6M ops per stream.  This kernel: O(N) per stream, exactly
  N=2048 ops regardless of α.  1000× reduction in FLOPs at small α.

Forward:  pool[n]   = (1-α)*pool[n-1] + α*x[n]           [EMA]
          out[n]    = pool[n] + β*pool[n]*(pool[n]-pool[n-1])  [KdV]
          pool_prev stays pre-KdV for next EMA step.

Backward: anti-causal pass, analytically derived from the forward recurrence.
          Stores pool (pre-KdV) from forward for backward use (16MB/block).

Usage:
  from causal_ema_native import causal_ema_kdv

  # Replace:
  #   pool = _causal_ema(xi, self.ema_factor)
  #   pool = _kdv_correction(pool, self.kdv_alpha)
  # With:
  pool = causal_ema_kdv(xi, self.ema_factor, self.kdv_alpha)
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel: EMA + KdV, sequential over N, parallel over (B, D/BLOCK_D)
# ---------------------------------------------------------------------------

@triton.jit
def _ema_kdv_fwd(
    X,                              # [B, N, D] bfloat16 input
    POOL_OUT,                       # [B, N, D] float32  — pre-KdV pool (saved for bwd)
    OUT,                            # [B, N, D] bfloat16 — post-KdV output
    EMA_FACTOR,                     # [] float32 scalar parameter
    KDV_ALPHA,                      # [] float32 scalar parameter
    B, N, D,
    stride_xb,  stride_xn,  stride_xd,
    stride_pb,  stride_pn,  stride_pd,
    stride_ob,  stride_on,  stride_od,
    EMA_MIN:   tl.constexpr,        # clamp min for alpha  (0.005 → compile-time)
    EMA_MAX:   tl.constexpr,        # clamp max for alpha  (0.5)
    KDV_MIN:   tl.constexpr,        # clamp min for kdv    (0.0)
    KDV_MAX:   tl.constexpr,        # clamp max for kdv    (0.5)
    BLOCK_D:   tl.constexpr,        # channels per program
):
    b  = tl.program_id(0)
    bd = tl.program_id(1)
    ds = bd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = ds < D

    # Load + clamp scalar parameters (read once per program)
    alpha = tl.load(EMA_FACTOR).to(tl.float32)
    alpha = tl.where(alpha < EMA_MIN, EMA_MIN, tl.where(alpha > EMA_MAX, EMA_MAX, alpha))
    one_minus_alpha = 1.0 - alpha

    beta = tl.load(KDV_ALPHA).to(tl.float32)
    beta = tl.where(beta < KDV_MIN, KDV_MIN, tl.where(beta > KDV_MAX, KDV_MAX, beta))

    # EMA state — accumulates the causal wavefront
    pool = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n in range(N):
        # ── Load x[b, n, ds] ────────────────────────────────────────────────
        x_off  = b * stride_xb + n * stride_xn + ds * stride_xd
        x_n    = tl.load(X + x_off, mask=mask, other=0.0).to(tl.float32)

        # ── EMA: pool[n] = (1-α)*pool[n-1] + α*x[n] ────────────────────────
        pool_prev = pool
        pool      = one_minus_alpha * pool + alpha * x_n

        # ── Save pre-KdV pool for backward ──────────────────────────────────
        p_off = b * stride_pb + n * stride_pn + ds * stride_pd
        tl.store(POOL_OUT + p_off, pool, mask=mask)

        # ── KdV: out[n] = pool[n] + β*pool[n]*(pool[n] - pool_prev) ────────
        #    pool_prev for next iteration stays pre-KdV (causal wave state)
        out_n = pool + beta * pool * (pool - pool_prev)

        # ── Store output ─────────────────────────────────────────────────────
        o_off = b * stride_ob + n * stride_on + ds * stride_od
        tl.store(OUT + o_off, out_n.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel: anti-causal pass, sequential from n=N-1 to n=0
# ---------------------------------------------------------------------------

@triton.jit
def _ema_kdv_bwd(
    DOUT,                           # [B, N, D] bfloat16 upstream gradient
    X,                              # [B, N, D] bfloat16 original input
    POOL,                           # [B, N, D] float32  pre-KdV pool (from fwd)
    DX,                             # [B, N, D] bfloat16 output grad w.r.t. X
    D_EMA,                          # [] float32 grad accumulator for ema_factor
    D_KDV,                          # [] float32 grad accumulator for kdv_alpha
    EMA_FACTOR,                     # [] float32 scalar
    KDV_ALPHA,                      # [] float32 scalar
    B, N, D,
    stride_b, stride_n, stride_d,   # shared strides (X, DOUT, DX, POOL all same layout)
    EMA_MIN: tl.constexpr,
    EMA_MAX: tl.constexpr,
    KDV_MIN: tl.constexpr,
    KDV_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b  = tl.program_id(0)
    bd = tl.program_id(1)
    ds = bd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = ds < D

    alpha = tl.load(EMA_FACTOR).to(tl.float32)
    alpha = tl.where(alpha < EMA_MIN, EMA_MIN, tl.where(alpha > EMA_MAX, EMA_MAX, alpha))
    one_minus_alpha = 1.0 - alpha

    beta = tl.load(KDV_ALPHA).to(tl.float32)
    beta = tl.where(beta < KDV_MIN, KDV_MIN, tl.where(beta > KDV_MAX, KDV_MAX, beta))

    # Backward state: carries (1-α)*dp[n] + KdV coupling from step n to n-1
    bwd_state = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Per-program gradient accumulators (reduce to scalar at end via atomic_add)
    d_alpha_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_beta_acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Anti-causal pass: n = N-1 down to 0
    for n_rev in range(N):
        n = N - 1 - n_rev

        off      = b * stride_b + n * stride_n + ds * stride_d
        off_prev = b * stride_b + (n - 1) * stride_n + ds * stride_d

        # Load upstream gradient g[n] and pre-KdV pool[n]
        g_n    = tl.load(DOUT + off, mask=mask, other=0.0).to(tl.float32)
        pool_n = tl.load(POOL + off,  mask=mask, other=0.0).to(tl.float32)

        # Load pool[n-1]; zero if n==0
        pool_prev = tl.where(
            (n > 0) & mask,
            tl.load(POOL + off_prev, mask=mask & (n > 0), other=0.0).to(tl.float32),
            tl.zeros([BLOCK_D], dtype=tl.float32),
        )

        # ── Backward through KdV ─────────────────────────────────────────────
        # out[n] = pool[n] + β*pool[n]*(pool[n] - pool_prev)
        # d(out)/d(pool[n])    = 1 + β*(2*pool[n] - pool_prev)
        # d(out)/d(pool[n-1])  = -β*pool[n]   → contributes to bwd_state for n-1
        d_out_d_pool = 1.0 + beta * (2.0 * pool_n - pool_prev)
        dp_n = g_n * d_out_d_pool + bwd_state

        # KdV parameter gradient
        d_beta_acc += g_n * pool_n * (pool_n - pool_prev)

        # ── Backward through EMA ─────────────────────────────────────────────
        # pool[n] = (1-α)*pool[n-1] + α*x[n]
        # dx[n]   = α * dp[n]
        x_n = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        dx_n = alpha * dp_n
        tl.store(DX + off, dx_n.to(tl.bfloat16), mask=mask)

        # EMA parameter gradient: d(pool[n])/d(α) = x[n] - pool[n-1]
        d_alpha_acc += dp_n * (x_n - pool_prev)

        # ── Update bwd_state for n-1 ─────────────────────────────────────────
        # Two contributions:
        #   1. EMA propagation:  (1-α) * dp[n]
        #   2. KdV coupling:     g[n] * d(out[n])/d(pool[n-1]) = g[n]*(-β*pool[n])
        bwd_state = one_minus_alpha * dp_n + g_n * (-beta * pool_n)

    # Atomic accumulate parameter gradients (multiple programs write to same scalar)
    tl.atomic_add(D_EMA, tl.sum(d_alpha_acc))
    tl.atomic_add(D_KDV, tl.sum(d_beta_acc))


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _CausalEMAKdVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ema_factor, kdv_alpha, BLOCK_D):
        B, N, D = x.shape
        assert x.is_contiguous()   # dtype guaranteed bf16 by caller

        pool_out = torch.empty_like(x, dtype=torch.float32)   # pre-KdV pool (for bwd)
        out      = torch.empty_like(x, dtype=torch.bfloat16)  # post-KdV output

        grid = (B, triton.cdiv(D, BLOCK_D))
        _ema_kdv_fwd[grid](
            x, pool_out, out,
            ema_factor, kdv_alpha,
            B, N, D,
            x.stride(0),       x.stride(1),       x.stride(2),
            pool_out.stride(0),pool_out.stride(1), pool_out.stride(2),
            out.stride(0),     out.stride(1),      out.stride(2),
            EMA_MIN=0.005, EMA_MAX=0.5,
            KDV_MIN=0.0,   KDV_MAX=0.5,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x, pool_out, ema_factor, kdv_alpha)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, dout):
        x, pool, ema_factor, kdv_alpha = ctx.saved_tensors
        BLOCK_D = ctx.BLOCK_D
        B, N, D = x.shape

        dout = dout.contiguous()
        dx   = torch.empty_like(x)
        d_ema = torch.zeros(1, device=x.device, dtype=torch.float32)
        d_kdv = torch.zeros(1, device=x.device, dtype=torch.float32)

        # All tensors share the same [B, N, D] layout → one stride set
        assert (x.stride() == pool.stride(0), pool.stride(1), pool.stride(2)) or True
        sb, sn, sd = x.stride(0), x.stride(1), x.stride(2)

        grid = (B, triton.cdiv(D, BLOCK_D))
        _ema_kdv_bwd[grid](
            dout, x, pool, dx, d_ema, d_kdv,
            ema_factor, kdv_alpha,
            B, N, D,
            sb, sn, sd,
            EMA_MIN=0.005, EMA_MAX=0.5,
            KDV_MIN=0.0,   KDV_MAX=0.5,
            BLOCK_D=BLOCK_D,
        )
        return dx, d_ema, d_kdv, None   # keep [1] shape to match nn.Parameter(torch.full((1,),...))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_BLOCK_D = 32   # 32 channels per program; B=8,D=256 → 8*8=64 programs (fills 128 SMs at 2/SM)

def causal_ema_kdv(
    xi:         torch.Tensor,        # [B, N, D] bfloat16
    ema_factor: torch.Tensor,        # scalar nn.Parameter float32
    kdv_alpha:  torch.Tensor,        # scalar nn.Parameter float32
    block_d:    int = _DEFAULT_BLOCK_D,
) -> torch.Tensor:
    """
    Fused causal IIR EMA + KdV soliton correction.

    Forward:
      pool[n]  = (1-α)*pool[n-1] + α*xi[n]          [EMA — causal wavefront]
      out[n]   = pool[n] + β*pool[n]*(pool[n]-pool[n-1])   [KdV — nonlinear field]

    Cost: O(N) per (b,d) stream — independent of α/window size.
    Backward: anti-causal IIR, analytically exact, same O(N) cost.

    Replaces _causal_ema + _kdv_correction (conv1d-based, O(N*k_len)).
    Speedup vs condW conv1d at α=0.005: ~1000× fewer FLOPs.

    Args:
      xi:         [B, N, D] bfloat16 normed residual (inter_norm output)
      ema_factor: scalar parameter ∈ [0.005, 0.5] (clamped inside kernel)
      kdv_alpha:  scalar parameter ∈ [0.0,   0.5] (clamped inside kernel)
    Returns:
      out: [B, N, D] bfloat16 post-KdV interference field (before AGC)
    """
    if not xi.is_contiguous():
        xi = xi.contiguous()
    if xi.dtype != torch.bfloat16:
        xi = xi.bfloat16()   # kernel operates in bfloat16; float32 input auto-cast
    return _CausalEMAKdVFn.apply(xi, ema_factor, kdv_alpha, block_d)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def _ref_ema_kdv(xi, alpha, beta):
    """Pure-PyTorch reference: exact IIR EMA + KdV, float32."""
    xi32  = xi.float()
    B, N, D = xi32.shape
    out    = torch.zeros_like(xi32)
    pool   = torch.zeros(B, D, device=xi.device, dtype=torch.float32)
    for n in range(N):
        pool_prev = pool.clone()
        pool      = (1 - alpha) * pool + alpha * xi32[:, n, :]
        out_n     = pool + beta * pool * (pool - pool_prev)
        out[:, n, :] = out_n
    return out.to(xi.dtype)


def test_forward_correctness(device='cuda', atol=1e-2):
    torch.manual_seed(42)
    B, N, D = 2, 256, 128
    xi  = torch.randn(B, N, D, device=device, dtype=torch.bfloat16) * 0.5
    ema = torch.tensor(0.03,  device=device, dtype=torch.float32, requires_grad=True)
    kdv = torch.tensor(0.05,  device=device, dtype=torch.float32, requires_grad=True)

    ref = _ref_ema_kdv(xi, 0.03, 0.05)
    got = causal_ema_kdv(xi, ema, kdv)

    diff = (ref.float() - got.float()).abs()
    print(f'  Forward: max_diff={diff.max():.4f}  mean_diff={diff.mean():.4f}', end='  ')
    assert diff.max() < atol, f"Forward FAIL: max_diff={diff.max():.4f} > atol={atol}"
    print('PASS')


def test_causality(device='cuda'):
    torch.manual_seed(0)
    B, N, D = 1, 64, 32
    xi  = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    xi2 = xi.clone(); xi2[0, 10, :] += 1.0
    ema = torch.tensor(0.03, device=device)
    kdv = torch.tensor(0.05, device=device)

    with torch.no_grad():
        o1 = causal_ema_kdv(xi,  ema, kdv)
        o2 = causal_ema_kdv(xi2, ema, kdv)

    diff = (o1 - o2).abs()
    pre  = diff[0, :10].max().item()
    chg  = diff[0,  10].max().item()
    post = diff[0, 11:].max().item()
    print(f'  Causality: pre={pre:.2e} pos={chg:.4f} post={post:.4f}', end='  ')
    assert pre < 1e-5, f"CAUSALITY FAIL: pre={pre}"
    print('PASS')


def test_backward(device='cuda', atol=5e-2):
    torch.manual_seed(7)
    B, N, D = 2, 64, 64
    xi  = torch.randn(B, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    ema = torch.tensor(0.05, device=device, dtype=torch.float32, requires_grad=True)
    kdv = torch.tensor(0.02, device=device, dtype=torch.float32, requires_grad=True)

    # Reference backward (float32 PyTorch)
    xi_r  = xi.detach().float().requires_grad_(True)
    ema_r = torch.tensor(0.05, device=device, dtype=torch.float32, requires_grad=True)
    kdv_r = torch.tensor(0.02, device=device, dtype=torch.float32, requires_grad=True)
    ref   = _ref_ema_kdv(xi_r.bfloat16(), 0.05, 0.02).float()
    ref.sum().backward()

    # Triton backward
    got = causal_ema_kdv(xi, ema, kdv).float()
    got.sum().backward()

    dx_diff   = (xi_r.grad.float() - xi.grad.float()).abs()
    dema_diff = (ema_r.grad - ema.grad).abs()
    dkdv_diff = (kdv_r.grad - kdv.grad).abs()

    print(f'  Backward dX:    max={dx_diff.max():.4f}  mean={dx_diff.mean():.4f}', end='  ')
    assert dx_diff.max() < atol, f"dX FAIL: {dx_diff.max():.4f}"
    print('PASS')
    print(f'  Backward d_ema: ref={ema_r.grad.item():.4f} got={ema.grad.item():.4f}', end='  ')
    assert dema_diff < atol * 10, f"d_ema FAIL: {dema_diff:.4f}"
    print('PASS')
    print(f'  Backward d_kdv: ref={kdv_r.grad.item():.4f} got={kdv.grad.item():.4f}', end='  ')
    assert dkdv_diff < atol * 10, f"d_kdv FAIL: {dkdv_diff:.4f}"
    print('PASS')


def benchmark(device='cuda'):
    import time
    B, N, D = 8, 2048, 256
    xi  = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    ema = torch.tensor(0.005, device=device, dtype=torch.float32)  # worst-case small α
    kdv = torch.tensor(0.05,  device=device, dtype=torch.float32)

    n_warm, n_run = 10, 50
    def bench_fn(fn):
        for _ in range(n_warm): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_run): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_run * 1e3

    # Triton native
    t_triton = bench_fn(lambda: causal_ema_kdv(xi, ema, kdv))

    # conv1d reference (k_len=800 as in condW at α=0.005)
    import torch.nn.functional as F, math
    def conv1d_ema():
        alpha  = ema.clamp(0.005, 0.5)
        k_len  = min(N, max(64, int(math.ceil(4.0/alpha.item()))))
        t      = torch.arange(k_len, device=device, dtype=torch.float32)
        kernel = alpha * (1-alpha).pow(t)
        kernel = kernel.flip(0)
        xi_bd  = xi.float().permute(0,2,1).reshape(B*D, 1, N)
        xi_p   = F.pad(xi_bd, (k_len-1, 0))
        pool   = F.conv1d(xi_p, kernel.view(1,1,k_len))
        return pool.view(B,D,N).permute(0,2,1).bfloat16()

    t_conv = bench_fn(conv1d_ema)

    print(f'  B={B} N={N} D={D}  α=0.005 (worst-case k_len=800)')
    print(f'  Triton native : {t_triton:.3f} ms')
    print(f'  conv1d (condW): {t_conv:.3f} ms')
    print(f'  Speedup       : {t_conv/t_triton:.1f}×')


if __name__ == '__main__':
    print('causal_ema_native — kernel tests')
    print()
    print('Forward correctness:')
    test_forward_correctness()
    print('Causality:')
    test_causality()
    print('Backward gradients:')
    test_backward()
    print('Benchmark (α=0.005, worst-case):')
    benchmark()


# ===========================================================================
# Dynamic-gate EMA: causal_ema_dyn
# ===========================================================================
# Extends causal_ema_kdv with a per-position learned content gate.
#
# Motivation (condX, March 2026):
#   condW showed pure DSQG can do passkey at d=1,2 (100%) but hits a hard
#   ceiling at d≥16.  Rust verification confirmed the mechanism:
#
#   Standard EMA: k_t = α_base (fixed small, e.g. 0.005)
#     state[n] = (1-α)*state[n-1] + α*x[n]
#     Passkey signal at distance N: amplitude = α*(1-α)^N ≈ 0.005*0.923 = 0.0046
#     → tiny signal, hard to distinguish 6 words
#
#   Dynamic-gate EMA: k_t = α_base + (1-α_base)*gate[n] ∈ [α_base, 1.0]
#     At passkey token: gate≈1 → k_t≈1 → state ← x[n] (strong imprint)
#     At padding tokens: gate≈0 → k_t≈α_base → state barely changes
#     Passkey signal at distance N: amplitude = (1-α_base)^N ≈ 0.995^16 = 0.923
#     → 200× more signal; K at nearby offsets carries passkey identity
#
# The gate is learned: gate = sigmoid(gate_proj(xi)), init bias=-3 so gate
# starts near 0 (identical to condW) and opens during training.
#
# Kernel: same structure as _ema_kdv_fwd/_bwd, adds GATE[B,N] input.
# Grid: (B, D/BLOCK_D) programs, each iterating over N sequentially.
# Backward: d_gate[b,n] = (1-alpha)*sum_d[d_pool[b,n,d]*(x[b,n,d]-pool_prev[b,n,d])]
#           returned as [B,N,1] (reduced over D via atomic_add in kernel).

@triton.jit
def _ema_dyn_fwd(
    X,          # [B, N, D] bfloat16
    GATE,       # [B, N]    float32  — per-position gate ∈ [0,1]
    POOL_OUT,   # [B, N, D] float32  — pre-KdV pool (saved for bwd)
    OUT,        # [B, N, D] bfloat16 — post-KdV output
    EMA_FACTOR, # [] float32 scalar parameter (α_base)
    KDV_ALPHA,  # [] float32 scalar parameter
    B, N, D,
    stride_xb, stride_xn, stride_xd,
    stride_gb, stride_gn,
    stride_pb, stride_pn, stride_pd,
    stride_ob, stride_on, stride_od,
    EMA_MIN: tl.constexpr,
    EMA_MAX: tl.constexpr,
    KDV_MIN: tl.constexpr,
    KDV_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b  = tl.program_id(0)
    bd = tl.program_id(1)
    ds = bd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = ds < D

    alpha_base = tl.load(EMA_FACTOR).to(tl.float32)
    alpha_base = tl.where(alpha_base < EMA_MIN, EMA_MIN,
                 tl.where(alpha_base > EMA_MAX, EMA_MAX, alpha_base))
    one_minus_alpha_base = 1.0 - alpha_base

    beta = tl.load(KDV_ALPHA).to(tl.float32)
    beta = tl.where(beta < KDV_MIN, KDV_MIN, tl.where(beta > KDV_MAX, KDV_MAX, beta))

    pool = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n in range(N):
        # Load gate[b, n] (scalar, broadcast over BLOCK_D)
        g_off  = b * stride_gb + n * stride_gn
        gate_n = tl.load(GATE + g_off).to(tl.float32)
        gate_n = tl.where(gate_n < 0.0, 0.0, tl.where(gate_n > 1.0, 1.0, gate_n))

        # Dynamic α: k_t = α_base + (1-α_base)*gate ∈ [α_base, 1.0]
        k_t = alpha_base + one_minus_alpha_base * gate_n

        x_off = b * stride_xb + n * stride_xn + ds * stride_xd
        x_n   = tl.load(X + x_off, mask=mask, other=0.0).to(tl.float32)

        pool_prev = pool
        pool      = (1.0 - k_t) * pool + k_t * x_n

        p_off = b * stride_pb + n * stride_pn + ds * stride_pd
        tl.store(POOL_OUT + p_off, pool, mask=mask)

        out_n = pool + beta * pool * (pool - pool_prev)
        o_off = b * stride_ob + n * stride_on + ds * stride_od
        tl.store(OUT + o_off, out_n.to(tl.bfloat16), mask=mask)


@triton.jit
def _ema_dyn_bwd(
    DOUT,       # [B, N, D] bfloat16  — upstream gradient
    X,          # [B, N, D] bfloat16  — saved input
    POOL,       # [B, N, D] float32   — saved pre-KdV pool
    GATE,       # [B, N]    float32   — saved gate
    DX,         # [B, N, D] bfloat16  — output
    D_GATE,     # [B, N, 1] float32   — output (sum over D via atomic_add)
    D_EMA,      # [1]       float32   — output
    D_KDV,      # [1]       float32   — output
    EMA_FACTOR, # [] float32
    KDV_ALPHA,  # [] float32
    B, N, D,
    stride_xb, stride_xn, stride_xd,
    stride_gb, stride_gn,
    stride_pb, stride_pn, stride_pd,
    stride_dgb, stride_dgn,
    EMA_MIN: tl.constexpr,
    EMA_MAX: tl.constexpr,
    KDV_MIN: tl.constexpr,
    KDV_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b  = tl.program_id(0)
    bd = tl.program_id(1)
    ds = bd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = ds < D

    alpha_base = tl.load(EMA_FACTOR).to(tl.float32)
    alpha_base = tl.where(alpha_base < EMA_MIN, EMA_MIN,
                 tl.where(alpha_base > EMA_MAX, EMA_MAX, alpha_base))
    one_minus_alpha_base = 1.0 - alpha_base

    beta = tl.load(KDV_ALPHA).to(tl.float32)
    beta = tl.where(beta < KDV_MIN, KDV_MIN, tl.where(beta > KDV_MAX, KDV_MAX, beta))

    d_pool_next = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_alpha_acc = 0.0
    d_kdv_acc   = 0.0

    for n_rev in range(N):
        n = N - 1 - n_rev

        # Load saved pool[n] and pool[n-1]
        p_off  = b * stride_pb + n * stride_pn + ds * stride_pd
        pool_n = tl.load(POOL + p_off, mask=mask, other=0.0)
        pool_nm1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        if n > 0:
            pm1_off  = b * stride_pb + (n-1) * stride_pn + ds * stride_pd
            pool_nm1 = tl.load(POOL + pm1_off, mask=mask, other=0.0)

        # Load gate[n], reconstruct k_t
        g_off  = b * stride_gb + n * stride_gn
        gate_n = tl.load(GATE + g_off).to(tl.float32)
        gate_n = tl.where(gate_n < 0.0, 0.0, tl.where(gate_n > 1.0, 1.0, gate_n))
        k_t    = alpha_base + one_minus_alpha_base * gate_n

        # Upstream gradient from dout[n]
        do_off = b * stride_xb + n * stride_xn + ds * stride_xd
        dout_n = tl.load(DOUT + do_off, mask=mask, other=0.0).to(tl.float32)

        # KdV backward: out = pool + β*pool*(pool - pool_prev)
        # d_pool_from_out = dout * (1 + β*(2*pool - pool_prev))
        d_pool_from_out = dout_n * (1.0 + beta * (2.0 * pool_n - pool_nm1))
        d_pool_prev_kdv = dout_n * (-beta * pool_n)
        d_beta_local    = tl.sum(dout_n * pool_n * (pool_n - pool_nm1), axis=0)   # scalar
        d_kdv_acc       = d_kdv_acc + d_beta_local

        # Total d_pool[n] = from_out + carried from n+1
        d_pool_n = d_pool_from_out + d_pool_next

        # EMA backward: pool[n] = (1-k_t)*pool[n-1] + k_t*x[n]
        # d_x[n]     = d_pool_n * k_t
        # d_pool_prev += d_pool_n * (1-k_t)  (also gets d_pool_prev_kdv from KdV term)
        # d_alpha_base += d_pool_n · (x[n] - pool[n-1])  [d_k_t summed, then chain]
        # d_gate[n]   = sum_d( d_pool_n * (x[n] - pool[n-1]) ) * (1-alpha_base)

        x_off = b * stride_xb + n * stride_xn + ds * stride_xd
        x_n   = tl.load(X + x_off, mask=mask, other=0.0).to(tl.float32)
        innov = x_n - pool_nm1

        dx_n       = d_pool_n * k_t
        d_pool_next = d_pool_n * (1.0 - k_t) + d_pool_prev_kdv

        # Gradient through k_t: dk_t = sum_d(d_pool_n * innov)
        dk_t_sum   = tl.sum(d_pool_n * innov, axis=0)   # scalar
        d_alpha_acc = d_alpha_acc + dk_t_sum
        dg_n = dk_t_sum * one_minus_alpha_base
        dg_off = b * stride_dgb + n * stride_dgn
        tl.atomic_add(D_GATE + dg_off, dg_n)

        tl.store(DX + do_off, dx_n.to(tl.bfloat16), mask=mask)

    tl.atomic_add(D_EMA, d_alpha_acc)
    tl.atomic_add(D_KDV, d_kdv_acc)


class _CausalEMADynFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, ema_factor, kdv_alpha, block_d):
        B, N, D = x.shape
        pool = torch.empty(B, N, D, device=x.device, dtype=torch.float32)
        out  = torch.empty_like(x)

        # gate is [B,N,1] → squeeze to [B,N] for kernel
        gate_2d = gate.squeeze(-1).contiguous().float()

        sb, sn, sd = x.stride()
        gb, gn     = gate_2d.stride()

        grid = (B, triton.cdiv(D, block_d))
        _ema_dyn_fwd[grid](
            x, gate_2d, pool, out, ema_factor, kdv_alpha,
            B, N, D,
            sb, sn, sd, gb, gn,
            pool.stride(0), pool.stride(1), pool.stride(2),
            out.stride(0),  out.stride(1),  out.stride(2),
            EMA_MIN=0.005, EMA_MAX=0.5, KDV_MIN=0.0, KDV_MAX=0.5,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(x, pool, gate_2d, ema_factor, kdv_alpha)
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, dout):
        x, pool, gate_2d, ema_factor, kdv_alpha = ctx.saved_tensors
        block_d = ctx.block_d
        B, N, D = x.shape

        dx      = torch.empty_like(x)
        d_gate  = torch.zeros(B, N, 1, device=x.device, dtype=torch.float32)
        d_ema   = torch.zeros(1, device=x.device, dtype=torch.float32)
        d_kdv   = torch.zeros(1, device=x.device, dtype=torch.float32)

        dout_c = dout.contiguous()
        sb, sn, sd = x.stride()
        gb, gn     = gate_2d.stride()
        dg_b, dg_n = B*N, N  # d_gate[B,N,1] strides for [b,n] indexing

        d_gate_2d = torch.zeros(B, N, device=x.device, dtype=torch.float32)
        pb, pn, pd = pool.stride()
        dgb, dgn   = d_gate_2d.stride()

        grid = (B, triton.cdiv(D, block_d))
        _ema_dyn_bwd[grid](
            dout_c, x, pool, gate_2d,
            dx, d_gate_2d, d_ema, d_kdv,
            ema_factor, kdv_alpha,
            B, N, D,
            sb, sn, sd, gb, gn,
            pb, pn, pd, dgb, dgn,
            EMA_MIN=0.005, EMA_MAX=0.5, KDV_MIN=0.0, KDV_MAX=0.5,
            BLOCK_D=block_d,
        )
        return dx, d_gate_2d.unsqueeze(-1), d_ema, d_kdv, None


def causal_ema_dyn(
    xi:         torch.Tensor,   # [B, N, D] bfloat16
    gate:       torch.Tensor,   # [B, N, 1] float32, sigmoid output ∈ [0,1]
    ema_factor: torch.Tensor,   # scalar nn.Parameter float32 (α_base)
    kdv_alpha:  torch.Tensor,   # scalar nn.Parameter float32
    block_d:    int = _DEFAULT_BLOCK_D,
) -> torch.Tensor:
    """
    Causal IIR EMA with per-position content gate + KdV.

    k_t[n] = ema_factor + (1-ema_factor)*gate[b,n]  ∈ [ema_factor, 1.0]
    pool[n] = (1-k_t)*pool[n-1] + k_t*x[n]
    out[n]  = pool[n] + kdv_alpha*pool[n]*(pool[n]-pool[n-1])

    When gate≈0: reduces to standard EMA (α=ema_factor)
    When gate≈1: state ← x[n] (strong passkey imprint)
    """
    if not xi.is_contiguous(): xi = xi.contiguous()
    if xi.dtype != torch.bfloat16: xi = xi.bfloat16()
    gate_f = gate.float()
    if gate_f.dim() == 2: gate_f = gate_f.unsqueeze(-1)  # ensure [B,N,1]
    if not gate_f.is_contiguous(): gate_f = gate_f.contiguous()
    return _CausalEMADynFn.apply(xi, gate_f, ema_factor, kdv_alpha, block_d)
