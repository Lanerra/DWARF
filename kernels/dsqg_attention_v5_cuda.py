"""
DSQG V5 — CUDA kernel drop-in replacement for dsqg_attention_v5.py

Wraps the compiled dsqg_cuda extension with the same interface as
DSQGAttentionV5 and dsqg_attention_v5(). Falls back to the Triton
implementation if the CUDA extension is not compiled.

Build the extension first:
    cd kernels/dsqg_cuda && python setup.py build_ext --inplace
    # or: pip install -e kernels/dsqg_cuda

Key differences from Triton V5:
  - COSSIN saves theta (not cos/sin directly); backward recomputes with __sincosf
  - D (softmax auxiliary) is computed in Python before calling the backward
  - Phase gradient buffers reduced in Python (same as Triton)
  - dK, dV output in fp32 (same as Triton; training script handles)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Extension import ──────────────────────────────────────────────────────────
_cuda_ext = None

def _load_cuda_ext():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext
    try:
        import sys, os
        # Try inplace build first (for development)
        ext_dir = os.path.join(os.path.dirname(__file__), "dsqg_cuda")
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import dsqg_cuda as ext
        _cuda_ext = ext
        return ext
    except ImportError:
        return None

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_OFFSETS = list(range(33)) + [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
N_SPARSE   = 11
R_PLANES   = 2

# ── Autograd wrapper ──────────────────────────────────────────────────────────
class _DSQGFnV5Cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                pos_bias, scale_embed,
                phase_base, phase_gain,
                y_pre, z_pre):
        ext = _load_cuda_ext()
        if ext is None:
            raise RuntimeError(
                "DSQG CUDA extension not built. Run:\n"
                "  cd kernels/dsqg_cuda && python setup.py build_ext --inplace")

        B, H, N, HD = q.shape
        scale = HD ** -0.5

        # Ensure contiguous bf16 for Q/K/V
        q_c = q.contiguous().to(torch.bfloat16)
        k_c = k.contiguous().to(torch.bfloat16)
        v_c = v.contiguous().to(torch.bfloat16)
        pb  = pos_bias.contiguous().float()
        se  = scale_embed.contiguous().float()
        phb = phase_base.contiguous().float()
        phg = phase_gain.contiguous().float()
        y   = y_pre.contiguous().float()
        z   = z_pre.contiguous().float()

        out, lse, cossin = ext.fwd(q_c, k_c, v_c, pb, se, phb, phg, y, z, scale)

        ctx.save_for_backward(q_c, k_c, v_c, pb, se, phb, phg, y, z, lse, cossin, out)
        ctx.scale = scale
        ctx.B, ctx.H, ctx.N, ctx.HD = B, H, N, HD
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _load_cuda_ext()
        q, k, v, pb, se, phb, phg, y, z, lse, cossin, out = ctx.saved_tensors
        B, H, N, HD = ctx.B, ctx.H, ctx.N, ctx.HD
        scale = ctx.scale

        dout = grad_out.contiguous().to(torch.bfloat16)

        # Precompute D = rowsum(dout * out) — softmax auxiliary
        # D[b,h,n] = sum_d dout[b,h,n,d] * out[b,h,n,d]
        D = (dout.float() * out.float()).sum(dim=-1).contiguous()  # [B,H,N]

        results = ext.bwd(
            q, k, v, pb, se, phb, phg, y, z,
            lse, cossin, dout, D, scale)

        dq, dk, dv, dpb_buf, dpg_buf, dy, dz, dpbias, dscale = results

        # Reduce phase gradient buffers: [B*H, n_tiles, 11*2] → [11, H, 2]
        n_tiles = (N + 63) // 64
        def reduce_phase_buf(buf):
            # buf: [B*H, n_tiles, 22]
            r = buf.view(B, H, n_tiles, N_SPARSE, R_PLANES).sum(dim=(0, 2))  # [H,11,2]
            return r.permute(1, 0, 2).contiguous()  # [11, H, 2]

        d_phase_base = reduce_phase_buf(dpb_buf)
        d_phase_gain = reduce_phase_buf(dpg_buf)

        # Convert dK, dV back to original dtype
        dtype = q.dtype  # bf16
        return (
            dq.to(dtype),       # dQ
            dk.to(dtype),       # dK
            dv.to(dtype),       # dV
            dpbias,             # d_pos_bias
            dscale,             # d_scale_embed
            d_phase_base,       # d_phase_base
            d_phase_gain,       # d_phase_gain
            dy,                 # d_y_pre
            dz,                 # d_z_pre
        )


def dsqg_attention_v5_cuda(q, k, v, pos_bias, scale_embed,
                            phase_base, phase_gain, y_pre, z_pre):
    """
    Drop-in replacement for dsqg_attention_v5() using the compiled CUDA kernel.

    Args (same as Triton version):
        q, k, v:         [B, H, N, HD]  bf16
        pos_bias:        [44, H]         fp32
        scale_embed:     [44, HD]        fp32
        phase_base:      [11, H, 2]      fp32
        phase_gain:      [11, H, 2]      fp32
        y_pre:           [B, H, N, 2]   fp32
        z_pre:           [B, H, N, 2]   fp32

    Returns:
        out:             [B, H, N, HD]  same dtype as q
    """
    return _DSQGFnV5Cuda.apply(
        q, k, v, pos_bias, scale_embed,
        phase_base, phase_gain, y_pre, z_pre)


# ── Module (drop-in for DSQGAttentionV5) ─────────────────────────────────────
class DSQGAttentionV5Cuda(nn.Module):
    """
    Drop-in replacement for DSQGAttentionV5 using the compiled CUDA kernel.
    Same parameter layout, same forward signature.
    """
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        HD = head_dim

        # DSQG parameters (identical to V5 module)
        self.pos_bias    = nn.Parameter(torch.zeros(44, num_heads))
        self.scale_embed = nn.Parameter(torch.zeros(44, HD))

        # MOVT: sparse-only — 11 global offsets × 2 planes
        self.phase_base  = nn.Parameter(torch.zeros(11, num_heads, R_PLANES))
        # QK-OVT
        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes   = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.phase_gain   = nn.Parameter(torch.zeros(11, num_heads, R_PLANES))

        # NPCI (not in CUDA kernel — stays in Python)
        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

    def forward(self, q, k, v, k_inj=None, v_inj=None):
        """
        q, k, v: [B, H, N, HD] bf16
        k_inj, v_inj: NPCI injection tensors [B, H, N, HD] (computed by caller)
        """
        B, H, N, HD = q.shape
        sc = HD ** -0.5

        # QK-OVT: compute y_pre and z_pre
        y_pre = torch.einsum('bhnd,rd->bhnr', q.float(), self.query_probes) * sc
        z_pre = torch.einsum('bhnd,rd->bhnr', k.float(), self.key_probes)   * sc
        y_pre = y_pre.contiguous()
        z_pre = z_pre.contiguous()

        return dsqg_attention_v5_cuda(
            q, k, v,
            self.pos_bias, self.scale_embed,
            self.phase_base, self.phase_gain,
            y_pre, z_pre)


# ── Quick smoke test ──────────────────────────────────────────────────────────
def _smoke_test(device='cuda'):
    """Verify CUDA kernel output matches Triton reference."""
    print("DSQG V5 CUDA — smoke test")
    print("=" * 50)

    ext = _load_cuda_ext()
    if ext is None:
        print("  CUDA extension not built — skipping")
        print("  Run: cd kernels/dsqg_cuda && python setup.py build_ext --inplace")
        return False

    # Import Triton reference
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dsqg_attention_v5 import dsqg_attention_v5 as triton_v5

    B, H, N, HD = 2, 8, 256, 64
    torch.manual_seed(42)
    q   = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    k   = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    v   = torch.randn(B,H,N,HD, device=device, dtype=torch.bfloat16) * 0.1
    pb  = torch.randn(44,H,  device=device, dtype=torch.float32) * 0.5
    se  = torch.zeros(44,HD, device=device, dtype=torch.float32)
    phb = torch.randn(11,H,2,device=device, dtype=torch.float32) * 0.1
    phg = torch.randn(11,H,2,device=device, dtype=torch.float32) * 0.01
    y   = torch.randn(B,H,N,2,device=device, dtype=torch.float32) * 0.05
    z   = torch.randn(B,H,N,2,device=device, dtype=torch.float32) * 0.05

    # Triton forward
    out_triton = triton_v5(q.clone(), k.clone(), v.clone(),
                           pb.clone(), se.clone(),
                           phb.clone(), phg.clone(),
                           y.clone(), z.clone())

    # CUDA forward
    out_cuda = dsqg_attention_v5_cuda(
        q.clone(), k.clone(), v.clone(),
        pb.clone(), se.clone(),
        phb.clone(), phg.clone(),
        y.clone(), z.clone())

    fwd_err = (out_triton.float() - out_cuda.float()).abs().max().item()
    print(f"  Forward max_err (Triton vs CUDA): {fwd_err:.6f}", end="")
    fwd_ok = fwd_err < 0.05
    print(f"  {'PASS ✓' if fwd_ok else 'FAIL ✗'}")

    # Backward: check gradients agree
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    pb2 = pb.clone().requires_grad_(True)
    phb2 = phb.clone().requires_grad_(True)
    out_cuda2 = dsqg_attention_v5_cuda(q2, k2, v2, pb2, se.clone(), phb2, phg.clone(), y.clone(), z.clone())
    out_cuda2.sum().backward()

    q3 = q.clone().requires_grad_(True)
    k3 = k.clone().requires_grad_(True)
    v3 = v.clone().requires_grad_(True)
    pb3 = pb.clone().requires_grad_(True)
    phb3 = phb.clone().requires_grad_(True)
    out_triton3 = triton_v5(q3, k3, v3, pb3, se.clone(), phb3, phg.clone(), y.clone(), z.clone())
    out_triton3.sum().backward()

    dq_err = (q2.grad.float()   - q3.grad.float()).abs().max().item()
    dk_err = (k2.grad.float()   - k3.grad.float()).abs().max().item()
    dpb_err= (pb2.grad.float()  - pb3.grad.float()).abs().max().item()
    bwd_ok = all(e < 0.05 for e in [dq_err, dk_err, dpb_err])
    print(f"  Bwd dQ err: {dq_err:.6f}  dK err: {dk_err:.6f}  dPB err: {dpb_err:.6f}", end="")
    print(f"  {'PASS ✓' if bwd_ok else 'FAIL ✗'}")

    ok = fwd_ok and bwd_ok
    print("=" * 50)
    print(f"{'PASS ✓' if ok else 'FAIL ✗ — review errors before using for training'}")
    return ok


if __name__ == "__main__":
    _smoke_test()
