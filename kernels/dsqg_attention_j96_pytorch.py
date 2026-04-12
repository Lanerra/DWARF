"""
DSQG Attention J=96 — Pure PyTorch implementation for N=4096

Uses the same math as V8 but avoids Triton static_range(96) compilation wall.
Triton can't compile static_range(96) — the unrolled IR blows the compiler limit.

This is the reference/fallback implementation. Slower than Triton V8 (~3-5×)
but correct and compilable. For a 33M-param model at BS=8/N=4096 this is fine.

J=96 offset set (greedy, targeting N=4096, from se096_derivation.rs):
  2-hop coverage: 75.29% at N=4096
  Spectral gap: 62.4 (Z/2048Z) = 12.3× se015
  EMA_INIT = 1/48 = 0.020833 (δ_relay_min=48, unchanged)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ALL_OFFSETS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 19, 21, 23, 28,
    48, 64, 96, 121, 161, 192, 212, 245, 273, 295, 342, 375, 384,
    413, 441, 473, 512, 549, 579, 593, 631, 653, 694, 716, 768,
    826, 846, 900, 936, 970, 1000, 1024, 1074, 1108, 1144, 1166,
    1190, 1218, 1244, 1288, 1322, 1385, 1423, 1451, 1497, 1522,
    1550, 1581, 1603, 1617, 1634, 1651, 1661, 1710, 1743, 1780,
    1810, 1820, 1852, 1860, 1876, 1886, 1897, 1903, 1916, 1926,
    1929, 1941, 1965, 1983, 2006, 2011, 2029, 2037, 2044, 2068,
    2097, 2113, 2199,
]
J        = len(ALL_OFFSETS)   # 96
J_SMALL  = 17                 # indices 0-16: δ≤28, no MOVT
J_LARGE  = 79                 # indices 17-95: δ≥48, MOVT applied
MAX_DELTA = max(ALL_OFFSETS)  # 2199
R_PLANES  = 2
assert J == 96 and J_SMALL + J_LARGE == J


def npci_rotate(x: torch.Tensor, x_delta: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
    """Norm-preserving causal injection: rotate x toward x_delta by angle theta.
    x, x_delta: [B, H, N, HD]  theta: [H]
    """
    cos_t = torch.cos(theta).view(1, -1, 1, 1)
    sin_t = torch.sin(theta).view(1, -1, 1, 1)
    x_norm     = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    xd_norm    = x_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    x_unit     = x / x_norm
    xd_unit    = x_delta / xd_norm
    perp       = xd_unit - (xd_unit * x_unit).sum(-1, keepdim=True) * x_unit
    perp_norm  = perp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    mask       = (perp_norm > x_norm * 1e-10).float()
    perp_unit  = perp / perp_norm * mask
    return (cos_t * x + sin_t * x_norm * perp_unit).to(x.dtype)


# How many offsets to process per chunk — tune for VRAM.
# At BS=8, N=4096, HD=64: each chunk of C offsets needs ~B*H*N*C*HD*4 bytes
# C=8: 8*8*4096*8*64*4 = 1.07 GB per gather tensor (Ka or Va) -- safe on 24GB
_CHUNK = 8


def _dsqg_j96_forward(q, k, v, pos_bias, scale_embed,
                      phase_base, phase_gain, y_pre, z_pre):
    """
    Pure-PyTorch DSQG forward for J=96, chunked to avoid OOM.
    Processes offsets in chunks of _CHUNK, accumulating online softmax (mi/li/acc).

    q,k,v:        [B, H, N, HD]   bfloat16 (cast to float32 internally)
    pos_bias:     [96, H]          float32
    scale_embed:  [96, HD]         float32
    phase_base:   [79, H, 2]       float32  MOVT angles (large offsets)
    phase_gain:   [79, H, 2]       float32  QK-OVT gains
    y_pre:        [B, H, N, 2]    float32  Q @ query_probes / sqrt(HD)
    z_pre:        [B, H, N, 2]    float32  K @ key_probes / sqrt(HD)
    Returns:      [B, H, N, HD]   bfloat16
    """
    B, H, N, HD = q.shape
    sc = HD ** -0.5

    qf = q.float()
    kf = k.float()
    vf = v.float()

    off = torch.tensor(ALL_OFFSETS, device=q.device, dtype=torch.long)  # [96]
    ni  = torch.arange(N, device=q.device)                               # [N]

    kp = F.pad(kf, (0, 0, MAX_DELTA, 0))  # [B, H, N+MAX_DELTA, HD]
    vp = F.pad(vf, (0, 0, MAX_DELTA, 0))
    zp = F.pad(z_pre, (0, 0, MAX_DELTA, 0))  # [B, H, N+MAX_DELTA, 2]

    # Online softmax accumulators
    mi  = torch.full((B, H, N), float('-inf'), device=q.device, dtype=torch.float32)
    li  = torch.zeros((B, H, N),              device=q.device, dtype=torch.float32)
    acc = torch.zeros((B, H, N, HD),          device=q.device, dtype=torch.float32)

    for chunk_start in range(0, J, _CHUNK):
        chunk_end = min(chunk_start + _CHUNK, J)
        c_idx = slice(chunk_start, chunk_end)   # which offsets in this chunk
        C = chunk_end - chunk_start

        off_c = off[c_idx]                       # [C]
        gi_c  = MAX_DELTA - off_c[None, :] + ni[:, None]  # [N, C]

        Ka_c = kp[:, :, gi_c, :]   # [B, H, N, C, HD]
        Va_c = vp[:, :, gi_c, :]   # [B, H, N, C, HD]

        # Scores for this chunk
        s_c  = (qf.unsqueeze(3) * Ka_c).sum(-1) * sc              # [B, H, N, C]
        s_c += pos_bias[c_idx].T[None, :, None, :]                 # [1, H, 1, C]
        s_c += (qf.unsqueeze(3) * scale_embed[c_idx][None, None, :, :]).sum(-1) * sc

        # Causal mask
        cmask = ni[:, None] < off_c[None, :]                       # [N, C]
        s_c   = s_c.masked_fill(cmask[None, None, :, :], float('-inf'))

        # MOVT for large offsets in this chunk
        Va_rot_c = Va_c.clone()
        for ci in range(C):
            global_i = chunk_start + ci
            if global_i >= J_SMALL:                                # large offset
                lg_i = global_i - J_SMALL                         # index into phase_base
                zi   = zp[:, :, gi_c[:, ci], :]                   # [B, H, N, 2]
                yi   = y_pre                                       # [B, H, N, 2]
                pb   = phase_base[lg_i]                            # [H, 2]
                pg   = phase_gain[lg_i]                            # [H, 2]
                theta = (pb[None, :, None, :] +
                         pg[None, :, None, :] * yi * zi)           # [B, H, N, 2]
                cos0 = torch.cos(theta[..., 0]); sin0 = torch.sin(theta[..., 0])
                cos1 = torch.cos(theta[..., 1]); sin1 = torch.sin(theta[..., 1])
                v0 = Va_c[:, :, :, ci, 0]; v1 = Va_c[:, :, :, ci, 1]
                Va_rot_c[:, :, :, ci, 0] = cos0 * v0 - sin0 * v1
                Va_rot_c[:, :, :, ci, 1] = sin0 * v0 + cos0 * v1
                v2 = Va_c[:, :, :, ci, 2]; v3 = Va_c[:, :, :, ci, 3]
                Va_rot_c[:, :, :, ci, 2] = cos1 * v2 - sin1 * v3
                Va_rot_c[:, :, :, ci, 3] = sin1 * v2 + cos1 * v3

        # Online softmax update: merge chunk scores with running (mi, li, acc)
        mn_c = s_c.max(dim=-1).values                              # [B, H, N]
        mn   = torch.maximum(mi, mn_c)

        # Rescale existing accumulator
        cor_old = torch.where(mi > float('-inf'),
                              torch.exp(mi - mn), torch.zeros_like(mi))  # [B, H, N]
        li  = li * cor_old
        acc = acc * cor_old.unsqueeze(-1)

        # Add new chunk
        exp_c = torch.exp(s_c - mn.unsqueeze(-1))                 # [B, H, N, C]
        exp_c = torch.nan_to_num(exp_c, nan=0.0, posinf=0.0)
        li   += exp_c.sum(-1)
        acc  += (exp_c.unsqueeze(-1) * Va_rot_c).sum(-2)          # [B, H, N, HD]
        mi    = mn

        del Ka_c, Va_c, Va_rot_c, s_c, exp_c, cor_old

    # Normalize
    li_safe = li.clamp(min=1e-12)
    out = acc / li_safe.unsqueeze(-1)                              # [B, H, N, HD]
    return out.to(q.dtype)


class DSQGAttentionJ96(nn.Module):
    """
    DSQG attention for J=96, N=4096. Pure PyTorch — no Triton.
    Same parameter layout as V8 but with J=96 and J_LARGE=79.
    """
    def __init__(self, embedding_dim, num_heads, seq_len=4096, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim
        assert HD >= 4, "HD must be >= 4 for 2-plane Givens rotation"

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

        self.phase_base   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))
        self.phase_gain   = nn.Parameter(torch.zeros(J_LARGE, num_heads, R_PLANES))

        self.query_probes = nn.Parameter(torch.zeros(R_PLANES, HD))
        self.key_probes   = nn.Parameter(torch.zeros(R_PLANES, HD))

        self.npci_theta_k = nn.Parameter(torch.zeros(num_heads))
        self.npci_theta_v = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = npci_rotate(k, k_delta, self.npci_theta_k)
            v = npci_rotate(v, v_delta, self.npci_theta_v)

        sc    = HD ** -0.5
        y_pre = torch.einsum('bhnd,rd->bhnr',
                             q.float(), self.query_probes.float()).mul(sc).contiguous()
        z_pre = torch.einsum('bhnd,rd->bhnr',
                             k.float(), self.key_probes.float()).mul(sc).contiguous()

        out = _dsqg_j96_forward(q, k, v,
                                 self.pos_bias.float(), self.scale_embed.float(),
                                 self.phase_base.float(), self.phase_gain.float(),
                                 y_pre, z_pre)

        out      = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Testing DSQGAttentionJ96 on {device}')
    print(f'J={J}, J_SMALL={J_SMALL}, J_LARGE={J_LARGE}, MAX_DELTA={MAX_DELTA}')

    D, H, N = 512, 8, 64
    m = DSQGAttentionJ96(D, H, seq_len=4096).to(device)
    print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')

    x = torch.randn(2, N, D, device=device, dtype=torch.bfloat16)
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        t0 = time.time()
        out = m(x)
        print(f'Forward: {out.shape}, time={time.time()-t0:.3f}s')

    # Backward
    loss = out.float().mean()
    loss.backward()
    print(f'Backward OK')

    # Longer sequence
    x2 = torch.randn(1, 512, D, device=device, dtype=torch.bfloat16)
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        out2 = m(x2)
    print(f'N=512 forward: {out2.shape}')

    print('All tests passed.')
