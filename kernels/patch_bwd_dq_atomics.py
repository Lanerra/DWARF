"""
patch_bwd_dq_atomics.py — Replace DPB/DSE atomic_adds in _bwd_dq_v8 with
private buffers + host-side reduction, matching the pattern already used
for phase_base/phase_gain in _bwd_dkdv_v8.

Applies to both dsqg_attention_v8_h100.py and dsqg_attention_v8_4090.py.
"""

import re, sys, pathlib

KERNELS = [
    pathlib.Path(__file__).parent / "dsqg_attention_v8_h100.py",
    pathlib.Path(__file__).parent / "dsqg_attention_v8_4090.py",
]

# ── 1. Kernel signature: replace DPB/DSE output tensors + old strides ────────

OLD_SIG_TENSORS = "    DQ, DPB, DSE, DY_PRE,"
NEW_SIG_TENSORS = "    DQ, DPB_BUF, DSE_BUF, DY_PRE,   # private per-program buffers (no atomics)"

OLD_SIG_STRIDES = """\
    stride_dpbi, stride_dpbh,
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dsei, stride_dsed,"""

NEW_SIG_STRIDES = """\
    stride_dpb_bh, stride_dpb_blk,   # DPB_BUF[bh, blk, i]
    stride_pbi,  stride_pbh,
    stride_sei,  stride_sed,
    stride_dse_bh, stride_dse_blk,   # DSE_BUF[bh, blk, i*HD + d]"""

# ── 2. Two identical atomic_add pairs inside the static_range loop ────────────
#    (appear once in i<14 branch and once in else branch — same text both times)

OLD_ATOMICS = """\
            tl.atomic_add(DPB + i*stride_dpbi + h*stride_dpbh,
                          tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.atomic_add(DSE + i*stride_dsei + ds*stride_dsed, tl.where(dm, dse_i, 0.0))"""

NEW_ATOMICS = """\
            tl.store(DPB_BUF + bh*stride_dpb_bh + blk*stride_dpb_blk + i,
                     tl.sum(tl.where(val, ds_v, 0.0)))
            dse_i = tl.sum(ds_v[:,None] * q, axis=0) * sc
            tl.store(DSE_BUF + bh*stride_dse_bh + blk*stride_dse_blk + i*HD + ds,
                     tl.where(dm, dse_i, 0.0), mask=dm)"""

# ── 3. backward(): replace dpb/dse alloc, update call args, add reduction ─────

OLD_BWD_ALLOC = """\
        dq     = torch.empty_like(q)
        dpb    = torch.zeros_like(pb)
        dse    = torch.zeros_like(se)
        dy_pre = torch.zeros_like(y_pre)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb, dse, dy_pre,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            dpb.stride(0),  dpb.stride(1),
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            dse.stride(0),  dse.stride(1),
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )

        dk     = torch.empty_like(k)
        dv     = torch.empty_like(v)
        dz_pre = torch.zeros_like(z_pre)

        blocks_n = (N + BN - 1) // BN
        _dev     = q.device"""

NEW_BWD_ALLOC = """\
        blocks_n = (N + BN - 1) // BN
        _dev     = q.device

        dq      = torch.empty_like(q)
        dy_pre  = torch.zeros_like(y_pre)
        # Private per-program buffers for dpb/dse — no atomics, reduce after kernel
        dpb_buf = torch.empty(B * H, blocks_n, J,       device=_dev, dtype=torch.float32)
        dse_buf = torch.empty(B * H, blocks_n, J * HD,  device=_dev, dtype=torch.float32)

        _bwd_dq_v8[g](
            q, k, v, pb, se, phase_base, phase_gain, y_pre, z_pre,
            dout, out, lse, D,
            dq, dpb_buf, dse_buf, dy_pre,
            q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
            k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
            v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            lse.stride(0),  lse.stride(1),  lse.stride(2),
            D.stride(0),    D.stride(1),    D.stride(2),
            dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
            blocks_n * J,   J,              # stride_dpb_bh, stride_dpb_blk
            pb.stride(0),   pb.stride(1),
            se.stride(0),   se.stride(1),
            blocks_n*J*HD,  J*HD,           # stride_dse_bh, stride_dse_blk
            phase_base.stride(0), phase_base.stride(1),
            phase_gain.stride(0), phase_gain.stride(1),
            y_pre.stride(0),      y_pre.stride(1),      y_pre.stride(2),
            z_pre.stride(0),      z_pre.stride(1),      z_pre.stride(2),
            dy_pre.stride(0),     dy_pre.stride(1),     dy_pre.stride(2),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            num_warps=NW, num_stages=NS,
        )
        # Reduce per-program partials → [J, H] and [J, HD]
        dpb = dpb_buf.view(B, H, blocks_n, J).sum(dim=(0, 2)).permute(1, 0).contiguous()
        dse = dse_buf.view(B, H, blocks_n, J, HD).sum(dim=(0, 1, 2)).contiguous()

        dk     = torch.empty_like(k)
        dv     = torch.empty_like(v)
        dz_pre = torch.zeros_like(z_pre)"""

PATCHES = [
    (OLD_SIG_TENSORS,  NEW_SIG_TENSORS),
    (OLD_SIG_STRIDES,  NEW_SIG_STRIDES),
    (OLD_ATOMICS,      NEW_ATOMICS),   # applied twice (both branches have identical text)
    (OLD_BWD_ALLOC,    NEW_BWD_ALLOC),
]

def patch(path):
    src = path.read_text()
    original = src
    for old, new in PATCHES:
        count = src.count(old)
        if count == 0:
            print(f"  WARN: pattern not found in {path.name}:\n    {old[:60]!r}")
            continue
        src = src.replace(old, new)
        print(f"  {path.name}: replaced {count}× — {old[:55]!r}")
    if src != original:
        path.write_text(src)
        print(f"  → wrote {path.name}")
    else:
        print(f"  → no changes needed in {path.name}")

for k in KERNELS:
    print(f"\nPatching {k.name} ...")
    patch(k)

print("\nDone.")
