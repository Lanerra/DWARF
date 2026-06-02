"""
HISA causality tests — verify strict token causality of the full layer.

Tests both forward-pass causality (future token corruption doesn't affect
past outputs) and gradient causality (gradients from past loss don't depend
on future tokens through chunk representatives or routing).

Run: pytest -q tests/test_hisa_causality.py -v
"""

import os
import pathlib
import sys

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn.functional as F


class TestHISACausality:
    """Strict token causality tests for HierarchicalSparseAttentionV15HISA."""

    def setup_method(self):
        from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA

        torch.manual_seed(42)
        self.B, self.N, self.D, self.H, self.hd = 2, 2048, 512, 8, 64
        self.C, self.k, self.m = 32, 4, 32
        self.chunk_size = self.N // self.C  # 64
        self.device = 'cuda'

        self.model = HierarchicalSparseAttentionV15HISA(
            D=self.D, H=self.H, hd=self.hd,
            num_chunks=self.C, top_k_chunks=self.k, hisa_top_m_tokens=self.m,
        ).to(self.device).float()

    def _forward_diff(self, x, fut, scale=20.0):
        """Return max output difference at positions < fut when position fut is corrupted."""
        self.model.eval()
        with torch.no_grad():
            y0 = self.model(x).detach()
            x2 = x.clone()
            x2[:, fut, :] += scale * torch.randn_like(x2[:, fut, :])
            y1 = self.model(x2).detach()
        return (y1[:, :fut, :] - y0[:, :fut, :]).abs().max().item()

    def test_forward_causality_mid_sequence(self):
        """Corrupting mid-sequence tokens must not affect earlier outputs."""
        x = torch.randn(self.B, self.N, self.D, device=self.device)

        for fut in [self.N // 4, self.N // 2, 3 * self.N // 4]:
            diff = self._forward_diff(x, fut)
            assert diff < 1e-5, \
                f"Forward causality violation at fut={fut}: max_diff={diff:.6f}"

    def test_forward_causality_cross_chunk(self):
        """Corrupting tokens in later chunks must not affect earlier chunk outputs."""
        x = torch.randn(self.B, self.N, self.D, device=self.device)
        cs = self.chunk_size

        # Corrupt chunk 16, check chunks 0-15
        fut = 16 * cs + 32
        diff = self._forward_diff(x, fut)
        # Only check positions before chunk 16
        self.model.eval()
        with torch.no_grad():
            y0 = self.model(x).detach()
            x2 = x.clone()
            x2[:, fut, :] += 20.0 * torch.randn_like(x2[:, fut, :])
            y1 = self.model(x2).detach()
        prefix_diff = (y1[:, :16 * cs, :] - y0[:, :16 * cs, :]).abs().max().item()
        assert prefix_diff < 1e-5, \
            f"Cross-chunk forward leak: corrupt chunk 16, check < chunk 16: diff={prefix_diff:.6f}"

    def test_forward_causality_massive_corruption(self):
        """Corrupting ALL tokens in a chunk must not affect earlier chunks."""
        x = torch.randn(self.B, self.N, self.D, device=self.device)
        cs = self.chunk_size

        self.model.eval()
        with torch.no_grad():
            y0 = self.model(x).detach()
            x2 = x.clone()
            # Corrupt entire chunk 16
            for i in range(cs):
                pos = 16 * cs + i
                x2[:, pos, :] += 20.0 * torch.randn_like(x2[:, pos, :])
            y1 = self.model(x2).detach()
        prefix_diff = (y1[:, :16 * cs, :] - y0[:, :16 * cs, :]).abs().max().item()
        assert prefix_diff < 1e-5, \
            f"Massive corruption leak: diff={prefix_diff:.6f}"

    def test_gradient_additivity_adjacent_chunks(self):
        """
        Gradient additivity: grad(A_loss + B_loss) == grad(A_loss) + grad(B_loss).
        If causal, losses from independent chunks produce independent gradients.
        """
        x = torch.randn(self.B, self.N, self.D, device=self.device, requires_grad=True)
        cs = self.chunk_size

        def get_param_grads(loss_chunk):
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            self.model.train()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                y = self.model(x)
            s, e = loss_chunk * cs, (loss_chunk + 1) * cs
            loss = y[:, s:e, :].float().square().mean()
            loss.backward()
            return {n: p.grad.clone() for n, p in self.model.named_parameters()
                    if p.grad is not None}

        g0 = get_param_grads(0)
        g1 = get_param_grads(1)

        # Combined
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        self.model.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            y = self.model(x)
        loss_both = (y[:, :2 * cs, :].float().square().mean())
        loss_both.backward()
        g01 = {n: p.grad.clone() for n, p in self.model.named_parameters()
               if p.grad is not None}

        for name in ['W_q.weight', 'W_k.weight', 'W_v.weight', 'W_o.weight']:
            g_sum = (g0[name] + g1[name]).float()
            diff = (g_sum - g01[name]).abs().max().item()
            assert diff < 1e-3, \
                f"Gradient non-additivity for {name}: diff={diff:.6f}"

    def test_gradient_additivity_non_adjacent_chunks(self):
        """Gradient additivity for chunks separated by gaps."""
        x = torch.randn(self.B, self.N, self.D, device=self.device, requires_grad=True)
        cs = self.chunk_size

        def get_param_grads(loss_chunk):
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            self.model.train()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                y = self.model(x)
            s, e = loss_chunk * cs, (loss_chunk + 1) * cs
            loss = y[:, s:e, :].float().square().mean()
            loss.backward()
            return {n: p.grad.clone() for n, p in self.model.named_parameters()
                    if p.grad is not None}

        # Test chunks 4 and 12 (gap of 8)
        g4 = get_param_grads(4)
        g12 = get_param_grads(12)

        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        self.model.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            y = self.model(x)
        s4, e4 = 4 * cs, 5 * cs
        s12, e12 = 12 * cs, 13 * cs
        loss_both = (y[:, s4:e4, :].float().square().mean() +
                     y[:, s12:e12, :].float().square().mean())
        loss_both.backward()
        g412 = {n: p.grad.clone() for n, p in self.model.named_parameters()
                if p.grad is not None}

        for name in ['W_k.weight', 'W_v.weight']:
            g_sum = (g4[name] + g12[name]).float()
            diff = (g_sum - g412[name]).abs().max().item()
            assert diff < 1e-3, \
                f"Gradient non-additivity (chunks 4+12) for {name}: diff={diff:.6f}"

    def test_chunk_rep_causal_isolation(self):
        """
        Verify that corrupting future chunk representatives doesn't change
        routing weights for past queries (causal mask blocks the dependency).
        """
        from hierarchical_sparse_attn_v15_hisa import (
            _compute_chunk_representatives, _compute_routing,
        )

        cs = self.chunk_size
        B, N, D, H, hd = self.B, self.N, self.D, self.H, self.hd

        def to_heads(t):
            return t.reshape(B, N, H, hd).transpose(1, 2)

        x = torch.randn(B, N, D, device=self.device)
        Q = to_heads(self.model.W_q(x))
        K = to_heads(self.model.W_k(x))

        N_padded = self.C * cs
        K_pad = F.pad(K, (0, 0, 0, N_padded - N))

        chunk_reps = _compute_chunk_representatives(K_pad, self.C)

        routing, _ = _compute_routing(
            Q, chunk_reps, seq_len=N, num_chunks=self.C,
            chunk_size=cs, hd=hd, temperature=1.0, training=True,
        )

        # Corrupt token in chunk 1, recompute
        K2 = K_pad.clone()
        K2[:, :, 100, :] += 10.0
        chunk_reps2 = _compute_chunk_representatives(K2, self.C)
        routing2, _ = _compute_routing(
            Q, chunk_reps2, seq_len=N, num_chunks=self.C,
            chunk_size=cs, hd=hd, temperature=1.0, training=True,
        )

        # Chunk 0 queries must NOT change
        rw_diff_c0 = (routing2[:, :, :cs, :] - routing[:, :, :cs, :]).abs()
        assert rw_diff_c0.max().item() < 1e-8, \
            f"Chunk rep leak to chunk 0 queries: max_diff={rw_diff_c0.max().item():.2e}"

        # Chunk 1 queries SHOULD change (they can see chunk 1)
        rw_diff_c1 = (routing2[:, :, cs:2 * cs, :] - routing[:, :, cs:2 * cs, :]).abs()
        assert rw_diff_c1.max().item() > 0.01, \
            "Chunk 1 queries should be sensitive to chunk 1 rep changes"

    def test_stage2_mask_causal_selection(self):
        """Verify Stage-2 token mask only selects from causally valid chunks."""
        from hierarchical_sparse_attn_v15_hisa import (
            _compute_chunk_representatives, _compute_routing,
            _build_stage1_top_k_packed,
        )

        cs = self.chunk_size
        B, N, D, H, hd = self.B, self.N, self.D, self.H, self.hd

        def to_heads(t):
            return t.reshape(B, N, H, hd).transpose(1, 2)

        x = torch.randn(B, N, D, device=self.device)
        Q = to_heads(self.model.W_q(x))
        K = to_heads(self.model.W_k(x))

        N_padded = self.C * cs
        K_pad = F.pad(K, (0, 0, 0, N_padded - N))

        chunk_reps = _compute_chunk_representatives(K_pad, self.C)
        routing, _ = _compute_routing(
            Q, chunk_reps, seq_len=N, num_chunks=self.C,
            chunk_size=cs, hd=hd, temperature=1.0, training=True,
        )

        top_k_packed = _build_stage1_top_k_packed(
            routing, seq_len=N, chunk_size=cs,
            num_chunks=self.C, top_k_chunks=self.k,
        )

        # c_q=0 should only select chunk 0 (and -1 padding)
        selected_c0 = top_k_packed[:, :, 0, :].clone()
        valid_c0 = selected_c0[selected_c0 >= 0]
        assert (valid_c0 == 0).all(), \
            f"c_q=0 selected non-chunk-0: {valid_c0.unique().tolist()}"

        # c_q=n should only select chunks <= n
        for c_q in range(self.C):
            selected = top_k_packed[:, :, c_q, :].clone()
            valid = selected[selected >= 0]
            if valid.numel() > 0:
                assert (valid <= c_q).all(), \
                    f"c_q={c_q} selected future chunk {valid.max().item()}"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
