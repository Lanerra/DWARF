#!/usr/bin/env python3
"""
Focused regression tests for HISA Stage 1/Stage 2 chunk selection.

Goal: once Stage 1 packs top-k chunks including self-chunk, Stage 2 must consume
that full packed selection rather than truncating it based on c_q. This also
covers the previously skipped first query chunk (c_q=0).
"""

import sys
import torch

sys.path.insert(0, '/home/dlewis3/Desktop/AI/DWARF/kernels')

from hierarchical_sparse_attn_v15_hisa import (
    HierarchicalSparseAttentionV15HISA,
    _build_stage1_top_k_packed,
    _prepare_stage2_selected_chunks,
)


def test_stage1_packs_the_first_query_chunk():
    # B=1, H=1, N=8, C=4, chunk_size=2, k=3
    routing_weights = torch.zeros(1, 1, 8, 4)
    routing_weights[:, :, :, 0] = 1.0

    top_k_packed = _build_stage1_top_k_packed(
        routing_weights,
        seq_len=8,
        chunk_size=2,
        num_chunks=4,
        top_k_chunks=3,
    )

    assert top_k_packed.shape == (1, 1, 4, 3)
    assert top_k_packed[0, 0, 0].tolist() == [0, -1, -1]


def test_stage2_keeps_first_chunk_selection_from_packed_selection():
    # Shape: (B=1, H=1, C=4, k=3)
    top_k_packed = torch.tensor(
        [[[
            [0, -1, -1],
            [0, 1, -1],
            [1, 0, 2],
            [2, 0, 3],
        ]]],
        dtype=torch.long,
    )

    selected, valid, ci = _prepare_stage2_selected_chunks(top_k_packed, c_q=0, num_chunks=4)

    assert selected.shape == (1, 1, 3)
    assert selected.tolist() == [[[0, -1, -1]]]
    assert valid.tolist() == [[[True, False, False]]]
    assert ci.tolist() == [[[0, 0, 0]]]


def test_stage2_keeps_self_chunk_from_packed_selection():
    top_k_packed = torch.tensor(
        [[[
            [0, -1, -1],
            [0, 1, -1],
            [1, 0, 2],
            [2, 0, 3],
        ]]],
        dtype=torch.long,
    )

    selected, valid, ci = _prepare_stage2_selected_chunks(top_k_packed, c_q=1, num_chunks=4)

    assert selected.shape == (1, 1, 3)
    assert selected.tolist() == [[[0, 1, -1]]]
    assert valid.tolist() == [[[True, True, False]]]
    assert ci.tolist() == [[[0, 1, 0]]]
    assert (selected[valid] == 1).any().item() is True


def test_stage2_preserves_all_valid_chunks_when_no_padding_present():
    top_k_packed = torch.tensor(
        [[[
            [0, -1, -1],
            [0, 1, -1],
            [1, 0, 2],
            [2, 0, 3],
        ]]],
        dtype=torch.long,
    )

    selected, valid, ci = _prepare_stage2_selected_chunks(top_k_packed, c_q=2, num_chunks=4)

    assert selected.tolist() == [[[1, 0, 2]]]
    assert valid.tolist() == [[[True, True, True]]]
    assert ci.tolist() == [[[1, 0, 2]]]
    assert set(selected[valid].tolist()) == {0, 1, 2}


def test_chunk0_empty_context_rows_do_not_produce_nans():
    if not torch.cuda.is_available():
        print('skip cuda regression (cuda unavailable)')
        return

    torch.manual_seed(0)
    device = 'cuda'
    model = HierarchicalSparseAttentionV15HISA(
        D=32, H=2, hd=16, num_chunks=4, top_k_chunks=2, hisa_top_m_tokens=1
    ).to(device)
    model.train()

    x = torch.randn(1, 64, 32, device=device, requires_grad=True)
    out = model(x)
    assert bool(out.isfinite().all().item())

    loss = out.square().mean()
    loss.backward()

    assert bool(loss.isfinite().item())
    assert x.grad is not None
    assert bool(x.grad.isfinite().all().item())


if __name__ == '__main__':
    test_stage1_packs_the_first_query_chunk()
    test_stage2_keeps_first_chunk_selection_from_packed_selection()
    test_stage2_keeps_self_chunk_from_packed_selection()
    test_stage2_preserves_all_valid_chunks_when_no_padding_present()
    test_chunk0_empty_context_rows_do_not_produce_nans()
    print('ok')
