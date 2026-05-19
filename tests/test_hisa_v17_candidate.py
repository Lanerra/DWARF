import os
import pathlib
import sys

import torch

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
for _d in [_project_root, os.path.join(_project_root, 'kernels')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from hierarchical_sparse_attn_v16_hisa_strict import (
    HierarchicalSparseAttentionV16HISAStrict,
    _build_stage1_top_k_per_token,
    _build_stage2_token_indices_per_token,
    _compute_chunk_representatives,
    _compute_past_only_routing,
)
from hierarchical_sparse_attn_v17_hisa_strict_triton import (
    HierarchicalSparseAttentionV17HISAStrictTriton,
    _build_stage1_top_k_per_token_prev_mandatory,
    _build_stage2_token_indices_per_token_triton,
)


def _make_pair(seed=0):
    torch.manual_seed(seed)
    ref = HierarchicalSparseAttentionV16HISAStrict(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
    ).eval()
    cand = HierarchicalSparseAttentionV17HISAStrictTriton(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
    ).eval()
    cand.load_state_dict(ref.state_dict())
    return ref, cand


def test_v17_candidate_matches_v16_forward_on_tiny_input():
    ref, cand = _make_pair(seed=11)
    x = torch.randn(2, 15, 16)

    with torch.no_grad():
        y_ref = ref(x)
        y_cand = cand(x)

    diff = (y_ref - y_cand).abs().max().item()
    assert diff < 1e-6


def test_v17_candidate_preserves_prefix_consistency():
    _, cand = _make_pair(seed=12)
    prefix = torch.randn(1, 11, 16)
    suffix = torch.randn(1, 9, 16)

    with torch.no_grad():
        y_prefix = cand(prefix)
        y_full = cand(torch.cat([prefix, suffix], dim=1))

    diff = (y_prefix - y_full[:, : prefix.size(1), :]).abs().max().item()
    assert diff < 1e-6


def test_v17_candidate_train_eval_agree_without_dropout():
    _, cand = _make_pair(seed=13)
    x = torch.randn(1, 12, 16)

    with torch.no_grad():
        cand.train()
        y_train = cand(x)
        cand.eval()
        y_eval = cand(x)

    diff = (y_train - y_eval).abs().max().item()
    assert diff < 1e-6


def test_v17_stage1_mandatory_prev_chunk_uses_fixed_budget():
    B, H, N, C = 1, 1, 256, 4
    chunk_size = 64
    top_k_chunks = 3

    routing_weights = torch.zeros(B, H, N, C)
    routing_weights[..., 0] = 0.90
    routing_weights[..., 1] = 0.60
    routing_weights[..., 2] = 0.30
    routing_weights[..., 3] = 0.10

    top_k_indices = _build_stage1_top_k_per_token_prev_mandatory(
        routing_weights,
        seq_len=N,
        chunk_size=chunk_size,
        num_chunks=C,
        top_k_chunks=top_k_chunks,
    )

    assert top_k_indices[0, 0, 0].tolist() == [-1, -1, -1]
    assert top_k_indices[0, 0, 63].tolist() == [-1, -1, -1]
    assert top_k_indices[0, 0, 64].tolist() == [0, -1, -1]
    assert top_k_indices[0, 0, 127].tolist() == [0, -1, -1]
    assert top_k_indices[0, 0, 128].tolist() == [1, 0, -1]
    assert top_k_indices[0, 0, 191].tolist() == [1, 0, -1]
    assert top_k_indices[0, 0, 192].tolist() == [2, 0, 1]
    assert top_k_indices[0, 0, 255].tolist() == [2, 0, 1]


def test_v17_candidate_defaults_to_triton_backward():
    cand = HierarchicalSparseAttentionV17HISAStrictTriton(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
    )
    assert cand.backward_impl == 'triton'


def test_triton_stage2_builder_matches_eager_reference_on_cuda():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(23)
    B, H, N, hd = 1, 2, 19, 8
    num_chunks = 4
    chunk_size = 5
    top_k_chunks = 2
    top_m = 4
    device = 'cuda'

    Q = torch.randn(B, H, N, hd, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, hd, device=device, dtype=torch.float32)
    pad_len = chunk_size * num_chunks - N
    K_pad = torch.nn.functional.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
    chunk_reps = _compute_chunk_representatives(K_pad, num_chunks)
    routing_weights = _compute_past_only_routing(
        Q,
        chunk_reps,
        seq_len=N,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        hd=hd,
    )
    top_k_indices = _build_stage1_top_k_per_token(
        routing_weights,
        seq_len=N,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        top_k_chunks=top_k_chunks,
    )

    eager_idx = _build_stage2_token_indices_per_token(
        Q,
        K_pad,
        top_k_indices,
        seq_len=N,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=top_m,
    )
    triton_idx = _build_stage2_token_indices_per_token_triton(
        Q,
        K_pad,
        top_k_indices,
        seq_len=N,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=top_m,
    )

    assert triton_idx.dtype == torch.long
    assert torch.equal(eager_idx, triton_idx)
