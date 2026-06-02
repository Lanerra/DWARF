import os
import pathlib
import sys

import torch

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
for _d in [_project_root, os.path.join(_project_root, 'kernels')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import hierarchical_sparse_attn_v17_hisa_strict_triton as hisa_v17

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
    _strict_hisa_module_forward,
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


def test_v17_candidate_routing_log_bias_is_train_only():
    _, baseline = _make_pair(seed=14)
    x = torch.randn(1, 130, 16)

    biased = HierarchicalSparseAttentionV17HISAStrictTriton(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
        routing_log_bias_scale=1.0,
    )
    biased.load_state_dict(baseline.state_dict())

    with torch.no_grad():
        baseline.eval()
        y_baseline = baseline(x)

        biased.train()
        y_train = biased(x)

        biased.eval()
        y_eval = biased(x)

    eval_diff = (y_eval - y_baseline).abs().max().item()
    train_diff = (y_train - y_eval).abs().max().item()
    assert eval_diff < 1e-6
    assert train_diff > 1e-6


def test_v17_stage1_entropy_stats_normalize_by_available_past_choices():
    routing_weights = torch.zeros(1, 1, 16, 4)
    routing_weights[..., 4:8, 0] = 1.0
    routing_weights[..., 8:12, 0] = 0.5
    routing_weights[..., 8:12, 1] = 0.5
    routing_weights[..., 12:16, 0] = 0.7
    routing_weights[..., 12:16, 1] = 0.2
    routing_weights[..., 12:16, 2] = 0.1

    raw_entropy, norm_entropy, floor_penalty = hisa_v17._compute_stage1_routing_entropy_stats(
        routing_weights,
        seq_len=16,
        chunk_size=4,
        num_chunks=4,
        entropy_target=0.8,
    )

    expected_chunk2_raw = torch.log(torch.tensor(2.0))
    p = torch.tensor([0.7, 0.2, 0.1])
    expected_chunk3_raw = -(p * p.log()).sum()
    expected_raw = (expected_chunk2_raw + expected_chunk3_raw) / 2.0
    expected_norm = (1.0 + (expected_chunk3_raw / torch.log(torch.tensor(3.0)))) / 2.0
    expected_penalty = torch.relu(torch.tensor(0.8) - expected_norm).square()

    assert torch.isclose(raw_entropy, expected_raw, atol=1e-6)
    assert torch.isclose(norm_entropy, expected_norm, atol=1e-6)
    assert torch.isclose(floor_penalty, expected_penalty, atol=1e-6)


def test_v17_candidate_exposes_entropy_regularizer_loss_only_in_training():
    torch.manual_seed(15)
    x = torch.randn(1, 130, 16, requires_grad=True)
    mod = HierarchicalSparseAttentionV17HISAStrictTriton(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
        routing_entropy_target=0.9,
        routing_entropy_reg_scale=1.5,
    ).train()

    y = mod(x)
    assert y.shape == x.shape
    assert 0.0 <= float(mod._routing_entropy_norm) <= 1.0
    assert mod._routing_entropy_reg_loss.requires_grad
    loss = y.square().mean() + mod._routing_entropy_reg_loss
    loss.backward()
    assert mod.W_q.weight.grad is not None
    assert mod.W_k.weight.grad is not None

    mod.eval()
    with torch.no_grad():
        _ = mod(x.detach())
    assert float(mod._routing_entropy_reg_loss) == 0.0


def test_v17_candidate_cuda_bias_backward_matches_eager_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(31)
    x = torch.randn(1, 130, 16, device='cuda', requires_grad=True)
    mod = HierarchicalSparseAttentionV17HISAStrictTriton(
        D=16,
        H=2,
        hd=8,
        num_chunks=4,
        top_k_chunks=2,
        hisa_top_m_tokens=4,
        routing_log_bias_scale=1.0,
        backward_impl='triton',
    ).train().cuda()

    y = mod(x)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert mod.W_q.weight.grad is not None
    assert mod.W_k.weight.grad is not None
    assert mod.W_v.weight.grad is not None
    assert mod.W_o.weight.grad is not None
    got = {
        'x': x.grad.detach().clone(),
        'W_q': mod.W_q.weight.grad.detach().clone(),
        'W_k': mod.W_k.weight.grad.detach().clone(),
        'W_v': mod.W_v.weight.grad.detach().clone(),
        'W_o': mod.W_o.weight.grad.detach().clone(),
    }

    x_ref = x.detach().clone().requires_grad_(True)
    W_q = mod.W_q.weight.detach().clone().requires_grad_(True)
    W_k = mod.W_k.weight.detach().clone().requires_grad_(True)
    W_v = mod.W_v.weight.detach().clone().requires_grad_(True)
    W_o = mod.W_o.weight.detach().clone().requires_grad_(True)
    y_ref = _strict_hisa_module_forward(
        x_ref,
        W_q,
        W_k,
        W_v,
        W_o,
        num_heads=mod.num_heads,
        hd=mod.hd,
        num_chunks_base=mod.num_chunks,
        top_k_chunks=mod.top_k_chunks,
        hisa_top_m_tokens=mod.hisa_top_m_tokens,
        chunk_size=mod.chunk_size,
        sparse_block_size=mod.sparse_block_size,
        routing_log_bias_scale=mod.routing_log_bias_scale,
        training=True,
        use_fused=False,
    )
    loss_ref = y_ref.square().mean()
    loss_ref.backward()
    assert x_ref.grad is not None
    assert W_q.grad is not None
    assert W_k.grad is not None
    assert W_v.grad is not None
    assert W_o.grad is not None
    want = {
        'x': x_ref.grad.detach(),
        'W_q': W_q.grad.detach(),
        'W_k': W_k.grad.detach(),
        'W_v': W_v.grad.detach(),
        'W_o': W_o.grad.detach(),
    }

    for name in got:
        diff = (got[name] - want[name]).abs().max().item()
        assert diff < 1e-5, f"{name} grad mismatch: {diff}"


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
