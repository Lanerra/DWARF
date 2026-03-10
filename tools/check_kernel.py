#!/usr/bin/env python3
"""
🔍 check_kernel.py — Kernel correctness checker for DSQG attention variants.

Catches forward/backward offset mismatch bugs BEFORE training by:
  1. Static AST parsing to verify forward/backward offset consistency
  2. Numerical forward check (Triton kernel vs Python reference)
  3. Numerical backward check (gradient cosine similarity > 0.99)
  4. Softmax sanity check (attention weights sum to ~1.0)

Usage:
  .venv/bin/python3 tools/check_kernel.py --kernel kernels/dsqg_attention_d41_35m.py
  .venv/bin/python3 tools/check_kernel.py --kernel kernels/dsqg_attention_v3.py
  .venv/bin/python3 tools/check_kernel.py  # checks all kernels/dsqg_attention*.py
"""

import ast
import sys
import os
import glob
import argparse
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# AST Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _const_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _get_static_range_arg(node):
    """If node is tl.static_range(N), return N as int. Otherwise None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == 'static_range':
        if node.args:
            return _const_value(node.args[0])
    return None


def _extract_list_constants(node):
    """Extract integer constants from a List or Tuple AST node."""
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            value = _const_value(element)
            if value is not None:
                values.append(value)
        return values
    return None


def _find_functions_by_prefix(tree, prefix):
    """Find top-level FunctionDefs whose names start with prefix."""
    return [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix)
    ]


def _is_name_or_attr(node, name):
    """Check if node is Name(name) or Attribute(*.name)."""
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Attribute):
        return node.attr == name
    return False


# ═══════════════════════════════════════════════════════════════════════════
# AST-based Offset Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_offsets_from_ast(filepath):
    """
    Parse kernel AST and extract offset configuration:
      - Module-level: _SPARSE_LIST, ALL_OFFSETS (range + sparse), assert J
      - Forward kernel: Phase 1 dense count, Phase 2 sparse tuple
      - Backward kernels: delta tuples from _bwd_dq and _bwd_dkdv
    """
    with open(filepath) as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    info = {}

    # ── Module-level: _SPARSE_LIST, ALL_OFFSETS, assert J ──────────────
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == '_SPARSE_LIST' and isinstance(node.value, ast.List):
                    info['sparse_list'] = _extract_list_constants(node.value)
                elif target.id == 'ALL_OFFSETS' and isinstance(node.value, ast.BinOp):
                    left = node.value.left
                    if isinstance(left, ast.Call) and _is_name_or_attr(left.func, 'list'):
                        if left.args and isinstance(left.args[0], ast.Call):
                            range_call = left.args[0]
                            if _is_name_or_attr(range_call.func, 'range') and range_call.args:
                                n = _const_value(range_call.args[0])
                                if n is not None:
                                    info['dense_range_n'] = n

        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            test = node.test
            if (isinstance(test.left, ast.Call) and
                    _is_name_or_attr(test.left.func, 'len') and
                    test.left.args and
                    isinstance(test.left.args[0], ast.Name) and
                    test.left.args[0].id == 'ALL_OFFSETS'):
                for comparator in test.comparators:
                    value = _const_value(comparator)
                    if value is not None:
                        info['assert_j'] = value

    if 'dense_range_n' in info and 'sparse_list' in info:
        info['module_offsets'] = list(range(info['dense_range_n'])) + info['sparse_list']

    # ── Forward kernel: _fwd_* ─────────────────────────────────────────
    forward_functions = _find_functions_by_prefix(tree, '_fwd')
    if forward_functions:
        _extract_forward_phases(forward_functions[0], info)

    # ── Backward kernels: _bwd_dq_*, _bwd_dkdv_* ──────────────────────
    for prefix, key in [('_bwd_dq', 'bwd_dq_deltas'), ('_bwd_dkdv', 'bwd_dkdv_deltas')]:
        backward_functions = _find_functions_by_prefix(tree, prefix)
        if backward_functions:
            deltas = _extract_backward_deltas(backward_functions[0])
            if deltas is not None:
                info[key] = deltas

    return info


def _extract_forward_phases(function_node, info):
    """Extract Phase 1 dense count and Phase 2 sparse tuple from forward kernel."""
    static_range_loops = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.For):
            static_range_argument = _get_static_range_arg(node.iter)
            if static_range_argument is not None:
                static_range_loops.append((node, static_range_argument))

    for loop_node, range_count in static_range_loops:
        sparse_values = None
        for statement in ast.walk(loop_node):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in ('sd', 'sd_'):
                        if isinstance(statement.value, ast.Subscript):
                            if isinstance(statement.value.value, ast.Tuple):
                                sparse_values = _extract_list_constants(statement.value.value)

        if sparse_values is not None:
            info['fwd_sparse_tuple'] = sparse_values
            info['fwd_sparse_count'] = range_count
        elif 'fwd_dense_count' not in info:
            info['fwd_dense_count'] = range_count


def _extract_backward_deltas(function_node):
    """Extract the full delta tuple from a backward kernel function."""
    for node in ast.walk(function_node):
        if isinstance(node, ast.For):
            if _get_static_range_arg(node.iter) is None:
                continue
            for statement in ast.walk(node):
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id == 'delta':
                            if isinstance(statement.value, ast.Subscript):
                                if isinstance(statement.value.value, ast.Tuple):
                                    return _extract_list_constants(statement.value.value)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Check ①: Offset Consistency (AST — no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════

def check_offset_consistency(filepath):
    """THE KEY CHECK: verify forward/backward offset consistency via AST."""
    info = extract_offsets_from_ast(filepath)
    results = []

    module_offsets = info.get('module_offsets')
    assert_j = info.get('assert_j')
    forward_dense = info.get('fwd_dense_count')
    forward_sparse = info.get('fwd_sparse_tuple')
    backward_dq = info.get('bwd_dq_deltas')
    backward_dkdv = info.get('bwd_dkdv_deltas')

    # (a) Module ALL_OFFSETS length matches assert
    if module_offsets is not None and assert_j is not None:
        ok = len(module_offsets) == assert_j
        detail = f'J={assert_j}, range({info["dense_range_n"]}) + {info["sparse_list"]}'
        results.append((
            'ALL_OFFSETS length == assert',
            ok,
            detail if ok else f'len={len(module_offsets)} but assert says J={assert_j}',
        ))
    else:
        results.append(('ALL_OFFSETS extraction', None, 'could not parse ALL_OFFSETS or assert'))

    # (b) Forward dense + sparse == ALL_OFFSETS
    if forward_dense is not None and forward_sparse is not None and module_offsets is not None:
        forward_offsets = list(range(forward_dense)) + forward_sparse
        ok = forward_offsets == module_offsets
        if ok:
            results.append((
                'Forward offsets == ALL_OFFSETS',
                True,
                f'dense=0..{forward_dense - 1} + sparse={forward_sparse}',
            ))
        else:
            results.append((
                'Forward offsets == ALL_OFFSETS',
                False,
                _describe_mismatch(forward_offsets, module_offsets, 'forward', 'ALL_OFFSETS'),
            ))
    else:
        parts = []
        if forward_dense is None:
            parts.append('no dense phase found')
        if forward_sparse is None:
            parts.append('no sparse phase found')
        results.append(('Forward offset extraction', None, '; '.join(parts) if parts else 'N/A'))

    # (c) Backward dQ deltas == ALL_OFFSETS
    if backward_dq is not None and module_offsets is not None:
        ok = backward_dq == module_offsets
        if ok:
            results.append(('Backward dQ deltas == ALL_OFFSETS', True, f'J={len(backward_dq)}'))
        else:
            results.append((
                'Backward dQ deltas == ALL_OFFSETS',
                False,
                _describe_mismatch(backward_dq, module_offsets, 'bwd_dq', 'ALL_OFFSETS'),
            ))
    else:
        results.append(('Backward dQ delta extraction', None, 'no delta tuple found'))

    # (d) Backward dKdV deltas == ALL_OFFSETS
    if backward_dkdv is not None and module_offsets is not None:
        ok = backward_dkdv == module_offsets
        if ok:
            results.append(('Backward dKdV deltas == ALL_OFFSETS', True, f'J={len(backward_dkdv)}'))
        else:
            results.append((
                'Backward dKdV deltas == ALL_OFFSETS',
                False,
                _describe_mismatch(backward_dkdv, module_offsets, 'bwd_dkdv', 'ALL_OFFSETS'),
            ))
    else:
        results.append(('Backward dKdV delta extraction', None, 'no delta tuple found'))

    # (e) Forward sparse tail == backward sparse tail (THE BUG CATCH)
    if forward_sparse is not None and backward_dq is not None:
        backward_tail = backward_dq[len(backward_dq) - len(forward_sparse):]
        ok = forward_sparse == backward_tail
        if ok:
            results.append(('Forward sparse == backward tail', True, f'{forward_sparse}'))
        else:
            results.append((
                'Forward sparse == backward tail',
                False,
                f'forward_sparse={forward_sparse} vs backward_tail={backward_tail}',
            ))

    # (f) dQ deltas == dKdV deltas
    if backward_dq is not None and backward_dkdv is not None:
        ok = backward_dq == backward_dkdv
        if ok:
            results.append(('dQ deltas == dKdV deltas', True, f'J={len(backward_dq)}'))
        else:
            results.append((
                'dQ deltas == dKdV deltas',
                False,
                _describe_mismatch(backward_dq, backward_dkdv, 'dQ', 'dKdV'),
            ))

    return results


def _describe_mismatch(list_a, list_b, name_a, name_b):
    """Describe the mismatch between two offset lists."""
    lines = []
    if len(list_a) != len(list_b):
        lines.append(f'length: {name_a}={len(list_a)} vs {name_b}={len(list_b)}')
    set_a, set_b = set(list_a), set(list_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    if only_a:
        lines.append(f'only in {name_a}: {only_a}')
    if only_b:
        lines.append(f'only in {name_b}: {only_b}')
    if not lines:
        for i, (a, b) in enumerate(zip(list_a, list_b)):
            if a != b:
                lines.append(f'first diff at [{i}]: {name_a}={a} vs {name_b}={b}')
                break
    return '; '.join(lines) if lines else 'unknown mismatch'


# ═══════════════════════════════════════════════════════════════════════════
# Checks ②③: Numerical Forward/Backward (GPU — Triton vs Python reference)
# ═══════════════════════════════════════════════════════════════════════════

def _import_kernel_module(filepath):
    """Dynamically import a kernel module from file path."""
    module_name = Path(filepath).stem
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_callable(module, prefix):
    """Find first callable (non-class) attribute matching prefix."""
    for name in sorted(dir(module)):
        if name.startswith(prefix):
            obj = getattr(module, name)
            if isinstance(obj, type):
                continue
            if callable(obj):
                return obj
    return None


def _cosine_similarity(tensor_a, tensor_b):
    """Cosine similarity between two tensors (flattened)."""
    a = tensor_a.flatten().float()
    b = tensor_b.flatten().float()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


def check_numerical(filepath):
    """Forward/backward correctness + finiteness checks (requires GPU)."""
    import torch

    results = []
    absolute_path = ROOT / filepath

    try:
        module = _import_kernel_module(absolute_path)
    except Exception as exception:
        return [('Module import', None, f'{type(exception).__name__}: {exception}')]

    all_offsets = getattr(module, 'ALL_OFFSETS', None)
    if all_offsets is None:
        return [('Numerical checks', None, 'no ALL_OFFSETS in module')]

    offset_count = len(all_offsets)
    attention_function = _find_callable(module, 'dsqg_attention_')
    reference_function = _find_callable(module, '_reference_')

    if attention_function is None:
        return [('Numerical checks', None, 'no dsqg_attention_* function found')]
    if reference_function is None:
        return [('Numerical checks', None, 'no _reference_* function found')]

    signature = inspect.signature(reference_function)
    parameter_names = list(signature.parameters.keys())

    device = 'cuda'
    configurations = [
        (1, 4, 64, 32, 'tiny'),
        (2, 8, 128, 32, 'small'),
    ]

    forward_max_diff = 0.0
    backward_min_cosine = 1.0
    backward_max_diff = 0.0
    finite_ok = True

    for batch, heads, sequence_length, head_dim, label in configurations:
        torch.manual_seed(42)
        query = torch.randn(batch, heads, sequence_length, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        key = torch.randn(batch, heads, sequence_length, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        value = torch.randn(batch, heads, sequence_length, head_dim, device=device, dtype=torch.bfloat16) * 0.1
        position_bias = torch.randn(offset_count, heads, device=device, dtype=torch.float32) * 0.5

        extra_tensors = {}
        if 'scale_embed' in parameter_names:
            extra_tensors['scale_embed'] = torch.randn(
                offset_count, head_dim, device=device, dtype=torch.float32) * 0.05
        if 'phase_embed' in parameter_names:
            extra_tensors['phase_embed'] = torch.zeros(
                offset_count, heads, device=device, dtype=torch.float32)

        extra_ordered = [extra_tensors[p] for p in parameter_names[4:] if p in extra_tensors]

        # ── Forward correctness ────────────────────────────────────────
        reference_output = reference_function(
            query.detach(), key.detach(), value.detach(), position_bias, *extra_ordered)
        kernel_output = attention_function(
            query.detach().clone(), key.detach().clone(), value.detach().clone(),
            position_bias, *extra_ordered)

        diff = (reference_output.float() - kernel_output.float()).abs().max().item()
        forward_max_diff = max(forward_max_diff, diff)

        if not torch.isfinite(kernel_output).all():
            finite_ok = False

        # ── Backward correctness ───────────────────────────────────────
        def run_backward(function):
            q_ = query.clone().detach().requires_grad_(True)
            k_ = key.clone().detach().requires_grad_(True)
            v_ = value.clone().detach().requires_grad_(True)
            extra_clone = {name: tensor.clone().detach().requires_grad_(True)
                          for name, tensor in extra_tensors.items()}
            extra_clone_ordered = [extra_clone[p] for p in parameter_names[4:] if p in extra_clone]
            function(q_, k_, v_, position_bias, *extra_clone_ordered).sum().backward()
            gradients = {'query': q_.grad, 'key': k_.grad, 'value': v_.grad}
            for name in extra_clone:
                gradients[name] = extra_clone[name].grad
            return gradients

        reference_gradients = run_backward(reference_function)
        kernel_gradients = run_backward(attention_function)

        for gradient_key in reference_gradients:
            reference_gradient = reference_gradients[gradient_key]
            kernel_gradient = kernel_gradients[gradient_key]
            if reference_gradient is None or kernel_gradient is None:
                continue

            abs_diff = (kernel_gradient.float() - reference_gradient.float()).abs().max().item()
            backward_max_diff = max(backward_max_diff, abs_diff)

            cosine = _cosine_similarity(kernel_gradient, reference_gradient).item()
            backward_min_cosine = min(backward_min_cosine, cosine)

            if not torch.isfinite(kernel_gradient).all():
                finite_ok = False

    results.append((
        'Forward correctness (vs reference)',
        forward_max_diff < 0.01,
        f'max_diff={forward_max_diff:.4f} (threshold 0.01)',
    ))
    results.append((
        'Backward correctness (vs reference)',
        backward_min_cosine > 0.99 and backward_max_diff < 0.05,
        f'cosine={backward_min_cosine:.4f} max_diff={backward_max_diff:.4f}',
    ))
    results.append((
        'Output + gradient finiteness',
        finite_ok,
        'no inf/nan' if finite_ok else '⚠ inf/nan detected',
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Check ④: Softmax Sanity (weight sums ≈ 1.0)
# ═══════════════════════════════════════════════════════════════════════════

def check_softmax_sanity(filepath):
    """Verify attention weights sum to ~1.0 using pure-Python reference."""
    import torch
    import torch.nn.functional as F

    info = extract_offsets_from_ast(filepath)
    module_offsets = info.get('module_offsets')
    if module_offsets is None:
        return [('Softmax sanity', None, 'could not extract offsets from AST')]

    offset_count = len(module_offsets)
    max_offset = max(module_offsets)
    device = 'cuda'
    batch, heads, sequence_length, head_dim = 1, 4, max(256, max_offset + 10), 32

    torch.manual_seed(42)
    query = torch.randn(batch, heads, sequence_length, head_dim, device=device) * 0.1
    key = torch.randn(batch, heads, sequence_length, head_dim, device=device) * 0.1
    position_bias = torch.randn(offset_count, heads, device=device) * 0.5
    scale_embed = torch.randn(offset_count, head_dim, device=device) * 0.05
    scale_factor = head_dim ** -0.5

    key_padded = F.pad(key, (0, 0, max_offset, 0))
    scores = torch.full(
        (batch, heads, sequence_length, offset_count), float('-inf'), device=device)

    position_indices = torch.arange(sequence_length, device=device)
    for j_index, delta in enumerate(module_offsets):
        key_at_offset = key_padded[:, :, max_offset - delta:max_offset - delta + sequence_length, :]
        content_score = (query * key_at_offset).sum(-1) * scale_factor
        bias_score = position_bias[j_index].view(1, heads, 1).expand(batch, heads, sequence_length)
        scale_score = (query * scale_embed[j_index].view(1, 1, 1, head_dim)).sum(-1) * scale_factor
        raw = content_score + bias_score + scale_score
        valid = position_indices >= delta
        scores[:, :, :, j_index] = torch.where(
            valid.view(1, 1, sequence_length), raw,
            torch.tensor(float('-inf'), device=device))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    weight_sums = weights.sum(dim=-1)

    valid_mask = weight_sums > 0.5
    if not valid_mask.any():
        return [('Softmax sanity', None, 'no valid positions found')]

    sums_at_valid = weight_sums[valid_mask]
    minimum_sum = sums_at_valid.min().item()
    maximum_sum = sums_at_valid.max().item()
    has_inf = torch.isinf(weights).any().item()

    ok = minimum_sum > 0.99 and maximum_sum < 1.01 and not has_inf
    detail = f'weight sums ∈ [{minimum_sum:.4f}, {maximum_sum:.4f}]'
    if has_inf:
        detail += ' ⚠ inf in weights!'

    return [('Softmax weight sums ≈ 1.0', ok, detail)]


# ═══════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_results(filepath, results):
    """Print formatted check results for a kernel file."""
    print()
    print('═' * 76)
    print(f'🔍 {filepath}')
    print('═' * 76)

    name_width = max((len(r[0]) for r in results), default=30) + 1

    print(f' {"Check":<{name_width}} │ {"Result":^8} │ Details')
    print(f'{"─" * (name_width + 1)}┼{"─" * 10}┼{"─" * 52}')

    for name, passed, detail in results:
        if passed is True:
            status = 'PASS ✓'
        elif passed is False:
            status = 'FAIL ✗'
        else:
            status = 'SKIP ─'
        print(f' {name:<{name_width}} │ {status:^8} │ {detail}')

    print('═' * 76)

    failures = [r for r in results if r[1] is False]
    if failures:
        print(f'  ⚠ {len(failures)} check(s) FAILED')
        for name, _, detail in failures:
            print(f'    ✗ {name}: {detail}')
    else:
        skipped = sum(1 for r in results if r[1] is None)
        passed = sum(1 for r in results if r[1] is True)
        print(f'  ✓ {passed} passed, {skipped} skipped')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='🔍 Kernel correctness checker for DSQG attention variants')
    parser.add_argument('--kernel', type=str, default=None,
                        help='Specific kernel file to check')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Skip GPU-based numerical checks (AST only)')
    arguments = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'kernels'))
    sys.path.insert(0, str(ROOT))

    if arguments.kernel:
        kernel_files = [arguments.kernel]
    else:
        kernel_files = sorted(glob.glob('kernels/dsqg_attention*.py'))

    if not kernel_files:
        print('No kernel files found matching kernels/dsqg_attention*.py')
        sys.exit(1)

    print(f'🔍 Checking {len(kernel_files)} kernel file(s)...')

    all_pass = True
    for filepath in kernel_files:
        results = []

        # ① Offset consistency (AST — always runs, no GPU)
        results.extend(check_offset_consistency(filepath))

        # ②③④ Numerical checks (GPU)
        if not arguments.no_gpu:
            try:
                results.extend(check_numerical(filepath))
            except Exception as exception:
                results.append(('Numerical checks', None,
                               f'{type(exception).__name__}: {exception}'))

            try:
                results.extend(check_softmax_sanity(filepath))
            except Exception as exception:
                results.append(('Softmax sanity', None,
                               f'{type(exception).__name__}: {exception}'))

        print_results(filepath, results)

        if any(r[1] is False for r in results):
            all_pass = False

    print()
    if all_pass:
        print('✓ All kernel checks passed')
    else:
        print('✗ Some kernel checks FAILED — fix before training!')
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
