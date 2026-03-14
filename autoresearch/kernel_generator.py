#!/usr/bin/env python3
"""
🔧 kernel_generator.py — Generates valid DSQG Triton kernels from config.

Eliminates forward/backward tuple mismatch bugs by deriving all
config-dependent values from a single (dense_width, sparse_list) spec.

The 7 substitutions (ALL derived from dense_width + sparse_list):
  1. _SPARSE_LIST = {sparse_list}
  2. ALL_OFFSETS = list(range({dense_width+1})) + _SPARSE_LIST
  3. assert len(ALL_OFFSETS) == {total}
  4. Forward Phase 1: tl.static_range({dense_width+1})
  5. Forward Phase 2: tl.static_range({len(sparse_list)}) + tuple + pbi base
  6. Backward dQ: tl.static_range({total}) + delta tuple
  7. Backward dKdV: tl.static_range({total}) + delta tuple

Plus: shape assertions, scale_embed param size, reference function padding.
"""

import re
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _format_delta_tuple(dense_width, sparse_list):
    """Format the backward delta tuple: 22 dense values per line, sparse on own line."""
    dense_offsets = list(range(dense_width + 1))
    continuation_indent = ' ' * 17

    dense_lines = []
    for i in range(0, len(dense_offsets), 22):
        chunk = dense_offsets[i:i + 22]
        dense_lines.append(', '.join(str(x) for x in chunk))

    sparse_str = ', '.join(str(x) for x in sparse_list)

    parts = []
    for i, line in enumerate(dense_lines):
        if i == 0:
            parts.append(f'delta = ({line},')
        else:
            parts.append(f'{continuation_indent}{line},')
    parts.append(f'{continuation_indent}{sparse_str})[i]')

    return '\n'.join(parts)


def generate_kernel(dense_width, sparse_list, output_path,
                    base_template="kernels/dsqg_attention_d41_35m.py"):
    """
    Generate a valid DSQG attention kernel for given (dense_width, sparse_list).

    Reads base_template, substitutes all config-dependent spots, writes output_path.
    Returns the output path string.
    """
    assert len(sparse_list) > 0, "sparse_list must be non-empty"
    assert all(isinstance(x, int) and x > 0 for x in sparse_list), \
        "sparse offsets must be positive integers"
    assert dense_width > 0, "dense_width must be positive"

    template_path = ROOT / base_template
    assert template_path.exists(), f"Template not found: {template_path}"

    dense_count = dense_width + 1
    total = dense_count + len(sparse_list)
    sparse_str = ', '.join(str(x) for x in sparse_list)
    max_offset = max(dense_width, max(sparse_list))

    with open(template_path) as f:
        source = f.read()

    # ── 1. _SPARSE_LIST ──────────────────────────────────────────────────────
    source = re.sub(
        r'_SPARSE_LIST\s*=\s*\[[\d,\s]+\]',
        f'_SPARSE_LIST = [{sparse_str}]',
        source)

    # ── 2. ALL_OFFSETS range ─────────────────────────────────────────────────
    source = re.sub(
        r'list\(range\(\d+\)\)\s*\+\s*_SPARSE_LIST',
        f'list(range({dense_count})) + _SPARSE_LIST',
        source)

    # ── 3. assert len(ALL_OFFSETS) == J ──────────────────────────────────────
    source = re.sub(
        r'assert\s+len\(ALL_OFFSETS\)\s*==\s*\d+',
        f'assert len(ALL_OFFSETS) == {total}',
        source)

    # ── 4. Forward Phase 1: for d in tl.static_range(N) ─────────────────────
    source = re.sub(
        r'for d in tl\.static_range\(\d+\):',
        f'for d in tl.static_range({dense_count}):',
        source)

    # ── 5. Forward Phase 2: sparse loop + tuple + pbi ────────────────────────
    source = re.sub(
        r'for si in tl\.static_range\(\d+\):',
        f'for si in tl.static_range({len(sparse_list)}):',
        source)
    source = re.sub(
        r'sd\s*=\s*\([\d,\s]+\)\[si\]',
        f'sd  = ({sparse_str})[si]',
        source)
    source = re.sub(
        r'pbi\s*=\s*\d+\s*\+\s*si',
        f'pbi = {dense_count} + si',
        source)

    # ── 6 & 7. Backward loops: static_range(total) + delta tuple ─────────────
    source = re.sub(
        r'for i in tl\.static_range\(\d+\):',
        f'for i in tl.static_range({total}):',
        source)
    delta_str = _format_delta_tuple(dense_width, sparse_list)
    source = re.sub(
        r'delta = \([\d,\s]+\)\[i\]',
        delta_str,
        source)

    # ── Shape assertions in autograd ─────────────────────────────────────────
    source = re.sub(
        r'assert pos_bias\.shape\s*==\s*\(\d+,\s*H\)',
        f'assert pos_bias.shape    == ({total}, H)',
        source)
    source = re.sub(
        r'assert scale_embed\.shape\s*==\s*\(\d+,\s*HD\)',
        f'assert scale_embed.shape == ({total}, HD)',
        source)

    # ── scale_embed / pos_bias tensor sizes ──────────────────────────────────
    source = re.sub(r'torch\.zeros\(\d+,\s*HD\b', f'torch.zeros({total}, HD', source)
    source = re.sub(r'torch\.randn\(\d+,\s*H\b', f'torch.randn({total}, H', source)
    source = re.sub(r'torch\.randn\(\d+,\s*HD\b', f'torch.randn({total}, HD', source)

    # ── Reference function padding ───────────────────────────────────────────
    source = re.sub(
        r"F\.pad\(k\.float\(\),\s*\(0,\s*0,\s*\d+,\s*0\)\)",
        f'F.pad(k.float(), (0, 0, {max_offset}, 0))',
        source)
    source = re.sub(
        r"F\.pad\(v\.float\(\),\s*\(0,\s*0,\s*\d+,\s*0\)\)",
        f'F.pad(v.float(), (0, 0, {max_offset}, 0))',
        source)
    source = re.sub(
        r'gi\s*=\s*\d+\s*-\s*off\[None,:\]\s*\+\s*ni\[:,None\]',
        f'gi  = {max_offset} - off[None,:] + ni[:,None]',
        source)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(source)

    return str(output_file)


def validate_kernel(kernel_path):
    """Run check_kernel.py AST validation on the generated kernel. Returns (passed, details)."""
    check_script = ROOT / 'tools' / 'check_kernel.py'
    assert check_script.exists(), f"check_kernel.py not found: {check_script}"

    result = subprocess.run(
        [sys.executable, str(check_script), '--kernel', str(kernel_path), '--no-gpu'],
        capture_output=True, text=True, cwd=str(ROOT))

    passed = result.returncode == 0
    details = (result.stdout + result.stderr).strip()
    return passed, details


def generate_and_validate(dense_width, sparse_list, output_path,
                          base_template="kernels/dsqg_attention_d41_35m.py"):
    """Generate kernel and run AST validation. Returns (path, passed, details)."""
    path = generate_kernel(dense_width, sparse_list, output_path, base_template)
    passed, details = validate_kernel(path)
    return path, passed, details


def main():
    parser = argparse.ArgumentParser(
        description='🔧 Generate DSQG attention kernel from (dense_width, sparse_list)')
    parser.add_argument('--dense', type=int, required=True,
                        help='Dense window width (e.g., 41 for 14M, 48 for 35M)')
    parser.add_argument('--sparse', type=str, required=True,
                        help='Sparse offsets, comma-separated (e.g., 48,128,384)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output kernel file path')
    parser.add_argument('--template', type=str,
                        default='kernels/dsqg_attention_d41_35m.py',
                        help='Base template kernel (default: d41_35m)')
    parser.add_argument('--validate', action='store_true',
                        help='Run check_kernel AST validation after generation')
    arguments = parser.parse_args()

    sparse_list = [int(x.strip()) for x in arguments.sparse.split(',')]
    dense_count = arguments.dense + 1
    total = dense_count + len(sparse_list)

    print(f'🔧 Generating DSQG kernel:')
    print(f'   dense_width = {arguments.dense}  (offsets 0..{arguments.dense})')
    print(f'   sparse_list = {sparse_list}')
    print(f'   total offsets = {total}  ({dense_count} dense + {len(sparse_list)} sparse)')
    print(f'   template = {arguments.template}')
    print(f'   output = {arguments.out}')

    if arguments.validate:
        path, passed, details = generate_and_validate(
            arguments.dense, sparse_list, arguments.out, arguments.template)
        print()
        print(details)
        if passed:
            print(f'\n✓ Kernel generated and validated: {path}')
        else:
            print(f'\n✗ Kernel generated but validation FAILED: {path}')
            sys.exit(1)
    else:
        path = generate_kernel(
            arguments.dense, sparse_list, arguments.out, arguments.template)
        print(f'\n✓ Kernel generated: {path}')


if __name__ == '__main__':
    main()
