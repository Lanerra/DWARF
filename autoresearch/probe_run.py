#!/usr/bin/env python3
"""
🧪 probe_run.py — Fixed-budget training probe for the autoresearch loop.

Generates a DSQG kernel from config, runs N training steps, evaluates,
and reports metrics in Karpathy autoresearch TSV format.

Usage:
  python autoresearch/probe_run.py \
    --dense 41 --sparse 48,128,384 \
    --steps 2000 \
    --model-size 14m \
    --tag my-experiment \
    --out autoresearch/results.tsv

Output format (tab-separated):
  commit  val_ppl  passkey_mean  peak_vram_gb  status  description
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_generator import generate_and_validate

MODEL_CONFIGS = {
    '14m': {
        'base_script': 'train/train_2048_14m_d41s3.py',
        'embedding_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ffn_dim': 1024,
        'full_attention_layer': 5,
    },
    '35m': {
        'base_script': 'train/train_2048_35m_d41.py',
        'embedding_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'ffn_dim': 2048,
        'full_attention_layer': 5,
    },
}

PROBE_PASSKEY_DISTANCES = [64, 128, 256, 512, 1024, 1536]
PROBE_PASSKEY_TRIALS = 5
BATCH_SIZE = 8
GRAD_ACCUM = 4


def _get_git_commit():
    """Get short git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=str(ROOT))
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except FileNotFoundError:
        return 'unknown'


def _build_probe_script(base_script_path, kernel_module_name,
                        dense_width, sparse_list, steps, tag, epochs=1):
    """Create a modified training script for the probe run."""
    with open(base_script_path) as f:
        script = f.read()

    total = dense_width + 1 + len(sparse_list)
    sparse_str = ', '.join(str(x) for x in sparse_list)
    train_sequences = steps * BATCH_SIZE * GRAD_ACCUM

    # ── Kernel import ────────────────────────────────────────────────────────
    script = re.sub(
        r'from dsqg_attention_\w+ import dsqg_attention_v3',
        f'from {kernel_module_name} import dsqg_attention_v3',
        script)

    # ── Fix paths — probe lives in autoresearch/runs/, not train/ ───────────
    # Kernel path: original uses parent.parent/kernels (works from train/)
    # Probe needs parent.parent.parent/kernels (two levels under ROOT)
    script = script.replace(
        "_pl.Path(__file__).parent.parent / 'kernels'",
        "_pl.Path(__file__).parent.parent.parent / 'kernels'"
    )
    # Tokenizer search: _script_dir is used to find results/tokenizer
    # Pin it to the train/ directory so all existing candidate paths resolve
    train_dir = str(ROOT / 'train')
    script = script.replace(
        '_script_dir     = os.path.dirname(os.path.abspath(__file__))',
        f"_script_dir     = {repr(train_dir)}  # pinned by probe_run.py"
    )

    # ── Training budget ──────────────────────────────────────────────────────
    script = re.sub(r'NUM_EPOCHS\s*=\s*\d+', f'NUM_EPOCHS      = {epochs}', script)
    if epochs > 1:
        # Multi-epoch: use full dataset each epoch, no sequence cap
        script = re.sub(
            r'MAX_TRAIN_SEQS\s*=\s*\S+',
            'MAX_TRAIN_SEQS  = None',
            script)
    else:
        script = re.sub(
            r'MAX_TRAIN_SEQS\s*=\s*\S+',
            f'MAX_TRAIN_SEQS  = {train_sequences}',
            script)

    # ── Passkey abbreviation ─────────────────────────────────────────────────
    distances_str = repr(PROBE_PASSKEY_DISTANCES)
    script = re.sub(
        r'PASSKEY_DISTANCES\s*=\s*\[[\d,\s]+\]',
        f'PASSKEY_DISTANCES = {distances_str}',
        script)
    script = re.sub(
        r'PASSKEY_TRIALS\s*=\s*\d+',
        f'PASSKEY_TRIALS    = {PROBE_PASSKEY_TRIALS}',
        script)

    # ── Save paths ───────────────────────────────────────────────────────────
    script = re.sub(
        r"SAVE_DIR\s*=\s*'[^']+'",
        f"SAVE_DIR    = 'autoresearch/checkpoints/{tag}'",
        script)
    script = re.sub(
        r"RESULT_FILE\s*=\s*'[^']+'",
        f"RESULT_FILE = 'autoresearch/results/{tag}.json'",
        script)

    # ── Offset config ────────────────────────────────────────────────────────
    script = re.sub(
        r'_DENSE_LOCAL_W\s*=\s*\d+',
        f'_DENSE_LOCAL_W     = {dense_width}',
        script)
    script = re.sub(
        r'_DYADIC_LONG_RANGE\s*=\s*\[[\d,\s]+\]',
        f'_DYADIC_LONG_RANGE = [{sparse_str}]',
        script)
    script = re.sub(
        r'assert len\(_COND_N_OFFSETS\)\s*==\s*\d+',
        f'assert len(_COND_N_OFFSETS) == {total}',
        script)

    # ── scale_embed size in model ────────────────────────────────────────────
    script = re.sub(
        r'torch\.zeros\(\d+,\s*HD\b',
        f'torch.zeros({total}, HD',
        script)

    # ── Peak VRAM tracking ───────────────────────────────────────────────────
    script = script.replace(
        "'per_epoch':               per_epoch_results,",
        "'per_epoch':               per_epoch_results,\n"
        "        'peak_vram_gb':            "
        "torch.cuda.max_memory_allocated() / 1e9 "
        "if torch.cuda.is_available() else 0.0,")

    return script


def _append_to_tsv(row, tsv_path):
    """Append a result row to the TSV file, creating header if needed."""
    columns = ['commit', 'val_ppl', 'passkey_mean', 'peak_vram_gb', 'status', 'description']
    tsv_file = Path(tsv_path)
    tsv_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not tsv_file.exists() or tsv_file.stat().st_size == 0
    with open(tsv_file, 'a') as f:
        if write_header:
            f.write('\t'.join(columns) + '\n')
        values = [str(row.get(column, '')) for column in columns]
        f.write('\t'.join(values) + '\n')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='🧪 Fixed-budget training probe for autoresearch loop')
    parser.add_argument('--dense', type=int, required=True,
                        help='Dense window width')
    parser.add_argument('--sparse', type=str, required=True,
                        help='Sparse offsets, comma-separated')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Training steps per epoch cap (default: 2000; ignored when --epochs > 1)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of full training epochs (default: 1; when > 1, uses full dataset each epoch)')
    parser.add_argument('--model-size', type=str, choices=['14m', '35m'],
                        default='14m', help='Model size (default: 14m)')
    parser.add_argument('--tag', type=str, required=True,
                        help='Experiment tag for naming')
    parser.add_argument('--out', type=str, default='autoresearch/results.tsv',
                        help='Output TSV file (default: autoresearch/results.tsv)')
    parser.add_argument('--template', type=str,
                        default='kernels/dsqg_attention_d41_35m.py',
                        help='Base kernel template')
    arguments = parser.parse_args()

    sparse_list = [int(x.strip()) for x in arguments.sparse.split(',')]
    config = MODEL_CONFIGS[arguments.model_size]
    dense_count = arguments.dense + 1
    total = dense_count + len(sparse_list)
    description = (
        f'd{arguments.dense}_s{"_".join(str(x) for x in sparse_list)}'
        f'_{arguments.model_size}')

    print('=' * 70)
    print(f'  🧪 Probe: {arguments.tag}')
    print(f'  dense={arguments.dense}  sparse={sparse_list}  total={total}')
    print(f'  model={arguments.model_size}  steps={arguments.steps}  epochs={arguments.epochs}')
    print('=' * 70)

    # ── 1. Generate and validate kernel ──────────────────────────────────────
    safe_tag = arguments.tag.replace('-', '_')
    kernel_path = str(ROOT / 'kernels' / f'dsqg_probe_{safe_tag}.py')
    print(f'\n  Generating kernel -> {kernel_path}')

    path, passed, details = generate_and_validate(
        arguments.dense, sparse_list, kernel_path, arguments.template)
    print(details)

    if not passed:
        print('\n  ✗ Kernel validation FAILED')
        _append_to_tsv({
            'commit': _get_git_commit(),
            'val_ppl': '',
            'passkey_mean': '',
            'peak_vram_gb': '',
            'status': 'crash',
            'description': f'{description} kernel_validation_failed',
        }, arguments.out)
        sys.exit(1)

    print(f'  ✓ Kernel validated')

    # ── 2. Build probe training script ───────────────────────────────────────
    kernel_module_name = Path(kernel_path).stem
    base_script = ROOT / config['base_script']
    probe_script = _build_probe_script(
        base_script, kernel_module_name,
        arguments.dense, sparse_list, arguments.steps, arguments.tag,
        epochs=arguments.epochs)

    probe_script_path = ROOT / 'autoresearch' / 'runs' / f'{safe_tag}_probe.py'
    probe_script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_script_path, 'w') as f:
        f.write(probe_script)

    print(f'  Probe script -> {probe_script_path}')

    # ── 3. Run training ─────────────────────────────────────────────────────
    result_json = ROOT / 'autoresearch' / 'results' / f'{safe_tag}.json'
    (ROOT / 'autoresearch' / 'results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'autoresearch' / 'checkpoints' / safe_tag).mkdir(
        parents=True, exist_ok=True)

    print(f'\n  Running training ({arguments.steps} steps)...')

    train_result = subprocess.run(
        [sys.executable, '-u', str(probe_script_path)],
        cwd=str(ROOT),
        env={
            **os.environ,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        })

    if train_result.returncode != 0:
        print(f'\n  ✗ Training crashed (exit code {train_result.returncode})')
        _append_to_tsv({
            'commit': _get_git_commit(),
            'val_ppl': '',
            'passkey_mean': '',
            'peak_vram_gb': '',
            'status': 'crash',
            'description': f'{description} training_crashed',
        }, arguments.out)
        sys.exit(1)

    # ── 4. Parse results ────────────────────────────────────────────────────
    if not result_json.exists():
        print(f'\n  ✗ Result file not found: {result_json}')
        _append_to_tsv({
            'commit': _get_git_commit(),
            'val_ppl': '',
            'passkey_mean': '',
            'peak_vram_gb': '',
            'status': 'crash',
            'description': f'{description} no_results_file',
        }, arguments.out)
        sys.exit(1)

    with open(result_json) as f:
        results = json.load(f)

    val_ppl = results.get('final_test_ppl', 0.0)
    passkey_mean = results.get('final_passkey_mean', 0.0)
    peak_vram = results.get('peak_vram_gb', 0.0)

    # ── 5. Write TSV ────────────────────────────────────────────────────────
    _append_to_tsv({
        'commit': _get_git_commit(),
        'val_ppl': f'{val_ppl:.3f}',
        'passkey_mean': f'{passkey_mean:.4f}',
        'peak_vram_gb': f'{peak_vram:.1f}',
        'status': 'ok',
        'description': description,
    }, arguments.out)

    # ── 6. Print summary ────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print(f'  🧪 Probe complete: {arguments.tag}')
    print(f'  val_ppl      = {val_ppl:.3f}')
    print(f'  passkey_mean = {passkey_mean:.4f}  ({passkey_mean * 100:.1f}%)')
    print(f'  peak_vram    = {peak_vram:.1f} GB')
    print(f'  status       = ok')
    print(f'  results      -> {result_json}')
    print(f'  tsv          -> {arguments.out}')
    print('=' * 70)


if __name__ == '__main__':
    main()
