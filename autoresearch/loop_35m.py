#!/usr/bin/env python3
"""
🔄 loop_35m.py — Tier 2 autoresearch loop for RunPod.

Pure Python. No LLM, no API key.
Reads candidates_35m.json, runs each pending probe at 35M scale,
updates the JSON in-place (crash-safe), fires Discord webhooks.

Usage:
  python autoresearch/loop_35m.py \
    --candidates autoresearch/candidates_35m.json \
    --steps 2000 \
    --webhook https://discord.com/api/webhooks/... \
    --out autoresearch/results_35m.tsv
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent


def _timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _log(message):
    print(f'[{_timestamp()}] {message}', flush=True)


def _load_candidates(path):
    with open(path) as f:
        return json.load(f)


def _save_candidates(data, path):
    """Atomic write: write to .tmp then rename (crash-safe)."""
    temporary = Path(str(path) + '.tmp')
    with open(temporary, 'w') as f:
        json.dump(data, f, indent=2)
    temporary.rename(path)


def _post_webhook(webhook_url, content):
    payload = json.dumps({'content': content}).encode('utf-8')
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST')
    try:
        urllib.request.urlopen(request, timeout=10)
    except Exception as error:
        _log(f'  ⚠  Webhook failed: {error}')


def _format_sparse(sparse_list):
    return '[' + ','.join(str(x) for x in sparse_list) + ']'


def _run_probe(candidate, steps, results_tsv):
    """Run probe_run.py as subprocess. Returns (success, result_json_path)."""
    dense = candidate['dense_width']
    sparse = candidate['sparse_list']
    sparse_str = ','.join(str(x) for x in sparse)
    tag = f'35m_d{dense}_s{"_".join(str(x) for x in sparse)}'

    command = [
        sys.executable, str(ROOT / 'autoresearch' / 'probe_run.py'),
        '--dense', str(dense),
        '--sparse', sparse_str,
        '--steps', str(steps),
        '--model-size', '35m',
        '--tag', tag,
        '--out', str(results_tsv),
    ]

    _log(f'  Running: {" ".join(command)}')

    result = subprocess.run(command, cwd=str(ROOT))
    result_json = ROOT / 'autoresearch' / 'results' / f'{tag}.json'

    return result.returncode == 0, result_json


def _parse_probe_results(result_json_path):
    """Parse the results JSON from probe_run.py."""
    with open(result_json_path) as f:
        data = json.load(f)

    return {
        'val_ppl': data.get('final_test_ppl', 0.0),
        'passkey_mean': data.get('final_passkey_mean', 0.0),
        'passkey_by_distance': data.get('final_passkey_by_d', {}),
    }


def _compute_status(candidate, baseline):
    """Determine keep/discard based on baseline comparison."""
    ppl_35m = candidate['35m_val_ppl']
    passkey_35m = candidate['35m_passkey_mean']
    baseline_ppl = baseline.get('14m_val_ppl', 999.0)
    baseline_passkey = baseline.get('14m_passkey_mean', 0.0)

    ppl_better = ppl_35m < baseline_ppl + 2.0
    passkey_acceptable = passkey_35m >= baseline_passkey - 0.15

    if ppl_better and passkey_acceptable:
        return 'keep'
    return 'discard'


def _classify_config(sparse_list):
    """Classify a sparse config into analysis classes."""
    classes = []
    if all(x <= 200 for x in sparse_list):
        classes.append('sparse_all_leq_200')
    if any(x >= 384 for x in sparse_list):
        classes.append('sparse_includes_384')
    if any(x >= 768 for x in sparse_list):
        classes.append('sparse_includes_768_plus')
    if len(sparse_list) <= 3:
        classes.append('sparse_3_or_fewer')
    if len(sparse_list) >= 5:
        classes.append('sparse_5_or_more')
    return classes


def _compute_rank_concordance(pairs_14m, pairs_35m):
    """
    Kendall-like rank concordance between 14M and 35M orderings.
    pairs = list of (value, index) for each candidate.
    Returns concordance in [0, 1].
    """
    if len(pairs_14m) < 2:
        return None

    rank_14m = [x[1] for x in sorted(pairs_14m)]
    rank_35m = [x[1] for x in sorted(pairs_35m)]

    concordant = 0
    total = 0
    for i in range(len(rank_14m)):
        for j in range(i + 1, len(rank_14m)):
            idx_i_14m = rank_14m.index(i)
            idx_j_14m = rank_14m.index(j)
            idx_i_35m = rank_35m.index(i)
            idx_j_35m = rank_35m.index(j)
            total += 1
            if (idx_i_14m - idx_j_14m) * (idx_i_35m - idx_j_35m) > 0:
                concordant += 1

    return concordant / total if total > 0 else None


def _build_calibration_update(data, run_tag):
    """Build calibration_update.json from completed candidates."""
    candidates = data['candidates']
    completed = [c for c in candidates if c['35m_status'] in ('keep', 'discard')]
    crashed = [c for c in candidates if c['35m_status'] == 'crash']

    if not completed:
        return None

    ppl_deltas = [c['transfer_delta_ppl'] for c in completed
                  if c['transfer_delta_ppl'] is not None]
    passkey_deltas = [c['transfer_delta_passkey'] for c in completed
                      if c['transfer_delta_passkey'] is not None]

    ppl_14m_35m_pairs = [
        (c['14m_val_ppl'], c['35m_val_ppl'], i)
        for i, c in enumerate(completed)
        if c['14m_val_ppl'] is not None and c['35m_val_ppl'] is not None
    ]

    passkey_14m_35m_pairs = [
        (c['14m_passkey_mean'], c['35m_passkey_mean'], i)
        for i, c in enumerate(completed)
        if c['14m_passkey_mean'] is not None and c['35m_passkey_mean'] is not None
    ]

    ppl_concordance = _compute_rank_concordance(
        [(p[0], p[2]) for p in ppl_14m_35m_pairs],
        [(p[1], p[2]) for p in ppl_14m_35m_pairs],
    )

    passkey_concordance = _compute_rank_concordance(
        [(p[0], p[2]) for p in passkey_14m_35m_pairs],
        [(p[1], p[2]) for p in passkey_14m_35m_pairs],
    )

    class_data = {}
    for candidate in completed:
        classes = _classify_config(candidate['sparse_list'])
        for config_class in classes:
            if config_class not in class_data:
                class_data[config_class] = []
            class_data[config_class].append(candidate)

    class_findings = []
    for config_class, members in sorted(class_data.items()):
        class_ppl_pairs = [
            (c['14m_val_ppl'], c['35m_val_ppl'], i)
            for i, c in enumerate(members)
            if c['14m_val_ppl'] is not None and c['35m_val_ppl'] is not None
        ]
        class_concordance = _compute_rank_concordance(
            [(p[0], p[2]) for p in class_ppl_pairs],
            [(p[1], p[2]) for p in class_ppl_pairs],
        )

        if class_concordance is not None and class_concordance > 0.7:
            verdict = 'reliable — 14M PPL predicts 35M well'
            recommendation = 'pursue'
        elif class_concordance is not None and class_concordance < 0.5:
            verdict = 'uncertain — 14M and 35M rank differently'
            recommendation = 'deprioritize until more data'
        else:
            verdict = 'insufficient data for conclusion'
            recommendation = 'gather more samples'

        class_findings.append({
            'class': config_class,
            'n': len(members),
            'ppl_concordance': class_concordance,
            'verdict': verdict,
            'recommendation': recommendation,
        })

    return {
        'updated_at': _timestamp(),
        '35m_run_tag': run_tag,
        'n_candidates_run': len(completed) + len(crashed),
        'n_keep': sum(1 for c in completed if c['35m_status'] == 'keep'),
        'n_discard': sum(1 for c in completed if c['35m_status'] == 'discard'),
        'n_crash': len(crashed),
        'transfer_stats': {
            'ppl_delta_mean': mean(ppl_deltas) if ppl_deltas else None,
            'ppl_delta_std': stdev(ppl_deltas) if len(ppl_deltas) >= 2 else None,
            'ppl_rank_concordance': ppl_concordance,
            'passkey_delta_mean': mean(passkey_deltas) if passkey_deltas else None,
            'passkey_rank_concordance': passkey_concordance,
        },
        'config_class_findings': class_findings,
        'free_text_findings': None,
    }


def main():
    parser = argparse.ArgumentParser(
        description='🔄 Tier 2 autoresearch loop for 35M probes on RunPod')
    parser.add_argument('--candidates', type=str, required=True,
                        help='Path to candidates_35m.json')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Training steps per probe (default: 2000)')
    parser.add_argument('--webhook', type=str, default=None,
                        help='Discord webhook URL (optional)')
    parser.add_argument('--out', type=str, default='autoresearch/results_35m.tsv',
                        help='Output TSV file (default: autoresearch/results_35m.tsv)')
    arguments = parser.parse_args()

    candidates_path = Path(arguments.candidates)
    assert candidates_path.exists(), f'Candidates file not found: {candidates_path}'

    data = _load_candidates(candidates_path)
    baseline = data['baseline']
    candidates = data['candidates']
    run_tag = data.get('14m_run_tag', 'unknown')

    pending = [c for c in candidates if c['35m_status'] == 'pending']

    _log('═' * 70)
    _log('  🔄 DWARF AutoResearch — 35M Loop')
    _log(f'  Candidates file: {candidates_path}')
    _log(f'  Run tag: {run_tag}')
    _log(f'  Total candidates: {len(candidates)}')
    _log(f'  Pending: {len(pending)}')
    _log(f'  Steps per probe: {arguments.steps}')
    _log(f'  Baseline: {baseline["name"]} '
         f'(PPL={baseline["14m_val_ppl"]}, '
         f'passkey={baseline["14m_passkey_mean"]})')
    _log('═' * 70)

    if not pending:
        _log('  No pending candidates. Exiting.')
        return

    results_tsv = ROOT / arguments.out

    for candidate in sorted(candidates, key=lambda c: c['rank']):
        if candidate['35m_status'] != 'pending':
            continue

        rank = candidate['rank']
        dense = candidate['dense_width']
        sparse = candidate['sparse_list']

        _log('')
        _log('─' * 70)
        _log(f'  📋 Candidate {rank}/{len(candidates)}: '
             f'dense={dense} sparse={_format_sparse(sparse)}')
        _log(f'  14M PPL={candidate["14m_val_ppl"]} '
             f'passkey={candidate["14m_passkey_mean"]}')
        _log(f'  Confidence: {candidate["confidence"]}')
        _log(f'  Rationale: {candidate["rationale"]}')
        _log('─' * 70)

        success, result_json = _run_probe(candidate, arguments.steps, results_tsv)

        if not success or not result_json.exists():
            _log(f'  ✗ Probe crashed')
            candidate['35m_status'] = 'crash'
            candidate['35m_steps'] = arguments.steps
            _save_candidates(data, candidates_path)

            if arguments.webhook:
                _post_webhook(arguments.webhook,
                              f'**35M probe CRASH**: dense={dense} '
                              f'sparse={_format_sparse(sparse)}\n'
                              f'Rank {rank} | Continuing to next candidate...')
            continue

        probe_results = _parse_probe_results(result_json)

        candidate['35m_val_ppl'] = round(probe_results['val_ppl'], 3)
        candidate['35m_passkey_mean'] = round(probe_results['passkey_mean'], 4)
        candidate['35m_passkey_by_distance'] = probe_results['passkey_by_distance']
        candidate['35m_steps'] = arguments.steps
        candidate['transfer_delta_ppl'] = round(
            probe_results['val_ppl'] - candidate['14m_val_ppl'], 3)
        candidate['transfer_delta_passkey'] = round(
            probe_results['passkey_mean'] - candidate['14m_passkey_mean'], 4)

        candidate['35m_status'] = _compute_status(candidate, baseline)

        _save_candidates(data, candidates_path)

        delta_ppl = candidate['transfer_delta_ppl']
        delta_passkey = candidate['transfer_delta_passkey']
        ppl_direction = 'better' if delta_ppl < 0 else 'worse'
        passkey_direction = 'better' if delta_passkey > 0 else 'worse'

        _log('')
        _log(f'  ✓ Probe complete')
        _log(f'  35M val_ppl:    {candidate["35m_val_ppl"]:.3f}')
        _log(f'  35M passkey:    {candidate["35m_passkey_mean"]:.4f} '
             f'({candidate["35m_passkey_mean"] * 100:.1f}%)')
        _log(f'  Transfer ΔPPL:     {delta_ppl:+.3f} ({ppl_direction} at scale)')
        _log(f'  Transfer Δpasskey: {delta_passkey:+.4f} ({passkey_direction} at scale)')
        _log(f'  Status: {candidate["35m_status"]}')

        if arguments.webhook:
            _post_webhook(arguments.webhook,
                          f'**35M probe result**: dense={dense} '
                          f'sparse={_format_sparse(sparse)}\n'
                          f'val_ppl: {candidate["35m_val_ppl"]:.3f} | '
                          f'passkey: {candidate["35m_passkey_mean"] * 100:.1f}% | '
                          f'status: {candidate["35m_status"]}\n'
                          f'Transfer delta PPL: {delta_ppl:+.3f} '
                          f'({ppl_direction} than 14M predicted)')

    completed = [c for c in candidates if c['35m_status'] in ('keep', 'discard')]
    crashed = [c for c in candidates if c['35m_status'] == 'crash']
    keeps = [c for c in candidates if c['35m_status'] == 'keep']

    _log('')
    _log('═' * 70)
    _log('  🏁 All candidates processed')
    _log(f'  Keep: {len(keeps)}  Discard: {len(completed) - len(keeps)}  Crash: {len(crashed)}')
    _log('')

    if keeps:
        _log('  📊 Final ranking (keeps only, by 35M PPL):')
        for i, candidate in enumerate(sorted(keeps, key=lambda c: c['35m_val_ppl']), 1):
            _log(f'    {i}. dense={candidate["dense_width"]} '
                 f'sparse={_format_sparse(candidate["sparse_list"])} '
                 f'PPL={candidate["35m_val_ppl"]:.3f} '
                 f'passkey={candidate["35m_passkey_mean"] * 100:.1f}%')

    _log('═' * 70)

    calibration = _build_calibration_update(data, run_tag)
    if calibration:
        calibration_path = ROOT / 'autoresearch' / 'calibration_update.json'
        with open(calibration_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        _log(f'  📝 Calibration update written to {calibration_path}')

        if arguments.webhook:
            summary_lines = [
                f'**35M autoresearch complete** ({run_tag})',
                f'Keep: {calibration["n_keep"]} | '
                f'Discard: {calibration["n_discard"]} | '
                f'Crash: {calibration["n_crash"]}',
            ]
            stats = calibration['transfer_stats']
            if stats['ppl_delta_mean'] is not None:
                summary_lines.append(
                    f'PPL transfer: Δ={stats["ppl_delta_mean"]:+.3f} '
                    f'(±{stats["ppl_delta_std"]:.3f})' if stats['ppl_delta_std']
                    else f'PPL transfer: Δ={stats["ppl_delta_mean"]:+.3f}')
            if stats['ppl_rank_concordance'] is not None:
                summary_lines.append(
                    f'PPL rank concordance: {stats["ppl_rank_concordance"]:.2f}')

            if keeps:
                best = min(keeps, key=lambda c: c['35m_val_ppl'])
                summary_lines.append(
                    f'Best: dense={best["dense_width"]} '
                    f'sparse={_format_sparse(best["sparse_list"])} '
                    f'PPL={best["35m_val_ppl"]:.3f}')

            _post_webhook(arguments.webhook, '\n'.join(summary_lines))

    _log('  Done. RunPod job can terminate.')


if __name__ == '__main__':
    main()
