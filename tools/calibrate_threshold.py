"""
calibrate_threshold.py — Coupling stability threshold calibration.

Uses existing training outcomes (stable/unstable) and theoretical coupling
coefficients to calibrate the stability threshold for coupling_stability.rs.

No GPU required. Pure arithmetic on known results.

Usage:
  .venv/bin/python3 tools/calibrate_threshold.py

  # After running compute_coupling.py, include measured values:
  .venv/bin/python3 tools/calibrate_threshold.py \
      --measured logs/coupling_condU_13M.json logs/coupling_condU_35M.json
"""

import sys, os, json, argparse, math

# ---------------------------------------------------------------------------
# Known outcomes from training runs
# ---------------------------------------------------------------------------
KNOWN_OUTCOMES = [
    # (label, D, H, L, injection_type, stable, notes)
    ('condU_13M',       256,  8,  6, 'kv',       True,  'condU 13M: 52.237 PPL, 38.3% passkey — clean'),
    ('condU_35M',       512,  8,  6, 'kv',       True,  'condU 35M: 38.542 PPL, 85.0% passkey — clean'),
    ('condU_37m_coeff', 384, 16,  6, 'kv',       True,  'condU 37M coeff probe: 43.434 PPL, 69.2% passkey — stable; designed to test coeff=36864'),
    ('condU_85M',       768, 12,  8, 'kv',       False, 'condU 85M: memorization pathology, val PPL ~3000'),
    ('condM_13M',       256,  8,  6, 'residual', True,  'condM_I2G0: stable, residual injection'),
    ('condM_85M',       640,  8, 12, 'residual', True,  'condM 85M: 36.042 PPL — clean'),
]


def theoretical_coupling(D, H, L, injection_type):
    """
    K/V injection:  coupling grows as H * D * L
                    (nonlinear softmax pathway amplifies with each additional
                     head, dimension, and layer the injected signal propagates through)
    Residual:       coupling grows as D
                    (linear additive to residual stream, bounded by stream dimension)
    """
    if injection_type == 'kv':
        return H * D * L
    else:
        return D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measured', nargs='*', default=[],
                        help='JSON files from compute_coupling.py to include measured values')
    args = parser.parse_args()

    print('=' * 70)
    print('COUPLING STABILITY THRESHOLD CALIBRATION')
    print('=' * 70)

    # Load measured coupling values if provided
    measured = {}
    for path in args.measured:
        if os.path.exists(path):
            data = json.load(open(path))
            for r in (data if isinstance(data, list) else [data]):
                measured[r['model']] = r.get('coupling_mean')

    # Compute theoretical coefficients
    print('\nTheoretical coupling coefficients by architecture:')
    print(f'{"Model":<15} {"Type":<10} {"Coeff":>10} {"Stable":>8} {"Measured":>10}')
    print('-' * 58)

    stable_coeffs = []
    unstable_coeffs = []

    for label, D, H, L, inj, stable, note in KNOWN_OUTCOMES:
        coeff = theoretical_coupling(D, H, L, inj)
        m_val = measured.get(label, None)
        m_str = f'{m_val:.4f}' if m_val else '(pending)'
        status = 'STABLE' if stable else 'UNSTABLE'
        print(f'{label:<15} {inj:<10} {coeff:>10,d} {status:>8} {m_str:>10}')
        if stable:
            stable_coeffs.append(coeff)
        else:
            unstable_coeffs.append(coeff)

    print()
    max_stable   = max(stable_coeffs)
    min_unstable = min(unstable_coeffs)

    print(f'Maximum STABLE coefficient:   {max_stable:,}')
    print(f'Minimum UNSTABLE coefficient: {min_unstable:,}')

    # Three threshold estimates
    conservative = max_stable + 1       # just above last known stable
    log_midpoint = int(math.exp((math.log(max_stable) + math.log(min_unstable)) / 2))
    arithmetic   = (max_stable + min_unstable) // 2

    print()
    print('Threshold estimates:')
    print(f'  Conservative  (just above max stable):  {conservative:,}')
    print(f'  Log midpoint  (geometric mean of gap):   {log_midpoint:,}')
    print(f'  Arithmetic    (midpoint of gap):         {arithmetic:,}')
    print()
    print(f'Recommended for Rust (conservative): {conservative:,}')
    print()

    # Validate against all known outcomes
    print('Validation against known outcomes (conservative threshold):')
    all_correct = True
    for label, D, H, L, inj, stable, note in KNOWN_OUTCOMES:
        coeff = theoretical_coupling(D, H, L, inj)
        predicted_stable = coeff < conservative
        correct = predicted_stable == stable
        mark = '✓' if correct else '✗ WRONG'
        print(f'  {mark} {label}: coeff={coeff:,}, predicted={predicted_stable}, actual={stable}')
        if not correct:
            all_correct = False

    print()
    if all_correct:
        print('All known outcomes correctly predicted.')
    else:
        print('WARNING: Some predictions incorrect — threshold may need adjustment.')

    # Rust constant output
    print()
    print('--- Rust constant (paste into coupling_stability.rs) ---')
    print(f'const STABILITY_THRESHOLD_KV: f64 = {float(conservative)};')
    print(f'// Calibrated from: max_stable={max_stable}, min_unstable={min_unstable}')
    print(f'// Conservative estimate. Update with measured coupling values when available.')
    if measured:
        print(f'// Measured coupling values incorporated: {list(measured.keys())}')

    # Save calibration record
    out = {
        'max_stable_theoretical':   max_stable,
        'min_unstable_theoretical': min_unstable,
        'threshold_conservative':   conservative,
        'threshold_log_midpoint':   log_midpoint,
        'threshold_arithmetic':     arithmetic,
        'recommended':              conservative,
        'measured_coupling':        measured,
        'outcomes':                 [
            {
                'label': label, 'D': D, 'H': H, 'L': L,
                'injection': inj, 'stable': stable,
                'theoretical_coupling': theoretical_coupling(D, H, L, inj),
            }
            for label, D, H, L, inj, stable, _ in KNOWN_OUTCOMES
        ],
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root, 'logs', 'coupling_threshold_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nCalibration saved to: {out_path}')


if __name__ == '__main__':
    main()
