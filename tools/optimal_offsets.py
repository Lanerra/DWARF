"""
optimal_offsets.py
Find the minimum offset set by greedily pruning offsets from the current J=44
set, using ablation probe importance scores and postage-stamp coverage checks.

Usage:
    python3 tools/optimal_offsets.py
"""

# ── Ablation data (from probe_condU_v5_35m results) ──────────────────────────
# (offset_value, passkey_when_zeroed_pct, ppl_when_zeroed)
BASELINE_PK  = 98.3
BASELINE_PPL = 39.53

ABLATION = [
    (0,    90.0, 41.69),
    (1,    93.3, 39.83),
    (2,    98.3, 42.02),
    (3,    51.7, 44.97),
    (4,    41.7, 46.83),
    (5,    88.3, 48.24),
    (6,    88.3, 49.24),
    (7,    85.0, 50.01),
    (8,    91.7, 50.64),
    (9,    88.3, 51.09),
    (10,   93.3, 51.40),
    (11,   93.3, 51.73),
    (12,   90.0, 51.98),
    (13,   86.7, 52.18),
    (14,   86.7, 52.33),
    (15,   88.3, 52.50),
    (16,   86.7, 52.64),
    (17,   86.7, 52.74),
    (18,   86.7, 52.83),
    (19,   91.7, 52.92),
    (20,   88.3, 52.98),
    (21,   93.3, 53.04),
    (22,   93.3, 53.13),
    (23,   90.0, 53.16),
    (24,   86.7, 53.22),
    (25,   86.7, 53.23),
    (26,   90.0, 53.28),
    (27,   86.7, 53.30),
    (28,   86.7, 53.29),
    (29,   86.7, 53.28),
    (30,   91.7, 53.30),
    (31,   86.7, 53.30),
    (32,   93.3, 53.25),
    (48,   90.0, 53.31),
    (64,   88.3, 53.21),
    (96,   93.3, 52.92),
    (128,  85.0, 52.62),
    (192,  90.0, 52.01),
    (256,  90.0, 51.41),
    (384,  93.3, 50.24),
    (512,  88.3, 49.16),
    (768,  96.7, 47.20),
    (1024, 93.3, 45.41),
    (1536, 91.7, 42.18),
]

CURRENT_OFFSETS = [row[0] for row in ABLATION]

# ── Coverage check ────────────────────────────────────────────────────────────

def reachable_set(offsets, L=5, max_val=1536):
    """BFS: all values reachable as sum of ≤L elements of offsets."""
    reachable = {0}
    frontier  = {0}
    for hop in range(L):
        new = set()
        for r in frontier:
            for o in offsets:
                v = r + o
                if v <= max_val and v not in reachable:
                    new.add(v)
        reachable |= new
        frontier = new
        if not frontier:
            break
    return reachable

def covers(offsets, targets, L=5):
    """Check if all target distances are reachable in ≤L hops."""
    r = reachable_set(offsets, L)
    missing = [t for t in targets if t not in r]
    return len(missing) == 0, missing

# Three coverage targets (increasingly strict):
TARGETS = {
    'passkey_only':  [1,2,4,8,16,32,64,128,256,512,1024,1536],
    'original_offsets': CURRENT_OFFSETS,  # every original offset reachable via relay
    'dense_full':    list(range(0, 1537)),  # every integer 0..1536 (postage stamp)
}

# ── Removal cost ──────────────────────────────────────────────────────────────
# Weighted sum: passkey drop (primary) + PPL cost (secondary, normalised)
MAX_PPL_DELTA = max(ppl - BASELINE_PPL for _, _, ppl in ABLATION)

def removal_cost(pk_when_zeroed, ppl_when_zeroed, pk_weight=1.0, ppl_weight=0.3):
    pk_drop   = (BASELINE_PK - pk_when_zeroed) / BASELINE_PK       # [0,1]
    ppl_delta = (ppl_when_zeroed - BASELINE_PPL) / MAX_PPL_DELTA   # [0,1]
    return pk_weight * pk_drop + ppl_weight * ppl_delta

costs = {
    off: removal_cost(pk, ppl)
    for off, pk, ppl in ABLATION
}

# Sort ascending: cheapest to remove first
sorted_offsets = sorted(CURRENT_OFFSETS, key=lambda o: costs[o])

# ── Greedy pruning ────────────────────────────────────────────────────────────

print("=" * 72)
print("  DSQG Optimal Offset Pruning")
print(f"  Baseline: J={len(CURRENT_OFFSETS)}  pk={BASELINE_PK}%  ppl={BASELINE_PPL}")
print("=" * 72)

for target_name, target_vals in TARGETS.items():
    print(f"\n── Coverage target: '{target_name}' ──")
    remaining = list(CURRENT_OFFSETS)
    dropped   = []
    ppl_sum   = 0.0  # cumulative PPL cost of drops (independent assumption)
    pk_min    = BASELINE_PK  # worst single-offset passkey if dropped

    for off in sorted_offsets:
        candidate = [o for o in remaining if o != off]
        ok, missing = covers(candidate, target_vals)
        if ok:
            remaining = candidate
            dropped.append(off)
            # track cost
            _, pk_z, ppl_z = next(r for r in ABLATION if r[0] == off)
            ppl_sum += (ppl_z - BASELINE_PPL)
            pk_min   = min(pk_min, pk_z)
        # else: keep it

    print(f"  Kept    ({len(remaining):2d}): {sorted(remaining)}")
    print(f"  Dropped ({len(dropped):2d}): {sorted(dropped)}")
    print(f"  Est. cumulative PPL cost (independence assumption): +{ppl_sum:.2f}")
    print(f"  Worst single-offset passkey (any one dropped):      {pk_min:.1f}%")
    ok2, miss2 = covers(remaining, target_vals)
    print(f"  Coverage check: {'PASS' if ok2 else f'FAIL (missing {miss2[:5]})'}")

# ── Print per-offset removal cost ranking ────────────────────────────────────
print("\n── Removal cost ranking (cheapest first) ──")
print(f"  {'δ':>6}  {'cost':>6}  {'pk_drop':>8}  {'ppl_Δ':>7}  {'passkey_z':>10}")
for off in sorted_offsets:
    _, pk_z, ppl_z = next(r for r in ABLATION if r[0] == off)
    pk_drop = BASELINE_PK - pk_z
    ppl_d   = ppl_z - BASELINE_PPL
    print(f"  {off:>6}  {costs[off]:>6.4f}  {pk_drop:>7.1f}pp  {ppl_d:>6.2f}  {pk_z:>9.1f}%")
