"""
analyze_j24d_redundancy.py
Find offset sets that extend j24d's J=24 with more coprime-pair redundancy,
fill the mid-range gap, and minimize J increase.

Uses actual j24d_int2_physics 39.5M ablation data from probe_j24d_int2_physics_run.log.
March 15 2026.
"""
from collections import deque
from itertools import combinations

# ─── j24d ablation data (from probe_j24d_int2_physics_run.log) ───────────────
# (passkey_drop_pp, ppl_delta) when that offset is zeroed
# baseline: PPL=42.07, passkey=91.7%

J24D_ABLATION = {
    1:    (20.0,  2.40),
    2:    ( 1.7,  1.87),
    3:    (31.7,  3.61),
    4:    (61.7,  4.97),
    5:    (14.2,  5.94),
    6:    (17.5,  6.92),
    7:    (25.9,  7.44),
    8:    (18.4,  8.18),
    9:    (17.5,  8.51),
    10:   (18.4,  8.78),
    13:   (20.0,  9.63),
    15:   (21.7,  9.80),
    16:   (10.9,  9.87),
    21:   (15.0, 10.27),
    23:   (17.5, 10.43),
    28:   (14.2, 10.56),
    48:   (15.9, 10.75),
    64:   (10.9, 10.78),
    96:   (15.0, 10.87),
    192:  (10.0, 10.24),
    384:  ( 7.5,  9.24),
    512:  ( 9.2,  8.52),
    768:  ( 3.4,  7.20),
    1024: ( 4.2,  5.79),
}

J24D_BASE = sorted(J24D_ABLATION.keys())

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]

# ─── relay coverage ───────────────────────────────────────────────────────────

def min_hops(offsets, target, max_hops=6):
    if target == 0: return 0
    visited = {0}
    queue = deque([(0, 0)])
    while queue:
        cur, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for o in offsets:
            nxt = cur + o
            if nxt == target:
                return depth + 1
            if nxt < target + 1 and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return -1  # not reachable

def coverage_ok(offsets, max_hops=5):
    return all(min_hops(offsets, d, max_hops) >= 0 for d in PASSKEY_DISTANCES)

def relay_stats(offsets):
    hops = [min_hops(offsets, d) for d in PASSKEY_DISTANCES]
    valid = [h for h in hops if h >= 0]
    per_dist = dict(zip(PASSKEY_DISTANCES, hops))
    return {
        'max': max(valid) if valid else 99,
        'mean': sum(valid)/len(valid) if valid else 99,
        'per_dist': per_dist,
        'covered': len(valid),
    }

# ─── coprime-pair redundancy ──────────────────────────────────────────────────

def count_relay_paths(offsets, target, max_hops=2):
    """Count distinct N-hop paths (as tuples) that sum to target using offsets."""
    offsets = sorted(set(offsets))
    if max_hops == 1:
        return 1 if target in offsets else 0
    count = 0
    def dfs(remaining, hops_left, path):
        nonlocal count
        if remaining == 0:
            count += 1
            return
        if hops_left == 0:
            return
        for o in offsets:
            if o <= remaining:
                dfs(remaining - o, hops_left - 1, path + [o])
    dfs(target, max_hops, [])
    return count

def redundancy_score(offsets):
    """
    For each passkey distance, count 2-hop relay paths available.
    Sum weighted by criticality (shorter distances more critical).
    """
    total = 0
    offsets_set = set(offsets)
    for d in PASSKEY_DISTANCES:
        paths_2hop = sum(1 for a in offsets if (d - a) in offsets_set and a > 0 and d - a > 0)
        paths_3hop = sum(1 for a in offsets for b in offsets
                        if a > 0 and b > 0 and (d - a - b) in offsets_set and (d - a - b) > 0)
        weight = 1.0 / (d ** 0.3)  # weight short distances more
        total += weight * (paths_2hop * 2 + paths_3hop * 0.5)
    return total

# ─── candidate offset pool ────────────────────────────────────────────────────
# Gaps in j24d: 11, 12, 14, 17-20, 22, 24-27, 29-47, 49-63, 65-95, 97-191, 193-383, etc.
# Focus on mid-range (29-63) and secondary coprime pairs (11, 12, 17, 22, 32, 36, 40)

CANDIDATE_ADDITIONS = [
    # Close mid-range gap
    32, 36, 40, 44,
    # Secondary coprime pairs / relay bridges
    11, 12, 17, 22,
    # Between 48 and 64
    56,
    # Between 28 and 48
    35, 42,
]

# ─── baseline analysis of J24D current set ───────────────────────────────────

print("=" * 65)
print("  Baseline: J24D current set (J=24)")
print("=" * 65)
stats = relay_stats(J24D_BASE)
redund = redundancy_score(J24D_BASE)
print(f"  J = {len(J24D_BASE)}")
print(f"  coverage_ok (5 hops): {coverage_ok(J24D_BASE)}")
print(f"  max relay depth: {stats['max']} hops")
print(f"  mean relay depth: {stats['mean']:.2f} hops")
print(f"  redundancy score: {redund:.2f}")
print(f"  per-distance hops: {stats['per_dist']}")
print()

# ─── evaluate single additions ────────────────────────────────────────────────

print("─" * 65)
print("  Single offset additions (δ → impact on relay structure)")
print("─" * 65)

single_results = []
for add in sorted(CANDIDATE_ADDITIONS):
    candidate = sorted(set(J24D_BASE) | {add})
    stats = relay_stats(candidate)
    redund = redundancy_score(candidate)
    # Rough score: redundancy gain per J increase
    baseline_redund = redundancy_score(J24D_BASE)
    gain = redund - baseline_redund
    # Count 2-hop paths for the most critical offsets (δ=3,4)
    cs = set(candidate)
    paths_for_3 = sum(1 for a in candidate if (3 - a) in cs and a > 0 and 3-a > 0)
    paths_for_4 = sum(1 for a in candidate if (4 - a) in cs and a > 0 and 4-a > 0)
    # Identify what critical distances this helps
    base_hops = relay_stats(J24D_BASE)['per_dist']
    improved = [d for d in PASSKEY_DISTANCES if stats['per_dist'].get(d, 99) < base_hops.get(d, 99)]
    single_results.append((gain, add, candidate, stats, redund, improved))

single_results.sort(reverse=True)
for gain, add, candidate, stats, redund, improved in single_results:
    cs = set(candidate)
    paths_4 = sum(1 for a in candidate if (4-a) in cs and a > 0 and 4-a > 0)
    paths_3 = sum(1 for a in candidate if (3-a) in cs and a > 0 and 3-a > 0)
    improved_str = str(improved) if improved else 'none'
    print(f"  +δ={add:4d}  J={len(candidate):2d}  redund_gain=+{gain:.2f}  "
          f"paths(δ=4)={paths_4}  paths(δ=3)={paths_3}  "
          f"max_hops={stats['max']}  improved={improved_str}")

print()

# ─── evaluate pairs of additions ─────────────────────────────────────────────

print("─" * 65)
print("  Best pairs of additions (top 15 by redundancy gain, J≤26)")
print("─" * 65)

pair_results = []
for a1, a2 in combinations(sorted(CANDIDATE_ADDITIONS), 2):
    candidate = sorted(set(J24D_BASE) | {a1, a2})
    if len(candidate) > 26:
        continue
    stats = relay_stats(candidate)
    redund = redundancy_score(candidate)
    baseline_redund = redundancy_score(J24D_BASE)
    gain = redund - baseline_redund
    cs = set(candidate)
    paths_4 = sum(1 for a in candidate if (4-a) in cs and a > 0 and 4-a > 0)
    base_hops = relay_stats(J24D_BASE)['per_dist']
    improved = [d for d in PASSKEY_DISTANCES if stats['per_dist'].get(d, 99) < base_hops.get(d, 99)]
    pair_results.append((gain, a1, a2, candidate, stats, improved))

pair_results.sort(reverse=True)
for gain, a1, a2, candidate, stats, improved in pair_results[:15]:
    print(f"  +δ={a1},{a2}  J={len(candidate):2d}  redund_gain=+{gain:.2f}  "
          f"max_hops={stats['max']}  mean={stats['mean']:.2f}  "
          f"improved_dists={improved}")

print()
print("─" * 65)
print("  Top 5 pairs: full offset sets")
print("─" * 65)
for gain, a1, a2, candidate, stats, improved in pair_results[:5]:
    print(f"  +{a1,a2}  J={len(candidate)}: {candidate}")

print()

# ─── specifically: check coprime-pair alternatives ────────────────────────────

print("─" * 65)
print("  Coprime-pair analysis: 2-hop paths to δ=4 and δ=3")
print("─" * 65)

print(f"  Current j24d: δ=4 2-hop paths = ", end="")
cs = set(J24D_BASE)
p4 = [(a, 4-a) for a in J24D_BASE if (4-a) in cs and a > 0 and 4-a > 0]
p3 = [(a, 3-a) for a in J24D_BASE if (3-a) in cs and a > 0 and 3-a > 0]
print(p4, f"  δ=3 2-hop paths =", p3)
print("  → Both δ=3 and δ=4 have no 2-hop alternatives in the dense local region.")
print("    (1+3=4 ✓ via δ=1+δ=3; 1+2=3 ✓ via δ=1+δ=2 — but δ=2 barely matters)")
print("    The issue is that δ=4 is on a SINGLE relay hop; 2+2 requires TWO δ=2 hops.")
print()

# Which additions create NEW paths to reach distance 4 and 3 in 2 hops?
print("  Additions that create NEW 2-hop paths to d=32, d=64, d=128 (mid-range):")
for add in sorted(CANDIDATE_ADDITIONS):
    cs = set(J24D_BASE) | {add}
    new_paths = {}
    for target in [32, 40, 48, 64, 128]:
        base_paths = [(a, target-a) for a in J24D_BASE if (target-a) in set(J24D_BASE) and a > 0 and target-a > 0]
        new_p = [(a, target-a) for a in sorted(cs) if (target-a) in cs and a > 0 and target-a > 0]
        added = [p for p in new_p if p not in base_paths and (p[1], p[0]) not in base_paths]
        if added:
            new_paths[target] = added
    if new_paths:
        print(f"    +δ={add}: {new_paths}")
