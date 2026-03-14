"""
analyze_j.py — J-reduction analysis for DWARF DSQG offset pruning.
Single modifiable file for the autoresearcher. Grep-able output only.
"""
import numpy as np
from collections import deque

# =============================================================================
# 🎛️ HYPOTHESIS KNOBS — Edit this section to test new J-set hypotheses
# Rules: MUST include delta=3 and delta=4. MUST keep 128 and 384 together (or drop both).
# delta=2 is FREE (0pp passkey cost) — always drop it.
# j_score = (44 - J) * 10 - estimated_pk_loss - estimated_ppl_delta * 2
# Higher j_score = better. Baseline J=44: j_score = 0.0
# =============================================================================

CANDIDATE_SET = sorted({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536})

# =============================================================================
# END OF HYPOTHESIS KNOBS
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 📊 ANALYSIS CODE
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Ablation data (March 11 probe, condU-v5 35M) ────────────────────────
# Each delta maps to (pk_drop_pp, ppl_delta) where:
#   pk_drop_pp = BASELINE_PK - pk_when_zeroed (in percentage points)
#   ppl_delta  = ppl_when_zeroed - 39.53 (PPL increase from removing that offset)

BASELINE_PK = 98.3
BASELINE_PPL = 39.53

ABLATION = {
    0: (8.3, 2.16),    1: (5.0, 0.30),    2: (0.0, 2.49),    3: (46.6, 5.44),
    4: (56.6, 7.30),   5: (10.0, 8.71),   6: (10.0, 9.71),   7: (13.3, 10.48),
    8: (6.6, 11.11),   9: (10.0, 11.56),  10: (5.0, 11.87),  11: (5.0, 12.20),
    12: (8.3, 12.45),  13: (11.6, 12.65), 14: (11.6, 12.80), 15: (10.0, 12.97),
    16: (11.6, 13.11), 17: (11.6, 13.21), 18: (11.6, 13.30), 19: (6.6, 13.39),
    20: (10.0, 13.45), 21: (5.0, 13.51),  22: (5.0, 13.60),  23: (8.3, 13.63),
    24: (11.6, 13.69), 25: (11.6, 13.70), 26: (8.3, 13.75),  27: (11.6, 13.77),
    28: (11.6, 13.76), 29: (11.6, 13.75), 30: (6.6, 13.77),  31: (11.6, 13.77),
    32: (5.0, 13.72),  48: (8.3, 13.78),  64: (10.0, 13.68), 96: (5.0, 13.39),
    128: (13.3, 13.09), 192: (8.3, 12.48), 256: (8.3, 11.88), 384: (5.0, 10.71),
    512: (10.0, 9.63), 768: (1.6, 7.67),  1024: (5.0, 5.88), 1536: (6.6, 2.65),
}

ALL_44 = sorted(ABLATION.keys())

# Co-dependent pair {128, 384}: joint removal is non-additive
CODEPENDENCY_128_384 = {
    "pk_drop_joint": 18.3,
    "ppl_delta_joint": 23.44,
}

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]


# ── 2b. Coverage check (BFS, L=5 hops) ──────────────────────────────────────

def covers_all(offsets, targets, L=5, max_value=1537):
    """Check if all target distances are reachable as sums of ≤L offsets."""
    reachable = np.zeros(max_value, dtype=bool)
    reachable[0] = True
    offsets_array = np.array(offsets, dtype=np.int64)

    for _ in range(L):
        new_reachable = reachable.copy()
        indices = np.where(reachable)[0]
        for index in indices:
            sums = index + offsets_array
            valid = sums[sums < max_value]
            new_reachable[valid] = True
        if np.array_equal(new_reachable, reachable):
            break
        reachable = new_reachable

    return all(reachable[t] for t in targets if t < max_value)


def covers_dense(offsets):
    """Check if every integer in [0, 32] is reachable in ≤5 hops."""
    return covers_all(offsets, list(range(0, 33)), L=5, max_value=33)


def min_hops_to(offsets, target):
    """Minimum number of hops to reach target distance via BFS."""
    if target == 0:
        return 0
    visited = {0}
    queue = deque([(0, 0)])
    offsets_set = set(offsets)
    max_value = target + 1

    while queue:
        current, depth = queue.popleft()
        if depth >= 5:
            continue
        for offset in offsets:
            next_value = current + offset
            if next_value == target:
                return depth + 1
            if next_value < max_value and next_value not in visited:
                visited.add(next_value)
                queue.append((next_value, depth + 1))

    return -1


def relay_depth_stats(offsets):
    """Compute min/mean/max hops across all passkey distances."""
    hops = []
    for distance in PASSKEY_DISTANCES:
        h = min_hops_to(offsets, distance)
        hops.append(h)
    valid_hops = [h for h in hops if h >= 0]
    if not valid_hops:
        return {"minimum": -1, "mean": -1.0, "maximum": -1}
    return {
        "minimum": min(valid_hops),
        "mean": sum(valid_hops) / len(valid_hops),
        "maximum": max(valid_hops),
    }


# ── 2c. Ablation cost estimation ────────────────────────────────────────────

dropped = sorted(set(ALL_44) - set(CANDIDATE_SET))

estimated_pk_loss = 0.0
estimated_ppl_delta = 0.0

both_128_384_dropped = (128 in dropped) and (384 in dropped)

for d in dropped:
    if both_128_384_dropped and d in (128, 384):
        continue
    estimated_pk_loss += ABLATION[d][0]
    estimated_ppl_delta += ABLATION[d][1]

if both_128_384_dropped:
    estimated_pk_loss += CODEPENDENCY_128_384["pk_drop_joint"]
    estimated_ppl_delta += CODEPENDENCY_128_384["ppl_delta_joint"]


# ── 2d. Validation checks ───────────────────────────────────────────────────

assert 3 in CANDIDATE_SET and 4 in CANDIDATE_SET, "Must include delta=3 and delta=4"

if (128 in CANDIDATE_SET) != (384 in CANDIDATE_SET):
    print("⚠️  WARNING: co-dependent pair {128, 384} is split — one kept, one dropped")

coverage_ok = covers_all(CANDIDATE_SET, PASSKEY_DISTANCES)
dense_ok = covers_dense(CANDIDATE_SET)

if not coverage_ok:
    print("❌ FAIL: passkey distance coverage broken")
if not dense_ok:
    print("⚠️  WARNING: dense [0,32] coverage broken")


# ── 2e. Score computation ────────────────────────────────────────────────────

j_size = len(CANDIDATE_SET)
j_score = (44 - j_size) * 10 - estimated_pk_loss - estimated_ppl_delta * 2

stats = relay_depth_stats(CANDIDATE_SET)
hops_to_1536 = min_hops_to(CANDIDATE_SET, 1536)


# ── 2f. Output ───────────────────────────────────────────────────────────────

print("---")
print(f"j_score:              {j_score:.2f}")
print(f"j_size:               {j_size}")
print(f"coverage_ok:          {'YES' if coverage_ok else 'NO'}")
print(f"estimated_pk_loss:    {estimated_pk_loss:.1f}")
print(f"estimated_ppl_delta:  {estimated_ppl_delta:.2f}")
print(f"dropped_offsets:      {dropped}")
print(f"min_hops_d1536:       {hops_to_1536}")
print(f"mean_relay_depth:     {stats['mean']:.1f}")
print(f"best_j_set:           {CANDIDATE_SET}")


# ─────────────────────────────────────────────────────────────────────────────
# 📝 LEARNINGS
# ─────────────────────────────────────────────────────────────────────────────
# Iteration 1: baseline J=44
