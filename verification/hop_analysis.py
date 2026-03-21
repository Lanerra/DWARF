"""
DWARF Relay Hop Analysis
True BFS computation (no monotone filter bias).
March 20, 2026 — written after GPT-5.4-Thinking identified the reachableInHops bug.
"""
from collections import deque, Counter

J13D = [1, 2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
J24  = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]
PASSKEY_DISTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_GAPS  = [d + 5 for d in PASSKEY_DISTS]
N = 2048

def true_bfs(offsets, N=2048):
    """Unbiased BFS — no filter, correct min-hop for all lags."""
    min_hops = [-1] * N
    min_hops[0] = 0
    q = deque([0])
    while q:
        node = q.popleft()
        for d in offsets:
            nxt = node + d
            if nxt < N and min_hops[nxt] == -1:
                min_hops[nxt] = min_hops[node] + 1
                q.append(nxt)
    return min_hops

def report(name, offsets):
    hops = true_bfs(offsets)
    dist = Counter(h for h in hops[1:] if h >= 0)
    unreachable = sum(1 for h in hops[1:] if h == -1)
    diameter = max(dist.keys()) if dist else 0

    print(f"\n{'='*60}")
    print(f"  {name} ({len(offsets)} offsets)")
    print(f"{'='*60}")
    print(f"  True relay diameter: {diameter}")
    print(f"  Unreachable lags: {unreachable}")
    print()
    print("  Hop distribution:")
    for k in sorted(dist):
        print(f"    {k} hops: {dist[k]:4d} lags")
    print()
    covered = {k: sum(v for h,v in dist.items() if h <= k) for k in range(1, diameter+1)}
    print("  Cumulative coverage:")
    for k, n in covered.items():
        print(f"    ≤{k} hops: {n:4d}/2047 ({n/2047*100:.1f}%)")
    print()
    print("  Passkey gap hop counts:")
    for d, g in zip(PASSKEY_DISTS, PASSKEY_GAPS):
        if g < N:
            h = hops[g]
            print(f"    d={d:5d}  gap={g:5d}  hops={h}")
        else:
            print(f"    d={d:5d}  gap={g:5d}  OUT OF RANGE")

    # Witnesses for max hop depth
    max_h = diameter
    witnesses = [i for i, h in enumerate(hops) if h == max_h]
    print(f"\n  Witnesses for max depth ({max_h} hops): {witnesses[:10]}")
    return hops

if __name__ == '__main__':
    print("DWARF True Relay Hop Analysis")
    print("(Unbiased BFS — correct min-hop computation)")
    hops_j13d = report("J13D", J13D)
    hops_j24  = report("J24",  J24)

    print("\n" + "="*60)
    print("  GPT-5.4 CORRECTNESS VERIFICATION")
    print("="*60)
    print(f"\n  J13D true diameter: {max(h for h in hops_j13d if h >= 0)}")
    print(f"  (Claimed diameter=3 in Lean Prompt 4 was WRONG)")
    print(f"\n  Passkey gaps all ≤3 hops? {all(hops_j13d[g] <= 3 for g in PASSKEY_GAPS if g < N)}")
    print(f"  (This is what matters for the paper — CORRECT)")
    print(f"\n  Additive capacity bound C(J+k,k) for J=13, k=3:")
    from math import comb
    print(f"    C(16,3) = {comb(16,3)} {'≥' if comb(16,3) >= N else '<'} {N}")
    print(f"    → 3-hop coverage of all [1,{N-1}] impossible with J=13 (needs C(J+3,3)≥{N})")
    print(f"    → Minimum J for full 3-hop coverage: J={next(j for j in range(1,100) if comb(j+3,3)>=N)}")
