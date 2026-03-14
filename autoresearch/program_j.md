# DWARF Math Autoresearcher — J Reduction

Same methodology as Karpathy's autoresearch. One file to modify. One metric. Loop forever.
No GPU required. Pure mathematical analysis and verification.

---

## What This Is

We know from the March 11 ablation probe that individual offsets have measurable passkey/PPL costs.
We want to find the smallest subset J* ⊂ {our 44 offsets} that preserves relay coverage
while minimizing ablation cost. This is the J-set we train the next model on.

The metric is **j_score** (higher = better):
```
j_score = (44 - J) * 10 - estimated_pk_loss - estimated_ppl_delta * 2
```
- Rewards smaller J (every offset dropped = +10 score)
- Penalizes estimated passkey loss (pp) from dropping offsets (independence assumption)
- Penalizes estimated PPL increase
- Baseline J=44: j_score = 0.0

**Coverage constraint**: candidate set MUST satisfy relay coverage (BFS, L=5 hops, max_dist=1536).
Sets failing coverage are logged as crashes and discarded.

---

## The One File You Modify

**`autoresearch/analyze_j.py`** — this is the only file you edit.

It has two sections:
1. **HYPOTHESIS KNOBS** (top): the current candidate J-set being tested
2. **ANALYSIS CODE** (below): coverage checks, scoring, printing — agent may extend this section

When run, it outputs a grep-able summary to stdout. That's all you read.

---

## Setup

1. Agree on a run tag (e.g. `j-mar11`)
2. Create branch: `git checkout -b autoresearch/<tag>` from current master
3. Read `autoresearch/analyze_j.py` fully — this is the ONLY file you modify
4. Read `autoresearch/j_results.tsv` for current state (start empty except header)
5. First run establishes the baseline j_score for the greedy J=44 (should be 0.0)
6. Confirm and begin

---

## The Experiment Loop

Run from DWARF root:
```bash
.venv/bin/python3 autoresearch/analyze_j.py > autoresearch/j_run.log 2>&1
```

**Extract results (DO NOT read the full log):**
```bash
grep "^j_score:\|^coverage_ok:\|^best_j_set:\|^estimated_pk_loss:\|^estimated_ppl_delta:\|^j_size:" autoresearch/j_run.log
```

**LOOP:**
1. Read `j_results.tsv` + `git log --oneline autoresearch/<tag>` (your only memory)
2. Form a hypothesis: a new CANDIDATE_SET with fewer offsets or better theoretical properties
3. Edit `autoresearch/analyze_j.py` — change CANDIDATE_SET, or extend ANALYSIS CODE
4. `git commit -am "J hypothesis: <one line>"`
5. Run the analysis
6. `grep "^j_score:\|^coverage_ok:\|^j_size:\|^estimated_pk_loss:\|^estimated_ppl_delta:" autoresearch/j_run.log`
7. If coverage_ok is FALSE: log as `crash`, git reset, try next idea
8. Log to `j_results.tsv`: commit, j_score, j_size, pk_loss, ppl_delta, status, description
9. If j_score improved (higher) → **keep** (advance branch)
10. If j_score equal or worse → **discard**: `git reset --soft HEAD~1`, try next idea

---

## Logging (j_results.tsv)

Tab-separated. DO NOT commit this file.

```
commit	j_score	j_size	estimated_pk_loss	estimated_ppl_delta	coverage_ok	status	description
```

Status: `keep`, `discard`, `crash`

Example:
```
commit	j_score	j_size	estimated_pk_loss	estimated_ppl_delta	coverage_ok	status	description
a1b2c3d	0.0	44	0.0	0.0	YES	keep	baseline J=44
b2c3d4e	45.2	39	8.3	1.2	YES	keep	dropped 5 cheapest offsets (2,10,21,22,30)
c3d4e5f	-15.0	37	45.0	5.0	YES	discard	dropped delta=3 (too costly)
d4e5f6g	crash	35	0.0	0.0	NO	crash	coverage fails — d=1024 unreachable
```

---

## What's In Scope

**In the HYPOTHESIS KNOBS section:**
- `CANDIDATE_SET`: the J-set being evaluated (always include δ=3 and δ=4)

**In the ANALYSIS CODE section (agent may extend):**
- Coverage checking logic (BFS/DP — extend for new coverage criteria)
- Ablation cost estimation (can add interaction terms for known co-dependent pairs)
- New theoretical analysis (Frobenius numbers, relay depth, coprime pair analysis)
- External candidate evaluation (ChatGPT set, DSQG-adapted, custom proposals)

**Out of scope:**
- Modifying kernel code
- Running actual GPU training
- Modifying the ablation data source

---

## Key Facts (Do Not Re-Test)

From March 11 ablation probe (probe_condU_v5_35m_results.json):
- δ=3: pk_drop=46.6pp, ppl_delta=+5.44 — CRITICAL, always keep
- δ=4: pk_drop=56.6pp, ppl_delta=+7.30 — MOST CRITICAL, always keep
- δ=2: pk_drop=0.0pp, ppl_delta=+2.49 — FREE, always drop first
- {128,384}: co-dependent — removal together → pk_drop=18.3pp, ppl_delta=+23.44
- Individual 128: pk_drop=13.3pp; individual 384: pk_drop=5.0pp (independence assumption underestimates joint)
- ChatGPT postage-stamp J=12: {1,6,8,21,60,93,104,154,378,414,539,938} — 8 of these NOT in our current 44
- Coverage verified: J=16 minimum from current set for dense [0,1536] coverage in ≤5 hops

**Known failure modes:**
- Dropping δ=3 or δ=4: catastrophic (never do this)
- Dropping {128} without {384} or vice versa: co-dependent, underestimated cost
- Coverage fails silently if you don't check BFS

---

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue.
If you exhaust obvious candidates, extend the analysis:
- Add interaction terms (which pairs are co-dependent beyond {128,384}?)
- Develop new coverage criteria (relay_depth, bottleneck analysis)
- Explore theoretical lower bounds (what is the TRUE minimum J for our 44-offset set?)
- Test external offset sets (ChatGPT, DSQG-adapted) even if they include new values — score them
- Ask: if we could add ONE offset not in our current set, which would let us drop the most existing offsets?
The human will interrupt you when they wake up.
