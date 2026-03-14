# DWARF Autoresearch

Autonomous architecture search for DSQG (Dyadic Sparse Query-Gate) attention.
Modeled after Karpathy's autoresearch. One file to modify. One metric. Loop forever.

---

## What We Know (Mechanistic Findings — March 11, 2026)

These are CONFIRMED findings from systematic ablation of the condU_v5 35M checkpoint.
Use them to form hypotheses. Do not re-test what is already confirmed.

### The Relay Chain Mechanism
DSQG doesn't do retrieval at inference. It TRAINS a relay chain into the weights:
- **pos_bias** [J, H] = the relay graph routing. ESSENTIAL. Without it: 11.7% passkey.
- **scale_embed** [J, HD] = amplitude modulation. NOT essential. Without it: 95% passkey.
  - scale_embed × 1.5 at inference → 100% passkey (free upgrade)
  - scale_embed must GROW during training (threshold ~0.74 |max|) to catalyze relay chain formation
- **IF physics (Kalman EMA, KdV)** = training scaffolding only. Not load-bearing at inference.
  - Zeroing all IF at inference: 96.7% passkey, +2.05 PPL
  - K injection specifically: essentially FREE (98.3% passkey, +0.18 PPL)
  - V injection costs PPL (+1.75) because it propagates useful values through residual stream

### Three-Stage Pipeline (CONFIRMED)
```
L0 (DSQG) → L1-L3 (optional depth) → L4 (DSQG staging) → L5 (full attention)
```
- L0 skip: 0% passkey, 8402 PPL — ESSENTIAL
- L1 skip: 100% passkey, 51.98 PPL — ENTIRELY OPTIONAL for retrieval
- L2 skip: 93.3% passkey — mostly optional
- L3 skip: 80% passkey — adds composition depth
- L4 skip: 0% passkey — ESSENTIAL (staging layer)
- L5 skip: 0% passkey — ESSENTIAL (full attention extraction)

**Implication: a 3-layer model (L0 + staging DSQG + full attention) might work.**

### Critical Components
- **L4 H7**: pk=36.7%, ppl=+6.0 when knocked out. Most critical individual head.
  It's the LOCAL head in the staging layer — counterintuitive but confirmed.
- **δ=3, δ=4**: Most critical offsets (56.6pp and 46.6pp passkey drop when removed).
  These are COPRIME (gcd=1, Frobenius=5) — model discovered coprime number theory via gradient descent.
  ALWAYS keep both δ=3 and δ=4 in the offset set.
- **δ=2**: Completely FREE for passkey (98.3% when removed). Can drop it.
- **{128, 384}**: Co-dependent pair — must keep together. Removal → 80% passkey + PPL jumps.

### Offset Theory
- J=44 current set covers [0,32] dense + 11 long-range sparse
- J=12 theoretical minimum (postage-stamp optimal): {1,6,8,21,60,93,104,154,378,414,539,938}
- J=12 DSQG-adapted (keeps coprime backbone): {0,1,3,4,8,21,60,93,154,378,539,938}
- J=16 minimum from current 44 offsets: {1,3,4,15,23,24,27,29,30,48,64,128,256,384,512,1024}
- δ=2, δ=0 (at non-trivial positions) are cheap to remove

### Known Working Config (baseline)
condU-v5 13M: D=256, H=8, L=6, FFN=1024, J=44 offsets, INTERFERENCE=3, FULL_ATTN_LAYER=5
Expected ep3: passkey ~20-40%, PPL ~55-70 (13M is slower to develop than 35M)
Expected ep10: passkey ~38%, PPL ~52

### Known Failures (Do Not Repeat)
- Pure DSQG without full attention: 0% passkey at ≤36M (confirmed condW, d49-d54)
- IF layers removed entirely during training: 0% passkey (d53, d54 — confirmed ≤35M)
- Hard spectral masking (condT): global heads starved of gradient → 0% passkey
- RoPE + DSQG (condM-v2): RoPE competes with pos_bias → worse than baseline

---

## Setup

1. Agree on a run tag (e.g. `ar-mar11`)
2. Create branch: `git checkout -b autoresearch/<tag>` from current master
3. Read `autoresearch/train_ar.py` fully — this is the ONLY file you modify
4. Read `autoresearch/results.tsv` for current state (start empty except header)
5. Confirm and begin

---

## The Experiment Loop

The ONLY file you edit is `autoresearch/train_ar.py`.
Run from DWARF root with: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u autoresearch/train_ar.py > autoresearch/run.log 2>&1`

**Always use `-u` (unbuffered stdout).** Without it, Python buffers ~8KB before flushing to the log file — with 3-epoch screen runs, the buffer won't purge until well after ep3, making real-time monitoring impossible.

**Extract results (DO NOT read the full log):**
```bash
grep "^passkey_ep3:\|^ppl_ep3:\|^ar_score:\|^passkey_ep1:\|^passkey_ep2:" autoresearch/run.log
```

**ar_score** (primary metric, higher = better):
```
ar_score = passkey_ep3 - max(0, ppl_ep3 - 55.0) * 0.5
```
- Rewards passkey, penalizes PPL above 55 (expected 13M baseline)
- First run establishes baseline ar_score; subsequent runs must improve it

**LOOP:**
1. Form a hypothesis based on mechanistic findings + prior results
2. Edit `train_ar.py` (experiment knobs section)
3. `git commit -am "hypothesis: <one line description>"`
4. Run the training (see command above)
5. `grep "^passkey_ep3:\|^ppl_ep3:\|^ar_score:" autoresearch/run.log`
6. If run crashed: `tail -n 30 autoresearch/run.log`, fix and retry
7. Log to `results.tsv`: commit hash, ar_score, passkey_ep3, ppl_ep3, status, description
8. If ar_score improved → **keep** (advance branch)
9. If ar_score equal or worse → **discard**: `git reset --soft HEAD~1`, revert and try next idea

**Memory:** On a new experiment, also read the prior git log:
```bash
git log --oneline autoresearch/<tag>
```
And current results.tsv. These are your only memory sources.

---

## Logging (results.tsv)

Tab-separated. DO NOT commit this file.

```
commit	ar_score	passkey_ep3	ppl_ep3	memory_mb	status	description
```

Status: `keep`, `discard`, `crash`

Example:
```
commit	ar_score	passkey_ep3	ppl_ep3	memory_mb	status	description
a1b2c3d	baseline	22.3	65.4	4200	keep	baseline: J=44 condU-v5 13M
b2c3d4e	25.1	30.0	60.2	4100	keep	J=12 DSQG-adapted (0,1,3,4,8,21,60,93,154,378,539,938)
c3d4e5f	18.0	20.0	72.0	3900	discard	J=9 minimal - relay chain too sparse at 13M
```

---

## What's In Scope

**J (offset set) is LOCKED at J=44.** The kernel has hardcoded assertions for J=44 and J_SPARSE=11.
Changing J requires kernel modification — that is handled separately by the math autoresearcher.
Do NOT touch the OFFSETS list in train_ar.py.

**Everything else is fair game.** In the EXPERIMENT KNOBS section:

### Architecture
- **NUM_LAYERS** (3, 4, 5, 6): Test three-stage pipeline (3 = L0+staging+full-attn minimal)
- **FULL_ATTN_LAYER** = always NUM_LAYERS-1 (update when changing NUM_LAYERS)
- **INTERFERENCE** position (0 to NUM_LAYERS-3): which DSQG layer gets IF scaffolding
  - None = no IF (risky — prior runs at ≤35M showed 0% passkey without IF during training)
- **EMBEDDING_DIM / NUM_HEADS / FFN_DIM**: Adjust together to stay ~13M params
  - e.g. D=320 H=8 FFN=1280 L=5 ≈ 13M — wider but shallower
  - e.g. D=256 H=8 FFN=1024 L=6 ≈ 13M — current baseline

### Training dynamics (highest leverage based on today's findings)
- **SCALE_EMBED_LR_MULT** (1.0, 3.0, 10.0): Critical. scale_embed must reach |max|≈0.74 for relay chain
  to form. Higher mult → faster threshold crossing → passkey emerges earlier.
- **SCALE_EMBED_INIT_VAL** (0.0, 0.01, 0.1): Higher init → head start on threshold.
  Combine with high LR_MULT for fastest phase transition.
- **POS_BIAS_LR_MULT** (1.0, 3.0): pos_bias is the routing structure — faster learning may help.
- **LR** (1e-4, 3e-4, 6e-4): Overall learning rate.
- **LR schedule**: cosine decay (current), linear, warmup-only-then-constant

### Physics parameters
- **EMA_CLAMP_FLOOR** (0.001, 0.01, 0.1): Kalman EMA minimum factor. Higher = shorter memory.
- **IF gate init**: zero-init (current), small positive — affects how IF scaffolding activates early
- **KdV alpha**: currently learned; try fixing to 0 (removes KdV, keeps EMA only)

### Optimizer
- **GRAD_ACCUM** (2, 4, 8): affects effective batch size and training stability
- **WEIGHT_DECAY** (0.01, 0.1, 0.3)
- **BETA2** (0.95, 0.99): Adam momentum — lower = faster adaptation

**Out of scope (requires kernel changes — handled by math autoresearcher):**
- OFFSETS list (J)
- Modifying kernel code in kernels/ directory
- Changing dataset or tokenizer
- Adding new packages

---

## Thermal Safety

Run on CUDA_VISIBLE_DEVICES=0 (4090) only. One experiment at a time.
Each screen takes ~15 minutes (3 epochs × ~300s on 4090 at 13M).
Total overnight budget: 6 screens × 15 min = ~1.5h active GPU time.
Add 5-minute idle gap between runs to allow cooling:
```bash
sleep 300 && CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 autoresearch/train_ar.py > autoresearch/run.log 2>&1
```

---

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue.
Keep forming hypotheses, running experiments, logging results.
If ar_score stops improving, try more radical changes:
- Drop to 3 layers (L0 + staging + full-attn)
- Try the ChatGPT postage-stamp offset set {1,6,8,21,60,93,104,154,378,414,539,938}
- Try removing IF entirely (training without IF — may get 0% passkey, that's informative)
- Try higher scale_embed_init to speed up phase transition
The human will interrupt you when they wake up.
