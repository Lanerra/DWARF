# DWARF AutoResearch

Autonomous research loop for DSQG attention architecture experiments.
Read this file fully before starting. It is updated by the human to steer research direction.

---

## Setup

To start a new experiment run:

1. **Agree on a run tag** with the user (e.g. `mar9-offsets`). Create branch:
   `git checkout -b autoresearch/<tag>` from main.

2. **Load context** — the RAG index has all prior experiment results, Holocron notes, design decisions:
   ```bash
   /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python3 /home/dlewis3/.openclaw/rag/startup_context.py
   ```
   Then query for specifics:
   ```bash
   .venv/bin/python3 /home/dlewis3/.openclaw/rag/query.py "DWARF d41 passkey results" --n 8
   .venv/bin/python3 /home/dlewis3/.openclaw/rag/query.py "softmax gain competition learnable offset" --n 5
   ```

3. **Check current results**:
   ```bash
   .venv/bin/python3 tools/compare_runs.py
   ```

4. **Initialize results.tsv** if it doesn't exist:
   ```
   commit	val_ppl	passkey_mean	peak_vram_gb	status	description
   ```
   Do NOT git-track results.tsv.

5. Confirm setup with user, then begin the experiment loop.

---

## What you CAN change

- **Sparse offset set** (`sparse_list`): choose from SPARSE_POOL = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
- **Dense width** (`dense_width`): integer, but budget = dense_width + len(sparse_list) ≤ 52 for 14M, ≤ 64 for 35M
- **Learning rate**: range [5e-5, 5e-4]
- **EMA init** (`ema_factor` init): range [0.005, 0.1]; empirically 0.03 is best
- **scale_embed init**: 0.0 (safe) or 0.5 (faster early crossing, validated at 14M)

## What you CANNOT change

- The kernel backward logic (only kernel_generator.py may produce kernels)
- The evaluation harness (`passkey_accuracy`, `evaluate`)
- The model architecture hyperparameters (D, H, L, FFN — these are fixed per scale)
- The dataset or tokenizer

---

## The metric

**Primary**: `val_ppl` after the fixed step budget (lower = better)
**Secondary**: `passkey_mean` across 6 distances [64,128,256,512,1024,1536] (higher = better)

A config **advances** (keep) if val_ppl improves. Passkey is a tiebreaker and a red flag:
- If val_ppl improves but passkey_mean drops significantly (>10pp), investigate before committing.
- If val_ppl is equal but passkey_mean improves, that is a keep.

---

## Pre-flight checks (MANDATORY before any GPU launch)

1. **Learnability check** — use `tools/offset_cli.py`:
   ```bash
   .venv/bin/python3 tools/offset_cli.py score --dense <W> --sparse <list>
   ```
   If `Min softmax weight < 1e-8`, the config is likely unlearnable within the budget. Skip it.

2. **Kernel validation** — always generate and validate:
   ```bash
   python autoresearch/kernel_generator.py --dense <W> --sparse <list> \
     --out kernels/dsqg_probe.py --validate
   ```
   If validation fails, **do not run**. Fix the config or skip.

3. **Path coverage** — check that target distances have nonzero path counts:
   ```bash
   .venv/bin/python3 tools/offset_cli.py score --dense <W> --sparse <list>
   ```
   Configs with 0 paths to d=256 and above are deprioritized (they rely on full-attention only for retrieval).

---

## The experiment loop

**The first run**: Always establish the baseline first. Run the known-best config (d41s3: dense=41, sparse=[48,128,384]) and record it. This is your baseline for all keep/discard decisions.

LOOP FOREVER:

1. **Hypothesize**: Based on current results.tsv and RAG context, choose the next config to try.
   Prefer configs that address the current bottleneck (see Research Directions below).
   Use `tools/offset_cli.py rank` to generate and rank candidates.

2. **Pre-flight**: Run learnability + kernel validation checks. Skip failures.

3. **Commit** the hypothesis:
   ```bash
   git add autoresearch/ kernels/dsqg_probe.py
   git commit -m "probe: <description>"
   ```

4. **Run probe** — redirect ALL output to log. Do NOT let training output flood your context:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python autoresearch/probe_run.py \
     --dense <W> --sparse <list> --steps 2000 --model-size 14m \
     --tag "<description>" --out autoresearch/results.tsv \
     > autoresearch/probe.log 2>&1
   ```
   **Timeout**: if the probe has not finished within 90 minutes, kill it (`kill <pid>`) and treat as crash.

5. **Read results** — grep only, do not read the full log:
   ```bash
   grep "val_ppl:\|passkey_mean:\|status:" autoresearch/probe.log | tail -5
   tail -1 autoresearch/results.tsv
   ```
   If grep output is empty, the run crashed. Run `tail -n 30 autoresearch/probe.log` to see the stack trace. Fix obvious bugs (typos, import errors) and retry once. If the idea itself is broken, log as crash and move on.

6. **Keep or discard**:
   - If val_ppl improved vs baseline: keep the commit
   - If equal or worse: `git reset --hard HEAD~1`, log as discard
   - Passkey improvement alone (with equal PPL): keep — it means the architecture found a better retrieval strategy

7. **Check events bus every 5 probes** for 35M feedback from Brico:
   ```bash
   python3 ~/.openclaw/workspace/events_cli.py poll
   ```
   If Brico posted insights about which config classes transferred well/poorly to 35M, update your strategy accordingly. Deprioritize classes Brico flagged as non-transferring.

8. **Update research directions** in this file if a significant finding warrants it.

9. Repeat.

**NEVER STOP**: Once the experiment loop has begun (after the initial baseline run), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human is asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — query the RAG, re-read Holocron notes, try combining prior near-misses, try more radical offset combinations. The loop runs until the human interrupts you, period.

---

## Research Directions (updated by human)

**Current focus**: Find the optimal sparse offset set for 14M scale (dense_width=41).

**Known results** (as of Mar 9, 2026):
- Best: d41s3 [48,128,384] → 52.457 PPL, 80.0% passkey (eval suite)
- d41s5 [48,128,384,768,1536] → 52.677 PPL, 41.7% passkey — long offsets hurt
- I3G0 reference: 52.948 PPL, 53.3% passkey
- condM reference: 54.529 PPL, 83.3% passkey

**Key constraints learned**:
- EMA window must exceed farthest sparse offset: `1/ema_factor > max(sparse_offsets)` at convergence
- Adding offsets > learnable cutoff (~99 at 16k steps) dilutes gradient without benefit
- All configs with 768 or 1536 as sparse offsets have failed empirically
- offset_cli score does not yet perfectly predict outcomes — calibrate as you gather data

**Open questions to explore**:
1. Is [48,64,96] or [48,96,128] better than [48,128,384] when all offsets are ≤ learnable cutoff?
2. Does dense_width=43 (one more) improve PPL without hurting passkey?
3. Does [48,128,256] (shorter max offset) match [48,128,384] passkey with better gradient?
4. What is the actual pareto front of (PPL, passkey) across 3-sparse configs?

**Simplicity criterion** (from Karpathy): All else being equal, simpler is better. A config with fewer offsets achieving equal results is a win.

---

## Output format (results.tsv)

Tab-separated. Header row:
```
commit	val_ppl	passkey_mean	peak_vram_gb	status	description
```
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what was tried, e.g. `dense=41 sparse=[48,96,192]`
- Do NOT commit results.tsv to git

---

## Important paths
- Training base (14M): `train/train_2048_14m_d41s3.py`
- Training base (35M): `train/train_2048_35m_d41.py`
- Kernel generator: `autoresearch/kernel_generator.py`
- Probe runner: `autoresearch/probe_run.py`
- Results: `autoresearch/results.tsv`
- Offset CLI: `tools/offset_cli.py`
- Kernel checker: `tools/check_kernel.py`
- Compare runs: `tools/compare_runs.py`
- RAG query: `.venv/bin/python3 /home/dlewis3/.openclaw/rag/query.py`
- Candidates handoff: `autoresearch/candidates_35m.json`
- Candidates schema: `autoresearch/candidates_35m_schema.md`
- 35M loop: `autoresearch/loop_35m.py`
- 35M results: `autoresearch/results_35m.tsv`
- Calibration data: `autoresearch/calibration_update.json`
- RunPod setup: `autoresearch/runpod_setup.sh`

---

## Handoff to 35M

When the 14M loop has accumulated enough confident results, hand off the best candidates
to Tier 2 (35M, RunPod H100). The 35M loop is **pure Python** — no Claude Code, no API key.

### When to generate candidates_35m.json

Generate the handoff file when ANY of these conditions are met:
- At least 5 candidates have `status=keep` in results.tsv
- The 14M search space is exhausted (all promising configs tested)
- A clear plateau is reached (last 5 runs all worse than best)

### How to generate it

1. Read `autoresearch/candidates_35m_schema.md` for the full JSON schema
2. Write `autoresearch/candidates_35m.json` with:
   - `generated_at`: current ISO timestamp
   - `14m_run_tag`: the current autoresearch branch name
   - `baseline`: the current best config from results.tsv
   - `candidates`: all `keep` configs ranked by 14M PPL (best first)
   - `calibration_notes`: free-form notes about patterns observed
3. For each candidate, set:
   - `confidence` based on offset_cli score + learnability analysis
   - All `35m_*` fields to `null`
   - `35m_status` to `"pending"`
   - `transfer_delta_ppl` and `transfer_delta_passkey` to `null`

### Minimum fields required per candidate

`rank`, `dense_width`, `sparse_list`, `14m_val_ppl`, `14m_passkey_mean`,
`14m_passkey_by_distance`, `14m_steps`, `rationale`, `confidence`, `35m_status`

### Signaling RunPod

After writing candidates_35m.json:
```bash
git add autoresearch/candidates_35m.json
git commit -m "🔄 handoff: candidates_35m.json for 35M probing"
git push origin HEAD
```
Then notify the user to start the RunPod loop:
```
tmux attach -t autoresearch
python autoresearch/loop_35m.py \
  --candidates autoresearch/candidates_35m.json \
  --steps 2000 \
  --out autoresearch/results_35m.tsv
```

---

## Reading 35M Feedback

At the **start of each new 14M run**, check for calibration data from the 35M tier.

### Check for calibration data

```bash
cat autoresearch/calibration_update.json
```

If `updated_at` is non-null, the 35M loop has completed at least one batch.

### Interpreting transfer_stats

- `ppl_delta_mean`: average (35M_ppl - 14M_ppl) across candidates.
  Negative means configs tend to perform **better** at 35M than 14M predicted.
  Positive means 14M overestimates quality.
- `ppl_rank_concordance`: how well 14M PPL ranking predicts 35M PPL ranking (0–1).
- `passkey_delta_mean`: average passkey improvement at 35M vs 14M.
- `passkey_rank_concordance`: how well 14M passkey ranking predicts 35M (0–1).

### Interpreting config_class_findings

Each entry classifies a group of configs (e.g., `sparse_all_leq_200`, `sparse_includes_384`)
and reports how reliably 14M predicts 35M for that class.

### Decision rules

| Condition | Action |
|---|---|
| `ppl_rank_concordance > 0.8` for a class | 14M results for that class are trustworthy. Increase exploration of that class. |
| `ppl_rank_concordance < 0.6` for a class | 14M is unreliable for that class. Deprioritize — only run at 35M to resolve. |
| `ppl_delta_mean < -1.0` | 14M is pessimistic. Configs may perform better at scale than expected. Widen acceptance threshold. |
| `ppl_delta_mean > 1.0` | 14M is optimistic. Tighten acceptance — only pass configs with clear 14M wins. |
| `passkey_rank_concordance < 0.5` | Passkey doesn't transfer reliably. Rely on PPL for ranking; use passkey only as red-flag detector. |

### Update internal strategy

After reading calibration data, adjust the 14M loop:
- Re-rank SPARSE_POOL priorities based on which config classes transfer reliably
- Adjust the PPL improvement threshold for `keep` vs `discard` decisions
- Note any config classes where 14M is systematically wrong
- Record strategy changes in the next `calibration_notes` when handing off
