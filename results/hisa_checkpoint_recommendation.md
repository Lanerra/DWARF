# HISA Checkpoint/Replay Recommendation

## Summary

**Accepted mode:** `DWARF_CKPT=all` (full gradient checkpointing, all layers)
**Replay V1:** Implemented but NOT recommended for production (fails benchmark gates)
**V2:** Skipped — unnecessary complexity

## Key Finding

`DWARF_CKPT=all` unlocks significantly higher batch sizes on D768+ models compared
to `none` or `full_attn` checkpoint strategies. This is the primary operational win.

## Benchmark Results

### D512-L10 (BS=2, GA=1, eager)

| Mode | CKPT | RECOMPUTE | VRAM | tok/s | Δ VRAM vs none | Δ tok/s vs none |
|---|---|---|---|---|---|---|
| baseline_none | none | none | 2581MB | 76910 | -- | -- |
| trainer_full_attn | full_attn | none | 2467MB | 56346 | -4.4% | -26.7% |
| recompute_out_lse | full_attn | out_lse | 2467MB | 56858 | -4.4% | -26.0% |
| recompute_all | full_attn | all | 2467MB | 55459 | -4.4% | -27.9% |
| trainer_all | all | none | 1521MB | 38913 | -41.1% | -49.4% |

### D768-L24 (eager, various BS)

| Mode | CKPT | BS | VRAM | tok/s | Notes |
|---|---|---|---|---|---|
| none | none | 2 | 6378MB | 25039 | Max BS=2, BS=4 OOMs |
| full_attn | full_attn | 2 | 6226MB | 23369 | Max BS=2, BS=4 OOMs |
| all | all | 2 | 2525MB | 17819 | |
| all | all | 4 | 3821MB | 20737 | |
| all | all | 8 | 6692MB | 21091 | **Sweet spot** |
| all | all | 10 | 8146MB | 19293 | |
| all | all | 11 | OOM | -- | |
| all (compile) | all | 8 | 6692MB | 20799 | Compile overhead: 7s |

### D768-L31 (eager, various BS)

| Mode | CKPT | BS | VRAM | tok/s |
|---|---|---|---|---|
| all | all | 2 | 2851MB | 14236 |
| all | all | 4 | 4166MB | 17628 |
| all | all | 8 | 7173MB | 17467 |
| all | all | 16 | 13177MB | 16562 |
| all | all | 24 | 19179MB | 16118 |
| all | all | 32 | OOM | -- |

### Saved State (per HISA layer, single fwd+bwd)

| Config | Mode | Saved Bytes | Δ vs none |
|---|---|---|---|
| D512-H8 | none | 141.2MB | -- |
| D512-H8 | out_lse / all | 136.9MB | -4.3MB (-3.1%) |
| D768-H16 | none | 236.6MB | -- |
| D768-H16 | out_lse / all | 230.0MB | -6.6MB (-2.8%) |

## Benchmark Gate Analysis

### Gate 1: Peak VRAM must decrease by >=10%

- Replay V1 (`out_lse`): **FAIL** — 0% VRAM improvement vs `full_attn` baseline
- Replay V1 vs `none`: **FAIL** — only 4.4% improvement
- `trainer_all`: **PASS** — 41% improvement on D512-L10

### Gate 2: Throughput regression <=25%

- Replay V1 (`out_lse`): **FAIL** — 26.0% regression vs `none`
- `trainer_all`: **FAIL** — 49.4% regression vs `none`

However, `trainer_all` unlocks batch sizes that `none`/`full_attn` cannot achieve,
which is the operational goal.

## Decision: Why Replay V1 Does Not Ship

1. **No measurable VRAM savings vs `full_attn` baseline.** The HISA layer's saved
   `out`/`lse_out` tensors are ~4MB at D512 scale and ~6.6MB at D768 scale — a
   tiny fraction of total peak VRAM.

2. **Throughput cost is real.** Replay adds a full forward kernel re-execution in
   backward, costing ~26% throughput. Same cost as `full_attn` checkpointing
   without additional VRAM benefit.

3. **The VRAM bottleneck is not `_DSRHISAAttendFn`.** The dominant saved-state
   contributors are Q/K/V projections and routing logits, not `out`/`lse_out`.

## Decision: Why V2 Is Skipped

The remaining VRAM bottleneck is outside `_DSRHISAAttendFn` saved state. Even V2
(recomputing routing/selections) would save at most ~20MB per HISA layer at D768
scale — still insufficient for the >=10% gate, with likely >40% throughput cost.

## Recommendation

### For D768-L24 on RTX 4090 (24GB):
- **Use `DWARF_CKPT=all` with BS=8 GA=4** (eff_batch=32): 6692MB VRAM, 21091 tok/s
- This is 4x the tokens/step vs `none`/`full_attn` (which max out at BS=2)
- Compile mode: same VRAM, slightly lower tok/s (20799), 7s compile overhead

### For D768-L31 on RTX 4090 (24GB):
- **Use `DWARF_CKPT=all` with BS=16 GA=2** (eff_batch=32): 13177MB VRAM, 16562 tok/s
- Or BS=24 GA=2 (eff_batch=48): 19179MB VRAM, 16118 tok/s

### For D512-L10 on RTX 4090 (24GB):
- **Use `DWARF_CKPT=all`**: 1521MB VRAM, 38913 tok/s
- Or `DWARF_CKPT=full_attn` if throughput matters more: 2467MB VRAM, 56346 tok/s

### General:
- **Do NOT enable `HISA_RECOMPUTE` for production.** Leave it as `none` (default).
- **`DWARF_CKPT=all` is the recommended production setting** for D768+ models.
- **`DWARF_CKPT=full_attn`** is viable for smaller models where throughput matters.

## Trainers Patched

All HISA trainers received the checkpoint gate fix (strategy checked before
DSR layer veto):
- `train/train_d512_l10_hisa_h16_v2_l3.py`
- `train/train_d768_l24_hisa_hd64.py`
- `train/train_d768_l31_hisa_hd32_tied.py`
- `train/train_d512_l20_hisa_hd32.py`
- `train/train_d512_l13_hisa_hd32.py`
- (+ 9 additional HISA trainer siblings)

## Files Changed

- `kernels/hierarchical_sparse_attn_v15_hisa.py` — refactored for replay (V1 implemented, not shipped)
- `scripts/bench_hisa_checkpoint_modes.py` — benchmark harness
- `tests/test_hisa_checkpoint_policy.py` — regression test
- `tests/test_hisa_recompute_equivalence.py` — exactness harness
- `train/*hisa*.py` — checkpoint gate fix (14 files)

## Test Results

- `test_hisa_smoke.py`: 2/2 passed
- `test_hisa_checkpoint_policy.py`: 11/11 passed
- `test_hisa_recompute_equivalence.py`: 5/5 passed
- **Total: 18/18 passed**
