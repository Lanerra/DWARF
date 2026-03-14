# 🔬 Layer/Interference/FullAttn Structural Analysis

**Total valid combinations analyzed:** 1221
**Champion-pattern combinations:** 75

## Section A: Reference Combo Breakdown (L=6)

**Configuration:** L=6, interference_interval=3, full_attn_layer=5

### Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| preprocessing_layers | 5 | DSQG layers before full attention |
| postprocessing_layers | 0 | DSQG layers after full attention |
| interference_block_count | 1 | Total interference blocks |
| interference_blocks_pre | 1 | IF blocks before full attention |
| interference_blocks_post | 0 | IF blocks after full attention |
| full_attn_ratio | 1.000 | Position as fraction of depth (1.0 = last layer) |
| interference_density_pre | 0.200 | IF coverage in pre-attn region |
| gap_before_full | 2 | Staging DSQG layers before full-attn |
| pre_post_ratio | 5.00 | Preprocessing to postprocessing ratio |

**Interference block positions:** [2]

### Interpretation

The reference config places full attention at the **last layer** (ratio=1.0), with all preprocessing happening before it. Interference blocks at layers 0 and 3 provide periodic cross-token mixing during the preprocessing phase, with a single staging DSQG layer (layer 4) between the last interference block and full attention.

## Section B: Champion-Pattern Combos by Layer Count

### L=4

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 4 | 2 | 3 | 1.000 | 1 | 1 | 1 | 3.00 |

### L=6

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 6 | 2 | 5 | 1.000 | 2 | 2 | 1 | 5.00 |
| 6 | 3 | 4 | 0.800 | 2 | 1 | 1 | 4.00 |

### L=7

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 7 | 2 | 5 | 0.833 | 2 | 2 | 1 | 5.00 |

### L=8

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 8 | 2 | 7 | 1.000 | 3 | 3 | 1 | 7.00 |
| 8 | 3 | 7 | 1.000 | 2 | 2 | 1 | 7.00 |

### L=9

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 9 | 3 | 8 | 1.000 | 2 | 2 | 2 | 8.00 |
| 9 | 2 | 7 | 0.875 | 3 | 3 | 1 | 7.00 |
| 9 | 3 | 7 | 0.875 | 3 | 2 | 1 | 7.00 |

### L=10

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 10 | 2 | 9 | 1.000 | 4 | 4 | 1 | 9.00 |
| 10 | 3 | 8 | 0.889 | 2 | 2 | 2 | 8.00 |

### L=11

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 11 | 3 | 10 | 1.000 | 3 | 3 | 1 | 10.00 |
| 11 | 2 | 9 | 0.900 | 4 | 4 | 1 | 9.00 |
| 11 | 3 | 8 | 0.800 | 2 | 2 | 2 | 4.00 |

### L=12

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 12 | 2 | 11 | 1.000 | 5 | 5 | 1 | 11.00 |
| 12 | 3 | 11 | 1.000 | 3 | 3 | 2 | 11.00 |
| 12 | 3 | 10 | 0.909 | 4 | 3 | 1 | 10.00 |
| 12 | 2 | 9 | 0.818 | 5 | 4 | 1 | 4.50 |

### L=13

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 13 | 2 | 11 | 0.917 | 5 | 5 | 1 | 11.00 |
| 13 | 3 | 11 | 0.917 | 3 | 3 | 2 | 11.00 |
| 13 | 3 | 10 | 0.833 | 4 | 3 | 1 | 5.00 |

### L=14

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 14 | 2 | 13 | 1.000 | 6 | 6 | 1 | 13.00 |
| 14 | 3 | 13 | 1.000 | 4 | 4 | 1 | 13.00 |
| 14 | 2 | 11 | 0.846 | 6 | 5 | 1 | 5.50 |
| 14 | 3 | 11 | 0.846 | 3 | 3 | 2 | 5.50 |

### L=15

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 15 | 3 | 14 | 1.000 | 4 | 4 | 2 | 14.00 |
| 15 | 2 | 13 | 0.929 | 6 | 6 | 1 | 13.00 |
| 15 | 3 | 13 | 0.929 | 5 | 4 | 1 | 13.00 |

### L=16

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 16 | 2 | 15 | 1.000 | 7 | 7 | 1 | 15.00 |
| 16 | 3 | 14 | 0.933 | 4 | 4 | 2 | 14.00 |
| 16 | 2 | 13 | 0.867 | 7 | 6 | 1 | 6.50 |
| 16 | 3 | 13 | 0.867 | 5 | 4 | 1 | 6.50 |

### L=17

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 17 | 3 | 16 | 1.000 | 5 | 5 | 1 | 16.00 |
| 17 | 2 | 15 | 0.938 | 7 | 7 | 1 | 15.00 |
| 17 | 3 | 14 | 0.875 | 4 | 4 | 2 | 7.00 |
| 17 | 2 | 13 | 0.812 | 7 | 6 | 1 | 4.33 |
| 17 | 3 | 13 | 0.812 | 5 | 4 | 1 | 4.33 |

### L=18

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 18 | 2 | 17 | 1.000 | 8 | 8 | 1 | 17.00 |
| 18 | 3 | 17 | 1.000 | 5 | 5 | 2 | 17.00 |
| 18 | 3 | 16 | 0.941 | 6 | 5 | 1 | 16.00 |
| 18 | 2 | 15 | 0.882 | 8 | 7 | 1 | 7.50 |
| 18 | 3 | 14 | 0.824 | 5 | 4 | 2 | 4.67 |

### L=19

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 19 | 2 | 17 | 0.944 | 8 | 8 | 1 | 17.00 |
| 19 | 3 | 17 | 0.944 | 5 | 5 | 2 | 17.00 |
| 19 | 3 | 16 | 0.889 | 6 | 5 | 1 | 8.00 |
| 19 | 2 | 15 | 0.833 | 8 | 7 | 1 | 5.00 |

### L=20

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 20 | 2 | 19 | 1.000 | 9 | 9 | 1 | 19.00 |
| 20 | 3 | 19 | 1.000 | 6 | 6 | 1 | 19.00 |
| 20 | 2 | 17 | 0.895 | 9 | 8 | 1 | 8.50 |
| 20 | 3 | 17 | 0.895 | 5 | 5 | 2 | 8.50 |
| 20 | 3 | 16 | 0.842 | 6 | 5 | 1 | 5.33 |

### L=21

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 21 | 3 | 20 | 1.000 | 6 | 6 | 2 | 20.00 |
| 21 | 2 | 19 | 0.950 | 9 | 9 | 1 | 19.00 |
| 21 | 3 | 19 | 0.950 | 7 | 6 | 1 | 19.00 |
| 21 | 2 | 17 | 0.850 | 9 | 8 | 1 | 5.67 |
| 21 | 3 | 17 | 0.850 | 6 | 5 | 2 | 5.67 |
| 21 | 3 | 16 | 0.800 | 7 | 5 | 1 | 4.00 |

### L=22

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 22 | 2 | 21 | 1.000 | 10 | 10 | 1 | 21.00 |
| 22 | 3 | 20 | 0.952 | 6 | 6 | 2 | 20.00 |
| 22 | 2 | 19 | 0.905 | 10 | 9 | 1 | 9.50 |
| 22 | 3 | 19 | 0.905 | 7 | 6 | 1 | 9.50 |
| 22 | 2 | 17 | 0.810 | 10 | 8 | 1 | 4.25 |
| 22 | 3 | 17 | 0.810 | 6 | 5 | 2 | 4.25 |

### L=23

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 23 | 3 | 22 | 1.000 | 7 | 7 | 1 | 22.00 |
| 23 | 2 | 21 | 0.955 | 10 | 10 | 1 | 21.00 |
| 23 | 3 | 20 | 0.909 | 6 | 6 | 2 | 10.00 |
| 23 | 2 | 19 | 0.864 | 10 | 9 | 1 | 6.33 |
| 23 | 3 | 19 | 0.864 | 7 | 6 | 1 | 6.33 |

### L=24

| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |
|---|--------------|-----------|-----------------|------|----------|-----|----------------|
| 24 | 2 | 23 | 1.000 | 11 | 11 | 1 | 23.00 |
| 24 | 3 | 23 | 1.000 | 7 | 7 | 2 | 23.00 |
| 24 | 3 | 22 | 0.957 | 8 | 7 | 1 | 22.00 |
| 24 | 2 | 21 | 0.913 | 11 | 10 | 1 | 10.50 |
| 24 | 3 | 20 | 0.870 | 7 | 6 | 2 | 6.67 |
| 24 | 2 | 19 | 0.826 | 11 | 9 | 1 | 4.75 |
| 24 | 3 | 19 | 0.826 | 8 | 6 | 1 | 4.75 |


## Section C: Predicted Optimal Config per Layer Count

Selection criteria: maximize full_attn_ratio → maximize n_if_pre → minimize interference_interval

| L | best_int_interval | best_full_attn | full_attn_ratio | n_if_blocks |
|---|-------------------|----------------|-----------------|-------------|
| 4 | 2 | 3 | 1.000 | 1 |
| 6 | 2 | 5 | 1.000 | 2 |
| 7 | 2 | 5 | 0.833 | 2 |
| 8 | 2 | 7 | 1.000 | 3 |
| 9 | 3 | 8 | 1.000 | 2 |
| 10 | 2 | 9 | 1.000 | 4 |
| 11 | 3 | 10 | 1.000 | 3 |
| 12 | 2 | 11 | 1.000 | 5 |
| 13 | 2 | 11 | 0.917 | 5 |
| 14 | 2 | 13 | 1.000 | 6 |
| 15 | 3 | 14 | 1.000 | 4 |
| 16 | 2 | 15 | 1.000 | 7 |
| 17 | 3 | 16 | 1.000 | 5 |
| 18 | 2 | 17 | 1.000 | 8 |
| 19 | 2 | 17 | 0.944 | 8 |
| 20 | 2 | 19 | 1.000 | 9 |
| 21 | 3 | 20 | 1.000 | 6 |
| 22 | 2 | 21 | 1.000 | 10 |
| 23 | 3 | 22 | 1.000 | 7 |
| 24 | 2 | 23 | 1.000 | 11 |

## Section D: Ratio Stability Analysis

### Full Attention Ratio

- Range across L values: [0.833, 1.000]
- Configs with full_attn at last layer (L-1): 17/20 (85.0%)
- **Stability:** ⚠️ VARIABLE (range = 0.167)

### Interference Density (Pre-Attention)

- Range across L values: [0.250, 0.478]

### Pattern Discovery

**Law 1:** Full attention is placed at the last layer (L-1) in 85.0% of optimal configs.

**Law 2:** Interference interval tends to be approximately L/2 to L/3, providing 2-3 interference blocks in the preprocessing region.

**Law 3:** Gap before full attention (staging area) is typically 1-2 layers, allowing final feature consolidation before global attention.

## Section E: Scale Extrapolation

Predicted optimal (interference_interval, full_attn_layer) for target architectures:

### 12M Parameters

| L | interference_interval | full_attn_layer | Notes |
|---|----------------------|-----------------|-------|
| 6 | 2 | 5 |  |
| 8 | 2 | 7 |  |
| 12 | 2 | 11 |  |

### 35M Parameters

| L | interference_interval | full_attn_layer | Notes |
|---|----------------------|-----------------|-------|
| 6 | 2 | 5 |  |
| 8 | 2 | 7 |  |
| 12 | 2 | 11 |  |

### 85M Parameters

| L | interference_interval | full_attn_layer | Notes |
|---|----------------------|-----------------|-------|
| 6 | 2 | 5 | |
| 12 | 2 | 11 | |
| 16 | 2 | 15 | |

### 200M Parameters

| L | interference_interval | full_attn_layer | Notes |
|---|----------------------|-----------------|-------|
| 12 | 2 | 11 | |
| 16 | 2 | 15 | |
| 24 | 2 | 23 | |