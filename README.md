# DWARF

**Dyadic Wave And Resonant Field Attention** — a hybrid sparse/dense attention architecture combining O(1)-KV-cache DSQG layers with a single full causal attention layer, trained jointly from initialization.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is DWARF?

DWARF replaces most of a transformer's attention layers with **DSQG** (Dyadic Sparse Q-K Gather) layers that attend to a fixed set of 44 sparse offsets — a dense local window plus semi-dyadic long-range taps. Because the offset set is fixed regardless of sequence length, each DSQG layer's KV cache at inference is a fixed-size circular buffer: **O(1) memory, not O(N)**.

One standard full causal attention layer remains, providing global context binding that sparse offsets alone cannot supply. The two layer types co-train from initialization: gradient signal from the full attention layer teaches the DSQG layers what to preprocess for it. This co-training is load-bearing — the preprocessing advantage is zero at epoch 1 and emerges entirely through joint training.

**condM** is the baseline hybrid: DSQG layers with ALiBi-style position biases and causal interference pooling.

**condU** extends condM with a physics-derived stack: Q-weighted scale gains (matched-filter retrieval), per-head IF amplifier gains, and Huygens-principle K/V injection at interference layers. These additions are derived from six signal-processing frameworks — heterodyne receivers, IF amplifiers, Huygens principle, matched filtering, phase-locked loops, and coherent field retrieval — that describe the same computational operations the architecture performs.

---

## Results

### Headline numbers

| Model | Params | Test PPL | Passkey mean | Benchmark mean |
|---|---|---|---|---|
| Standard 13M | 21.6M | 64.07 | -- | 31.3% |
| condM 13M | 14.0M | 52.88 | 46.7% (5-sample) | 32.0% |
| condU 13M | 14.1M | 52.24 | 38.3% (50-sample) | 31.7% |
| Standard 85M | 101.4M | 39.45 | -- | 31.5% |
| condU 35M hybrid | 38.7M | 38.54 | 85.0% (5-sample) | 32.6% |
| condM 85M | 88.3M | 36.04 | -- | 32.5% |

Passkey task: next-word prediction of a memorized key buried in a 2048-token context, evaluated at 12 retrieval distances (d=1..1536). 50-sample measurements are reported where available; 5-sample results carry high variance and are noted.

Benchmark mean: average accuracy across HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and LAMBADA.

**condM 85M** (88.3M params) beats the standard 85M baseline (101.4M params) by **3.4 PPL** with **13% fewer parameters**. The condU 35M hybrid (38.7M params) beats the standard 85M on benchmark accuracy (32.6% vs 31.5%) at 62% fewer parameters.

### External benchmark detail (6-task suite)

| Model | PPL | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | LAMBADA | Mean |
|---|---|---|---|---|---|---|---|---|
| Standard 13M | 64.07 | 24.9% | 54.3% | 30.4% | 19.7% | 52.3% | 6.3% | 31.3% |
| condM 13M | 52.88 | 24.5% | 55.9% | 34.0% | 20.1% | 51.1% | 6.2% | 32.0% |
| condU 13M | 52.24 | 24.5% | 54.1% | 33.3% | 21.1% | 50.0% | 7.4% | 31.7% |
| Standard 85M | 39.45 | 24.9% | 54.9% | 28.9% | 18.4% | 52.2% | 9.8% | 31.5% |
| condU 35M hybrid | 38.54 | 24.7% | 56.3% | 34.6% | 18.1% | 50.7% | 10.9% | 32.6% |
| condM 85M | 36.04 | 25.1% | 55.0% | 31.2% | 19.7% | 51.1% | 13.0% | 32.5% |

All 13M models cluster within 1.6% mean benchmark accuracy regardless of PPL or passkey. LAMBADA is the exception: it is the only task in the suite sensitive to memory mechanism type at 13M scale, because it requires content-addressed lookup to predict the final word of a passage from earlier context. condU 35M (10.9%) and condM 85M (13.0%) show clear improvement, consistent with scale enabling genuine retrieval.

### Ablation series (N=2048, 13M scale)

Six offset-set ablations on the condU 13M baseline, each changing one variable:

| Run | Key change | Test PPL | Passkey mean | Finding |
|---|---|---|---|---|
| condU baseline | V3 offsets, linspace init | 52.24 | 38.3% (50-sample) | Reference |
| Run C | Interleaved offset set | 52.86 | 45.0% (5-sample) | Fills d=32 gap; best custom passkey |
| Run D | Octave blocks, near-zero init | 55.49 | 35.0% (5-sample) | PPL penalty from sparse short-range |
| Run E | Dense 0-39, near-zero init | 52.44 | 25.0% (5-sample) | Best PPL of custom runs; passkey destroyed |
| Run F | Dense 0-15, gap 16-31, octave blocks | 52.73 | 21.7% (5-sample) | Cliff-edge commitment; gradient starvation |

Meta-finding: all six ablations failed to improve on the condU baseline simultaneously on both PPL and passkey. The V3 design (dense 0-31 + semi-dyadic long-range + linspace init) was near-optimal from physics derivation. The ablation series is mechanistic confirmation of why it works, not a search that found it.

### Layer position ablation (condM 13M)

| Full-attention layer position | Test PPL | Passkey mean | Notes |
|---|---|---|---|
| Layer 0 | 70.08 | 8.3% | No preprocessing; uniquely bad |
| Layer 3 | 54.54 | 66.7% | Equivalent PPL; stronger OOD long-range |
| Layer 5 | 54.53 | 83.3% | Best passkey; direct output path |

Layer 0 is qualitatively different from layers 1-5. Layers 1-5 produce equivalent final PPL; the position only affects retrieval and long-range generalization. At least one preprocessing DSQG layer is required before full attention.

---

## Architecture

### DSQG offset set (V3, N=2048)

```
Dense local:    delta = 0, 1, 2, ..., 31          (32 offsets)
Semi-dyadic:    delta = 48, 64, 96, 128, 192,
                        256, 384, 512, 768,
                        1024, 1536               (11 offsets)
Total: 43 unique offsets (+1 for delta=0 self-attention)
```

Long-range offsets follow a semi-dyadic pattern (two taps per octave), giving maximum path diversity per offset budget. The dense 0-31 local window ensures the delta=1 copy attractor competes against 31 peers rather than dominating.

### DSQG attention (one layer)

```
QKV projection: x -> Q, K, V  [D x D, no bias]
For each offset delta in {0, 1, ..., 1536}:
    score[delta, h, n] = (Q[h, n] . K[h, n-delta]) / sqrt(d_head)
                       + pos_bias[delta, h]           # ALiBi-style learned bias
                       + Q[h, n] . scale_embed[delta]  # Q-weighted scale gain (condU only)
    [mask: delta > n positions set to -inf for causality]
weight = softmax(score) over 43 offsets
output = sum_delta weight[delta, h, n] * V[h, n-delta]
output = out_proj(attn_output * sigmoid(gate_proj(x)))  # gate bias init to 0
x = x + output
x = x + FFN(LayerNorm(x))
```

The gate bias is initialized to 0 (sigmoid output near 0.5 at init) to prevent early gate saturation. This is not "forced field retrieval" — the residual connection means information always bypasses through x regardless.

### condM interference (pooling layers)

```
pool = causal_mean(x)   # cumulative mean up to position n; O(N), vectorized
x += sigmoid(gate(x)) * proj(pool)
```

Adds distributional context from the sequence prefix to the residual stream. Model can suppress via gating.

### condU interference (Huygens K/V injection)

```
pool = causal_mean(x)
k_delta = inter_k_proj(pool)
v_delta = inter_v_proj(pool)
k = k + k_delta
v = v + v_delta
```

Injects prefix context directly into K and V, modifying what attention retrieves. Model cannot gate this out. Differs fundamentally from condM's residual addition.

### condU IF amplifier

```
if_gain: [num_heads]  # per-head scalar, ones-init
output_per_head *= if_gain[h]
```

Learned per-head gain normalization. At convergence: most-global heads (low pos_bias slope) amplify (gain > 1.0); most-local heads (high pos_bias slope) attenuate (gain < 1.0). Convergence is stable — gains freeze by epoch 7 of 10 and do not drift thereafter.

At 13M, final gains (50-sample, 10-epoch run):
```
h0 (most global): 1.030
h1:               1.026
h2:               0.994
h3:               0.987
h4:               0.947
h5:               0.946
h6:               0.970
h7 (most local):  0.949
```

### Layer stacks

**condM 85M** (D=640, H=8, d_head=80, L=12):
```
[DSQG, DSQG, DSQG+pool, DSQG, DSQG, DSQG+pool, DSQG, DSQG, DSQG+pool, DSQG, DSQG, Full]
```

**condU 13M** (D=256, H=8, d_head=32, L=6):
```
[DSQG, DSQG, DSQG+HK/V, DSQG, DSQG, Full]
```

**condU 39M** (D=512, H=8, d_head=64, L=6):
```
[DSQG, DSQG, DSQG+HK/V, DSQG, DSQG, Full]
```

**condU 85M** (D=768, H=12, d_head=64, L=8) — *negative result: Huygens K/V injection memorizes training signatures at this scale; see paper §7.3*:
```
[DSQG, DSQG, DSQG, DSQG+HK/V, DSQG, DSQG, DSQG, Full]
```

### Inference KV cache

DSQG layers: circular buffer of depth max(offset) = 1,536 tokens per layer. Fixed size regardless of context length. For a 7B model at 100K context: standard attention ~52 GB KV cache; DSQG layers ~1.5 GB fixed.

The full causal attention layer retains a standard O(N) KV cache. For a purely O(1) deployment, this layer can be windowed at inference or replaced with DSQG at larger scales where per-layer retrieval capacity is sufficient.

---

## Key Findings

**Co-training is load-bearing.** The DSQG preprocessing advantage is zero at epoch 1 — all layer positions produce identical initial PPL. The full advantage emerges through joint training. Memory systems cannot be retrofitted onto pretrained backbones; the gradient pathway between provider and consumer must form from initialization. DeepSeek Engram finds the same constraint from the other direction.

**Content-gated routing is non-negotiable.** Without Q-K inner products at each offset (i.e., pure position-based routing), DWARF produces ~99 PPL. Adding full Q-K formulation dropped PPL by 14.3 in one change. This is the single most important architectural decision in the history of the project.

**PPL is insufficient to evaluate DSQG architectures.** Two documented cases: (1) condL collapsed to a delta=1 copy attractor while PPL was still improving; the collapse was visible in generation before PPL degraded. (2) condP at 65.1 PPL generates coherent sentences; standard 13M at 64.1 PPL (0.99 PPL better) generates severe word-level copy loops. Copy attractor exploitation inflates the standard transformer's PPL. Generation samples at every checkpoint are required.

**Distraction by density.** Dense local offsets (Run E: 0-39) give the model so many equivalent retrieval paths that no specific offset receives enough gradient to commit to long-range retrieval. The sparse semi-dyadic long-range structure forces selectivity. Run E had the best PPL of all ablations; it also had the worst passkey.

**Delta-effective metric.** For a passkey task with cue length C and intro tail T, the retrieval offset needed is delta_eff = d + C + T (not just d). V3's success at d=32 (60% passkey in Run C) was correctly predicted by verifying that delta=39 (= 32 + 5 + 2) is directly covered by the interleaved offset set. The aggregate path count metric is wrong; direct 1-hop delta_eff coverage is the correct predictor.

**Co-adaptive division of labor.** When a full attention layer co-trains with DSQG layers, the two systems spontaneously divide the retrieval task: full attention handles short-range, DSQG claims long-range. The boundary and specialization are not designed in; they emerge from gradient competition. Same principle as condM: the architecture must co-train for division of labor to form.

**IF gain specialization is monotonic and stable.** At 13M with 50-sample passkey measurement, per-head IF gains converge by epoch 7 to a smooth gradient from most-global (amplifying) to most-local (attenuating). The specialization does not drift after convergence. This is consistent with the IF amplifier physics: global heads need to amplify weak long-range signals; local heads can attenuate without loss.

**Scale changes the Pareto frontier.** At 13M, all architectures cluster within 1.6% on the 6-task benchmark suite regardless of PPL or passkey — the suite cannot distinguish memory mechanism types at small scale. The retrieval/PPL tradeoff becomes measurable at 35M+ and is a hard engineering constraint at 7B+ context lengths where O(1) KV cache is necessary.

---

## Training

All training uses FineWeb-Edu (sample-10BT) with the condI BPE tokenizer (36K vocab, 2048 sequence length). Token budget is Chinchilla-normalized per model size: each model is trained for 10 epochs with per-epoch sequence counts set so that epoch 7 approximately reaches 100% Chinchilla-optimal (20 x N tokens). This removes the tokens-per-parameter confound from cross-scale comparisons.

```
MAX_TRAIN_SEQS = int(20 * n_params / (7 * 2048))

13M:  19,611 seqs/epoch  [Chinchilla optimal at epoch 7]
39M:  54,031 seqs/epoch  [Chinchilla optimal at epoch 7]
85M: ~125,000 seqs/epoch [Chinchilla optimal at epoch 7; requires 300K docs]
```

Note: the ablation series (condM, condU, Runs A-F, condV, condW at 13M scale) used a fixed 52,716 seqs/epoch for all runs. These are valid comparisons within scale but are not Chinchilla-normalized. The 13M condU Chinchilla-normalized rerun (this README reflects its results) reproduces the original to within 0.03 PPL.

### Local training

```bash
git clone https://github.com/Lanerra/DWARF.git
cd DWARF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# condU 13M (~6 min/epoch on RTX 4090)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_condU.py \
  2>&1 | tee logs/condU_run.log

# condM 85M (reference architecture, ~45 min/epoch on RTX 4090)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_condM_layer_ablation.py \
  2>&1 | tee logs/condM_85m_run.log
```

### RunPod (H200, 85M condU)

```bash
git clone https://github.com/Lanerra/DWARF.git /root/DWARF
cd /root/DWARF
pip install -r requirements.txt

# condU 85M with 300K docs (~Chinchilla optimal at epoch 7)
python -u train/train_2048_85m_condU.py 2>&1 | tee logs/85m_condU_run.log
```

---

## Evaluation

```bash
# External benchmarks (HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, LAMBADA)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 evals/eval_external.py \
  --arch condu_13m \
  --checkpoint checkpoints/condU/best.pt \
  --label "condU 13M"

# Available --arch values: condu_13m, condu_27m, condu_35m, condm_13m, standard_13m
```

---

## Rust Verification

Mathematical properties of all architecture components are verified in Rust before training runs.

```bash
cd verification
PATH="$HOME/.cargo/bin:$PATH" cargo test --release -- --nocapture
```

182 tests pass. 3 known failures in `cond_d_db4` (KdV nonlinear instability — condD was an abandoned experiment; failures document its instability, not a bug).

Verified modules include: DSQG algebraic equivalence and causality, Huygens K/V injection, IF amplifier gain and SNR, coherent scale retrieval (Q-weighted matched filter), offset optimizer and delta-effective coverage metric, receiver chain interaction (heterodyne + AGC + PLL + Kalman), beamforming coherence, and full Run E/F offset-set analyses.

---

## Repository Structure

```
DWARF/
├── train/                           # Training scripts
│   ├── train_2048_condU.py          # condU 13M (current best at 13M)
│   ├── train_2048_27m_condU.py      # condU 39M
│   ├── train_2048_85m_condU.py      # condU 85M (in training)
│   ├── train_2048_condM_layer_ablation.py  # condM (reference hybrid)
│   ├── train_2048_27m_condM.py      # condM 27M
│   ├── train_2048_standard_baseline.py     # standard transformer reference
│   ├── train_2048_condV.py          # condV (condM + RMSNorm/SwiGLU/RoPE)
│   ├── train_2048_condW.py          # condW (pure DSQG, no full attention)
│   └── ...                          # ablation series (condN, condP, condX, Runs A-F)
├── evals/
│   └── eval_external.py             # 6-task benchmark evaluator
├── kernels/
│   ├── dsqg_attention_v3.py         # Triton kernel (N=2048, 43 offsets)
│   ├── dsqg_attention_v4.py         # Triton kernel (N=4096, 46 offsets)
│   └── ...                          # ablation kernels (Runs C-F)
├── verification/                    # Rust verification crate (182 passing tests)
│   └── src/
│       ├── huygens_kv_injection.rs
│       ├── if_amplifier.rs
│       ├── coherent_scale_retrieval.rs
│       ├── offset_optimizer.rs
│       ├── dsqg.rs
│       └── ...
├── rag/
│   ├── ingest.py                    # build/update ChromaDB semantic index
│   ├── query.py                     # semantic search CLI
│   ├── startup_context.py           # session startup: recent activity summary
│   └── chroma_db/                   # persistent vector store
├── logs/                            # training logs, eval results (JSON)
├── checkpoints/                     # saved model checkpoints
├── results/
│   └── 2048_condI_tokenizer.json    # shared BPE tokenizer (36K vocab)
├── requirements.txt
└── README.md
```

---

## Status

Completed:
- condM and condU at 13M scale, full ablation series
- condM and standard at 27M and 85M scale
- condU at 39M Chinchilla-normalized (38.293 PPL, 90.0% passkey)
- condM at 85M Chinchilla-normalized (36.042 PPL, 83.3% passkey) — headline result
- 6-task external benchmark sweep across all models
- Rust verification crate (182 tests)
- 50-sample passkey infrastructure
- Paper complete, arXiv submission in progress

Negative results (documented in paper):
- condU 85M: Huygens K/V injection exhibits scale-dependent memorization pathology at ≥85M parameters; train loss converges, val PPL catastrophic (~3000). Component-level failure, not architectural — condM 85M trains correctly. See paper §7.3 for root cause and proposed fix.

Planned:
- External benchmark sweep for condU 39M
- Few-shot ICL and copy task evaluation on existing checkpoints

---

## Citation

Paper submitted to arXiv (cs.LG). Citation will be updated with arXiv ID on acceptance.

```bibtex
@misc{lewis2026dwarf,
  author       = {Lewis, Dennis},
  title        = {{DWARF}: Dyadic Wave And Resonant Field Attention with {O}(1) {KV} Cache},
  year         = {2026},
  howpublished = {\url{https://github.com/Lanerra/DWARF}},
  note         = {arXiv preprint, cs.LG. Preprint at https://arxiv.org/abs/[ID TBD]}
}
```

---

## License

Copyright 2026 Dennis Lewis. Licensed under the [Apache License 2.0](LICENSE).

The research process from condA through the current condU architecture, including all negative results, intermediate architectures, and ablation findings, is documented in full and will be released under the same license. The path matters as much as the destination.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Lanerra/DWARF)
