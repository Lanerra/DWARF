# DWARF

**Dyadic Windowed Attention with Resonant Fusion** — a hybrid sparse/dense attention architecture combining O(1)-KV-cache DSQG layers with a single full softmax attention layer, trained jointly from initialization.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is DWARF?

DWARF is a 5:1 hybrid transformer: five **DSQG** (Dyadic Sparse Q·K Gather) layers followed by one standard full causal softmax attention layer. The two layer types are co-trained from scratch — gradient signal from the full attention layer teaches the DSQG layers what to provide.

**Architecture:** `[DSQG, DSQG, DSQG, DSQG, DSQG, Full]`

**DSQG layers.** Each DSQG layer attends to a fixed set of 44 sparse offsets: a dense local window (δ=0..32) plus semi-dyadic long-range taps (δ=48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536). Attention is content-addressed via Q·K inner products over the sparse offsets, with learned ALiBi-style position biases. Because the offset set is fixed regardless of sequence length, the KV cache at inference is a fixed-size circular buffer of depth 3,072 tokens — **O(1) KV cache** regardless of context length.

**Full attention layer.** One standard causal softmax attention layer (O(N²)) provides global context binding — exact, full-sequence content retrieval that DSQG's bounded window cannot provide alone.

**The co-training principle.** The DSQG layers are not preprocessing in a generic sense — they learn to preprocess *specifically* for what the full attention layer needs, driven by shared gradient signal. The evidence: at epoch 1 (before co-training has established this pathway), all layer positions produce identical PPL. The preprocessing advantage emerges entirely through training. This mirrors DeepSeek Engram's finding from the other direction: Engram works when trained from scratch; retrofitting memory modules onto pretrained backbones fails because the provider has never received gradient signal from the consumer.

**Headline results:**
- **condM 13M** (13.98M params): **54.53 PPL** — beats standard 13M transformer by **9.54 PPL**
- **condM 27M** (26.46M params): **44.5 PPL** — beats pre-V4 100M standard transformer with **26% of the parameters**

---

## Key Properties

- **5/6 layers O(1) KV cache at inference.** The five DSQG layers require only the last 3,072 tokens of K,V — a fixed-size circular buffer regardless of context length. For a 7B model at 100K-token context: standard attention requires ~52 GB KV cache; DWARF requires ~1.5 GB fixed. The full attention layer adds standard O(N) KV cache for one layer only.
- **Sparse attention is immune to softmax dilution.** DSQG softmax is always over 44 offsets regardless of sequence length. Tokens beyond offset 1536 can never dilute the attention distribution. This is a deployment advantage that grows with context length.
- **Co-training is load-bearing.** DSQG preprocessing advantage is zero at epoch 1 (null result); it emerges entirely from joint training. This is both an empirical finding and a design constraint: the architecture cannot be retrofitted onto pretrained backbones.
- **Beats standard transformers at every scale tested.** condM 13M outperforms standard 13M by 9.54 PPL with the same parameter budget. condM 27M outperforms the pre-V4 100M standard with 26% of the parameters.
- **DSQG generation quality superior at small scale.** Standard 13M transformers exhibit severe word-level copy attractors at greedy decoding from the first training epoch. condM avoids this: the sparse offset structure forces each offset to carry genuine gradient signal rather than letting copy dominate.

---

## Architecture

### condM — current best architecture

```
Layer stack: [DSQG₀, DSQG₁, DSQG₂*, DSQG₃, DSQG₄, Full₅*]
  * interference pooling (causal mean) at layers 2, 5

DSQG layer:
  QKV projection → Q, K, V  [bias=False; gate projection keeps bias]
  offsets Δ = {0,1,...,32} ∪ {48,64,96,128,192,256,384,512,768,1024,1536}  (44 total)
  score_{δ,h,n} = (Q_{h,n} · K_{h,n-δ}) / √d_head  +  pos_bias_{δ,h}
  weight = softmax(score) over all 44 offsets  [causal: mask δ > n]
  output = Σ_δ weight_{δ,h,n} · V_{h,n-δ}
  gate   = sigmoid(W_gate · x + b)  [b initialized to 2.0 → gate ≈ 0.88]
  output = gate ⊙ out_proj(output)

  At interference layers (2, 5):
    pool = causal_mean(x)     [cumsum / position; vectorized, O(N)]
    x   += sigmoid(W_gate · x) ⊙ W_proj(pool)

Full causal attention layer (layer 5):
  Standard scaled dot-product attention  [O(N²), full context]
  Applied after 5 DSQG layers have preprocessed the representation

Shared:
  pos_bias: ALiBi-style init — −log(1 + max(δ, 0.2)) · α_h
            δ=0 initialized with small negative bias to reduce identity attractor
  Norms: LayerNorm (baseline) / RMSNorm (condM-v2)
  FFN: GELU with 4D hidden (baseline) / SwiGLU with 8D/3 hidden (condM-v2)

Parameters (13M variant, tied embeddings):
  condM baseline: 13,984,480
  condM-v2:       13,439,200  (RMSNorm, no QKV biases, no abs pos embed)

Position encoding:
  DSQG layers: relative via learned pos_bias (ALiBi-style)
  Full attention: absolute pos embed P[pos] (baseline) / RoPE (condM-v2)
```

### condM-v2 — freebie improvements bundle (in training)

condM-v2 adds training hygiene improvements with no architectural capability change:
- **RMSNorm** replaces LayerNorm (no bias parameter, ~15% faster kernel)
- **SwiGLU FFN** replaces GELU (hidden_dim=682 = 8D/3 for iso-parameter match; selectivity gating)
- **RoPE** in full attention, drops absolute P[pos] (purely relative position encoding throughout)
- **Vectorized causal mean pooling** (cumsum, O(N), replaces Python loop EMA)
- QKV/out_proj **bias=False** (gate_proj retains bias=2.0)
- **Scaled residual init**: out_proj and FFN down_proj σ = 0.02/√(2L) (prevents residual variance growth)
- **Embedding output scaling** × √D (corrects variance mismatch for tied embeddings)
- **LR warmup**: linear 1% of steps → cosine decay
- **bf16 autocast**, **cudnn.benchmark**, **AdamW param groups** (0.0 decay on norms/biases/pos_bias)

Expected improvement: ~1–3 PPL over baseline (primarily SwiGLU + RoPE).

### condN / condP — pure DSQG architectures (prior best)

Pure DSQG without a full attention layer. The stepping stones that led to condM.

```
condN: offsets {0..32} ∪ {48,64,96,128,192,256,384,512,768,1024,1536}  (44 offsets)  → 70.8 PPL
condP: offsets {0..64} ∪ {96,128,192,256,384,512,768,1024,1536}         (74 offsets)  → 65.057 PPL
```

condP's extended dense window (0-64) gives it direct 1-hop access to lag positions 33-47 that condN misses, at the cost of reduced long-range modeling capacity. Per-position loss eval confirms condP's advantage at positions 33-47 is directly measurable.

---

## Results

### Main ablation table (13M scale, OpenWebText, 2048 seq len, 10 epochs, 32k BPE vocab)

| Condition | Params | PPL | Key change |
|---|---|---|---|
| Standard transformer 13M | 21.6M | 64.07 | Reference — greedy generation: copy loops from epoch 1 |
| condA (FFT baseline) | ~13M | 86.8 | Baseline DWARF wave field |
| condB (+ dispersion) | ~13M | 87.9 | +1.1 PPL — dispersion noise at 13M |
| condC (causal Morlet) | ~13M | 87.2 | +0.4 PPL vs condA |
| condG (D4 DWT, no Q-gate) | ~13M | 99.4 | D4 introduced; j₀ collapse without Q-gate |
| condI (Q-scale pre-gather) | ~13M | 93.3 | Partial Q·K formulation |
| condJ (full OPWF Q·K gather) | ~13M | 85.1 | Key breakthrough: content-gated routing |
| condK (+ pos bias + ELU + RG init) | ~13M | 84.7 | condJ ablation, pooling removed |
| condK+RP (+ interference pooling) | ~13M | 83.3 | Pooling confirmed −1.6 PPL |
| condL+RP (24 dyadic, no dense) | ~13M | *terminated ep4* | δ=1 structural collapse; PPL/generation divergence |
| condN (dense-32 + dyadic DSQG) | ~14M | 70.8 | Eliminates δ=1 collapse basin |
| condP (dense-64 + dyadic DSQG) | ~14M | 65.057 | Best pure DSQG — within 0.99 PPL of standard |
| **condM 13M (5:1 hybrid)** | **13.98M** | **54.53** | **+9.54 PPL over standard 13M** |

### Multi-scale results

| Model | Params | Test PPL | Notes |
|---|---|---|---|
| Standard 13M | 21.6M | 64.07 | Greedy: copy loops |
| condM 13M | 13.98M | **54.53** | Beats standard 13M by 9.54 PPL |
| condP 27M | 26.78M | 52.756 | Pure DSQG at scale |
| Standard 27M | 37.97M | 50.683 | Greedy: copy loops all epochs |
| **condM 27M** | **26.46M** | **44.5** | **Beats pre-V4 100M standard (45.96) with 26% of parameters** |
| Pre-V4 100M standard | ~100M | 45.96 | Reference |
| condA 85M | ~85M | 57.7 | Wave field at scale |
| condB 85M | ~85M | 58.1 | +0.4 PPL dispersion (vs +1.1 at 13M — penalty halves at scale) |

### Layer position ablation (condM 13M variants)

Tests where to place the full attention layer within the 6-layer stack.

| Layer position | Architecture | Test PPL | Passkey (mean) | Finding |
|---|---|---|---|---|
| Layer 0 | [Full, DSQG×5] | 70.078 | 8.3% | No preprocessing → uniquely bad |
| Layer 3 | [DSQG×3, Full, DSQG×2] | 54.54 | 66.7% | Equivalent PPL; better OOD long-range |
| Layer 5 | [DSQG×5, Full] | **54.529** | **83.3%** | Best passkey; direct path to output |

**Key finding:** Full attention placement within positions 1–5 is equivalent in final PPL (difference ≤ 0.01). Layer 0 is uniquely bad — without any preprocessing DSQG layers, the co-training gradient pathway cannot establish. Layer 5 preferred for retrieval-intensive tasks (best passkey). Layer 3 shows +11 PPL advantage at 1536-2046 token context on wikitext-103 (OOD long-range generalization).

The transition from "bad" (layer 0) to "good" (layers 1–5) is a threshold effect, not a gradient. This is direct evidence that **at least one preprocessing DSQG layer is required** — the co-training pathway requires some DSQG processing before the full attention layer.

### Passkey retrieval (10-way classification, 5 trials per distance)

| Model | d=1 | d=4 | d=16 | d=64 | d=256 | d=512 | d=1024 | d=1536 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| condM layer0 | 20% | 20% | 20% | 0% | 0% | 0% | 0% | 0% | 8.3% |
| Standard 27M | 40% | 40% | 20% | 20% | 40% | 20% | 40% | 20% | 31.7% |
| condP 27M | 100% | 40% | 60% | **0%** | 0% | 0% | 0% | 0% | 36.7% |
| condM layer3 | 60% | 60% | 100% | 100% | 80% | 40% | 40% | 20% | 66.7% |
| condM 27M | 100% | 100% | 60% | 60% | 80% | 40% | 60% | 20% | 71.7% |
| **condM layer5** | **100%** | **100%** | **100%** | **100%** | **100%** | **60%** | **40%** | **20%** | **83.3%** |

**condP passkey cliff:** 100% at d≤2, drops to 0% at exactly d=64 — the dense-64 offset boundary. condP has direct 1-hop coverage to d=64 but cannot retrieve beyond it. condM recovers long-range retrieval via the full attention layer.

**condM layer3 short-distance degradation:** Drops to 60% at d=1–8 (vs 100% for layer5). Two post-attention DSQG layers partially override short-distance retrieval signal, substituting distribution-dominant tokens. Recovers to 100% at d=16–64.

---

## Generation Quality

**Critical finding: PPL is insufficient to evaluate DWARF architectures.** Two documented cases of PPL/generation divergence:

1. **condL+RP (ep4):** PPL improved while all heads converged on δ=1 copy-previous-token. Generation was degenerate before PPL degraded. Generation sampling at every epoch is required.

2. **condP vs standard transformer:** Standard 13M at PPL 64.07 produces severe word-level copy loops at greedy decoding from epoch 1 onward. condP at 65.057 PPL (0.99 PPL worse) produces coherent sentences throughout. The standard transformer's PPL lead is partially attributable to copy attractor exploitation, not genuine language modeling.

**Architecture-determined attractor taxonomy (13M scale):**

| Architecture | Attractor type | Example |
|---|---|---|
| Standard transformer 13M | **Word-level copy** | "stormy stormy stormy..." — degenerate |
| condN (dense-32) | **Concept-level** | "the length of the diagonal, the length of the diagonal" — geometrically adjacent concept |
| condP (dense-64) | **Narrative-phrase** | "the idea of a new person is to be a new person" — higher abstraction |
| condM 13M | **Mild copy, coherent at T≥0.5** | Greedy: some word loops; T=0.7: coherent political/academic sentences |
| condM 27M | **Near-coherent greedy** | "cap" filler attractor (not copy); T=0.7: coherent multi-clause sentences |

Architecture determines *which semantic tier* becomes the attractor anchor. condM's copy attractors are milder than the standard transformer's and dissolve at T=0.5–0.7. Scale further reduces attractor depth: condM 27M greedy is qualitatively better than condM 13M.

---

## Key Findings

**Content-gated routing is non-negotiable.** Without Q·K inner products at each offset tap (condG→condI), DWARF reduces to position-only routing (~99 PPL). Adding full Q·K formulation (condJ) drops PPL by 14.3 in one change. This is the single most important architectural decision.

**Memory and intelligence must co-train.** DSQG preprocessing layers show zero advantage at epoch 1 — all layer positions (0, 3, 5) produce identical initial PPL (~162). The 15.55 PPL advantage of preprocessing (layer 5 vs layer 0 at test) is entirely learned through co-training. This mirrors DeepSeek's Engram finding: Engram works when trained from scratch; retrofitting onto pretrained models fails because the gradient pathway between provider and consumer never forms.

**The hybrid is qualitatively different from pure DSQG.** condM 13M beats standard 13M by 9.54 PPL. condP 13M (the best pure DSQG) is within 0.99 PPL of the standard. The full attention layer provides global context binding that DSQG's bounded window cannot approximate — it is not a marginal improvement but a qualitative architectural change.

**Coverage density as structural regularization.** Sparse dyadic offsets without dense local coverage create structural collapse attractors (condL+RP: δ=1 dominates, terminated ep4). condN's dense-32 region puts δ=1 in competition with 32 peers, eliminating the collapse basin. This is an architectural constraint, not a tuning knob.

**PPL/generation divergence is structural.** Two clean cases show that specific architectural choices create failure modes invisible in PPL. condL+RP: collapse happened before PPL degraded. Standard transformer: copy attractor is observable from generation but not from PPL alone. Generate samples at every epoch checkpoint.

**DSQG dilution immunity.** Softmax in DSQG is always over 44 offsets regardless of sequence length. Tokens beyond offset 1536 cannot dilute the attention distribution. This is a deployment advantage that grows with context length — the attention quality at position 1 is identical whether the sequence has 128 or 100K tokens.

**Semi-dyadic offset structure.** The long-range offsets follow {3·2^k, 4·2^k} for k=4..9, giving two samples per octave. "Dyadic" in the name refers to the dyadic scale spacing; the actual pattern is semi-dyadic (two taps per octave, ratios ×1.33 and ×1.5).

**Bounded KV cache at inference.** DSQG layers only need the last 3,072 tokens of K,V at inference — a fixed-size circular buffer. For a 7B model at 100K tokens: standard attention ~52 GB KV cache; DWARF ~1.5 GB fixed. The full attention layer adds one standard KV cache (O(N)) for the global binding layer.

**Per-position loss structure is dataset-driven, not architecture-driven.** Fine-grained per-position loss on wikitext-103 shows the same banded loss structure across condM and the standard transformer — confirming these are text distribution artifacts, not architectural support-hole effects.

---

## Evaluation Infrastructure

`benchmarks/eval_suite.py` — unified benchmark script (canonical for paper results):
- Calibration (entropy, effective vocabulary, top-1 confidence)
- Distance-conditioned PPL on wikitext-103 (6 distance bands, 0–2047)
- Passkey retrieval (10-way classification, 12 distances, 5 trials each)
- Few-shot copy accuracy

```bash
# Run full eval suite on a registered model
CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_suite.py --model condm_layer5

# Available models: standard_13m, standard_27m, condp_13m, condp_27m,
#                   condm_layer0, condm_layer3, condm_layer5, condm_27m
```

`benchmarks/eval_position_loss.py` — fine-grained per-position loss (2047 positions):
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_position_loss.py --all
```

---

## 85M Scale (Planned)

condM 85M is the next planned run (RunPod H100). Pre-registered prediction: **41–46 PPL**.

Basis: condM 27M beats condM 13M by 10 PPL (54.5 → 44.5). Scaling from 27M to 85M at the same improvement ratio would project ~37–40 PPL — likely optimistic. The pre-V4 100M standard at 45.96 PPL is the primary comparison point; condM 85M should beat it by approximately 4–8 PPL based on the pattern at 27M.

Blockers: (1) condM-v2 13M training completing (confirms freebie bundle quality), (2) GPU availability on RunPod H100.

---

## Rust Verification

`verification/` contains a Rust crate verifying mathematical properties of architecture components before training runs.

```bash
cd verification
PATH="$HOME/.cargo/bin:$PATH" cargo test -- --nocapture
```

**65 tests total:**
- OPWF algebraic decomposition and math equivalence (error < 1e-13)
- Causality verification (pre-10: 0.000, post-10: >0)
- ELU normalizer variance, RG init scale separation
- DSQG coverage: all 44 unique dyadic offsets correctly addressed
- Rank bottleneck analysis (13M/85M/7B field capacity)
- **condM actual architecture** (5:1 hybrid, `condm_actual.rs` — 4 tests)
- Coverage density metrics and collapse attractor gradient dynamics
- Copy attractor feedback loop analysis

3 known failures in `cond_d_db4` (KdV nonlinear instability) — expected, condD was abandoned.

---

## Quick Start

### Local GPU

```bash
git clone https://github.com/Lanerra/DWARF.git
cd DWARF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train condM (best architecture, 13M, ~26 min/epoch on RTX 4090)
CUDA_VISIBLE_DEVICES=0 python -u benchmarks/train_2048_condM_layer_ablation.py \
  2>&1 | tee benchmarks/results/condM_run.log

# Train condM-v2 (RMSNorm + SwiGLU + RoPE bundle, 13M)
CUDA_VISIBLE_DEVICES=0 python -u benchmarks/train_2048_condM_v2.py \
  2>&1 | tee benchmarks/results/condM_v2_run.log

# Train condP (best pure DSQG, 13M, ~26 min/epoch)
CUDA_VISIBLE_DEVICES=0 python -u benchmarks/train_2048_condP.py \
  2>&1 | tee benchmarks/results/condP_run.log
```

### RunPod (H100, 27M scale)

```bash
git clone https://github.com/Lanerra/DWARF.git
cd DWARF
bash runpod_setup.sh

# condM 27M (~4.5h on H100, BATCH_SIZE=32)
python -u benchmarks/train_2048_27m_condM.py 2>&1 | tee benchmarks/results/condM_27m_run.log
```

### Evaluate an existing checkpoint

```bash
# Full eval suite (passkey + distance PPL + calibration + few-shot copy)
CUDA_VISIBLE_DEVICES=1 python benchmarks/eval_suite.py --model condm_layer5

# Per-position loss (2047 positions, support structure analysis)
CUDA_VISIBLE_DEVICES=1 python benchmarks/eval_position_loss.py --model condm_layer5

# All models
CUDA_VISIBLE_DEVICES=1 python benchmarks/eval_suite.py --all
CUDA_VISIBLE_DEVICES=1 python benchmarks/eval_position_loss.py --all
```

---

## Repository Structure

```
DWARF/
├── benchmarks/
│   ├── train_2048_condA.py          # FFT baseline
│   ├── train_2048_condB.py          # + learned dispersion
│   ├── train_2048_condC.py          # + Morlet wavelet
│   ├── train_2048_condG.py          # D4 DWT baseline
│   ├── train_2048_condJ.py          # + full OPWF Q·K gather (breakthrough)
│   ├── train_2048_condK.py          # + pos bias, ELU, RG init
│   ├── train_2048_condK_pooling.py  # condK + interference pooling
│   ├── train_2048_condL.py          # DSQG 24 dyadic (collapse; terminated)
│   ├── train_2048_condN.py          # ★ dense-32 + dyadic DSQG (70.8 PPL)
│   ├── train_2048_condP.py          # ★ dense-64 + dyadic DSQG (65.057 PPL)
│   ├── train_2048_condM_layer_ablation.py  # ★ 5:1 hybrid condM (54.53 PPL)
│   ├── train_2048_condM_v2.py       # condM + RMSNorm/SwiGLU/RoPE bundle
│   ├── train_2048_27m_condM.py      # condM at 27M (44.5 PPL)
│   ├── train_2048_27m_condP.py      # condP at 27M
│   ├── train_2048_27m_standard_baseline.py  # standard 27M reference
│   ├── eval_suite.py                # ★ unified eval (canonical benchmark)
│   ├── eval_position_loss.py        # per-position loss (2047 positions)
│   ├── eval_temperature_sweep.py    # temperature sweep
│   ├── train_2048_standard_baseline.py  # standard 13M reference
│   ├── 2048_condI_tokenizer.json    # shared BPE tokenizer (32k vocab)
│   ├── results/                     # JSON results for all completed runs
│   └── logs/                        # eval suite outputs
├── rag/
│   ├── ingest.py                    # build/update ChromaDB semantic index
│   ├── query.py                     # semantic search CLI
│   ├── chroma_db/                   # persistent vector store (~2,300 chunks)
│   └── README.md
├── verification/                    # Rust verification crate (65 tests)
│   └── src/
│       ├── condm_actual.rs          # condM 5:1 hybrid verification
│       ├── coverage_attractor.rs    # coverage density + collapse/copy attractor
│       ├── condl_ablation.rs        # ELU normalizer + RG init
│       ├── rank_bottleneck.rs       # field rank capacity analysis
│       ├── dsqg.rs                  # DSQG coverage verification
│       └── ...
├── runpod_setup.sh
├── requirements.txt
└── README.md
```

---

## Citation

No preprint yet. One is forthcoming. For now:

```bibtex
@misc{lewis2026dwarf,
  author       = {Lewis, Dennis},
  title        = {{DWARF}: Dyadic Windowed Attention with Resonant Fusion},
  year         = {2026},
  howpublished = {\url{https://github.com/Lanerra/DWARF}},
  note         = {Preprint forthcoming}
}
```

---

## License

Copyright 2026 Dennis Lewis. Licensed under the [Apache License 2.0](LICENSE).

This repository was forked from [badaramoni/wave-field-llm](https://github.com/badaramani/wave-field-llm), which was MIT-licensed at the time of fork. The original FFT-convolution wave field architecture (condA–condK) has been substantially redesigned; DWARF is an independent architecture developed from that starting point.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Lanerra/DWARF)
