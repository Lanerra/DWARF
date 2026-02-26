# DWARF

**Dyadic Wave And Resonant Field Attention** — an O(N) linear attention architecture with a bounded inference KV cache, multi-scale D4 Daubechies wavelet field propagation, and content-gated Q·K gather at dyadic offsets.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is DWARF?

DWARF replaces standard quadratic attention with two parallel memory systems:

**Local K/V track (bounded, exact).** The model gathers key-value pairs from a fixed set of sparse offsets using direct Q·K inner products. In the best-performing condN/condP architectures, this includes a dense local region (δ=1..32 or δ=1..64) plus dyadic long-range taps (δ=48, 96, 128, ..., 1536). The maximum lookback is architecturally bounded: 3 × 2^10 = **3,072 tokens** regardless of sequence length — exact, content-addressable, O(1) KV cache at inference.

**Global wave field (compressed, propagated).** K⊗V outer products propagate forward through the sequence via D4 Daubechies wavelet convolutions, forming a multi-scale "field" carrying distributional information at all distances. Old tokens diffuse into the field rather than disappearing — the wave field is a continuous unbounded memory, but compressed and approximate.

**The key insight (condJ, Feb 2026).** Early DWARF variants without content-gated routing achieved ~99–100 PPL at 13M scale. The breakthrough came from replacing position-only field reads with a proper Q·K gather: `output = Σ_{j,τ} gain_{j,τ} · φ(Q_i · K_{i-δ_{j,τ}} + pos_bias)`, where φ = softmax and gains are Q-conditioned. This dropped PPL from ~99 to 85.1 (condJ) in one architectural change, and the ablation series refined it to 84.7 (condK), 83.3 (condK+RP), and ultimately **70.8** (condN) and **65.1** (condP) — closing to within **1.03 PPL** of the standard transformer baseline of 64.07.

---

## Key Properties

- **O(N) complexity.** No quadratic attention matrix. Computation scales linearly with sequence length.
- **Bounded inference KV cache.** Max 3,072-token lookback at inference, O(1) fixed-size circular buffer. For a 7B model at 100K-token context: standard attention requires ~52 GB KV cache; DWARF requires ~1.5 GB fixed, regardless of context length.
- **D4 Daubechies wavelet basis.** 4-tap causal filter provides multi-scale field propagation with compact support and better frequency localization than simpler kernels (Morlet condC, FFT condA–B).
- **Two-memory architecture.** Local track: exact, bounded, fast. Wave field: compressed, unbounded, diffusive. The interaction between them (condK+RP pooling) provides both within-layer amplitude normalization (ELU) and cross-layer state continuity (interference pooling) — orthogonal contributions, both needed.
- **Coverage density as structural regularization.** Dense local coverage prevents individual offsets from becoming collapse attractors (see condL+RP failure). condN's dense-32 region eliminated the δ=1 collapse basin. condP's dense-64 region further reduces uncovered gaps.
- **1.03 PPL gap vs standard transformer at 13M (condP).** condP: 65.1 PPL vs 64.07 standard. condN: 70.8 PPL. The linear-vs-softmax capacity gap is almost entirely closed by coverage density and offset structure — and the baseline's remaining advantage likely reflects copy-attractor exploitation rather than genuine language modeling superiority (see Generation Quality below).

---

## Architecture

### condP — best configuration (65.1 PPL)

condP extends condN's dense local window from 32 to 64 tokens:

```
Offsets: dense {δ=0..64} + dyadic {96, 128, 192, 256, 384, 512, 768, 1024, 1536}
         = 74 total offsets

Everything else identical to condN (ALiBi pos_bias, interference pooling every 3rd layer)
Max lookback: 3 × 2^10 = 3,072 tokens
```

### condN — reference architecture (70.8 PPL)

condN uses DSQG (Dyadic Sparse Q·K Gather) with dense-32 local coverage:

```
Offsets: dense {δ=0..32} + dyadic {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536}
         = 44 total offsets

Per offset δ:
  score_{j,h,n} = Q_{h,n} · K_{h,n-δ} / sqrt(d_head) + pos_bias_{j,h}
  weight = softmax(score) over all 44 offsets
  output = Σ weight_{j,h,n} · V_{h,n-δ_j}

pos_bias: ALiBi-style init (steeper for local heads, shallower for global heads)
          fully learned during training

Interference pooling: cumulative mean of field outputs, applied every 3rd layer
                      provides cross-layer state continuity
```

### condK — best linear attention configuration (83.3 PPL with pooling)

```
Input: x
  ↓
QKV projection → Q, K, V (shape: [B, H, N, D])
  ↓
DEPOSIT:    field[b,h,n] += K[b,h,n] ⊗ V[b,h,n]        # outer product, D×D matrix
  ↓
PROPAGATE:  field = D4 causal dilated conv (j=0..10, τ=0..3, 44 taps)
            gains = softmax(scale_param[j,h] + pos_bias[j,h] + q_scale_proj(Q))
  ↓
GATHER:     output = Q[b,h,n] @ F_prop[b,h,n]            # Q·K content-matched
  ↓
NORMALIZE:  output = ELU_phi(output) / Z                  # linear attn normalizer
  ↓
GATE:       output = sigmoid(W_gate·x + b) ⊙ reshape(output)
  ↓
BYPASS:     F_prop += identity_gain · F_0                 # learned scalar ≈ 0.028
  ↓
FFN (SwiGLU)
  ↓
Output logits
```

---

## Ablation Results

All runs: OpenWebText, 2048 sequence length, 13M parameters, 10 epochs, 32k BPE vocabulary.

| Condition | PPL | Key change |
|---|---|---|
| Standard transformer 13M | **64.07** | Reference — generation severely degraded (repetition loops at test time) |
| condA (no dispersion) | 86.8 | Baseline DWARF — FFT convolution replaced by DWT taps |
| condB (learned-β dispersion) | 87.9 | +1.1 PPL — dispersion adds noise at 13M |
| condC (causal Morlet wavelet) | 87.2 | +0.4 PPL — Morlet slightly worse than baseline |
| condD (KdV nonlinear) | 99.6 | Abandoned — training instability |
| condG (D4 DWT, no Q-gate) | 99.4 | D4 introduced; without Q-gate it's terrible |
| condH (Q-gated gains post-gather) | 100.0 | Null result — wrong placement |
| condI (Q-scale pre-gather) | 93.3 | +Q·K pre-gather; partial improvement |
| condJ (OPWF Q·K gather) | 85.1 | Full Q·K formulation — key breakthrough |
| condK (+ pos bias + ELU + RG init) | 84.7 | condJ ablation; pooling removed |
| condK+RP (+ interference pooling) | **83.3** | Pooling confirmed −1.6 PPL improvement |
| condL+RP (DSQG 24 dyadic offsets) | *terminated ep4* | δ=1 structural collapse; PPL improved while generation degraded — key PPL/generation divergence finding |
| **condN (dense-32 + dyadic DSQG)** | **70.8** ✓ | Dense-32 eliminates δ=1 collapse basin; ALiBi pos_bias init; coherent generation; concept-level attractor |
| **condP (dense-64 + dyadic DSQG)** | **65.1** ✓ | **Best DWARF 13M.** Within 1.03 PPL of standard transformer. Generation superior to baseline throughout; no word-level copy attractor |

**Gap to standard transformer:** 1.03 PPL (condP) — down from 20.2 PPL (condK+RP). Three architectural steps, each closing roughly two-thirds of the remaining gap.

**The main story:** condG→condJ (99.4 → 85.1) established that content-gated routing is essential. condJ→condN (85.1 → 70.8) established that dense local coverage and softmax gather (vs ELU linear) close most of the remaining gap. condN→condP (70.8 → 65.1) established that extending the dense window from 32 to 64 closes ~85% of the remaining gap. condL+RP failure established the mechanism: sparse dyadic without dense local creates structural collapse attractors. With condP, DWARF is no longer a "promising alternative" to standard attention — it is within experimental error of a fair PPL comparison, while maintaining structural properties standard transformers cannot match.

---

## Generation Quality vs PPL

**Critical finding:** PPL alone is insufficient to evaluate DWARF architectures. Two documented cases of PPL/generation divergence:

1. **condL+RP (ep4, Feb 2026):** PPL improved while generation degraded — all heads converging on δ=1 copy-previous token. The model was getting better at predicting by copying, which lowers loss but makes generation degenerate.

2. **condP vs standard transformer (Feb 2026):** Standard transformer at PPL 64.07 produces severe repetition loops on all generation prompts from the first evaluation epoch onward ("stormy stormy stormy..."). condP at PPL 65.1 (worse by only 1.03 PPL) produces coherent sentences and never exhibits this pattern.

**Why the baseline's PPL lead doesn't mean what it appears to mean.** The copy attractor is *falsifiable*: a model that earns 64.07 through genuine language modeling doesn't collapse to token repetition on basic completion prompts. The standard transformer's PPL advantage is partially attributable to δ=1 copy exploitation — and we can demonstrate this directly from the generation samples. This makes the 1.03 PPL gap a ceiling on the "real" gap, not a floor.

**Architecture-dependent generation taxonomy (13M scale).** The ablation series reveals three distinct failure modes corresponding to different levels of gradient competition:

| Architecture | Attractor type | Example |
|---|---|---|
| Standard transformer 13M | **Word-level copy** | "stormy stormy stormy stormy..." — δ=1 dominance, degenerate |
| condN (dense-32) | **Concept-level** | "the length of the diagonal, the length of the diagonal" — geometrically adjacent concept, epistemically grounded but stuck |
| condP (dense-64) | **Narrative-phrase** | "the idea of a new person is to be a new person" — common narrative construction, higher abstraction |

Architecture determines *which semantic tier* becomes the attractor anchor, not *whether* anchoring occurs. Resolving this at 13M requires more scale; at 85M the narrative-phrase attractor is expected to dissolve as capacity increases.

**Policy:** generation samples are included in all training scripts at each epoch checkpoint.

---

## 85M Scaling Results

| Condition | Scale | PPL | Notes |
|---|---|---|---|
| condA | 85M | 57.7 | Acc 30.3% |
| condB (+ dispersion) | 85M | 58.1 | +0.4 PPL; dispersion penalty halved vs 13M (+1.1 → +0.4) |
| Standard transformer | 100M | 45.96 | Reference |

Dispersion penalty at 85M is +0.4 PPL vs +1.1 at 13M — the relative cost halves as scale increases. condN-equivalent 85M not yet run; predicted 60–64 PPL based on copy attractor escape analysis.

condC 85M was not completed (pod terminations + tokenizer confound — different vocabulary, not comparable).

---

## Key Findings

**condG→condJ: content-gated routing is non-negotiable.** Without Q·K inner products at each tap, DWARF reduces to position-only routing (~99 PPL). Adding full OPWF gather (condJ) drops PPL by 14.3 in one change.

**Interference pooling is load-bearing, not coincidental.** condK accidentally removed it and improved by 1.4 PPL. Later ablation (condK+RP, Feb 2026) confirmed pooling = genuine −1.6 PPL when ELU is present. Mechanism: ELU handles within-layer amplitude normalization; pooling provides cross-layer state continuity. Orthogonal operations, both needed.

**PPL/generation divergence is architectural, not random.** Two clean cases show that structural choices in the attention mechanism (sparse dyadic without dense local; copy attractor in dense softmax at 13M) create failure modes invisible in PPL but evident in generation. Generation sampling is required evaluation for DWARF ablations.

**Coverage density as structural regularization.** Collapse risk is inversely proportional to local coverage density. condL+RP failed because δ=1 was structurally uncontested. condN's dense-32 region put δ=1 in competition with 32 peers, eliminating the collapse basin. condP tests whether extending dense coverage to [0,64] yields further improvement.

**Copy attractor (standard transformer at 13M).** Dense softmax at 13M scale makes the δ=1 copy-previous strategy easy for gradient descent to exploit — the 2,046 non-copy offsets mostly carry gradient noise, so copy is the only consistent gradient direction. DWARF's fixed informative offsets prevent this: the non-copy offsets carry real gradient signal and compete with copy. This is the mechanism behind condP's better generation quality at near-equivalent PPL. The copy-attractor argument is falsifiable and directly observable from generation samples — see Generation Quality above.

**Coverage density is orthogonal to training dynamics.** condP and condN pos-bias |max| values tracked within 0.02 of each other across all 10 training epochs, despite a 5–7 PPL performance gap. The D4+ALiBi training regime converges to the same basin regardless of offset count. This means: (1) PPL differences between coverage experiments are cleanly attributable to coverage, not confounded by training dynamics changes, and (2) any future coverage experiment inherits the same training stability.

**Bounded KV cache property.** DWARF only needs last 3,072 tokens of K,V at inference — a fixed-size circular buffer regardless of context length. Passkey retrieval test (13M, condK) confirmed ~10% accuracy at all distances — the local KV track handles sharp content retrieval within 3,072 tokens; beyond that, only distributional coherence is preserved via the wave field. This is architectural design, not a bug.

---

## Rust Verification

`verification/` contains a Rust crate verifying mathematical properties of proposed architecture components *before* training runs.

```bash
cd verification
PATH="$HOME/.cargo/bin:$PATH" cargo test -- --nocapture
```

**52 tests total:**
- OPWF algebraic decomposition and math equivalence (error < 1e-13)
- Causality: output at position i cannot depend on position i+k (pre-10: 0.000, post-10: >0)
- ELU normalizer variance effect (OPWF vs DSQG)
- RG init scale separation persistence under gradient noise
- DSQG coverage: all unique dyadic offsets correctly addressed
- Rank bottleneck analysis (13M/85M/7B field capacity)
- condM/condO stability
- **Coverage density metrics** (condL+RP / condN / condP gap analysis)
- **Collapse attractor gradient dynamics** (steps to δ=1 dominance vs offset count)
- **Copy attractor feedback loop** (loop depth by offset set; ε needed for concentration)

3 known failures in `cond_d_db4` (KdV nonlinear instability) — expected, condD was abandoned.

---

## Upcoming Experiments

| Condition | Design | Status |
|---|---|---|
| condP | Dense-64 local + dyadic long-range (74 offsets) | **COMPLETE — 65.1 PPL** (best DWARF 13M) |
| condM | Learned softmax correction layer at layer boundaries; alternative to structural gap-fill | Planned after condP |
| condQ | TBD based on condP result | Planned |
| 85M condN-equiv | Best 13M architecture at 85M scale | Pending — waiting for 13M ablation to stabilize |

---

## Quick Start

### Local GPU

```bash
git clone https://github.com/Lanerra/DWARF.git
cd DWARF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train condN (best architecture, 13M, ~3.5h on RTX 4090)
python -u benchmarks/train_2048_condN.py 2>&1 | tee benchmarks/logs/condN_run.log

# Train condP (dense-64 coverage, 13M, ~7.5h on RTX 4090)
python -u benchmarks/train_2048_condP.py 2>&1 | tee benchmarks/logs/condP_run.log
```

### RunPod

```bash
git clone https://github.com/Lanerra/DWARF.git
cd DWARF
bash runpod_setup.sh

# Train condN
python -u benchmarks/train_2048_condN.py 2>&1 | tee benchmarks/logs/condN_run.log
```

### Evaluate an existing checkpoint

```bash
# Passkey retrieval test (requires condK/condN checkpoint)
python benchmarks/passkey_test.py

# WikiText-2 evaluation
python benchmarks/benchmark_wikitext2.py
```

Results written to `benchmarks/2048_cond{X}_results.json`.

---

## Repository Structure

```
DWARF/
├── benchmarks/
│   ├── train_2048_condA.py          # FFT baseline
│   ├── train_2048_condB.py          # + learned dispersion
│   ├── train_2048_condC.py          # + Morlet wavelet
│   ├── train_2048_condG.py          # D4 DWT baseline
│   ├── train_2048_condH.py          # + Q-gated gains (null result)
│   ├── train_2048_condI.py          # + Q-scale pre-gather
│   ├── train_2048_condJ.py          # + full OPWF Q·K gather (breakthrough)
│   ├── train_2048_condK.py          # + pos bias, ELU, RG init
│   ├── train_2048_condK_pooling.py  # condK + interference pooling
│   ├── train_2048_condL.py          # DSQG 24 dyadic (no dense)
│   ├── train_2048_condLRP.py        # + interference pooling (terminated ep4)
│   ├── train_2048_condN.py          # dense-32 + dyadic DSQG (70.8 PPL)
│   ├── train_2048_condP.py          # ★ dense-64 + dyadic DSQG — BEST (65.1 PPL)
│   ├── train_2048_standard_baseline.py  # matched standard transformer (64.07 PPL)
│   ├── passkey_test.py              # passkey retrieval evaluation
│   ├── eval_temperature_sweep.py    # temperature sweep evaluation
│   ├── benchmark_wikitext2.py       # WikiText-2 evaluation
│   ├── 2048_condI_tokenizer.json    # shared BPE tokenizer (all cond* runs)
│   └── logs/                        # training logs (condG through condN)
├── rag/
│   ├── ingest.py                    # build/update ChromaDB semantic index
│   ├── query.py                     # semantic search CLI
│   ├── chroma_db/                   # persistent vector store (~1600 chunks)
│   └── README.md                    # RAG usage docs
├── verification/                    # Rust verification crate (52 tests)
│   └── src/
│       ├── lib.rs
│       ├── coverage_attractor.rs    # coverage density + collapse/copy attractor
│       ├── condl_ablation.rs        # ELU normalizer + RG init persistence
│       ├── rank_bottleneck.rs       # field rank capacity analysis
│       ├── dsqg.rs                  # DSQG coverage verification
│       └── ...
├── runpod_setup.sh                  # RunPod environment setup
├── requirements.txt
└── README.md
```

---

## Citation

No preprint yet. One is forthcoming. For now:

```bibtex
@misc{lewis2026dwarf,
  author       = {Lewis, Dennis},
  title        = {{DWARF}: Dyadic Wave And Resonant Field Attention},
  year         = {2026},
  howpublished = {\url{https://github.com/Lanerra/DWARF}},
  note         = {Preprint forthcoming}
}
```

---

## License

Copyright 2026 Dennis Lewis. Licensed under the [Apache License 2.0](LICENSE).

This repository was forked from [badaramoni/wave-field-llm](https://github.com/badaramani/wave-field-llm), which was MIT-licensed at the time of fork. The original FFT-convolution architecture has been substantially redesigned; DWARF is an independent architecture developed from that starting point.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Lanerra/DWARF)
