# DWARF

**Dyadic Wave And Resonant Field Attention** — an O(N) linear attention architecture using D4 Daubechies wavelets and content-gated Q·K gather at dyadic offsets.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is DWARF?

DWARF is a linear attention mechanism built around two memory systems:

**Local K/V store (bounded, exact).** At each position, the model gathers key-value pairs from a fixed set of 44 dyadic offsets: positions at distances 2^0, 2^1, ..., 2^10, each with 4 D4 filter taps (τ=0..3). These are direct reads from the sequence — exact, content-addressable, and bounded. The maximum lookback is 3 × 2^10 = 3,072 tokens regardless of sequence length, which is the architecturally bounded KV cache at inference.

**Global wave field (compressed, propagated).** K⊗V outer products are propagated forward through the sequence via D4 Daubechies wavelet convolutions, forming a multi-scale "field" that carries distributional information at all distances. This field captures long-range correlations in a compressed form.

**The key insight (condJ/K).** Early DWARF variants without content-gated routing achieved ~99-100 PPL at 13M scale — no better than the first working version. The breakthrough came from replacing the field gather with a proper Q·K inner product at each tap: φ(Q_i · K_{i-offset} + bias_{j,τ}), where φ = ELU+1 (linear attention feature map). This made routing content-dependent rather than purely positional, and dropped PPL from ~99 to 85.1 (condJ) and 84.7 (condK) through the ablation series.

---

## Key Properties

- **O(N) complexity.** No quadratic attention matrix. Computation scales linearly with sequence length.
- **Bounded inference KV cache.** Architecturally: max 3,072-token lookback (3 × 2^10). *Caveat:* passkey retrieval at 13M shows ~10% accuracy (chance) at all distances. The wave field preserves distributional coherence at distance but not specific-content retrieval — this is a known limitation at current scale, not a bug.
- **D4 Daubechies wavelet basis.** 4-tap D4 filter taps provide multi-scale causal convolution with compact support. Better frequency localization than Morlet (condC), without the KdV instability issues (condD).
- **Multi-scale hierarchy.** 11 scales (j=0..10), 4 taps each = 44 dyadic offsets total. Scale gains are Q-conditioned (softmax over 11 scales per head), so each query selects which scales to attend to.
- **~20 PPL gap vs standard transformer.** DWARF condK achieves 84.7 PPL at 13M (OpenWebText) vs 64.5 for a matched standard transformer. This gap is the target of ongoing ablations (condL+RP, condN).

---

## Architecture

condK — current best configuration:

```
Input tokens
     │
     ▼
┌─────────────────────────────────┐
│  Embedding + Positional Embed   │
└─────────────────────────────────┘
     │
     ▼ (per layer × 6)
┌─────────────────────────────────────────────────────┐
│  DWARFAttention                                     │
│                                                     │
│  x ──▶ QKV projection ──▶ Q, K, V                  │
│              │                                      │
│              ▼                                      │
│   D4 DWT causal convolution                         │
│   (j=0..10, τ=0..3 → 44 taps)                      │
│   propagates K⊗V field forward                     │
│              │                                      │
│              ▼                                      │
│   Q·K gather at dyadic offsets                      │
│   φ(Q_i · K_{i-offset} + pos_bias_{j,τ,h})         │
│   φ(x) = elu(x) + 1  [linear attn feature map]     │
│              │                                      │
│   Q-conditioned scale gains                         │
│   (softmax over 11 scales, per-head)                │
│              │                                      │
│   ELU normalizer (prevents output explosion)        │
│              │                                      │
│   sigmoid output gate                               │
│              │                                      │
│   identity bypass (small residual path)             │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  FFN (SwiGLU)   │
└─────────────────┘
     │
     ▼
   Output logits
```

**condK+RP** adds interference pooling (cumulative mean every 3rd layer) on top of condK. Currently in training; ep7 PPL ~85.5, final estimate ~83.3.

**Parameter breakdown (13M model, condK):**
- Standard transformer components (embed, FFN, LM head): ~12.4M
- Per-tap position bias: 44 taps × 8 heads × 6 layers = 2,112 params
- Q-scale projection: negligible
- Scale gain: 11 × 8 = 88 params per layer
- Total: ~13M

---

## Ablation Results

All runs: OpenWebText, 2048 sequence length, 13M parameters, 10 epochs, 32k BPE vocabulary.

| Condition | PPL | Key change |
|---|---|---|
| Standard transformer 13M | **64.5** | Reference |
| condA (no dispersion) | 86.8 | Baseline DWARF — FFT convolution replaced by DWT taps |
| condB (learned-β dispersion) | 87.9 | +1.1 PPL — dispersion adds noise at 13M |
| condC (causal Morlet wavelet) | 87.2 | +0.4 PPL — Morlet slightly worse than baseline |
| condD (KdV nonlinear) | 99.6 | Abandoned — training instability |
| condG (D4 DWT, no Q-gate) | 99.4 | D4 introduced; without Q-gate it's terrible |
| condH (Q-gated gains post-gather) | 100.0 | Null result — wrong placement |
| condI (Q-scale pre-gather) | 93.3 | +Q·K pre-gather; partial improvement |
| condJ (OPWF Q·K gather) | 85.1 | Full Q·K formulation — key breakthrough |
| condK (+ pos bias + ELU + RG init) | **84.7** | Best 13M result to date |
| condK+RP (+ interference pooling) | ~83.3 | *In training* — ep7: 85.5, est. final: 83.3 |

The condG→condJ jump (99.4 → 85.1) is the main story: content-gated routing is what makes this architecture work.

condA–condC used the older FFT convolution approach; condG+ use D4 DWT. The condG baseline being ~99 PPL shows that D4 DWT alone doesn't help — it needed the Q·K gather (condJ) to become useful.

---

## 85M Scaling

Early scaling results (OpenWebText, 2048 seq len, condA and condB architectures):

| Condition | Scale | PPL |
|---|---|---|
| condA | 85M | 57.7 |
| condB (+ dispersion) | 85M | 58.1 |
| Standard transformer | 13M | 64.5 |

The dispersion penalty at 85M is +0.4 PPL (vs +1.1 at 13M) — the relative cost halves as scale increases, suggesting dispersion may break even or help at larger scale.

condC 85M was not completed (pod terminations; not comparable).

85M condK+RP will run after the 13M ablation series completes.

---

## Rust Verification

`verification/` contains a Rust crate with 58 tests verifying the mathematical properties of the architecture:

- **55 passing** — covering: OPWF algebraic decomposition, causality (output at position i cannot depend on position i+k), DSQG coverage (all unique dyadic offsets correctly addressed), condM/condO stability, field normalizer behavior.
- **3 known failures** — all in `cond_d_db4`: KdV nonlinear term instability. Expected and confirmed; condD was abandoned for this reason.

To run:
```bash
cd verification
PATH="$HOME/.cargo/bin:$PATH" cargo test
```

---

## Upcoming Experiments

**condL+RP: DSQG with D4 warm-start + interference pooling**
Direct Q·K Softmax Gather over 24 unique dyadic offsets. Uses condK checkpoint as warm-start, adds interference pooling. Tests whether a softmax formulation over the dyadic offset set (rather than ELU linear attention) improves content routing.

**condN: Dense local + dyadic long-range**
Dense-32 local attention (positions i-1..i-32) plus dyadic long-range DSQG at 44 offsets. Same compute budget as condK. Tests the hypothesis that the remaining PPL gap is partly from weak local context (the nearest 32 tokens).

**85M condK+RP**
After condN completes at 13M, the best-performing variant scales to 85M. Interested in whether the ~20 PPL gap vs standard transformer shrinks with scale (as condA→85M and condB→85M suggest it will).

---

## Quick Start

### RunPod (recommended for training)

```bash
# On a fresh RunPod instance with PyTorch image
git clone https://github.com/dlewis3/DWARF.git
cd DWARF
bash runpod_setup.sh

# Train condK (13M, ~8h on A100)
CUDA_VISIBLE_DEVICES=0 python -u benchmarks/train_2048_condK.py \
  2>&1 | tee benchmarks/logs/condK_run.log
```

### Local GPU

```bash
git clone https://github.com/dlewis3/DWARF.git
cd DWARF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train condK
CUDA_VISIBLE_DEVICES=0 python -u benchmarks/train_2048_condK.py \
  2>&1 | tee benchmarks/logs/condK_run.log
```

Results are written to `benchmarks/2048_condK_results.json`. The training script handles BPE tokenizer training, OpenWebText download, and checkpointing automatically.

### Run passkey retrieval test

```bash
# Requires a trained condK checkpoint in checkpoints/
python benchmarks/passkey_test.py
```

---

## Repository Structure

```
DWARF/
├── benchmarks/
│   ├── train_2048_condG.py          # D4 DWT baseline
│   ├── train_2048_condH.py          # + Q-gated gains
│   ├── train_2048_condI.py          # + Q-scale pre-gather
│   ├── train_2048_condJ.py          # + full Q·K OPWF gather
│   ├── train_2048_condK.py          # + pos bias, ELU, RG init (best)
│   ├── train_2048_condK_pooling.py  # condK with interference pooling
│   ├── train_2048_condL.py          # condL experiments
│   ├── train_2048_condLRP.py        # condL + interference pooling
│   ├── train_2048_condN.py          # dense local + dyadic long-range
│   ├── passkey_test.py              # passkey retrieval evaluation
│   ├── benchmark_wikitext2.py       # WikiText-2 evaluation
│   ├── 2048_condI_tokenizer.json    # shared BPE tokenizer (all cond* runs)
│   ├── 2048_condG_results.json      # per-epoch training results
│   ├── 2048_condH_results.json
│   ├── 2048_condI_results.json
│   ├── 2048_condJ_results.json
│   ├── 2048_condK_results.json
│   ├── 2048_passkey_results.json
│   ├── logs/                        # training logs
│   └── archive/                     # pre-condG scripts (historical record)
├── checkpoints/                     # model checkpoints
├── verification/                    # Rust verification crate (58 tests)
├── runpod_setup.sh                  # RunPod environment setup
├── requirements.txt
├── 100m_benchmark_results.json      # 100M standard transformer reference
├── 85m_ablation_results.json        # 85M condA/B/C results
├── 2048_dispersion_ablation_results.json
├── seqlen_sweep_results.json
└── README.md
```

---

## Citation

No preprint yet, but one is coming. For now:

```bibtex
@misc{lewis2026dwarf,
  author       = {Lewis, Dennis},
  title        = {{DWARF}: Dyadic Wave And Resonant Field Attention},
  year         = {2026},
  howpublished = {\url{https://github.com/dlewis3/DWARF}},
  note         = {Preprint forthcoming}
}
```

---

## License

Copyright 2026 Dennis Lewis. Licensed under the [Apache License 2.0](LICENSE).

This repository was forked from [badaramoni/wave-field-llm](https://github.com/badaramoni/wave-field-llm), which was MIT-licensed at the time of fork. The original FFT-convolution architecture and related code have been removed; DWARF is an independent architecture developed from that starting point.
