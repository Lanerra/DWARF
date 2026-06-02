# DWARF

**Dyadic Wave And Resonant Field Attention** — a pure sparse attention architecture combining O(1)-KV-cache DSQG layers with HISA (Hierarchical Sparse Attention) layers, trained jointly from initialization. Dense (full) attention has been deprecated.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is DWARF?

DWARF replaces all transformer attention layers with sparse alternatives — no full causal attention layer. The two layer types are:

- **DSQG** (Dyadic Sparse Q-K Gather) — attends to a fixed set of 96 sparse offsets (triadic partitioning: three groups of 32). Because the offset set is fixed regardless of sequence length, each DSQG layer's KV cache at inference is a fixed-size circular buffer: **O(1) memory, not O(N)**.
- **HISA** (Hierarchical Sparse Attention) — two-stage chunked attention: Stage 1 selects top-k chunks per query chunk (self-chunk guaranteed), Stage 2 selects top-m key tokens within each selected chunk. Uses Triton kernels with compact token refinement, reducing both QK and V work.

The DSQG layers handle local and long-range relay patterns through their fixed offset structure. HISA provides the global context binding that fixed sparse offsets alone cannot supply, but does so sparsely rather than through full attention. Both layer types co-train from initialization: gradient signal from HISA teaches DSQG layers what to preprocess for it. This co-training is load-bearing — the preprocessing advantage is zero at epoch 1 and emerges entirely through joint training.

---

## Results

### Current May 2026 HISA-DSQG Result

The current best DWARF recipe is a pure sparse HISA-DSQG model:

| Model | Params | Training | Val PPL | Passkey |
|---|---:|---|---:|---:|
| D512/L10 Muon HISA-DSQG | 46.1M | 1 epoch at 2048 + 1 CPT epoch at 8192 | 8.74 | 96.8% at 8192 |

Architecture summary:

- D=512, L=10, H=8, head_dim=64, FFN=1536, tied LM head.
- DSQG V20-compatible sparse layers with R_PLANES=4 and staggered MOVT planes.
- Original J=96 triadic offsets split into A/B/C groups of 32.
- HISA at L3 with C=128, top_k=4, HISA_m=64 during 8192 CPT.
- No full attention layer.

The most important extrapolation result so far:

| Exact context length | Passkey accuracy | Effective passkey distance |
|---:|---:|---:|
| 8192 | 90.0% (9/10) | 8181-8182 |
| 16384 | 100.0% (10/10) | 16373-16374 |
| 32768 | 40.0% (4/10) | 32757-32758 |

This checkpoint was trained at 2048 and 8192 context only. The 16384 result used the same checkpoint and the same original 96 offsets, with no 16k training and no architecture change. This supports the current working interpretation that the offsets act as a reusable relay basis rather than a fixed table of maximum supported distances.

External benchmarks remain weak and essentially flat after the new CPT: HellaSwag 25.95%, PIQA 53.86%, ARC-Easy 28.25%, ARC-Challenge 19.40%, Winogrande 51.30%, LAMBADA 7.24%. The current claim is token-efficient language modeling plus long-context retrieval/transport, not broad benchmark competence.

Relevant files:

- Training: `train/train_d512_l10_muon_40_30_20_10.py`
- CPT: `train/train_d512_l10_muon_cpt8192_boundary.py`
- Long-context probe: `evals/eval_cpt_long_passkey.py`
- Result JSON: `results/cpt_long_passkey_16k_32k_perword.json`

---

## License

Copyright 2026 Dennis Lewis. Licensed under the [Apache License 2.0](LICENSE).

The research process from condA through the current condU architecture, including all negative results, intermediate architectures, and ablation findings, is documented in full and will be released under the same license. The path matters as much as the destination.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Lanerra/DWARF)
