# BRIEF: Build train_d512_l13_moonshot.py

## Task
Create `/home/dlewis3/Desktop/AI/DWARF/train/train_d512_l13_moonshot.py` — a complete training script for the DWARF Moonshot D512/L13 configuration.

## Reference
Use `/home/dlewis3/Desktop/AI/DWARF/train/train_d512_l13_triadic_aabbc_4090_bf16.py` as the base. This script achieved PPL=33.64, passkey=98.3%. Copy it and modify.

## Required Changes from Reference

### 1. Layer Layout — AABBC pattern with IF at multiple depths
The reference uses AABBC layout with IF only at L2 (preIF). The moonshot adds IF at L5 and L11:

```python
LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L0
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L1
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, True),    # L2 + preIF (before FA)
    ('FA', None, 0, 0, False),                       # L3: FullAttention
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L4
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L5 + IF ← NEW
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L6
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L7
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L8
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L9
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L10
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L11 + IF ← NEW
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L12
]
```

### 2. Batch size: BS=16, GA=8 (eff_batch=128)
```python
BATCH_SIZE     = int(os.environ.get('DWARF_BS', '16'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '8'))
```

### 3. SE_LR_MULT = 15.0 (formula value for D=512)
```python
SCALE_EMBED_LR_MULT  = 15.0
```

### 4. Checkpoint name
```python
CKPT_BASE_NAME    = 'd512_l13_moonshot'
```

### 5. Print header — update to say "Moonshot"
```python
print('  🌲 DWARF D512-L13 Moonshot — all proven mechanisms, AABBC + multi-IF')
```

### 6. KEEP EVERYTHING ELSE from the reference script:
- DSQGAttentionGrouped class (with phase_gain, NOT phase_gain_vec or phase_gate)
- DSQGBlockTriadic with IF/interference mechanism
- FullCausalAttention + FullAttentionBlock with gate_proj
- FFN with GELU
- Chunked CE loss (CE_CHUNK=512)
- EMA physics (EMA_INIT=0.020833, EMA_FLOOR=0.00001)
- NPCI (npci_theta_k, npci_theta_v)
- QK-OVT (query_probes, key_probes)
- MOVT (phase_base, phase_gain)
- Scale embeddings (init=0.15, separate param group)
- Gate projection on both DSQG and FA blocks
- AdamW8bit optimizer
- Cosine LR schedule with 100-step warmup
- Passkey eval (12 distances × 50 trials)
- BPETokenizerWrapper
- Full-attn checkpoint saving
- Resume support

## DO NOT ADD
- Learnable offsets (Phase B — separate script later)
- DSR/HSA/TopK
- phase_gain_vec or phase_gate
- Huygens K/V
- RoPE/SwiGLU/RMSNorm
- torch.compile
- Liger fused CE (keep disabled by default)

## Verification
After creating the file, verify:
1. It imports successfully: `cd /home/dlewis3/Desktop/AI/DWARF && .venv/bin/python3 -c "import ast; ast.parse(open('train/train_d512_l13_moonshot.py').read()); print('parse OK')"`
2. The LAYER_LAYOUT has 13 entries with FA at index 3
3. IF blocks are at L2, L5, L11 (indices 2, 5, 11 have has_if=True)
4. BATCH_SIZE=16, GRAD_ACCUM=8, SE_LR_MULT=15.0
5. CE_CHUNK=512 is present
6. gate_proj is in both DSQGAttentionGrouped and FullCausalAttention

## RAG for additional context
```bash
.venv/bin/python3 /home/dlewis3/.openclaw/rag/query.py "DWARF d512 triadic aabbc training configuration IF interference" --n 5
```
