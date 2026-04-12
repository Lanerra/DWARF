# Tooling Update Pass — Shared Model Registry

## Goal
Eliminate the repeated 4-dict pattern across eval/viz tools by creating a single shared model registry that all tools import. Every new model registration should require editing exactly ONE file.

## Problem
Currently 4 tools each maintain their own copies of:
- `_TRAIN_SCRIPTS` dict (arch → train script path)
- `_MODEL_CLASSES` dict (arch → class name string)
- `_OFFSET_SETS` dict (arch → offset list)
- `_ARCH_CONFIGS` dict (arch → D/L/H/FFN/full_attn_layer)
- `_PREIF_ARCHS`, `_FINEWEB_TOK_ARCHS`, `_IS_STANDARD` sets

Adding a new arch (e.g. triadic_l13) required 16+ manual edits across:
- `evals/eval_suite.py` (MODEL_REGISTRY + build_model handler)
- `evals/eval_external.py` (pulls from eval_suite, but has its own model loading)
- `tools/visualize_attention.py`
- `tools/relay_path_tracer.py`
- `tools/probe_residual_stream.py`

An additional complication was discovered: the old tools assume a single flat offset set per arch, but triadic models have per-layer offset sets (each DSQGBlockTriadic.attn.offsets_dev is different). The registry needs to accommodate this.

## Solution: `tools/model_registry.py`

Create `/home/dlewis3/Desktop/AI/DWARF/tools/model_registry.py` — a single source of truth.

### Schema

Each entry in `MODEL_REGISTRY` is a dict with:
```python
{
    # Required
    'train_script': 'train/train_moonshot_58m_4090_bf16.py',  # relative to REPO root
    'model_class':  'AutoresearchTransformerPhysics',
    'D': 512, 'H': 8, 'FFN': 2048, 'L': 8,
    'full_attn_layer': 2,
    
    # Optional — defaults shown
    'offsets': SE015_OFFSETS,           # default se015; can be list or 'per_layer'
    'vocab_size': None,                 # None = read from train script's VOCAB_SIZE global
    'seq_len': 2048,
    'tokenizer': 'fineweb_32k',         # 'fineweb_32k' | 'cond_i' | explicit path
    'scale_embed_init': 0.15,
    'init_kwargs': {},                  # extra kwargs for model __init__ beyond standard
    'no_num_layers': False,             # True for TriadicJ96 (layer count baked into script)
    'is_standard': False,               # standard transformer (no DSQG)
    'label': 'Human-readable label',
    'checkpoint': 'autoresearch/checkpoints/foo.pt',  # relative to REPO root
    'params_ref': None,
}
```

For `offsets='per_layer'`: tools should call `block.attn.offsets_dev.tolist()` per layer instead of using a global offset list.

### Constants

```python
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SE015_OFFSETS = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]
SE096_OFFSETS = [...]  # the J=96 set from train scripts

TOKENIZER_PATHS = {
    'fineweb_32k': os.path.join(REPO, 'results', 'fineweb_tokenizer_32k.json'),
    'cond_i':      os.path.join(REPO, 'results', '2048_condI_tokenizer.json'),
}
```

### `build_model(arch_name, device='cpu')` function

Centralised model construction:
```python
def build_model(arch_name, device='cpu'):
    """Load and instantiate model for arch_name. Returns (model, cfg)."""
    cfg = MODEL_REGISTRY[arch_name]
    # 1. Import train script
    # 2. Get vocab_size (from cfg or train script global)
    # 3. Call cls(**kwargs) — use no_num_layers flag for TriadicJ96
    # 4. Load checkpoint if cfg['checkpoint'] is set
    # 5. Return model, cfg
```

### `load_checkpoint(model, checkpoint_path)` function

Shared checkpoint loading with standard key stripping:
```python
def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = state.get('model_state_dict', state.get('model', state))
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v for k,v in state.items()}
    state = {k:v for k,v in state.items() if 'causal_mask' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
```

### `get_tokenizer(arch_name)` function

```python
def get_tokenizer(arch_name):
    cfg = MODEL_REGISTRY[arch_name]
    tok_key = cfg.get('tokenizer', 'fineweb_32k')
    path = TOKENIZER_PATHS.get(tok_key, tok_key)  # fall back to treating as path
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)
```

## MODEL_REGISTRY contents

Populate with all archs currently in eval_suite.py MODEL_REGISTRY and relay_path_tracer.py/_TRAIN_SCRIPTS. Key entries to include (at minimum):

```python
MODEL_REGISTRY = {
    # Moonshot lineage
    'moonshot_58m': {
        'train_script': 'train/train_moonshot_58m_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 512, 'H': 8, 'FFN': 2048, 'L': 8, 'full_attn_layer': 2,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'label': 'Moonshot-58M ep2 (J24 se015, D=512, FA@L2)',
        'checkpoint': 'autoresearch/checkpoints/moonshot_58m_best.pt',
    },
    # d768_l32 family
    'd768_l32': { ... },
    'd768_l32_cont': { ... },
    'd768_l32_mixed_frozen': { ... },
    'd768_l32_mixed_scratch': { ... },
    # d768_l24
    'd768_l24': { ... },
    # d1024_267m  
    'd1024_267m': { ... },
    # Triadic family
    'triadic_l13': {
        'train_script': 'train/train_d512_l13_triadic_j96_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',   # <-- special: use block.attn.offsets_dev per layer
        'no_num_layers': True,    # TriadicJ96.__init__ doesn't take num_layers
        'tokenizer': 'fineweb_32k',
        'label': 'Triadic-J96 L13 D512 ep3 (J=32/layer, 4 triads, FA@L3)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_triadic_j96_best.pt',
        'params_ref': 48_359_073,
    },
    'triadic_l9': { ... },
    # ... etc for all models currently in eval_suite MODEL_REGISTRY
}
```

## Updating the tools to use the registry

After creating `model_registry.py`, update each tool to import from it:

### `tools/relay_path_tracer.py`
Replace the 4 dicts + `load_model` function with:
```python
from model_registry import MODEL_REGISTRY, build_model, get_tokenizer, SE015_OFFSETS

def load_model(arch, checkpoint_path, root, device):
    model, cfg = build_model(arch, device=device)
    full_attn_layer = cfg['full_attn_layer']
    offsets = cfg['offsets'] if cfg['offsets'] != 'per_layer' else SE015_OFFSETS
    return model, full_attn_layer, cfg
```

For the offset ablation loop, check `cfg['offsets'] == 'per_layer'` and use `block.attn.offsets_dev.tolist()` per block.

### `tools/probe_residual_stream.py`
Same pattern — replace `_TRAIN_SCRIPTS`, `_MODEL_CLASSES`, `_PREIF_ARCHS`, `load_model` with imports from registry.

### `tools/visualize_attention.py`
Replace `_TRAIN_SCRIPTS`, `_MODEL_CLASSES`, `_OFFSET_SETS`, `_PREIF_ARCHS`, `_FINEWEB_TOK_ARCHS`, `_instantiate_model` with registry imports. The `ALL_OFFSETS` global override hack for triadic (added in tonight's patches) gets replaced with proper per-layer handling via `cfg['offsets'] == 'per_layer'`.

### `evals/eval_suite.py`
The `MODEL_REGISTRY` in eval_suite.py is more detailed (includes calibration params, tokenizer paths, etc). Keep the eval_suite registry as-is for now — it's already working. The shared `model_registry.py` primarily benefits the 3 tools.

## File to create

`/home/dlewis3/Desktop/AI/DWARF/tools/model_registry.py`

## Files to update

1. `tools/relay_path_tracer.py` — replace 4 dicts with registry imports
2. `tools/probe_residual_stream.py` — same
3. `tools/visualize_attention.py` — same, plus fix per-layer offset handling

## Test

After changes, verify these commands still work:

```bash
cd /home/dlewis3/Desktop/AI/DWARF

# relay_path_tracer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 tools/relay_path_tracer.py \
  --arch triadic_l13 \
  --checkpoint autoresearch/checkpoints/d512_l13_triadic_j96_best.pt \
  --distance 512 --n_trials 5

# probe_residual_stream  
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 tools/probe_residual_stream.py \
  --arch triadic_l13 \
  --checkpoint autoresearch/checkpoints/d512_l13_triadic_j96_best.pt \
  --distance 512

# visualize_attention
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 tools/visualize_attention.py \
  --arch triadic_l13 \
  --checkpoint autoresearch/checkpoints/d512_l13_triadic_j96_best.pt \
  --out_dir /tmp/viz_test/
```

All three should complete without errors.

## Key constraint

Do not break any existing arch. All archs that currently work in relay_path_tracer/probe_residual/visualize_attention must still work after the refactor. Run a quick smoke test on `moonshot_58m` and `d768_l32` in addition to `triadic_l13`.

RAG:
  /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python3 /home/dlewis3/.openclaw/rag/query.py "model registry eval tools dispatch table" --n 5
