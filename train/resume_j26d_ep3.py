"""
Resume j26d from ep2 checkpoint → run ep3 only.

Patches the j26d training module at runtime:
  - SCREEN_EPOCHS = 1  (only ep3)
  - Loads ep2 weights before training starts

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/resume_j26d_ep3.py \
    > logs/run_j26d_ep3_resume.log 2>&1 &
"""
import os, sys, pathlib
import torch

# ── point to repo root ────────────────────────────────────────────────────
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(ROOT, 'train'))
sys.path.insert(0, os.path.join(ROOT, 'kernels'))
sys.path.insert(0, os.path.join(ROOT, 'autoresearch'))
sys.path.insert(0, os.path.join(ROOT, '/tmp/dwarf-j17d/kernels'))
sys.path.insert(0, '/tmp/dwarf-j17d/kernels')
sys.path.insert(0, '/tmp/dwarf-j17d')
os.chdir(ROOT)

CHECKPOINT = "autoresearch/checkpoints/f52350c_j26d_int2_physics_best.pt"

# ── Load the training module source and patch it ──────────────────────────
with open(os.path.join(ROOT, 'train', 'train_j26d_int2_physics_bf16.py')) as f:
    src = f.read()

# Patch: run only 1 epoch (will be ep3 in terms of data)
src = src.replace('SCREEN_EPOCHS = 3', 'SCREEN_EPOCHS = 1')

# Patch: inject checkpoint load right after model is built (before optimizer)
INJECT = f"""
    # ── Resume from ep2 checkpoint ────────────────────────────────────────
    _ckpt_path = "{CHECKPOINT}"
    print(f'  Loading ep2 checkpoint: {{_ckpt_path}}')
    _sd = torch.load(_ckpt_path, map_location=device, weights_only=False)
    # Strip _orig_mod prefix (from torch.compile in ep1/ep2)
    _sd_clean = {{k.replace("._orig_mod", ""): v for k, v in _sd.items()}}
    _missing, _unexpected = model.load_state_dict(_sd_clean, strict=False)
    print(f'  Resume: {{len(_missing)}} missing, {{len(_unexpected)}} unexpected keys')
    # ── End resume ────────────────────────────────────────────────────────
"""

# Insert after "model = AutoresearchTransformerPhysics(...).to(device)"
insert_marker = ').to(device)\n\n    try:'
src = src.replace(insert_marker, ').to(device)\n' + INJECT + '\n    try:', 1)

# Also fix the epoch label to show "ep3" in output
src = src.replace(
    "print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL",
    "print(f'Ep {epoch + 2}/{3} | Val PPL"
)

# Execute the patched source
exec(compile(src, 'train_j26d_int2_physics_bf16.py', 'exec'))
