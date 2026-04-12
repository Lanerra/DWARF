"""
🗄️ Cache FA layer activations from the teacher checkpoint.

Captures the FA block's additive contribution (before residual add)
for use in TopK sparse attention distillation training.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u scripts/cache_fa_activations.py
"""

import os, sys, time
import torch

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
for _d in [os.path.join(_project_root, 'kernels'), _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from train.train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16 import (
    TriadicJ96, EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS,
    FULL_ATTN_LAYER, VOCAB_SIZE, MAX_SEQ_LEN, SCALE_EMBED_INIT_VAL, DROPOUT,
)

TEACHER_CKPT = 'autoresearch/checkpoints/d512_l13_triadic_aabbc_mixed_scratch_best.pt'
DATASET_PATH = 'logs/mixed_encoded_2048_fineweb_tok.pt'
TRAIN_SEQS = 20_000
VAL_SEQS = 1_000
BATCH_SIZE = 32

TRAIN_OUT = 'logs/fa_cache_teacher_train_20k.pt'
VAL_OUT = 'logs/fa_cache_teacher_val_1k.pt'


def build_model(device):
    model = TriadicJ96(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
    ).to(device)

    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    fa_layer = model.full_attn_layer
    print(f'  FA layer index: {fa_layer} (from model.full_attn_layer)')
    print(f'  FA block type: {type(model.blocks[fa_layer]).__name__}')
    return model, fa_layer


@torch.inference_mode()
def cache_activations(model, fa_layer, data, device):
    captured = []

    def _hook(module, inp, out):
        captured.append(out.detach())

    handle = model.blocks[fa_layer].attn.register_forward_hook(_hook)

    num_seqs = len(data)
    all_outputs = []

    for start in range(0, num_seqs, BATCH_SIZE):
        batch = data[start:start + BATCH_SIZE]
        x = batch[:, :-1].to(device, non_blocking=True)

        captured.clear()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(x)

        assert len(captured) == 1, f"Expected 1 hook capture, got {len(captured)}"
        all_outputs.append(captured[0].cpu().to(torch.bfloat16))

        if (start // BATCH_SIZE) % 50 == 0:
            print(f'    {start}/{num_seqs} sequences processed', flush=True)

    handle.remove()
    return torch.cat(all_outputs, dim=0)


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('medium')
    t0 = time.time()

    print('=' * 70)
    print('  🗄️ Cache FA teacher activations for TopK distillation')
    print('=' * 70)

    assert os.path.isfile(TEACHER_CKPT), f'Teacher checkpoint not found: {TEACHER_CKPT}'
    assert os.path.isfile(DATASET_PATH), f'Dataset not found: {DATASET_PATH}'

    print(f'  Teacher: {TEACHER_CKPT}')
    print(f'  Dataset: {DATASET_PATH}')

    print('\n  Loading model...')
    model, fa_layer = build_model(device)

    print('\n  Loading dataset...')
    dataset = torch.load(DATASET_PATH, weights_only=True)
    if isinstance(dataset, dict):
        all_data = dataset.get('train', dataset.get('data'))
    else:
        all_data = dataset
    all_data = all_data.long()

    assert len(all_data) >= TRAIN_SEQS + VAL_SEQS, (
        f'Dataset has {len(all_data)} sequences, need {TRAIN_SEQS + VAL_SEQS}')

    train_data = all_data[:TRAIN_SEQS]
    val_data = all_data[TRAIN_SEQS:TRAIN_SEQS + VAL_SEQS]
    print(f'  Train: {len(train_data)} seqs, Val: {len(val_data)} seqs')

    print('\n  Caching train activations...')
    train_cache = cache_activations(model, fa_layer, train_data, device)
    print(f'  Train cache shape: {train_cache.shape}, dtype: {train_cache.dtype}')

    print('\n  Caching val activations...')
    val_cache = cache_activations(model, fa_layer, val_data, device)
    print(f'  Val cache shape: {val_cache.shape}, dtype: {val_cache.dtype}')

    os.makedirs(os.path.dirname(TRAIN_OUT) or '.', exist_ok=True)
    torch.save(train_cache, TRAIN_OUT)
    print(f'  Saved: {TRAIN_OUT} ({os.path.getsize(TRAIN_OUT) / 1e9:.1f} GB)')

    torch.save(val_cache, VAL_OUT)
    print(f'  Saved: {VAL_OUT} ({os.path.getsize(VAL_OUT) / 1e9:.1f} GB)')

    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
