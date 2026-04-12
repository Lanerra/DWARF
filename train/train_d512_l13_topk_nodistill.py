#!/usr/bin/env python3
"""
🔬 DWARF D=512 L=13 TopK Sparse Attention — NO DISTILLATION (ablation)

Tests whether TopK sparse attention can learn relay routing from SCRATCH
with no teacher guidance - just standard CE loss on language modeling.

This is an ABLATION of the distillation experiment:
- Same architecture as TopK distillation (D512/L13, TopKSparseAttention k=64)
- NO distillation loss (pure CE only)
- NO warm-start from teacher (random init for TopK projections)
- Same data, hyperparameters

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_topk_nodistill.py
"""

import math, os, subprocess, sys, time, types
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    _BNB = True
except ImportError:
    _BNB = False

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pathlib as _pl
_root = str(_pl.Path(__file__).resolve().parent.parent)
for _d in [os.path.join(_root, 'kernels'), _root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from train.train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16 import (
    TriadicJ96, EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS,
    FULL_ATTN_LAYER, VOCAB_SIZE, MAX_SEQ_LEN, SCALE_EMBED_INIT_VAL, DROPOUT,
    SCALE_EMBED_LR_MULT, DSQGAttentionGrouped,
    TOKENIZER_CANDIDATES, BPETokenizerWrapper,
    PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE,
    _PASSKEY_WORDS, _FILLER_SENTENCE, _INTRO_TEMPLATE, _RETRIEVAL_CUE,
    _amp_context, passkey_accuracy,
)
from kernels.topk_sparse_attn import TopKSparseAttention

TOPK_K          = 64
LR              = 3e-4
BATCH_SIZE      = 4
GRAD_ACCUM      = 8
MAX_TRAIN_SEQS  = 20_000
MAX_VAL_SEQS    = 1_000
EPOCHS          = 3
WARMUP_STEPS    = 100

DATASET_PATH    = 'logs/mixed_encoded_2048_fineweb_tok.pt'
CHECKPOINT_DIR  = 'autoresearch/checkpoints'
CKPT_NAME       = 'd512_l13_topk_nodistill'


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i + BATCH_SIZE, :-1].to(device, non_blocking=True)
        y = data[i:i + BATCH_SIZE, 1:].to(device, non_blocking=True).long()
        with _amp_context(device):
            logits = model(x)
        loss = F.cross_entropy(
            logits.contiguous().reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def train():
    device = 'cuda'
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    head_dim = EMBEDDING_DIM // NUM_HEADS
    fa_layer = FULL_ATTN_LAYER

    print('=' * 70)
    print('  🔬 DWARF D512-L13 TopK NO DISTILLATION (ablation)')
    print(f'  Replacing FA@L{fa_layer} with TopKSparseAttention(k={TOPK_K})')
    print('  NO distillation loss - pure CE only')
    print('  NO warm-start - random init')
    print('=' * 70)
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM} H={NUM_HEADS} hd={head_dim} '
          f'L={NUM_LAYERS} FFN={FFN_DIM}')
    print(f'  TopK k={TOPK_K}')
    print(f'  BS={BATCH_SIZE} × GA={GRAD_ACCUM} = '
          f'eff_batch={BATCH_SIZE * GRAD_ACCUM}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    assert tok_path is not None, 'Tokenizer not found'
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'  Tokenizer: {tok_path}')

    assert os.path.isfile(DATASET_PATH), f'Missing: {DATASET_PATH}'
    raw = torch.load(DATASET_PATH, weights_only=True)
    all_data = (raw.get('train', raw.get('data'))
                if isinstance(raw, dict) else raw).long()
    assert len(all_data) >= MAX_TRAIN_SEQS + MAX_VAL_SEQS
    train_data = all_data[:MAX_TRAIN_SEQS]
    val_data = all_data[MAX_TRAIN_SEQS:MAX_TRAIN_SEQS + MAX_VAL_SEQS]
    del raw, all_data
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs')

    # ── Build model with TopK replacing FA ──
    model = TriadicJ96(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS, ffn_dim=FFN_DIM, seq_len=MAX_SEQ_LEN,
        full_attn_layer=fa_layer,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL, dropout=DROPOUT,
    ).to(device)

    # Replace FA with TopK sparse attention (RANDOM INIT - no warm-start)
    model.blocks[fa_layer].attn = TopKSparseAttention(
        D=EMBEDDING_DIM, H=NUM_HEADS, hd=head_dim, k=TOPK_K,
    ).to(device)
    print(f'  Replaced blocks[{fa_layer}].attn → '
          f'TopKSparseAttention(k={TOPK_K}) [RANDOM INIT]')

    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {number_of_parameters:,} '
          f'({number_of_parameters / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')

    # ── Checkpoint all blocks for memory efficiency ──
    def _checkpoint_all_except_topk(self, block_idx):
        return block_idx != self.full_attn_layer
    model._should_checkpoint_block = types.MethodType(
        _checkpoint_all_except_topk, model)

    # ── Optimizer ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    se_ids = {id(p) for p in model.scale_embed_parameters()}
    se_params = [p for p in model.parameters() if id(p) in se_ids]
    other_params = [p for p in model.parameters() if id(p) not in se_ids]

    AdamW = bnb.optim.AdamW8bit if _BNB else torch.optim.AdamW
    optimizer = AdamW([
        {'params': other_params, 'lr': LR},
        {'params': se_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    total_steps = EPOCHS * steps_per_epoch

    def _lr_main(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[_lr_main, lambda _step: 1.0])

    print(f'  steps_per_epoch={steps_per_epoch}  total_steps={total_steps}')
    print(f'  Optimizer: {"AdamW8bit" if _BNB else "AdamW"}')
    print()

    # ── Training ──
    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}
    global_step = 0
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        optimizer.zero_grad(set_to_none=True)
        step_times = deque(maxlen=20)
        epoch_ce, epoch_count = 0.0, 0

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                seq_idx = indices[idx_start:idx_start + BATCH_SIZE]
                batch = train_data[seq_idx]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True).long()

                with _amp_context(device):
                    logits = model(x)

                # Pure CE loss - NO distillation
                ce_loss = F.cross_entropy(
                    logits.contiguous().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )

                loss = ce_loss / GRAD_ACCUM
                loss.backward()

                epoch_ce += ce_loss.item()
                epoch_count += 1

                del logits, ce_loss, loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            step_ms = (time.time() - t0) * 1000
            step_times.append(step_ms)

            if global_step % 50 == 0:
                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000)
                avg_ce = epoch_ce / epoch_count
                print(f'  [ep{epoch} step{global_step:4d}] '
                      f'ce={avg_ce:.4f} '
                      f'lr={scheduler.get_last_lr()[0]:.2e} '
                      f'{tok_s:.0f} tok/s', flush=True)

        # ── Validation ──
        val_loss = evaluate(model, val_data, device)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        # ── Passkey eval ──
        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100

        print(f'\n{"=" * 70}')
        print(f'  Epoch {epoch}/{EPOCHS}')
        print(f'  Train: ce={epoch_ce / epoch_count:.4f}')
        print(f'  Val:   loss={val_loss:.4f}  ppl={val_ppl:.2f}')
        print(f'  Passkey: mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print(f'  {" ".join(parts)}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules()
                   if isinstance(m, DSQGAttentionGrouped)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} '
                  f'|max|={se_all.max():.4f}')
        print(f'  Physics: {model.physics_summary()}')
        print(f'{"=" * 70}\n')

        clean = {k.replace('._orig_mod', ''): v
                 for k, v in model.state_dict().items()}
        torch.save({
            'model_state_dict': clean,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'config': {'topk_k': TOPK_K, 'fa_layer': fa_layer},
        }, os.path.join(CHECKPOINT_DIR, f'{CKPT_NAME}_ep{epoch}.pt'))
        print(f'  Saved: {CKPT_NAME}_ep{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': clean,
                'config': {'topk_k': TOPK_K, 'fa_layer': fa_layer},
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_NAME}_best.pt'))
            print(f'  ✓ New best: {CKPT_NAME}_best.pt')

    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f'\n{"=" * 70}')
    print('  🔬 TopK NO DISTILL Ablation Summary')
    print(f'{"=" * 70}')
    for ep in range(1, EPOCHS + 1):
        print(f'  ep{ep}: ppl={ppl_results.get(ep, 999):.2f}  '
              f'passkey={passkey_results.get(ep, 0):.1f}%')
    print(f'  peak_vram={peak_mb:.0f}MB  elapsed={elapsed:.0f}s')
    print(f'  params={number_of_parameters / 1e6:.1f}M  topk_k={TOPK_K}')

    pk_final = passkey_results.get(EPOCHS, 0)
    if pk_final >= 70:
        print('\n  ✅ MECHANISM VALIDATED — TopK learns relay from scratch!')
    elif pk_final >= 30:
        print('\n  ⚠️ PARTIAL — TopK can partially learn relay without distillation')
    else:
        print('\n  ❌ BELOW THRESHOLD — distillation is necessary for relay')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
