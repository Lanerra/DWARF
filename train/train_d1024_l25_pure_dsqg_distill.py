#!/usr/bin/env python3
"""
🔬 DWARF D=1024 L=25 Pure DSQG Distillation — from cached FA teacher

PURE DSQG student: no TopK, no FA. Standard triadic DSQG everywhere
including at L/4 (layer 6). Distillation loss is MSE between the
student's DSQG output at layer 6 and the cached teacher FA activations.

Loss: LM (CE) + λ·MSE(student_DSQG_out_L6, teacher_FA_cache)
  λ anneals 1.0 → 0.1 over training

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d1024_l25_pure_dsqg_distill.py
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

from train.train_d1024_l25_triadic_aabbc_h200_bf16 import (
    TriadicJ96, EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS,
    FULL_ATTN_LAYER, VOCAB_SIZE, MAX_SEQ_LEN, SCALE_EMBED_INIT_VAL, DROPOUT,
    SCALE_EMBED_LR_MULT, DSQGAttentionGrouped,
    LAYER_LAYOUT, GROUP_A, GROUP_B, GROUP_C,
    J_SMALL_A, J_LARGE_A, J_SMALL_B, J_LARGE_B, J_SMALL_C, J_LARGE_C,
    TOKENIZER_CANDIDATES, BPETokenizerWrapper,
    PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE,
    _PASSKEY_WORDS, _FILLER_SENTENCE, _INTRO_TEMPLATE, _RETRIEVAL_CUE,
    _amp_context, passkey_accuracy,
)

TOPK_K          = 64   # placeholder — not used in pure DSQG
LR              = 2e-4
BATCH_SIZE      = int(os.environ.get('DWARF_BS', '1'))
GRAD_ACCUM      = int(os.environ.get('DWARF_GA', '16'))
MAX_TRAIN_SEQS  = 10_000
MAX_VAL_SEQS    = 1_000
EPOCHS          = 3
WARMUP_STEPS    = 100
LAMBDA_START    = 1.0
LAMBDA_END      = 0.1

TEACHER_CKPT    = 'autoresearch/checkpoints/d1024_l25_triadic_aabbc_mixed_scratch_best.pt'
FA_CACHE_TRAIN  = 'logs/fa_cache_d1024_teacher_train_10k'
FA_CACHE_VAL    = 'logs/fa_cache_d1024_teacher_val_1k'
DATASET_PATH    = 'logs/mixed_encoded_2048_fineweb_tok.pt'
CHECKPOINT_DIR  = 'autoresearch/checkpoints'
CKPT_NAME       = 'd1024_l25_pure_dsqg_distill'


def _get_layer_group_config(fa_layer):
    """Get DSQG config (offsets, j_small, j_large) for the FA layer position."""
    entry = LAYER_LAYOUT[fa_layer]
    if entry[0] == "FA":
        # FA layer slot: use GROUP_C offsets (the group at L/4 position)
        return "C", GROUP_C, J_SMALL_C, J_LARGE_C
    label, offsets, j_small, j_large, _ = entry
    return label, offsets, j_small, j_large


@torch.inference_mode()
def cache_fa_activations(model, data, fa_layer, device, cache_path, label):
    """Run teacher model and cache FA layer outputs to disk in chunks."""
    chunk_size = 2000
    n_chunks = (len(data) + chunk_size - 1) // chunk_size
    all_chunks_exist = all(
        os.path.isfile(f'{cache_path}.chunk{i}') for i in range(n_chunks)
    )
    if all_chunks_exist:
        print(f'  ✓ FA cache chunks exist: {cache_path}.chunk* ({n_chunks} chunks)')
        return

    print(f'  ⏳ Caching {label} FA activations ({len(data)} seqs) → {cache_path} ({n_chunks} chunks)')
    model.eval()
    batch_size = 2
    sequence_length = data.shape[1] - 1

    _fa_out = {}
    def _hook(module, inp, out):
        _fa_out['val'] = out.detach().cpu().to(torch.bfloat16)
    handle = model.blocks[fa_layer].attn.register_forward_hook(_hook)

    for chunk_idx in range(n_chunks):
        chunk_path = f'{cache_path}.chunk{chunk_idx}'
        if os.path.isfile(chunk_path):
            print(f'    Chunk {chunk_idx} already exists, skipping')
            continue
            
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(data))
        n_seqs = end - start
        
        print(f'  📦 Chunk {chunk_idx}/{n_chunks-1}: seqs {start}-{end}')
        chunk_cache = torch.empty(n_seqs, sequence_length, EMBEDDING_DIM, dtype=torch.bfloat16)
        
        for i in range(start, end, batch_size):
            batch_end = min(i + batch_size, end)
            x = data[i:batch_end, :-1].to(device, non_blocking=True)
            with _amp_context(device):
                model(x)
            chunk_cache[i - start:batch_end - start] = _fa_out['val']
            if ((i - start) // batch_size) % 100 == 0:
                print(f'    {(i - start)}/{n_seqs}', flush=True)

        torch.save(chunk_cache, chunk_path)
        print(f'    ✓ Saved: {chunk_path} ({chunk_cache.shape})')
        del chunk_cache
        import gc; gc.collect()

    handle.remove()
    print(f'  ✓ All {n_chunks} chunks saved for {cache_path}')


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

    # Get DSQG config for layer 6 (GROUP_C layout at FA position)
    group_label, dsqg_offsets, j_small, j_large = _get_layer_group_config(fa_layer)

    print('=' * 70)
    print('  🔬 DWARF D1024-L25 Pure DSQG Distillation (cached FA teacher)')
    print(f'  FA@L{fa_layer} → DSQGAttentionGrouped({group_label})')
    print('=' * 70)
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM} H={NUM_HEADS} hd={head_dim} '
          f'L={NUM_LAYERS} FFN={FFN_DIM}')
    print(f'  DSQG L6: {group_label} offsets (J={len(dsqg_offsets)}) '
          f'j_small={j_small} j_large={j_large}')
    print(f'  λ: {LAMBDA_START} → {LAMBDA_END} (linear)')
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
    if isinstance(raw, dict):
        all_train = raw.get('train', raw.get('data')).long()
        all_val = raw.get('val')
        if all_val is not None:
            all_val = all_val.long()
    else:
        all_train = raw.long()
        all_val = None

    assert len(all_train) >= MAX_TRAIN_SEQS
    train_data = all_train[:MAX_TRAIN_SEQS]
    if all_val is not None and len(all_val) >= MAX_VAL_SEQS:
        val_data = all_val[:MAX_VAL_SEQS]
    else:
        val_data = all_train[MAX_TRAIN_SEQS:MAX_TRAIN_SEQS + MAX_VAL_SEQS]
    del raw, all_train, all_val
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs')

    # ── Cache FA activations if needed (chunks) ──
    train_chunk_size = 2000
    n_train_chunks = (len(train_data) + train_chunk_size - 1) // train_chunk_size
    val_chunk_size = 1000
    n_val_chunks = 1
    
    train_chunks_exist = all(
        os.path.isfile(f'{FA_CACHE_TRAIN}.chunk{i}') for i in range(n_train_chunks)
    )
    val_chunks_exist = all(
        os.path.isfile(f'{FA_CACHE_VAL}.chunk{i}') for i in range(n_val_chunks)
    )
    
    if not train_chunks_exist or not val_chunks_exist:
        print('\n  📦 Building FA activation cache from teacher...')
        assert os.path.isfile(TEACHER_CKPT), f'Missing teacher: {TEACHER_CKPT}'
        teacher = TriadicJ96(
            vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS, ffn_dim=FFN_DIM, seq_len=MAX_SEQ_LEN,
            full_attn_layer=fa_layer,
            scale_embed_init_val=SCALE_EMBED_INIT_VAL, dropout=DROPOUT,
        ).to(device)
        ckpt = torch.load(TEACHER_CKPT, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        state = {k.replace('._orig_mod', ''): v for k, v in state.items()}
        teacher.load_state_dict(state, strict=False)
        del ckpt, state
        teacher.eval()

        cache_fa_activations(teacher, train_data, fa_layer, device,
                             FA_CACHE_TRAIN, 'train')
        cache_fa_activations(teacher, val_data, fa_layer, device,
                             FA_CACHE_VAL, 'val')
        del teacher
        torch.cuda.empty_cache()
        print()

    # Load from chunks
    print(f'  📂 Loading FA cache from chunks...')
    train_chunks = [torch.load(f'{FA_CACHE_TRAIN}.chunk{i}', mmap=True) for i in range(n_train_chunks)]
    fa_cache_train = torch.cat(train_chunks, dim=0)
    del train_chunks; import gc; gc.collect()
    
    val_chunks = [torch.load(f'{FA_CACHE_VAL}.chunk{i}', mmap=True) for i in range(n_val_chunks)]
    fa_cache_val = torch.cat(val_chunks, dim=0)
    del val_chunks; gc.collect()
    print(f'  FA cache train: {fa_cache_train.shape}  '
          f'val: {fa_cache_val.shape}')

    # ── Build student with PURE DSQG at L/4 ──
    model = TriadicJ96(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS, ffn_dim=FFN_DIM, seq_len=MAX_SEQ_LEN,
        full_attn_layer=fa_layer,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL, dropout=DROPOUT,
    ).to(device)

    # Create DSQGAttentionGrouped for layer 6 with same group config as LAYER_LAYOUT
    dsqg_layer_6 = DSQGAttentionGrouped(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        offsets=dsqg_offsets,
        j_small=j_small,
        j_large=j_large,
        seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    model.blocks[fa_layer].attn = dsqg_layer_6
    print(f'  Replaced blocks[{fa_layer}].attn → '
          f'DSQGAttentionGrouped({group_label}, J={len(dsqg_offsets)})')

    # ── Load teacher weights, warm-start DSQG from FA projections ──
    assert os.path.isfile(TEACHER_CKPT)
    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('._orig_mod', ''): v for k, v in state.items()}
    del ckpt

    qkv_key = f'blocks.{fa_layer}.attn.qkv_proj.weight'
    out_key = f'blocks.{fa_layer}.attn.out_proj.weight'
    gate_key = f'blocks.{fa_layer}.attn.gate_proj.weight'
    if qkv_key in state:
        qkv_w = state[qkv_key]
        D = EMBEDDING_DIM
        # DSQG qkv_proj same shape as teacher qkv_proj [3D, D]
        state[f'blocks.{fa_layer}.attn.qkv_proj.weight'] = qkv_w.clone()
        print('  Warm-started DSQG qkv_proj from teacher FA qkv_proj')
    if out_key in state:
        state[f'blocks.{fa_layer}.attn.out_proj.weight'] = state[out_key].clone()
        print('  Warm-started DSQG out_proj from teacher FA out_proj')
    if gate_key in state:
        state[f'blocks.{fa_layer}.attn.gate_proj.weight'] = state[gate_key].clone()
        print('  Warm-started DSQG gate_proj from teacher FA gate_proj')

    # Filter: remove FA-only keys the teacher doesn't have (DSQG inits them randomly)
    fa_prefix = f'blocks.{fa_layer}.attn.'
    keep_keys = {
        f'blocks.{fa_layer}.attn.qkv_proj.weight',
        f'blocks.{fa_layer}.attn.qkv_proj.bias',
        f'blocks.{fa_layer}.attn.out_proj.weight',
        f'blocks.{fa_layer}.attn.out_proj.bias',
        f'blocks.{fa_layer}.attn.gate_proj.weight',
        f'blocks.{fa_layer}.attn.gate_proj.bias',
    }
    state = {k: v for k, v in state.items()
             if not k.startswith(fa_prefix) or k in keep_keys}
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    print(f'  Loaded weights: {len(missing)} missing, '
          f'{len(unexpected)} unexpected')
    for k in missing:
        print(f'    [missing] {k}')

    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {number_of_parameters:,} '
          f'({number_of_parameters / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')

    # ── Checkpoint all blocks EXCEPT DSQG layer 6 (preserve hook gradients) ──
    def _checkpoint_all_except_dsqg(self, block_idx):
        return block_idx != self.full_attn_layer
    model._should_checkpoint_block = types.MethodType(
        _checkpoint_all_except_dsqg, model)

    # ── Hook captures DSQG output at layer 6 WITH gradient graph intact ──
    _fa_out = {}
    def _hook(module, inp, out):
        _fa_out['val'] = out
    hook_handle = model.blocks[fa_layer].attn.register_forward_hook(_hook)
    print(f'  Hook on blocks[{fa_layer}].attn DSQG output (grad-enabled)')

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
        epoch_ce, epoch_distill, epoch_count = 0.0, 0.0, 0

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            frac = global_step / max(total_steps, 1)
            lambda_d = LAMBDA_START * (1.0 - frac) + LAMBDA_END * frac

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

                ce_loss = F.cross_entropy(
                    logits.contiguous().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )

                teacher_act = fa_cache_train[seq_idx].to(
                    device, dtype=torch.bfloat16)
                student_act = _fa_out['val']
                distill_loss = F.mse_loss(
                    student_act.float(), teacher_act.float())

                loss = (ce_loss + lambda_d * distill_loss) / GRAD_ACCUM
                loss.backward()

                epoch_ce += ce_loss.item()
                epoch_distill += distill_loss.item()
                epoch_count += 1

                del logits, ce_loss, distill_loss, loss, teacher_act

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
                avg_dist = epoch_distill / epoch_count
                print(f'  [ep{epoch} step{global_step:4d}] '
                      f'ce={avg_ce:.4f} distill={avg_dist:.6f} '
                      f'λ={lambda_d:.3f} '
                      f'lr={scheduler.get_last_lr()[0]:.2e} '
                      f'{tok_s:.0f} tok/s', flush=True)

        # ── Validation ──
        val_loss = evaluate(model, val_data, device)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        model.eval()
        val_dist_total, val_dist_n = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val_data), BATCH_SIZE):
                bs = min(BATCH_SIZE, len(val_data) - i)
                xv = val_data[i:i + bs, :-1].to(device, non_blocking=True)
                with _amp_context(device):
                    model(xv)
                tv = fa_cache_val[i:i + bs].to(device, dtype=torch.bfloat16)
                val_dist_total += F.mse_loss(
                    _fa_out['val'].float(), tv.float()).item()
                val_dist_n += 1
        val_distill = val_dist_total / max(val_dist_n, 1)

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100

        print(f'\n{"=" * 70}')
        print(f'  Epoch {epoch}/{EPOCHS}')
        print(f'  Train: ce={epoch_ce / epoch_count:.4f}  '
              f'distill={epoch_distill / epoch_count:.6f}')
        print(f'  Val:   loss={val_loss:.4f}  ppl={val_ppl:.2f}  '
              f'distill_mse={val_distill:.6f}')
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
            'config': {'fa_layer': fa_layer, 'group_label': group_label},
        }, os.path.join(CHECKPOINT_DIR, f'{CKPT_NAME}_ep{epoch}.pt'))
        print(f'  Saved: {CKPT_NAME}_ep{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': clean,
                'config': {'fa_layer': fa_layer, 'group_label': group_label},
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_NAME}_best.pt'))
            print(f'  ✓ New best: {CKPT_NAME}_best.pt')

    hook_handle.remove()
    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f'\n{"=" * 70}')
    print('  🔬 Pure DSQG Distillation Summary (D1024-L25)')
    print(f'{"=" * 70}')
    for ep in range(1, EPOCHS + 1):
        print(f'  ep{ep}: ppl={ppl_results.get(ep, 999):.2f}  '
              f'passkey={passkey_results.get(ep, 0):.1f}%')
    print(f'  peak_vram={peak_mb:.0f}MB  elapsed={elapsed:.0f}s')
    print(f'  params={number_of_parameters / 1e6:.1f}M')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
