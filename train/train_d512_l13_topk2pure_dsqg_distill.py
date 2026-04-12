#!/usr/bin/env python3
"""
🔬 DWARF D=512 L=13 TopK→Pure DSQG Distillation

Two-step distillation chain:
  Step 1: Run TopK teacher in eval mode, cache blocks[3] outputs (TopK attention)
  Step 2: Train pure DSQG student at blocks[3] with MSE against cached TopK outputs

Rationale: TopK's routing is learnable (soft attention) vs FA's content-addressed
hard lookup. TopK outputs may be more directly translatable to DSQG fixed-offset
routing than FA outputs were. If this works: confirms relay = learnable routing topology.

Loss: LM (CE) + λ·MSE(student_DSQG_L3, teacher_TopK_L3)
  λ anneals 1.0 → 0.1 over training

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_topk2pure_dsqg_distill.py
"""

import math, os, subprocess, sys, time, types
from collections import deque
import contextlib
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
    LAYER_LAYOUT, GROUP_A, GROUP_B, GROUP_C,
    J_SMALL_A, J_LARGE_A, J_SMALL_B, J_LARGE_B, J_SMALL_C, J_LARGE_C,
    TOKENIZER_CANDIDATES, BPETokenizerWrapper,
    PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE,
    _PASSKEY_WORDS, _FILLER_SENTENCE, _INTRO_TEMPLATE, _RETRIEVAL_CUE,
    _amp_context, passkey_accuracy,
)
from kernels.topk_sparse_attn import TopKSparseAttention

TOPK_K          = 64
LR              = 2.5e-4          # mixed-domain LR for D=512
BATCH_SIZE      = int(os.environ.get('DWARF_BS', '2'))
GRAD_ACCUM      = int(os.environ.get('DWARF_GA', '8'))
MAX_TRAIN_SEQS  = 20_000
MAX_VAL_SEQS    = 1_000
EPOCHS          = 3
WARMUP_STEPS    = 100
LAMBDA_START    = 1.0
LAMBDA_END      = 0.1

TEACHER_CKPT    = 'autoresearch/checkpoints/d512_l13_topk_distill_best.pt'
TOPK_CACHE_TRAIN = 'logs/topk_cache_d512_l3_train_20k'
TOPK_CACHE_VAL   = 'logs/topk_cache_d512_l3_val_1k'
DATASET_PATH    = 'logs/mixed_encoded_2048_fineweb_tok.pt'
CHECKPOINT_DIR  = 'autoresearch/checkpoints'
CKPT_NAME       = 'd512_l13_topk2pure_dsqg_distill'

TOPK_LAYER      = FULL_ATTN_LAYER   # = 3 for D=512 L=13


# =============================================================================
# TEACHER: D512 TopK model (for caching)
# =============================================================================

def build_teacher_model(device):
    """Build the D512 TopK teacher model and load checkpoint."""
    print(f"  Building D512 TopK teacher from {TEACHER_CKPT}")
    head_dim = EMBEDDING_DIM // NUM_HEADS
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

    # Replace blocks[3] with TopKSparseAttention (same as original training)
    model.blocks[TOPK_LAYER].attn = TopKSparseAttention(
        D=EMBEDDING_DIM, H=NUM_HEADS, hd=head_dim, k=TOPK_K,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Teacher loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


# =============================================================================
# CACHE TopK ACTIVATIONS
# =============================================================================

@torch.inference_mode()
def cache_topk_activations(model, data, device, cache_path, label):
    """Run teacher TopK model and cache blocks[3] outputs to disk in chunks."""
    import os
    chunk_size = 2000
    n_chunks = (len(data) + chunk_size - 1) // chunk_size
    all_chunks_exist = all(
        os.path.isfile(f'{cache_path}.chunk{i}') for i in range(n_chunks)
    )
    if all_chunks_exist:
        print(f"  ✓ TopK cache chunks exist: {cache_path}.chunk* ({n_chunks} chunks)")
        return

    print(f"  ⏳ Caching {label} TopK activations at L{TOPK_LAYER} ({len(data)} seqs) → {cache_path} ({n_chunks} chunks)")
    model.eval()
    batch_size = 4
    sequence_length = data.shape[1] - 1

    _topk_out = {}
    def _hook(module, inp, out):
        # out is [B, N, D] from TopKSparseAttention.forward
        _topk_out['val'] = out.detach().cpu().to(torch.bfloat16)
    handle = model.blocks[TOPK_LAYER].attn.register_forward_hook(_hook)

    for chunk_idx in range(n_chunks):
        chunk_path = f'{cache_path}.chunk{chunk_idx}'
        if os.path.isfile(chunk_path):
            print(f"    Chunk {chunk_idx} already exists, skipping")
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(data))
        n_seqs = end - start

        print(f"  📦 Chunk {chunk_idx}/{n_chunks-1}: seqs {start}-{end}")
        # TopK output: [B, N, D] = [B, 2047, 512]
        chunk_cache = torch.empty(n_seqs, sequence_length, EMBEDDING_DIM, dtype=torch.bfloat16)

        for i in range(start, end, batch_size):
            batch_end = min(i + batch_size, end)
            x = data[i:batch_end, :-1].to(device, non_blocking=True)
            with _amp_context(device):
                model(x)
            chunk_cache[i - start:batch_end - start] = _topk_out['val']
            if ((i - start) // batch_size) % 100 == 0:
                print(f"    {(i - start)}/{n_seqs}", flush=True)

        torch.save(chunk_cache, chunk_path)
        print(f"    ✓ Saved: {chunk_path} ({chunk_cache.shape})")
        del chunk_cache
        import gc; gc.collect()

    handle.remove()
    print(f"  ✓ All {n_chunks} chunks saved for {cache_path}")


# =============================================================================
# EVALUATION
# =============================================================================

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


# =============================================================================
# PASSKEY EVALUATION
# =============================================================================

def run_passkey(model, tokenizer, device, distances, trials, batch_size):
    """
    Batch-passkey evaluation matching the reference implementation structure.
    Uses flat _PASSKEY_WORDS list + _INTRO_TEMPLATE['{word}'], _FILLER_SENTENCE, _RETRIEVAL_CUE.
    """
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE).ids
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE).ids
    pad_id = 0

    # Pre-encode word -> token id
    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word).ids or tokenizer.encode(word).ids
        if not encoded:
            raise ValueError(f'Could not encode passkey word: {word}')
        word_token_ids[word] = encoded[0]

    hit = {d: 0 for d in distances}
    counts = {d: 0 for d in distances}

    for dist in distances:
        for i in range(trials):
            target = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target)).ids
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if dist > available:
                continue

            # Build filler of exactly 'dist' tokens
            filler = []
            while len(filler) < dist:
                filler.extend(filler_ids)
            filler = filler[:dist]

            full_seq = intro_ids + filler + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue

            # Pad to MAX_SEQ_LEN
            padded = full_seq + [pad_id] * (MAX_SEQ_LEN - len(full_seq))
            last_pos = len(full_seq) - 1  # position of last token (cue's last word)

            # Candidate words for softmax
            others = [w for w in _PASSKEY_WORDS if w != target]
            cand_words = [target] + others[:9]
            cand_ids = [word_token_ids[w] for w in cand_words]

            # Forward pass
            input_ids = torch.tensor([padded], device=device)
            with _amp_context(device):
                logits = model(input_ids)  # [1, MAX_SEQ_LEN, V]
            logits_last = logits[0, last_pos]  # [V]
            probs = F.softmax(logits_last.float(), dim=-1)
            predicted_id = torch.argmax(probs).item()
            predicted_word = tokenizer.decode([predicted_id]).strip().lower()
            target_word = target.strip().lower()

            counts[dist] += 1
            if predicted_word == target_word:
                hit[dist] += 1

    # Avoid div-by-zero for distances with no valid trials
    return {d: hit[d] / counts[d] * 100 if counts[d] > 0 else 0.0 for d in distances}


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

def compute_distill_loss(student_out, teacher_cache, seq_idx, device):
    """
    Compute MSE between student's DSQG output at layer TOPK_LAYER and
    cached teacher TopK output.

    student_out: [B, N, D] — student's blocks[TOPK_LAYER].attn output
    teacher_cache: [B, N, D] — cached TopK outputs from teacher
    """
    B = student_out.shape[0]
    # Align batch indices
    start = (seq_idx * B) % teacher_cache.shape[0]
    end = min(start + B, teacher_cache.shape[0])
    actual_B = end - start
    if actual_B < B:
        # Pad or wrap — shouldn't happen with 20K seqs
        teacher_batch = teacher_cache[start:end]
        pad = student_out[actual_B:].detach()
        teacher_batch = F.pad(teacher_batch, (0,0,0,0))
        teacher_batch[:B-actual_B] = pad
    else:
        teacher_batch = teacher_cache[start:end]
    teacher_act = teacher_batch.to(student_out.device, non_blocking=True)
    return F.mse_loss(student_out.float(), teacher_act.float())


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = 'cuda'
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print(f"🔬 DWARF D512-L13 TopK→Pure DSQG Distillation")
    print(f"  TopK teacher → pure DSQG student at L{TOPK_LAYER}")
    print(f"  LR={LR}, BS={BATCH_SIZE}×GA={GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  git={git_hash}")
    print()

    # ── Tokenizer & Data ──────────────────────────────────────────────────────
    tok_path = 'results/fineweb_tokenizer_32k.json'
    for candidate in TOKENIZER_CANDIDATES:
        if os.path.exists(candidate):
            tok_path = candidate
            break
    print(f"📂 Tokenizer: {tok_path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    print(f"📂 Loading dataset: {DATASET_PATH}")
    data = torch.load(DATASET_PATH, mmap=True)
    train_data = data['train'][:MAX_TRAIN_SEQS]
    val_data   = data['val'][:MAX_VAL_SEQS]
    print(f"  train: {train_data.shape}  val: {val_data.shape}")

    # ── Build Teacher & Cache TopK Activations ────────────────────────────────
    print("\n📂 Loading teacher (D512 TopK model)...")
    teacher = build_teacher_model(device)
    print(f"  Teacher ready. Caching TopK outputs at L{TOPK_LAYER}...")

    os.makedirs('logs', exist_ok=True)
    cache_topk_activations(teacher, train_data, device, TOPK_CACHE_TRAIN, "train")
    cache_topk_activations(teacher, val_data, device, TOPK_CACHE_VAL, "val")

    # Unload teacher from GPU (free VRAM)
    del teacher
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Load cached activations
    print(f"\n📂 Loading cached TopK activations...")
    n_train_chunks = (len(train_data) + 1999) // 2000
    n_val_chunks   = (len(val_data) + 1999) // 2000
    topk_cache_train = torch.cat([
        torch.load(f'{TOPK_CACHE_TRAIN}.chunk{i}', weights_only=False)
        for i in range(n_train_chunks)
    ])
    topk_cache_val = torch.cat([
        torch.load(f'{TOPK_CACHE_VAL}.chunk{i}', weights_only=False)
        for i in range(n_val_chunks)
    ])
    print(f"  TopK cache train: {topk_cache_train.shape}")
    print(f"  TopK cache val:   {topk_cache_val.shape}")

    # ── Build Pure DSQG Student Model ────────────────────────────────────────
    print("\n🔧 Building pure DSQG student model...")
    student = TriadicJ96(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
    ).to(device)

    # At TOPK_LAYER=3, blocks[3].attn is already DSQGAttentionGrouped from TriadicJ96.
    # No weight loading from teacher (random init) — pure student.
    print(f"  Student built. Layout: L{TOPK_LAYER}=DSQG (no teacher warm-start)")
    print(f"  Layout: {', '.join(f'L{i}:{_l[0]}' for i, _l in enumerate(LAYER_LAYOUT) if _l[0] != 'FA')}")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Register Hook on Student L{TOPK_LAYER} for Distillation ─────────────────
    _student_layer_out = {}
    def _student_hook(module, inp, out):
        # out: [B, N, D] from DSQGAttentionGrouped.forward
        _student_layer_out['val'] = out
    handle = student.blocks[TOPK_LAYER].attn.register_forward_hook(_student_hook)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    assert (BATCH_SIZE * GRAD_ACCUM) % 2 == 0, "effective batch must be even"
    eff_batch = BATCH_SIZE * GRAD_ACCUM
    lr_mult = SCALE_EMBED_LR_MULT
    scale_embed_lr = LR * lr_mult
    opt_params = [
        {'params': student.parameters(), 'lr': LR, 'weight_decay': 0.1},
    ]

    if _BNB:
        optimizer = bnb.optim.AdamW8bit(opt_params, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(opt_params, betas=(0.9, 0.95))

    total_steps = (len(train_data) // eff_batch) * EPOCHS
    warmup_steps = WARMUP_STEPS
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.01, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n  Optimizer: {'AdamW8bit' if _BNB else 'AdamW8bit'}")
    print(f"  LR={LR}, scale_embed_LR={scale_embed_lr:.2e} (×{lr_mult:.1f})")
    print(f"  steps_per_epoch={len(train_data)//eff_batch}, total_steps={total_steps}")

    # ── Mixed Precision ───────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    seq_idx = 0
    step = 0
    best_ppl = float('inf')

    for epoch in range(1, EPOCHS + 1):
        student.train()
        epoch_ce, epoch_dist, epoch_tokens = 0.0, 0.0, 0
        buf_loss, buf_steps = deque(maxlen=50), 0
        t_ep = time.time()

        for i in range(0, len(train_data) - eff_batch + 1, eff_batch):
            # Lambda annealing: 1.0 → 0.1 linear
            progress = step / max(total_steps, 1)
            lam = LAMBDA_END + (LAMBDA_START - LAMBDA_END) * (1.0 - progress)

            # Data batch
            xb = train_data[i:i + eff_batch, :-1].to(device, non_blocking=True)
            yb = train_data[i:i + eff_batch, 1:].to(device, non_blocking=True).long()

            # Forward (mixed precision)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = student(xb)  # [B, N, V]
                ce_loss = F.cross_entropy(
                    logits.contiguous().reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                )
                # Distillation: MSE between student L3 output and cached TopK
                student_act = _student_layer_out['val']  # [B, N, D]
                # Compute MSE against aligned cache entries
                B_eff = eff_batch
                cache_start = (seq_idx * B_eff) % (len(topk_cache_train) - B_eff)
                teacher_act = topk_cache_train[cache_start:cache_start + B_eff].to(device, non_blocking=True)
                distill_loss = F.mse_loss(student_act.float(), teacher_act.float())
                loss = ce_loss + lam * distill_loss

            # Backward
            scaler.scale(loss).backward()
            if ((i // eff_batch) + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

            epoch_ce   += ce_loss.item() * xb.shape[0]
            epoch_dist += distill_loss.item() * xb.shape[0]
            epoch_tokens += xb.shape[0] * xb.shape[1]
            buf_loss.append(loss.item())
            buf_steps += 1
            seq_idx += 1

            if buf_steps % 100 == 0:
                avg_loss = sum(buf_loss) / len(buf_loss)
                lr_now = scheduler.get_last_lr()[0]
                tok_s = int(epoch_tokens / max(time.time() - t_ep, 0.1))
                print(f"  [ep{epoch} step{step}] ce={avg_loss:.4f} distill={distill_loss.item():.4f} "
                      f"λ={lam:.3f} lr={lr_now:.2e} {tok_s} tok/s")

        # ── Epoch Eval ──────────────────────────────────────────────────────
        student.eval()
        val_loss = evaluate(student, val_data, device)
        val_ppl = math.exp(val_loss)
        hit = run_passkey(student, tokenizer, device, PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE)
        mean_passkey = sum(hit.values()) / len(hit)

        # scale_embed stats
        se = student.blocks[TOPK_LAYER].attn.scale_embed
        se_max = se.detach().abs().max().item()
        se_mean = se.detach().abs().mean().item()

        elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f" Epoch {epoch}/{EPOCHS}")
        print(f" Train: ce={epoch_ce/max(epoch_tokens,1)*math.exp2(10):.4f} "
              f"distill={epoch_dist/max(epoch_tokens,1)*math.exp2(10):.6f}")
        print(f" Val: loss={val_loss:.4f} ppl={val_ppl:.2f}")
        print(f" Passkey: mean={mean_passkey:.1%}")
        for d in sorted(hit):
            print(f"  d={d}: {hit[d]:.0f}%", end='')
        print()
        print(f" scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}")
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f" peak_vram={peak_vram:.0f}MB elapsed={elapsed:.0f}s")
        print(f"{'='*70}\n")

        # Checkpoint
        ckpt_path = f'{CHECKPOINT_DIR}/{CKPT_NAME}_ep{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_ppl': val_ppl,
            'passkey': mean_passkey,
            'git_hash': git_hash,
        }, ckpt_path)
        print(f"  ✓ Saved: {ckpt_path}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_path = f'{CHECKPOINT_DIR}/{CKPT_NAME}_best.pt'
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': student.state_dict(),
                'val_ppl': val_ppl,
                'passkey': mean_passkey,
                'git_hash': git_hash,
            }, best_path)
            print(f"  ✓ Saved best: {best_path}")

    handle.remove()
    total_elapsed = time.time() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n✅ Training complete. total={total_elapsed:.0f}s peak_vram={peak_vram:.0f}MB")


if __name__ == '__main__':
    train()
