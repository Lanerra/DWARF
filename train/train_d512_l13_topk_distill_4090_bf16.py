"""
🔬 DWARF D=512 L=13 TopK Sparse Attention — distillation from FA teacher

Mechanism validation: can content-addressed top-k lookup at L/4 replace
the full attention layer while preserving passkey retrieval?

Architecture: D=512, H=8 (hd=64), L=13, FFN=1024, TIED lm_head
  FA block at L3 replaced with TopKSparseAttention (k=64)
  All 13 DSQG blocks untouched. SO(2) MOVT, preIF/EMA, NPCI preserved.

Loss: LM (chunked CE) + λ·MSE(student_FA_out, teacher_FA_cache)
  λ anneals 1.0 → 0.1 over training

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d512_l13_topk_distill_4090_bf16.py
"""

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available, using standard AdamW")

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from train.train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16 import (
    TriadicJ96, EMBEDDING_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS,
    FULL_ATTN_LAYER, VOCAB_SIZE, MAX_SEQ_LEN, SCALE_EMBED_INIT_VAL, DROPOUT,
    SCALE_EMBED_LR_MULT, EMA_INIT,
    DSQGAttentionGrouped,
    GROUP_A, GROUP_B, GROUP_C, GROUP_T,
    TOKENIZER_CANDIDATES, BPETokenizerWrapper,
    PASSKEY_DISTANCES, PASSKEY_TRIALS, PASSKEY_BATCH_SIZE,
    _PASSKEY_WORDS, _FILLER_SENTENCE, _INTRO_TEMPLATE, _RETRIEVAL_CUE,
    _amp_context, passkey_accuracy, _unwrap_compiled_module,
)
from kernels.topk_sparse_attn import TopKSparseAttention

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

TOPK_K          = 64
LR              = 3e-4
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
MAX_TRAIN_SEQS  = 20_000
MAX_VAL_SEQS    = 1_000
CE_CHUNK        = 512
SCREEN_EPOCHS   = 3

TEACHER_CKPT    = 'autoresearch/checkpoints/d512_l13_triadic_aabbc_mixed_scratch_best.pt'
FA_CACHE_TRAIN  = 'logs/fa_cache_teacher_train_20k.pt'
FA_CACHE_VAL    = 'logs/fa_cache_teacher_val_1k.pt'
DATASET_PATH    = 'logs/mixed_encoded_2048_fineweb_tok.pt'
CHECKPOINT_DIR  = 'autoresearch/checkpoints'
CKPT_BASE_NAME  = 'd512_l13_topk_distill'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()

# =============================================================================
# EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    batch_size = max(1, BATCH_SIZE // 2)
    for i in range(0, len(data) - batch_size + 1, batch_size):
        x = data[i:i + batch_size, :-1].to(device, non_blocking=True)
        y = data[i:i + batch_size, 1:].to(device, non_blocking=True)
        with _amp_context(device):
            logits = model(x)
        T, V = logits.size(1), logits.size(2)
        batch_loss = 0.0
        for c in range(0, T, CE_CHUNK):
            lc = logits[:, c:c + CE_CHUNK, :].reshape(-1, V).float()
            yc = y[:, c:c + CE_CHUNK].reshape(-1)
            batch_loss += F.cross_entropy(lc, yc, reduction='sum').item()
        total_loss += batch_loss
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = 'cuda'
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    HD = EMBEDDING_DIM // NUM_HEADS

    print('=' * 70)
    print('  🔬 DWARF D512-L13 TopK Sparse Attention — distillation from FA')
    print(f'  FA@L{FULL_ATTN_LAYER} → TopKSparseAttention(k={TOPK_K})')
    print('=' * 70)
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={HD}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  TopK k={TOPK_K}')
    print(f'  λ annealing: 1.0 → 0.1 (linear)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE * GRAD_ACCUM}')
    print(f'  git={git_hash}')

    # ── Tokenizer ──
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    assert tok_path is not None, 'Tokenizer not found'
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'  Tokenizer: {tok_path}')

    # ── Dataset ──
    assert os.path.isfile(DATASET_PATH), f'Dataset not found: {DATASET_PATH}'
    _raw = torch.load(DATASET_PATH, weights_only=True)
    if isinstance(_raw, dict):
        all_data = _raw.get('train', _raw.get('data'))
    else:
        all_data = _raw
    all_data = all_data.long()
    assert len(all_data) >= MAX_TRAIN_SEQS + MAX_VAL_SEQS, (
        f'Dataset has {len(all_data)} sequences, need {MAX_TRAIN_SEQS + MAX_VAL_SEQS}')

    train_data = all_data[:MAX_TRAIN_SEQS]
    val_data = all_data[MAX_TRAIN_SEQS:MAX_TRAIN_SEQS + MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs')

    # ── Teacher FA cache ──
    assert os.path.isfile(FA_CACHE_TRAIN), f'FA cache not found: {FA_CACHE_TRAIN}'
    assert os.path.isfile(FA_CACHE_VAL), f'FA cache not found: {FA_CACHE_VAL}'
    fa_cache_train = torch.load(FA_CACHE_TRAIN, mmap=True)
    fa_cache_val = torch.load(FA_CACHE_VAL, mmap=True)
    print(f'  FA cache train: {fa_cache_train.shape}, val: {fa_cache_val.shape}')

    # ── Model ──
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

    # ── Replace FA block's attention with TopKSparseAttention ──
    fa_layer = model.full_attn_layer
    model.blocks[fa_layer].attn = TopKSparseAttention(
        D=EMBEDDING_DIM, H=NUM_HEADS, hd=HD, k=TOPK_K,
    ).to(device)
    print(f'  Replaced blocks[{fa_layer}].attn with TopKSparseAttention(k={TOPK_K})')

    # ── Load pretrained weights (FA excluded) ──
    assert os.path.isfile(TEACHER_CKPT), f'Checkpoint not found: {TEACHER_CKPT}'
    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  Loaded pretrained weights from {TEACHER_CKPT}')
    print(f'    missing keys:    {len(missing)} (TopK params — initialized fresh)')
    print(f'    unexpected keys: {len(unexpected)} (old FA params — discarded)')
    for key in missing:
        print(f'      [missing] {key}')
    for key in unexpected:
        print(f'      [unexpected] {key}')

    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {number_of_parameters:,} ({number_of_parameters / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')

    # ── Register hook to capture student FA output ──
    _student_fa_out = {}

    def _fa_hook(module, inp, out):
        _student_fa_out['val'] = out

    hook_handle = model.blocks[fa_layer].attn.register_forward_hook(_fa_hook)
    print(f'  Hook registered on blocks[{fa_layer}].attn (TopKSparseAttention)')

    # ── Optimizer ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    scale_embed_ids = {id(p) for p in model.scale_embed_parameters()}
    scale_embed_params = [p for p in model.parameters() if id(p) in scale_embed_ids]
    other_params = [p for p in model.parameters() if id(p) not in scale_embed_ids]
    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': other_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)

    def _lr_lambda(step, group_idx):
        if group_idx == 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda s: _lr_lambda(s, 0), lambda s: _lr_lambda(s, 1)])

    print(f'  total_steps={total_steps}, LR={LR}')

    # ── Training loop ──
    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}
    global_step = 0
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        step_times = deque(maxlen=20)

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                seq_indices = indices[idx_start:idx_start + BATCH_SIZE]
                batch = train_data[seq_indices]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)

                with _amp_context(device):
                    logits = model(x)

                # ── Chunked CE (LM loss) ──
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                T = logits_flat.size(0)
                grad_logits = torch.empty_like(logits_flat)
                total_ce = 0.0
                for chunk_start in range(0, T, CE_CHUNK):
                    chunk_end = min(chunk_start + CE_CHUNK, T)
                    chunk = logits_flat[chunk_start:chunk_end].detach().requires_grad_(True)
                    chunk_loss = F.cross_entropy(
                        chunk, y_flat[chunk_start:chunk_end], reduction='sum')
                    chunk_loss.backward()
                    grad_logits[chunk_start:chunk_end] = chunk.grad
                    total_ce += chunk_loss.item()
                lm_loss_val = total_ce / T

                # ── Distillation loss (MSE) ──
                teacher_target = fa_cache_train[seq_indices].to(
                    device, dtype=torch.bfloat16)
                student_output = _student_fa_out['val']
                distill_loss = F.mse_loss(student_output, teacher_target)

                # ── Lambda annealing: 1.0 → 0.1 ──
                frac = global_step / max(total_steps, 1)
                lambda_d = 1.0 * (1.0 - frac) + 0.1 * frac

                # ── Combined backward ──
                # CE gradient through logits
                logits_flat.backward(grad_logits / (T * GRAD_ACCUM), retain_graph=True)
                # Distillation gradient through FA hook
                (lambda_d * distill_loss / GRAD_ACCUM).backward()

                distill_loss_val = distill_loss.item()
                del logits, logits_flat, y_flat, grad_logits, student_output, teacher_target

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            global_step += 1

            step_ms = (time.time() - t0) * 1000
            step_times.append(step_ms)

            if step % 50 == 0:
                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000.0)
                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| LM {lm_loss_val:.4f} | Distill {distill_loss_val:.6f} '
                      f'| λ={lambda_d:.3f} | {tok_s:.0f} tok/s', flush=True)

        # ── Validation ──
        val_loss = evaluate(model, val_data, device)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        # ── Distillation loss on val set ──
        model.eval()
        val_distill_total, val_distill_count = 0.0, 0
        val_batch_size = max(1, BATCH_SIZE // 2)
        with torch.inference_mode():
            for i in range(0, len(val_data) - val_batch_size + 1, val_batch_size):
                x = val_data[i:i + val_batch_size, :-1].to(device, non_blocking=True)
                with _amp_context(device):
                    model(x)
                teacher_val = fa_cache_val[i:i + val_batch_size].to(
                    device, dtype=torch.bfloat16)
                val_distill_total += F.mse_loss(
                    _student_fa_out['val'], teacher_val).item()
                val_distill_count += 1

        val_distill = val_distill_total / max(val_distill_count, 1)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': clean_state,
                'config': {
                    'topk_k': TOPK_K,
                    'fa_layer': fa_layer,
                    'embedding_dim': EMBEDDING_DIM,
                    'num_heads': NUM_HEADS,
                    'ffn_dim': FFN_DIM,
                    'num_layers': NUM_LAYERS,
                },
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt'))
            marker = ' *'

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f} '
              f'| Val Distill MSE {val_distill:.6f}{marker}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules() if isinstance(m, DSQGAttentionGrouped)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        resume_state = {
            'model_state_dict': {k.replace('._orig_mod', ''): v
                                 for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
        }
        torch.save(resume_state,
                   os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_ep{epoch}_resume.pt'))

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    # ── Summary ──
    hook_handle.remove()
    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final = ppl_results.get(SCREEN_EPOCHS, 999.0)

    print('\n' + '=' * 70)
    print('  🔬 TopK Distillation Results')
    print('=' * 70)
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {number_of_parameters / 1e6:.1f}')
    print(f'topk_k: {TOPK_K}')
    print(f'description: TopK distillation D={EMBEDDING_DIM} L={NUM_LAYERS} '
          f'k={TOPK_K} FA@L{FULL_ATTN_LAYER} λ=1.0→0.1 '
          f'20K seqs, 3 epochs')

    if passkey_final >= 70:
        print('\n  ✅ MECHANISM VALIDATED — passkey ≥ 70%')
    elif passkey_final >= 30:
        print('\n  ⚠️ PARTIAL — consider k=128 or longer training')
    else:
        print('\n  ❌ BELOW THRESHOLD — FA may be irreplaceable at this scale')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
