#!/usr/bin/env python3
"""
DWARF D512-L13 TopK Sparse Attention Distillation - Dual GPU
Teacher (FA) on 3090 (cuda:1), Student (TopK) on 4090 (cuda:0)
"""

import os
import sys
import time
import math
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Import the triadic AABBC architecture
sys.path.insert(0, str(Path(__file__).parent))
from train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16 import TriadicJ96
from tokenizers import Tokenizer

# Constants from the base training script
SCALE_EMBED_INIT = 0.15
LR_MULT = 20.0
EMA_INIT = 0.020833

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
D               = 512
H               = 8
NUM_LAYERS      = 13
FFN_MULT        = 2
VOCAB_SIZE      = 32768  # Model provisioned for 32768, tokenizer uses 32000
MAX_SEQ_LEN     = 2048
TOPK_K          = 64

TEACHER_CKPT    = "autoresearch/checkpoints/d512_l13_triadic_aabbc_mixed_scratch_best.pt"
TEACHER_DEVICE  = "cuda:0"  # 4090 (same as student - avoid Triton cross-device issues)
STUDENT_DEVICE  = "cuda:0"  # 4090

BATCH_SIZE      = 16        # Reduced for single-GPU (teacher + student)
GRAD_ACCUM      = 4
EPOCHS          = 3
LR              = 3e-4
WEIGHT_DECAY    = 0.1
WARMUP_STEPS    = 100
MAX_TRAIN_SEQS  = 20_000
MAX_VAL_SEQS    = 1_000

# Distillation loss weighting (anneals from 1.0 to 0.1)
LAMBDA_START    = 1.0
LAMBDA_END      = 0.1

CKPT_DIR        = Path("autoresearch/checkpoints")
LOG_DIR         = Path("logs")
TOKENIZER_PATH  = "results/fineweb_tokenizer_32k.json"
DATA_TRAIN      = LOG_DIR / "fineweb_edu_encoded_2048_v2.pt"
DATA_VAL        = LOG_DIR / "fineweb_edu_encoded_2048_val.pt"

# ═══════════════════════════════════════════════════════════════════
#  TopK Sparse Attention Layer
# ═══════════════════════════════════════════════════════════════════
class TopKSparseAttention(nn.Module):
    """
    TopK sparse attention: compute full O(N²) scores, keep top-k per query,
    apply causal mask, softmax over sparse set.
    """
    def __init__(self, d_model, num_heads, k, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.k = k
        self.dropout = dropout
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.register_buffer("causal_mask", 
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool)))
    
    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim
        
        q = self.W_q(x).view(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.W_k(x).view(B, T, H, hd).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, hd).transpose(1, 2)
        
        # Full attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, T)
        
        # Apply causal mask (set future positions to -inf)
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Top-k selection per query
        topk_vals, topk_idx = torch.topk(scores, min(self.k, T), dim=-1, largest=True)
        
        # Create sparse mask: only keep top-k positions
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_idx, True)
        
        # Zero out non-topk scores
        scores_sparse = scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Softmax over sparse set
        attn = F.softmax(scores_sparse, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Weighted sum
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        
        return out


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                     cwd=Path(__file__).parent.parent).decode().strip()
    except:
        return "unknown"


def chunked_cross_entropy(logits, targets, chunk_size=4096):
    """Memory-efficient cross-entropy with vocabulary chunking."""
    B, T, V = logits.shape
    logits_2d = logits.reshape(-1, V)
    targets_1d = targets.reshape(-1)
    
    losses = []
    for i in range(0, V, chunk_size):
        chunk_logits = logits_2d[:, i:i+chunk_size]
        chunk_targets = targets_1d.clone().long()
        
        # Mask targets outside this chunk
        mask = (chunk_targets >= i) & (chunk_targets < i+chunk_size)
        chunk_targets[~mask] = 0  # Dummy value
        chunk_targets = chunk_targets - i
        
        chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, reduction='none')
        chunk_loss = chunk_loss * mask.float()
        losses.append(chunk_loss)
    
    total_loss = torch.stack(losses).sum(dim=0)
    return total_loss.sum() / (targets_1d != -100).sum()


def load_data(path, max_seqs, split='train'):
    data = torch.load(path, map_location='cpu')
    # Handle both dict and tensor formats
    if isinstance(data, dict):
        data = data[split]
    if data.shape[0] > max_seqs:
        data = data[:max_seqs]
    return data


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  🔬 DWARF D512-L13 TopK Sparse Attention — Dual GPU Distillation")
    print("  Teacher (FA) on 3090 (cuda:1), Student (TopK) on 4090 (cuda:0)")
    print("=" * 70)
    
    # Device info
    print(f"  Teacher GPU: {torch.cuda.get_device_name(1)}")
    print(f"  Student GPU: {torch.cuda.get_device_name(0)}")
    print(f"  D={D}, H={H}, hd={D//H}, L={NUM_LAYERS}, FFN={D*FFN_MULT}")
    print(f"  TopK k={TOPK_K}")
    print(f"  λ annealing: {LAMBDA_START} → {LAMBDA_END} (linear)")
    print(f"  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={EPOCHS}")
    print(f"  Batch: BS={BATCH_SIZE} × GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  git={get_git_hash()}")
    print(f"  Tokenizer: {TOKENIZER_PATH}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_vocab_size = tokenizer.get_vocab_size()
    print(f"  Tokenizer vocab: {tok_vocab_size}, Model vocab: {VOCAB_SIZE}")
    assert tok_vocab_size <= VOCAB_SIZE, f"Tokenizer size {tok_vocab_size} exceeds model {VOCAB_SIZE}"
    
    # Load data
    train_data = load_data(DATA_TRAIN, MAX_TRAIN_SEQS, split='train')
    val_data = load_data(DATA_TRAIN, MAX_VAL_SEQS, split='val')  # Val split is in the same file
    print(f"  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs")
    
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ──────────────────────────────────────────────────────────────
    #  Load Teacher Model (3090, eval mode, no gradients)
    # ──────────────────────────────────────────────────────────────
    # Determine FA layer from triadic layout (should be layer 3 for D512-L13)
    FA_LAYER = 3
    
    teacher = TriadicJ96(
        vocab_size=VOCAB_SIZE,
        embedding_dim=D,
        num_heads=H,
        ffn_dim=D * FFN_MULT,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FA_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT,
        dropout=0.1
    ).to(TEACHER_DEVICE)
    
    ckpt = torch.load(TEACHER_CKPT, map_location=TEACHER_DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    teacher.load_state_dict(state)
    teacher.eval()
    
    # Disable gradients for teacher and ensure all buffers are on the right device
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Explicitly move all buffers to the teacher device (Triton kernels need this)
    for name, buffer in teacher.named_buffers():
        if buffer.device != torch.device(TEACHER_DEVICE):
            buffer.data = buffer.data.to(TEACHER_DEVICE)
    
    # FA layer is always at the predefined index for this architecture
    fa_layer_idx = FA_LAYER
    print(f"  Teacher FA layer: blocks[{fa_layer_idx}]")
    # Verify it's actually an FA layer
    assert hasattr(teacher.blocks[fa_layer_idx].attn, 'qkv_proj'), \
        f"blocks[{fa_layer_idx}] is not an FA layer (missing qkv_proj)"
    
    # ──────────────────────────────────────────────────────────────
    #  Build Student Model (4090, TopK replaces FA)
    # ──────────────────────────────────────────────────────────────
    student = TriadicJ96(
        vocab_size=VOCAB_SIZE,
        embedding_dim=D,
        num_heads=H,
        ffn_dim=D * FFN_MULT,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FA_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT,
        dropout=0.1
    ).to(STUDENT_DEVICE)
    
    # Replace FA with TopK
    student.blocks[fa_layer_idx].attn = TopKSparseAttention(
        d_model=D, num_heads=H, k=TOPK_K, dropout=0.0
    ).to(STUDENT_DEVICE)
    print(f"  Replaced blocks[{fa_layer_idx}].attn with TopKSparseAttention(k={TOPK_K})")
    
    # Load pretrained weights (will skip TopK params)
    result = student.load_state_dict(state, strict=False)
    print(f"  Loaded pretrained weights from {TEACHER_CKPT}")
    print(f"    missing keys:    {len(result.missing_keys)} (TopK params — initialized fresh)")
    print(f"    unexpected keys: {len(result.unexpected_keys)} (old FA params — discarded)")
    
    student.train()
    num_params = sum(p.numel() for p in student.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # ──────────────────────────────────────────────────────────────
    #  Hook to capture student FA output
    # ──────────────────────────────────────────────────────────────
    student_fa_output = {}
    def hook_fn(module, input, output):
        student_fa_output['activations'] = output.detach()
    
    hook = student.blocks[fa_layer_idx].attn.register_forward_hook(hook_fn)
    print(f"  Hook registered on student blocks[{fa_layer_idx}].attn (TopKSparseAttention)")
    
    # ──────────────────────────────────────────────────────────────
    #  Optimizer & Scheduler
    # ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    print(f"  total_steps={total_steps}, LR={LR}\n")
    
    # ──────────────────────────────────────────────────────────────
    #  Training Loop
    # ──────────────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        student.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_distill_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (batch,) in enumerate(train_loader):
            # Lambda annealing
            progress = global_step / total_steps
            lambda_distill = LAMBDA_START + (LAMBDA_END - LAMBDA_START) * progress
            
            # Move to devices
            batch_teacher = batch.to(TEACHER_DEVICE)
            batch_student = batch.to(STUDENT_DEVICE)
            
            inputs = batch_student[:, :-1]
            targets = batch_student[:, 1:]
            
            # ────────────────────────────────────────────────────
            #  Teacher forward (no grad)
            # ────────────────────────────────────────────────────
            with torch.no_grad():
                _ = teacher(batch_teacher[:, :-1])
                # Extract FA activations from teacher
                teacher_fa_acts = None
                def teacher_hook(module, input, output):
                    nonlocal teacher_fa_acts
                    teacher_fa_acts = output.detach()
                
                temp_hook = teacher.blocks[fa_layer_idx].attn.register_forward_hook(teacher_hook)
                _ = teacher(batch_teacher[:, :-1])
                temp_hook.remove()
                
                # Transfer to student device
                teacher_fa_acts = teacher_fa_acts.to(STUDENT_DEVICE)
            
            # ────────────────────────────────────────────────────
            #  Student forward
            # ────────────────────────────────────────────────────
            logits = student(inputs)
            student_fa_acts = student_fa_output['activations']
            
            # ────────────────────────────────────────────────────
            #  Combined loss
            # ────────────────────────────────────────────────────
            ce_loss = chunked_cross_entropy(logits, targets)
            distill_loss = F.mse_loss(student_fa_acts, teacher_fa_acts)
            
            loss = ce_loss + lambda_distill * distill_loss
            loss = loss / GRAD_ACCUM
            
            loss.backward()
            
            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_ce_loss += ce_loss.item()
            epoch_distill_loss += distill_loss.item()
            
            # ────────────────────────────────────────────────────
            #  Gradient accumulation step
            # ────────────────────────────────────────────────────
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 50 == 0:
                    print(f"  [ep{epoch} step{global_step:4d}] "
                          f"loss={loss.item()*GRAD_ACCUM:.4f} "
                          f"ce={ce_loss.item():.4f} "
                          f"distill={distill_loss.item():.4f} "
                          f"λ={lambda_distill:.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = epoch_loss / len(train_loader)
        avg_ce = epoch_ce_loss / len(train_loader)
        avg_distill = epoch_distill_loss / len(train_loader)
        
        # ────────────────────────────────────────────────────────
        #  Validation
        # ────────────────────────────────────────────────────────
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch_student = batch.to(STUDENT_DEVICE)
                inputs = batch_student[:, :-1]
                targets = batch_student[:, 1:]
                logits = student(inputs)
                val_loss += chunked_cross_entropy(logits, targets).item()
        
        val_loss /= len(val_loader)
        val_ppl = math.exp(min(val_loss, 10))
        
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch}/{EPOCHS}")
        print(f"  Train: loss={avg_loss:.4f}  ce={avg_ce:.4f}  distill={avg_distill:.4f}")
        print(f"  Val:   loss={val_loss:.4f}  ppl={val_ppl:.2f}")
        print(f"{'='*70}\n")
        
        # ────────────────────────────────────────────────────────
        #  Checkpointing
        # ────────────────────────────────────────────────────────
        ckpt_path = CKPT_DIR / f"d512_l13_topk_distill_dual_gpu_ep{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CKPT_DIR / "d512_l13_topk_distill_dual_gpu_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, best_path)
            print(f"  ✓ New best: {best_path}\n")
    
    hook.remove()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
