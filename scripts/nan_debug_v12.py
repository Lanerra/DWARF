"""
NaN Diagnostic Script for train_d768_l16_mixed_scratch_v12_4090_bf16.py

This script replicates the training setup with extensive NaN detection to
identify exactly where NaN first appears in the training pipeline.

Run from repo root:
  .venv/bin/python3 scripts/nan_debug_v12.py 2>&1 | tee logs/nan_debug_v12.log
"""

import contextlib
import math
import os
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM = 768
NUM_HEADS = 12
FFN_DIM = 1536
NUM_LAYERS = 16
FULL_ATTN_LAYER = 4
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT = 20.0
EMA_INIT = 0.010
EMA_FLOOR = 0.00001
LR = 2.5e-4

BATCH_SIZE = 32
GRAD_ACCUM = 4
MAX_STEPS = 250
CHECKPOINT_STRATEGY = os.environ.get('DWARF_CKPT', 'every_other')

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
    print("AdamW8bit: enabled (bitsandbytes)")
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available")

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
    print("Liger fused CE: enabled")
except ImportError:
    _LIGER_AVAILABLE = False
    print("WARNING: liger_kernel not installed")

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v12 import DSQGAttentionV12 as DSQGAttentionKernel
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

print(f"Kernel: V12")


def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)


def _agc_normalize(pool, eps=1e-6):
    D = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


_VERBOSE_LAYER_DEBUG = int(os.environ.get('VERBOSE_LAYER', '-1'))

class DSQGBlockV6Physics(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False, layer_idx=0):
        super().__init__()
        self.interference = interference
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = DSQGAttentionKernel(embedding_dim, num_heads,
                                        seq_len=seq_len, dropout=dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))

    def _debug_tensor(self, name, t):
        if self.layer_idx == _VERBOSE_LAYER_DEBUG:
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            tmin = t.min().item() if not has_nan else float('nan')
            tmax = t.max().item() if not has_nan else float('nan')
            print(f"      L{self.layer_idx} {name}: NaN={has_nan} Inf={has_inf} min={tmin:.4e} max={tmax:.4e}")

    def forward(self, x):
        self._debug_tensor("input", x)

        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR)
            self._debug_tensor("pool_after_ema", pool)
            pool = _agc_normalize(pool)
            self._debug_tensor("pool_after_agc", pool)
            inter = torch.sigmoid(self.inter_gate(xi)) * pool
            self._debug_tensor("inter", inter)
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            self._debug_tensor("k_delta", k_delta)
            self._debug_tensor("v_delta", v_delta)
            kv_inject = (k_delta, v_delta)

        norm1_out = self.norm1(x)
        self._debug_tensor("norm1_out", norm1_out)

        attn_out = self.attn(norm1_out, kv_inject=kv_inject)
        self._debug_tensor("attn_out", attn_out)

        x = x + attn_out
        self._debug_tensor("after_residual1", x)

        norm2_out = self.norm2(x)
        self._debug_tensor("norm2_out", norm2_out)

        ffn_out = self.ffn(norm2_out)
        self._debug_tensor("ffn_out", ffn_out)

        x = x + ffn_out
        self._debug_tensor("output", x)

        return x


class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(
            self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
            p=self.dropout_p, training=self.training)


class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AutoresearchTransformerPhysicsDebug(nn.Module):
    """Debug version with NaN checks after each layer."""

    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer,
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout, layer_idx=i))
            else:
                has_if = (i == full_attn_layer - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if, layer_idx=i))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights(scale_embed_init_val)

        self.nan_check_enabled = False
        self.first_nan_layer = None
        self.first_nan_step = None

    def _init_weights(self, scale_embed_init_val):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)
        for m in self.modules():
            if isinstance(m, DSQGAttentionKernel):
                nn.init.normal_(m.phase_base, 0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes, 0.0, 0.01)
                nn.init.normal_(m.phase_gain, 0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        if block_idx == self.full_attn_layer:
            return True
        if block_idx == self.full_attn_layer - 1:
            return True
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.full_attn_layer
        return False

    def _check_nan(self, x, step, layer_idx, location):
        if not self.nan_check_enabled:
            return
        if torch.isnan(x).any() or torch.isinf(x).any():
            if self.first_nan_layer is None:
                self.first_nan_layer = layer_idx
                self.first_nan_step = step
                nan_count = torch.isnan(x).sum().item()
                inf_count = torch.isinf(x).sum().item()
                print(f"\n  !!! NaN/Inf DETECTED at Step {step}, Layer {layer_idx}, {location} !!!")
                print(f"      NaN count: {nan_count}, Inf count: {inf_count}")
                print(f"      Shape: {x.shape}, dtype: {x.dtype}")
                print(f"      Stats: min={x[~torch.isnan(x)].min().item() if (~torch.isnan(x)).any() else 'all NaN':.6f}, "
                      f"max={x[~torch.isnan(x)].max().item() if (~torch.isnan(x)).any() else 'all NaN':.6f}")

    def forward_hidden_debug(self, idx, step):
        B, N = idx.shape
        x = self.drop(self.embedding(idx))
        self._check_nan(x, step, -1, "after_embedding")

        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
            self._check_nan(x, step, i, f"after_block_{i}")

        out = self.norm(x)
        self._check_nan(out, step, NUM_LAYERS, "after_final_norm")
        return out

    def forward_hidden(self, idx):
        return self.forward_hidden_debug(idx, -1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionKernel):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p


def check_gradients_for_nan(model, step):
    """Check all parameter gradients for NaN/Inf."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                print(f"\n  !!! GRADIENT NaN/Inf at Step {step}, Param: {name} !!!")
                print(f"      NaN count: {nan_count}, Inf count: {inf_count}")
                print(f"      Grad shape: {param.grad.shape}, dtype: {param.grad.dtype}")
                grad_finite = param.grad[~torch.isnan(param.grad) & ~torch.isinf(param.grad)]
                if grad_finite.numel() > 0:
                    print(f"      Finite grad stats: min={grad_finite.min().item():.6e}, "
                          f"max={grad_finite.max().item():.6e}, "
                          f"abs_max={grad_finite.abs().max().item():.6e}")
                return name
    return None


def check_params_for_nan(model, step):
    """Check all parameters for NaN/Inf after optimizer step."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            print(f"\n  !!! PARAMETER NaN/Inf at Step {step}, Param: {name} !!!")
            print(f"      NaN count: {nan_count}, Inf count: {inf_count}")
            print(f"      Shape: {param.shape}, dtype: {param.dtype}")
            return name
    return None


def run_diagnostic():
    print("=" * 70)
    print("  NaN DIAGNOSTIC — DWARF D768-L16 V12")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM}, MAX_STEPS={MAX_STEPS}")
    print(f"LR={LR}, SCALE_EMBED_LR_MULT={SCALE_EMBED_LR_MULT}")
    print(f"AdamW8bit: {'enabled' if _BNB_AVAILABLE else 'DISABLED'}")
    print(f"Liger CE: {'enabled' if _LIGER_AVAILABLE else 'DISABLED'}")

    _encoded_cache = 'logs/mixed_encoded_2048_fineweb_tok.pt'
    if not os.path.exists(_encoded_cache):
        raise FileNotFoundError(f'Dataset not found: {_encoded_cache}')
    print(f'Loading dataset from {_encoded_cache}')
    _cache = torch.load(_encoded_cache, weights_only=True)
    train_data = _cache['train'].long()
    print(f'  train: {len(train_data):,} seqs')

    model = AutoresearchTransformerPhysicsDebug(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)
    model.nan_check_enabled = True

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())

    print(f"\nscale_embed param count: {sum(p.numel() for p in scale_embed_params)}")
    print(f"  (should be {24 * 64 * NUM_LAYERS} = {24*64*(NUM_LAYERS-1)} for non-FA layers)")

    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    if _LIGER_AVAILABLE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    print("\nRunning kernel warmup (matching production setup)...")
    _wb = min(BATCH_SIZE, len(train_data))
    _wx = train_data[:_wb, :-1].to(device)
    _wy = train_data[:_wb, 1:].to(device)

    print("  Testing forward-only (no backward) multiple times...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            for _i in range(3):
                _test_h = model.forward_hidden(_wx)
                has_nan = torch.isnan(_test_h).any().item()
                print(f"    Forward #{_i+1}: NaN={has_nan}, min={_test_h.min().item():.4f}, max={_test_h.max().item():.4f}")
                del _test_h

    print("  Now running forward+backward warmup...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if _LIGER_AVAILABLE:
            _whidden = model.forward_hidden(_wx)
            print(f"  Warmup hidden shape: {_whidden.shape}, dtype: {_whidden.dtype}")
            print(f"  Warmup hidden stats: min={_whidden.min().item():.4f}, max={_whidden.max().item():.4f}")
            print(f"  out.weight shape: {model.out.weight.shape}, dtype: {model.out.weight.dtype}")
            _wloss = liger_ce_fn(
                _whidden.view(-1, _whidden.size(-1)),
                model.out.weight,
                _wy.view(-1))
            print(f"  Warmup loss: {_wloss.item():.4f}")
            _wloss.backward()
            del _whidden, _wloss
        else:
            _wout = model.out(model.forward_hidden(_wx))
            _wloss = F.cross_entropy(_wout.view(-1, VOCAB_SIZE), _wy.view(-1))
            _wloss.backward()
            del _wout, _wloss
    print("  Forward+backward complete, now testing forward after backward...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            _test_h = model.forward_hidden(_wx)
            has_nan = torch.isnan(_test_h).any().item()
            print(f"    Forward after backward (before zero_grad): NaN={has_nan}")
            del _test_h

    optimizer.zero_grad(set_to_none=True)
    print("  Called optimizer.zero_grad(set_to_none=True)")

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            _test_h = model.forward_hidden(_wx)
            has_nan = torch.isnan(_test_h).any().item()
            print(f"    Forward after zero_grad: NaN={has_nan}")
            del _test_h

    del _wx, _wy
    torch.cuda.synchronize()
    print("  Kernel warmup complete.")

    print("\nRunning sanity check: same batch after zero_grad...")
    _wx2 = train_data[:_wb, :-1].to(device)
    _wy2 = train_data[:_wb, 1:].to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _whidden2 = model.forward_hidden(_wx2)
        print(f"  Post-zero_grad hidden shape: {_whidden2.shape}, dtype: {_whidden2.dtype}")
        has_nan = torch.isnan(_whidden2).any().item()
        has_inf = torch.isinf(_whidden2).any().item()
        print(f"  Post-zero_grad hidden NaN: {has_nan}, Inf: {has_inf}")
        if not has_nan and not has_inf:
            print(f"  Post-zero_grad hidden stats: min={_whidden2.min().item():.4f}, max={_whidden2.max().item():.4f}")
    del _wx2, _wy2, _whidden2

    print("\nChecking input data validity...")
    indices = torch.randperm(len(train_data))
    batch0 = train_data[indices[:BATCH_SIZE]]
    print(f"  Batch shape: {batch0.shape}")
    print(f"  Token ID range: [{batch0.min().item()}, {batch0.max().item()}]")
    print(f"  Expected vocab range: [0, {VOCAB_SIZE-1}]")
    if batch0.max().item() >= VOCAB_SIZE:
        print("  !!! WARNING: Token IDs exceed vocab size !!!")
    if batch0.min().item() < 0:
        print("  !!! WARNING: Negative token IDs !!!")

    print("\n" + "=" * 70)
    print("  STARTING DIAGNOSTIC RUN")
    print("=" * 70)

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    first_nan_step = None
    first_nan_location = None

    for acc_step in range(MAX_STEPS // GRAD_ACCUM + 1):
        for ga in range(GRAD_ACCUM):
            idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
            if idx_start >= len(train_data):
                continue

            batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)

            if _LIGER_AVAILABLE:
                with _amp_context(device):
                    hidden = model.forward_hidden_debug(x, step)

                    if torch.isnan(hidden).any():
                        first_nan_location = "forward_hidden"
                        first_nan_step = step
                        break

                    loss = liger_ce_fn(
                        hidden.view(-1, hidden.size(-1)),
                        model.out.weight,
                        y.view(-1))

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n  !!! LOSS NaN/Inf at Step {step} !!!")
                        print(f"      Loss value: {loss.item()}")
                        first_nan_location = "loss"
                        first_nan_step = step
                        break

                (loss / GRAD_ACCUM).backward()
                loss_val = loss.item()
                del hidden, loss
            else:
                with _amp_context(device):
                    hidden = model.forward_hidden_debug(x, step)
                    logits = model.out(hidden)
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE),
                    y.view(-1))
                (loss / GRAD_ACCUM).backward()
                loss_val = loss.item()
                del hidden, logits, loss

            nan_grad_param = check_gradients_for_nan(model, step)
            if nan_grad_param:
                first_nan_location = f"gradient:{nan_grad_param}"
                first_nan_step = step
                break

        if first_nan_step is not None:
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        nan_param = check_params_for_nan(model, step)
        if nan_param:
            first_nan_location = f"param_after_step:{nan_param}"
            first_nan_step = step
            break

        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 10 == 0:
            se_vals = []
            for m in model.modules():
                if isinstance(m, DSQGAttentionKernel):
                    se_vals.append(m.scale_embed.detach().abs())
            se_max = torch.cat(se_vals).max().item() if se_vals else 0.0
            print(f"  Step {step}/{MAX_STEPS} | Loss {loss_val:.4f} | SE |max|={se_max:.4f}")

        if step >= MAX_STEPS:
            break

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)

    if first_nan_step is not None:
        print(f"\n  FIRST NaN DETECTED:")
        print(f"    Step: {first_nan_step}")
        print(f"    Location: {first_nan_location}")
        if model.first_nan_layer is not None:
            print(f"    Layer: {model.first_nan_layer}")
    else:
        print(f"\n  NO NaN DETECTED in {MAX_STEPS} steps")
        print("  The issue may be intermittent or require more steps")

    se_vals = []
    for m in model.modules():
        if isinstance(m, DSQGAttentionKernel):
            se_vals.append(m.scale_embed.detach())
    if se_vals:
        se_all = torch.cat(se_vals)
        print(f"\n  Final scale_embed stats:")
        print(f"    |mean|={se_all.abs().mean().item():.4f}")
        print(f"    |max|={se_all.abs().max().item():.4f}")
        print(f"    NaN count: {torch.isnan(se_all).sum().item()}")


if __name__ == '__main__':
    run_diagnostic()
