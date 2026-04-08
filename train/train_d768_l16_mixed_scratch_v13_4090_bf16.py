"""
🧪 DWARF D=768 L=16 — FROM-SCRATCH MIXED-DOMAIN PRETRAINING (4090 / V12)

Architecture: D=768, H=12 (hd=64), L=16, FFN=1536, J=24 (se015 offsets), TIED lm_head
  L0-L2:  DSQGBlockV6Physics  IF=False  ← 3 pre-FA warm-up relay layers
  L3:     DSQGBlockV6Physics  IF=True   ← preIF@L3
  L4:     FullAttentionBlock            ← FA@L4 (25% depth, 11 post-FA relay layers)
  L5-15:  DSQGBlockV6Physics  IF=False  ← 11 post-FA relay layers

EXPERIMENT: Does D=768/L=16 form relay natively on mixed-domain corpus?

CONTEXT:
  - D=512/L=8 failed (global_loss_std=0.798, topology capacity floor confirmed)
  - D=768/L=32 H3 succeeded (passkey 70% ep2, percolation step ~1000)
  - This tests minimum viable relay depth at D=768 under mixed gradients
  - D=768 coherence length ≈ 22 layers (D/35), so 11 post-FA may be borderline

HYPOTHESES:
  H-pass: 11 post-FA layers sufficient for topology crystallisation at D=768
           → expect passkey ep2 ≥ 50% (vs 70% for L=32)
  H-fail: Topology crystallisation stalls — gradient noise + depth floor identical
           to D=512 failure mode, just at a higher dimension

KEY DISCRIMINATOR vs D=512:
  D=512: global_loss_std=0.798 (8× higher than FineWeb), SE|max| grew fast to 2.54
         with NO consolidation plateau → topology formation failed despite amplitude
  D=768: Watch for Phase B consolidation plateau (SE|max| ~2.0-2.1 for 200-400 steps)
         Plateau present → topology crystallising; absent → same failure as D=512

HYPERPARAMETERS (corrected for mixed-domain):
  - LR_MULT = 20.0  (base 18.37 × 1.087 mixed-domain factor; validated d768_l32 H3)
  - EMA_INIT = 0.010 (vs 0.0208 FineWeb; model learns 690-822t window on mixed)
  - LR = 2.5e-4 (vs 3e-4 FineWeb; smooths 2.6× higher loss variance)

KERNEL: V12 (gather-before-attend, 8.39× speedup, 16.2% MFU on 4090)
  - Auto-detects sm_89 (4090) → streaming gather path, no materialise
  - Benchmark: 5.43 ms/step vs V8's 45.53 ms/step (B=8, N=2047)

Dataset: fineweb_wiki_80_20_encoded_2048.pt  (FineWeb-Edu 80% + Wikipedia 20%)
  - 234,418 train / 5,582 val sequences
  - 80% FineWeb-Edu / 20% Wikipedia
  - Encoded with fineweb_tokenizer_32k.json (32K BPE)

Chinchilla budget:
  ~111M params × 20 = 2.22B Chinchilla-optimal tokens
  22% Chinchilla: ~488M tokens = 238K seqs @ 2048 tokens
  Using all 234,418 available ≈ 22% Chinchilla

Config:
  BS=16, GA=8 → eff_batch=128 (4090 VRAM-safe with V12 streaming path)
  Gradient checkpointing: every_other (required for L=16 on 4090)

Run from repo root:
  tmux new-session -d -s d768_l16_mixed_v13 \\
    ".venv/bin/python3 -u train/train_d768_l16_mixed_scratch_v13_4090_bf16.py \\
     2>&1 | tee logs/run_d768_l16_mixed_v13.log"
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 768
NUM_HEADS        = 12         # hd = 768/12 = 64 ✓
FFN_DIM          = 1536       # 2×D — confirmed optimal
NUM_LAYERS       = 16
FULL_ATTN_LAYER  = 4          # 25% of 16 = L4; 11 post-FA relay layers; preIF@L3

MAX_TRAIN_SEQS       = 452_000   # all available ≈ 41% Chinchilla/epoch (~124% over 3 epochs) for ~111M params
SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 20.0     # mixed-domain correction: 18.37 × 1.087

# Corrected EMA for mixed-domain: model learns 690-822t window; init closer to target
EMA_INIT  = 0.010               # vs 0.0208 FineWeb-only; window≈100t init → learns ~700t
EMA_FLOOR = 0.00001

LR            = 2.5e-4          # vs 3e-4 FineWeb; smooths 2.6× higher loss variance
SCREEN_EPOCHS = 3

PASSKEY_ABORT_THRESHOLD = 0.20  # 20% — strictly above random (10%) baseline

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
from collections import deque

MAX_TRAIN_SEQS = int(os.environ.get('MAX_TRAIN_SEQS_OVERRIDE', MAX_TRAIN_SEQS))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ── bitsandbytes (AdamW8bit) ───────────────────────────────────────────────────
try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
    print("AdamW8bit: enabled (bitsandbytes)")
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available — falling back to standard AdamW")

# ── Liger fused CE — required, not optional ───────────────────────────────────
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
    print("Liger fused CE: enabled")
except ImportError:
    _LIGER_AVAILABLE = False
    print("WARNING: liger_kernel not installed — install with: pip install liger-kernel")
    print("  Falling back to chunked CE (slower)")

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"


def get_gpu_peak_flops(device="cuda"):
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    if "H100" in name:   return 989e12
    elif "H200" in name: return 1979e12
    elif "4090" in name: return 330e12
    elif "3090" in name: return 142e12
    elif "A100" in name: return 312e12
    return None

VOCAB_SIZE     = 32000
BATCH_SIZE     = 32
GRAD_ACCUM     = 4           # eff_batch=128
CE_CHUNK       = 512
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

ENABLE_TORCH_COMPILE = os.getenv('DWARF_ENABLE_COMPILE', '0') == '1'
CHECKPOINT_STRATEGY  = os.getenv('DWARF_CKPT', 'every_other').lower()
PASSKEY_BATCH_SIZE   = int(os.getenv('DWARF_PASSKEY_BATCH', '16'))

# ── Kernel import — V12 default, V8 4090 fallback ────────────────────────────
import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

_USE_V13 = os.getenv('DWARF_USE_V13', '1') == '1'   # V13 on by default for this script
_USE_V12 = os.getenv('DWARF_USE_V12', '0') == '1'
_USE_V11 = os.getenv('DWARF_USE_V11', '0') == '1'

if _USE_V13:
    from kernels.dsqg_attention_v13 import DSQGAttentionV13 as DSQGAttentionKernel
    _kernel_label = 'V13'
    print("Kernel: V13 (two-pass + window clustering)")
elif _USE_V12:
    try:
        from dsqg_attention_v12 import DSQGAttentionV12 as DSQGAttentionKernel
        print("Kernel: V12 (gather-before-attend, autotuned)")
        _kernel_label = 'V12'
    except ImportError as e:
        print(f"WARNING: V12 not available ({e}), falling back to V8 4090")
        from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionKernel
        _kernel_label = 'V8-4090-fallback'
elif _USE_V11:
    from dsqg_attention_v11 import DSQGAttentionV11 as DSQGAttentionKernel
    _kernel_label = 'V11'
    print("Kernel: V11 (FlexAttention)")
else:
    from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionKernel
    _kernel_label = 'V8-4090'
    print("Kernel: V8 4090 (set DWARF_USE_V12=1 for V12)")

# npci_rotate is unused at inference but imported for module compatibility
try:
    from dsqg_attention_v8_4090 import npci_rotate
except ImportError:
    def npci_rotate(x, *a, **kw): return x

from causal_ema_scan import causal_ema_scan as _causal_ema_scan

assert len(OFFSETS) == 24


def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    return getattr(module, '_orig_mod', module)


def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)


def _agc_normalize(pool, eps=1e-6):
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Model ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV6Physics(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionKernel(embedding_dim, num_heads,
                                         seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor   = nn.Parameter(torch.full((1,), EMA_INIT))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR)
            pool = _agc_normalize(pool)
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            kv_inject = (k_delta, v_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
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
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AutoresearchTransformerPhysics(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer,
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.drop            = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                has_if = (i == full_attn_layer - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight  # tied lm_head
        self._init_weights(scale_embed_init_val)

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
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        if block_idx == self.full_attn_layer:     return True
        if block_idx == self.full_attn_layer - 1: return True
        if CHECKPOINT_STRATEGY == 'all':          return True
        if CHECKPOINT_STRATEGY == 'every_other':  return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':    return block_idx == self.full_attn_layer
        return False

    def forward(self, idx):
        B, N = idx.shape
        x    = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def forward_hidden(self, idx):
        """Return pre-lm_head hidden states [B, N, D] for Liger fused CE."""
        B, N = idx.shape
        x    = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

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

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)


# ── Data utilities ────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    bs = 4
    for i in range(0, len(data) - bs + 1, bs):
        x = data[i:i+bs, :-1].to(device, non_blocking=True)
        y = data[i:i+bs,  1:].to(device, non_blocking=True)
        with _amp_context(device):
            logits = model(x)
        T, V = logits.size(1), logits.size(2)
        batch_loss = 0.0
        for c in range(0, T, CE_CHUNK):
            lc = logits[:, c:c+CE_CHUNK, :].reshape(-1, V).float()
            yc = y[:, c:c+CE_CHUNK].reshape(-1)
            batch_loss += F.cross_entropy(lc, yc, reduction='sum').item()
        total_loss   += batch_loss
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    pad_id     = 0
    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        if not encoded:
            raise ValueError(f'Could not encode passkey word: {word}')
        word_token_ids[word] = encoded[0]

    results = {}
    for d in PASSKEY_DISTANCES:
        seqs, last_pos, cand_rows = [], [], []
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available:
                continue

            filler = []
            while len(filler) < d:
                filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue

            seqs.append(full_seq + [pad_id] * (MAX_SEQ_LEN - len(full_seq)))
            last_pos.append(len(full_seq) - 1)
            cand_words = [target] + others[:9]
            cand_rows.append([word_token_ids[w] for w in cand_words])

        if not seqs:
            results[d] = 0.0
            continue

        ids  = torch.tensor(seqs,      dtype=torch.long, device=device)
        pos  = torch.tensor(last_pos,  dtype=torch.long, device=device)
        cand = torch.tensor(cand_rows, dtype=torch.long, device=device)

        correct = 0
        total   = ids.size(0)
        for start in range(0, total, PASSKEY_BATCH_SIZE):
            ids_b  = ids[start:start + PASSKEY_BATCH_SIZE]
            pos_b  = pos[start:start + PASSKEY_BATCH_SIZE]
            cand_b = cand[start:start + PASSKEY_BATCH_SIZE]
            with _amp_context(device):
                logits = model(ids_b)
            row         = torch.arange(ids_b.size(0), device=device)
            next_logits = logits[row, pos_b, :]
            cand_logits = torch.gather(next_logits, 1, cand_b)
            correct    += (cand_logits.argmax(dim=1) == 0).sum().item()

        results[d] = correct / total
    return results


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir):
    full_attn_block = _unwrap_compiled_module(model.blocks[model.full_attn_layer])
    state_dict = {}
    for name, param in full_attn_block.named_parameters():
        state_dict[f"blocks.{model.full_attn_layer}.{name}"] = param.data.clone()
    payload = {
        "full_attn_block": state_dict,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads":     NUM_HEADS,
            "ffn_dim":       FFN_DIM,
            "seq_len":       MAX_SEQ_LEN,
            "source_script": "train/train_d768_l16_mixed_scratch_v13_4090_bf16.py",
            "source_layer":  FULL_ATTN_LAYER,
            "num_layers":    NUM_LAYERS,
            "num_offsets":   len(OFFSETS),
            "epoch":         epoch,
            "git_hash":      git_hash,
            "note": (
                f"D768-L16-FA4 FROM-SCRATCH MIXED V12: D={EMBEDDING_DIM} H={NUM_HEADS} "
                f"L={NUM_LAYERS} FFN={FFN_DIM} J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER} "
                f"preIF@L{FULL_ATTN_LAYER-1}. Epoch {epoch}/3. "
                f"Dataset: 80% FineWeb-Edu / 20% Wikipedia. "
                f"Kernel: {_kernel_label}, LR_MULT=20.0, EMA_INIT=0.010"
            ),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"d768_l16_mixed_scratch_v13_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Percolation tracker ────────────────────────────────────────────────────────

class PercolationTracker:
    """Track SE|max| history to detect crossing, plateau, and Phase B→C transition."""
    def __init__(self):
        self.crossed   = False
        self.cross_step = None
        self.history   = []   # (step, se_max)
        self.plateau_warned = False

    def update(self, step, se_max):
        self.history.append((step, se_max))
        if not self.crossed and se_max >= 2.0:
            self.crossed    = True
            self.cross_step = step
            print(f'\n  🎯 PERCOLATION THRESHOLD CROSSED: EP1 STEP {step} — SE|max|={se_max:.4f}')

    def phase_label(self, se_max):
        if not self.crossed:
            return '↑ approaching' if se_max >= 1.6 else ''
        # Check if we're in consolidation plateau or resumed growth
        if len(self.history) >= 3:
            recent = [h[1] for h in self.history[-3:]]
            if max(recent) - min(recent) < 0.05:   # <5% variance over last 3 checkpoints
                return '⏸ consolidating (Phase B)'
            elif se_max > self.history[-2][1]:
                return '↑ Phase C (amplitude resuming)'
        return '✓'


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF D768-L16 FA@L4 — FROM-SCRATCH MIXED-DOMAIN (V12 / 4090)')
    print('  Dataset: 80% FineWeb-Edu / 20% Wikipedia')
    print('  Question: Does 11 post-FA layers suffice for D=768 mixed relay?')
    print('  Watch: Phase B consolidation plateau at steps 800-1200 after crossing')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Kernel: {_kernel_label}')
    print(f'  Liger fused CE: {"enabled" if USE_LIGER_CE else "DISABLED (chunked CE fallback)"}')
    print(f'  AdamW8bit: {"enabled" if _BNB_AVAILABLE else "DISABLED (standard AdamW fallback)"}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}, '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t, learns ~700t on mixed)')
    print(f'  LR={LR}, MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found in: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}  (vocab={tokenizer.vocab_size():,})')

    _encoded_cache = 'logs/fineweb_wiki_80_20_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading mixed dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train'].long()
        val_data   = _cache['val'].long()
    else:
        raise FileNotFoundError(
            f'Mixed dataset not found: {_encoded_cache}\n'
            f'Expected: logs/fineweb_wiki_80_20_encoded_2048.pt')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    model = AutoresearchTransformerPhysics(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())

    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params,     'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}
    perc_tracker    = PercolationTracker()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Liger CE setup ────────────────────────────────────────────────────────
    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    # ── Kernel warmup ─────────────────────────────────────────────────────────
    _WARMUP_BS = 4             # small warmup BS to avoid OOM during Triton JIT compile
    print(f'Warming up kernel (warmup BS={_WARMUP_BS}, forward-only)...')
    _wb = min(_WARMUP_BS, len(train_data))
    _wx = train_data[:_wb, :-1].to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if USE_LIGER_CE:
            _whidden = model.forward_hidden(_wx)
            del _whidden
        else:
            _wout = model(_wx)
            del _wout
    del _wx
    torch.cuda.synchronize()
    print('  Kernel warmup complete.')

    # ── MFU tracking ──────────────────────────────────────────────────────────
    gpu_peak_flops  = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step  = 6 * n_params * tokens_per_step
    mfu_window      = deque(maxlen=20)

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        loss_window     = deque(maxlen=20)

        for acc_step in range(steps_per_epoch):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                        loss   = liger_ce_fn(
                            hidden.view(-1, hidden.size(-1)),
                            model.out.weight,
                            y.view(-1))
                    (loss / GRAD_ACCUM).backward()
                    loss_val = loss.item()
                    del hidden, loss
                else:
                    with _amp_context(device):
                        logits = model(x)
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    y_flat      = y.reshape(-1)
                    T           = logits_flat.size(0)
                    grad_logits = torch.empty_like(logits_flat)
                    total_loss  = 0.0
                    for cs in range(0, T, CE_CHUNK):
                        ce   = min(cs + CE_CHUNK, T)
                        chk  = logits_flat[cs:ce].detach().requires_grad_(True)
                        cl   = F.cross_entropy(chk, y_flat[cs:ce], reduction='sum')
                        cl.backward()
                        grad_logits[cs:ce] = chk.grad
                        total_loss += cl.item()
                    logits_flat.backward(grad_logits / (T * GRAD_ACCUM))
                    loss_val = total_loss / T
                    del logits, logits_flat, y_flat, grad_logits

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            t1.record()
            torch.cuda.synchronize()
            step_ms = t0.elapsed_time(t1)
            mfu_window.append(step_ms)
            loss_window.append(loss_val)
            step += 1

            if step % 200 == 0:
                se_vals = []
                for m in model.modules():
                    if isinstance(m, DSQGAttentionKernel):
                        se_vals.append(m.scale_embed.detach().abs())
                se_max = torch.cat(se_vals).max().item() if se_vals else 0.0

                perc_tracker.update(step, se_max)
                phase = perc_tracker.phase_label(se_max)

                avg_ms      = sum(mfu_window) / len(mfu_window)
                tok_per_sec = tokens_per_step / (avg_ms / 1000.0)
                loss_std    = (sum((l - sum(loss_window)/len(loss_window))**2
                               for l in loss_window) / len(loss_window)) ** 0.5

                mfu_str = ''
                if gpu_peak_flops is not None:
                    mfu = (flops_per_step / (avg_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'

                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} (σ={loss_std:.3f}) '
                      f'| SE |max|={se_max:.4f} {phase}'
                      f'{mfu_str} | {tok_per_sec:.0f} tok/s', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, 'd768_l16_mixed_scratch_v13_best.pt'))
            marker = ' *'

        torch.save({
            'epoch': epoch,
            'model_state_dict': {k.replace('._orig_mod', ''): v
                                 for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_ppl':  val_ppl,
        }, os.path.join(CHECKPOINT_DIR, f'd768_l16_mixed_scratch_v13_ep{epoch}.pt'))

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker} '
              f'| Loss σ={sum((l - sum(loss_window)/len(loss_window))**2 for l in loss_window)**0.5 / len(loss_window):.4f}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionKernel):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            se_max = se_all.max().item()
            status = ('ABOVE THRESHOLD ✓' if se_max >= 2.0 else
                      f'BELOW THRESHOLD ✗ (need {2.0 - se_max:.4f} more)')
            print(f'  SE|mean|={se_all.mean():.4f} |max|={se_max:.4f} — {status}')

        print(f'  Physics: {model.physics_summary()}')
        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100

        relay_status = 'relay active ✓' if pk_mean > 0.5 else 'random — relay not yet formed'
        print(f'  Passkey mean={pk_mean * 100:.1f}%  ({relay_status})')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        if epoch >= 2 and pk_mean < PASSKEY_ABORT_THRESHOLD:
            print(f'\n  ⛔ ABORT: Passkey {pk_mean*100:.1f}% < {PASSKEY_ABORT_THRESHOLD*100:.0f}% '
                  f'— relay formation failed (capacity floor at D=768/L=16?)')
            break

        sys.stdout.flush()

    elapsed_s = time.time() - t_start
    memory_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

    print('\n--- RESULTS ---')
    print(f'LR={LR} LR_MULT={SCALE_EMBED_LR_MULT} EMA_INIT={EMA_INIT}')
    print(f'percolation_step: {perc_tracker.cross_step or "NOT CROSSED"}')
    if len(perc_tracker.history) >= 2:
        print(f'global_loss_std: (see per-step σ above)')
    for ep in sorted(passkey_results):
        print(f'passkey_ep{ep}: {passkey_results[ep]:.1f}%')
    for ep in sorted(ppl_results):
        print(f'ppl_ep{ep}: {ppl_results[ep]:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_params / 1e6:.1f}')
    print(f'description: D768-L16-FA4 MIXED-SCRATCH-V12 D={EMBEDDING_DIM} H={NUM_HEADS} '
          f'hd=64 L={NUM_LAYERS} FFN={FFN_DIM} J=24 se015 '
          f'FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1} '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers '
          f'kernel={_kernel_label} LR_MULT={SCALE_EMBED_LR_MULT} EMA_INIT={EMA_INIT}')


if __name__ == '__main__':
    train()
