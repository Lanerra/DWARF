"""
🚀 DWARF D=768 L=32 — ~169.8M params, coherence-ceiling test, cold-start

Architecture: D=768, H=12 (hd=64), L=24, FFN=1536, J=24 (se015 offsets), TIED lm_head
  L0-L6:  DSQGBlockV6Physics  IF=False  ← 7 pre-FA warm-up relay layers
  L7:     DSQGBlockV6Physics  IF=True   ← preIF@L7
  L8:     FullAttentionBlock            ← FA@L8 (25% depth, 23 post-FA relay layers)
  L9-31:  DSQGBlockV6Physics  IF=False  ← 23 post-FA relay layers

Hypothesis: The emergent FA distributed routing observed in 267M (D=1024, L=24) is
  driven by post-FA depth (17 layers), not D alone. D=768/L=24 matches 267M's post-FA
  depth at 17 layers. D=768 coherence ≈ 18 layers covers 17 post-FA layers (D=512 only
  ≈12, insufficient). If FA develops the same distributed structure as 267M, depth is
  the mechanism. If not, D is the mechanism.

Comparison:
  d768_l16 (111.5M, L=16, 11 post-FA): FA shows sparse pattern-matching activation
  d1024_267m (267M, L=24, 17 post-FA):  FA shows diffuse diagonal relay routing
  d768_l24 (132M, L=24, 17 post-FA):    < this run — tests depth vs D hypothesis

Derived quantities:
  hd = 768/12 = 64 ✓ (validated head dimension)
  Relay/residual ratio = (J=24 × hd=64) / D=768 = 2.0× (validated)
  LR_MULT = 15 × √(768/512) = 18.37
  Chinchilla for 132M: 132M × 20 = 2.64B tokens
  Actual param count: ~154M (32 extra DSQG layers vs 111M estimate)
  Chinchilla for 154M: 154M × 20 = 3.08B tokens
  22% Chinchilla: 0.22 × 3.08B / 2048 ≈ 331,055 seqs → MAX_TRAIN_SEQS = 331_000

Baselines:
  Moonshot-58M ep2   (PPL=35.04, passkey=99.2%, ar_score=80.90) — D=512 L=8
  d768_l16 ep3       (PPL=34.42, passkey=100.0%)                 — D=768 L=16
  267M D=1024 ep1    (PPL=27.75, passkey=93.3%)                  — D=1024 L=24

Config:
  - Tokenizer: fineweb_tokenizer_32k.json  (32K BPE, FineWeb proper)
  - Dataset:   fineweb_edu_encoded_2048_v2.pt (~2.01M seqs, 4.13B tokens)
  - EMA_INIT = 0.0208 (= 1/δ_relay_min for J24D se015)
  - SCALE_EMBED_INIT = 0.15, LR_MULT = 18.37
  - Batch: BS=128 × GRAD_ACCUM=1 = eff_batch=128 (H200)
  - Cold start, ~154M parameters (tied lm_head)

Screen: 3 epochs × 331,000 seqs (22% Chinchilla for ~154M params)

Run (from repo root, on H100 pod):
  .venv/bin/python3 -u train/train_d768_l32_h100_bf16.py \\
    2>&1 | tee logs/run_d768_l32.log
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 768
NUM_HEADS        = 12         # hd = 768/12 = 64 ✓ (validated head dimension)
FFN_DIM          = 1536       # 2×D — confirmed optimal by FFN ablation
NUM_LAYERS       = 32
FULL_ATTN_LAYER  = 8          # 25% of 32 = L8; 23 post-FA relay layers (L9-31); preIF@L7

MAX_TRAIN_SEQS       = 128_000   # 80/20 FineWeb+Wiki mixed dataset
SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 18.37    # μP: 15 × √(768/512)

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# Empirically validated for J24D (se015): trains to α≈0.0207, 0.6% error
# δ_relay_min = 48 (first offset after local cluster [1..28])
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 1e-4   # reduced LR for continued training
SCREEN_EPOCHS = 3
RESUME_CHECKPOINT = 'autoresearch/checkpoints/d768_l32_fa8_ep3_resume.pt'

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
from collections import deque
MAX_TRAIN_SEQS = int(os.environ.get('MAX_TRAIN_SEQS_OVERRIDE', MAX_TRAIN_SEQS))  # pipeline override
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ── Liger fused CE ─────────────────────────────────────────────────────────────
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") == "1"


def get_gpu_peak_flops(device="cuda"):
    """Return peak BF16 TFLOPs for the detected GPU."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    if "H100" in name:
        return 989e12
    elif "H200" in name:
        return 1979e12
    elif "4090" in name:
        return 330e12
    elif "3090" in name:
        return 142e12
    elif "A100" in name:
        return 312e12
    return None


VOCAB_SIZE     = 32000
BATCH_SIZE     = 128
GRAD_ACCUM     = 1    # effective batch = 128
CE_CHUNK       = 512  # chunked CE token stride — avoids materialising full (BS×2047×32K) fp32 grad tensor
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50   # n=50: ±7pp noise floor
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

ENABLE_TORCH_COMPILE = os.getenv('DWARF_ENABLE_COMPILE', '0') == '1'
COMPILE_MODE         = os.getenv('DWARF_COMPILE_MODE', 'reduce-overhead')
CHECKPOINT_STRATEGY  = os.getenv('DWARF_CKPT', 'every_other').lower()   # none | full_attn | every_other | all
PASSKEY_BATCH_SIZE   = int(os.getenv('DWARF_PASSKEY_BATCH', '32'))

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_h100 import DSQGAttentionV8_H100 as DSQGAttentionV6, npci_rotate
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

assert len(OFFSETS) == 24


def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    return getattr(module, '_orig_mod', module)

# ── Physics helpers ───────────────────────────────────────────────────────────

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
        self.attn  = DSQGAttentionV6(embedding_dim, num_heads,
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
        # Tied lm_head — shares weights with embedding table
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
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
            if isinstance(m, DSQGAttentionV6):
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.full_attn_layer
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
        """Return pre-lm_head hidden states [B, N, D] for chunked CE computation."""
        B, N = idx.shape
        x    = self.drop(self.embedding(idx))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def full_attn_parameters(self):
        for p in self.blocks[self.full_attn_layer].parameters():
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
def evaluate(model, data, device, CE_CHUNK=512):
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
            target = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others = [w for w in _PASSKEY_WORDS if w != target]
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

        ids  = torch.tensor(seqs,     dtype=torch.long, device=device)
        pos  = torch.tensor(last_pos, dtype=torch.long, device=device)
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
            "source_script": "train/train_d768_l32_h100_bf16.py",
            "source_layer":  FULL_ATTN_LAYER,
            "num_layers":    NUM_LAYERS,
            "num_offsets":   len(OFFSETS),
            "epoch":         epoch,
            "git_hash":      git_hash,
            "note": (
                f"D768-L32-FA8: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} "
                f"FFN={FFN_DIM} J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER} "
                f"preIF@L{FULL_ATTN_LAYER-1}. Epoch {epoch}/3. Cold start."
            ),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"d768_l32_fa8_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🚀 DWARF D768-L32 FA@L8 — D=768 H=12 hd=64 L=24 FFN=1536 J=24, cold start')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}, '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers '
          f'(matches 267M post-FA depth), fineweb_tokenizer_32k')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Liger not available, using chunked CE')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}')
    print(f'  Post-FA relay layers: {NUM_LAYERS - FULL_ATTN_LAYER - 1}  '
          f'(267M has 17; d768_l16 had 11)')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}  passkey_batch_size={PASSKEY_BATCH_SIZE}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found in: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}  (vocab={tokenizer.vocab_size():,})')

    _encoded_cache = 'logs/fineweb_wiki_80_20_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train'].long()
        val_data   = _cache['val'].long()
    else:
        raise FileNotFoundError(
            f'Pre-encoded dataset not found: {_encoded_cache}\n'
            f'Run scripts/build_dataset_fineweb.py first.')

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

    if ENABLE_TORCH_COMPILE:
        try:
            for i, block in enumerate(model.blocks):
                if type(block).__name__ == "FullAttentionBlock":
                    try:
                        model.blocks[i] = torch.compile(block, fullgraph=False, dynamic=False, mode=COMPILE_MODE)
                    except TypeError:
                        model.blocks[i] = torch.compile(block, fullgraph=False)
                    print(f"  torch.compile applied to FullAttentionBlock at layer {i} (mode={COMPILE_MODE})")
                    break
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    else:
        print('  torch.compile disabled by default (set DWARF_ENABLE_COMPILE=1 to opt in)')

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if 'RESUME_CHECKPOINT' in dir() or True:
        _resume = globals().get('RESUME_CHECKPOINT', None)
        if _resume and os.path.exists(_resume):
            print(f'Loading checkpoint: {_resume}')
            ckpt = torch.load(_resume, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
            model.load_state_dict(state, strict=True)
            print('  Checkpoint loaded successfully.')
        else:
            print(f'  No resume checkpoint found at {_resume!r}, starting fresh.')


    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = torch.optim.AdamW([
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Kernel warmup pass ────────────────────────────────────────────────────
    # WARMUP_BS must match BATCH_SIZE — Triton specialises kernels on (B, N);
    # mismatched warmup causes recompilation stall on first real step.
    _WARMUP_BS = BATCH_SIZE
    print(f'Warming up Triton kernels (BS={_WARMUP_BS} dummy pass)...')
    _wb       = min(_WARMUP_BS, len(train_data))
    _wx       = train_data[:_wb, :-1].to(device)
    _wy       = train_data[:_wb, 1:].to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if USE_LIGER_CE:
            _whidden = model.forward_hidden(_wx)
            _liger_ce_fn = LigerFusedLinearCrossEntropyLoss()
            _wloss = _liger_ce_fn(
                _whidden.view(-1, _whidden.size(-1)),
                model.out.weight,
                _wy.view(-1)
            )
            _wloss.backward()
            del _whidden, _wloss
        else:
            _wout = model(_wx)
            _wlogits_flat = _wout.reshape(-1, _wout.size(-1))
            _wy_flat      = _wy.reshape(-1)
            _wT           = _wlogits_flat.size(0)
            _wgrad        = torch.empty_like(_wlogits_flat)
            for _wcs in range(0, _wT, CE_CHUNK):
                _wce    = min(_wcs + CE_CHUNK, _wT)
                _wchunk = _wlogits_flat[_wcs:_wce].detach().requires_grad_(True)
                _wloss  = F.cross_entropy(_wchunk, _wy_flat[_wcs:_wce], reduction='sum')
                _wloss.backward()
                _wgrad[_wcs:_wce] = _wchunk.grad
            _wlogits_flat.backward(_wgrad / _wT)
            del _wout, _wlogits_flat, _wy_flat, _wloss
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy
    torch.cuda.synchronize()
    print('  kernel warmup complete.')

    # ── MFU tracking setup ─────────────────────────────────────────────────────
    gpu_peak_flops = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step = 6 * n_params * tokens_per_step
    mfu_window = deque(maxlen=20)
    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            t_start_event = torch.cuda.Event(enable_timing=True)
            t_end_event   = torch.cuda.Event(enable_timing=True)
            t_start_event.record()

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
                        loss = liger_ce_fn(
                            hidden.view(-1, hidden.size(-1)),
                            model.out.weight,
                            y.view(-1)
                        )
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
                    for chunk_start in range(0, T, CE_CHUNK):
                        chunk_end  = min(chunk_start + CE_CHUNK, T)
                        chunk      = logits_flat[chunk_start:chunk_end].detach().requires_grad_(True)
                        chunk_loss = F.cross_entropy(
                            chunk, y_flat[chunk_start:chunk_end], reduction='sum')
                        chunk_loss.backward()
                        grad_logits[chunk_start:chunk_end] = chunk.grad
                        total_loss += chunk_loss.item()
                    logits_flat.backward(grad_logits / (T * GRAD_ACCUM))
                    loss_val = total_loss / T
                    del logits, logits_flat, y_flat, grad_logits

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            t_end_event.record()
            torch.cuda.synchronize()
            step_ms = t_start_event.elapsed_time(t_end_event)
            mfu_window.append(step_ms)
            step += 1

            if step % 200 == 0:
                avg_step_ms = sum(mfu_window) / len(mfu_window)
                tok_per_sec = tokens_per_step / (avg_step_ms / 1000.0)
                mfu_str = ''
                if gpu_peak_flops is not None:
                    mfu = (flops_per_step / (avg_step_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f}{mfu_str} | {tok_per_sec:.0f} tok/s', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, 'd768_l32_fa8_best.pt'))
            marker = ' *'

        # Save resume checkpoint (full model + optimizer state for continuation)
        torch.save({
            'epoch': epoch,
            'model_state_dict': {k.replace('._orig_mod', ''): v
                                 for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_ppl':  val_ppl,
        }, os.path.join(CHECKPOINT_DIR, f'd768_l32_fa8_ep{epoch}_resume.pt'))

        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionV6):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} '
                  f'|max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')
        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s     = time.time() - t_start
    memory_mb     = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
    # ar_score relative to Moonshot-58M ep2 (project baseline)
    PPL_BASELINE     = 35.04
    PASSKEY_BASELINE = 99.2
    ar_score = (passkey_final - PASSKEY_BASELINE) + (PPL_BASELINE - ppl_final) * 0.5

    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score: {ar_score:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_params / 1e6:.1f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'num_offsets: {len(OFFSETS)}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init: {EMA_INIT}')
    print(f'description: D768-L32-FA8 D={EMBEDDING_DIM} H={NUM_HEADS} hd=64 L={NUM_LAYERS} '
          f'FFN={FFN_DIM} J=24 se015, cold start, fineweb_tokenizer_32k, '
          f'FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1} '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers')


if __name__ == '__main__':
    train()
