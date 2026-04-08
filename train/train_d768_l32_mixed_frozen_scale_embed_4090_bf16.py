"""
🧪 EXPERIMENT: Mixed-Domain Continuation with scale_embed FROZEN

Hypothesis being tested:
  The mixed-domain relay collapse (FineWeb 60%/PG19 25%/Stack 15%, March 29 2026)
  may have been driven primarily by unconstrained scale_embed amplitude decay
  rather than direct K/Q erosion by domain gradient conflict.

  Previous run (unfrozen scale_embed):
    scale_embed: 2.12 → 1.36 (fell below maintenance threshold)
    passkey:     ~100% → 79.2%

  This run freezes scale_embed to isolate the confound.

Expected outcomes:
  - If passkey holds ≥90%: collapse was amplitude-driven, not K/Q-driven.
    Diverse pretraining is viable with scale_embed frozen.
  - If passkey degrades anyway: K/Q weights erode under PG19/Stack gradients
    regardless of scale_embed. Domain coherence constraint is fundamental.

Architecture: D=768, H=12 (hd=64), L=32, FFN=1536, J=24 (se015 offsets)
  L0-L6:  DSQGBlockV6Physics  IF=False
  L7:     DSQGBlockV6Physics  IF=True   ← preIF@L7
  L8:     FullAttentionBlock            ← FA@L8 (25% depth)
  L9-L31: DSQGBlockV6Physics  IF=False  ← 23 post-FA relay layers

Resume from: autoresearch/checkpoints/d768_l32_fa8_best.pt
  (cont_ep2: PPL=25.85, passkey=98.3%, FineWeb 80%/Wikipedia 20%)

Dataset: logs/mixed_encoded_2048_fineweb_tok.pt
  (60% FineWeb-Edu / 25% PG19 / 15% The Stack — 234,418 seqs, fineweb_tokenizer_32k)

scale_embed: FROZEN (requires_grad=False, excluded from optimizer)
  Rationale: LR_MULT=18.37 at D=768; unfrozen SFT caused 2.07→0.43 collapse.
  Same mechanism applies to continued pretraining. Freeze prevents amplitude decay.

Outputs:
  autoresearch/checkpoints/d768_l32_mixed_frozen_se_ep{N}.pt   (full resume)
  autoresearch/checkpoints/d768_l32_mixed_frozen_se_best.pt    (best val loss)

Run (from repo root, on 4090):
  tmux new-session -d -s mixed_frozen 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u \\
    train/train_d768_l32_mixed_frozen_scale_embed_4090_bf16.py \\
    2>&1 | tee logs/run_d768_l32_mixed_frozen_se.log'
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
           48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 768
NUM_HEADS        = 12         # hd = 768/12 = 64 ✓
FFN_DIM          = 1536       # 2×D
NUM_LAYERS       = 32
FULL_ATTN_LAYER  = 8          # FA@L8 (25% depth), preIF@L7

# Mixed dataset: 60% FineWeb-Edu / 25% PG19 / 15% The Stack
MIXED_DATASET        = 'logs/mixed_encoded_2048_fineweb_tok.pt'
MAX_TRAIN_SEQS       = 220_000   # use most of 234K; leave headroom for val split
MAX_VAL_SEQS         = 5_000

SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 18.37    # μP: 15 × √(768/512) — used for reference only; SE is frozen

EMA_INIT  = 0.0208              # 1/δ_relay_min = 1/48
EMA_FLOOR = 0.00001

LR            = 5e-5            # Conservative: continued pretraining, not cold start
SCREEN_EPOCHS = 2               # 2 epochs sufficient to diagnose
RESUME_CHECKPOINT = 'autoresearch/checkpoints/d768_l32_fa8_best.pt'
CHECKPOINT_PREFIX = 'd768_l32_mixed_frozen_se'

# Passkey abort threshold: if passkey mean drops below this, stop early
PASSKEY_ABORT_THRESHOLD = 0.80  # 80% — relay clearly degrading

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
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

# ── Liger fused CE ─────────────────────────────────────────────────────────────
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"


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
BATCH_SIZE     = 16
GRAD_ACCUM     = 8    # effective batch = 128 (matches pretraining)
CE_CHUNK       = 512
MAX_SEQ_LEN    = 2048

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
PASSKEY_BATCH_SIZE = int(os.getenv('DWARF_PASSKEY_BATCH', '32'))

# Warmup uses a smaller batch to avoid OOM from full logits materialisation.
# The training loop uses chunked CE so is cheaper than the warmup pass.
# Note: mismatched warmup BS causes a one-time Triton recompile stall at
# step 1 — acceptable; not a crash.
WARMUP_BATCH_SIZE = 4

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
            if isinstance(m, DSQGAttentionV6):
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def forward(self, idx):
        B, N = idx.shape
        x    = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and i % 2 == 0:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def forward_hidden(self, idx):
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

    def freeze_scale_embed(self):
        """Freeze all scale_embed parameters. Call before building optimizer."""
        frozen_count = 0
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                m.scale_embed.requires_grad_(False)
                frozen_count += 1
        return frozen_count

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)

    def scale_embed_stats(self):
        se_vals = []
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            return se_all.mean().item(), se_all.max().item()
        return 0.0, 0.0


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


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 EXPERIMENT: Mixed-Domain Continued Pretraining, scale_embed FROZEN')
    print('  Resume: d768_l32_fa8_best.pt (cont_ep2: PPL=25.85, passkey=98.3%)')
    print('  Dataset: 60% FineWeb-Edu / 25% PG19 / 15% The Stack')
    print('  scale_embed: FROZEN — testing amplitude-vs-K/Q collapse hypothesis')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Liger not available, using chunked CE')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}, '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers')
    print(f'  LR={LR}, BS={BATCH_SIZE}×GRAD_ACCUM={GRAD_ACCUM}='
          f'eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  Passkey abort threshold: {PASSKEY_ABORT_THRESHOLD*100:.0f}%')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}  (vocab={tokenizer.vocab_size():,})')

    if not os.path.exists(MIXED_DATASET):
        raise FileNotFoundError(
            f'Mixed dataset not found: {MIXED_DATASET}\n'
            f'Expected: 60% FineWeb-Edu / 25% PG19 / 15% Stack, fineweb_tokenizer_32k')
    print(f'Loading mixed dataset from {MIXED_DATASET}')
    _cache     = torch.load(MIXED_DATASET, weights_only=True)
    all_data   = _cache['train'].long() if 'train' in _cache else _cache.long()
    val_data   = _cache['val'].long()   if 'val'   in _cache else all_data[-MAX_VAL_SEQS:]

    if len(all_data) > MAX_TRAIN_SEQS + MAX_VAL_SEQS:
        train_data = all_data[:MAX_TRAIN_SEQS]
    else:
        train_data = all_data[:max(len(all_data) - MAX_VAL_SEQS, 1)]
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

    print(f'Parameters: {model.param_count():,} ({model.param_count()/1e6:.1f}M)')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(RESUME_CHECKPOINT):
        raise FileNotFoundError(f'Checkpoint not found: {RESUME_CHECKPOINT}')
    print(f'Loading checkpoint: {RESUME_CHECKPOINT}')
    ckpt  = torch.load(RESUME_CHECKPOINT, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=True)
    se_mean, se_max = model.scale_embed_stats()
    print(f'  Checkpoint loaded. scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}')

    # ── FREEZE scale_embed — CRITICAL ─────────────────────────────────────────
    n_frozen = model.freeze_scale_embed()
    print(f'  *** scale_embed FROZEN ({n_frozen} parameters) ***')
    print(f'  scale_embed will NOT be updated during this run.')
    print(f'  If relay degrades, cause is K/Q erosion, not amplitude collapse.')

    # ── Optimizer (scale_embed excluded) ─────────────────────────────────────
    trainable_params = list(model.non_scale_embed_parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = model.param_count()
    print(f'  Trainable: {n_trainable:,} / {n_total:,} params '
          f'({100*n_trainable/n_total:.1f}% — scale_embed frozen)')

    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)(
        trainable_params,
        lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    best_val_loss = float('inf')
    passkey_results, ppl_results = {}, {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Kernel warmup ─────────────────────────────────────────────────────────
    print(f'Warming up Triton kernels (warmup BS={WARMUP_BATCH_SIZE}, train BS={BATCH_SIZE})...')
    _wb   = min(WARMUP_BATCH_SIZE, len(train_data))
    _wx   = train_data[:_wb, :-1].to(device)
    _wy   = train_data[:_wb, 1:].to(device)
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
            _wlf = _wout.reshape(-1, _wout.size(-1))
            _wyf = _wy.reshape(-1)
            _wT  = _wlf.size(0)
            _wg  = torch.empty_like(_wlf)
            for _wcs in range(0, _wT, CE_CHUNK):
                _wce  = min(_wcs + CE_CHUNK, _wT)
                _wc   = _wlf[_wcs:_wce].detach().requires_grad_(True)
                _wl   = F.cross_entropy(_wc, _wyf[_wcs:_wce], reduction='sum')
                _wl.backward()
                _wg[_wcs:_wce] = _wc.grad
            _wlf.backward(_wg / _wT)
            del _wout, _wlf, _wyf, _wl
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy
    torch.cuda.synchronize()
    print('  Kernel warmup complete.')

    # ── MFU tracking setup ─────────────────────────────────────────────────────
    n_params = model.param_count()
    gpu_peak_flops = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step = 6 * n_params * tokens_per_step
    mfu_window = deque(maxlen=20)
    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    # ── Eval at start (baseline before any mixed-domain training) ─────────────
    print('\n--- Baseline eval (cont_ep2, before mixed-domain exposure) ---')
    val_loss_0 = evaluate(model, val_data, device)
    print(f'  Val PPL (baseline): {math.exp(min(val_loss_0, 20)):.2f}')
    pk0 = passkey_accuracy(model, tokenizer, device)
    pk0_mean = sum(pk0.values()) / len(pk0)
    print(f'  Passkey mean (baseline): {pk0_mean*100:.1f}%')
    parts = [f'd={d}:{int(pk0[d]*100)}%' for d in PASSKEY_DISTANCES]
    print('  ' + '  '.join(parts))
    se_mean, se_max = model.scale_embed_stats()
    print(f'  scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f} (should be ~2.1)')
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
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
                            hidden.contiguous().reshape(-1, hidden.size(-1)),
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
                se_mean, se_max = model.scale_embed_stats()
                avg_step_ms = sum(mfu_window) / len(mfu_window)
                tok_per_sec = tokens_per_step / (avg_step_ms / 1000.0)
                mfu_str = ''
                if gpu_peak_flops is not None:
                    mfu = (flops_per_step / (avg_step_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'
                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} '
                      f'| SE |max|={se_max:.4f} (frozen, should be stable)'
                      f'{mfu_str} | {tok_per_sec:.0f} tok/s',
                      flush=True)

        # ── Epoch eval ────────────────────────────────────────────────────────
        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        se_mean, se_max = model.scale_embed_stats()
        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}')
        print(f'  scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f} '
              f'(FROZEN — confirm no change from baseline)')
        print(f'  Physics: {model.physics_summary()}')

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean*100:.1f}%  '
              f'(baseline={pk0_mean*100:.1f}%, delta={((pk_mean-pk0_mean)*100):+.1f}pp)')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        # Save checkpoint (never overwrites baseline d768_l32_fa8_best.pt)
        save_path = os.path.join(CHECKPOINT_DIR, f'{CHECKPOINT_PREFIX}_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ppl':  val_ppl,
            'passkey_mean': pk_mean,
            'scale_embed_frozen': True,
            'experiment': 'mixed_domain_frozen_scale_embed',
        }, save_path)
        print(f'  Saved: {save_path}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import shutil
            shutil.copy(save_path,
                        os.path.join(CHECKPOINT_DIR, f'{CHECKPOINT_PREFIX}_best.pt'))
            print(f'  → New best: {CHECKPOINT_PREFIX}_best.pt')

        # ── Abort if relay clearly collapsing ────────────────────────────────
        if pk_mean < PASSKEY_ABORT_THRESHOLD:
            print(f'\n  ⚠️  ABORT: passkey {pk_mean*100:.1f}% < threshold '
                  f'{PASSKEY_ABORT_THRESHOLD*100:.0f}%')
            print('  Conclusion: K/Q weights erode under mixed-domain gradient.')
            print('  Domain coherence constraint is fundamental, not amplitude-driven.')
            break
        else:
            print(f'  ✓  Passkey above abort threshold '
                  f'({PASSKEY_ABORT_THRESHOLD*100:.0f}%). Continuing.')

        sys.stdout.flush()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_s  = time.time() - t_start
    memory_mb  = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

    print('\n' + '=' * 70)
    print('  EXPERIMENT SUMMARY')
    print('=' * 70)
    print(f'  Baseline (cont_ep2):  PPL=25.85, passkey={pk0_mean*100:.1f}%')
    for ep in range(1, SCREEN_EPOCHS + 1):
        if ep in ppl_results:
            delta_pk = passkey_results.get(ep, 0.0) - pk0_mean * 100
            print(f'  Ep{ep} (mixed+frozen): PPL={ppl_results[ep]:.2f}, '
                  f'passkey={passkey_results.get(ep, 0.0):.1f}% '
                  f'(delta={delta_pk:+.1f}pp)')

    final_pk = passkey_results.get(max(passkey_results.keys(), default=1), 0.0)
    if final_pk >= PASSKEY_ABORT_THRESHOLD * 100:
        print('\n  → RESULT: Relay held under mixed-domain pretraining with frozen scale_embed.')
        print('    Hypothesis SUPPORTED: prior collapse was amplitude-driven.')
        print('    Diverse pretraining is viable with scale_embed frozen.')
    else:
        print('\n  → RESULT: Relay degraded despite frozen scale_embed.')
        print('    Hypothesis NOT SUPPORTED: K/Q weights erode directly under mixed-domain gradient.')
        print('    Domain coherence constraint is fundamental.')

    print(f'\n  elapsed_s: {elapsed_s:.1f}')
    print(f'  memory_mb: {memory_mb:.1f}')

    # Machine-readable summary for autoresearch pipeline
    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'passkey_baseline: {pk0_mean*100:.1f}')
    print(f'scale_embed_frozen: True')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'experiment: mixed_domain_frozen_scale_embed')


if __name__ == '__main__':
    train()
