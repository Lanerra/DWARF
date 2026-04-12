"""
🧪 DWARF D=512 L=8 — MIXED-DOMAIN HYPERPARAMETER VALIDATION (4090)

Purpose: Validate the adjusted mixed-domain hyperparameters derived from analysis
of H3 from-scratch failures:
  - LR_MULT: 15.0 → 20.0  (D=512 μP corrected for mixed-domain gradient dilution)
  - EMA_INIT: 0.0208 → 0.010  (closer to learned mixed-domain window of ~690-822t)
  - LR: 3e-4 → 2.5e-4  (reduce 2.6× higher loss variance from domain competition)

Architecture: D=512, H=8 (hd=64), L=8, FFN=1024 (2×D), J=24 (se015), TIED lm_head
FA@L2 (25% depth), preIF@L1
Dataset: Mixed 60% FineWeb-Edu / 25% PG19 / 15% The Stack
Hardware: RTX 4090 — BS=32, GRAD_ACCUM=4 (eff_batch=128)

Hypothesis: With corrected hyperparameters the relay should:
  1. Cross percolation threshold (~2.0) at step ~800 (not ~1000 as in H3)
  2. Show lower loss variance (σ < 0.06 vs H3's 0.098)
  3. Achieve ep2 passkey > 70% (better than H3 D768-L32 baseline)

Run:
  cd /home/dlewis3/Desktop/AI/DWARF
  tmux new-session -d -s d512_mixed \\
    ".venv/bin/python3 -u train/train_d512_l8_mixed_hyp_4090_bf16.py \\
     2>&1 | tee logs/run_d512_l8_mixed_hyp.log"
  tmux attach -t d512_mixed
"""

# =============================================================================
# CONFIG
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
assert len(OFFSETS) == 24

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 1024    # 2×D
NUM_LAYERS       = 8
FULL_ATTN_LAYER  = 2       # 25% depth (L/4)

# ── Adjusted mixed-domain hyperparameters ─────────────────────────────────────
LR               = 2.5e-4  # was 3e-4; reduce gradient competition variance
SCALE_EMBED_LR_MULT = 20.0 # was 15.0 (D=512 μP); corrected for ~30% mixed gradient dilution
EMA_INIT         = 0.010   # was 0.0208 (≈48t window); mixed corpus learns ≈690-822t
# ──────────────────────────────────────────────────────────────────────────────

SCALE_EMBED_INIT_VAL = 0.15
EMA_FLOOR            = 0.00001
SCREEN_EPOCHS        = 3
PASSKEY_ABORT_THRESHOLD = 0.20

MAX_TRAIN_SEQS  = 234_418
MAX_VAL_SEQS    = 5_582
VOCAB_SIZE       = 32000
BATCH_SIZE       = 16
GRAD_ACCUM       = 8
CE_CHUNK         = 512
MAX_SEQ_LEN      = 2048

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
PASSKEY_BATCH_SIZE = 32
CHECKPOINT_DIR     = 'autoresearch/checkpoints'

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]
MIXED_DATASET_PATH = 'logs/mixed_encoded_2048_fineweb_tok.pt'

CHECKPOINT_STRATEGY  = 'every_other'
ENABLE_TORCH_COMPILE = False

# =============================================================================

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available, using standard AdamW")

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_h100 import DSQGAttentionV8_H100 as DSQGAttentionV6, npci_rotate
from causal_ema_scan import causal_ema_scan as _causal_ema_scan


def get_gpu_peak_flops(device="cuda"):
    if not torch.cuda.is_available(): return None
    name = torch.cuda.get_device_name(device)
    if "H200" in name: return 1979e12
    if "H100" in name: return 989e12
    if "4090" in name: return 330e12
    if "3090" in name: return 142e12
    if "A100" in name: return 312e12
    return None


def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


class DSQGBlock(nn.Module):
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
            pool    = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR)
            pool    = _agc_normalize(pool)
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


class DWARFModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer,
                 scale_embed_init_val=0.15, dropout=0.1):
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
                blocks.append(DSQGBlock(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight  # tied
        self._init_weights(scale_embed_init_val)

    def _init_weights(self, se_init):
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
                if se_init != 0.0:
                    nn.init.constant_(m.scale_embed, se_init)

    def _should_checkpoint(self, i):
        if i in (self.full_attn_layer, self.full_attn_layer - 1):
            return True
        if CHECKPOINT_STRATEGY == 'all':      return True
        if CHECKPOINT_STRATEGY == 'every_other': return i % 2 == 0
        return False

    def forward(self, idx):
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint(i):
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def forward_hidden(self, idx):
        x = self.drop(self.embedding(idx))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_params(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                yield m.scale_embed

    def phase_params(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                yield m.phase_gain
                yield m.phase_gate
                yield m.query_probes
                yield m.key_probes

    def non_scale_embed_params(self):
        exclude_ids = {id(p) for p in self.scale_embed_params()}
        exclude_ids.update(id(p) for p in self.phase_params())
        for p in self.parameters():
            if id(p) not in exclude_ids:
                yield p

    def se_stats(self):
        vals = torch.cat([m.scale_embed.detach().abs()
                          for m in self.modules()
                          if isinstance(m, DSQGAttentionV6)])
        return vals.mean().item(), vals.max().item()

    def ema_summary(self):
        parts = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlock) and block.interference:
                a = abs(block.ema_factor.item()) + EMA_FLOOR
                parts.append(f'b{i}: α={a:.4f}(w≈{round(1/max(a, EMA_FLOOR))}t)')
        return '  '.join(parts)


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
    for i in range(0, len(data) - 4 + 1, 4):
        x = data[i:i+4, :-1].to(device, non_blocking=True)
        y = data[i:i+4,  1:].to(device, non_blocking=True)
        with _amp_context(device):
            logits = model(x)
        T, V = logits.size(1), logits.size(2)
        for c in range(0, T, CE_CHUNK):
            lc = logits[:, c:c+CE_CHUNK, :].reshape(-1, V).float()
            yc = y[:, c:c+CE_CHUNK].reshape(-1)
            total_loss   += F.cross_entropy(lc, yc, reduction='sum').item()
            total_tokens += yc.numel()
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    pad_id     = 0
    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        enc = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        if not enc:
            raise ValueError(f'Could not encode: {word}')
        word_token_ids[word] = enc[0]

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
            cand_rows.append([word_token_ids[target]] +
                             [word_token_ids[w] for w in others[:9]])

        if not seqs:
            results[d] = 0.0
            continue

        ids  = torch.tensor(seqs,     dtype=torch.long, device=device)
        pos  = torch.tensor(last_pos, dtype=torch.long, device=device)
        cand = torch.tensor(cand_rows, dtype=torch.long, device=device)
        correct = 0
        for start in range(0, ids.size(0), PASSKEY_BATCH_SIZE):
            ib = ids[start:start + PASSKEY_BATCH_SIZE]
            pb = pos[start:start + PASSKEY_BATCH_SIZE]
            cb = cand[start:start + PASSKEY_BATCH_SIZE]
            with _amp_context(device):
                logits = model(ib)
            row         = torch.arange(ib.size(0), device=device)
            next_logits = logits[row, pb, :]
            cand_logits = torch.gather(next_logits, 1, cb)
            correct    += (cand_logits.argmax(dim=1) == 0).sum().item()
        results[d] = correct / ids.size(0)
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF D512-L8 FA@L2 — MIXED-DOMAIN HYPERPARAMETER VALIDATION')
    print(f'  LR={LR}  LR_MULT={SCALE_EMBED_LR_MULT}  EMA_INIT={EMA_INIT}')
    print(f'  (vs FineWeb baseline: LR=3e-4, LR_MULT=15.0, EMA_INIT=0.0208)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Liger CE: {"enabled" if USE_LIGER_CE else "disabled"}')
    print(f'  AdamW8bit: {"enabled" if _BNB_AVAILABLE else "disabled (standard AdamW)"}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}')
    print(f'  scale_embed_init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  Batch: BS={BATCH_SIZE}×GRAD_ACCUM={GRAD_ACCUM}=eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  git={git_hash}')

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Tokenizer: {tok_path}  (vocab={tokenizer.vocab_size():,})')

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.exists(MIXED_DATASET_PATH):
        raise FileNotFoundError(f'Mixed dataset not found: {MIXED_DATASET_PATH}')
    print(f'Loading mixed dataset from {MIXED_DATASET_PATH}')
    cache      = torch.load(MIXED_DATASET_PATH, weights_only=True)
    train_data = cache['train'].long()
    val_data   = cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DWARFModel(
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
    print(f'Parameters: {n_params:,} ({n_params/1e6:.1f}M)')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    _AdamW   = bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW
    optimizer = _AdamW([
        {'params': list(model.non_scale_embed_params()), 'lr': LR},
        {'params': list(model.scale_embed_params()),     'lr': LR * SCALE_EMBED_LR_MULT},
        {'params': list(model.phase_params()),           'lr': LR * 50, 'name': 'phase'},
    ], weight_decay=0.1, betas=(0.9, 0.95))
    print(f'  phase params LR: {LR * 50:.2e} (50× base)')

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}
    loss_history    = []  # track for variance comparison

    # ── Kernel warmup ─────────────────────────────────────────────────────────
    print(f'Warming up Triton kernels (BS={BATCH_SIZE})...')
    _wb = min(BATCH_SIZE, len(train_data))
    _wx = train_data[:_wb, :-1].to(device)
    _wy = train_data[:_wb,  1:].to(device)
    with _amp_context(device):
        if USE_LIGER_CE:
            _wh = model.forward_hidden(_wx)
            _wl = liger_ce_fn(_wh.view(-1, _wh.size(-1)), model.out.weight, _wy.view(-1))
            _wl.backward()
            del _wh, _wl
        else:
            _wo = model(_wx)
            _wf = _wo.reshape(-1, _wo.size(-1))
            _wyf = _wy.reshape(-1)
            _wloss = F.cross_entropy(_wf, _wyf, reduction='mean')
            _wloss.backward()
            del _wo, _wf, _wyf, _wloss
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy
    torch.cuda.synchronize()
    print('  Warmup complete.')

    # ── Main training loop ────────────────────────────────────────────────────
    gpu_peak_flops  = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step  = 6 * n_params * tokens_per_step
    mfu_window      = deque(maxlen=20)
    percolation_logged = False

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        epoch_losses    = []

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
                            hidden.contiguous().reshape(-1, hidden.size(-1)),
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
                    total_ce    = 0.0
                    for cs in range(0, T, CE_CHUNK):
                        ce    = min(cs + CE_CHUNK, T)
                        chunk = logits_flat[cs:ce].detach().requires_grad_(True)
                        cl    = F.cross_entropy(chunk, y_flat[cs:ce], reduction='sum')
                        cl.backward()
                        grad_logits[cs:ce] = chunk.grad
                        total_ce += cl.item()
                    logits_flat.backward(grad_logits / (T * GRAD_ACCUM))
                    loss_val = total_ce / T
                    del logits, logits_flat, y_flat, grad_logits

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            t1.record()
            torch.cuda.synchronize()
            step_ms = t0.elapsed_time(t1)
            mfu_window.append(step_ms)
            step += 1
            loss_history.append(loss_val)
            epoch_losses.append(loss_val)

            # Log percolation crossing
            se_mean, se_max = model.se_stats()
            if not percolation_logged and se_max >= 2.0:
                print(f'\n  🎯 PERCOLATION THRESHOLD CROSSED: '
                      f'EP{epoch} STEP {step} — SE|max|={se_max:.4f}',
                      flush=True)
                percolation_logged = True

            if step % 200 == 0:
                # Compute loss std dev for last 200 steps (gradient competition metric)
                recent = loss_history[-200:]
                loss_std = (sum((x - sum(recent)/len(recent))**2
                               for x in recent) / len(recent)) ** 0.5

                avg_ms  = sum(mfu_window) / len(mfu_window)
                tok_s   = tokens_per_step / (avg_ms / 1000.0)
                mfu_str = ''
                if gpu_peak_flops:
                    mfu = (flops_per_step / (avg_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'

                perc_marker = ''
                if se_max >= 2.0:
                    perc_marker = ' ✓'
                elif se_max >= 1.6:
                    perc_marker = ' ↑'

                print(
                    f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                    f'| Loss {loss_val:.4f} (σ={loss_std:.3f}) '
                    f'| SE|max|={se_max:.4f}{perc_marker}'
                    f'{mfu_str} | {tok_s:.0f} tok/s',
                    flush=True)

        # ── End-of-epoch eval ─────────────────────────────────────────────────
        import statistics
        epoch_std = statistics.stdev(epoch_losses) if len(epoch_losses) > 1 else 0.0

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, 'd512_l8_mixed_hyp_best.pt'))
            marker = ' *'

        torch.save({
            'epoch':               epoch,
            'model_state_dict':    {k.replace('._orig_mod', ''): v
                                    for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':            val_loss,
            'val_ppl':             val_ppl,
            'config': {
                'LR': LR, 'LR_MULT': SCALE_EMBED_LR_MULT, 'EMA_INIT': EMA_INIT,
                'D': EMBEDDING_DIM, 'H': NUM_HEADS, 'L': NUM_LAYERS,
                'FFN': FFN_DIM, 'FA_layer': FULL_ATTN_LAYER,
            },
        }, os.path.join(CHECKPOINT_DIR, f'd512_l8_mixed_hyp_ep{epoch}.pt'))

        se_mean, se_max = model.se_stats()
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker} '
              f'| Loss σ={epoch_std:.4f} (target: <0.06)')
        print(f'  SE|mean|={se_mean:.4f} |max|={se_max:.4f}'
              + (' ✓ above threshold' if se_max >= 2.0 else ' ✗ below threshold'))
        print(f'  EMA: {model.ema_summary()}')
        print(f'  Passkey mean={pk_mean*100:.1f}%')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        if epoch >= 2 and pk_mean < PASSKEY_ABORT_THRESHOLD:
            print(f'\n  ⛔ ABORT: passkey {pk_mean*100:.1f}% < {PASSKEY_ABORT_THRESHOLD*100:.0f}%')
            break

        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_s = time.time() - t_start
    mem_mb    = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    pk_final  = passkey_results.get(max(passkey_results, default=1), 0.0)
    ppl_final = ppl_results.get(max(ppl_results, default=1), 999.0)

    global_loss_std = (sum((x - sum(loss_history)/len(loss_history))**2
                           for x in loss_history) / max(len(loss_history),1)) ** 0.5

    print('\n--- RESULTS ---')
    print(f'LR={LR}  LR_MULT={SCALE_EMBED_LR_MULT}  EMA_INIT={EMA_INIT}')
    print(f'percolation_step: {"crossed" if percolation_logged else "NOT CROSSED"}')
    print(f'global_loss_std: {global_loss_std:.4f}  (H3 baseline: 0.098, FW: 0.038, target: <0.06)')
    for ep in sorted(passkey_results):
        print(f'passkey_ep{ep}: {passkey_results[ep]:.1f}%')
    for ep in sorted(ppl_results):
        print(f'ppl_ep{ep}: {ppl_results[ep]:.2f}')
    print(f'memory_mb: {mem_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'description: D512-L8-FA2 MIXED-HYP D={EMBEDDING_DIM} H={NUM_HEADS} '
          f'L={NUM_LAYERS} FFN={FFN_DIM} FA@L{FULL_ATTN_LAYER} '
          f'LR={LR} LR_MULT={SCALE_EMBED_LR_MULT} EMA_INIT={EMA_INIT}')


if __name__ == '__main__':
    train()
