"""
DWARF 24M J24-Ramanujan screen — 4090, cold-start

Architecture: D=512, H=8 (hd=64), L=8, FFN=1024, J=24 (J24-Ram v1 offsets)
  L0:  DSQGBlockV6Physics  IF=False  ← pure DSQG relay
  L1:  DSQGBlockV6Physics  IF=True   ← preIF (single layer before FA)
  L2:  FullAttentionBlock            ← FA@L2
  L3-7: DSQGBlockV6Physics IF=False  ← post-FA relay

Vocab: 8K (fineweb_tokenizer_8k.json)
Tied lm_head.
~24.9M parameters.

J24-Ramanujan v1 offsets: se015 minus {1,2}, plus {69,1536}
  [3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,69,96,192,384,512,768,1024,1536]
  - Gains 12/12 2-hop passkey coverage  (vs se015's 9/12)
  - Spectral gap: 0.2638                (vs se015's 0.2325, +13.5%)
  - Key paths:
      d=128: gap=133 = 64+69       (was 3-hop in se015)
      d=256: gap=261 = 192+69      (was 3-hop in se015)
      d=1536: gap=1541 = 5+1536    (was 3-hop in se015)

Dataset: fineweb_edu_encoded_2048_8k.pt (87,484 train seqs, 8K tokenizer)
         Already present from frozen-FA probe work (Mar 21).

EMA_INIT = 0.0208 = 1/48 = 1/δ_relay_min
  J24-Ram v1 shares the same δ_relay_min=48 as se015 (first sparse offset unchanged).

Screen: 3 epochs × 87K seqs — directional signal on passkey by distance.
        Full Chinchilla for 24.9M = ~243K seqs.

Comparison target: Moonshot-58M ep2 (35.04 PPL / 99.2% passkey / ar_score=80.90)
Key question: does J24-Ram v1 improve d=128, d=256, d=1536 passkey vs se015?

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_24m_j24ram_4090_bf16.py \\
    > logs/run_24m_j24ram.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

# J24-Ramanujan v1: se015 minus {1,2}, plus {69,1536}
# Verified via Rust: 12/12 2-hop, norm_gap=0.2638 vs se015's 0.2325
OFFSETS = [3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
           48, 64, 69, 96, 192, 384, 512, 768, 1024, 1536]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8        # hd = 512/8 = 64
FFN_DIM          = 1024     # 2×D (relay carries memory load; FFN is nonlinear transform only)
NUM_LAYERS       = 8
FULL_ATTN_LAYER  = 2
VOCAB_SIZE       = 8000     # fineweb_tokenizer_8k.json (actual vocab size)

# Dataset: pre-encoded 8K-tokenized, 87K train seqs from 100K doc cache
DATASET_PATH     = 'logs/fineweb_edu_encoded_2048_8k.pt'
TOKENIZER_PATH   = 'results/fineweb_tokenizer_8k.json'

# Chinchilla-optimal for 24.9M: 20 × 24.9M / 2048 = 243K seqs
# Using 87K (available dataset) — ~36% Chinchilla, screen-level
MAX_TRAIN_SEQS   = 87_000
MAX_VAL_SEQS     = 2_048

SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# J24-Ram v1 retains δ_relay_min=48 (same as se015 — sparse offsets start at 48)
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3
BATCH_SIZE    = 16
GRAD_ACCUM    = 8    # effective batch = 128
MAX_SEQ_LEN   = 2048

CHECKPOINT_DIR = 'autoresearch/checkpoints'

# Resume from ep1 best checkpoint — skip epochs already completed
# Set to 1 to train from scratch, 2 to resume after ep1, etc.
START_EPOCH = 3
RESUME_CKPT = 'autoresearch/checkpoints/j24ram_24m_ep2_resume.pt'  # load if START_EPOCH > 1

# ar_score baselines (autoresearch reference: J16D fulldata pure champion)
PPL_BASELINE     = 61.75
PASSKEY_BASELINE = 18.3

PASSKEY_DISTANCES  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50   # n=50: ±7pp noise vs ±20pp at n=20 (inflation risk)
PASSKEY_BATCH_SIZE = 32
_PASSKEY_WORDS     = ['apple', 'banana', 'orange', 'cherry', 'grape',
                      'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE   = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE    = 'the secret word is {word} .'
_RETRIEVAL_CUE     = 'the secret word is'

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
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
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionV6
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

assert len(OFFSETS) == 24, f"Expected 24 offsets, got {len(OFFSETS)}"


def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ── Physics helpers ───────────────────────────────────────────────────────────

def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)


def _kdv_correction(pool, kdv_alpha):
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    return pool + alpha * pool * (pool - pool_prev)


def _agc_normalize(pool, eps=1e-6):
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Model blocks ─────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.0):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV6Physics(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.0, interference=False):
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
            self.kdv_alpha    = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR)
            pool = _kdv_correction(pool, self.kdv_alpha)
            pool = _agc_normalize(pool)
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = inter.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            k_delta = self.inter_k_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            kv_inject = (k_delta, v_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
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
        return F.dropout(self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.0):
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
                 ffn_dim, seq_len, full_attn_layer, scale_embed_init_val=0.0,
                 dropout=0.0):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(embedding_dim, num_heads, ffn_dim, dropout))
            else:
                has_if = (i == full_attn_layer - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        # Tied lm_head (saves ~4.19M params vs untied at V=8K, D=512)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

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
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

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

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                kdv   = block.kdv_alpha.item()
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t) kdv={kdv:.4f}')
        return '  '.join(entries)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, data, device, batch_size=8):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(data) - batch_size + 1, batch_size):
            x = data[i:i+batch_size, :-1].to(device, non_blocking=True)
            y = data[i:i+batch_size,  1:].to(device, non_blocking=True)
            logits = model(x)
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss   += loss.item() * y.numel()
            total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    pad_id     = tokenizer.token_to_id('<|endoftext|>') or 0

    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word) or tokenizer.encode(word)
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

        ids   = torch.tensor(seqs,  dtype=torch.long, device=device)
        pos   = torch.tensor(last_pos, dtype=torch.long, device=device)
        cand  = torch.tensor(cand_rows, dtype=torch.long, device=device)
        correct, total = 0, ids.size(0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, total, PASSKEY_BATCH_SIZE):
                ids_b  = ids[start:start+PASSKEY_BATCH_SIZE]
                pos_b  = pos[start:start+PASSKEY_BATCH_SIZE]
                cand_b = cand[start:start+PASSKEY_BATCH_SIZE]
                logits = model(ids_b)
                row    = torch.arange(ids_b.size(0), device=device)
                cand_logits = torch.gather(logits[row, pos_b], 1, cand_b)
                correct += (cand_logits.argmax(dim=1) == 0).sum().item()

        results[d] = correct / total
    return results


def save_checkpoint(model, name):
    clean = {k.replace('._orig_mod', ''): v for k, v in model.state_dict().items()}
    torch.save(clean, os.path.join(CHECKPOINT_DIR, name))


def save_full_attn_checkpoint(model, epoch, git_hash):
    block = model.blocks[model.full_attn_layer]
    state = {f"blocks.{model.full_attn_layer}.{n}": p.data.clone()
             for n, p in block.named_parameters()}
    payload = {
        "full_attn_block": state,
        "config": {
            "embedding_dim":  EMBEDDING_DIM,
            "num_heads":      NUM_HEADS,
            "ffn_dim":        FFN_DIM,
            "num_layers":     NUM_LAYERS,
            "num_offsets":    len(OFFSETS),
            "offsets":        OFFSETS,
            "vocab_size":     VOCAB_SIZE,
            "epoch":          epoch,
            "git_hash":       git_hash,
            "note": (
                f"J24-Ram 24M: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} "
                f"FFN={FFN_DIM} J=24-Ram-v1 FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1}. "
                f"8K vocab tied. Epoch {epoch}/3. fineweb_tokenizer_8k."
            ),
        },
    }
    path = os.path.join(CHECKPOINT_DIR, f"j24ram_24m_ep{epoch}_full_attn.pt")
    torch.save(payload, path)
    print(f"  Saved FullAttn checkpoint: {path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start  = time.time()
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  DWARF 24M J24-Ram v1 screen — D=512 H=8 L=8 FFN=1024 V=8K tied')
    print('  J24-Ram v1: se015 - {1,2} + {69,1536} | 12/12 2-hop | gap=0.2638')
    print('  FA@L2, preIF@L1, fineweb_tokenizer_8k, EMA_INIT=1/δ_relay_min=0.0208')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        _cc = torch.cuda.get_device_capability()
        _path = 'sm_89 (4090 Ada — tuned)' if (_cc[0]==8 and _cc[1]==9) else f'sm_{_cc[0]}{_cc[1]}'
        print(f'  Kernel path: {_path}')
    print(f'  Offsets ({len(OFFSETS)}): {OFFSETS}')
    print(f'  FFN={FFN_DIM} (2×D — relay carries memory load)')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS} (~36% Chinchilla for 24.9M)')
    print(f'  git={git_hash}')

    # ── Load tokenizer ──────────────────────────────────────────────────────
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_PATH}')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f'  Tokenizer: {TOKENIZER_PATH} (vocab={tokenizer.get_vocab_size()})')
    assert tokenizer.get_vocab_size() == VOCAB_SIZE, \
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != VOCAB_SIZE {VOCAB_SIZE}"

    # Wrap for convenience in passkey eval
    class _TokWrap:
        def __init__(self, t): self._t = t
        def encode(self, s): return self._t.encode(s).ids
        def token_to_id(self, s): return self._t.token_to_id(s)
        def vocab_size(self): return self._t.get_vocab_size()
    tok = _TokWrap(tokenizer)

    # ── Load dataset ────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f'Dataset not found: {DATASET_PATH}\n'
            f'Re-encode with fineweb_tokenizer_8k.json from the 100K doc cache.')
    print(f'  Loading dataset: {DATASET_PATH}')
    _cache     = torch.load(DATASET_PATH, weights_only=True)
    train_data = _cache['train'].long()
    val_data   = _cache['val'].long()
    print(f'  Raw: train={len(train_data):,}  val={len(val_data):,} seqs')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  Using: train={len(train_data):,}  val={len(val_data):,} seqs')

    # ── Build model ─────────────────────────────────────────────────────────
    model = DWARFModel(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL, dropout=0.0,
    ).to(device)

    n_params = model.param_count()
    print(f'\n  Parameters: {n_params:,} ({n_params/1e6:.1f}M)')

    # ── Resume from checkpoint if requested ─────────────────────────────────
    if START_EPOCH > 1:
        if not os.path.exists(RESUME_CKPT):
            raise FileNotFoundError(f'Resume checkpoint not found: {RESUME_CKPT}')
        state = torch.load(RESUME_CKPT, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f'  Resumed from {RESUME_CKPT} (starting at epoch {START_EPOCH})')

    # ── Warmup (forward only, single batch, avoids first-backward surprise) ─
    print('  kernel warmup...', end=' ', flush=True)
    _wx = torch.zeros(1, 64, dtype=torch.long, device=device)
    with _amp_context(device):
        _ = model(_wx)
    del _wx
    torch.cuda.reset_peak_memory_stats()
    print(f'done (peak during warmup: {torch.cuda.max_memory_allocated()//1024//1024} MB)')

    # ── Optimizer ───────────────────────────────────────────────────────────
    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params,     'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    total_steps     = SCREEN_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    # Fast-forward scheduler to match epochs already completed
    if START_EPOCH > 1:
        completed_steps = (START_EPOCH - 1) * steps_per_epoch
        for _ in range(completed_steps):
            scheduler.step()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}

    for epoch in range(START_EPOCH, SCREEN_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        optimizer.zero_grad(set_to_none=True)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x = batch[:, :-1].to(device, non_blocking=True)
                y = batch[:, 1:].to(device, non_blocking=True)
                with _amp_context(device):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if (acc_step + 1) % 200 == 0:
                print(f'  Step {acc_step+1}/{steps_per_epoch} '
                      f'| Loss {loss.item() * GRAD_ACCUM:.4f}', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, 'j24ram_24m_best.pt')
            save_checkpoint(model, f'j24ram_24m_ep{epoch}_resume.pt')
            marker = ' *'

        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules() if isinstance(m, DSQGAttentionV6)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')
        save_full_attn_checkpoint(model, epoch, git_hash)

        pk      = passkey_accuracy(model, tok, device)
        pk_mean = sum(pk.values()) / len(pk) * 100
        passkey_results[epoch] = pk_mean
        print(f'  Passkey mean={pk_mean:.1f}%')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6

    ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ar_score = (passkey_final - PASSKEY_BASELINE) - max(0, ppl_final - PPL_BASELINE) * 0.5

    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score: {ar_score:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_params/1e6:.1f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'num_offsets: {len(OFFSETS)}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init: {EMA_INIT}')
    print(f'description: J24-Ram 24M — D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} '
          f'FFN={FFN_DIM} J=24-Ram-v1 8K-tied. se015-{{1,2}}+{{69,1536}}. '
          f'12/12 2-hop. Screen on 87K seqs.')


if __name__ == '__main__':
    train()
