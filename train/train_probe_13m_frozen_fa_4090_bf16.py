"""
🧪 DWARF Frozen-FA Probe — 37M DSQG + frozen moonshot-58M FA donor

Architecture: D=512, H=8 (hd=64), L=8, FFN=2048, J=24 (se015 offsets), V=32K
  L0:  DSQGBlockV6Physics  IF=False  ← trainable
  L1:  DSQGBlockV6Physics  IF=True   ← trainable (preIF)
  L2:  FullAttentionBlock            ← FROZEN (loaded from moonshot_58m_ep2_full_attn.pt)
  L3-7: DSQGBlockV6Physics IF=False  ← trainable

Hypothesis: FA trained on moonshot-58M has learned strong retrieval (99.2% passkey).
Can a fresh DSQG stack learn to co-adapt against a frozen FA that already knows
how to retrieve? If yes: relay mechanism generalizes across DSQG initializations.
If no: co-adaptation requires joint training from scratch.

Config:
  - Tokenizer: fineweb_tokenizer_32k.json  (32K BPE, FineWeb-Edu proper)
               EOS id = 0  (<|endoftext|>)
  - Dataset:   logs/fineweb_edu_encoded_2048_v2.pt (~2M seqs, 1M docs)
  - FA donor:  autoresearch/checkpoints/moonshot_58m_ep2_full_attn.pt
               (moonshot-58M ep2: PPL=35.04, passkey=99.2%, ar_score=80.90)
               FA is vocab-agnostic (D=512 projections only, no embedding/lm_head)
  - EMA_INIT = 0.0208 (= 1/48 = 1/δ_relay_min; validated for J24D, 0.6% error)
  - SCALE_EMBED_INIT = 0.1, LR_MULT = 15.0
  - Batch: BS=16 × GRAD_ACCUM=8 → eff_batch=128
  - ~37.4M total params; FA frozen → ~34.0M trainable
  - MAX_TRAIN_SEQS=121K (probe budget — same as autoresearch standard)

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_probe_13m_frozen_fa_4090_bf16.py \\
    > logs/run_probe_frozen_fa_32k.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8          # hd = 512/8 = 64
FFN_DIM          = 2048
NUM_LAYERS       = 8
FULL_ATTN_LAYER  = 2

VOCAB_SIZE       = 32000

# Chinchilla-optimal: 20 × 37.4M params / 2048 tokens/seq ≈ 365K seqs
MAX_TRAIN_SEQS      = 121_232
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# Same J24D offset set as moonshot; validated for se015 to 0.6% error
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

# FA donor checkpoint (moonshot-58M ep2)
FA_DONOR_CKPT = 'autoresearch/checkpoints/moonshot_58m_ep2_full_attn.pt'

# Dataset paths
ENCODED_DATASET = 'logs/fineweb_edu_encoded_2048_v2.pt'
TOKENIZER_PATH  = 'results/fineweb_tokenizer_32k.json'

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

CHECKPOINT_DIR = 'autoresearch/checkpoints'
BEST_CKPT_NAME = 'probe_frozen_fa_32k_best.pt'
RESUME_CKPT    = 'autoresearch/checkpoints/probe_frozen_fa_32k_resume.pt'

MAX_SEQ_LEN  = 2048
MAX_VAL_SEQS = 2048
BATCH_SIZE   = 16
GRAD_ACCUM   = 8   # eff_batch = 128

# =============================================================================

import json, math, os, subprocess, sys, time
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

from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionV6, npci_rotate
from causal_ema_scan import causal_ema_scan as _causal_ema_scan


# ── Physics helpers (unchanged from moonshot) ─────────────────────────────────

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


# ── Model (identical to moonshot except vocab size) ───────────────────────────

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
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))
            self.kdv_alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)
            pool = _kdv_correction(pool, self.kdv_alpha)
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
        return F.dropout(self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
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
        self.pos_embed       = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop            = nn.Dropout(dropout)
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
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        # Note: untied lm_head for 8K vocab (embedding is small, saves little to tie)
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

    def load_frozen_fa(self, ckpt_path):
        """Load FA weights from donor checkpoint and freeze them."""
        ck = torch.load(ckpt_path, map_location='cpu')
        raw_sd = ck['full_attn_block']

        # Remap keys: strip 'blocks.N._orig_mod.' or 'blocks.N.' prefix
        # (torch.compile adds _orig_mod; loading into non-compiled model)
        prefix_variants = [
            f'blocks.{self.full_attn_layer}._orig_mod.',
            f'blocks.{self.full_attn_layer}.',
        ]
        clean_sd = {}
        for k, v in raw_sd.items():
            remapped = k
            for pfx in prefix_variants:
                if k.startswith(pfx):
                    remapped = k[len(pfx):]
                    break
            clean_sd[remapped] = v

        fa_block = self.blocks[self.full_attn_layer]
        missing, unexpected = fa_block.load_state_dict(clean_sd, strict=True)
        if missing:
            raise RuntimeError(f'Missing keys in FA donor: {missing}')
        if unexpected:
            print(f'  Warning: unexpected keys in FA donor: {unexpected}')

        # Freeze FA — no gradients, no optimizer updates
        for p in fa_block.parameters():
            p.requires_grad_(False)

        n_frozen = sum(p.numel() for p in fa_block.parameters())
        print(f'  FA donor loaded from {ckpt_path}')
        print(f'  FA frozen: {n_frozen:,} params ({n_frozen/1e6:.2f}M)')
        return ck['config']

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            if self.training:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                if m.scale_embed.requires_grad:
                    yield m.scale_embed

    def non_scale_embed_trainable_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in se_ids:
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


# ── Data & eval utilities ──────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    def token_to_id(self, tok):
        return self.tokenizer.token_to_id(tok)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device)
        y = data[i:i+BATCH_SIZE,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for d in PASSKEY_DISTANCES:
        correct, n_valid = 0, 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available:
                continue
            filler   = []
            while len(filler) < d:
                filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target] + others[:9])[
                            logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


def save_resume_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                           ppl_results, passkey_results):
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss':    best_val_loss,
        'ppl_results':      ppl_results,
        'passkey_results':  passkey_results,
    }, RESUME_CKPT)
    print(f'  Saved resume checkpoint: {RESUME_CKPT}')


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF Frozen-FA Probe — 37M DSQG + frozen moonshot-58M FA')
    print('  D=512 H=8 hd=64 L=8 J=24 V=32K  FA@L2 frozen (ep2 donor)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # Load dataset
    if not os.path.exists(ENCODED_DATASET):
        raise FileNotFoundError(f'Encoded dataset not found: {ENCODED_DATASET}')
    print(f'Loading pre-encoded dataset: {ENCODED_DATASET}')
    _cache     = torch.load(ENCODED_DATASET, weights_only=True)
    train_data = _cache['train'].long()
    val_data   = _cache['val'].long()
    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs  (V=32K)')

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))
    assert tokenizer.vocab_size() == VOCAB_SIZE, \
        f'Vocab mismatch: expected {VOCAB_SIZE}, got {tokenizer.vocab_size()}'
    eos_id = tokenizer.token_to_id('<|endoftext|>')
    print(f'  Tokenizer: {TOKENIZER_PATH}  vocab={VOCAB_SIZE}  EOS={eos_id}')

    # Build model
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

    # Load and freeze FA donor
    donor_config = model.load_frozen_fa(FA_DONOR_CKPT)
    print(f'  Donor: {donor_config.get("note", FA_DONOR_CKPT)}')

    n_total     = model.param_count()
    n_trainable = model.trainable_param_count()
    n_frozen    = n_total - n_trainable
    print(f'  Total params:     {n_total:,} ({n_total/1e6:.2f}M)')
    print(f'  Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)')
    print(f'  Frozen (FA):      {n_frozen:,} ({n_frozen/1e6:.2f}M)')

    # Check for resume checkpoint
    start_epoch     = 1
    best_val_loss   = float('inf')
    ppl_results     = {}
    passkey_results = {}

    if os.path.exists(RESUME_CKPT):
        print(f'\n  Resuming from {RESUME_CKPT}')
        ck = torch.load(RESUME_CKPT, map_location=device)
        # Load model weights (FA will be overwritten then re-frozen)
        model.load_state_dict(ck['model_state_dict'])
        # Re-freeze FA (load_state_dict overwrites requires_grad)
        for p in model.blocks[FULL_ATTN_LAYER].parameters():
            p.requires_grad_(False)
        start_epoch   = ck['epoch'] + 1
        best_val_loss = ck['best_val_loss']
        ppl_results   = ck['ppl_results']
        passkey_results = ck['passkey_results']
        print(f'  Resuming from epoch {start_epoch}')
    else:
        print(f'\n  Cold start for DSQG stack; FA loaded from donor.')

    # Optimizer — trainable params only
    scale_embed_params = list(model.scale_embed_parameters())
    other_params       = list(model.non_scale_embed_trainable_parameters())
    print(f'  scale_embed params: {sum(p.numel() for p in scale_embed_params):,}')
    print(f'  other trainable:    {sum(p.numel() for p in other_params):,}')

    optimizer = (bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW)([
        {'params': other_params,       'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if os.path.exists(RESUME_CKPT) and 'optimizer_state_dict' in ck:
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        print('  Optimizer + scheduler state restored.')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f'\n  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
    print(f'  git={git_hash}')

    for epoch in range(start_epoch, SCREEN_EPOCHS + 1):
        model.train()
        # FA block stays frozen even in train() mode
        model.blocks[FULL_ATTN_LAYER].eval()

        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')
                sys.stdout.flush()

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_CKPT_NAME))
            marker = ' *'

        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionV6):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} |max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

        save_resume_checkpoint(model, optimizer, scheduler, epoch,
                               best_val_loss, ppl_results, passkey_results)

    elapsed_s     = time.time() - t_start
    memory_mb     = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
    PPL_BASELINE     = 61.75
    PASSKEY_BASELINE = 18.3
    ar_score = (passkey_final - PASSKEY_BASELINE) - max(0, ppl_final - PPL_BASELINE) * 0.5

    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}: {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}: {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score: {ar_score:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'num_params_M: {n_total/1e6:.2f}')
    print(f'num_trainable_M: {n_trainable/1e6:.2f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'num_offsets: {len(OFFSETS)}')
    print(f'vocab_size: {VOCAB_SIZE}')
    print(f'fa_frozen: True')
    print(f'fa_donor: moonshot_58m_ep2 (PPL=35.04, passkey=99.2%)')
    print(f'description: Frozen-FA probe — 32K vocab, L=8 D=512 H=8 J=24, FA@L2 frozen from moonshot-58M ep2, Chinchilla 365K seqs')


if __name__ == '__main__':
    train()
