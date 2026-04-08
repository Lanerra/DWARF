"""
🧪 DWARF Frozen-FA Probe — 13M/34M, 4090, frozen FA from Moonshot-58M ep2

Architecture: D=512, H=8 (hd=64), L=8, FFN=2048, J=24 (se015 offsets)
  L0:  DSQGBlockV6Physics  IF=False  ← pure DSQG relay
  L1:  DSQGBlockV6Physics  IF=True   ← preIF (single layer before FA)
  L2:  FullAttentionBlock            ← FA@L2 — FROZEN from moonshot_58m_ep2_full_attn.pt
  L3-7: DSQGBlockV6Physics IF=False  ← post-FA relay layers

Key difference vs moonshot: FA is FROZEN from day 1. DSQG layers cold-start and
learn to route into a fixed, already-trained retrieval block. Tests whether
co-adaptive training is required or if a pre-trained FA is good enough.

Config:
  - Tokenizer: fineweb_tokenizer_8k.json  (8K BPE, trained on 100K FineWeb-Edu cache)
                EOS id = 0  (<|endoftext|>)
  - Dataset:   fineweb_edu_encoded_2048_8k.pt  (~87K train seqs, 178M tokens)
               pre-encoded with fineweb_tokenizer_8k
  - Vocab:     8000  (reduces embedding/lm_head from 16.4M → 4.1M params each)
  - FA Donor:  autoresearch/checkpoints/moonshot_58m_ep2_full_attn.pt
  - EMA_INIT = 0.0208 (= 1/48 = 1/δ_relay_min; J24D validated 0.6% error)
  - SCALE_EMBED_INIT = 0.1, LR_MULT = 15.0
  - Batch: BS=16 × GRAD_ACCUM=8 → eff_batch=128
  - ~87K train seqs (30% Chinchilla for 33.7M tied-weight params) — probe budget

EMA_INIT RULE: Always derive from offset set before training.
  For J24D (se015): empirically validated 1/δ_relay_min = 1/48 = 0.0208

Hypothesis: DSQG layers can learn to route into a frozen pre-trained FA,
skipping the co-adaptive phase transition and potentially reaching good
passkey performance faster than a cold-start run.

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_frozen_fa_probe_13m_4090_bf16.py \\
    > logs/run_frozen_fa_probe_13m.log 2>&1 &
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

# Probe budget: use all available 87K seqs (30% Chinchilla for ~33.7M tied params)
MAX_TRAIN_SEQS      = 87_484
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# Empirically validated for J24D: se015 trains to α≈0.0207, error 0.6%
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

VOCAB_SIZE  = 8000
BATCH_SIZE  = 16
GRAD_ACCUM  = 8    # effective batch = 128
MAX_SEQ_LEN = 2048
MAX_VAL_SEQS = 2048

DATASET_PATH   = 'logs/fineweb_edu_encoded_2048_8k.pt'
TOKENIZER_PATH = 'results/fineweb_tokenizer_8k.json'
FA_DONOR_PATH  = 'autoresearch/checkpoints/moonshot_58m_ep2_full_attn.pt'
CHECKPOINT_DIR = 'autoresearch/checkpoints'

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

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

assert len(OFFSETS) == 24


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
                 interference_interval=None,   # unused; kept for compat
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed       = nn.Embedding(seq_len + 2, embedding_dim)
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
        # Weight tying: lm_head shares embedding weights
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

    def load_frozen_fa(self, checkpoint_path):
        """Load FA weights from donor checkpoint and freeze them.

        Donor keys look like: blocks.2._orig_mod.<subkey>  (torch.compile artifact)
        Target keys look like: blocks.2.<subkey>
        Strips the _orig_mod prefix automatically.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        donor_sd = ckpt['full_attn_block']
        donor_config = ckpt.get('config', {})
        print(f'  FA donor: {checkpoint_path}')
        print(f'  Donor config: D={donor_config.get("embedding_dim")}, '
              f'H={donor_config.get("num_heads")}, epoch={donor_config.get("epoch")}')

        # Strip _orig_mod prefix (torch.compile artifact) and blocks.N. prefix
        target_layer = self.full_attn_layer
        target_block = self.blocks[target_layer]
        target_sd    = target_block.state_dict()

        remapped = {}
        for k, v in donor_sd.items():
            # k = "blocks.2._orig_mod.norm1.weight" → strip "blocks.2._orig_mod."
            prefix = f'blocks.{target_layer}._orig_mod.'
            prefix_noc = f'blocks.{target_layer}.'
            if k.startswith(prefix):
                new_k = k[len(prefix):]
            elif k.startswith(prefix_noc):
                new_k = k[len(prefix_noc):]
            else:
                print(f'  WARNING: unexpected key {k}, skipping')
                continue
            remapped[new_k] = v

        missing   = set(target_sd.keys()) - set(remapped.keys())
        unexpected = set(remapped.keys()) - set(target_sd.keys())
        if missing:
            print(f'  WARNING: missing FA keys: {missing}')
        if unexpected:
            print(f'  WARNING: unexpected FA keys: {unexpected}')

        target_block.load_state_dict(remapped, strict=True)
        print(f'  Loaded {len(remapped)} FA weight tensors')

        # Freeze all FA parameters
        for p in target_block.parameters():
            p.requires_grad_(False)
        frozen_count = sum(p.numel() for p in target_block.parameters())
        print(f'  Frozen {frozen_count:,} FA parameters ({frozen_count/1e6:.2f}M)')

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
    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)


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


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF Frozen-FA Probe — 13M/34M, D=512 H=8 L=8 J=24')
    print('  FA@L2 FROZEN from moonshot_58m_ep2. DSQG cold-start.')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))
    print(f'  Tokenizer: {TOKENIZER_PATH} ({tokenizer.vocab_size()} vocab)')
    eos_id = tokenizer.token_to_id('<|endoftext|>')
    print(f'  EOS id: {eos_id}')

    # Load dataset
    print(f'Loading dataset from {DATASET_PATH}')
    cache      = torch.load(DATASET_PATH, weights_only=True)
    train_data = cache['train'].long()
    val_data   = cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    # Build model
    model = AutoresearchTransformerPhysics(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    # Load and freeze FA from donor checkpoint
    model.load_frozen_fa(FA_DONOR_PATH)

    n_total     = model.param_count()
    n_trainable = model.trainable_param_count()
    n_frozen    = n_total - n_trainable
    print(f'Parameters: {n_total/1e6:.2f}M total | {n_trainable/1e6:.2f}M trainable | {n_frozen/1e6:.2f}M frozen FA')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}, V={VOCAB_SIZE}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t)')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS} ({100*MAX_TRAIN_SEQS/(n_trainable*20/2048):.0f}% Chinchilla for trainable params)')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  git={git_hash}')

    # Optimizer — only trainable (non-FA) parameters
    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_trainable_parameters())
    print(f'  Optimizer groups: {len(non_scale_embed_params)} non-SE params, {len(scale_embed_params)} SE params')

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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_ckpt_name  = 'frozen_fa_probe_13m_best.pt'

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        # Keep FA in eval mode (frozen — no dropout, no grad)
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

            if step % 50 == 0:
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item() * GRAD_ACCUM:.4f}')
                sys.stdout.flush()

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, best_ckpt_name))
            marker = ' *'

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

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

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
    print(f'num_params_M: {n_total / 1e6:.2f}')
    print(f'num_trainable_M: {n_trainable / 1e6:.2f}')
    print(f'num_frozen_M: {n_frozen / 1e6:.2f}')
    print(f'num_layers: {NUM_LAYERS}')
    print(f'num_offsets: {len(OFFSETS)}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init: {EMA_INIT}')
    print(f'description: Frozen-FA Probe 13M — D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} '
          f'FFN={FFN_DIM} J=24 se015, FA frozen from moonshot_58m_ep2, V=8K, cold DSQG')


if __name__ == '__main__':
    train()
