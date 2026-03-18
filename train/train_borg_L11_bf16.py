"""
🔬 DWARF Autoresearch — Borg Adaptation Run (FullAttn-first, 13M)

Step 2 of Borg Collective experiment: train a FullAttn-first model where the
FullAttn block is warm-started from extracted J26D weights, adapted to receive
raw embeddings at L0 instead of processed DSQG residuals.

Architecture (D=512, H=8, L=3, FFN=2048 → ~15M params):
  Block 0:  FullAttentionBlock  ← warm-started from J26D extracted weights
  Block 1:  DSQGBlockV6Physics  ← randomly initialized
  Block 2:  DSQGBlockV6Physics  ← randomly initialized

J26D's FullAttn was trained at L5 receiving residual stream activations after
5 DSQG blocks. Placing it at L0 causes distribution mismatch (raw embeddings ≠
processed residuals). This run adapts the block to the new input distribution.

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_borg_adapt_13m_bf16.py \
    > logs/run_borg_adapt_13m.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 2048
NUM_LAYERS       = 11
INTERFERENCE     = 2
FULL_ATTN_LAYER  = 5

MAX_TRAIN_SEQS      = 121_232
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

EMA_INIT  = 0.00035
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 2

FREEZE_FULL_ATTN_AFTER_EPOCH = 0

EXTRACTED_CKPT = "autoresearch/checkpoints/borg_midattn_gen2_ep3_full_attn.pt"

# =============================================================================

import json, math, os, subprocess, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE     = 32000
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
MAX_SEQ_LEN    = 2048
NUM_DOCS       = 100_000
MAX_VAL_SEQS   = 5_582

FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_CANDIDATES = [
    'benchmarks/logs/condm_tokenizer.json',
    'results/2048_condI_tokenizer.json',
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 20
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8 import DSQGAttentionV8 as DSQGAttentionV6, npci_rotate

assert len(OFFSETS) == 24

# ── condV physics helpers ─────────────────────────────────────────────────────

from causal_ema_scan import causal_ema_scan as _causal_ema_scan

def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor,
                floor: float = EMA_FLOOR) -> torch.Tensor:
    """Causal EMA — Triton scan (O(B·N·D) memory vs O(B·D·N·K) conv)."""
    return _causal_ema_scan(xi, ema_factor, floor=floor)


def _kdv_correction(pool: torch.Tensor,
                    kdv_alpha: torch.Tensor) -> torch.Tensor:
    """KdV soliton: pool += α * pool * Δpool. Zero-init → identity at start."""
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    return pool + alpha * pool * (pool - pool_prev)


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AGC: normalise to unit RMS per token. No learnable params."""
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
    """V8 DSQG attention + condV interference (EMA + KdV + AGC)."""
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

            pool = _causal_ema(xi, self.ema_factor, floor=EMA_FLOOR)
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
                 ffn_dim, seq_len, full_attn_layer, interference_interval,
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
                has_if = (interference_interval is not None and
                          i % interference_interval == interference_interval - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
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

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

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
        """Yield all parameters from the FullAttentionBlock at layer 0."""
        for p in self.blocks[self.full_attn_layer].parameters():
            yield p

    def non_full_attn_parameters(self):
        """Yield all parameters except those in the FullAttentionBlock."""
        fa_ids = {id(p) for p in self.full_attn_parameters()}
        for p in self.parameters():
            if id(p) not in fa_ids:
                yield p

    def physics_summary(self):
        """Log EMA and KdV state for all interference blocks."""
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = block.ema_factor.item()
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


def load_data():
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
    else:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True)
        texts = []
        for item in ds:
            if len(item['text']) < 5_000:
                continue
            texts.append(item['text'])
            if len(texts) >= NUM_DOCS:
                break
        os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
        with open(FW_CACHE_FILE, 'w') as fp:
            json.dump(texts, fp)
    n = len(texts)
    return {'train': texts[:int(n * 0.95)],
            'val':   texts[int(n * 0.95):int(n * 0.95) + 2500]}


def encode_split(split_texts, tokenizer, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


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


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir):
    """Save just the FullAttn block weights after an epoch (for precompute experiment)."""
    full_attn_block = model.blocks[model.full_attn_layer]
    state_dict = {}
    for name, param in full_attn_block.named_parameters():
        state_dict[f"blocks.{model.full_attn_layer}.{name}"] = param.data.clone()

    payload = {
        "full_attn_block": state_dict,
        "config": {
            "embedding_dim":     EMBEDDING_DIM,
            "num_heads":         NUM_HEADS,
            "ffn_dim":           FFN_DIM,
            "seq_len":           MAX_SEQ_LEN,
            "source_script":     "train/train_borg_adapt_13m_bf16.py",
            "source_layer":      FULL_ATTN_LAYER,
            "num_layers":        NUM_LAYERS,
            "epoch":             epoch,
            "note": (
                "FullAttn adapted to L0 input distribution (raw embeddings). "
                f"Epoch {epoch} of Borg adaptation run."
            ),
        },
    }

    out_path = os.path.join(checkpoint_dir, f"borg_L11_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(no_warmstart: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🔬 DWARF Autoresearch — BORG MID-ATTN RUN')
    print('  FullAttn at L2 (middle), warm-started from J26D (2 DSQG blocks before FA)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  IF interval={INTERFERENCE}, Full attn layer={FULL_ATTN_LAYER} (MIDDLE, L5 of 11, Gen2 FA frozen)')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
    print(f'  FREEZE_FULL_ATTN_AFTER_EPOCH={FREEZE_FULL_ATTN_AFTER_EPOCH}')
    print(f'  git={git_hash}')

    splits = load_data()
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    _encoded_cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
    else:
        train_data = encode_split(splits['train'], tokenizer, 'Train')
        val_data   = encode_split(splits['val'],   tokenizer, 'Val')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    model = AutoresearchTransformerPhysics(
        vocab_size=tokenizer.vocab_size(), embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=INTERFERENCE,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    # ── Warm-start FullAttn from J26D extracted weights ──────────────────────
    ckpt_suffix = 'control' if no_warmstart else 'warmstart'
    best_ckpt_name = f'borg_L11_best.pt'
    if no_warmstart:
        print(f'\n  [control] Random-init FullAttn — skipping J26D warm-start')
        print(f'  Checkpoint: {best_ckpt_name}\n')
    if not no_warmstart and os.path.exists(EXTRACTED_CKPT):
        print(f'\n  Loading J26D FullAttn weights from {EXTRACTED_CKPT}...')
        extracted = torch.load(EXTRACTED_CKPT, map_location="cpu")
        full_attn_weights = extracted["full_attn_block"]

        remapped = {}
        for k, v in full_attn_weights.items():
            # Strip _orig_mod (torch.compile artifact) and remap any source block index
            # to the target FULL_ATTN_LAYER — works regardless of source checkpoint's FA position
            import re as _re
            new_key = _re.sub(r'blocks\.\d+\._orig_mod\.', f'blocks.{FULL_ATTN_LAYER}.', k)
            new_key = _re.sub(r'blocks\.\d+\.', f'blocks.{FULL_ATTN_LAYER}.', new_key)
            remapped[new_key] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f'  Loaded {len(remapped)} weights')
        print(f'  Missing (expected — DSQG blocks random init): {len(missing)}')
        if unexpected:
            print(f'  WARNING unexpected keys: {unexpected}')
    elif not no_warmstart:
        print(f'  WARNING: {EXTRACTED_CKPT} not found — training from scratch')

    try:
        for i, block in enumerate(model.blocks):
            if type(block).__name__ == "FullAttentionBlock":
                model.blocks[i] = torch.compile(block, fullgraph=False)
                print(f"  compiled FullAttnBlock at layer {i}")
        print("  torch.compile applied to FullAttnBlock(s)")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

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

    full_attn_frozen = False

    for epoch in range(1, SCREEN_EPOCHS + 1):
        if FREEZE_FULL_ATTN_AFTER_EPOCH is not None and epoch > FREEZE_FULL_ATTN_AFTER_EPOCH and not full_attn_frozen:
            print(f'\n  ── Freezing FullAttn block (epoch {epoch}) ──')
            for p in model.full_attn_parameters():
                p.requires_grad = False
            full_attn_frozen = True
            print('  FullAttn frozen.')

        model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item() * GRAD_ACCUM:.4f}')

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

        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

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
        print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score:       {ar_score:.2f}')
    print(f'memory_mb:      {memory_mb:.1f}')
    print(f'elapsed_s:      {elapsed_s:.1f}')
    print(f'num_params_M:   {n_params / 1e6:.1f}')
    print(f'num_layers:     {NUM_LAYERS}')
    print(f'num_offsets:    24')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init:       {EMA_INIT}')
    print(f'description:    Borg L=11 — {n_params/1e6:.1f}M, Gen2 FA frozen@L5 (mid), 10 fresh DSQG, '
          f'D={EMBEDDING_DIM}, L={NUM_LAYERS}, 5 DSQG each side of FA')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Borg adaptation run')
    parser.add_argument('--no-warmstart', action='store_true',
                        help='Skip J26D weight loading (random-init control)')
    args = parser.parse_args()
    if args.no_warmstart:
        # Patch the global so the warm-start block is skipped
        _real_exists = os.path.exists
        os.path.exists = lambda p: False if p == EXTRACTED_CKPT else _real_exists(p)
        # Use distinct checkpoint names so control and warm-start don't collide
        import train_borg_adapt_13m_bf16 as _self
        _self.CHECKPOINT_DIR  # ensure module loaded
        # Monkey-patch checkpoint name by modifying the save path at runtime
        # — handled by checking sys.argv in train() is cleaner; just note in log
        print("\n  [--no-warmstart] Random-init control — skipping J26D weight load")
        print("  Checkpoint: borg_L11_best.pt\n")
        # Patch the checkpoint name global
        import sys as _sys
        # We'll use a simple flag read inside train()
    train(no_warmstart=args.no_warmstart)
