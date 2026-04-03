"""
🔬 DWARF D=768 L=32 — SFT on OpenHermes-2.5 + Open-Orca GPT-4 blend

Architecture: D=768, H=12 (hd=64), L=32, FFN=1536, FA@L8
  Same model class as train_d768_l32_h100_bf16.py — MUST use that class to match checkpoint.

Purpose: Test whether relay mechanism survives SFT.
  - Baseline checkpoint: d768_l32_fa8_best.pt (cont_ep2, PPL=25.85, passkey=98.3%)
  - Dataset: openhermes_orca_sft_encoded.pt (498K examples: 300K OpenHermes + 200K Orca-GPT4)
  - Labels: 0/1 mask — 1=answer tokens (supervised), 0=prompt/context (ignored in CE)
  - Passkey canary after each epoch — abort if mean <85%

Key hypothesis: SQuAD/TriviaQA passages keep FA retrieval gradient alive.
  Prior smoltalk SFT degraded relay 99.2%→72.5% because smoltalk lacks retrieval structure.
  Retrieval-aligned dataset should preserve relay while teaching instruction following.

Config:
  - LR = 5e-5 (conservative for SFT)
  - 3 epochs, BS=8, GRAD_ACCUM=16 (eff_batch=128)
  - CE only on answer tokens (label=1)
  - Passkey abort threshold: mean <85%

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d768_l32_oh_orca_sft_4090_bf16.py \\
    2>&1 | tee logs/run_d768_l32_oh_orca_sft.log
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 768
NUM_HEADS        = 12
FFN_DIM          = 1536
NUM_LAYERS       = 32
FULL_ATTN_LAYER  = 8

EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001
SCALE_EMBED_INIT_VAL = 0.15   # only used for _init_weights; checkpoint overrides
SCALE_EMBED_LR_MULT  = 18.37

LR            = 5e-5           # conservative SFT LR
SFT_EPOCHS    = 3
BATCH_SIZE    = 8
GRAD_ACCUM    = 16                # eff_batch = 128
CE_CHUNK      = 512
MAX_SEQ_LEN   = 1024  # OH/Orca mean=411t; longer than retrieval set, still saves ~2x vs 2048

RESUME_CHECKPOINT = 'autoresearch/checkpoints/d768_l32_fa8_best.pt'
SFT_DATASET_PATH  = 'logs/oh_orca_75k_sft_encoded.pt'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

PASSKEY_ABORT_THRESHOLD = 85.0   # % — abort SFT if relay degrades below this
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50

# =============================================================================

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

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


VOCAB_SIZE  = 32000

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'every_other').lower()
PASSKEY_BATCH_SIZE  = int(os.getenv('DWARF_PASSKEY_BATCH', '16'))

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


# ── Physics helpers ───────────────────────────────────────────────────────────

def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)

def _agc_normalize(pool, eps=1e-6):
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Model (identical class to train_d768_l32_h100_bf16.py) ───────────────────

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
        x = self.drop(self.embedding(idx))
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
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
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)


# ── Passkey eval ──────────────────────────────────────────────────────────────

_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape',
                    'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'


def _load_tokenizer():
    candidates = [
        'results/fineweb_tokenizer_32k.json',
        'results/fineweb_v32k_v2_tokenizer.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            from tokenizers import Tokenizer as HFTokenizer
            tok = HFTokenizer.from_file(path)
            tok.no_truncation()
            tok.no_padding()

            def _encode(text):
                return tok.encode(text).ids

            def _decode(ids):
                return tok.decode(ids)

            print(f'  Tokenizer: {path}')
            return _encode, _decode
    raise FileNotFoundError('No tokenizer found')


def _build_passkey_prompt(encode, word, filler_tokens, distance):
    intro   = encode(_INTRO_TEMPLATE.format(word=word))
    cue     = encode(_RETRIEVAL_CUE)
    n_fill  = max(0, distance - len(intro))
    tokens  = intro + filler_tokens[:n_fill] + cue
    return tokens, len(intro)


@torch.no_grad()
def evaluate_passkey(model, device, encode, decode):
    model.eval()
    filler_tokens = []
    while len(filler_tokens) < 2000:
        filler_tokens += encode(_FILLER_SENTENCE)

    results = {}
    for dist in PASSKEY_DISTANCES:
        correct = 0
        for trial in range(PASSKEY_TRIALS):
            word     = _PASSKEY_WORDS[trial % len(_PASSKEY_WORDS)]
            candidates = _PASSKEY_WORDS
            prompt, _ = _build_passkey_prompt(encode, word, filler_tokens, dist)
            if len(prompt) > 2048 - 1:
                prompt = prompt[:2048 - 1]

            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            with _amp_context(device):
                logits = model(ids)
            next_logits = logits[0, -1]

            # Score each candidate word
            best_word, best_score = None, float('-inf')
            for cand in candidates:
                cand_ids = encode(' ' + cand)
                if not cand_ids:
                    cand_ids = encode(cand)
                score = sum(next_logits[t].item() for t in cand_ids[:1])
                if score > best_score:
                    best_score, best_word = score, cand
            if best_word == word:
                correct += 1

        acc = correct / PASSKEY_TRIALS * 100
        results[dist] = acc

    mean_acc = sum(results.values()) / len(results)
    return results, mean_acc


# ── SFT data collation ────────────────────────────────────────────────────────

def collate_sft_batch(examples, pad_id=0):
    """
    examples: list of (ids_list, labels_list) where labels are 0/1 mask.
    Returns: input_ids [B, T], targets [B, T] with -100 where label==0.
    """
    max_len = min(max(len(e[0]) for e in examples), MAX_SEQ_LEN)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    targets   = torch.full((len(examples), max_len), -100,   dtype=torch.long)

    for i, (ids, labels) in enumerate(examples):
        ids    = ids[:max_len]
        labels = labels[:max_len]
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids,    dtype=torch.long)
        for j, (tok, lbl) in enumerate(zip(ids, labels)):
            if lbl == 1:
                targets[i, j] = tok

    # Shift: predict next token
    # input  = tokens[:-1], target = tokens[1:] masked by label
    x = input_ids[:, :-1].contiguous()
    y = targets[:,   1:].contiguous()
    return x, y


# ── Val loss on SFT data ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sft_val(model, val_data, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    indices = list(range(min(len(val_data), max_batches * BATCH_SIZE)))
    for start in range(0, len(indices), BATCH_SIZE):
        batch = [val_data[i] for i in indices[start:start + BATCH_SIZE]]
        x, y  = collate_sft_batch(batch)
        x, y  = x.to(device), y.to(device)
        mask  = (y != -100)
        if not mask.any():
            continue
        with _amp_context(device):
            hidden = model.forward_hidden(x)
        n_tok = y.shape[1]
        chunk_loss, chunk_count = 0.0, 0
        for c_start in range(0, n_tok, CE_CHUNK):
            c_end   = min(c_start + CE_CHUNK, n_tok)
            h_chunk = hidden[:, c_start:c_end, :]
            y_chunk = y[:, c_start:c_end]
            mask_c  = (y_chunk != -100)
            if not mask_c.any():
                continue
            with _amp_context(device):
                logits_chunk = model.out(h_chunk).float()
            loss = F.cross_entropy(
                logits_chunk.reshape(-1, VOCAB_SIZE),
                y_chunk.reshape(-1),
                ignore_index=-100,
                reduction='sum')
            chunk_loss  += loss.item()
            chunk_count += mask_c.sum().item()
        total_loss   += chunk_loss
        total_tokens += chunk_count

    if total_tokens == 0:
        return float('inf')
    return math.exp(total_loss / total_tokens)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if USE_LIGER_CE:
        print('Using Liger fused CE')
    else:
        print('Liger not available, using standard CE')

    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = 'unknown'
    print(f'Git: {git_hash}')

    # ── Load tokenizer ────────────────────────────────────────────────────────
    encode, decode = _load_tokenizer()

    # ── Load SFT dataset ──────────────────────────────────────────────────────
    print(f'Loading SFT dataset: {SFT_DATASET_PATH}')
    sft_data = torch.load(SFT_DATASET_PATH, map_location='cpu', weights_only=False)
    train_examples = sft_data['train']
    val_examples   = sft_data['val']
    print(f'  Train: {len(train_examples):,}  Val: {len(val_examples):,}')
    print(f'  Composition: {sft_data.get("composition", {})}')

    # ── Build model ───────────────────────────────────────────────────────────
    print('Building model...')
    model = AutoresearchTransformerPhysics(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=2048,   # must match pretraining; MAX_SEQ_LEN=512 is only for SFT data collation
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    print(f'Parameters: {model.param_count():,} ({model.param_count() / 1e6:.1f}M)')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading checkpoint: {RESUME_CHECKPOINT}')
    ckpt  = torch.load(RESUME_CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=True)
    print('  Checkpoint loaded successfully.')

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f'Warming up Triton kernels (BS={BATCH_SIZE})...')
    _wb  = min(BATCH_SIZE, len(train_examples))
    _wx  = collate_sft_batch(train_examples[:_wb])[0].to(device)
    with _amp_context(device):
        _ = model(_wx)
    del _wx
    torch.cuda.empty_cache()
    print('  Warmup complete.')

    # ── Baseline passkey ──────────────────────────────────────────────────────
    print('\n=== Baseline passkey (pre-SFT) ===')
    pk_results, pk_mean = evaluate_passkey(model, device, encode, decode)
    dist_str = '  '.join(f'd={d}:{v:.0f}%' for d, v in pk_results.items())
    print(f'Passkey mean={pk_mean:.1f}%')
    print(dist_str)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Freeze scale_embed during SFT — it crystallized during pretraining and
    # has effective LR = 5e-5 × 18.37 = 9.2e-4, enough to destroy relay topology
    # in one epoch. Freezing it preserves the relay routing map entirely.
    # CRITICAL: must also exclude from optimizer entirely — AdamW's momentum/
    # variance state can still update parameters even with requires_grad=False.
    for p in model.scale_embed_parameters():
        p.requires_grad_(False)

    # Only include parameters that actually need gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': trainable_params, 'lr': LR},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_examples) / BATCH_SIZE / GRAD_ACCUM)
    total_steps     = SFT_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR * 0.1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_ppl = float('inf')

    # ── MFU tracking setup ─────────────────────────────────────────────────────
    n_params = model.param_count()
    gpu_peak_flops = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step = 6 * n_params * tokens_per_step
    mfu_window = deque(maxlen=20)
    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f'\nSFT training: {SFT_EPOCHS} epochs × {steps_per_epoch} steps '
          f'(eff_batch={BATCH_SIZE * GRAD_ACCUM}, LR={LR:.0e})')

    import random
    indices = list(range(len(train_examples)))

    global_step = 0
    for epoch in range(1, SFT_EPOCHS + 1):
        model.train()
        random.shuffle(indices)
        epoch_loss, epoch_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        acc_loss = 0.0
        for step_idx in range(steps_per_epoch):
            t_start_event = torch.cuda.Event(enable_timing=True)
            t_end_event   = torch.cuda.Event(enable_timing=True)
            t_start_event.record()

            for acc in range(GRAD_ACCUM):
                batch_start = (step_idx * GRAD_ACCUM + acc) * BATCH_SIZE
                batch_idxs  = indices[batch_start: batch_start + BATCH_SIZE]
                if not batch_idxs:
                    continue
                batch = [train_examples[i] for i in batch_idxs]
                x, y  = collate_sft_batch(batch)
                x, y  = x.to(device), y.to(device)

                mask = (y != -100)
                if not mask.any():
                    continue

                n_ans_tokens = mask.sum().item()

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(x)
                        loss = liger_ce_fn(
                            hidden.view(-1, hidden.size(-1)),
                            model.out.weight,
                            y.view(-1)
                        )
                    loss_val = loss.item()
                    (loss / GRAD_ACCUM).backward()
                    del hidden, loss
                else:
                    with _amp_context(device):
                        logits = model(x).float()
                    loss = F.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE),
                        y.reshape(-1),
                        ignore_index=-100,
                        reduction='sum')
                    loss_val = loss.item()
                    (loss / max(n_ans_tokens, 1) / GRAD_ACCUM).backward()

                acc_loss     += loss_val / max(n_ans_tokens, 1) if not USE_LIGER_CE else loss_val
                epoch_loss   += loss_val if not USE_LIGER_CE else loss_val * n_ans_tokens
                epoch_tokens += n_ans_tokens

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            t_end_event.record()
            torch.cuda.synchronize()
            step_ms = t_start_event.elapsed_time(t_end_event)
            mfu_window.append(step_ms)
            global_step += 1

            if (step_idx + 1) % 200 == 0:
                avg = acc_loss / 200
                elapsed = time.time() - t0
                avg_step_ms = sum(mfu_window) / len(mfu_window)
                tok_per_sec = tokens_per_step / (avg_step_ms / 1000.0)
                mfu_str = ''
                if gpu_peak_flops is not None:
                    mfu = (flops_per_step / (avg_step_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'
                print(f'  Ep {epoch} Step {step_idx+1}/{steps_per_epoch} | '
                      f'Loss {avg:.4f}{mfu_str} | {tok_per_sec:.0f} tok/s | {elapsed:.0f}s')
                acc_loss = 0.0

        # ── Epoch eval ────────────────────────────────────────────────────────
        val_ppl = evaluate_sft_val(model, val_examples, device)
        se_vals = [m.scale_embed.abs().mean().item()
                   for m in model.modules()
                   if isinstance(m, DSQGAttentionV6)]
        se_mean = sum(se_vals) / len(se_vals) if se_vals else 0
        se_max  = max(se_vals) if se_vals else 0

        phys = model.physics_summary()
        marker = ' *' if val_ppl < best_val_ppl else ''
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        print(f'\nEp {epoch}/{SFT_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')
        print(f'  scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}')
        print(f'  Physics: {phys}')

        # ── Passkey canary ────────────────────────────────────────────────────
        pk_results, pk_mean = evaluate_passkey(model, device, encode, decode)
        dist_str = '  '.join(f'd={d}:{v:.0f}%' for d, v in pk_results.items())
        print(f'  Passkey mean={pk_mean:.1f}%')
        print(f'  {dist_str}')

        # ── Save checkpoint ───────────────────────────────────────────────────
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_ppl': val_ppl,
            'passkey_mean': pk_mean,
            'git_hash': git_hash,
        }, os.path.join(CHECKPOINT_DIR, f'd768_l32_oh_orca_sft_ep{epoch}_resume.pt'))

        if val_ppl <= best_val_ppl:
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_ppl': val_ppl,
                'passkey_mean': pk_mean,
                'epoch': epoch,
                'git_hash': git_hash,
            }, os.path.join(CHECKPOINT_DIR, 'd768_l32_oh_orca_sft_best.pt'))

        # ── Relay abort check ─────────────────────────────────────────────────
        if pk_mean < PASSKEY_ABORT_THRESHOLD:
            print(f'\n⚠️  RELAY DEGRADED: passkey mean {pk_mean:.1f}% < '
                  f'{PASSKEY_ABORT_THRESHOLD}% threshold. Stopping SFT.')
            break

    print('\n=== SFT Complete ===')
    print(f'Best val PPL: {best_val_ppl:.2f}')
    print(f'Final passkey: {pk_mean:.1f}%')
    print(f'Relay {"PRESERVED" if pk_mean >= PASSKEY_ABORT_THRESHOLD else "DEGRADED"}')


if __name__ == '__main__':
    main()
