"""
🔬 DWARF Clean Curve — 50M BF16 (4090)

Clean curve 50M — D=512, L=8, FFN=3072, V=32K, J24D-int2 physics

Architecture: J24D-int2 + condV physics (EMA + KdV + AGC)
  - V8 kernel with J=24 offsets
  - IF fires at layers where i % 2 == 1: [1, 3, 5]
  - Full attention at layer L-1 (layer 7)
  - BF16, no GradScaler

Run (from /home/dlewis3/Desktop/AI/DWARF):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_curve_50m_bf16.py \
    > logs/run_curve_50m.log 2>&1 &
"""

# =============================================================================
# PATHS — easy to change
# =============================================================================

TOKENIZER_PATH = 'results/fineweb_v32k_v2_tokenizer.json'
DATASET_PATH = 'logs/fineweb_v32k_encoded.pt'
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_NAME = 'curve_50m_best.pt'

# =============================================================================
# ARCHITECTURE KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM = 512
NUM_HEADS = 8
FFN_DIM = 3072
NUM_LAYERS = 8
VOCAB_SIZE = 32000
INTERFERENCE = 2
FULL_ATTN_LAYER = 7

MAX_TRAIN_SEQS = 121_232
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT = 15.0

EMA_INIT = 0.003
EMA_FLOOR = 0.0003

LR = 3e-4
SCREEN_EPOCHS = 3
WARMUP_STEPS = 200

# =============================================================================

import math
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
GRAD_ACCUM = 2
MAX_SEQ_LEN = 2048
MAX_VAL_SEQS = 5_582

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS = 20
_PASSKEY_WORDS = ['apple', 'banana', 'orange', 'cherry', 'grape',
                  'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8 import DSQGAttentionV8

assert len(OFFSETS) == 24
assert FULL_ATTN_LAYER == NUM_LAYERS - 1

# ── condV physics helpers ─────────────────────────────────────────────────────

_EMA_KERNEL_LEN = 256


def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor,
                floor: float = 0.0003) -> torch.Tensor:
    """Causal EMA via depthwise conv with exponential kernel."""
    B, N, D = xi.shape
    alpha = ema_factor.clamp(floor, 0.5)
    k_len = min(_EMA_KERNEL_LEN, N)
    t = torch.arange(k_len, device=xi.device, dtype=torch.float32)
    kernel = alpha.float() * (1.0 - alpha.float()).pow(t)
    kernel = (kernel / kernel.sum()).flip(0)
    xi_f = xi.float()
    xi_bd = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p = F.pad(xi_bd, (k_len - 1, 0))
    pool = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(xi.dtype)


def _kdv_correction(pool: torch.Tensor,
                    kdv_alpha: torch.Tensor) -> torch.Tensor:
    """KdV soliton: pool += α * pool * Δpool. Zero-init → identity at start."""
    alpha = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    return pool + alpha * pool * (pool - pool_prev)


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AGC: normalise to unit RMS per token. No learnable params."""
    D = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Model ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV8Physics(nn.Module):
    """V8 DSQG attention + condV interference (EMA + KdV + AGC)."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = DSQGAttentionV8(embedding_dim, num_heads,
                                    seq_len=seq_len, dropout=dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))
            self.kdv_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD = self.num_heads, self.head_dim

            pool = _causal_ema(xi, self.ema_factor, floor=EMA_FLOOR)
            pool = _kdv_correction(pool, self.kdv_alpha)
            pool = _agc_normalize(pool)

            inter = torch.sigmoid(self.inter_gate(xi)) * pool
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

        orig_dtype = q.dtype
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        out = out.to(orig_dtype)

        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CurveTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer, interference_interval,
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                has_if = (interference_interval is not None and
                          i % interference_interval == interference_interval - 1)
                blocks.append(DSQGBlockV8Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
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
            if isinstance(m, DSQGAttentionV8):
                nn.init.normal_(m.phase_base, 0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes, 0.0, 0.01)
                nn.init.normal_(m.phase_gain, 0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def forward(self, idx):
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device).unsqueeze(0)
        x = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV8):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def physics_summary(self):
        """Log EMA and KdV state for all interference blocks."""
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV8Physics) and block.interference:
                alpha = block.ema_factor.item()
                kdv = block.kdv_alpha.item()
                win = round(1.0 / max(alpha, EMA_FLOOR))
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


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device)
        y = data[i:i+BATCH_SIZE, 1:].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)
    results = {}
    for d in PASSKEY_DISTANCES:
        correct, n_valid = 0, 0
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
            ids = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct += int(([target] + others[:9])[
                           logits[0][cand_ids].argmax().item()] == target)
            n_valid += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        git_hash = 'unknown'

    if_blocks = [i for i in range(NUM_LAYERS)
                 if i != FULL_ATTN_LAYER and i % INTERFERENCE == INTERFERENCE - 1]

    print('=' * 70)
    print('  🔬 Clean curve 50M — D=512, L=8, FFN=3072, V=32K, J24D-int2 physics')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}, V={VOCAB_SIZE}')
    print(f'  head_dim={EMBEDDING_DIM // NUM_HEADS}')
    print(f'  IF interval={INTERFERENCE}, Full attn layer={FULL_ATTN_LAYER}')
    print(f'  IF blocks at: {if_blocks}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch={BATCH_SIZE}, GradAccum={GRAD_ACCUM}, BF16')
    print(f'  git={git_hash}')

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_PATH}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER_PATH))
    print(f'Loaded tokenizer from {TOKENIZER_PATH}')

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f'Encoded dataset not found: {DATASET_PATH}')
    print(f'Loading pre-encoded dataset from {DATASET_PATH}')
    _cache = torch.load(DATASET_PATH, weights_only=True)
    train_data = _cache['train']
    val_data = _cache['val']

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    model = CurveTransformer(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=INTERFERENCE,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    try:
        for i, block in enumerate(model.blocks):
            if type(block).__name__ == "FullAttentionBlock":
                model.blocks[i] = torch.compile(block, fullgraph=False)
                print(f"  compiled FullAttnBlock at layer {i}")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = torch.optim.AdamW([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(
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
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))
            marker = ' *'

        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionV8):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} '
                  f'|max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final = ppl_results.get(SCREEN_EPOCHS, 999.0)

    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
    print(f'memory_mb:      {memory_mb:.1f}')
    print(f'elapsed_s:      {elapsed_s:.1f}')
    print(f'num_params_M:   {n_params / 1e6:.1f}')
    print(f'num_layers:     {NUM_LAYERS}')
    print(f'num_offsets:    24')
    print(f'description:    Clean curve 50M — D=512, L=8, FFN=3072, V=32K, J24D-int2 physics')


if __name__ == '__main__':
    train()
