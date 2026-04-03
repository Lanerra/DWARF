"""
🚀 DWARF D=768 — 94.2M params, D-scaling probe, cold-start

Architecture: D=768, H=12 (hd=64), L=16, FFN=1536, J=24 (se015 offsets), TIED lm_head
  L0-L2:  DSQGBlockV6Physics  IF=False  ← 3 pre-FA warm-up relay layers
  L3:     DSQGBlockV6Physics  IF=True   ← preIF@L3 (one layer before FA)
  L4:     FullAttentionBlock            ← FA@L4 (25% depth, validated recipe)
  L5-15:  DSQGBlockV6Physics  IF=False  ← 11 post-FA relay layers

Design rationale:
  Scaling ladder: Moonshot-58M (D=512) → D=768 (this) → 267M (D=1024)
  D=768, H=12 keeps HD=64 exactly — no kernel changes required.
  LR_MULT = 15 × √(768/512) = 18.4  (μP √D scaling rule, validated at D=512 and D=1024)
  DSQG:FA ratio = 15:1  (same as depth16_55m_fa4, validated pattern)
  Relay bandwidth (J×HD=1536) : residual stream (D=768) = 2.0×  (gap_ratio threshold)
  FFN = 2×D = 1536  (confirmed optimal by FFN ablation; relay carries memory load)

Validated baselines:
  Moonshot-58M ep2     D=512 H=8  L=8  (PPL=35.04, passkey=99.2%, ar_score=80.90)
  Depth16-FA@L4 ep3    D=512 H=8  L=16 (PPL=44.49, passkey=99.2%, ar_score=80.87)
  267M ep2             D=1024 H=16 L=24 (PPL=22.67, passkey=100%, ar_score=~78)

This run fills the gap between Moonshot and 267M.

Config:
  - Tokenizer: fineweb_tokenizer_32k.json  (32K BPE, FineWeb proper)
               EOS id = 0  (<|endoftext|>)
  - Dataset:   fineweb_edu_encoded_2048_v2.pt (~2.01M seqs, 4.13B tokens)
  - EMA_INIT = 0.0208 (= 1/δ_relay_min = 1/48, empirically validated for J24D se015)
  - SCALE_EMBED_INIT = 0.15  (slightly above 0.1; empirically safe, speeds percolation)
  - LR_MULT = 18.4  (= 15 × √(768/512), μP √D scaling)
  - Batch: BS=16 × GRAD_ACCUM=8 = eff_batch=128
  - CHECKPOINT_STRATEGY = every_other  (L=16 needs it; D=768 activations bigger than D=512)
  - Cold start (no warm-start checkpoint)
  - ~94.2M parameters (tied lm_head)

22% Chinchilla budget:
  100% Chinchilla = 20 × 94.2M params / 2048 seq_len = 920K seqs
  22% target = 202,500 seqs  (~3 epochs × 67.5K steps/epoch at eff_batch=128)
  Expected runtime: ~3.9h on 4090

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_d768_94m_4090_bf16.py \\
    > logs/run_d768_94m.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

EMBEDDING_DIM    = 768
NUM_HEADS        = 12         # hd = 768/12 = 64  (kernel-tuned sweet spot, no changes required)
FFN_DIM          = 1536       # 2×D — confirmed optimal by FFN ablation (relay carries memory load)
NUM_LAYERS       = 16
FULL_ATTN_LAYER  = 4          # FA@L4 = 25% depth — validated recipe from depth16_55m_fa4
                               # L0-L2: pre-FA warmup; L3: preIF; L5-15: 11 post-FA relay

MAX_TRAIN_SEQS       = 202_500   # 22% Chinchilla (920K × 0.22)
SCALE_EMBED_INIT_VAL = 0.15      # slightly above 0.1; safe at this scale, speeds percolation crossing
SCALE_EMBED_LR_MULT  = 18.4      # 15 × √(D/512) = 15 × √(768/512) = 18.37 ≈ 18.4

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# Empirically validated for J24D (se015): trains to α≈0.0207, 0.6% error
# δ_relay_min = 48 (first offset after local cluster [1..28])
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
MAX_TRAIN_SEQS = int(os.environ.get('MAX_TRAIN_SEQS_OVERRIDE', MAX_TRAIN_SEQS))
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

VOCAB_SIZE     = 32000
BATCH_SIZE     = 16
GRAD_ACCUM     = 8    # effective batch = 128
CE_CHUNK       = 512  # chunked CE — avoids materialising full (BS×2047×32K) fp32 grad tensor
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50   # n=50: ±7pp noise floor (n=20 risks inflation on borderline models)
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

ENABLE_TORCH_COMPILE = os.getenv('DWARF_ENABLE_COMPILE', '0') == '1'
COMPILE_MODE         = os.getenv('DWARF_COMPILE_MODE', 'default')
CHECKPOINT_STRATEGY  = os.getenv('DWARF_CKPT', 'every_other').lower()  # none|full_attn|every_other|all
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

assert len(OFFSETS) == 24, f"Expected 24 offsets (se015), got {len(OFFSETS)}"


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
        return F.dropout(
            self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
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


class AutoresearchTransformerPhysics(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer,
                 scale_embed_init_val=0.0, dropout=0.0):
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
        result = self.tokenizer.encode(text)
        return result.ids if hasattr(result, 'ids') else list(result)
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    bs = 4  # conservative: 4×2047×32K×2 bytes ≈ 0.5 GB; D=768 activations larger
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
            row          = torch.arange(ids_b.size(0), device=device)
            next_logits  = logits[row, pos_b, :]
            cand_logits  = torch.gather(next_logits, 1, cand_b)
            correct     += (cand_logits.argmax(dim=1) == 0).sum().item()

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
            "source_script": "train/train_d768_94m_4090_bf16.py",
            "source_layer":  FULL_ATTN_LAYER,
            "num_layers":    NUM_LAYERS,
            "num_offsets":   len(OFFSETS),
            "epoch":         epoch,
            "git_hash":      git_hash,
            "note": (
                f"D768-94M-FA4: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} "
                f"FFN={FFN_DIM} J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER} "
                f"preIF@L{FULL_ATTN_LAYER-1}. Epoch {epoch}/3. Cold start."
            ),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"d768_94m_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🚀 DWARF D=768 — D=768 H=12 hd=64 L=16 FFN=1536 J=24, cold start')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}, '
          f'{NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        _cc   = torch.cuda.get_device_capability()
        _path = ('sm_90 (H100/H200 — tuned)' if (_cc[0] == 9 and _cc[1] == 0) else
                 'sm_89 (4090 Ada — tuned)' if (_cc[0] == 8 and _cc[1] == 9) else
                 f'sm_{_cc[0]}{_cc[1]}')
        print(f'  Kernel path: {_path}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER} ({FULL_ATTN_LAYER/NUM_LAYERS*100:.0f}% depth), '
          f'preIF@L{FULL_ATTN_LAYER-1}')
    print(f'  Post-FA relay: {NUM_LAYERS - FULL_ATTN_LAYER - 1} layers  '
          f'DSQG:FA = {NUM_LAYERS - 1}:1')
    print(f'  Relay bandwidth (J×HD) = {len(OFFSETS) * (EMBEDDING_DIM // NUM_HEADS)} '
          f'({len(OFFSETS)}×{EMBEDDING_DIM//NUM_HEADS})')
    print(f'  Relay/residual ratio = {len(OFFSETS) * (EMBEDDING_DIM // NUM_HEADS) / EMBEDDING_DIM:.2f}×')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,} (22% Chinchilla), Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = '
          f'eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}  '
          f'passkey_batch_size={PASSKEY_BATCH_SIZE}')
    print(f'  git={git_hash}')

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found in: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'  Tokenizer: {tok_path}  (vocab={tokenizer.vocab_size():,})')

    # ── Dataset ───────────────────────────────────────────────────────────────
    _encoded_cache = 'logs/fineweb_edu_encoded_2048_v2.pt'
    if not os.path.exists(_encoded_cache):
        raise FileNotFoundError(
            f'Pre-encoded dataset not found: {_encoded_cache}\n'
            f'Run scripts/build_dataset_fineweb.py first.')
    print(f'  Loading pre-encoded dataset from {_encoded_cache}')
    _cache     = torch.load(_encoded_cache, weights_only=True)
    train_data = _cache['train'].long()
    val_data   = _cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoresearchTransformerPhysics(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=0.0,   # dropout=0 confirmed: dropout severs relay chains during training
    ).to(device)

    if ENABLE_TORCH_COMPILE:
        try:
            for i, block in enumerate(model.blocks):
                if type(block).__name__ == 'FullAttentionBlock':
                    try:
                        model.blocks[i] = torch.compile(
                            block, fullgraph=False, dynamic=False, mode=COMPILE_MODE)
                    except TypeError:
                        model.blocks[i] = torch.compile(block, fullgraph=False)
                    print(f'  torch.compile applied to FullAttentionBlock at L{i} '
                          f'(mode={COMPILE_MODE})')
                    break
        except Exception as e:
            print(f'  torch.compile skipped: {e}')
    else:
        print('  torch.compile disabled (set DWARF_ENABLE_COMPILE=1 to opt in)')

    n_params = model.param_count()
    print(f'  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    # ── Optimizer ─────────────────────────────────────────────────────────────
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

    best_val_loss = float('inf')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Kernel warmup ─────────────────────────────────────────────────────────
    # Forward + backward dummy pass to compile all Triton specialisations
    # before the training loop. Without this, the first real step stalls
    # silently for 5–30 min on a cold Triton cache.
    print(f'  Warming up Triton kernels (dummy fwd+bwd)...')
    _wb  = min(4, len(train_data))
    _wx  = train_data[:_wb, :-1].to(device)
    _wy  = train_data[:_wb,  1:].to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy, _wout, _wlogits_flat, _wy_flat, _wloss, _wgrad
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        print(f'  kernel warmup complete. '
              f'Peak VRAM so far: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
    else:
        print('  kernel warmup complete.')

    # ── Training loop ─────────────────────────────────────────────────────────
    ppl_results = {}
    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

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
            step += 1

            if step % 200 == 0:
                elapsed = time.time() - t_start
                print(f'  Ep {epoch}/{SCREEN_EPOCHS} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} | {elapsed/60:.1f}m elapsed', flush=True)

        # ── Epoch validation ──────────────────────────────────────────────────
        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = ' *'
            best_path = os.path.join(CHECKPOINT_DIR, 'd768_94m_best.pt')
            torch.save(model.state_dict(), best_path)

        se_max  = max((m.scale_embed.abs().max().item()
                       for m in model.modules()
                       if isinstance(m, DSQGAttentionV6)), default=0.0)
        se_mean = sum(m.scale_embed.abs().mean().item()
                      for m in model.modules()
                      if isinstance(m, DSQGAttentionV6))
        se_count = sum(1 for m in model.modules() if isinstance(m, DSQGAttentionV6))
        se_mean  = se_mean / max(se_count, 1)

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')
        print(f' scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}')
        print(f' Physics: {model.physics_summary()}')

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f' Peak VRAM: {peak_gb:.1f} GB')

        passkey = passkey_accuracy(model, tokenizer, device)
        passkey_mean = sum(passkey.values()) / len(passkey)
        print(f' Passkey mean={passkey_mean*100:.1f}%')
        dist_str = '  '.join(f'd={d}:{v*100:.0f}%' for d, v in passkey.items())
        print(f' {dist_str}')

        # ar_score: relative to Moonshot-58M ep2 baseline (ppl=35.04, passkey=99.2%)
        ar_passkey = passkey_mean * 100
        ar_ppl     = val_ppl
        ar_score   = (ar_passkey - 99.2) + (35.04 - ar_ppl) * 0.5
        print(f' ar_score: {ar_score:.2f}')

        # Save per-epoch resume checkpoint and FA block
        resume_path = os.path.join(CHECKPOINT_DIR, f'd768_94m_ep{epoch}_resume.pt')
        torch.save({
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_ppl': val_ppl,
        }, resume_path)
        print(f'  Saved resume checkpoint: {resume_path}')
        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print('\n' + '=' * 70)
    print(f'  Training complete in {elapsed/3600:.2f}h')
    print(f'  D={EMBEDDING_DIM} H={NUM_HEADS} hd={EMBEDDING_DIM//NUM_HEADS} '
          f'L={NUM_LAYERS} FFN={FFN_DIM} J={len(OFFSETS)}')
    print(f'  FA@L{FULL_ATTN_LAYER} (25% depth), {NUM_LAYERS - FULL_ATTN_LAYER - 1} post-FA relay layers')
    print(f'  scale_embed_init={SCALE_EMBED_INIT_VAL}  LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  ema_init: {EMA_INIT}')
    print(f'  description: D=768 94M — {n_params/1e6:.1f}M, D=768, L={NUM_LAYERS}, '
          f'FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}, J=24, 4090')
    for ep, ppl in ppl_results.items():
        print(f'  ppl_ep{ep}: {ppl:.2f}')
    print('=' * 70)


if __name__ == '__main__':
    train()
