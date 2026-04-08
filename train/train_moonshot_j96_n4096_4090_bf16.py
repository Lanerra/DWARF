"""
🚀 DWARF Moonshot J=96 — N=4096 context window, 4090, cold start

Architecture: D=512, H=8 (hd=64), L=8, FFN=1024, J=96 (se096@N=4096 offsets)
  L0:  DSQGBlockV6Physics  IF=False  ← pure DSQG relay
  L1:  DSQGBlockV6Physics  IF=True   ← preIF (single layer before FA)
  L2:  FullAttentionBlock            ← FA@L2 (L/4 = 8/4 = 2, zone-boundary resonance)
  L3-7: DSQGBlockV6Physics IF=False  ← 5 post-FA relay layers

Key differences from Moonshot-58M (se015, N=2048):
  - J=96 offsets (greedy, targeting N=4096; full 2-hop to N=2353, multi-layer for d>2353)
  - N=4096 sequence length (dataset: fineweb_edu_encoded_4096_v1.pt)
  - FFN=1024 (2×D, same ratio — model slightly smaller due to embedding dim scaling)
  - EMA_INIT=0.020833 (= 1/48, δ_relay_min=48, unchanged from se015)
  - SCALE_EMBED_LR_MULT=15.0 (D=512, unchanged)
  - Kernel: dsqg_attention_v8_j96_4096 (V8 with 96-offset static_range)

Offset set (J=96, greedy targeting N=4096):
  Local (δ≤28, 17 offsets, no MOVT):
    1,2,3,4,5,6,7,8,9,10,13,15,16,19,21,23,28
  Medium+Long (δ≥48, 79 offsets, MOVT applied):
    48,64,96,121,161,192,212,245,273,295,342,375,384,413,441,473,512,
    549,579,593,631,653,694,716,768,826,846,900,936,970,1000,1024,1074,
    1108,1144,1166,1190,1218,1244,1288,1322,1385,1423,1451,1497,1522,
    1550,1581,1603,1617,1634,1651,1661,1710,1743,1780,1810,1820,1852,
    1860,1876,1886,1897,1903,1916,1926,1929,1941,1965,1983,2006,2011,
    2029,2037,2044,2068,2097,2113,2199

Passkey distances tested (N=4096 regime):
  Standard: 1,2,4,8,16,32,64,128,256,512,1024,1536
  Extended: 2048,3072 (new territory vs se015)

2-hop coverage of passkey gaps (d+5):
  ✓ d=1,2,4,8,16,32,64,256,512,1024,1536,2048  (12/14)
  ✗ d=128 (gap=133, 3-hop — same as se015)
  ✗ d=3072 (gap=3077, 3-hop — multi-layer routing required)

Spectral gap: 62.4 (Z/2048Z) = 12.3× se015 → faster relay mixing

Chinchilla: ~33.5M params → 20×33.5M = 670M optimal tokens
  Per epoch at N=4096: 1,007,296 seqs × 4096 = 4.13B tokens >> optimal
  → cap at MAX_TRAIN_SEQS = 163_000 seqs (667M tokens ≈ Chinchilla optimal)
  3 epochs = 3× Chinchilla (adequate for relay crystallization + PPL convergence)

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_moonshot_j96_n4096_4090_bf16.py \\
    > logs/run_moonshot_j96_n4096.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

# J=96 greedy offset set targeting N=4096 (from se096_derivation.rs)
OFFSETS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 19, 21, 23, 28,
    48, 64, 96, 121, 161, 192, 212, 245, 273, 295, 342, 375, 384,
    413, 441, 473, 512, 549, 579, 593, 631, 653, 694, 716, 768,
    826, 846, 900, 936, 970, 1000, 1024, 1074, 1108, 1144, 1166,
    1190, 1218, 1244, 1288, 1322, 1385, 1423, 1451, 1497, 1522,
    1550, 1581, 1603, 1617, 1634, 1651, 1661, 1710, 1743, 1780,
    1810, 1820, 1852, 1860, 1876, 1886, 1897, 1903, 1916, 1926,
    1929, 1941, 1965, 1983, 2006, 2011, 2029, 2037, 2044, 2068,
    2097, 2113, 2199,
]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8          # hd = 512/8 = 64  ← binding constraint (hd≥64)
FFN_DIM          = 1024       # 2×D
NUM_LAYERS       = 8
FULL_ATTN_LAYER  = 2          # L/4 = 2, zone-boundary resonance

# Chinchilla-optimal for ~33.5M params: 20×33.5M / 4096 ≈ 163K seqs (667M tokens)
MAX_TRAIN_SEQS      = 163_000
SCALE_EMBED_INIT_VAL = 0.15   # slightly higher than Moonshot — helps percolation at J=96
SCALE_EMBED_LR_MULT  = 15.0   # μP: LR_MULT = 15×√(D/512) = 15.0 for D=512

# EMA_INIT = 1/δ_relay_min = 1/48 = 0.020833
# δ_relay_min=48 is unchanged from se015 (48 is the first non-local offset in J=96 set)
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

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

VOCAB_SIZE     = 32000
BATCH_SIZE     = 2            # pure-PyTorch DSQG is memory-heavy at N=4096
GRAD_ACCUM     = 64           # effective batch = 128
CE_CHUNK       = 256          # smaller chunks for N=4096 (longer sequences)
MAX_SEQ_LEN    = 4096
MAX_VAL_SEQS   = 2_784        # full val set from fineweb_edu_encoded_4096_v1.pt

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
]
# Extended passkey distances for N=4096 regime
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072]
PASSKEY_TRIALS    = 50
PASSKEY_BATCH_SIZE = 2        # very small — pure-PyTorch + N=4096 is memory-heavy
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'all').lower()   # checkpoint all blocks — needed for pure-PyTorch DSQG VRAM

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_j96_pytorch import DSQGAttentionJ96 as DSQGJ96, npci_rotate

assert len(OFFSETS) == 96, f"Expected 96 offsets, got {len(OFFSETS)}"


def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    return getattr(module, '_orig_mod', module)


from causal_ema_scan import causal_ema_scan as _causal_ema_scan

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


class DSQGBlockJ96(nn.Module):
    """V8-J96 DSQG attention + condV interference (EMA + AGC). J=96 offsets, N=4096."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGJ96(embedding_dim, num_heads,
                              seq_len=MAX_SEQ_LEN, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)
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


class MoonshotJ96(nn.Module):
    """Moonshot architecture with J=96 offsets and N=4096 context window."""
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer,
                 scale_embed_init_val=0.15, dropout=0.1):
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
                blocks.append(DSQGBlockJ96(
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
            if isinstance(m, DSQGJ96):
                nn.init.normal_(m.phase_base,   0.0, 0.01)
                nn.init.normal_(m.query_probes, 0.0, 0.01)
                nn.init.normal_(m.key_probes,   0.0, 0.01)
                nn.init.normal_(m.phase_gain,   0.0, 0.001)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx):
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.full_attn_layer
        return False  # 'none' default — L=8 fits without checkpointing

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
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
            if isinstance(m, DSQGJ96):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockJ96) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)


# ── Data utilities ─────────────────────────────────────────────────────────────

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
    bs = max(1, BATCH_SIZE // 2)  # smaller batches for eval at N=4096
    for i in range(0, len(data) - bs + 1, bs):
        x = data[i:i+bs, :-1].to(device, non_blocking=True)
        y = data[i:i+bs,  1:].to(device, non_blocking=True)
        with _amp_context(device):
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
            "embedding_dim": EMBEDDING_DIM, "num_heads": NUM_HEADS,
            "ffn_dim": FFN_DIM, "seq_len": MAX_SEQ_LEN,
            "source_script": "train/train_moonshot_j96_n4096_4090_bf16.py",
            "source_layer": FULL_ATTN_LAYER, "num_layers": NUM_LAYERS,
            "num_offsets": len(OFFSETS), "epoch": epoch, "git_hash": git_hash,
            "note": (f"Moonshot J=96 N=4096: D={EMBEDDING_DIM} H={NUM_HEADS} "
                     f"L={NUM_LAYERS} J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER}. "
                     f"Epoch {epoch}/3."),
        },
    }
    out_path = os.path.join(checkpoint_dir, f"moonshot_j96_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🚀 DWARF Moonshot J=96 N=4096 — D=512 H=8 hd=64 L=8 cold start')
    print('  FA@L2, preIF@L1, J=96 (se096@N=4096), EMA_INIT=1/48=0.020833')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        _cc   = torch.cuda.get_device_capability()
        _path = ('sm_90 (H100)' if ((_cc[0]==9 and _cc[1]==0) or _cc[0]>9)
                 else 'sm_89 (4090 Ada)' if (_cc[0]==8 and _cc[1]==9)
                 else f'sm_{_cc[0]}{_cc[1]}')
        print(f'  Kernel path: {_path}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  J=96 offsets, N={MAX_SEQ_LEN}, MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}')
    print(f'  scale_embed_init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  LR={LR}, epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print(f'  git={git_hash}')
    print()
    print('  NOTE: Pure-PyTorch kernel (no Triton) -- no JIT warmup needed.')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'\nLoaded tokenizer from {tok_path}')

    encoded_path = 'logs/fineweb_edu_encoded_4096_v1.pt'
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(
            f'Dataset not found: {encoded_path}\n'
            'Run reshape script first: python3 -c "import torch; ..." (see session notes)')
    print(f'Loading pre-encoded dataset from {encoded_path}')
    _cache     = torch.load(encoded_path, weights_only=True)
    train_data = _cache['train'].long()
    val_data   = _cache['val'].long()

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,} seqs  ({len(train_data)*MAX_SEQ_LEN/1e9:.2f}B tokens)')
    print(f'  val:   {len(val_data):,} seqs')

    model = MoonshotJ96(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    n_params = model.param_count()
    print(f'\nParameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_ckpt_name = 'moonshot_j96_n4096_best.pt'

    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
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
                    chunk_loss = F.cross_entropy(chunk, y_flat[chunk_start:chunk_end], reduction='sum')
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

            if step % 100 == 0:
                se_vals = [m.scale_embed.detach().abs()
                           for m in model.modules() if isinstance(m, DSQGJ96)]
                se_max  = torch.cat(se_vals).max().item() if se_vals else 0.0
                print(f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f} | SE|max|={se_max:.4f}', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state   = {k.replace('._orig_mod', ''): v
                             for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, best_ckpt_name))
            marker = ' *'

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules() if isinstance(m, DSQGJ96)]
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} '
                  f'|max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

        resume_state = {k.replace('._orig_mod', ''): v
                        for k, v in model.state_dict().items()}
        torch.save(resume_state,
                   os.path.join(CHECKPOINT_DIR, f'moonshot_j96_ep{epoch}_resume.pt'))
        print(f'  Saved resume checkpoint: moonshot_j96_ep{epoch}_resume.pt')

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s     = time.time() - t_start
    memory_mb     = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
    PPL_BASELINE     = 35.04   # Moonshot-58M se015 ep2
    PASSKEY_BASELINE = 99.2
    ar_score = (passkey_final - PASSKEY_BASELINE) + (PPL_BASELINE - ppl_final) * 0.5

    print('\n' + '=' * 70)
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score:       {ar_score:.2f}')
    print(f'memory_mb:      {memory_mb:.1f}')
    print(f'elapsed_s:      {elapsed_s:.1f}')
    print(f'num_params_M:   {n_params / 1e6:.1f}')
    print(f'num_layers:     {NUM_LAYERS}')
    print(f'num_offsets:    {len(OFFSETS)}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init:       {EMA_INIT}')
    print(f'description:    Moonshot J=96 N=4096 — D=512 H=8 L=8 FFN=1024 '
          f'J=96 se096@N4096, cold start, fineweb_edu_encoded_4096_v1, '
          f'EMA_INIT=1/delta_relay_min=1/48')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
