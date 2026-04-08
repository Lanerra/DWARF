"""
🧪 DWARF D=768 L=32 — FROM-SCRATCH MIXED-DOMAIN + PERCOLATION RESET ON δ=4

Identical to train_d768_l32_mixed_scratch_h100_bf16.py EXCEPT:

  PERCOLATION RESET MECHANISM
  ───────────────────────────
  Background: Mixed-domain from-scratch training (H3) confirmed relay forms natively
  (ep2 passkey 70%), but δ=4 settles into an inhibitory state. Ablation shows:
    - With δ=4 active: d=1536 passkey = 10% (random)
    - With δ=4 zeroed: d=1536 passkey = 90% (relay path exposed but routing breaks)

  The problem: δ=4 (co-prime with δ=3) is REQUIRED to bootstrap the percolation
  cascade during ep1. But after the relay crystallises, mixed-domain gradients
  (PG19/Stack local coherence patterns) settle δ=4 into a basin that actively
  suppresses long-range relay signal.

  The fix: immediately after scale_embed |max| crosses the percolation threshold
  (~2.0) for the first time, re-initialise δ=4 weights + optimizer momentum.
  At that moment the other 23 offsets have crystallised their topology — the relay
  field in the residual stream is now strong enough to pull a freshly-initialised
  δ=4 into alignment rather than letting it drift into the inhibitory basin.

  RESET PROCEDURE (one-shot, never re-fires):
    1. Re-init all blocks' scale_embed[KILL_IDX] → constant SCALE_EMBED_INIT_VAL (0.15)
    2. Zero all blocks' pos_bias[KILL_IDX]
    3. Zero AdamW optimizer exp_avg + exp_avg_sq rows for those params (kill momentum)

  HYPOTHESIS: δ=4 will re-learn in a relay-active environment and settle into a
  constructive (or at least neutral) role, recovering d=1536 passkey to 80%+.

  FAILURE MODE: Mixed-domain gradients overpower the relay field and δ=4 drifts
  back into the inhibitory basin regardless of when it's reset.

Architecture: D=768, H=12 (hd=64), L=32, FFN=1536, J=24 (se015), TIED lm_head
Dataset:      60% FineWeb-Edu / 25% PG19 / 15% The Stack (mixed_encoded_2048_fineweb_tok.pt)
Hardware:     H200 — BS=128, GRAD_ACCUM=1, eff_batch=128

Run (from repo root, on H100 pod):
  tmux new-session -d -s d4reset \\
    ".venv/bin/python3 -u train/train_d768_l32_mixed_scratch_d4reset_h100_bf16.py \\
     2>&1 | tee logs/run_d768_l32_d4reset.log"
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

MAX_TRAIN_SEQS       = 234_418
SCALE_EMBED_INIT_VAL = 0.15
SCALE_EMBED_LR_MULT  = 20.0
EMA_INIT  = 0.010
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

# ── Percolation reset knobs ───────────────────────────────────────────────────
KILL_DELTA           = 4
KILL_IDX             = OFFSETS.index(KILL_DELTA)   # = 3
PERCOLATION_THRESHOLD = 2.0   # scale_embed |max| crossing point
# ─────────────────────────────────────────────────────────────────────────────

PASSKEY_ABORT_THRESHOLD = 0.20

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
from collections import deque
MAX_TRAIN_SEQS = int(os.environ.get('MAX_TRAIN_SEQS_OVERRIDE', MAX_TRAIN_SEQS))
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

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"


def get_gpu_peak_flops(device="cuda"):
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    if "H100" in name:   return 989e12
    elif "H200" in name: return 1979e12
    elif "4090" in name: return 330e12
    elif "3090" in name: return 142e12
    elif "A100" in name: return 312e12
    return None

VOCAB_SIZE     = 32000
BATCH_SIZE     = 128
GRAD_ACCUM     = 1
CE_CHUNK       = 512
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

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

ENABLE_TORCH_COMPILE = os.getenv('DWARF_ENABLE_COMPILE', '0') == '1'
COMPILE_MODE         = os.getenv('DWARF_COMPILE_MODE', 'reduce-overhead')
CHECKPOINT_STRATEGY  = os.getenv('DWARF_CKPT', 'every_other').lower()
PASSKEY_BATCH_SIZE   = int(os.getenv('DWARF_PASSKEY_BATCH', '32'))

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_h100 import DSQGAttentionV8_H100 as DSQGAttentionV8, npci_rotate
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

USE_V13 = os.getenv('DWARF_USE_V13', '0') == '1'
USE_V12 = os.getenv('DWARF_USE_V12', '0') == '1'
USE_V11 = os.getenv('DWARF_USE_V11', '0') == '1'

if USE_V13:
    from dsqg_attention_v13 import DSQGAttentionV13
    DSQGAttentionV6 = DSQGAttentionV13
    print(f"Using DSQG Attention V13 (DWARF_USE_V13=1)")
elif USE_V12:
    from dsqg_attention_v12 import DSQGAttentionV12
    DSQGAttentionV6 = DSQGAttentionV12
    print(f"Using DSQG Attention V12 (DWARF_USE_V12=1)")
elif USE_V11:
    from dsqg_attention_v11 import DSQGAttentionV11
    DSQGAttentionV6 = DSQGAttentionV11
    print(f"Using DSQG Attention V11 (DWARF_USE_V11=1)")
else:
    DSQGAttentionV6 = DSQGAttentionV8
    print(f"Using DSQG Attention V8 (default)")

assert len(OFFSETS) == 24


def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    return getattr(module, '_orig_mod', module)


def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)


def _agc_normalize(pool, eps=1e-6):
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Percolation reset ─────────────────────────────────────────────────────────

def fire_percolation_reset(model, optimizer, kill_idx, scale_embed_init_val):
    """
    One-shot reinitialisation of δ=4 (index kill_idx) weights and optimizer
    momentum across all DSQG layers.

    Called immediately when scale_embed |max| first crosses PERCOLATION_THRESHOLD.
    At that moment the relay topology in offsets 0–2, 4–23 is crystallised;
    resetting kill_idx lets it re-learn against an active relay field.
    """
    print(f'\n  ════════════════════════════════════════════════════════════')
    print(f'  🔄 PERCOLATION RESET — reinitialising δ={KILL_DELTA} (idx={kill_idx})')
    print(f'  Relay topology crystallised; δ={KILL_DELTA} will re-learn from scratch')
    print(f'  ════════════════════════════════════════════════════════════')

    reset_count = 0
    for block in model.blocks:
        raw_block = _unwrap_compiled_module(block)
        if not isinstance(raw_block, DSQGBlockV6Physics):
            continue
        attn = raw_block.attn
        if not isinstance(attn, DSQGAttentionV6):
            continue

        with torch.no_grad():
            # scale_embed[kill_idx]: reinit to same constant used at model init
            attn.scale_embed[kill_idx].fill_(scale_embed_init_val)
            # pos_bias[kill_idx]: zero out (original init is zero)
            attn.pos_bias[kill_idx].zero_()

        # Wipe optimizer momentum for scale_embed param at this row
        se_param = attn.scale_embed
        if se_param in optimizer.state:
            st = optimizer.state[se_param]
            for key in ('exp_avg', 'exp_avg_sq',
                        'exp_avg_sq_row', 'exp_avg_sq_col'):  # covers 8bit variants
                if key in st:
                    if st[key].dim() >= 1:
                        st[key][kill_idx].zero_()

        # Wipe optimizer momentum for pos_bias param at this row
        pb_param = attn.pos_bias
        if pb_param in optimizer.state:
            st = optimizer.state[pb_param]
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in st:
                    if st[key].dim() >= 1:
                        st[key][kill_idx].zero_()

        reset_count += 1

    print(f'  Reset {reset_count} DSQG layers: scale_embed[{kill_idx}]→{scale_embed_init_val} '
          f'pos_bias[{kill_idx}]→0, optimizer momentum zeroed\n')
    return reset_count


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
        if block_idx == self.full_attn_layer:
            return True
        if block_idx == self.full_attn_layer - 1:
            return True
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

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win   = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)

    def get_kill_idx_scale_embed_stats(self, kill_idx):
        """Return (mean, max) of |scale_embed[kill_idx]| across all DSQG layers."""
        vals = []
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                vals.append(m.scale_embed[kill_idx].detach().abs().mean().item())
        return (sum(vals) / len(vals), max(vals)) if vals else (0.0, 0.0)


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
def evaluate(model, data, device, CE_CHUNK=512):
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


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir, suffix=''):
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
            "source_script": "train/train_d768_l32_mixed_scratch_d4reset_h100_bf16.py",
            "source_layer":  FULL_ATTN_LAYER,
            "num_layers":    NUM_LAYERS,
            "num_offsets":   len(OFFSETS),
            "epoch":         epoch,
            "git_hash":      git_hash,
            "kill_delta":    KILL_DELTA,
            "kill_idx":      KILL_IDX,
            "percolation_threshold": PERCOLATION_THRESHOLD,
        },
    }
    tag = f'_ep{epoch}{suffix}'
    out_path = os.path.join(checkpoint_dir, f'd768_l32_mixed_d4reset{tag}_full_attn.pt')
    torch.save(payload, out_path)
    print(f'  Saved FullAttn checkpoint: {out_path}')


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start  = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DWARF D768-L32 FA@L8 — MIXED + PERCOLATION RESET ON δ=4')
    print(f'  Kill offset: δ={KILL_DELTA} (index {KILL_IDX})')
    print(f'  Reset fires when: scale_embed |max| ≥ {PERCOLATION_THRESHOLD}')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Liger CE: {"enabled" if USE_LIGER_CE else "disabled (chunked CE fallback)"}')
    print(f'  bitsandbytes AdamW8bit: {"enabled" if _BNB_AVAILABLE else "disabled (standard AdamW)"}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  FA@L{FULL_ATTN_LAYER}, preIF@L{FULL_ATTN_LAYER-1}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR_MULT={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found in: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}  (vocab={tokenizer.vocab_size():,})')

    _encoded_cache = 'logs/mixed_encoded_2048_fineweb_tok.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading mixed dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train'].long()
        val_data   = _cache['val'].long()
    else:
        raise FileNotFoundError(f'Mixed dataset not found: {_encoded_cache}')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
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

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    _AdamW = bnb.optim.AdamW8bit if _BNB_AVAILABLE else torch.optim.AdamW
    optimizer = _AdamW([
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

    # ── Kernel warmup ─────────────────────────────────────────────────────────
    _WARMUP_BS = min(BATCH_SIZE, 16)  # limit warmup to avoid OOM during JIT
    print(f'Warming up Triton kernels (BS={_WARMUP_BS})...')
    _wb = min(_WARMUP_BS, len(train_data))
    _wx = train_data[:_wb, :-1].to(device)
    _wy = train_data[:_wb, 1:].to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if USE_LIGER_CE:
            _liger_ce_fn_warmup = LigerFusedLinearCrossEntropyLoss()
            _whidden = model.forward_hidden(_wx)
            _wloss = _liger_ce_fn_warmup(
                _whidden.view(-1, _whidden.size(-1)), model.out.weight, _wy.view(-1))
            _wloss.backward()
            del _whidden, _wloss
        else:
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
            del _wout, _wlogits_flat, _wy_flat, _wloss
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy
    torch.cuda.synchronize()
    print('  Kernel warmup complete.')

    gpu_peak_flops  = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    flops_per_step  = 6 * n_params * tokens_per_step
    mfu_window      = deque(maxlen=20)
    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss()

    # ── Percolation reset state ───────────────────────────────────────────────
    percolation_reset_fired = False
    reset_fired_at_step     = None
    reset_fired_at_epoch    = None

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

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
                        loss = liger_ce_fn(
                            hidden.view(-1, hidden.size(-1)),
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
                    total_ce    = 0.0
                    for cs in range(0, T, CE_CHUNK):
                        ce      = min(cs + CE_CHUNK, T)
                        chunk   = logits_flat[cs:ce].detach().requires_grad_(True)
                        cl      = F.cross_entropy(chunk, y_flat[cs:ce], reduction='sum')
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

            # ── Per-step percolation threshold check ──────────────────────────
            if not percolation_reset_fired:
                se_max_live = max(
                    m.scale_embed.detach().abs().max().item()
                    for m in model.modules()
                    if isinstance(m, DSQGAttentionV6)
                )
                if se_max_live >= PERCOLATION_THRESHOLD:
                    fire_percolation_reset(
                        model, optimizer,
                        kill_idx=KILL_IDX,
                        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
                    )
                    percolation_reset_fired = True
                    reset_fired_at_step  = step
                    reset_fired_at_epoch = epoch
                    print(f'  [reset] percolation_reset_fired=True '
                          f'(ep{epoch} step {step}/{steps_per_epoch}, '
                          f'SE |max|={se_max_live:.4f})', flush=True)

            # ── Periodic logging ──────────────────────────────────────────────
            if step % 200 == 0:
                se_vals = [m.scale_embed.detach().abs()
                           for m in model.modules() if isinstance(m, DSQGAttentionV6)]
                se_all  = torch.cat(se_vals)
                se_max  = se_all.max().item()
                # Also show kill-idx row specifically
                kill_se_mean, kill_se_max = model.get_kill_idx_scale_embed_stats(KILL_IDX)

                threshold_marker = ' ✓ ABOVE' if se_max >= PERCOLATION_THRESHOLD else (
                    ' ↑ approaching' if se_max >= 1.6 else '')
                reset_marker = (
                    f' [RESET fired ep{reset_fired_at_epoch} s{reset_fired_at_step}]'
                    if percolation_reset_fired else ' [reset pending]'
                )
                avg_ms  = sum(mfu_window) / len(mfu_window)
                tok_s   = tokens_per_step / (avg_ms / 1000.0)
                mfu_str = ''
                if gpu_peak_flops:
                    mfu = (flops_per_step / (avg_ms / 1000.0)) / gpu_peak_flops * 100
                    mfu_str = f' | MFU {mfu:.1f}%'
                print(
                    f'  Ep{epoch} Step {step}/{steps_per_epoch} '
                    f'| Loss {loss_val:.4f} '
                    f'| SE |max|={se_max:.4f}{threshold_marker}'
                    f' | δ={KILL_DELTA} SE mean={kill_se_mean:.4f} max={kill_se_max:.4f}'
                    f'{reset_marker}'
                    f'{mfu_str} | {tok_s:.0f} tok/s',
                    flush=True,
                )

        # ── End-of-epoch eval ─────────────────────────────────────────────────
        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v
                           for k, v in model.state_dict().items()}
            torch.save(clean_state,
                       os.path.join(CHECKPOINT_DIR, 'd768_l32_mixed_d4reset_best.pt'))
            marker = ' *'

        torch.save({
            'epoch':               epoch,
            'model_state_dict':    {k.replace('._orig_mod', ''): v
                                    for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':            val_loss,
            'val_ppl':             val_ppl,
            'percolation_reset_fired':   percolation_reset_fired,
            'reset_fired_at_step':       reset_fired_at_step,
            'reset_fired_at_epoch':      reset_fired_at_epoch,
        }, os.path.join(CHECKPOINT_DIR, f'd768_l32_mixed_d4reset_ep{epoch}.pt'))

        print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = [m.scale_embed.detach().abs()
                   for m in model.modules() if isinstance(m, DSQGAttentionV6)]
        if se_vals:
            se_all = torch.cat(se_vals)
            se_max = se_all.max().item()
            kill_se_mean, kill_se_max = model.get_kill_idx_scale_embed_stats(KILL_IDX)
            print(f'  scale_embed |max|={se_max:.4f} '
                  f'| δ={KILL_DELTA} SE mean={kill_se_mean:.4f} max={kill_se_max:.4f}'
                  f'{"  [reset fired at ep" + str(reset_fired_at_epoch) + " s" + str(reset_fired_at_step) + "]" if percolation_reset_fired else "  [reset not yet fired]"}')

        print(f'  Physics: {model.physics_summary()}')
        save_full_attn_checkpoint(model, epoch, git_hash, CHECKPOINT_DIR)

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        if epoch >= 2 and pk_mean < PASSKEY_ABORT_THRESHOLD:
            print(f'\n  ⛔ ABORT: Passkey {pk_mean*100:.1f}% < {PASSKEY_ABORT_THRESHOLD*100:.0f}% '
                  f'— relay formation failed.')
            break

        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_s = time.time() - t_start
    memory_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    passkey_final = passkey_results.get(max(passkey_results.keys(), default=1), 0.0)
    ppl_final     = ppl_results.get(max(ppl_results.keys(), default=1), 999.0)
    PPL_BASELINE     = 35.04
    PASSKEY_BASELINE = 99.2
    ar_score = (passkey_final - PASSKEY_BASELINE) + (PPL_BASELINE - ppl_final) * 0.5

    print('\n--- RESULTS ---')
    print(f'percolation_reset_fired: {percolation_reset_fired}')
    if percolation_reset_fired:
        print(f'reset_fired_at: ep{reset_fired_at_epoch} step{reset_fired_at_step}')
    for ep in sorted(passkey_results):
        print(f'passkey_ep{ep}: {passkey_results[ep]:.1f}%')
    for ep in sorted(ppl_results):
        print(f'ppl_ep{ep}: {ppl_results[ep]:.2f}')
    print(f'ar_score: {ar_score:.2f}')
    print(f'memory_mb: {memory_mb:.1f}')
    print(f'elapsed_s: {elapsed_s:.1f}')
    print(f'description: D768-L32-FA8 MIXED+D4RESET '
          f'D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} FFN={FFN_DIM} '
          f'FA@L{FULL_ATTN_LAYER} kill_delta={KILL_DELTA} threshold={PERCOLATION_THRESHOLD}')


if __name__ == '__main__':
    train()
