"""
🚀 DWARF Moonshot-58M — Mixed dataset screen (FineWeb 60% + PG19 25% + Stack 15%), 4090, cold-start

Architecture: D=512, H=8 (hd=64), L=8, FFN=2048, J=24 (se015 offsets)
  L0:  DSQGBlockV6Physics  IF=False  ← pure DSQG relay
  L1:  DSQGBlockV6Physics  IF=True   ← preIF (single layer before FA)
  L2:  FullAttentionBlock            ← FA@L2 (empirically optimal placement)
  L3-7: DSQGBlockV6Physics IF=False  ← post-FA relay layers

Config:
  - Tokenizer: fineweb_tokenizer_32k.json  (32K BPE, FineWeb proper)
                EOS id = 0  (<|endoftext|>)
  - Dataset:   mixed_encoded_2048_v1.pt (3.36M seqs, 6.88B tokens)
               60% FineWeb-Edu + 25% PG19 + 15% The Stack (dedup)
               pre-encoded with fineweb_tokenizer_32k
  - EMA_INIT = 0.0208 (= 1/48 = 1/δ_relay_min; empirically validated for J24D,
               se015 trains to α≈0.0207, 0.6% error)
  - SCALE_EMBED_INIT = 0.1, LR_MULT = 15.0
  - Batch: BS=32 × GRAD_ACCUM=4 → eff_batch=128
  - Cold start (no warm-start checkpoint)
  - ~45.6M parameters

EMA_INIT RULE: Always derive from offset set before training.
  For J24D (se015): empirically validated 1/δ_relay_min = 1/48 = 0.0208
  For J13D: 1/√(δ_local × δ_relay) = 1/√(5×8) = 0.0447 (0.4% error)
  If unsure: use 0.003 (abs+floor parameterization self-corrects, but converges slower)

TODO (audit Mar 20 2026):
  - dropout=0.1 never ablated at this scale; modern LLMs use 0.0 — worth testing
  - grad_clip=1.0 never varied — flag for future experiment
  - No LR warmup — stable so far but untested at >100M scale

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_moonshot_58m_mixed_4090_bf16.py \\
    > logs/run_moonshot_58m_mixed.log 2>&1 &
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

# Mixed dataset screen: 121K seqs (same budget as autoresearch screens for fair comparison)
# Chinchilla-optimal for ~45.6M params = 445,312 seqs; this is 27% of that.
# Available dataset: 3.36M seqs (6.88B tokens, 60% FineWeb / 25% PG19 / 15% Stack)
MAX_TRAIN_SEQS      = 121_000
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0

# EMA_INIT = 1/δ_relay_min = 1/48 ≈ 0.0208
# Empirically validated for J24D: se015 trains to α≈0.0207, error 0.6%
# δ_relay_min = 48 (first offset after local cluster [1..28])
EMA_INIT  = 0.0208
EMA_FLOOR = 0.00001

LR            = 3e-4
SCREEN_EPOCHS = 3

EXTRACTED_CKPT = None  # cold start — no warm-start checkpoint

# =============================================================================

import contextlib, json, math, os, subprocess, sys, time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

# H100: enable TF32 tensor cores for free speedup on FP32 ops (optimizer states, reductions)
torch.set_float32_matmul_precision('high')
# Reduce VRAM fragmentation — critical on large models
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

VOCAB_SIZE     = 32000
BATCH_SIZE     = 16   # 4090 24GB safe with every_other grad_ckpt
GRAD_ACCUM     = 8    # effective batch = 128
CE_CHUNK       = 512  # chunked CE token stride — avoids ~2GB fp32 gradient at vocab=32K
MAX_SEQ_LEN    = 2048
MAX_VAL_SEQS   = 5_582

FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_CANDIDATES = [
    'results/mixed_tokenizer_32k.json',     # mixed: 32K BPE on 60/25/15 FW/PG19/Stack
    'results/fineweb_tokenizer_32k.json',   # fallback: FineWeb-only tokenizer
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
PASSKEY_BATCH_SIZE = 16   # keep passkey eval memory low (was 64, OOM risk)
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

ENABLE_TORCH_COMPILE = os.getenv('DWARF_ENABLE_COMPILE', '0') == '1'  # disabled: silent C++ crash during compile init on 4090
# reduce-overhead (CUDAGraphs) is incompatible with sub-module compilation when the compiled block's
# outputs are held in the autograd graph of surrounding non-compiled (Triton) layers — the CUDAGraph
# replays and overwrites buffers that the backward still needs.  Use default (Inductor, no CUDAGraphs).
COMPILE_MODE         = os.getenv('DWARF_COMPILE_MODE', 'default')
CHECKPOINT_STRATEGY  = os.getenv('DWARF_CKPT', 'every_other').lower()  # none | every_other | all

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionV6, npci_rotate

assert len(OFFSETS) == 24



def _amp_context(device: str):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    return getattr(module, '_orig_mod', module)

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

            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)  # abs() prevents dead zone
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
                # Pre-FA IF only: IF on the single layer immediately before FA, pure DSQG everywhere else
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
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.full_attn_layer
        return False

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
            if isinstance(m, DSQGAttentionV6):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def full_attn_parameters(self):
        """Yield all parameters from the FullAttentionBlock."""
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
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR  # matches abs() parameterization
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
            if len(texts) >= 100_000:  # fallback streaming limit
                break
        os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
        with open(FW_CACHE_FILE, 'w') as fp:
            json.dump(texts, fp)
    n = len(texts)
    return {'train': texts[:int(n * 0.95)],
            'val':   texts[int(n * 0.95):int(n * 0.95) + 2500]}


def encode_split(split_texts, tokenizer, split_name):
    from tokenizers import Tokenizer as _Tokenizer
    _tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    _raw_tok = _Tokenizer.from_file(_tok_path)
    eos_id = _raw_tok.token_to_id('<|endoftext|>')  # 0 for fineweb_tokenizer_32k
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(eos_id)
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device, non_blocking=True)
        y = data[i:i+BATCH_SIZE,  1:].to(device, non_blocking=True)
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

        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        pos = torch.tensor(last_pos, dtype=torch.long, device=device)
        cand = torch.tensor(cand_rows, dtype=torch.long, device=device)

        correct = 0
        total = ids.size(0)
        for start in range(0, total, PASSKEY_BATCH_SIZE):
            ids_b = ids[start:start + PASSKEY_BATCH_SIZE]
            pos_b = pos[start:start + PASSKEY_BATCH_SIZE]
            cand_b = cand[start:start + PASSKEY_BATCH_SIZE]
            with _amp_context(device):
                logits = model(ids_b)
            row = torch.arange(ids_b.size(0), device=device)
            next_logits = logits[row, pos_b, :]
            cand_logits = torch.gather(next_logits, 1, cand_b)
            correct += (cand_logits.argmax(dim=1) == 0).sum().item()

        results[d] = correct / total
    return results


def save_full_attn_checkpoint(model, epoch, git_hash, checkpoint_dir):
    """Save just the FullAttn block weights after each epoch."""
    full_attn_block = _unwrap_compiled_module(model.blocks[model.full_attn_layer])
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
            "source_script":     "train/train_moonshot_58m_mixed_4090_bf16.py",
            "source_layer":      FULL_ATTN_LAYER,
            "num_layers":        NUM_LAYERS,
            "num_offsets":       len(OFFSETS),
            "epoch":             epoch,
            "git_hash":          git_hash,
            "note": (
                f"Moonshot-58M: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} "
                f"J={len(OFFSETS)} FA@L{FULL_ATTN_LAYER} preIF@L{FULL_ATTN_LAYER-1}. "
                f"Epoch {epoch}/3. fineweb_tokenizer_32k. Cold start."
            ),
        },
    }

    out_path = os.path.join(checkpoint_dir, f"moonshot_58m_mixed_ep{epoch}_full_attn.pt")
    torch.save(payload, out_path)
    print(f"  Saved FullAttn checkpoint: {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🚀 DWARF Moonshot-58M — D=512 H=8 hd=64 L=8 J=24, cold start')
    print('  FA@L2, preIF@L1, fineweb_tokenizer_32k, EMA_INIT=1/δ_relay_min=0.0208')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        _cc = torch.cuda.get_device_capability()
        _path = 'sm_90 (H100)' if ((_cc[0]==9 and _cc[1]==0) or _cc[0]>9) else \
                'sm_89 (4090 Ada — tuned)' if (_cc[0]==8 and _cc[1]==9) else \
                f'sm_{_cc[0]}{_cc[1]} (generic)'
        print(f'  Kernel path: {_path}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Pre-FA IF only: IF on layers < {FULL_ATTN_LAYER}, pure DSQG on layers >= {FULL_ATTN_LAYER}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
    print(f'  Batch: BS={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}  passkey_batch_size={PASSKEY_BATCH_SIZE}')
    print(f'  git={git_hash}')

    splits = load_data()
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    _encoded_cache = 'logs/mixed_encoded_2048_v4.pt'  # 60% FineWeb / 25% PG19 / 15% Stack, mixed_tokenizer_32k
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train'].long()   # int32 on disk → int64 required by cross_entropy
        val_data   = _cache['val'].long()
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
        interference_interval=None,  # unused; IF placement is hardcoded as preIF@(FA-1)
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    # ── Cold start — no warm-start checkpoint ────────────────────────────────
    best_ckpt_name = 'moonshot_58m_mixed_best.pt'
    print(f'\n  Cold start — random init, no warm-start checkpoint')

    if ENABLE_TORCH_COMPILE:
        try:
            for i, block in enumerate(model.blocks):
                if type(block).__name__ == "FullAttentionBlock":
                    try:
                        model.blocks[i] = torch.compile(block, fullgraph=False, dynamic=False, mode=COMPILE_MODE)
                    except TypeError:
                        model.blocks[i] = torch.compile(block, fullgraph=False)
                    print(f"  torch.compile applied to FullAttentionBlock at layer {i} (mode={COMPILE_MODE})")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    else:
        print('  torch.compile disabled by default (set DWARF_ENABLE_COMPILE=1 to opt in)')

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
                # Chunked CE — avoids ~4 GB fp32 gradient spike at vocab=32K
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

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss_val:.4f}', flush=True)

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            clean_state = {k.replace('._orig_mod', ''): v for k, v in model.state_dict().items()}
            torch.save(clean_state, os.path.join(CHECKPOINT_DIR, best_ckpt_name))
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

        # Save full-model resume checkpoint (weights only, no optimizer — for
        # per-epoch acquisition-order eval and future warm-starts).
        resume_state = {k.replace('._orig_mod', ''): v
                        for k, v in model.state_dict().items()}
        resume_path  = os.path.join(CHECKPOINT_DIR, f'moonshot_58m_mixed_ep{epoch}_resume.pt')
        torch.save(resume_state, resume_path)
        print(f'  Saved resume checkpoint: {resume_path}')

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
    PPL_BASELINE     = 35.04
    PASSKEY_BASELINE = 99.2
    ar_score = (passkey_final - PASSKEY_BASELINE) + (PPL_BASELINE - ppl_final) * 0.5

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
    print(f'num_offsets:    {len(OFFSETS)}')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init:       {EMA_INIT}')
    print(f'description:    Moonshot 58M — D={EMBEDDING_DIM} H={NUM_HEADS} hd=64 L={NUM_LAYERS} '
          f'FFN={FFN_DIM} J=24 se015, cold start, fineweb_tokenizer_32k, EMA_INIT=1/δ_relay_min')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
