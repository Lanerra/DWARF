"""
condX-v2 35M — Learnable Bypass Gate, scaled to ~35M (FineWeb-Edu)

Relationship to condX 13M:
  condX 13M showed a striking pattern at 13M scale:
    - Consistently better PPL than condV (+3 PPL) AND passkey developing simultaneously
    - Hard cliff at d=64: retrieval plateau at d=1-32, 0% at d≥64
    - PPL beats condV AND condU v3 baselines (51.5 at ep7, heading below 52)
    - Passkey migrates one tier per ~2 epochs (d=32→d=64 between ep5 and ep6)

  condU exhibited the same pattern at 13M (modest passkey), then exploded:
    condU 13M: 52.237 PPL / 38.3% passkey
    condU 25M: 43.434 PPL / 69.2% passkey
    condU 35M: 38.293 PPL /  90.0% passkey  ← direct comparison target

  Architecture change from condX 35M: replaces hard bypass with learnable gate,
  initialized at bypass_alpha=-10.0 (gate≈0.000045 — effectively pure contaminated Q).

Motivation from 13M gate sweep experiments (March 8, 2026):
  Both condX (hard bypass, gate=1.0 training) and condX-v2 (learnable gate, converged
  at gate≈0.10) sweep results showed gate=0 is strictly optimal for passkey:
    condX   checkpoint: peak 46.7% at gate=0 → 31.7% at gate=1.0 (monotonic decrease)
    condX-v2 checkpoint: peak 54.2% at gate=0 → 20.0% at gate=1.0 (monotonic decrease)
  The bypass costs ~20-30pp passkey regardless of what the weights were trained on.
  "Contamination" by the Huygens field IS the reference-beam learning mechanism.

  Starting at bypass_alpha=-10 (gate≈0) lets the model:
    1. Develop holographic retrieval from epoch 1 (pure contaminated Q, condV-like)
    2. Allow the gradient to discover whether any clean Q blend is useful at 35M
    3. If gradient stays near -10: confirms condV is ceiling at 35M
    4. If gradient nudges positive: 35M equilibrium differs from 13M (interesting)

Architecture:
  condX-v2 13M architecture scaled to 35M:
    EMBEDDING_DIM = 512  (was 256)
    FFN_DIM       = 2048 (was 1024)
    NUM_HEADS     = 8    (same)
    NUM_LAYERS    = 6    (same)
    INTERFERENCE  = 3    (same)
    FULL_ATTN_LAYER = 5  (same)
    bypass_alpha init = -10.0  (gate≈0 — vs -3.0 in condX-v2 13M)

  MAX_TRAIN_SEQS is computed dynamically in main() to land Chinchilla at ep7.

Baseline references:
  condU 35M:    38.293 PPL / 90.0% passkey  ← direct target
  condU v5 38M: 39.998 PPL / 98.3% passkey  ← upper bound reference
  condX-v2 13M: 52.685 PPL / 45.0% passkey  (gate=0 sweep: 54.2%)

Run (4090):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_35m_condX_v2.py \\
    2>&1 | tee benchmarks/logs/condX_v2_35m_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (identical to condV — fair comparison) ────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000

EMBEDDING_DIM   = 512
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 2048
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5

# ── Dataset config ─────────────────────────────────────────────────────────────

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
MAX_TRAIN_SEQS  = None     # computed dynamically in main() → Chinchilla at ep 7

# ── Passkey eval ───────────────────────────────────────────────────────────────

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Save paths ─────────────────────────────────────────────────────────────────

SAVE_DIR    = 'checkpoints/condX_v2_35m_bf16'
RESULT_FILE = 'benchmarks/logs/condX_v2_35m_bf16_results.json'

# ── Offset set (condN / condU / condV identical) ───────────────────────────────

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44

# ── Kernel import ──────────────────────────────────────────────────────────────

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_attention_v3 import dsqg_attention_v3


# ══════════════════════════════════════════════════════════════════════════════
# DSQG Attention (V3 — Q-weighted scale gains + IF amplifier)
# Identical to condV — no changes here.
# ══════════════════════════════════════════════════════════════════════════════

class DSQGAttentionQW(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in _COND_N_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(44, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        out = dsqg_attention_v3(q, k, v, self.pos_bias, self.scale_embed)
        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()
            se   = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean':   se.abs().mean().item(),
            'scale_embed_abs_max':    se.abs().max().item(),
            'if_gain':                gain.tolist(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Full Causal Attention — condX modification:
#   Accepts optional clean_residual for Q computation.
#   When clean_residual is provided:
#     Q ← q_proj(clean_residual)   ← uncontaminated query perspective
#     K ← k_proj(x)                ← Huygens-enriched context
#     V ← v_proj(x)                ← Huygens-enriched values
#   Q stays clean; K/V benefit from interference preprocessing.
#   Same principle as within DSQG blocks: local oscillator never contaminated.
# ══════════════════════════════════════════════════════════════════════════════

class FullCausalAttentionBypass(nn.Module):
    """
    Full causal attention with LEARNABLE bypass gate, initialized near gate=0.

    bypass_alpha = nn.Parameter(torch.tensor(-10.0))
    gate = sigmoid(bypass_alpha) ≈ 0.000045 at init → effectively pure contaminated Q.

    q_input = gate * clean_residual + (1 - gate) * x

    At init: Q is purely from x (condV-like). Gradient decides whether/how much
    to shift toward clean_residual. If gradient finds no value: stays near -10.
    If 35M finds equilibrium: bypass_alpha drifts toward it from below.

    Init at -10 (vs -3.0 in condX-v2 13M) because gate sweep showed gate=0 is
    strictly optimal for passkey — starting at zero prevents any ramp-up cost.

    Diagnostics:
      bypass_alpha — raw learned parameter
      bypass_gate  — sigmoid(bypass_alpha), effective clean blend weight
      clean_norm / full_norm / q_cosim / q_delta_norm — residual/Q diagnostics
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        D              = embedding_dim
        self.q_proj    = nn.Linear(D, D, bias=True)      # Q only
        self.kv_proj   = nn.Linear(D, 2 * D, bias=True)  # K + V together
        self.out_proj  = nn.Linear(D, D, bias=True)
        self.gate_proj = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

        # Learnable gate: sigmoid(-10) ≈ 0.000045 → effectively gate=0 at init.
        # Informed by 13M gate sweep: gate=0 is optimal for passkey on both
        # condX and condX-v2 checkpoints. Start near the optimum; let gradient
        # find the 35M equilibrium (if any) from below.
        self.bypass_alpha = nn.Parameter(torch.tensor(-10.0))

        self._last_diagnostics = {}

    def forward(self, x, clean_residual=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        if clean_residual is not None:
            alpha   = torch.sigmoid(self.bypass_alpha)
            q_input = alpha * clean_residual + (1.0 - alpha) * x
        else:
            q_input = x

        q = self.q_proj(q_input)
        k, v = self.kv_proj(x).split(D, dim=-1)

        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)

        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        result   = F.dropout(self.out_proj(out_flat * gate),
                             p=self.dropout_p, training=self.training)

        # Diagnostics
        if clean_residual is not None:
            with torch.no_grad():
                alpha_val = torch.sigmoid(self.bypass_alpha).item()
                cn    = clean_residual.norm(dim=-1).mean().item()
                fn    = x.norm(dim=-1).mean().item()
                q_full    = self.q_proj(x).view(B, N, H, HD).permute(0, 2, 1, 3)
                q_blended = self.q_proj(q_input).view(B, N, H, HD).permute(0, 2, 1, 3)
                cosim  = F.cosine_similarity(q_blended.flatten(0,2),
                                             q_full.flatten(0,2), dim=-1).mean().item()
                qdelta = (q_blended - q_full).norm(dim=-1).mean().item()
            self._last_diagnostics = {
                'bypass_alpha': self.bypass_alpha.item(),
                'bypass_gate':  alpha_val,
                'clean_norm':   cn,
                'full_norm':    fn,
                'q_cosim':      cosim,
                'q_delta_norm': qdelta,
            }
        return result

    def attn_summary(self):
        return {'type': 'full_causal_bypass',
                'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0] * NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0] * NUM_HEADS}


# ══════════════════════════════════════════════════════════════════════════════
# Receiver-chain helpers (condV interference mechanism — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_EMA_KERNEL_LEN = 256

def _causal_ema(xi, ema_factor):
    B, N, D = xi.shape
    alpha   = ema_factor.clamp(0.005, 0.5)
    k_len   = min(_EMA_KERNEL_LEN, N)
    t       = torch.arange(k_len, device=xi.device, dtype=torch.float32)
    kernel  = alpha.float() * (1.0 - alpha.float()).pow(t)
    kernel  = kernel / kernel.sum()
    kernel  = kernel.flip(0)
    xi_f    = xi.float()
    xi_bd   = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p    = F.pad(xi_bd, (k_len - 1, 0))
    pool    = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(xi.dtype)

def _kdv_correction(pool, kdv_alpha):
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    delta     = pool - pool_prev
    return pool + alpha * pool * delta

def _agc_normalize(pool, eps=1e-6):
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ══════════════════════════════════════════════════════════════════════════════
# DSQG Block (condV receiver-chain interference — unchanged from condV)
# ══════════════════════════════════════════════════════════════════════════════

class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionQW(embedding_dim, num_heads,
                                      seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor   = nn.Parameter(torch.full((1,), 0.03))
            self.kdv_alpha    = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            pool     = _causal_ema(xi, self.ema_factor)
            pool     = _kdv_correction(pool, self.kdv_alpha)
            pool     = _agc_normalize(pool)
            inter    = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta  = self.inter_k_proj(inter).view(B,N,H,HD).permute(0,2,1,3).contiguous()
            v_delta  = self.inter_v_proj(inter).view(B,N,H,HD).permute(0,2,1,3).contiguous()
            kv_inject = (k_delta, v_delta)

        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x

    def block_summary(self):
        if not self.interference:
            return {}
        return {'ema_factor': self.ema_factor.item(),
                'kdv_alpha':  self.kdv_alpha.item()}


# ══════════════════════════════════════════════════════════════════════════════
# Full Attention Block — wraps FullCausalAttentionBypass
# ══════════════════════════════════════════════════════════════════════════════

class FullAttentionBypassBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = FullCausalAttentionBypass(embedding_dim, num_heads, dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x, clean_residual=None):
        # Norm the clean residual the same way as x (for fair Q comparison)
        clean_normed = self.norm1(clean_residual) if clean_residual is not None else None
        x = x + self.attn(self.norm1(x), clean_residual=clean_normed)
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def last_diagnostics(self):
        return self.attn._last_diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# FFN (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ══════════════════════════════════════════════════════════════════════════════
# condX Transformer
#
# Layer topology (6 layers, INTERFERENCE=3, FULL_ATTN_LAYER=5):
#   0: DSQGBlock (no interference)
#   1: DSQGBlock (no interference)  ← x_clean saved AFTER this
#   2: DSQGBlock (interference=True) ← Huygens K/V computed here
#   3: DSQGBlock (no interference)
#   4: DSQGBlock (no interference)
#   5: FullAttentionBypassBlock     ← Q from x_clean, K/V from x_full
#
# The clean residual is saved after the last non-interference block
# preceding the interference block (layer 1), so it represents the
# residual state BEFORE any Huygens processing.
# ══════════════════════════════════════════════════════════════════════════════

class CondXTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer=FULL_ATTN_LAYER,
                 interference_interval=INTERFERENCE, dropout=0.1):
        super().__init__()
        self.embedding        = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed        = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop             = nn.Dropout(dropout)
        self.full_attn_layer  = full_attn_layer
        self.interference_interval = interference_interval

        # Identify where to save the clean residual:
        # The last non-interference DSQG block before the full_attn_layer.
        # With interference at (i % interval == interval-1), we find the
        # interference block index just before full_attn_layer.
        self._clean_save_after = None
        for i in range(full_attn_layer):
            if i != full_attn_layer and (i % interference_interval == interference_interval - 1):
                self._clean_save_after = i - 1   # save after block before interference

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBypassBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                blocks.append(DSQGBlock(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout,
                    interference=(i % interference_interval == interference_interval - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        # Weight-tied output projection (same pattern as condV)
        self.head   = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))

        x_clean = None
        for i, block in enumerate(self.blocks):
            if isinstance(block, FullAttentionBypassBlock):
                x = block(x, clean_residual=x_clean)
            else:
                x = block(x)
                # Save clean residual after the designated block
                if i == self._clean_save_after:
                    x_clean = x.detach().clone()

        return self.head(self.norm(x))  # weight-tied to embedding

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg_blocks  = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        full_block   = next((b for b in self.blocks
                             if isinstance(b, FullAttentionBypassBlock)), None)
        if not dsqg_blocks:
            return {}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        inter_blocks = [b for b in dsqg_blocks if b.interference]
        result = {
            'pos_bias_abs_mean': sum(s['pos_bias_abs_mean'] for s in summaries) / n,
            'pos_bias_abs_max':  max(s['pos_bias_abs_max']  for s in summaries),
            'pos_bias_mean_per_head': [
                sum(s['pos_bias_mean_per_head'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
            'scale_embed_abs_mean': sum(s['scale_embed_abs_mean'] for s in summaries) / n,
            'scale_embed_abs_max':  max(s['scale_embed_abs_max']  for s in summaries),
            'if_gain': [
                sum(s['if_gain'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
            'ema_factors': [b.ema_factor.item() for b in inter_blocks],
            'kdv_alphas':  [b.kdv_alpha.item()  for b in inter_blocks],
        }
        # Bypass diagnostics
        if full_block is not None:
            result.update(full_block.last_diagnostics)
        return result

    def layer_types(self):
        types = []
        for b in self.blocks:
            if isinstance(b, FullAttentionBypassBlock):
                types.append('FULL_BYPASS')
            elif isinstance(b, DSQGBlock):
                types.append('DSQG_INT' if b.interference else 'DSQG')
            else:
                types.append('?')
        return types


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities (identical to condV)
# ══════════════════════════════════════════════════════════════════════════════

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_data(num_docs=NUM_DOCS):
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
    else:
        from datasets import load_dataset
        print(f'Loading FineWeb-Edu ({FW_SUBSET})...')
        ds = load_dataset(FW_DATASET_NAME, name=FW_SUBSET,
                          split='train', streaming=True)
        texts = []; examined = 0
        for item in ds:
            examined += 1
            if len(item['text']) < FW_MIN_CHARS: continue
            texts.append(item['text'])
            if len(texts) % 10_000 == 0:
                print(f'  {len(texts):,} docs (examined {examined:,})')
            if len(texts) >= num_docs: break
        os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
        with open(FW_CACHE_FILE, 'w') as fp:
            json.dump(texts, fp)
    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95) : int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500 : int(n * 0.95) + 5000],
    }


def encode_split(split_texts, tokenizer, max_seq_len, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // max_seq_len) * max_seq_len
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size + 1, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=150,
             temperature=1.0, top_p=0.9):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                logits      = model(ids[:, -MAX_SEQ_LEN:])
                logits_last = logits[0, -1]
                if temperature <= 0.01:
                    next_id = logits_last.argmax()
                else:
                    probs = F.softmax(logits_last / temperature, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    mask   = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs      /= sorted_probs.sum()
                    next_id = sorted_idx[torch.multinomial(sorted_probs, 1)]
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        gen = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


def causality_check(model, device):
    print('Running causality check...')
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        out1, out2 = model(x1), model(x2)
        diff = (out1 - out2).abs()
    pre  = diff[0, :10].max().item()
    pos  = diff[0,  10].max().item()
    post = diff[0, 11:].max().item()
    print(f'  Pre-10:  {pre:.8f}  (expect 0.0)')
    print(f'  Pos-10:  {pos:.6f}  (expect >0)')
    print(f'  Post-10: {post:.6f}  (expect >0)')
    ok = pre < 1e-6
    print(f'  {"PASS" if ok else "FAIL"}')
    return ok


def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for d in PASSKEY_DISTANCES:
        correct = 0; n_valid = 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available: continue
            filler = []
            while len(filler) < d: filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN: continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target] + others[:9])[logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_data, val_data, test_data, tokenizer, device='cuda'):
    os.makedirs(SAVE_DIR, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    # BF16 — no GradScaler needed (4090 has native BF16 tensor cores; no inf/nan scan overhead)

    GEN_PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
    ]

    best_val_loss      = float('inf')
    best_val_ppl       = float('inf')
    best_epoch         = 0
    t0                 = time.time()
    per_epoch_results  = []

    tokens_per_epoch = len(train_data) * (MAX_SEQ_LEN - 1)
    chin_tokens      = 20 * model.param_count()
    chin_epoch       = chin_tokens / tokens_per_epoch
    print(f'\n  Tokens/epoch: {tokens_per_epoch:,}')
    print(f'  Chinchilla:   {chin_tokens:,} tokens (epoch ~{chin_epoch:.2f})')
    print(f'  Clean residual saved after block: {model._clean_save_after}\n')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        running_loss    = 0.0

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            scheduler.step(); step += 1
            running_loss += loss.item() * GRAD_ACCUM

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

        train_loss = running_loss / max(step, 1)
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0
        chin_pct   = epoch * tokens_per_epoch / chin_tokens * 100

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt'))
            marker = ' * BEST'

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'val_ppl': val_ppl, 'chinchilla_pct': chin_pct,
        }, os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt'))

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} '
              f'| {elapsed:.0f}s ({chin_pct:.0f}%C)')

        ss = model.attn_summary()
        head_means  = ss['pos_bias_mean_per_head']
        most_local  = int(max(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        most_global = int(min(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} '
              f'most-local=h{most_local} most-global=h{most_global}')
        print(f'  scale_embed:   |mean|={ss["scale_embed_abs_mean"]:.4f} '
              f'|max|={ss["scale_embed_abs_max"]:.4f}')
        gains = ss['if_gain']
        gain_str = '  '.join(f'h{h}:{gains[h]:.2f}' for h in range(NUM_HEADS))
        print(f'  IF gains:      {gain_str}')

        # condX-v2 bypass gate diagnostics
        if 'bypass_alpha' in ss:
            print(f'  Bypass gate:   alpha={ss["bypass_alpha"]:.4f}  '
                  f'gate={ss["bypass_gate"]:.6f}  '
                  f'(gate→0=condV-like, gate→1=fully-clean-Q)')
            print(f'  Bypass diag:   clean_norm={ss["clean_norm"]:.4f}  '
                  f'full_norm={ss["full_norm"]:.4f}  '
                  f'q_cosim={ss["q_cosim"]:.4f}  '
                  f'q_delta_norm={ss["q_delta_norm"]:.4f}')

        if ss['ema_factors']:
            ema_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['ema_factors']))
            kdv_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['kdv_alphas']))
            print(f'  EMA factors:   {ema_str}')
            print(f'  KdV alphas:    {kdv_str}')

        print('  -- Generation (T=0.7) --')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.7)):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')
        print('  --')

        print('  Passkey...')
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        above50 = sum(1 for v in pk.values() if v >= 0.5)
        print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        per_epoch_results.append({
            'epoch': epoch, 'val_ppl': val_ppl, 'train_loss': train_loss,
            'chinchilla_pct': chin_pct, 'elapsed_s': elapsed,
            'passkey_mean': pk_mean,
            'passkey_by_d': {str(d): v for d, v in pk.items()},
            'scale_embed_abs_mean': ss['scale_embed_abs_mean'],
            'scale_embed_abs_max':  ss['scale_embed_abs_max'],
            'if_gain': ss['if_gain'],
            'ema_factors': ss['ema_factors'],
            'kdv_alphas':  ss['kdv_alphas'],
            'bypass_alpha':        ss.get('bypass_alpha'),
            'bypass_gate':         ss.get('bypass_gate'),
            'bypass_clean_norm':   ss.get('clean_norm'),
            'bypass_full_norm':    ss.get('full_norm'),
            'bypass_q_cosim':      ss.get('q_cosim'),
            'bypass_q_delta_norm': ss.get('q_delta_norm'),
        })
        sys.stdout.flush()

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condX TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')
    print(f'  condV (Huygens+Kalman):  52.207 PPL | delta = {test_ppl - 52.207:+.3f}')
    print(f'  condU (V3 baseline):     52.237 PPL | delta = {test_ppl - 52.237:+.3f}')
    print(f'  condM (reference):       54.529 PPL | delta = {test_ppl - 54.529:+.3f}')

    print('\n  -- Temperature sweep (best checkpoint) --')
    sweep_results = {}
    for temp in [0.0, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        print(f'\n  [{label}]')
        gens = generate(model, tokenizer, GEN_PROMPTS, device,
                        temperature=temp, top_p=0.9)
        sweep_results[label] = gens
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')

    pk_final      = passkey_accuracy(model, tokenizer, device)
    pk_final_mean = sum(pk_final.values()) / len(pk_final)
    above50_final = sum(1 for v in pk_final.values() if v >= 0.5)
    print(f'\n  Final passkey: mean={pk_final_mean*100:.1f}%  '
          f'({above50_final}/{len(pk_final)} distances >50%)')
    parts = [f'd={d}:{int(pk_final[d]*100)}%' for d in PASSKEY_DISTANCES]
    print('  ' + '  '.join(parts))
    print(f'  condV ref:  36.7% passkey')
    print(f'  condU ref:  38.3% passkey')
    print(f'  condM ref:  83.3% passkey')

    ss_final = model.attn_summary()
    gains = ss_final['if_gain']
    gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
    print(f'\n  Final IF gains: {gain_str}')
    print(f'  Final scale_embed: |mean|={ss_final["scale_embed_abs_mean"]:.4f} '
          f'|max|={ss_final["scale_embed_abs_max"]:.4f}')
    if ss_final.get('ema_factors'):
        ema_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss_final['ema_factors']))
        kdv_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss_final['kdv_alphas']))
        print(f'  Final EMA factors: {ema_str}')
        print(f'  Final KdV alphas:  {kdv_str}')
    if 'clean_norm' in ss_final:
        print(f'  Final bypass: clean_norm={ss_final["clean_norm"]:.4f}  '
              f'full_norm={ss_final["full_norm"]:.4f}  '
              f'q_cosim={ss_final["q_cosim"]:.4f}  '
              f'q_delta_norm={ss_final["q_delta_norm"]:.4f}')

    results = {
        'experiment':          'condX_v2_35m_learnable_gate_scaled',
        'hypothesis':          'gate=0 is optimal (sweep-confirmed); 35M may find non-zero equilibrium',
        'kernel':              'dsqg_attention_v3',
        'base':                'condV (Kalman-EMA + KdV + AGC), D=512',
        'change':              'Learnable gate bypass_alpha=-10 init (gate≈0); condX-v2 arch at 35M',
        'condX_v2_13m_result': '52.685 PPL / 45.0% passkey; gate sweep peak 54.2% at gate=0',
        'gate_sweep_finding':  'gate=0 optimal on both condX and condX-v2 checkpoints; monotonic decrease to gate=1',
        'final_test_ppl':      test_ppl,
        'final_passkey_mean':  pk_final_mean,
        'final_passkey_by_d':  {str(d): v for d, v in pk_final.items()},
        'per_epoch':           per_epoch_results,
        'temperature_sweep':   sweep_results,
        'attn_summary':        ss_final,
        'condV_ppl':           52.207,
        'condV_passkey':       0.367,
        'condU_ppl':           52.237,
        'condU_passkey':       0.383,
        'condM_ppl':           54.529,
        'condM_passkey':       0.833,
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  condX-v2 35M — Learnable Bypass Gate, Scaled (FineWeb-Edu)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Kernel:    dsqg_attention_v3')
    print(f'  Base arch: condV (Kalman-EMA + KdV + AGC) + bypass gate, D=512')
    print(f'  Gate init: bypass_alpha=-10.0 → gate≈0.000045 (effectively condV Q)')
    print(f'  Sweep:     13M gate sweep showed gate=0 is strictly optimal for passkey')
    print(f'  Target:    condU 35M = 38.293 PPL / 90.0% passkey')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits = load_data(NUM_DOCS)
    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    _repo_root      = os.path.dirname(_script_dir)
    _tok_candidates = [
        os.path.join(_repo_root, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    if tok_path:
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'\nLoaded condI BPE tokenizer from {tok_path}')
    else:
        raise FileNotFoundError('condI tokenizer not found — tried: ' +
                                str(_tok_candidates))

    _encoded_cache = os.path.join(_repo_root, 'logs', 'fineweb_encoded_2048.pt')
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache} ...')
        _cache = torch.load(_encoded_cache, weights_only=True)
        train_data_full = _cache['train']
        val_data        = _cache['val']
        test_data       = _cache['test']
    else:
        print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
        train_data_full = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
        val_data        = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
        test_data       = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    # ── Build model first so we can compute Chinchilla-calibrated MAX_TRAIN_SEQS ──
    model = CondXTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params    = model.param_count()
    layer_types = model.layer_types()

    # Chinchilla-calibrate: land at ep 7 (same as condU 35M)
    # MAX_TRAIN_SEQS = (20 * params) / (NUM_EPOCHS * tokens_per_seq)
    tokens_per_seq  = MAX_SEQ_LEN - 1                     # 2047
    chin_total      = 20 * n_params                       # Chinchilla token budget
    chin_at_ep7     = chin_total / 7                      # tokens per epoch for ep-7 landing
    max_train_seqs  = int(chin_at_ep7 / tokens_per_seq)   # sequences per epoch
    max_train_seqs  = min(max_train_seqs, len(train_data_full))

    idx        = torch.randperm(len(train_data_full))[:max_train_seqs]
    train_data = train_data_full[idx]

    print(f'\ncondX 35M: {n_params:,} parameters')
    print(f'  Layer types:  {layer_types}')
    print(f'  Clean save after block: {model._clean_save_after}')
    print(f'  Chinchilla budget:  {chin_total:,} tokens  ({chin_total/1e6:.1f}M)')
    print(f'  Max train seqs:     {max_train_seqs:,}  (Chinchilla lands at ep7)')
    print(f'  Tokens/epoch:       {max_train_seqs * tokens_per_seq:,}')
    print(f'  Chinchilla ep:      {max_train_seqs * tokens_per_seq / chin_total * NUM_EPOCHS:.2f}')
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}  test: {len(test_data):,} seqs')

    if not causality_check(model, device):
        return

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
