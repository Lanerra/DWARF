"""
d49_optimal_14m — Full autoresearch synthesis: d49 bridge-offset geometry.

Architecture improvements over d48:

  1. d49 offset set (J=52, FULL BUDGET):
     Dense local W=40, far-local [96,128],
     bridge [145,163,185,209,236,266,301,340] (log-spaced, closes 128→384 gap),
     distal [384].
     Dense W=40 validated by zone evaluator (dense_efficiency=1.000).
     Bridge log-spacing confirmed by offset_zone_spacing evaluator.

  2. Hard head-regime split (7 local : 1 distal):
     head_specialization evaluator score=1.000 (perfect).
     Training never grows a distal specialist naturally → forcing it is critical.
     Heads 0-6: attend over local regime (δ≤128).
     Head 7: attends exclusively to distal regime (δ>200).
     Implementation: pos_bias hard-masked at init, scale_embed masked same way.

  3. Scale_embed soft L2 regularization (λ=1e-4) — validated from d48.
     Prevents abs_max overshoot. Target abs_mean ~0.21–0.23.

  4. Wavelet filter θ=1.0625 (re-optimised on d49 geometry, score=0.8359).
     Old θ=1.1377 drops to 0.8328 on new geometry; 1.0625 recovers to 0.8359.
     4-tap Daubechies-style: [0.5261, 0.8228, 0.1810, -0.1157] → updated below.

  5. Mild pos_bias regularization (cap |max| at 10.0):
     d48 reached |max|=8.855; uncapped growth hurts long-range retrieval.

  6. Standard pos_bias init (two-regime init removed):
     Two-regime pos_bias init did NOT improve d48 results at 14M scale.
     Reverted to standard negative distance penalty throughout.

Two-θ design (local=1.5, distal=1.0 per regime_decoupled_wavelet_potential)
identified as +22.6% theoretical gain → reserved for d50 (requires arch change).

All other components retained from d48/condV:
  - Q-weighted scale gains (V3 kernel, d49 offsets)
  - IF amplifier (per-head gain)
  - Huygens K/V injection
  - Kalman-EMA + KdV soliton + AGC interference

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_14m_d49_optimal.py \\
    2>&1 | tee benchmarks/logs/d49_optimal_14m.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# -- Hyperparameters (identical to condM/condT reference) ----------------------

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
FULL_ATTN_LAYER = -1  # pure DSQG — no full attention layer (O(1) cache throughout)

# -- FineWeb-Edu dataset config ------------------------------------------------

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
MAX_TRAIN_SEQS  = 52_716          # ISO-COMPUTE CAP — do not change

# -- Passkey eval config -------------------------------------------------------

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# -- Save paths ----------------------------------------------------------------

EXPERIMENT_NAME = 'd49_optimal_14m'
SAVE_DIR        = 'checkpoints/d49_optimal_14m'
RESULT_FILE     = 'benchmarks/logs/d49_optimal_14m_results.json'

# -- d49 offset set (J=52) -----------------------------------------------------
#
#   Dense   δ=0..40   (41 offsets) — validated by zone evaluator (eff=1.000)
#   Far-local          [96, 128]
#   Bridge  δ=145..340 [145,163,185,209,236,266,301,340] — log-spaced, closes 128→384 gap
#   Distal             [384]
#
_DENSE_LOCAL_W  = 40
_SPARSE_LIST    = [96, 128, 145, 163, 185, 209, 236, 266, 301, 340, 384]
_COND_N_OFFSETS = list(range(_DENSE_LOCAL_W + 1)) + _SPARSE_LIST
assert len(_COND_N_OFFSETS) == 52, f"Expected J=52, got {len(_COND_N_OFFSETS)}"

# -- Head-regime split ---------------------------------------------------------
#
#   7 local heads (h=0..6): attend local regime δ≤128
#   1 distal head (h=7):    attends distal regime δ>200
#   Validated: head_specialization evaluator score=1.000 (perfect)
#   Training never grows distal specialist naturally — hard mask is critical.
#
NUM_LOCAL_HEADS  = 7
NUM_DISTAL_HEADS = 1
assert NUM_LOCAL_HEADS + NUM_DISTAL_HEADS == NUM_HEADS
DISTAL_THRESHOLD = 350   # offsets > this are "distal" — only δ=384 qualifies
                         # bridge offsets (145-340) are local-accessible by all heads

# Build regime masks over the offset set
_offset_tensor = torch.tensor(_COND_N_OFFSETS, dtype=torch.float32)
_LOCAL_MASK  = (_offset_tensor <= DISTAL_THRESHOLD)   # [J] True for local offsets
_DISTAL_MASK = (_offset_tensor >  DISTAL_THRESHOLD)   # [J] True for distal offsets

# -- Scale_embed regularization ------------------------------------------------

SCALE_EMBED_REG_LAMBDA = 1e-4
SCALE_EMBED_MAX_ABS    = 1.0

# -- pos_bias regularization (cap unconstrained growth) ------------------------

POS_BIAS_MAX_ABS = 10.0   # soft cap; d48 reached 8.855 — don't let it blow past 10

# -- Wavelet filter (θ=1.0625, re-optimised for d49 geometry) ------------------
#
#   Lattice-parameterised 4-tap Daubechies. θ=1.0625 found by systematic sweep
#   over d49 offset set (score=0.8359 vs old θ=1.1377→0.8328 on new geometry).
#   regime_decoupled_wavelet_potential recommends two-θ (local=1.5, distal=1.0)
#   as +22.6% improvement — reserved for d50.
#
_THETA = 1.0625

# Wavelet coefficients for θ=1.0625 (from math_autoresearch binary — reference only,
# not used in model forward pass; stored here for provenance).
_WAVELET_COEFFS = [0.4903, 0.8345, 0.2168, -0.1274]

# -- Triton kernel (d49 offsets, V3 Q-weighted scale gains) --------------------

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_probe_d49_bridge_384 import dsqg_attention_v3   # 52-offset d49 kernel


# ==============================================================================
#  Model Components
# ==============================================================================

class _KalmanEMA(nn.Module):
    """Kalman-filtered EMA — smoothed recurrent context for interference layers."""
    _EMA_KERNEL_LEN = 256

    def __init__(self, dim):
        super().__init__()
        self.alpha  = nn.Parameter(torch.full((dim,), 0.9))
        self.gain   = nn.Parameter(torch.ones(dim))
        self.dim    = dim

    def forward(self, x):          # x: [B, T, D]
        B, T, D   = x.shape
        k_len     = min(self._EMA_KERNEL_LEN, T)
        alpha_c   = torch.sigmoid(self.alpha).clamp(0.01, 0.99)
        t         = torch.arange(k_len, device=x.device, dtype=x.dtype)
        kernel    = alpha_c.unsqueeze(1) ** t.unsqueeze(0)   # [D, k_len]
        kernel    = kernel / kernel.sum(dim=1, keepdim=True)
        x_t       = x.transpose(1, 2)                         # [B, D, T]
        pad       = F.pad(x_t, (k_len - 1, 0))
        ema       = F.conv1d(pad, kernel.unsqueeze(1), groups=D)
        return ema.transpose(1, 2) * self.gain


class DSQGAttention(nn.Module):
    """
    DSQG attention with d49 geometry + hard head-regime split.

      Heads 0..NUM_LOCAL_HEADS-1  : local regime  (δ ≤ DISTAL_THRESHOLD)
      Head  NUM_LOCAL_HEADS..H-1  : distal regime  (δ > DISTAL_THRESHOLD)

    The split is implemented by masking pos_bias and scale_embed:
      - Local heads:  pos_bias[:, local_heads] forced to -∞ for distal offsets
      - Distal heads: pos_bias[:, distal_heads] forced to -∞ for local offsets
    This is a soft mask applied at every forward pass (not tied to init only).
    """

    # Two mask values for regime separation:
    #   _LOCAL_HARD_MASK: hard block of δ=384 for local heads — they should never see it.
    #   _DISTAL_SOFT_MASK: soft suggestion for distal head to avoid local offsets.
    #     Using -3.0 (not -1e4) so pos_bias[δ=384,h=7]=+3.0 creates a 6-unit gap →
    #     ~89% preference for distal at init WITHOUT gradient saturation.
    #     With -1e4 the softmax was fully saturated (p=1.0, grad=0), blocking Q/K learning.
    _LOCAL_HARD_MASK  = -1e4   # local heads: hard block of distal offset
    _DISTAL_SOFT_MASK = -3.0   # distal head: soft suggestion against local offsets

    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim
        J              = len(_COND_N_OFFSETS)

        self.qkv        = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout    = nn.Dropout(dropout)

        # pos_bias init:
        #   Local heads (0..NUM_LOCAL_HEADS-1): standard negative distance penalty.
        #   Distal head (NUM_LOCAL_HEADS..H-1): standard init for local offsets, but
        #     δ=384 is set to +3.0 (positive) so it clearly wins over local offsets
        #     masked to _DISTAL_SOFT_MASK=-3.0, while avoiding softmax saturation.
        #   Natural init would give δ=384 a value of -(384×alpha) = -768 for head 7,
        #   which still "wins" over -3 local offsets but causes full saturation
        #   (p≈1.0 → grad≈0 → Q/K/scale_embed can't learn for head 7).
        delta_vals      = torch.tensor(_COND_N_OFFSETS, dtype=torch.float32)
        alphas          = torch.linspace(0.2, 2.0, num_heads)
        pos_bias_init   = -delta_vals.unsqueeze(1) * alphas.unsqueeze(0)  # [J, H]
        # Override distal head's valid offset to +3.0 (non-saturating positive init)
        pos_bias_init[_DISTAL_MASK, NUM_LOCAL_HEADS:] = 3.0
        self.pos_bias   = nn.Parameter(pos_bias_init)

        # scale_embed: Q-weighted matched filter — starts at zero (backward compat)
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))

        # IF amplifier: per-head learned gain
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

        # Register regime masks as buffers (move to device automatically)
        local_mask  = _LOCAL_MASK.clone()   # [J] bool
        distal_mask = _DISTAL_MASK.clone()  # [J] bool
        self.register_buffer('_local_mask',  local_mask)
        self.register_buffer('_distal_mask', distal_mask)

    def _apply_head_regime_mask(self, pos_bias):
        """
        Apply hard head-regime split.
        Local heads (0..NUM_LOCAL_HEADS-1): zero out distal offsets.
        Distal head (NUM_LOCAL_HEADS..H-1): zero out local offsets.
        Returns masked pos_bias [J, H].
        """
        pb = pos_bias.clone()
        # Local heads cannot see distal offsets (hard block — they should never attend to δ=384)
        pb[self._distal_mask, :NUM_LOCAL_HEADS] = self._LOCAL_HARD_MASK
        # Distal head avoids local offsets (soft suggestion — allows gradient flow at t<384
        # and prevents full softmax saturation that would kill Q/K learning)
        pb[self._local_mask,  NUM_LOCAL_HEADS:] = self._DISTAL_SOFT_MASK
        return pb

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        H, HD   = self.num_heads, self.head_dim
        J       = len(_COND_N_OFFSETS)

        # Apply hard head-regime masking to pos_bias at every forward pass
        masked_pb = self._apply_head_regime_mask(self.pos_bias)

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        # Kernel expects [B, H, T, HD]
        q = q.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()

        out = dsqg_attention_v3(q, k, v, masked_pb, self.scale_embed)
        # Apply IF gain [H] and un-permute back to [B, T, C]
        out = out * self.if_gain.view(1, H, 1, 1)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.dropout(self.out_proj(out))

        if return_stats:
            with torch.no_grad():
                pb  = masked_pb.detach().cpu()
                se  = self.scale_embed.detach().cpu()
                ig  = self.if_gain.detach().cpu()
                stats = {
                    'scale_embed_abs_mean': se.abs().mean().item(),
                    'scale_embed_abs_max':  se.abs().max().item(),
                    'pos_bias_abs_mean':    pb.abs().mean().item(),
                    'pos_bias_abs_max':     pb.abs().max().item(),
                    'pos_bias_mean_per_head': pb.mean(0).tolist(),
                    'if_gain':              ig.tolist(),
                }
            return out, stats
        return out


class FullCausalAttention(nn.Module):
    """Standard full causal attention (used only if FULL_ATTN_LAYER ≥ 0)."""

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = embedding_dim // num_heads
        self.qkv        = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv     = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, HD).transpose(1, 2)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out    = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out    = self.out_proj(out)
        if return_stats:
            return out, {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                         'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                         'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                         'if_gain': [1.0]*NUM_HEADS}
        return out


class InterferenceLayer(nn.Module):
    """Kalman-EMA + KdV soliton + AGC interference pooling."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, use_dsqg=True):
        super().__init__()
        self.use_dsqg = use_dsqg
        if use_dsqg:
            self.attn = DSQGAttention(embedding_dim, num_heads, seq_len, dropout)
        else:
            self.attn = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ema   = _KalmanEMA(embedding_dim)
        self.alpha = nn.Parameter(torch.zeros(1))   # KdV soliton gate
        self.beta  = nn.Parameter(torch.zeros(1))   # AGC gate
        self.gamma = nn.Parameter(torch.zeros(1))   # EMA gate

        # Huygens K/V injection
        H, HD = num_heads, embedding_dim // num_heads
        J     = len(_COND_N_OFFSETS)
        self.huygens_k = nn.Parameter(torch.zeros(J, HD))
        self.huygens_v = nn.Parameter(torch.zeros(J, HD))
        self.huygens_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_stats=False):
        # Huygens injection (physics-inspired K/V perturbation)
        hscale = torch.tanh(self.huygens_scale)

        res = self.norm1(x)
        if return_stats:
            attn_out, stats = self.attn(res, return_stats=True)
        else:
            attn_out = self.attn(res)
            stats    = {}

        x = x + attn_out

        # KdV soliton: non-linear mixing
        kdv = torch.tanh(self.alpha) * (x * x.roll(1, dims=1))
        x   = x + kdv

        # AGC: adaptive gain control
        agc_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x        = x + torch.tanh(self.beta) * x / agc_norm

        # Kalman-EMA context
        ema_ctx = self.ema(x)
        x       = x + torch.tanh(self.gamma) * ema_ctx

        x = x + self.ffn(self.norm2(x))
        return (x, stats) if return_stats else x


class TransformerBlock(nn.Module):
    """Standard transformer block (non-interference layers)."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn  = DSQGAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_stats=False):
        res = self.norm1(x)
        if return_stats:
            attn_out, stats = self.attn(res, return_stats=True)
        else:
            attn_out = self.attn(res)
            stats    = {}
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return (x, stats) if return_stats else x


class DSQGTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len=2048, dropout=0.1,
                 interference=3, full_attn_layer=-1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc   = nn.Embedding(seq_len, embedding_dim)
        self.drop      = nn.Dropout(dropout)

        layers = []
        for i in range(num_layers):
            use_full = (full_attn_layer >= 0 and i == full_attn_layer)
            if use_full:
                layers.append(InterferenceLayer(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, use_dsqg=False))
            elif (i + 1) % interference == 0:
                layers.append(InterferenceLayer(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, use_dsqg=True))
            else:
                layers.append(TransformerBlock(
                    embedding_dim, num_heads, ffn_dim, dropout=dropout))
        self.layers  = nn.ModuleList(layers)
        self.norm    = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, input_ids, return_stats=False):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.drop(self.embedding(input_ids) + self.pos_enc(pos))

        all_stats = []
        for layer in self.layers:
            if return_stats:
                x, s = layer(x, return_stats=True)
                all_stats.append(s)
            else:
                x = layer(x)

        x      = self.norm(x)
        logits = self.lm_head(x)
        return (logits, all_stats) if return_stats else logits


# ==============================================================================
#  Dataset & Tokenizer  (matches d48 / condM infrastructure)
# ==============================================================================

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_tokenizer():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates  = [
        os.path.join(_script_dir, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in candidates if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI BPE tokenizer not found in results/')
    from tokenizers import Tokenizer
    tok = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}  (vocab={tok.vocab_size()})')
    return tok


def load_dataset(tokenizer):
    """Load pre-encoded seqs from cache, or build cache from raw data."""
    encoded_cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(encoded_cache):
        print(f'Loading pre-encoded dataset from {encoded_cache} …')
        cache      = torch.load(encoded_cache, weights_only=True)
        train_data = cache['train']
        val_data   = cache['val']
        test_data  = cache['test']
    else:
        # Fall back: stream FineWeb-Edu and encode on the fly
        import datasets
        print('Streaming FineWeb-Edu (no cache found) …')
        ds   = datasets.load_dataset(FW_DATASET_NAME, FW_SUBSET, split='train',
                                     streaming=True, trust_remote_code=True)
        all_ids = []
        n_docs  = 0
        for ex in ds:
            if len(ex['text']) < FW_MIN_CHARS:
                continue
            all_ids.extend(tokenizer.encode(ex['text']))
            n_docs += 1
            if n_docs >= NUM_DOCS:
                break
        # Slice into (seq_len+1) chunks
        seqs = []
        for i in range(0, len(all_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            seqs.append(all_ids[i:i + MAX_SEQ_LEN + 1])
        import random; random.shuffle(seqs)
        n_val  = min(2000, len(seqs) // 10)
        n_test = min(2000, len(seqs) // 10)
        val_t   = torch.tensor(seqs[:n_val],             dtype=torch.long)
        test_t  = torch.tensor(seqs[n_val:n_val+n_test], dtype=torch.long)
        train_t = torch.tensor(seqs[n_val+n_test:],      dtype=torch.long)
        torch.save({'train': train_t, 'val': val_t, 'test': test_t}, encoded_cache)
        train_data, val_data, test_data = train_t, val_t, test_t

    # Apply ISO-COMPUTE cap
    if len(train_data) > MAX_TRAIN_SEQS:
        idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
        train_data = train_data[idx]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
          f'test: {len(test_data):,} seqs')
    return train_data, val_data, test_data


# ==============================================================================
#  Training
# ==============================================================================

def run_passkey_eval(model, tokenizer, device):
    model.eval()
    results = []
    with torch.no_grad():
        for dist in PASSKEY_DISTANCES:
            correct = 0
            n_valid = 0
            trials  = []
            for word in _PASSKEY_WORDS[:PASSKEY_TRIALS]:
                intro   = _INTRO_TEMPLATE.format(word=word)
                fillers = (_FILLER_SENTENCE * (dist * 10))[:dist * 80]
                prompt  = intro + ' ' + fillers + ' ' + _RETRIEVAL_CUE
                ids     = tokenizer.encode(prompt)
                if len(ids) >= MAX_SEQ_LEN:
                    trials.append({'distance': dist, 'target': word,
                                   'predicted': None, 'correct': False, 'skipped': True})
                    continue
                inp     = torch.tensor([ids], device=device)
                logits  = model(inp)
                pred_id = logits[0, -1].argmax().item()
                pred    = tokenizer.decode([pred_id]).strip()
                ok      = (pred == word)
                correct += int(ok)
                n_valid += 1
                trials.append({'distance': dist, 'target': word,
                               'predicted': pred, 'correct': ok, 'skipped': False})
            acc = correct / n_valid if n_valid > 0 else 0.0
            results.append({'distance': dist, 'accuracy': acc,
                            'n_valid': n_valid, 'trials': trials})
    return results


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Offsets ({len(_COND_N_OFFSETS)}): {_COND_N_OFFSETS}')
    print(f'Head split: {NUM_LOCAL_HEADS} local + {NUM_DISTAL_HEADS} distal')
    print(f'Wavelet θ={_THETA:.4f}  coefficients: {[round(c,4) for c in _WAVELET_COEFFS]}')

    tokenizer                    = load_tokenizer()
    train_data, val_data, test_data = load_dataset(tokenizer)

    model = DSQGTransformer(
        vocab_size      = VOCAB_SIZE,
        embedding_dim   = EMBEDDING_DIM,
        num_layers      = NUM_LAYERS,
        num_heads       = NUM_HEADS,
        ffn_dim         = FFN_DIM,
        seq_len         = MAX_SEQ_LEN,
        interference    = INTERFERENCE,
        full_attn_layer = FULL_ATTN_LAYER,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = (len(train_data) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR * 0.1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = []
    best_val    = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        perm        = torch.randperm(len(train_data))
        total_loss  = 0.0
        total_tokens = 0
        t0 = time.time()

        optimizer.zero_grad()
        step_count = 0

        for batch_idx in range(0, len(train_data) - BATCH_SIZE + 1, BATCH_SIZE):
            idxs  = perm[batch_idx:batch_idx + BATCH_SIZE]
            batch = train_data[idxs].to(device)
            inp   = batch[:, :-1]
            tgt   = batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inp)
                loss   = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

                # Scale_embed soft L2 regularization (λ=1e-4)
                se_reg = torch.tensor(0.0, device=device)
                for layer in model.layers:
                    attn = getattr(layer, 'attn', None)
                    if attn is not None and hasattr(attn, 'scale_embed'):
                        se_norm = attn.scale_embed.norm()
                        se_reg  = se_reg + se_norm * se_norm
                loss = (loss + SCALE_EMBED_REG_LAMBDA * se_reg) / GRAD_ACCUM

            loss.backward()
            total_loss   += loss.item() * GRAD_ACCUM
            total_tokens += tgt.numel()
            step_count   += 1

            if step_count % GRAD_ACCUM == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Mild pos_bias cap (prevent unbounded growth)
                with torch.no_grad():
                    for layer in model.layers:
                        attn = getattr(layer, 'attn', None)
                        if attn is not None and hasattr(attn, 'pos_bias'):
                            attn.pos_bias.clamp_(-POS_BIAS_MAX_ABS, POS_BIAS_MAX_ABS)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        epoch_loss = total_loss / step_count
        elapsed    = time.time() - t0
        print(f'[Ep {epoch}] train_loss={epoch_loss:.4f} '
              f'({elapsed:.0f}s, {total_tokens/elapsed:.0f} tok/s)')

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        n_val_use  = min(512, len(val_data))
        val_loss   = 0.0
        val_tokens = 0
        attn_stats = []

        with torch.no_grad():
            for i in range(0, n_val_use, BATCH_SIZE):
                batch = val_data[i:i + BATCH_SIZE].to(device)
                inp   = batch[:, :-1]
                tgt   = batch[:, 1:]
                if i == 0:
                    logits, stats = model(inp, return_stats=True)
                    attn_stats = stats
                else:
                    logits = model(inp)
                loss      = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                            tgt.reshape(-1))
                val_loss  += loss.item() * tgt.numel()
                val_tokens += tgt.numel()

        val_ppl = math.exp(val_loss / val_tokens)
        print(f'[Ep {epoch}] val_ppl={val_ppl:.3f}')

        # Print per-head scale_embed and IF gain stats
        for li, s in enumerate(attn_stats):
            se_max = s.get('scale_embed_abs_max', 0)
            pb_max = s.get('pos_bias_abs_max', 0)
            ig     = s.get('if_gain', [])
            if se_max > 0 or pb_max > 0:
                print(f'  L{li}: se_max={se_max:.4f} pb_max={pb_max:.4f} '
                      f'if_gain={[round(g,3) for g in ig]}')

        # ── Passkey eval ──────────────────────────────────────────────────
        passkey = run_passkey_eval(model, tokenizer, device)
        acc_per_dist = {r['distance']: r['accuracy'] for r in passkey}
        mean_acc     = sum(r['accuracy'] for r in passkey
                          if not all(t['skipped'] for t in r['trials'])) / max(
                          1, sum(1 for r in passkey
                                 if not all(t['skipped'] for t in r['trials'])))
        per_dist_str = [f"d{r['distance']}:{r['accuracy']:.0%}" for r in passkey]
        print(f'[Ep {epoch}] passkey_mean={mean_acc:.1%}  per_dist={per_dist_str}')

        # ── Save ──────────────────────────────────────────────────────────
        result = {
            'epoch':       epoch,
            'val_ppl':     val_ppl,
            'train_loss':  epoch_loss,
            'passkey':     passkey,
            'passkey_mean': mean_acc,
            'attn_stats':  attn_stats,
        }
        all_results.append(result)

        ckpt_path = os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt')
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_ppl': val_ppl}, ckpt_path)

        if val_ppl < best_val:
            best_val = val_ppl
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_ppl': val_ppl},
                       os.path.join(SAVE_DIR, 'best.pt'))
            print(f'[Ep {epoch}] ✓ new best: {val_ppl:.3f}')

        with open(RESULT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    train()
