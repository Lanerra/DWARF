"""
Wavelet-Augmented FFN — Parallel DWT Path + Standard FFN (14M scale, 3-Epoch Probe)

Keeps the full standard FFN and ADDS a wavelet path in parallel, mixed with learned α:
  out = (1 - α) * FFN(x) + α * IDWT(gated_GELU(DWT(x)))

Key properties:
  - Optimized 4-tap wavelet (lattice θ=1.1377, from math autoresearch)
  - Level selection: ALTERNATING — GELU at levels [0, 2, 4, 6], linear at [1, 3, 5, 7]
  - Full D=256 transform (cross-head mixing)
  - Approx coefficients: PASS (no nonlinearity on DC component)
  - Wavelet path: ~513 learned params per block (scale + bias (zero-init ReZero))
  - Full FFN retained: 525,568 params per block (unchanged)
  - Mixing: learned sigmoid(α), init at -2.0 → ~12% wavelet contribution

Architecture: d48 offsets (dense=48, sparse=[96,128,384], J=52)
  FULL_ATTN_LAYER = -1 (pure: all 6 layers are DSQG blocks)
  INTERFERENCE = 3 (receiver-chain every 3rd block)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_14m_d48_wavelet_aug_ffn.py \
    2>&1 | tee benchmarks/logs/d48_wavelet_aug_14m.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# -- Hyperparameters -----------------------------------------------------------

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 3
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
FULL_ATTN_LAYER = -1

EXPERIMENT_NAME = 'd48_wavelet_aug_14m'

# -- FineWeb-Edu dataset config ------------------------------------------------

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
MAX_TRAIN_SEQS  = 52_716       # iso-compute cap — matches all 14M autoresearch probes

# -- Passkey eval config -------------------------------------------------------

PASSKEY_DISTANCES = [64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# -- Save paths ----------------------------------------------------------------

SAVE_DIR    = 'checkpoints/d48_wavelet_aug_14m'
RESULT_FILE = 'benchmarks/logs/d48_wavelet_aug_14m_results.json'

# -- d48 offset set ------------------------------------------------------------

_DENSE_LOCAL_W     = 48
_DYADIC_LONG_RANGE = [96, 128, 384]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 52

# -- Triton kernel (V3 -- Q-weighted scale gains) -----------------------------

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_probe_d48_s96_128_384 import dsqg_attention_v3


# -- Optimized wavelet filter (lattice parameterization) ----------------------

OPTIMAL_THETA = 1.1377


def lattice_4tap(theta: float) -> list:
    """
    4-tap orthogonal wavelet filter via lattice factorization.
    Single rotation angle θ parameterizes the filter.
    D4 Daubechies corresponds to θ = π/3.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    denominator = 2.0 * math.sqrt(2.0)
    return [
        (1.0 - c + s) / denominator,
        (1.0 + c + s) / denominator,
        (1.0 + c - s) / denominator,
        (1.0 - c - s) / denominator,
    ]


def qmf_highpass(lowpass: list) -> list:
    """QMF conjugate mirror filter: highpass[k] = (-1)^(k+1) * lowpass[N-1-k]."""
    length = len(lowpass)
    return [
        ((-1) ** (k + 1)) * lowpass[length - 1 - k]
        for k in range(length)
    ]


def build_dwt_matrix(dimension: int, number_of_levels: int,
                     lowpass_filter: list) -> torch.Tensor:
    """
    Build the full dimension×dimension orthogonal DWT matrix.
    DWT(x) = x @ W.T, IDWT(y) = y @ W  (for batch/row-vector form).
    """
    lowpass = torch.tensor(lowpass_filter, dtype=torch.float32)
    highpass = torch.tensor(qmf_highpass(lowpass_filter), dtype=torch.float32)

    W = torch.eye(dimension)
    current_length = dimension

    for _level in range(number_of_levels):
        half = current_length // 2
        level_matrix = torch.zeros(current_length, current_length)

        for i in range(half):
            for k, coefficient in enumerate(lowpass):
                column = (2 * i + k) % current_length
                level_matrix[i, column] += coefficient
            for k, coefficient in enumerate(highpass):
                column = (2 * i + k) % current_length
                level_matrix[half + i, column] += coefficient

        if _level == 0:
            W = level_matrix
        else:
            top = W[:current_length, :]
            W[:current_length, :] = level_matrix @ top

        current_length = half

    return W


# -- WaveletAugmentedFFN ------------------------------------------------------

class WaveletAugmentedFFN(nn.Module):
    """
    Parallel wavelet path augmenting a standard FFN.

    Standard FFN runs at full capacity (D → 4D → D).
    Wavelet path: DWT → per-bin learned affine (scale+bias) → IDWT.  Purely linear
    — no GELU needed here; the FFN path supplies all nonlinearity.  Math autoresearch
    (59 iterations) confirmed GELU on wavelet levels is redundant (score identical).
    Output: FFN(x) + wavelet(x).  wavelet path is zero-init (ReZero-style) — starts
    at 0, grows as wavelet_scale learns nonzero values wherever frequency routing helps.
    """
    def __init__(self, embedding_dimension: int, ffn_multiplier: int = 4,
                 number_of_levels: int = 7, theta: float = OPTIMAL_THETA):
        super().__init__()
        # ── Standard FFN (full capacity, unchanged) ──
        self.w1 = nn.Linear(embedding_dimension, embedding_dimension * ffn_multiplier)
        self.w2 = nn.Linear(embedding_dimension * ffn_multiplier, embedding_dimension)
        self.activation = nn.GELU()

        # ── Wavelet path (~2*D params, ReZero-style zero init) ──
        # scale=zeros → wavelet path starts at 0, model grows it from there.
        # No identity leak at init; trains as pure FFN until wavelet routing helps.
        self.wavelet_scale = nn.Parameter(torch.zeros(embedding_dimension))
        self.wavelet_bias  = nn.Parameter(torch.zeros(embedding_dimension))

        # ── DWT matrix (fixed orthogonal basis, not learned) ──
        lowpass_filter = lattice_4tap(theta)
        W = build_dwt_matrix(embedding_dimension, number_of_levels, lowpass_filter)
        self.register_buffer('W_dwt', W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Standard FFN path ──
        ffn_out = self.w2(self.activation(self.w1(x)))

        # ── Wavelet path: DWT → learned affine → IDWT (starts at 0) ──
        x_freq = x @ self.W_dwt.T                                 # forward DWT
        x_freq = x_freq * self.wavelet_scale + self.wavelet_bias  # per-bin gain (0 at init)
        x_wave = x_freq @ self.W_dwt                              # inverse DWT

        # Purely additive — wavelet path starts at 0, grows as scale learns
        return ffn_out + x_wave


# -- DSQGAttentionQW -----------------------------------------------------------

class DSQGAttentionQW(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        head_dimension = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in _COND_N_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(52, head_dimension))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))
        self.dropout = nn.Dropout(dropout)

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


# -- FullCausalAttention -------------------------------------------------------

class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return F.dropout(self.out_proj(out_flat * gate),
                         p=self.dropout_p, training=self.training)

    def attn_summary(self):
        return {'type': 'full_causal',
                'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0] * NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0] * NUM_HEADS}


# -- Receiver-chain helpers ----------------------------------------------------

_EMA_KERNEL_LEN = 256

def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor) -> torch.Tensor:
    B, N, D = xi.shape
    alpha   = ema_factor.clamp(0.005, 0.5)
    k_len   = min(_EMA_KERNEL_LEN, N)
    t      = torch.arange(k_len, device=xi.device, dtype=torch.float32)
    kernel = alpha.float() * (1.0 - alpha.float()).pow(t)
    kernel = kernel / kernel.sum()
    kernel = kernel.flip(0)
    xi_f  = xi.float()
    xi_bd = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p  = F.pad(xi_bd, (k_len - 1, 0))
    pool  = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(xi.dtype)


def _kdv_correction(pool: torch.Tensor, kdv_alpha: torch.Tensor) -> torch.Tensor:
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    delta     = pool - pool_prev
    return pool + alpha * pool * delta


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# -- DSQGBlock with WaveletAugmentedFFN ----------------------------------------

class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, use_checkpoint=False, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionQW(
            embedding_dim, num_heads, seq_len=seq_len, dropout=dropout)
        self.wavelet_augmented_ffn = WaveletAugmentedFFN(
            embedding_dim, ffn_multiplier=ffn_dim // embedding_dim)

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
            pool = _causal_ema(xi, self.ema_factor)
            pool = _kdv_correction(pool, self.kdv_alpha)
            pool = _agc_normalize(pool)
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = self.inter_k_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            v_delta = self.inter_v_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            kv_inject = (k_delta, v_delta)

        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.wavelet_augmented_ffn(self.norm2(x))
        return x

    def block_summary(self):
        if not self.interference:
            return {}
        return {
            'ema_factor': self.ema_factor.item(),
            'kdv_alpha':  self.kdv_alpha.item(),
        }


# -- FullAttentionBlock --------------------------------------------------------

class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.fc1   = nn.Linear(embedding_dim, ffn_dim)
        self.fc2   = nn.Linear(ffn_dim, embedding_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.drop(F.gelu(self.fc1(self.norm2(x)))))
        return x


# -- CondMTransformer ----------------------------------------------------------

class CondMTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer=FULL_ATTN_LAYER,
                 interference_interval=INTERFERENCE, dropout=0.1):
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
                blocks.append(DSQGBlock(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, use_checkpoint=False,
                    interference=(i % interference_interval == interference_interval - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
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
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg_blocks = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        if not dsqg_blocks:
            return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                    'pos_bias_mean_per_head': [0.0] * NUM_HEADS,
                    'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                    'if_gain': [1.0] * NUM_HEADS,
                    'ema_factors': [], 'kdv_alphas': [],
                    'wavelet_mix_alpha': 0.0, 'wavelet_scale_abs_mean': 0.0,
                    'wavelet_scale_abs_max': 0.0}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        inter_blocks = [b for b in dsqg_blocks if b.interference]

        wavelet_scales = [
            b.wavelet_augmented_ffn.wavelet_scale.detach()
            for b in dsqg_blocks
        ]

        return {
            'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean']    for s in summaries) / n,
            'pos_bias_abs_max':       max(s['pos_bias_abs_max']     for s in summaries),
            'pos_bias_mean_per_head': [
                sum(s['pos_bias_mean_per_head'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
            'scale_embed_abs_mean':   sum(s['scale_embed_abs_mean'] for s in summaries) / n,
            'scale_embed_abs_max':    max(s['scale_embed_abs_max']  for s in summaries),
            'if_gain': [
                sum(s['if_gain'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
            'ema_factors': [b.ema_factor.item() for b in inter_blocks],
            'kdv_alphas':  [b.kdv_alpha.item()  for b in inter_blocks],
            'wavelet_scale_abs_mean': sum(
                s.abs().mean().item() for s in wavelet_scales) / n,
            'wavelet_scale_abs_max': max(
                s.abs().max().item() for s in wavelet_scales),
            'wavelet_mix_alpha': 0.0,  # not used in this architecture; placeholder for results schema
        }


# -- Data utilities ------------------------------------------------------------

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_data(num_docs=NUM_DOCS):
    import json as _json
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = _json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
    else:
        from datasets import load_dataset
        print(f'Loading FineWeb-Edu ({FW_SUBSET}) -- seeking {num_docs:,} docs...')
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
            _json.dump(texts, fp)
        print(f'  Cached {len(texts):,} docs')
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


def dwt_invertibility_check(device):
    """Verify that the DWT matrix roundtrips correctly (no nonlinearity)."""
    print('Running DWT invertibility check...')
    lowpass_filter = lattice_4tap(OPTIMAL_THETA)
    number_of_levels = int(math.log2(EMBEDDING_DIM)) - 1
    W = build_dwt_matrix(EMBEDDING_DIM, number_of_levels, lowpass_filter).to(device)
    x_test = torch.randn(4, 16, EMBEDDING_DIM, device=device)
    with torch.no_grad():
        x_freq = x_test @ W.T
        x_rec = x_freq @ W
    max_error = (x_rec - x_test).abs().max().item()
    print(f'  ||IDWT(DWT(x)) - x||_inf = {max_error:.2e}')
    ok = max_error < 1e-4
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


def build_model(device='cuda'):
    from tokenizers import Tokenizer
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _tok_candidates = [
        os.path.join(_script_dir, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    assert tok_path, 'condI tokenizer not found'
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    model = CondMTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
    ).to(device)
    return model


# -- Training loop -------------------------------------------------------------

def train(model, train_data, val_data, test_data, tokenizer, device='cuda'):
    os.makedirs(SAVE_DIR, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda')

    GEN_PROMPTS = [
        'It was a dark and stormy',
        'The length of the hypotenuse',
        'The President of the United',
        'Once upon a time there was',
        'The results indicate that',
    ]

    best_val_loss     = float('inf')
    best_val_ppl      = float('inf')
    best_epoch        = 0
    t0                = time.time()
    per_epoch_results = []

    tokens_per_epoch = len(train_data) * (MAX_SEQ_LEN - 1)
    chin_tokens      = 20 * model.param_count()
    chin_epoch       = chin_tokens / tokens_per_epoch
    print(f'\n  Tokens/epoch: {tokens_per_epoch:,}')
    print(f'  Chinchilla:   {chin_tokens:,} tokens (epoch ~{chin_epoch:.2f})\n')

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
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
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
        print(f'  Wavelet scale |mean|: {ss["wavelet_scale_abs_mean"]:.4f}  |max|: {ss["wavelet_scale_abs_max"]:.4f}')
        print(f'  Wavelet scale |mean|: {ss["wavelet_scale_abs_mean"]:.4f} '
              f'|max|: {ss["wavelet_scale_abs_max"]:.4f}')

        print('  -- Generation samples (greedy) --')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
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
            'if_gain': ss['if_gain'],
            'ema_factors': ss['ema_factors'],
            'kdv_alphas':  ss['kdv_alphas'],
            'wavelet_mix_alpha': ss['wavelet_mix_alpha'],
            'wavelet_scale_abs_mean': ss['wavelet_scale_abs_mean'],
            'wavelet_scale_abs_max': ss['wavelet_scale_abs_max'],
        })
        sys.stdout.flush()

    # -- Final evaluation ------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  {EXPERIMENT_NAME} TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')

    print('\n  -- Temperature sweep (best checkpoint) --')
    sweep_results = {}
    for temp in [0.0, 0.5, 0.7, 1.0]:
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

    ss = model.attn_summary()
    gains = ss['if_gain']
    gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
    print(f'\n  Final IF gains: {gain_str}')
    print(f'  Final scale_embed: |mean|={ss["scale_embed_abs_mean"]:.4f} '
          f'|max|={ss["scale_embed_abs_max"]:.4f}')
    print(f'  Final wavelet scale |mean|: {ss["wavelet_scale_abs_mean"]:.4f}  |max|: {ss["wavelet_scale_abs_max"]:.4f}')
    print(f'  Final wavelet scale: |mean|={ss["wavelet_scale_abs_mean"]:.4f} '
          f'|max|={ss["wavelet_scale_abs_max"]:.4f}')
    if ss['ema_factors']:
        ema_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['ema_factors']))
        kdv_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['kdv_alphas']))
        print(f'  Final EMA factors:  {ema_str}   (window~{1/ss["ema_factors"][0]:.0f}t)')
        print(f'  Final KdV alphas:   {kdv_str}')

    results = {
        'experiment':              EXPERIMENT_NAME,
        'kernel':                  'dsqg_attention_v3',
        'architecture':            'wavelet_augmented_ffn_d48',
        'full_attn_layer':         FULL_ATTN_LAYER,
        'offsets':                 {'dense': _DENSE_LOCAL_W, 'sparse': _DYADIC_LONG_RANGE, 'total': 52},
        'final_test_ppl':          test_ppl,
        'final_passkey_mean':      pk_final_mean,
        'final_passkey_by_d':      {str(d): v for d, v in pk_final.items()},
        'per_epoch':               per_epoch_results,
        'peak_vram_gb':            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        'temperature_sweep':       sweep_results,
        'attn_summary':            ss,
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')
    return results


# -- Main ----------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print(f'  {EXPERIMENT_NAME} -- Wavelet-Augmented FFN + Pure DSQG (14M scale, FineWeb-Edu)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Kernel: dsqg_attention_v3 (d48 offsets)')
    print(f'  Wavelet-Augmented FFN: lattice \u03b8={OPTIMAL_THETA:.4f}, '
          f'alternating levels [0,2,4,6], full D=256 transform')
    lowpass_filter = lattice_4tap(OPTIMAL_THETA)
    print(f'  Filter coefficients: [{", ".join(f"{c:.4f}" for c in lowpass_filter)}]')
    print(f'  Wavelet path: zero-init (ReZero-style) — starts at 0, grows as scale learns')
    print(f'  Architecture: INTERFERENCE=3, gate=0, ALL DSQG (no full attention)')
    print(f'  Offsets: dense={_DENSE_LOCAL_W}, sparse={_DYADIC_LONG_RANGE}, J=52')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits = load_data(NUM_DOCS)
    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    _tok_candidates = [
        os.path.join(_script_dir, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    if tok_path:
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'\nLoaded condI BPE tokenizer from {tok_path}')
    else:
        raise FileNotFoundError('condI tokenizer not found')

    _encoded_cache = 'benchmarks/logs/fineweb_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache} ...')
        _cache = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
        test_data  = _cache['test']
        if MAX_TRAIN_SEQS is not None and len(train_data) > MAX_TRAIN_SEQS:
            idx = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
            train_data = train_data[idx]
        print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
              f'test: {len(test_data):,} seqs (from cache)')
    else:
        print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
        train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
        if MAX_TRAIN_SEQS is not None and len(train_data) > MAX_TRAIN_SEQS:
            idx = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
            train_data = train_data[idx]
            print(f'  Capped to {MAX_TRAIN_SEQS:,} train sequences (iso-compute)')
        val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
        test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = CondMTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    layer_types = ['FULL' if i == FULL_ATTN_LAYER else 'DSQG'
                   for i in range(NUM_LAYERS)]

    ffn_params_per_block = EMBEDDING_DIM * FFN_DIM * 2 + FFN_DIM + EMBEDDING_DIM
    wavelet_params_per_block = 2 * EMBEDDING_DIM + 1
    dsqg_block_count = sum(1 for i in range(NUM_LAYERS) if i != FULL_ATTN_LAYER)
    total_ffn_params = ffn_params_per_block * dsqg_block_count
    total_wavelet_params = wavelet_params_per_block * dsqg_block_count

    print(f'\n{EXPERIMENT_NAME}: {n_params:,} parameters')
    print(f'  Standard FFN params: {total_ffn_params:,} ({ffn_params_per_block:,}/block)')
    print(f'  Wavelet path params: {total_wavelet_params:,} '
          f'({wavelet_params_per_block:,}/block = scale + bias (zero-init ReZero))')
    print(f'  Layer types: {layer_types}')

    if not dwt_invertibility_check(device):
        print('DWT invertibility check FAILED \u2014 aborting.')
        return

    if not causality_check(model, device):
        return

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
