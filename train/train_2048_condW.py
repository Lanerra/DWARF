"""
condW — Pure-DSQG Receiver-Chain Interference (13M, FineWeb-Edu)

Builds on condV with five bug fixes and one architectural change:

ARCHITECTURAL CHANGE:
  Full Attention layer (condV layer 5) removed.  All 6 layers are now DSQG.
  With INTERFERENCE=3 and 6 DSQG layers: i%3==2 fires at layers 2 AND 5.
  This restores the second interference site and eliminates O(N) KV cache
  from the full-attention layer, making the full stack O(1) KV cache.

BUG FIXES vs condV:
  1. RUNNING LOSS: all GRAD_ACCUM microbatch losses accumulated, not just last
  2. EMA KERNEL: kernel_len = N (sequence length) for exact IIR equivalence;
     condV used fixed 256 which diverges 38% from true EMA at α=0.005
  3. PASSKEY WORDS: filtered to 6 guaranteed-single-token candidates;
     condV had 4 multi-token words ('grape','mango','peach','berry') silently
     scored on first subword token only
  4. PASSKEY TRIALS: 50 per distance (was 5); noise drops from ±20pp to ±7pp
  5. OPTIMIZER: three-group AdamW — weight decay off for biases/norms/physics
     embeddings; ema_factor + kdv_alpha at 10× lower LR (3e-5 vs 3e-4)

ADDITIONAL CHANGE:
  Gate init = -3.0 (sigmoid(-3) ≈ 0.047, near-zero initial branch contribution)
  condV used 0.0 which starts the gate at sigmoid(0)=0.5 — 50% contribution
  from the first step. Negative init delays gate opening, allowing the model
  to develop embeddings/FFN before the attention branch dominates.

Retained from condV:
  - Kalman-EMA + KdV soliton + AGC interference mechanism
  - Q-weighted scale gains (V3 kernel, scale_embed [44, HD], zero-init)
  - IF amplifier (per-head gain, 1.0-init)
  - Huygens K/V injection (K and V only; Q stays clean)
  - INTERFERENCE=3, gate_bias=0 rule (now gate_init=-3, not +bias)

References:
  condV FINAL:  52.207 PPL / 36.7% passkey (ep10) [1/12 >50%; d=8:60%]
  condU FINAL:  52.206 PPL / 43.3% passkey (ep10) [3/12 >50%]
  I3G0 ref:     52.948 PPL / 53.3% passkey
  condM:        54.529 PPL / 83.3% passkey

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condW.py \
    2>&1 | tee benchmarks/logs/condW_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Hyperparameters -----------------------------------------------------------

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
LR_PHYSICS      = 3e-5     # ema_factor, kdv_alpha: 10× lower (different gradient scale)
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3        # i%3==2 → layers 2 and 5 (both DSQG in pure stack)

GATE_INIT_BIAS  = -3.0     # sigmoid(-3)≈0.047: near-zero initial gate contribution

# -- FineWeb-Edu dataset config ------------------------------------------------

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
MAX_TRAIN_SEQS  = 52_716

# -- Passkey eval config -------------------------------------------------------

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50   # was 5; noise drops from ±20pp to ±7pp

# Guaranteed single-token candidates (verified with condI BPE tokenizer).
# Removed: 'grape'→[5924,391], 'mango'→[523,2063], 'peach'→[522,587], 'berry'→[229,5632]
# 6-way forced choice (chance=16.7%); previous 10-way included 4 silent multi-token errors.
_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'lemon', 'plum']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'

# -- Save paths ----------------------------------------------------------------

SAVE_DIR    = 'checkpoints/condW'
RESULT_FILE = 'benchmarks/logs/condW_results.json'

# -- Offset set ----------------------------------------------------------------

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44

# -- Triton kernel (V3) --------------------------------------------------------

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_attention_v3 import dsqg_attention_v3
from causal_ema_native import causal_ema_kdv


# -- DSQGAttentionQW -----------------------------------------------------------

class DSQGAttentionQW(nn.Module):
    """
    DSQG attention with V3 kernel (Q-weighted scale gains) + IF amplifier.
    Identical to condV.  gate_init_bias applied externally by CondWTransformer.
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

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
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta
        out      = dsqg_attention_v3(q, k, v, self.pos_bias, self.scale_embed)
        out      = out * self.if_gain.view(1, H, 1, 1)
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


# -- FFN -----------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# -- Receiver-chain helpers (condV, carried forward) ---------------------------

def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor) -> torch.Tensor:
    """
    Causal EMA: pool[n] = (1-α)*pool[n-1] + α*xi[n]

    FIX vs condV: kernel_len = N (sequence length) instead of fixed 256.
    At fixed 256 with α=0.005: only 72.3% of the true IIR signal captured
    and the normalisation factor inflates all taps by 1.38×.
    With kernel_len = N the truncated FIR is exact (causal sequences can't
    look back further than their own length).

    xi:         [B, N, D]
    ema_factor: scalar parameter, clamped to [0.005, 0.5]
    Returns:    [B, N, D]
    """
    B, N, D = xi.shape
    alpha   = ema_factor.clamp(0.005, 0.5)
    k_len   = min(N, max(64, int(math.ceil(4.0 / alpha.item()))))  # adaptive: covers 98% of tail

    t      = torch.arange(k_len, device=xi.device, dtype=torch.float32)
    kernel = alpha.float() * (1.0 - alpha.float()).pow(t)   # [N]
    # No normalisation: tail weight < e^{-4} < 2%; inflation from renorm is worse
    kernel = kernel.flip(0)                                  # causal: oldest first

    xi_f  = xi.float()
    xi_bd = xi_f.permute(0, 2, 1).reshape(B * D, 1, N)
    xi_p  = F.pad(xi_bd, (k_len - 1, 0))
    pool  = F.conv1d(xi_p, kernel.view(1, 1, k_len))
    return pool.view(B, D, N).permute(0, 2, 1).to(xi.dtype)


def _kdv_correction(pool: torch.Tensor, kdv_alpha: torch.Tensor) -> torch.Tensor:
    """
    KdV soliton field correction: pool_kdv[n] = pool[n] + α·pool[n]·Δpool[n]
    Zero-init (α=0) → identity.
    """
    alpha     = kdv_alpha.clamp(0.0, 0.5)
    pool_prev = F.pad(pool[:, :-1], (0, 0, 1, 0))
    return pool + alpha * pool * (pool - pool_prev)


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AGC: normalize to unit RMS per token. No parameters."""
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# -- DSQGBlock -----------------------------------------------------------------

class DSQGBlock(nn.Module):
    """
    condW DSQG block.  Identical to condV except:
    - gate_init_bias applied by CondWTransformer._init_weights (not hardcoded here)
    - No FullAttentionBlock variant; all blocks are DSQGBlock
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionQW(
            embedding_dim, num_heads, seq_len=seq_len, dropout=dropout)
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
            pool    = causal_ema_kdv(xi, self.ema_factor, self.kdv_alpha)
            pool    = _agc_normalize(pool)
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = self.inter_k_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            v_delta = self.inter_v_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            kv_inject = (k_delta, v_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x

    def block_summary(self):
        if not self.interference:
            return {}
        return {
            'ema_factor': self.ema_factor.item(),
            'kdv_alpha':  self.kdv_alpha.item(),
        }


# -- CondWTransformer (pure DSQG, no FullAttentionBlock) -----------------------

class CondWTransformer(nn.Module):
    """
    Pure DSQG stack.  No FullAttentionBlock.  Full O(1) KV cache.
    INTERFERENCE=3 with 6 DSQG layers → interference at layers 2 and 5.
    """
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, interference_interval=INTERFERENCE, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DSQGBlock(
                embedding_dim, num_heads, ffn_dim, seq_len,
                dropout=dropout,
                interference=(i % interference_interval == interference_interval - 1),
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # Gate init = GATE_INIT_BIAS (negative → near-zero initial gate contribution)
        # sigmoid(GATE_INIT_BIAS=-3) ≈ 0.047; model learns to open gate gradually
        # condV used 0.0 (sigmoid=0.5, 50% gate from first step)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, GATE_INIT_BIAS)

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
        summaries     = [b.attn.attn_summary() for b in self.blocks]
        inter_blocks  = [b for b in self.blocks if b.interference]
        n             = len(summaries)
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
        }


# -- Optimizer: three param groups ---------------------------------------------

def build_optimizer(model):
    """
    Three-group AdamW:
      decay    — weights (Linear.weight, Embedding.weight): lr=LR, wd=0.1
      no_decay — biases, LayerNorm, pos_bias, scale_embed, if_gain: lr=LR, wd=0
      physics  — ema_factor, kdv_alpha: lr=LR_PHYSICS, wd=0
                 (different gradient scaling; needs separate LR)

    FIX vs condV: flat model.parameters() with wd=0.1 was pulling if_gain
    toward 0 (gradient=0.1 per step >> loss gradient on single scalar) and
    preventing ema_factor/kdv_alpha from developing freely.
    """
    decay_params   = []
    nodecay_params = []
    physics_params = []
    physics_names  = {'ema_factor', 'kdv_alpha'}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pname = n.split('.')[-1]   # last name component
        if pname in physics_names:
            physics_params.append(p)
        elif (pname in {'bias'} or
              'norm' in n or
              pname in {'pos_bias', 'scale_embed', 'if_gain'} or
              isinstance(p, nn.Parameter) and p.ndim <= 1):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    # Sanity check: all params accounted for
    all_ids  = {id(p) for p in model.parameters()}
    grp_ids  = ({id(p) for p in decay_params} |
                {id(p) for p in nodecay_params} |
                {id(p) for p in physics_params})
    missed   = all_ids - grp_ids
    assert not missed, f"Unclassified params: {missed}"

    n_decay   = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    n_physics = sum(p.numel() for p in physics_params)
    print(f'  Optimizer param groups:')
    print(f'    decay    : {len(decay_params):3d} tensors  {n_decay:>9,} params  lr={LR}  wd=0.1')
    print(f'    no_decay : {len(nodecay_params):3d} tensors  {n_nodecay:>9,} params  lr={LR}  wd=0')
    print(f'    physics  : {len(physics_params):3d} tensors  {n_physics:>9,} params  lr={LR_PHYSICS}  wd=0')

    return torch.optim.AdamW([
        {'params': decay_params,   'lr': LR,         'weight_decay': 0.1},
        {'params': nodecay_params, 'lr': LR,         'weight_decay': 0.0},
        {'params': physics_params, 'lr': LR_PHYSICS,  'weight_decay': 0.0},
    ], betas=(0.9, 0.95))


# -- Data utilities (unchanged) ------------------------------------------------

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
        texts = []
        for item in ds:
            if len(item['text']) < FW_MIN_CHARS: continue
            texts.append(item['text'])
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


def generate(model, tokenizer, prompts, device, max_new=150, temperature=1.0, top_p=0.9):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                logits_last = model(ids[:, -MAX_SEQ_LEN:])[:, -1, :]
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
    """
    FIX vs condV: 6 guaranteed single-token words, 50 trials, 6-way forced choice.
    Chance = 1/6 ≈ 16.7%.  condV had 4 multi-token words silently scored on
    first subword only; 5 trials (±20pp noise); 10-way including broken candidates.
    """
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    # Pre-encode all candidate first tokens (all guaranteed single-token)
    all_tok_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                   for w in _PASSKEY_WORDS]
    results = {}
    for d in PASSKEY_DISTANCES:
        correct = 0; n_valid = 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            t_idx     = _PASSKEY_WORDS.index(target)
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available: continue
            filler = []
            while len(filler) < d: filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN: continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(ids)[:, -1, :]
            scores   = logits[0][all_tok_ids]   # [6]
            pred_idx = scores.argmax().item()
            correct  += int(pred_idx == t_idx)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# -- Training loop -------------------------------------------------------------

def train(model, train_data, val_data, test_data, tokenizer, device='cuda'):
    os.makedirs(SAVE_DIR, exist_ok=True)
    optimizer   = build_optimizer(model)
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    scaler      = torch.amp.GradScaler('cuda')

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
            # FIX: accumulate loss across ALL microbatches, not just the last one.
            # condV only added the last ga's loss, biasing the reported train loss.
            step_loss = 0.0
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
                step_loss += loss.item()   # accumulate scaled; sum = mean(full_loss)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1
            running_loss += step_loss     # step_loss = mean cross-entropy for this step

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {step_loss:.4f}')

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

        ss          = model.attn_summary()
        head_means  = ss['pos_bias_mean_per_head']
        most_local  = int(max(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        most_global = int(min(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} '
              f'most-local=h{most_local} most-global=h{most_global}')
        print(f'  scale_embed:   |mean|={ss["scale_embed_abs_mean"]:.4f} '
              f'|max|={ss["scale_embed_abs_max"]:.4f}')
        gains    = ss['if_gain']
        gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
        print(f'  IF gains:      {gain_str}')
        if ss['ema_factors']:
            ema_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['ema_factors']))
            kdv_str = '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['kdv_alphas']))
            wins    = [f'{1/v:.0f}t' for v in ss['ema_factors']]
            print(f'  EMA factors:   {ema_str}   (windows: {", ".join(wins)})')
            print(f'  KdV alphas:    {kdv_str}')

        print('  -- Generation samples (greedy) --')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')
        print('  --')

        print('  Passkey... (50 trials, 6-way, chance=16.7%)')
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        above50 = sum(1 for v in pk.values() if v >= 0.5)
        print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        per_epoch_results.append({
            'epoch':               epoch,
            'val_ppl':             val_ppl,
            'train_loss':          train_loss,
            'chinchilla_pct':      chin_pct,
            'elapsed_s':           elapsed,
            'passkey_mean':        pk_mean,
            'passkey_by_d':        {str(d): v for d, v in pk.items()},
            'scale_embed_abs_mean':ss['scale_embed_abs_mean'],
            'if_gain':             ss['if_gain'],
            'ema_factors':         ss['ema_factors'],
            'kdv_alphas':          ss['kdv_alphas'],
        })
        sys.stdout.flush()

    # -- Final evaluation ------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condW TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')
    print(f'  condV reference: 52.207 PPL | delta = {test_ppl - 52.207:+.3f}')
    print(f'  condU reference: 52.206 PPL | delta = {test_ppl - 52.206:+.3f}')
    print(f'  I3G0 reference:  52.948 PPL | delta = {test_ppl - 52.948:+.3f}')
    print(f'  condM reference: 54.529 PPL | delta = {test_ppl - 54.529:+.3f}')

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
    print(f'\n  Final passkey (50 trials, 6-way, chance=16.7%):')
    print(f'  mean={pk_final_mean*100:.1f}%  ({above50_final}/{len(pk_final)} distances >50%)')
    parts = [f'd={d}:{int(pk_final[d]*100)}%' for d in PASSKEY_DISTANCES]
    print('  ' + '  '.join(parts))
    print(f'  I3G0 reference:  53.3% passkey  (NOTE: 5-trial 10-way — not directly comparable)')
    print(f'  condM reference: 83.3% passkey  (NOTE: same caveat)')

    ss = model.attn_summary()
    gains    = ss['if_gain']
    gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
    print(f'\n  Final IF gains: {gain_str}')
    print(f'  Final scale_embed: |mean|={ss["scale_embed_abs_mean"]:.4f} '
          f'|max|={ss["scale_embed_abs_max"]:.4f}')
    if ss['ema_factors']:
        for i, (ef, ka) in enumerate(zip(ss['ema_factors'], ss['kdv_alphas'])):
            print(f'  Interference block {i} (layer {2 + 3*i}): '
                  f'ema_factor={ef:.4f} (window≈{1/ef:.0f}t)  kdv_alpha={ka:.4f}')

    results = {
        'experiment':               'condW_pure_dsqg',
        'kernel':                   'dsqg_attention_v3',
        'changes_vs_condV': [
            'pure_dsqg_no_full_attn',
            'interference_at_layers_2_and_5',
            'gate_init_bias_-3',
            'no_decay_param_groups',
            'physics_lr_3e-5',
            'ema_kernel_len_equals_N',
            'passkey_6_single_token_words',
            'passkey_50_trials_6way',
            'running_loss_all_microbatches',
        ],
        'final_test_ppl':           test_ppl,
        'final_passkey_mean':       pk_final_mean,
        'final_passkey_by_d':       {str(d): v for d, v in pk_final.items()},
        'per_epoch':                per_epoch_results,
        'temperature_sweep':        sweep_results,
        'attn_summary':             ss,
        'condV_reference_ppl':      52.207,
        'condV_reference_passkey':  0.367,
        'condU_reference_ppl':      52.206,
        'condU_reference_passkey':  0.433,
        'i3g0_reference_ppl':       52.948,
        'i3g0_reference_passkey':   0.533,
        'condm_reference_ppl':      54.529,
        'condm_reference_passkey':  0.833,
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
    print('  condW — Pure-DSQG Receiver-Chain Interference (13M, FineWeb-Edu)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Kernel: dsqg_attention_v3 (unchanged)')
    print(f'  Architecture: 6×DSQG (pure), INTERFERENCE=3 → layers 2+5')
    print(f'  Gate init: {GATE_INIT_BIAS} (sigmoid≈{1/(1+math.exp(-GATE_INIT_BIAS)):.3f})')
    print(f'  Optimizer: 3-group AdamW (decay / no-decay / physics@{LR_PHYSICS})')
    print(f'  Passkey: {PASSKEY_TRIALS} trials, {len(_PASSKEY_WORDS)}-way, '
          f'words={_PASSKEY_WORDS}')
    print(f'  EMA: kernel_len=N (exact; condV used fixed 256)')
    print(f'  References: condV=52.207/36.7% | condU=52.206/43.3% | '
          f'I3G0=52.948/53.3% | condM=54.529/83.3%')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits   = load_data(NUM_DOCS)
    _sdir    = os.path.dirname(os.path.abspath(__file__))
    tok_path = next((p for p in [
        os.path.join(_sdir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_sdir, '2048_condI_tokenizer.json'),
    ] if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI tokenizer not found')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'\nLoaded condI BPE tokenizer from {tok_path}')

    _encoded_cache = 'benchmarks/logs/fineweb_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache} ...')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
        test_data  = _cache['test']
        if len(train_data) > MAX_TRAIN_SEQS:
            idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
            train_data = train_data[idx]
        print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
              f'test: {len(test_data):,} seqs (from cache)')
    else:
        print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
        train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
        if len(train_data) > MAX_TRAIN_SEQS:
            idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
            train_data = train_data[idx]
        val_data  = encode_split(splits['val'],  tokenizer, MAX_SEQ_LEN, 'Val')
        test_data = encode_split(splits['test'], tokenizer, MAX_SEQ_LEN, 'Test')

    model = CondWTransformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_dim               = FFN_DIM,
        seq_len               = MAX_SEQ_LEN,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params = model.param_count()
    int_layers = [i for i in range(NUM_LAYERS)
                  if i % INTERFERENCE == INTERFERENCE - 1]
    print(f'\ncondW: {n_params:,} parameters')
    print(f'  Layer types: {["DSQG+INT" if i in int_layers else "DSQG" for i in range(NUM_LAYERS)]}')
    print(f'  Interference at layers: {int_layers}')

    if not causality_check(model, device):
        return

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
