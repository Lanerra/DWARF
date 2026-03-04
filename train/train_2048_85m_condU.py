"""
condU 85M — Q-Weighted Scale Gains, Scaling Series (RunPod H200 SXM)

Scales condU to 85M (D=768, H=12) — the headline result of the condU
scaling series (13M → 35M → 85M). Same physics stack and hybrid structure
as condU 35M, scaled up:

  - Q-weighted scale gains (V3 kernel, scale_embed [44, HD], zero-init)
  - IF amplifier (per-head learnable gain, init 1.0)
  - Huygens K/V injection (interference → K,V only; Q unchanged)
  - gate=0 (correct design from gate-bias ablation)
  - INTERFERENCE=3, FULL_ATTN_LAYER=7 (last layer in L=8 stack)

PURPOSE
-------
  1. Headline condU result: does the condU physics stack beat condM 85M
     (36.042 PPL, 83.3% passkey) at matched scale?
  2. Complete the 13M → 35M → 85M power-law scaling curve.
  3. Cross-domain physics → ML translation at 85M param publication scale.

ARCHITECTURE
  D=768, H=12, d_head=64, L=8, FFN=3072
  HD=64 — power-of-2, Triton V3 kernel fully native (zero waste groups)
  Layer types: [DSQG, DSQG, DSQG+INT, DSQG, DSQG, DSQG+INT, DSQG, FULL]
  INT at layers 2 and 5 (i%3 == 2); FULL at layer 7
  ~82M parameters (tied input/output embeddings)

HYPERPARAMETERS (H200 SXM — 141 GB HBM3e)
  B=8, GA=4 (same effective batch as condM 85M H200 run)
  LR=3e-4, cosine, AdamW, 10 epochs, FineWeb-Edu 100K docs
  MAX_TRAIN_SEQS=52_716 (iso-compute with all prior runs)

Run on RunPod H200 SXM:
  cd /root/DWARF
  python3 -u benchmarks/train_2048_85m_condU.py \\
    2>&1 | tee benchmarks/logs/condU_85m_run.log

References:
  condM 85M:    36.042 PPL, 83.3% passkey (88M params, L=12, full_layer=11)
  condU 35M:    38.542 PPL, 85.0% passkey (39M params, L=6,  full_layer=5)
  Standard 85M: 39.447 PPL              (101M params, L=12, plain attn)
  condM 27M:    44.500 PPL, 83.3% passkey
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ────────────────────────────────────────────────────────────

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 10
BATCH_SIZE    = 8
GRAD_ACCUM    = 4
LR            = 3e-4
MAX_SEQ_LEN   = 2048
NUM_DOCS      = 100_000

EMBEDDING_DIM   = 768     # D=768  (HD=64 with H=12, power-of-2, Triton-safe)
NUM_LAYERS      = 8
NUM_HEADS       = 12      # d_head = 64
FFN_DIM         = 3072    # 4 × EMBEDDING_DIM
INTERFERENCE    = 3
FULL_ATTN_LAYER = 7       # last layer in L=8 stack (7 DSQG preprocessors)

MAX_TRAIN_SEQS = 52_716   # iso-compute with all prior condU/condM runs

# ── FineWeb-Edu dataset config ────────────────────────────────────────────────

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'logs/condm_fineweb_edu_doc_cache.json'

# ── Passkey eval ──────────────────────────────────────────────────────────────

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Save paths ────────────────────────────────────────────────────────────────

SAVE_DIR    = 'checkpoints/2048_condU_85m_checkpoints'
RESULT_FILE = 'results/condU_85m_results.json'

# ── condN offset set (44 offsets — shared across all condU/condM runs) ─────────

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44

# ── Triton kernel (V3 — Q-weighted scale gains) ────────────────────────────────
# D=768, H=12 → HD=64 (power-of-2): Triton kernel runs natively with zero waste.

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_attention_v3 import dsqg_attention_v3

def dsqg_attention_backend(q, k, v, pos_bias, scale_embed):
    return dsqg_attention_v3(q, k, v, pos_bias, scale_embed)


# ── DSQGAttentionQW — Q-Weighted scale gains + IF amplifier ───────────────────

class DSQGAttentionQW(nn.Module):
    """
    condU DSQG attention with Q-weighted scale gains and IF amplifier.

    Physics parameters:
      pos_bias    [44, H]   — global learned frequency prior (ALiBi-style init)
      scale_embed [44, HD]  — Q-matched-filter (zero init → pure pos_bias at ep0)
      if_gain     [H]       — per-head IF amplifier (1.0 init)

    Huygens K/V injection: kv_inject=(k_delta, v_delta) added to K and V.
    Q unchanged — local oscillator stays clean.
    gate_bias=0.0 — correct design enforced explicitly.
    """
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)  # gate=0 correct design

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in _COND_N_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias    = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.scale_embed = nn.Parameter(torch.zeros(44, HD))  # zero init
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

        # Huygens K/V injection (Q unchanged — local oscillator stays clean)
        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        out = dsqg_attention_backend(q, k, v, self.pos_bias, self.scale_embed)

        # IF amplifier: per-head gain before projection
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


# ── FullCausalAttention ────────────────────────────────────────────────────────

class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)  # gate=0
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
        return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0] * NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0] * NUM_HEADS}


# ── FFN ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ── DSQGBlock ─────────────────────────────────────────────────────────────────

class DSQGBlock(nn.Module):
    """
    condU DSQGBlock with Huygens K/V injection.
    interference=True adds the interference pooling block (Huygens K/V source).
    """
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

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi      = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            counts  = torch.arange(1, N + 1, device=xi.device,
                                   dtype=xi.dtype).view(1, N, 1)
            pool    = xi.cumsum(dim=1) / counts
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            kv_inject = (k_delta, v_delta)

        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


# ── FullAttentionBlock ────────────────────────────────────────────────────────

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


# ── CondUTransformer ──────────────────────────────────────────────────────────

class CondUTransformer(nn.Module):
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
                    dropout=dropout,
                    interference=(i % interference_interval == interference_interval - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight  # tied embeddings
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        # Enforce gate=0 on all gate_proj layers (correct design)
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
                    'if_gain': [1.0] * NUM_HEADS}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
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
        }


# ── Data utilities ────────────────────────────────────────────────────────────

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
        print(f'Loading FineWeb-Edu ({FW_SUBSET}) — seeking {num_docs:,} docs...')
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
        print(f'  Cached {len(texts):,} docs')
    # Shuffle with fixed seed so val/test are drawn from the same distribution
    # as training — NOT from the stream tail which can be OOD.
    import random as _random
    _random.seed(42)
    shuffled = texts[:]
    _random.shuffle(shuffled)
    return {
        'val':   shuffled[:2500],
        'test':  shuffled[2500:5000],
        'train': shuffled[5000:],
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
        with torch.amp.autocast('cuda'):   # match training precision
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
                with torch.amp.autocast('cuda'):   # match training precision
                    logits = model(ids[:, -MAX_SEQ_LEN:])
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
            with torch.amp.autocast('cuda'):   # match training precision
                logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target] + others[:9])[logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ── Training loop ──────────────────────────────────────────────────────────────

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
        gains    = ss['if_gain']
        gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
        print(f'  IF gains:      {gain_str}')

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
        })
        sys.stdout.flush()

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))

    print(f'\n  condU 85M TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')
    print(f'  condM 85M reference: 36.042 PPL | delta = {test_ppl - 36.042:+.3f}')
    print(f'  condU 35M reference: 38.542 PPL | delta = {test_ppl - 38.542:+.3f}')
    print(f'  Standard 85M ref:    39.447 PPL | delta = {test_ppl - 39.447:+.3f}')
    print(f'  condM 27M reference: 44.500 PPL | delta = {test_ppl - 44.500:+.3f}')

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
    print(f'  condM 85M reference: 83.3% passkey')
    print(f'  condU 35M reference: 85.0% passkey')
    print(f'  condM 27M reference: 83.3% passkey')

    ss = model.attn_summary()
    gains    = ss['if_gain']
    gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
    print(f'\n  Final IF gains: {gain_str}')
    print(f'  Final scale_embed: |mean|={ss["scale_embed_abs_mean"]:.4f} '
          f'|max|={ss["scale_embed_abs_max"]:.4f}')

    results = {
        'experiment':              'condU_85m_qweighted_scale',
        'kernel':                  'dsqg_attention_v3',
        'architecture':            f'condU: 7xDSQGQW + 1xFullAttn (layer {FULL_ATTN_LAYER})',
        'embedding_dim':           EMBEDDING_DIM,
        'num_heads':               NUM_HEADS,
        'head_dim':                EMBEDDING_DIM // NUM_HEADS,
        'ffn_dim':                 FFN_DIM,
        'num_layers':              NUM_LAYERS,
        'physics':                 ['q_weighted_scale_gains', 'if_amplifier',
                                    'huygens_kv_injection'],
        'final_test_ppl':          test_ppl,
        'final_passkey_mean':      pk_final_mean,
        'final_passkey_by_d':      {str(d): v for d, v in pk_final.items()},
        'per_epoch':               per_epoch_results,
        'temperature_sweep':       sweep_results,
        'attn_summary':            ss,
        'condm_85m_ref_ppl':       36.042,
        'condm_85m_ref_passkey':   0.833,
        'condu_35m_ref_ppl':       38.542,
        'condu_35m_ref_passkey':   0.850,
        'standard_85m_ref_ppl':    39.447,
        'condm_27m_ref_ppl':       44.500,
        'condm_27m_ref_passkey':   0.833,
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.set_float32_matmul_precision('high')   # enable TF32 on Ampere/Hopper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  condU 85M — Q-Weighted Scale Gains + IF Amplifier + Huygens K/V')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, d_head={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Full attention at layer {FULL_ATTN_LAYER} (0-indexed)')
    print(f'  Layer types: ' +
          ', '.join('FULL' if i == FULL_ATTN_LAYER
                    else ('DSQG+INT' if i % INTERFERENCE == INTERFERENCE - 1 else 'DSQG')
                    for i in range(NUM_LAYERS)))
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Kernel: dsqg_attention_v3 (HD={EMBEDDING_DIM//NUM_HEADS}, '
          f'BLOCK_HD={EMBEDDING_DIM//NUM_HEADS}, scale_embed [44,{EMBEDDING_DIM//NUM_HEADS}])')
    print(f'  Physics: QW scale gains + IF amplifier + Huygens K/V injection')
    print(f'  Architecture: INTERFERENCE={INTERFERENCE}, gate=0, '
          f'1 FullAttn @ layer {FULL_ATTN_LAYER}')
    print(f'  References: condM_85M=36.042/83.3%pk | condU_35M=38.542/85.0%pk '
          f'| Standard_85M=39.447')

    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    splits = load_data(NUM_DOCS)

    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    _tok_candidates = [
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

    _encoded_cache = 'logs/fineweb_encoded_2048.pt'
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
            print(f'  Capped to {MAX_TRAIN_SEQS:,} train seqs (iso-compute)')
        val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
        test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')
        torch.save({'train': train_data, 'val': val_data, 'test': test_data},
                   _encoded_cache)
        print(f'  Saved encoded cache → {_encoded_cache}')

    model = CondUTransformer(
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
    layer_types = []
    for i in range(NUM_LAYERS):
        if i == FULL_ATTN_LAYER:
            layer_types.append('FULL')
        elif i % INTERFERENCE == INTERFERENCE - 1:
            layer_types.append('DSQG+INT')
        else:
            layer_types.append('DSQG')
    print(f'\ncondU 85M: {n_params:,} parameters')
    print(f'  Layer types: {layer_types}')

    if not causality_check(model, device):
        return

    # torch.compile — mode='default' (not reduce-overhead; CUDA graphs conflict
    # with Triton V3 kernel's save_for_backward tensors)
    print('  Compiling model with torch.compile (mode=default)...')
    model = torch.compile(model, mode='default')
    print('  Compile done.')

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
