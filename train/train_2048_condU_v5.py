"""
condU-v5 — MOVT + QK-OVT + NPCI (13M, FineWeb-Edu)

Extends condU (V3 kernel) with three new mechanisms from GPT-5.4-Pro analysis,
confirmed mathematically in the Rust verification crate (March 6, 2026):

  1. MOVT (Multi-Plane Orthogonal Value Transport)
     Replaces scalar phase_embed [44,H] with phase_base [44,H,2] — two
     disjoint rotation planes (0,1) and (2,3) per (offset, head).
     Planes are disjoint → commutative (order doesn't matter).
     Zero-init → identity → backward-compatible with V3.

  2. QK-OVT (Query-Key Conditioned Orthogonal Value Transport)
     The rotation angles become content-dependent:
       theta_m = phase_base[j,h,m] + phase_gain[j,h,m] * y[n,m] * z[t,m]
     where y = Q @ query_probes.T / sqrt(HD) (precomputed, varies per n)
           z = K @ key_probes.T / sqrt(HD)   (precomputed, varies per t)
     All parameters zero-init: starts as pure MOVT, learns content-dependence.
     This is the "fifth mechanism" in the DSQG taxonomy: content-dependent
     V readout, orthogonal to score computation.

  3. NPCI (Norm-Preserving Coupled Injection)
     Replaces raw additive K/V injection with bounded-angle rotation:
       k' = cos(theta_k[h]) * k + sin(theta_k[h]) * ||k|| * u_hat_perp
     Preserves ||k'|| = ||k|| exactly. Targets K/V coupling instability.
     Uses relative threshold (not absolute) for the degenerate case.
     Zero-init → raw additive (backward-compatible at epoch 0).

Performance fix (V4 → V5):
   V5 loads V contiguously [BLOCK_N, HD] and applies rotations in-register.
   Targets ~V3 timing (~400s/epoch at 13M vs V4's ~1200s/epoch).

Architecture (matches condU 35M paper baseline):
  EMBEDDING_DIM = 512, NUM_LAYERS = 6, NUM_HEADS = 8, FFN_DIM = 2048
  INTERFERENCE = 3, FULL_ATTN_LAYER = 5
  ~35M parameters | Chinchilla at ~ep6.5 | coupling = 8*512*6 = 24,576 ✓

Key logging additions vs condU:
  Per epoch: phase_base stats (mean, max, local/sparse, by-head, plane-diff)
             phase_gain stats, probe norms, npci_theta per head
  These track whether MOVT planes diverge, QK-OVT activates, NPCI rotates.

Baseline references:
  condU V3 13M:  52.237 PPL, 38.3% passkey
  condU 35M:     38.542 PPL, 85.0% passkey (paper headline)
  standard 85M:  39.447 PPL, 96.7% passkey

Run:
  cd /home/dlewis3/Desktop/AI/DWARF
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_condU_v5.py \
    2>&1 | tee benchmarks/logs/condU_v5_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (identical to all 13M condU runs) ─────────────────────────

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

# ── Passkey eval ───────────────────────────────────────────────────────────────

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 50
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Save paths ─────────────────────────────────────────────────────────────────

SAVE_DIR    = 'checkpoints/condU_v5'
RESULT_FILE = 'logs/condU_v5_results.json'

# ── Kernel import ──────────────────────────────────────────────────────────────

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
_cuda_ext_dir = str(_pl.Path(__file__).parent.parent / 'kernels' / 'dsqg_cuda')
for _d in [_kernel_dir, _cuda_ext_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Use CUDA kernel if available, fall back to Triton.
# Strategy: monkey-patch the module-level dsqg_attention_v5 function so that
# DSQGAttentionV5.forward() (which calls it by name) transparently uses the
# CUDA kernel. No subclassing needed — identical call signature.
import dsqg_attention_v5 as _v5_module
try:
    import dsqg_cuda as _dsqg_cuda_ext  # noqa: F401 (side-effect: loads .so)
    from dsqg_attention_v5_cuda import dsqg_attention_v5_cuda as _cuda_attn_fn
    _v5_module.dsqg_attention_v5 = _cuda_attn_fn   # hot-swap kernel
    _USE_CUDA_KERNEL = True
    print('[kernel] CUDA extension loaded — using compiled CUDA backward')
except ImportError:
    _USE_CUDA_KERNEL = False
    print('[kernel] CUDA extension not found — falling back to Triton')

from dsqg_attention_v5 import DSQGAttentionV5, npci_rotate

# ── Offset set ─────────────────────────────────────────────────────────────────

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44


# ── DSQGBlock V5 ───────────────────────────────────────────────────────────────

class DSQGBlockV5(nn.Module):
    """
    condU-v5 DSQG block: V5 kernel + Huygens interference with NPCI injection.
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionV5(embedding_dim, num_heads,
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
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            counts  = torch.arange(1, N+1, device=xi.device,
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


# ── FullCausalAttention (unchanged) ───────────────────────────────────────────

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
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B,N,H,HD).permute(0,2,1,3)
        k = k.view(B,N,H,HD).permute(0,2,1,3)
        v = v.view(B,N,H,HD).permute(0,2,1,3)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        out_flat = out.permute(0,2,1,3).reshape(B,N,D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return F.dropout(self.out_proj(out_flat * gate),
                         p=self.dropout_p, training=self.training)

    def attn_summary(self):
        return {'type': 'full_causal',
                'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0]*NUM_HEADS,
                'phase_base_abs_mean': 0.0, 'phase_base_plane_diff': 0.0,
                'phase_gain_abs_mean': 0.0,
                'npci_theta_k': [0.0]*NUM_HEADS, 'npci_theta_v': [0.0]*NUM_HEADS}


# ── FFN (unchanged) ────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ── FullAttentionBlock (unchanged) ────────────────────────────────────────────

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


# ── CondUV5Transformer ────────────────────────────────────────────────────────

class CondUV5Transformer(nn.Module):
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
                blocks.append(DSQGBlockV5(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout,
                    interference=(i % interference_interval ==
                                  interference_interval - 1)))
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
        # Nudge init for phase_base: N(0, 0.01) — breaks zero-symmetry
        # phase_gain, query_probes, key_probes, npci_theta: remain zero
        for m in self.modules():
            if isinstance(m, DSQGAttentionV5):
                nn.init.normal_(m.phase_base,    0.0, 0.01)   # MOVT: breaks zero-symmetry
                nn.init.normal_(m.query_probes,  0.0, 0.01)   # QK-OVT: enables y_pre ≠ 0
                nn.init.normal_(m.key_probes,    0.0, 0.01)   # QK-OVT: enables z_pre ≠ 0
                nn.init.normal_(m.phase_gain,    0.0, 0.001)  # QK-OVT: smaller — gain coeff

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
        dsqg_blocks = [b for b in self.blocks if isinstance(b, DSQGBlockV5)]
        if not dsqg_blocks:
            return {}
        sums = [b.attn.attn_summary() for b in dsqg_blocks]
        n    = len(sums)

        def avg(key):  return sum(s[key] for s in sums) / n
        def mx(key):   return max(s[key] for s in sums)
        def avg_list(key): return [sum(s[key][h] for s in sums)/n
                                   for h in range(NUM_HEADS)]
        return {
            'pos_bias_abs_mean':           avg('pos_bias_abs_mean'),
            'pos_bias_abs_max':            mx('pos_bias_abs_max'),
            'pos_bias_mean_per_head':      avg_list('pos_bias_mean_per_head'),
            'scale_embed_abs_mean':        avg('scale_embed_abs_mean'),
            'scale_embed_abs_max':         mx('scale_embed_abs_max'),
            'if_gain':                     avg_list('if_gain'),
            'phase_base_abs_mean':         avg('phase_base_abs_mean'),
            'phase_base_abs_max':          mx('phase_base_abs_max'),
            'phase_base_local_mean':       avg('phase_base_local_mean'),
            'phase_base_sparse_mean':      avg('phase_base_sparse_mean'),
            'phase_base_by_head':          avg_list('phase_base_by_head'),
            'phase_base_sparse_by_head':   avg_list('phase_base_sparse_by_head'),
            'phase_base_plane_diff':       avg('phase_base_plane_diff'),
            'phase_base_p0_sparse':        avg_list('phase_base_p0_sparse'),
            'phase_base_p1_sparse':        avg_list('phase_base_p1_sparse'),
            'phase_gain_abs_mean':         avg('phase_gain_abs_mean'),
            'phase_gain_abs_max':          mx('phase_gain_abs_max'),
            'phase_gain_sparse_mean':      avg('phase_gain_sparse_mean'),
            'phase_gain_by_head':          avg_list('phase_gain_by_head'),
            'query_probe_norm':            sums[0]['query_probe_norm'],
            'key_probe_norm':              sums[0]['key_probe_norm'],
            'npci_theta_k':                avg_list('npci_theta_k'),
            'npci_theta_v':                avg_list('npci_theta_v'),
        }


# ── Data utilities (identical to condU) ───────────────────────────────────────

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
            _json.dump(texts, fp)
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
        x = data[i:i+batch_size, :-1].to(device)
        y = data[i:i+batch_size,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
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
                ids = torch.cat([ids, next_id.view(1,1)], dim=1)
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
            cand_ids = [(tokenizer.encode(' '+w) or tokenizer.encode(w))[0]
                        for w in [target]+others[:9]]
            correct  += int(([target]+others[:9])[
                             logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ── Training loop ─────────────────────────────────────────────────────────────

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

    best_val_loss = float('inf')
    best_val_ppl  = float('inf')
    best_epoch    = 0
    t0            = time.time()
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
        t_epoch_start   = time.time()

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start : idx_start + BATCH_SIZE]]
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
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item()*GRAD_ACCUM:.4f}')

        epoch_time = time.time() - t_epoch_start
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
              f'| {elapsed:.0f}s ({chin_pct:.0f}%C) | ep={epoch_time:.0f}s')

        ss = model.attn_summary()

        # ── Attention diagnostics ──────────────────────────────────────────
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

        # MOVT
        print(f'  phase_base:    |mean|={ss["phase_base_abs_mean"]:.4f} '
              f'|max|={ss["phase_base_abs_max"]:.4f} '
              f'sparse={ss["phase_base_sparse_mean"]:.4f} '
              f'plane_diff={ss["phase_base_plane_diff"]:.4f}')
        pb_sp = ss['phase_base_sparse_by_head']
        pb_str = '  '.join(f'h{h}:{pb_sp[h]:.3f}' for h in range(NUM_HEADS))
        print(f'  phase_base_sp: {pb_str}')
        p0 = ss['phase_base_p0_sparse']; p1 = ss['phase_base_p1_sparse']
        p0_str = '  '.join(f'h{h}:{p0[h]:+.3f}' for h in range(NUM_HEADS))
        p1_str = '  '.join(f'h{h}:{p1[h]:+.3f}' for h in range(NUM_HEADS))
        print(f'  plane0 sparse: {p0_str}')
        print(f'  plane1 sparse: {p1_str}')

        # QK-OVT
        print(f'  phase_gain:    |mean|={ss["phase_gain_abs_mean"]:.4f} '
              f'|max|={ss["phase_gain_abs_max"]:.4f} '
              f'sparse={ss["phase_gain_sparse_mean"]:.4f}')
        pg_sp = ss['phase_gain_by_head']
        pg_str = '  '.join(f'h{h}:{pg_sp[h]:.4f}' for h in range(NUM_HEADS))
        print(f'  phase_gain_hd: {pg_str}')

        # QK-OVT probes
        qpn = ss['query_probe_norm']; kpn = ss['key_probe_norm']
        print(f'  probe norms:   q=[{qpn[0]:.4f},{qpn[1]:.4f}] '
              f'k=[{kpn[0]:.4f},{kpn[1]:.4f}]')

        # NPCI
        thk = ss['npci_theta_k']; thv = ss['npci_theta_v']
        thk_str = '  '.join(f'h{h}:{thk[h]:+.4f}' for h in range(NUM_HEADS))
        thv_str = '  '.join(f'h{h}:{thv[h]:+.4f}' for h in range(NUM_HEADS))
        print(f'  npci_theta_k:  {thk_str}')
        print(f'  npci_theta_v:  {thv_str}')

        # ── Generation ────────────────────────────────────────────────────
        print('  -- Generation (T=0.7) --')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.7)):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')
        print('  --')

        # ── Passkey ───────────────────────────────────────────────────────
        print('  Passkey...')
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        above50 = sum(1 for v in pk.values() if v >= 0.5)
        print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        per_epoch_results.append({
            'epoch':                     epoch,
            'val_ppl':                   val_ppl,
            'train_loss':                train_loss,
            'chinchilla_pct':            chin_pct,
            'elapsed_s':                 elapsed,
            'epoch_time_s':              epoch_time,
            'passkey_mean':              pk_mean,
            'passkey_by_d':              {str(d): v for d, v in pk.items()},
            # DSQG core
            'scale_embed_abs_mean':      ss['scale_embed_abs_mean'],
            'if_gain':                   ss['if_gain'],
            # MOVT
            'phase_base_abs_mean':       ss['phase_base_abs_mean'],
            'phase_base_sparse_mean':    ss['phase_base_sparse_mean'],
            'phase_base_plane_diff':     ss['phase_base_plane_diff'],
            'phase_base_by_head':        ss['phase_base_by_head'],
            'phase_base_sparse_by_head': ss['phase_base_sparse_by_head'],
            'phase_base_p0_sparse':      ss['phase_base_p0_sparse'],
            'phase_base_p1_sparse':      ss['phase_base_p1_sparse'],
            # QK-OVT
            'phase_gain_abs_mean':       ss['phase_gain_abs_mean'],
            'phase_gain_sparse_mean':    ss['phase_gain_sparse_mean'],
            'phase_gain_by_head':        ss['phase_gain_by_head'],
            'query_probe_norm':          ss['query_probe_norm'],
            'key_probe_norm':            ss['key_probe_norm'],
            # NPCI
            'npci_theta_k':              ss['npci_theta_k'],
            'npci_theta_v':              ss['npci_theta_v'],
        })
        sys.stdout.flush()

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condU-v5 TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')
    print(f'  condU V3 baseline: 52.237 PPL | delta = {test_ppl - 52.237:+.3f}')
    print(f'  condM 85M:         36.042 PPL | delta = {test_ppl - 36.042:+.3f}')

    print('\n  -- Temperature sweep --')
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

    ss = model.attn_summary()
    results = {
        'experiment':            'condU_v5_movt_qkovt_npci',
        'kernel':                'dsqg_attention_v5_cuda' if _USE_CUDA_KERNEL else 'dsqg_attention_v5',
        'architecture': {
            'embedding_dim': EMBEDDING_DIM, 'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS, 'ffn_dim': FFN_DIM,
            'full_attn_layer': FULL_ATTN_LAYER,
            'interference': INTERFERENCE,
        },
        'new_mechanisms':        ['MOVT_r2', 'QK-OVT', 'NPCI'],
        'final_test_ppl':        test_ppl,
        'final_passkey_mean':    pk_final_mean,
        'final_passkey_by_d':    {str(d): v for d, v in pk_final.items()},
        'per_epoch':             per_epoch_results,
        'temperature_sweep':     sweep_results,
        'attn_summary':          ss,
        'references': {
            'condU_v3_13M_ppl':      52.237,
            'condU_v3_13M_passkey':  0.383,
            'condU_35M_ppl':         38.542,
            'condU_35M_passkey':     0.850,
            'standard_85M_ppl':      39.447,
            'standard_85M_passkey':  0.967,
        },
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  condU-v5 — MOVT + QK-OVT + NPCI (13M, FineWeb-Edu)')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Kernel: dsqg_attention_v5')
    print(f'  Architecture: D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Coupling coeff: H*D*L = {NUM_HEADS}*{EMBEDDING_DIM}*{NUM_LAYERS} = {NUM_HEADS*EMBEDDING_DIM*NUM_LAYERS:,} (threshold: 36,865)')
    print(f'  New: MOVT(r=2 planes) + QK-OVT + NPCI (all zero-init @ start)')
    print(f'  References: condU-35M-V3=38.542/85.0% | std-85M=39.447/96.7%')

    os.makedirs('logs', exist_ok=True)

    splits = load_data(NUM_DOCS)

    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    _tok_candidates = [
        os.path.join(_script_dir, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '..', 'benchmarks', 'results', '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    if tok_path:
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'\nLoaded condI BPE tokenizer from {tok_path}')
    else:
        raise FileNotFoundError('condI tokenizer not found — expected at results/2048_condI_tokenizer.json')

    _encoded_cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache} ...')
        _cache = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
        test_data  = _cache['test']
        print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
              f'test: {len(test_data):,} seqs')
    else:
        print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
        train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
        val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
        test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = CondUV5Transformer(
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
    layer_types = ['FULL' if i == FULL_ATTN_LAYER else 'DSQGv5'
                   for i in range(NUM_LAYERS)]
    print(f'\ncondU-v5: {n_params:,} parameters')
    print(f'  Layer types: {layer_types}')
    HD = EMBEDDING_DIM // NUM_HEADS
    print(f'  New params per DSQGv5 layer (sparse-only MOVT — 11 global offsets):')
    print(f'    phase_base  [11, {NUM_HEADS}, 2] = {11*NUM_HEADS*2:,}  (MOVT, nudge-init)')
    print(f'    phase_gain  [11, {NUM_HEADS}, 2] = {11*NUM_HEADS*2:,}  (QK-OVT, zero-init)')
    print(f'    query_probes [2, {HD}]   = {2*HD:,}    (QK-OVT, zero-init)')
    print(f'    key_probes   [2, {HD}]   = {2*HD:,}    (QK-OVT, zero-init)')
    print(f'    npci_theta_k [{NUM_HEADS}]       = {NUM_HEADS:,}       (NPCI, zero-init)')
    print(f'    npci_theta_v [{NUM_HEADS}]       = {NUM_HEADS:,}       (NPCI, zero-init)')

    MAX_TRAIN_SEQS = 52_716
    if len(train_data) > MAX_TRAIN_SEQS:
        idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
        train_data = train_data[idx]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
          f'test: {len(test_data):,} seqs (capped to {MAX_TRAIN_SEQS:,})')

    if not causality_check(model, device):
        return

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
