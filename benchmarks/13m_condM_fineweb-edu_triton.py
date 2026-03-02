"""
condM 13M — FineWeb-Edu Dataset, Documents Filtered to >= 2048 Tokens

Hypothesis under test:
  OpenWebText documents average well under 2048 tokens, so training sequences
  are multi-document concatenations. Long-range positions within a training
  sequence (d >= 256) often cross a document boundary — the model sees
  unrelated content at those offsets, never receiving consistent gradient
  signal for pos_bias specialization at long range.

  FineWeb-Edu filtered to >= 2048 tokens ensures every training sequence is
  sourced entirely from a single document. Long-range positions within the
  sequence reflect genuine document-internal coherence (narrative, argument,
  citation patterns), giving pos_bias weights a consistent signal to learn.

  Prediction: earlier passkey emergence and better d >= 256 accuracy vs OWT
  baseline, despite identical architecture and training budget.

Architecture: IDENTICAL to train_2048_condM_layer_ablation_triton.py
  - 5 DSQG layers + 1 full causal attention (layer 5, no postprocessing)
  - Triton/custom DSQG kernel (DSQGAttentionN_Fused)
  - 13M parameters: D=256, H=8, L=6, FFN=1024, interference=3
  - Tied embeddings, condI BPE tokenizer (32K vocab)

Dataset changes vs OWT baseline:
  - Source:   HuggingFaceFW/fineweb-edu, sample-10BT subset (public, no auth)
  - Filter:   documents must tokenize to >= MAX_SEQ_LEN (2048) tokens exactly
  - Encoding: IDENTICAL to OWT baseline (concatenate-and-reshape/packing).
              Keeps encoding as a controlled variable — only dataset changes.
              Because docs are >=2048 tokens, cross-doc boundaries are rare.
  - Count:    first NUM_DOCS=100_000 qualifying documents (same as OWT runs)

Per-epoch passkey eval: same 10-way forced-choice format as eval_suite.py
Results:    benchmarks/logs/condm_fineweb_edu_results.json
Checkpoint: checkpoints/condm_fineweb_edu/best.pt  (+ epoch_NN.pt per epoch)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/13m_condM_fineweb-edu_triton.py \
    2>&1 | tee benchmarks/logs/condm_fineweb_edu_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ── Hyperparameters (identical to condM OWT baseline) ─────────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000      # qualifying documents (>= MAX_SEQ_LEN tokens each)
MAX_TRAIN_SEQS  = 52_716       # cap train sequences to match OWT baseline (iso-compute)

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5            # last layer — same as reference condM

# ── FineWeb-Edu dataset config ─────────────────────────────────────────────────

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'   # ~10B tokens, ~2.5M docs, public (no auth)
FW_MIN_CHARS    = 5_000           # fast pre-filter before tokenizing
                                  # (~5000 chars => ~1250 tokens; exact check follows)

# ── Passkey eval config (identical to eval_suite.py / chinchilla_repeated) ────

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Save paths ─────────────────────────────────────────────────────────────────

SAVE_DIR    = 'checkpoints/condm_fineweb_edu'
RESULT_FILE = 'benchmarks/logs/condm_fineweb_edu_results.json'

# ── condN offset set (identical to all condM variants) ────────────────────────

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44


# ── Triton kernel import ───────────────────────────────────────────────────────

import pathlib as _pl
_kernel_dir = str(_pl.Path.home() / '.openclaw/workspace/kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)
from dsqg_attention_v2 import DSQGAttentionN_Fused as DSQGAttentionN


# ── FullCausalAttention (byte-for-byte identical to layer_ablation_triton) ────

class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
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
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)

        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return F.dropout(self.out_proj(out_flat * gate),
                         p=self.dropout_p, training=self.training)

    def attn_summary(self):
        return {'type': 'full_causal', 'pos_bias_abs_mean': 0.0,
                'pos_bias_abs_max': 0.0, 'pos_bias_mean_per_head': [0.0] * NUM_HEADS}


# ── FFN (identical) ───────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ── DSQGBlock (identical) ──────────────────────────────────────────────────────

class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, use_checkpoint=True, interference=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.interference   = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionN(
            embedding_dim, num_heads, seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_pool = nn.Linear(embedding_dim, embedding_dim)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        if self.use_checkpoint:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_fn, x, use_reentrant=False)
        else:
            x = x + self._attn_fn(x)

        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)

        x = x + self.ffn(self.norm2(x))
        return x


# ── FullAttentionBlock (identical) ────────────────────────────────────────────

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


# ── CondMTransformer (identical) ──────────────────────────────────────────────

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
                    dropout=dropout, use_checkpoint=True,
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
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'gate_proj'):
                nn.init.constant_(block.attn.gate_proj.bias, 2.0)

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
                    'pos_bias_mean_per_head': [0.0] * NUM_HEADS}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        return {
            'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean'] for s in summaries) / n,
            'pos_bias_abs_max':       max(s['pos_bias_abs_max']  for s in summaries),
            'pos_bias_mean_per_head': [
                sum(s['pos_bias_mean_per_head'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
        }


# ── BPE tokenizer wrapper (identical) ─────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


# ── Dataset loading: FineWeb-Edu, filtered to >= MAX_SEQ_LEN tokens ───────────

# Path for caching the filtered doc list — avoids re-streaming on every run
FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'


def load_fineweb_edu(tokenizer, num_docs=NUM_DOCS):
    """
    Stream FineWeb-Edu (sample-10BT), keeping only documents that tokenize to
    at least MAX_SEQ_LEN tokens. Returns raw text lists for train/val/test.

    On first run: streams from HuggingFace, filters, saves texts to FW_CACHE_FILE.
    On subsequent runs: loads from FW_CACHE_FILE directly (no streaming needed).

    Two-stage filter (first run only):
      1. Character count >= FW_MIN_CHARS  (fast, no tokenization)
      2. Token count >= MAX_SEQ_LEN       (exact, using condI tokenizer)
    """
    import json as _json

    # ── Cache hit: load and return immediately ─────────────────────────────
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu doc list from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = _json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
        n = len(texts)
        return {
            'train': texts[:int(n * 0.95)],
            'val':   texts[int(n * 0.95) : int(n * 0.95) + 2500],
            'test':  texts[int(n * 0.95) + 2500 : int(n * 0.95) + 5000],
        }

    # ── Cache miss: stream, filter, save ──────────────────────────────────
    from datasets import load_dataset

    print(f'Loading FineWeb-Edu ({FW_SUBSET}) — seeking {num_docs:,} docs '
          f'with >= {MAX_SEQ_LEN} tokens...')
    print(f'  Pre-filter: >= {FW_MIN_CHARS:,} chars before tokenizing')
    print(f'  Will cache results to: {FW_CACHE_FILE}')

    ds       = load_dataset(FW_DATASET_NAME, name=FW_SUBSET,
                            split='train', streaming=True)
    texts    = []
    examined = 0

    for item in ds:
        examined += 1
        text = item['text']

        # Stage 1: fast character-length pre-filter
        if len(text) < FW_MIN_CHARS:
            continue

        # Stage 2: exact token count
        toks = tokenizer.encode(text)
        if len(toks) >= MAX_SEQ_LEN:
            texts.append(text)
            if len(texts) % 10_000 == 0:
                print(f'  {len(texts):,} qualifying docs '
                      f'(examined {examined:,}, '
                      f'pass-rate {len(texts)/examined*100:.1f}%)')

        if len(texts) >= num_docs:
            break

    print(f'  Done. {len(texts):,} qualifying docs from {examined:,} examined '
          f'({len(texts)/max(examined,1)*100:.1f}% pass-rate)')

    # Save cache so future runs skip streaming entirely
    os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
    with open(FW_CACHE_FILE, 'w') as fp:
        _json.dump(texts, fp)
    print(f'  Cached to {FW_CACHE_FILE} ({os.path.getsize(FW_CACHE_FILE)/1e6:.1f} MB)')

    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95) : int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500 : int(n * 0.95) + 5000],
    }


# ── Sequence encoding: identical packing to OWT baseline ─────────────────────
#
# Using the same concatenate-and-reshape strategy as train_2048_condM_layer_ablation_triton.py.
# This is intentional: we want a direct comparison where the ONLY variable is
# dataset quality and document length distribution, not encoding strategy.
#
# Because FineWeb-Edu docs are filtered to >= MAX_SEQ_LEN tokens, cross-document
# boundaries within a packed sequence will be far less frequent than with OWT
# (OWT avg doc ~600 tokens => ~3-4 docs/sequence; FW-Edu >=2048 tokens => ~1 doc/sequence).
# That benefit emerges naturally from the length filter without changing encoding.

def encode_split(split_texts, tokenizer, max_seq_len, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)  # EOS/doc boundary marker
    n    = (len(tokens) // max_seq_len) * max_seq_len
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


# ── Evaluate, generate, causality check (identical to base script) ────────────

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size, batch_size):
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


# ── Passkey evaluation (identical to chinchilla_repeated / eval_suite.py) ─────

def passkey_accuracy(model, tokenizer, device):
    """10-way forced-choice word retrieval — exact format from eval_suite.py."""
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

    # Chinchilla markers
    tokens_per_epoch = len(train_data) * (MAX_SEQ_LEN - 1)
    chin_tokens      = 20 * model.param_count()
    chin_epoch       = chin_tokens / tokens_per_epoch
    print(f'\n  Train sequences: {len(train_data):,}')
    print(f'  Tokens/epoch:    {tokens_per_epoch:,}')
    print(f'  Chinchilla:      {chin_tokens:,} tokens (epoch ~{chin_epoch:.2f})\n')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start : idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda'):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0
        chin_pct   = epoch * tokens_per_epoch / chin_tokens * 100

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt'))
            marker = ' * BEST'

        # Per-epoch checkpoint (enables post-hoc analysis)
        torch.save({
            'epoch':             epoch,
            'model_state_dict':  model.state_dict(),
            'val_ppl':           val_ppl,
            'chinchilla_pct':    chin_pct,
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

        print('  ── Generation samples (greedy) ──')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')
        print('  ──')

        # Per-epoch passkey eval
        print('  Passkey...')
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        above50 = sum(1 for v in pk.values() if v >= 0.5)
        print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
        parts = [f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))

        per_epoch_results.append({
            'epoch':          epoch,
            'val_ppl':        val_ppl,
            'train_loss':     train_loss,
            'chinchilla_pct': chin_pct,
            'elapsed_s':      elapsed,
            'passkey_mean':   pk_mean,
            'passkey_by_d':   {str(d): v for d, v in pk.items()},
        })
        sys.stdout.flush()

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condM (FineWeb-Edu) TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')

    # Temperature sweep
    print('\n  ── Temperature sweep (best checkpoint) ──')
    sweep_results = {}
    for temp in [0.0, 0.5, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        print(f'\n  [{label}]')
        gens = generate(model, tokenizer, GEN_PROMPTS, device,
                        temperature=temp, top_p=0.9)
        sweep_results[label] = gens
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} -> {repr(gen[:80])}')

    # Final passkey on best checkpoint
    pk_final      = passkey_accuracy(model, tokenizer, device)
    pk_final_mean = sum(pk_final.values()) / len(pk_final)
    above50_final = sum(1 for v in pk_final.values() if v >= 0.5)
    print(f'\n  Final passkey (best ckpt): mean={pk_final_mean*100:.1f}%  '
          f'({above50_final}/{len(pk_final)} distances >50%)')
    parts = [f'd={d}:{int(pk_final[d]*100)}%' for d in PASSKEY_DISTANCES]
    print('  ' + '  '.join(parts))

    ss = model.attn_summary()

    # ── Emergence summary table ────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  condM 13M -- FineWeb-Edu Emergence (packed encoding, direct OWT comparison)')
    print(f'  Chinchilla epoch ~{chin_epoch:.2f}  |  '
          f'OWT reference: 54.529 PPL / 83.3% passkey')
    print('=' * 72)
    print(f'  {"Ep":<4} {"Tokens":>8} {"  %Chin":>7} {"ValPPL":>8} {"Passkey":>9}')
    for r in per_epoch_results:
        ep_tok = r['epoch'] * tokens_per_epoch
        cf = ' <- ~CHINCHILLA' if abs(r['epoch'] - chin_epoch) < 0.6 else ''
        print(f'  {r["epoch"]:<4} {ep_tok//1_000_000:>6}M '
              f'{r["chinchilla_pct"]:>7.0f}% '
              f'{r["val_ppl"]:>8.1f} '
              f'{r["passkey_mean"]*100:>8.1f}%{cf}')
    print('=' * 72)

    results = {
        'experiment':             'condm_fineweb_edu',
        'dataset':                f'{FW_DATASET_NAME}/{FW_SUBSET}',
        'min_doc_tokens':         MAX_SEQ_LEN,
        'encoding':               'packed_identical_to_owt_baseline',
        'num_docs':               NUM_DOCS,
        'chinchilla_epoch':       chin_epoch,
        'final_test_ppl':         test_ppl,
        'final_passkey_mean':     pk_final_mean,
        'final_passkey_by_d':     {str(d): v for d, v in pk_final.items()},
        'per_epoch':              per_epoch_results,
        'temperature_sweep':      sweep_results,
        'attn_summary':           ss,
        'owt_baseline_ppl':       54.529,
        'owt_baseline_passkey':   0.833,
    }

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 72)
    print('  condM 13M -- FineWeb-Edu (>=2048 tok/doc, document-aware encoding)')
    print('=' * 72)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'  Dataset:        {FW_DATASET_NAME}/{FW_SUBSET}')
    print(f'  Min doc tokens: {MAX_SEQ_LEN}  |  Num docs: {NUM_DOCS:,}')
    print(f'  Encoding:       packed (identical to OWT baseline — direct comparison)')
    print(f'  Architecture:   5 DSQG + 1 FullAttn (layer {FULL_ATTN_LAYER}) | Triton kernel')

    os.makedirs('benchmarks/logs', exist_ok=True)

    # Load condI BPE tokenizer (same as all OWT runs)
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
        raise FileNotFoundError(
            'condI tokenizer not found. Tried:\n' +
            '\n'.join(f'  {p}' for p in _tok_candidates))

    # Load and filter FineWeb-Edu (tokenizer required for exact length check)
    splits = load_fineweb_edu(tokenizer, num_docs=NUM_DOCS)

    # Document-aware encoding
    print(f'\nEncoding data (document-aware, max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    if len(train_data) > MAX_TRAIN_SEQS:
        idx = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
        train_data = train_data[idx]
        print(f'  Capped to {MAX_TRAIN_SEQS:,} train sequences (OWT iso-compute)')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    # Build model (identical architecture to OWT baseline)
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

    n_params    = model.param_count()
    layer_types = ['FULL' if i == FULL_ATTN_LAYER else 'DSQG'
                   for i in range(NUM_LAYERS)]
    print(f'\ncondM (FineWeb-Edu): {n_params:,} parameters')
    print(f'  Layer types: {layer_types}')

    if not causality_check(model, device):
        return

    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
