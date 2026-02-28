"""
DWARF 85M condM — Scaling: 11:1 Hybrid at ~88M Parameters

Architecture
------------
  D=640, H=8, d_head=80, L=12, FFN=2560
  Layers 0-10 : condN DSQG (44 offsets, interference every 3rd)
  Layer    11 : Full O(N^2) causal attention (SDPA / FlashAttention)
  ~88.3M parameters (vs 84.5M standard transformer baseline -- within 5%)

Purpose
-------
  Final scaling point in the condM ablation series:
    13M condM  -> test PPL 54.529
    27M condM  -> test PPL 44.5
    85M condM  -> expected ~41-46 PPL

  Primary claim: condM hybrid outperforms standard transformer at iso-scale.
    Standard transformer 27M  -> test PPL 50.683
    Standard transformer 85M  -> test PPL 57.7

H200 Optimisations
------------------
  - BATCH_SIZE=64 (2x the 27M H100 run; 141 GB HBM3e has headroom to spare)
  - torch.set_float32_matmul_precision('high')  -> TF32 on Hopper for free
  - AMP bf16 (native Hopper dtype)
  - SDPA / FlashAttention in FullCausalAttention (unchanged from 27M)
  - Gradient checkpointing in DSQGBlock (use_checkpoint=True)
  - Linear LR warmup (5% of steps) -> cosine decay to 10% of peak LR
  - Two checkpoint files:
      best.pt              -- plain state_dict (eval_suite-compatible)
      checkpoint_latest.pt -- full resumable dict (model/optimizer/epoch)
  - Per-epoch numbered checkpoints for pod-termination recovery

RunPod command (H200 SXM pod, /workspace volume)
-------------------------------------------------
  git clone https://github.com/dlewis3/DWARF /workspace/DWARF
  cd /workspace/DWARF
  pip install tokenizers datasets
  tmux new-session -d -s condM85m -x 220 -y 50
  tmux send-keys -t condM85m \
    "python3 -u benchmarks/train_2048_85m_condM.py \
    2>&1 | tee benchmarks/logs/85m_condM_run.log" Enter
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# H200: TF32 matmuls on Hopper -- free ~10% throughput
torch.set_float32_matmul_precision('high')

# ---- Hyperparameters ---------------------------------------------------------

VOCAB_SIZE     = 32000
NUM_EPOCHS     = 10
BATCH_SIZE     = 64       # H200 141 GB -- comfortable at B=64 with grad checkpointing
GRAD_ACCUM     = 1        # effective batch = 64
LR             = 3e-4     # same as 27M condM
WARMUP_FRAC    = 0.05     # linear warmup over first 5% of total steps
ETA_MIN_FRAC   = 0.10     # cosine tail floor = 10% of peak LR
MAX_SEQ_LEN    = 2048
NUM_DOCS       = 100_000

EMBEDDING_DIM  = 640      # 13M:256 -> 27M:400 -> 85M:640
NUM_LAYERS     = 12       # 11 DSQG + 1 full causal
NUM_HEADS      = 8        # d_head = 80
FFN_DIM        = 2560     # 4 x EMBEDDING_DIM
INTERFERENCE   = 3        # pooling every 3rd DSQG layer (layers 2, 5, 8)

FULL_ATTN_LAYER = 11      # last layer -- maximum DSQG preprocessing depth
CHECKPOINT_DIR  = '2048_85m_condM_checkpoints'

# ---- condN offset set --------------------------------------------------------

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) |
                             set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == 44

# ---- DSQG Attention (condN, identical to 27M condM) -------------------------

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len

        if offsets is None:
            offsets = _COND_N_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets],
                                  dtype=torch.float32)
        self.pos_bias = nn.Parameter(
            -delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scale  = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))

        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)

        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        gathered_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(gathered_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
        }


# ---- Full Causal Attention (SDPA / FlashAttention) --------------------------

class FullCausalAttention(nn.Module):
    """
    Standard full causal attention, O(N^2).
    Uses torch SDPA with is_causal=True -- dispatches to FlashAttention on
    Hopper (H200) automatically. No RoPE; absolute position is carried by
    pos_embed and preserved through skip connections (critical for passkey).
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.qkv_proj      = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj      = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj     = nn.Linear(embedding_dim, embedding_dim, bias=True)
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
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return F.dropout(self.out_proj(out_flat * gate),
                         p=self.dropout_p, training=self.training)

    def attn_summary(self):
        return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0] * NUM_HEADS}


# ---- FFN ---------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ---- DSQG Block --------------------------------------------------------------

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


# ---- Full Attention Block ----------------------------------------------------

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


# ---- condM Transformer -------------------------------------------------------

class CondMTransformer(nn.Module):
    """
    (num_layers-1) DSQG blocks + 1 full causal attention block at full_attn_layer.
    Tied input/output embeddings. Absolute position embedding.
    No RoPE -- absolute positions required for passkey retrieval via skip connections.
    """
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
                    interference=(
                        i % interference_interval == interference_interval - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight   # tied
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


# ---- Data utilities (identical to condN / 27M condM) -------------------------

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_data(num_docs=NUM_DOCS):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {num_docs:,} docs)...')
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0: print(f'  {i+1:,} docs...')
    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
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
    for i in range(0, len(data) - batch_size, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size,  1:].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(ids[:, -MAX_SEQ_LEN:])
                logits_last = logits[0, -1].float()
                if temperature <= 0.01:
                    next_id = logits_last.argmax()
                else:
                    probs = F.softmax(logits_last / temperature, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    mask   = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum()
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
        out1 = model(x1).float()
        out2 = model(x2).float()
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


# ---- Training loop -----------------------------------------------------------

GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]


def train(model, train_data, val_data, test_data, tokenizer,
          save_dir=CHECKPOINT_DIR, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

    total_steps  = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    warmup_steps = max(50, int(total_steps * WARMUP_FRAC))

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
        eta_min=LR * ETA_MIN_FRAC)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps])

    scaler = torch.amp.GradScaler('cuda')

    print(f'\n  Total steps: {total_steps}  |  Warmup: {warmup_steps} steps')

    best_val_loss, best_val_ppl, best_epoch = float('inf'), float('inf'), 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1

            if step % 200 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'  Step {step}/{steps_per_epoch} | '
                      f'Loss {loss.item() * GRAD_ACCUM:.4f} | LR {current_lr:.2e}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best.pt'))

        # Full resumable checkpoint (model + optimizer + epoch metadata)
        torch.save({
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'epoch':        epoch,
            'val_ppl':      val_ppl,
            'best_val_ppl': best_val_ppl,
            'best_epoch':   best_epoch,
            'elapsed_s':    elapsed,
        }, os.path.join(save_dir, 'checkpoint_latest.pt'))

        # Numbered epoch copy for pod-termination safety
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f'epoch_{epoch:02d}.pt'))

        marker = ' * BEST' if is_best else ''
        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')

        ss = model.attn_summary()
        head_means  = ss['pos_bias_mean_per_head']
        most_local  = int(max(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        most_global = int(min(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} '
              f'most-local=h{most_local} most-global=h{most_global}')

        print('  -- Generation samples (greedy) --')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
            print(f'    {repr(prompt):35s} -> {repr(gen[:80])}')
        print('  --')
        sys.stdout.flush()

    # ---- Final evaluation on best checkpoint ---------------------------------

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condM 85M TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')

    print('\n  -- Temperature sweep (best checkpoint) --')
    sweep_results = {}
    for temp in [0.0, 0.5, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        print(f'\n  [{label}]')
        gens = generate(model, tokenizer, GEN_PROMPTS, device,
                        temperature=temp, top_p=0.9)
        sweep_results[label] = gens
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt):35s} -> {repr(gen[:80])}')

    ss = model.attn_summary()
    total_s = time.time() - t0

    print('\n' + '=' * 70)
    print('  85M condM ABLATION SUMMARY')
    print('=' * 70)
    print(f'  {"Standard 13M (reference)":<52} {"64.07":>8}')
    print(f'  {"Standard 27M":<52} {"50.683":>8}')
    print(f'  {"Standard 85M (condA baseline)":<52} {"57.70":>8}')
    print(f'  {"condM 13M":<52} {"54.529":>8}')
    print(f'  {"condM 27M":<52} {"44.500":>8}')
    print(f'  {"condM 85M (this run)":<52} {test_ppl:>8.3f}')
    delta_std = test_ppl - 57.70
    delta_27m = test_ppl - 50.683
    print(f'\n  vs standard 85M: {delta_std:+.3f} PPL')
    print(f'  vs standard 27M: {delta_27m:+.3f} PPL')
    print(f'  Total training time: {total_s/3600:.2f} h')

    return {
        'test_ppl':          test_ppl,
        'test_loss':         test_loss,
        'best_val_ppl':      best_val_ppl,
        'best_epoch':        best_epoch,
        'total_time_s':      total_s,
        'standard_85m_ppl':  57.70,
        'standard_27m_ppl':  50.683,
        'condm_13m_ppl':     54.529,
        'condm_27m_ppl':     44.5,
        'full_attn_layer':   FULL_ATTN_LAYER,
        'n_dsqg_layers':     NUM_LAYERS - 1,
        'architecture':      (f'condM 85M: {NUM_LAYERS-1}xDSQG + '
                              f'1xfull causal (layer {FULL_ATTN_LAYER})'),
        'temperature_sweep': sweep_results,
        'attn_summary':      ss,
    }


# ---- Main --------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('  DWARF 85M condM -- 11:1 Hybrid (11 DSQG + 1 Full Causal Attention)')
    print(f'  D={EMBEDDING_DIM}  H={NUM_HEADS}  d_head={EMBEDDING_DIM//NUM_HEADS}  '
          f'L={NUM_LAYERS}  FFN={FFN_DIM}  full_layer={FULL_ATTN_LAYER}')
    print('=' * 70)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU:    {props.name}  ({props.total_memory / 1e9:.1f} GB)')
        print(f'  CUDA:   {torch.version.cuda}')
        print(f'  PyTorch:{torch.__version__}')
        print(f'  Flash SDPA: {torch.backends.cuda.flash_sdp_enabled()}')

    os.makedirs('benchmarks/logs', exist_ok=True)

    splits = load_data(NUM_DOCS)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _tok_candidates = [
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
        '/workspace/DWARF/benchmarks/results/2048_condI_tokenizer.json',
    ]
    tok_path = next((p for p in _tok_candidates if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(
            'condI BPE tokenizer not found. Tried:\n' +
            '\n'.join(f'  {p}' for p in _tok_candidates))
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer: {tok_path}  ({tokenizer.vocab_size():,} tokens)')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
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

    n_params    = model.param_count()
    layer_types = ['FULL' if i == FULL_ATTN_LAYER else 'DSQG'
                   for i in range(NUM_LAYERS)]
    print(f'\ncondM 85M: {n_params:,} parameters')
    print(f'  Layers:      {layer_types}')
    print(f'  DSQG offsets: {len(_COND_N_OFFSETS)} (condN set: dense-32 + dyadic)')
    print(f'  Interference: every {INTERFERENCE}rd DSQG layer (layers 2, 5, 8)')
    print(f'  Batch:        B={BATCH_SIZE} GA={GRAD_ACCUM} (effective {BATCH_SIZE*GRAD_ACCUM})')
    print(f'  LR:           {LR}  warmup={WARMUP_FRAC*100:.0f}%  '
          f'eta_min={ETA_MIN_FRAC*100:.0f}% of LR')

    if not causality_check(model, device):
        print('CAUSALITY CHECK FAILED -- aborting')
        return

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir=CHECKPOINT_DIR, device=device)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, '2048_85m_condM_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {results_path}')
    print(f'  Best checkpoint -> {CHECKPOINT_DIR}/best.pt')


if __name__ == '__main__':
    main()
