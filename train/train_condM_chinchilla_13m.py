"""
DWARF condM — Chinchilla-Optimized 13.97M
H200 SXM Edition

Architecture: exact condM 13M (layer5) — no changes
  D=256, H=8, d_head=32, L=6, FFN=1024
  Layers 0-4: condN DSQG (44 offsets, interference at layer 2)
  Layer   5:  Full O(N^2) causal attention
  ~13.97M parameters (tied embeddings)

Chinchilla recipe
  14M params x 20 tok/param = ~280M unique tokens (minimum)
  400K OWT docs -> ~200K training sequences -> ~410M raw tokens
  Single epoch — zero data repetition
  Test PPL directly comparable to condM 13M (54.529) — same architecture

H200 SXM optimizations
  B=128, GA=1          — large batch saturates 141GB HBM3e tensor cores
  bfloat16             — H200 SXM native precision (faster than fp16 here)
  torch.compile        — Triton kernel fusion, reduce-overhead mode
  pin_memory + prefetch — overlap CPU->GPU transfer with compute
  LR=6e-4              — sqrt-scaled from B=32 baseline (3e-4 x sqrt(4) = 6e-4)
  5% warmup + cosine to 10% floor
  Tokenized data cache — skip retokenization on reruns (saves ~15 min)

Expected
  ~1,500 gradient steps | ~20-30 min on H200 SXM | ~$1-2
  Target PPL: 42-50 (vs condM 13M undertrained: 54.529)
  Answers: how much does data scaling improve the architecture?

Run on RunPod H200 SXM:
  pip install datasets tokenizers   # if needed
  cd /workspace/DWARF
  python3 -u benchmarks/train_condM_chinchilla_13m.py \
    2>&1 | tee benchmarks/logs/condM_chinchilla_13m_run.log

Scale via env vars:
  NUM_DOCS=800000 BATCH_SIZE=256 python3 -u benchmarks/train_condM_chinchilla_13m.py
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ---- Hyperparameters (override via env) --------------------------------------

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 1                       # Chinchilla: single pass, no repetition
NUM_DOCS      = int(os.environ.get('NUM_DOCS',    400_000))
BATCH_SIZE    = int(os.environ.get('BATCH_SIZE',  64))
GRAD_ACCUM    = 1                       # H200 SXM handles B=128 natively
LR            = float(os.environ.get('LR',        6e-4))   # sqrt-scaled for B=128
WARMUP_FRAC   = 0.05
LR_FLOOR_FRAC = 0.10
MAX_SEQ_LEN   = 2048

# ---- Architecture (identical to condM 13M layer5) ----------------------------

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8        # d_head = 32
FFN_DIM         = 1024    # 4 x D
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5        # last layer — maximum preprocessing depth
CHECKPOINT_DIR  = '2048_condM_chinchilla_13m_checkpoints'

# ---- condN offset set --------------------------------------------------------

_COND_N_OFFSETS = sorted(
    set(range(0, 33)) | {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536})
assert len(_COND_N_OFFSETS) == 44

# ---- DSQG Attention ----------------------------------------------------------

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.register_buffer('offsets', torch.tensor(_COND_N_OFFSETS, dtype=torch.long))
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in _COND_N_OFFSETS],
                                  dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD]
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)  # [B, H, N, HD]
        scale   = HD ** -0.5
        max_off = int(self.offsets[-1].item())         # offsets sorted ascending
        # Pad once at the front: [B, H, max_off+N, HD]
        k_pad   = F.pad(k, (0, 0, max_off, 0))
        v_pad   = F.pad(v, (0, 0, max_off, 0))
        # Build all gather indices in one shot: [num_offsets, N]
        # gather_idx[i,n] = max_off - offsets[i] + n  →  k_pad position for token n at lag offsets[i]
        n_idx      = torch.arange(N, device=x.device)                          # [N]
        gather_idx = max_off - self.offsets.unsqueeze(1) + n_idx.unsqueeze(0)  # [44, N]
        # Single advanced-index gather (1 CUDA kernel, replaces 44-iter loop)
        K_all = k_pad[:, :, gather_idx, :].permute(0, 1, 3, 2, 4).contiguous()  # [B,H,N,44,HD]
        V_all = v_pad[:, :, gather_idx, :].permute(0, 1, 3, 2, 4).contiguous()  # [B,H,N,44,HD]
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale  # [B, H, N, 44]
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        pos_idx = torch.arange(N, device=x.device).unsqueeze(1)               # [N, 1]
        scores  = scores.masked_fill(
            (pos_idx < self.offsets.unsqueeze(0)).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)  # [B, H, N, HD]
        flat  = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.dropout(self.out_proj(flat * torch.sigmoid(self.gate_proj(x))))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
        return {'pos_bias_abs_mean': pb.abs().mean().item(),
                'pos_bias_abs_max':  pb.abs().max().item(),
                'pos_bias_mean_per_head': pb.mean(0).tolist()}


class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
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
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(self.out_proj(flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


class FFN(nn.Module):
    def __init__(self, d, f, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, f); self.fc2 = nn.Linear(f, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, d, h, f, seq_len, dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = DSQGAttentionN(d, h, seq_len, dropout)
        self.ffn   = FFN(d, f, dropout)
        if interference:
            self.inter_norm = nn.LayerNorm(d)
            self.inter_gate = nn.Linear(d, d)
            self.inter_pool = nn.Linear(d, d)

    def _attn_fn(self, x): return self.attn(self.norm1(x))

    def forward(self, x):
        x = x + self._attn_fn(x)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N+1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
            pool = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        return x + self.ffn(self.norm2(x))


class FullAttentionBlock(nn.Module):
    def __init__(self, d, h, f, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = FullCausalAttention(d, h, dropout)
        self.ffn   = FFN(d, f, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class CondMTransformer(nn.Module):
    """condM 13M — identical architecture to the undertrained ablation run."""
    def __init__(self, vocab_size, D, L, H, FFN_dim, seq_len,
                 full_attn_layer=5, interference_interval=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.pos_embed = nn.Embedding(seq_len + 2, D)
        self.drop      = nn.Dropout(dropout)
        blocks = []
        for i in range(L):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(D, H, FFN_dim, dropout))
            else:
                inter = (i % interference_interval == interference_interval - 1
                         and i != full_attn_layer)
                blocks.append(DSQGBlock(D, H, FFN_dim, seq_len, dropout, inter))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(D)
        self.out    = nn.Linear(D, vocab_size, bias=False)
        self.out.weight = self.embedding.weight   # tied embeddings
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
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        sums = [b.attn.attn_summary() for b in dsqg]
        n    = len(sums)
        return {'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean'] for s in sums) / n,
                'pos_bias_abs_max':       max(s['pos_bias_abs_max']  for s in sums),
                'pos_bias_mean_per_head': [sum(s['pos_bias_mean_per_head'][h] for s in sums)/n
                                          for h in range(NUM_HEADS)]}


# ---- Data utilities ----------------------------------------------------------

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def _cache_path(num_docs):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f'.chinchilla_cache_{num_docs}docs.pt')


def load_and_tokenize(tokenizer, num_docs):
    cache = _cache_path(num_docs)
    if os.path.exists(cache):
        print(f'  Loading tokenized cache ({os.path.getsize(cache)/1e9:.2f} GB): {cache}')
        return torch.load(cache, weights_only=True)

    from datasets import load_dataset
    print(f'  Streaming {num_docs:,} OWT docs (first time — will cache to disk)...')
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    t0    = time.time()
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 50_000 == 0:
            print(f'    {i+1:,} docs | {time.time()-t0:.0f}s')

    n       = len(texts)
    n_train = int(n * 0.950)
    n_val   = int(n * 0.025)
    splits  = {'train': texts[:n_train],
               'val':   texts[n_train:n_train + n_val],
               'test':  texts[n_train + n_val:]}

    def encode_split(split_texts, name):
        tokens = []
        for t in split_texts:
            tokens.extend(tokenizer.encode(t)); tokens.append(3)
        trunc = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
        seqs  = torch.tensor(tokens[:trunc], dtype=torch.int32).view(-1, MAX_SEQ_LEN)
        print(f'    {name}: {len(seqs):,} sequences '
              f'({len(seqs)*MAX_SEQ_LEN/1e6:.0f}M tokens)')
        return seqs

    print('  Tokenizing...')
    data = {k: encode_split(v, k) for k, v in splits.items()}
    torch.save(data, cache)
    print(f'  Cached -> {cache}  ({os.path.getsize(cache)/1e9:.2f} GB)')
    return data


# ---- Evaluation / generation -------------------------------------------------

@torch.no_grad()
def evaluate(model, data, device, dtype):
    model.eval()
    total_loss = total_tokens = 0
    bs = min(BATCH_SIZE, 64)   # smaller eval batch — no need for full size
    for i in range(0, len(data) - bs, bs):
        x = data[i:i+bs, :-1].long().to(device)
        y = data[i:i+bs,  1:].long().to(device)
        with torch.amp.autocast('cuda', dtype=dtype):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


GEN_PROMPTS = ['It was a dark and stormy', 'The length of the hypotenuse',
               'The President of the United', 'Once upon a time there was',
               'The results indicate that']

def generate(model, tokenizer, device, dtype, temperature=0.7, max_new=100):
    model.eval(); results = []
    for prompt in GEN_PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits = model(ids[:, -MAX_SEQ_LEN:])
                last = logits[0, -1].float()
                nxt  = (last.argmax() if temperature <= 0.01
                        else torch.multinomial(F.softmax(last/temperature, dim=-1), 1))
                ids  = torch.cat([ids, nxt.view(1, 1)], dim=1)
        results.append(tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())[:120])
    return results


def causality_check(model, device, dtype):
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        with torch.amp.autocast('cuda', dtype=dtype):
            d  = (model(x1).float() - model(x2).float()).abs()
    ok = d[0, :10].max().item() < 1e-5
    print(f'  Causality: {"PASS" if ok else "FAIL!"}  '
          f'pre-10 max={d[0,:10].max():.8f}  pos-10 max={d[0,10:].max():.4f}')
    return ok


# ---- Training loop -----------------------------------------------------------

def train(model, data, tokenizer, save_dir, device, dtype):
    os.makedirs(save_dir, exist_ok=True)
    train_data = data['train']
    val_data   = data['val']
    test_data  = data['test']

    total_steps  = math.ceil(len(train_data) / BATCH_SIZE)
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    lr_min       = LR * LR_FLOOR_FRAC

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (lr_min + (LR - lr_min) * cos_val) / LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # BF16 on H200 SXM: GradScaler not needed (bf16 doesn't underflow like fp16)
    scaler    = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    log_every = max(1, total_steps // 20)
    t0        = time.time()

    print(f'\n  Total steps:    {total_steps}')
    print(f'  Warmup steps:   {warmup_steps}')
    print(f'  Unique tokens:  {total_steps * BATCH_SIZE * MAX_SEQ_LEN / 1e6:.0f}M '
          f'({total_steps * BATCH_SIZE * MAX_SEQ_LEN / model.param_count():.1f} tok/param)\n')

    model.train()
    indices      = torch.randperm(len(train_data))
    running_loss = 0.0
    optimizer.zero_grad()

    for step in range(total_steps):
        end = min((step + 1) * BATCH_SIZE, len(train_data))
        batch = train_data[indices[step * BATCH_SIZE:end]]
        if len(batch) < 2: continue
        x, y  = batch[:, :-1].long().to(device), batch[:, 1:].long().to(device)

        with torch.amp.autocast('cuda', dtype=dtype):
            loss = F.cross_entropy(
                model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item()
        if (step + 1) % log_every == 0 or step == total_steps - 1:
            avg   = running_loss / log_every; running_loss = 0.0
            tps   = (step + 1) * BATCH_SIZE * MAX_SEQ_LEN / (time.time() - t0)
            lr_now = scheduler.get_last_lr()[0]
            print(f'  Step {step+1:>5}/{total_steps} | Loss {avg:.4f} '
                  f'| LR {lr_now:.2e} | {tps/1e6:.2f}M tok/s '
                  f'| {time.time()-t0:.0f}s elapsed')
        sys.stdout.flush()

    # ---- Final eval ----------------------------------------------------------
    val_loss  = evaluate(model, val_data,  device, dtype)
    test_loss = evaluate(model, test_data, device, dtype)
    val_ppl   = math.exp(min(val_loss,  20))
    test_ppl  = math.exp(min(test_loss, 20))
    elapsed   = time.time() - t0

    torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
    print(f'\n  Val PPL: {val_ppl:.3f}')

    ss = model.attn_summary()
    hm = ss['pos_bias_mean_per_head']
    print(f'  pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
          f'|max|={ss["pos_bias_abs_max"]:.4f} '
          f'most-local=h{max(range(NUM_HEADS), key=lambda h: abs(hm[h]))} '
          f'most-global=h{min(range(NUM_HEADS), key=lambda h: abs(hm[h]))}')

    print(f'\n  -- Generation (T=0.7) --')
    for p, g in zip(GEN_PROMPTS, generate(model, tokenizer, device, dtype)):
        print(f'  {repr(p):35s} -> {repr(g[:100])}')

    print(f'\n  {"="*62}')
    print(f'  RESULTS — condM 13M: Undertrained vs Chinchilla')
    print(f'  {"="*62}')
    print(f'  {"Model":<44} {"Test PPL":>8}')
    print(f'  {"-"*53}')
    print(f'  {"Standard 13M (100K docs x10 epochs)":<44} {"64.07":>8}')
    print(f'  {"condM 13M (100K docs x10 epochs)":<44} {"54.529":>8}')
    print(f'  {"condM 27M (100K docs x10 epochs)":<44} {"44.500":>8}')
    print(f'  {"condM 13M CHINCHILLA (this run)":<44} {test_ppl:>8.3f}  <- NEW')
    print(f'  {"-"*53}')
    delta = test_ppl - 54.529
    print(f'  vs undertrained condM 13M: {delta:+.3f} PPL '
          f'({"improvement" if delta < 0 else "regression"})')
    unique_tokens = total_steps * BATCH_SIZE * MAX_SEQ_LEN
    print(f'\n  Total time:    {elapsed:.0f}s ({elapsed/3600:.2f}h)')
    print(f'  Unique tokens: {unique_tokens/1e9:.3f}B')
    print(f'  Tok/param:     {unique_tokens/model.param_count():.1f}')
    print(f'  {"="*62}')

    return {'test_ppl': test_ppl, 'val_ppl': val_ppl,
            'total_time_s': elapsed, 'num_docs': NUM_DOCS,
            'batch_size': BATCH_SIZE, 'total_steps': total_steps,
            'n_params': model.param_count(),
            'unique_tokens': unique_tokens,
            'tok_per_param': unique_tokens / model.param_count(),
            'reference': {'condM_13m_undertrained': 54.529,
                          'standard_13m_undertrained': 64.07,
                          'condM_27m_undertrained': 44.5}}


# ---- Main --------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # H200 SXM: sm_90 supports bfloat16 natively and faster than fp16
    if torch.cuda.is_available():
        cap   = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    print('=' * 65)
    print('  condM 13.97M — Chinchilla-Optimized | H200 SXM Edition')
    print(f'  D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} '
          f'FFN={FFN_DIM} full_attn=layer{FULL_ATTN_LAYER}')
    print(f'  Docs: {NUM_DOCS:,} | Epochs: {NUM_EPOCHS} | '
          f'B={BATCH_SIZE} GA={GRAD_ACCUM} | LR={LR:.1e}')
    print(f'  dtype: {dtype} | device: {device}')
    print('=' * 65)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {props.name}  ({props.total_memory/1e9:.0f}GB)')

    os.makedirs('benchmarks/logs', exist_ok=True)

    # Tokenizer
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = next((p for p in [
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ] if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI tokenizer not found — '
                                'check benchmarks/results/2048_condI_tokenizer.json')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'  Tokenizer: {os.path.basename(tok_path)}  (vocab={VOCAB_SIZE})')

    # Data
    data = load_and_tokenize(tokenizer, NUM_DOCS)
    print(f'  Train: {len(data["train"]):,} seqs | '
          f'Val: {len(data["val"]):,} | Test: {len(data["test"]):,}')

    # Model
    model = CondMTransformer(
        vocab_size=tokenizer.vocab_size(), D=EMBEDDING_DIM, L=NUM_LAYERS,
        H=NUM_HEADS, FFN_dim=FFN_DIM, seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER, interference_interval=INTERFERENCE,
    ).to(device)

    layer_types  = ['FULL' if i == FULL_ATTN_LAYER else 'DSQG'
                    for i in range(NUM_LAYERS)]
    inter_layers = [i for i in range(NUM_LAYERS)
                    if i != FULL_ATTN_LAYER
                    and i % INTERFERENCE == INTERFERENCE - 1]
    n_params = model.param_count()
    print(f'\n  Params: {n_params:,}  '
          f'(ref condM 13M: 13,984,480 — '
          f'{"MATCH" if abs(n_params - 13_984_480) < 10_000 else "CHECK"})')
    print(f'  Layers: {layer_types}')
    print(f'  Interference at: {inter_layers}')
    print(f'  Chinchilla tokens/param target: 20 -> '
          f'{n_params * 20 / 1e6:.0f}M tokens minimum')

    if not causality_check(model, device, dtype):
        print('Causality FAILED — aborting'); return

    print('  Running eager (no torch.compile — fast on H200 SXM for small models)')

    results = train(model, data, tokenizer, CHECKPOINT_DIR, device, dtype)

    out_path = os.path.join(_script_dir, 'condM_chinchilla_13m_results.json')
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f'\n  Results -> {out_path}')


if __name__ == '__main__':
    main()
