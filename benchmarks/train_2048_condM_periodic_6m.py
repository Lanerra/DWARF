"""
DWARF condM-periodic 6.84M — 3:1 Hybrid Proof of Concept

Architecture: [DSQG, DSQG, DSQG, Full]
  D=160, H=8, d_head=20, L=4, FFN=640
  Layers 0-2: condN DSQG (44 offsets, interference at layer 2)
  Layer   3: Full O(N^2) causal attention
  ~6.84M parameters

Purpose
-------
  Quick 3:1 periodic hybrid proof-of-concept at half the condM 13M scale.
  Answers: does a single DSQG preprocessing block before full attention
  already match condM 13M (5 DSQG blocks), at half the parameters?

  condM 13M (layer5): D=256, L=6 → 13.97M params, test PPL 54.529
  condM 13M (layer3): D=256, L=6 → 13.97M params, test PPL 54.480
  condM-periodic 6.84M (this): D=160, L=4 → 6.84M params, test PPL ?

  If this beats condM 13M at half params: the 3:1 pattern is highly efficient.
  Expected: probably ~58-64 PPL (better than condN, competitive with condM at scale).

Run on local RTX 4090:
  CUDA_VISIBLE_DEVICES=0 python3 -u benchmarks/train_2048_condM_periodic_6m.py \\
    2>&1 | tee benchmarks/logs/condM_periodic_6m_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ---- Hyperparameters ---------------------------------------------------------

VOCAB_SIZE     = 32000
NUM_EPOCHS     = 10
BATCH_SIZE     = 8
GRAD_ACCUM     = 4        # effective batch = 32 (same as 13M condM runs)
LR             = 3e-4
MAX_SEQ_LEN    = 2048
NUM_DOCS       = 100_000

EMBEDDING_DIM  = 160
NUM_LAYERS     = 4        # [DSQG, DSQG, DSQG, Full]
NUM_HEADS      = 8        # d_head = 20
FFN_DIM        = 640      # 4 x D
INTERFERENCE   = 3        # pooling at DSQG layer 2 (i%3==2, i not in full_attn_layers)

FULL_ATTN_LAYERS = {3}    # single full attention at layer 3 (last)
CHECKPOINT_DIR   = '2048_condM_periodic_6m_checkpoints'

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
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scale  = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k)); V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta]], dim=2))
        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        scores = scores.masked_fill(
            (n_idx < self.offsets.unsqueeze(0)).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha  = F.softmax(scores, dim=-1)
        out    = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        flat   = out.permute(0, 2, 1, 3).reshape(B, N, D)
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
        H, HD = self.num_heads, self.head_dim
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

    def attn_summary(self):
        return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0] * NUM_HEADS}


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
        x = x + torch.utils.checkpoint.checkpoint(
            self._attn_fn, x, use_reentrant=False)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
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


class CondMPeriodicTransformer(nn.Module):
    def __init__(self, vocab_size, D, L, H, FFN_dim, seq_len,
                 full_attn_layers=None, interference_interval=3, dropout=0.1):
        super().__init__()
        if full_attn_layers is None:
            full_attn_layers = {L - 1}
        self.embedding = nn.Embedding(vocab_size, D)
        self.pos_embed = nn.Embedding(seq_len + 2, D)
        self.drop      = nn.Dropout(dropout)
        blocks = []
        for i in range(L):
            if i in full_attn_layers:
                blocks.append(FullAttentionBlock(D, H, FFN_dim, dropout))
            else:
                blocks.append(DSQGBlock(D, H, FFN_dim, seq_len, dropout,
                    interference=(i % interference_interval == interference_interval - 1
                                  and i not in full_attn_layers)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(D)
        self.out    = nn.Linear(D, vocab_size, bias=False)
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
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        if not dsqg: return {'pos_bias_abs_mean':0.,'pos_bias_abs_max':0.,'pos_bias_mean_per_head':[0.]*NUM_HEADS}
        sums = [b.attn.attn_summary() for b in dsqg]
        n = len(sums)
        return {
            'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean'] for s in sums) / n,
            'pos_bias_abs_max':       max(s['pos_bias_abs_max']  for s in sums),
            'pos_bias_mean_per_head': [sum(s['pos_bias_mean_per_head'][h] for s in sums)/n
                                       for h in range(NUM_HEADS)],
        }


# ---- Data utilities ----------------------------------------------------------

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()

def load_data():
    from datasets import load_dataset
    print(f'Loading OpenWebText ({NUM_DOCS:,} docs)...')
    ds = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= NUM_DOCS: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0: print(f'  {i+1:,}...')
    n = len(texts)
    return {'train': texts[:int(n*.95)],
            'val':   texts[int(n*.95):int(n*.95)+2500],
            'test':  texts[int(n*.95)+2500:int(n*.95)+5000]}

def encode_split(split_texts, tokenizer, split_name):
    tokens = []
    for t in split_texts: tokens.extend(tokenizer.encode(t)); tokens.append(3)
    n = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    seqs = torch.tensor(tokens[:n], dtype=torch.long).view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs

@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device)
        y = data[i:i+BATCH_SIZE,  1:].to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * y.numel(); total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)

def generate(model, tokenizer, prompts, device, temperature=0.0):
    model.eval(); results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(80):
                with torch.amp.autocast('cuda'):
                    logits = model(ids[:, -MAX_SEQ_LEN:])
                logits_last = logits[0, -1].float()
                next_id = logits_last.argmax() if temperature <= 0.01 else \
                    torch.multinomial(F.softmax(logits_last / temperature, dim=-1), 1)
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        results.append(tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())[:100])
    return results

def causality_check(model, device):
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        diff = (model(x1).float() - model(x2).float()).abs()
    ok = diff[0, :10].max().item() < 1e-6
    print(f'  Causality: {"PASS" if ok else "FAIL"}  (pre-10 max: {diff[0,:10].max():.8f})')
    return ok


# ---- Training loop -----------------------------------------------------------

GEN_PROMPTS = ['It was a dark and stormy', 'The length of the hypotenuse',
               'The President of the United', 'Once upon a time there was',
               'The results indicate that']

def train(model, train_data, val_data, test_data, tokenizer, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = torch.amp.GradScaler('cuda')

    best_val_loss, best_val_ppl, best_epoch = float('inf'), float('inf'), 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0; optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda'):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1
            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item()*GRAD_ACCUM:.4f}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0
        is_best    = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

        marker = ' * BEST' if is_best else ''
        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')
        ss = model.attn_summary()
        hm = ss['pos_bias_mean_per_head']
        print(f'  pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'most-local=h{max(range(NUM_HEADS),key=lambda h:abs(hm[h]))} '
              f'most-global=h{min(range(NUM_HEADS),key=lambda h:abs(hm[h]))}')
        print('  -- greedy --')
        for p, g in zip(GEN_PROMPTS, generate(model, tokenizer, GEN_PROMPTS, device)):
            print(f'    {repr(p):35s} -> {repr(g[:75])}')
        sys.stdout.flush()

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condM-periodic 6M TEST: PPL {test_ppl:.3f}')
    print(f'  Reference — condM 13M (layer5): 54.529')
    print(f'  Reference — condM 13M (layer3): 54.480')
    print(f'  Params: {model.param_count():,} vs condM 13M: 13,984,480')
    return {'test_ppl': test_ppl, 'best_val_ppl': best_val_ppl,
            'best_epoch': best_epoch, 'total_time_s': time.time() - t0,
            'architecture': '[DSQG×3, Full]',
            'full_attn_layers': sorted(FULL_ATTN_LAYERS),
            'n_params': model.param_count()}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 65)
    print('  condM-periodic 6.84M -- [DSQG, DSQG, DSQG, Full]')
    print(f'  D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} FFN={FFN_DIM}')
    print(f'  Full attention at layers: {sorted(FULL_ATTN_LAYERS)}')
    print('=' * 65)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {props.name} ({props.total_memory/1e9:.1f} GB)')

    os.makedirs('benchmarks/logs', exist_ok=True)
    splits    = load_data()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path  = next((p for p in [
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ] if os.path.exists(p)), None)
    if tok_path is None: raise FileNotFoundError('condI tokenizer not found')
    from tokenizers import Tokenizer
    tokenizer  = BPETokenizerWrapper(Tokenizer.from_file(tok_path))

    train_data = encode_split(splits['train'], tokenizer, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, 'Test')

    model = CondMPeriodicTransformer(
        vocab_size=tokenizer.vocab_size(), D=EMBEDDING_DIM, L=NUM_LAYERS,
        H=NUM_HEADS, FFN_dim=FFN_DIM, seq_len=MAX_SEQ_LEN,
        full_attn_layers=FULL_ATTN_LAYERS, interference_interval=INTERFERENCE,
    ).to(device)

    layer_types = ['FULL' if i in FULL_ATTN_LAYERS else 'DSQG' for i in range(NUM_LAYERS)]
    print(f'\n  Params: {model.param_count():,}')
    print(f'  Layers: {layer_types}')
    inter_layers = [i for i in range(NUM_LAYERS) if i not in FULL_ATTN_LAYERS
                    and i % INTERFERENCE == INTERFERENCE - 1]
    print(f'  Interference at: {inter_layers}')

    if not causality_check(model, device): return

    results = train(model, train_data, val_data, test_data, tokenizer,
                    CHECKPOINT_DIR, device)
    out_path = os.path.join(_script_dir, 'condM_periodic_6m_results.json')
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f'  Results -> {out_path}')

if __name__ == '__main__':
    main()
