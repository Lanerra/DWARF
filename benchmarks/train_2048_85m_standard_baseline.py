"""
Standard Transformer Baseline — 85M Scale

Comparison target for condM 85M scaling validation.

ARCHITECTURE
------------
Pre-norm causal transformer, same D/H/L/FFN as condM 85M:
  D=640, H=8, d_head=80, L=12, FFN=2560
  Standard causal self-attention (full N×N softmax, O(N²))
  Uses F.scaled_dot_product_attention (Flash Attention on H100/H200)
  Untied out_proj — consistent with 13M and 27M standard baselines

PARAMETER COUNTS
  condM 85M:       88,267,552  (tied embeddings + DSQG layers)
  Standard 85M:  ~101,361,920  (untied embeddings, same D/H/L/FFN)

The untied/tied difference is consistent with all prior baselines.
condM wins with fewer parameters — that's the result.

HYPERPARAMETERS
  Match condM 85M exactly: B=64, GA=1, LR=3e-4, warmup 5%, cosine to 10%.
  Same data, same tokenizer, same epochs.

RunPod (H100 or H200 SXM pod)
------------------------------
  git clone https://github.com/Lanerra/wave-field-llm /workspace/DWARF
  cd /workspace/DWARF
  pip install tokenizers datasets
  mkdir -p benchmarks/logs
  tmux new-session -d -s std85m -x 220 -y 50
  tmux send-keys -t std85m \
    "python3 -u benchmarks/train_2048_85m_standard_baseline.py \
    2>&1 | tee benchmarks/logs/85m_standard_run.log" Enter

Monitor:
  tmux attach -t std85m
  tail -f benchmarks/logs/85m_standard_run.log

Pull results when done:
  rsync -avz root@<pod-ip>:/workspace/DWARF/benchmarks/logs/85m_standard_run.log ./benchmarks/logs/
  rsync -avz root@<pod-ip>:/workspace/DWARF/2048_85m_standard_checkpoints/best.pt \
        ./checkpoints/2048_85m_standard_checkpoints/best.pt

Expected runtime: ~5-7h on H100 SXM5, ~3-4h on H200 SXM.
Memory: B=64 fits comfortably on both. Reduce to B=32 + GA=2 if OOM.
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')   # TF32 on Hopper

# ── Hyperparameters ───────────────────────────────────────────────────────────

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 10
BATCH_SIZE    = 64       # H200: comfortable. H100: fits; reduce to 32+GA=2 if OOM.
GRAD_ACCUM    = 1
LR            = 3e-4
WARMUP_FRAC   = 0.05     # linear warmup over first 5% of steps
ETA_MIN_FRAC  = 0.10     # cosine floor = 10% of peak LR
MAX_SEQ_LEN   = 2048
NUM_DOCS      = 100_000

# Same dimensions as condM 85M — only the attention mechanism differs
EMBEDDING_DIM = 640
NUM_LAYERS    = 12
NUM_HEADS     = 8        # d_head = 80
FFN_DIM       = 2560     # 4 × EMBEDDING_DIM

CHECKPOINT_DIR = '2048_85m_standard_checkpoints'

GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]


# ── Standard Causal Self-Attention ────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.dropout   = dropout
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).transpose(1, 2)
        k = k.view(B, N, H, HD).transpose(1, 2)
        v = v.view(B, N, H, HD).transpose(1, 2)
        # is_causal=True → Flash Attention causal mask, no explicit N×N matrix
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


# ── FFN ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ── Transformer Block (pre-norm) ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.ln2  = nn.LayerNorm(embedding_dim)
        self.ffn  = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Model ─────────────────────────────────────────────────────────────────────

class StandardTransformer85M(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(seq_len, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final  = nn.LayerNorm(embedding_dim)
        self.out_proj   = nn.Linear(embedding_dim, vocab_size, bias=False)  # untied
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.ln_final(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(num_docs=NUM_DOCS):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {num_docs:,} docs)...')
    ds, texts = load_dataset('openwebtext', split='train', streaming=True), []
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0:
            print(f'  {i+1:,} docs...')
    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
    }


def encode_split(split_texts, tokenizer, max_seq_len, name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // max_seq_len) * max_seq_len
    seqs = torch.tensor(tokens[:n], dtype=torch.long).view(-1, max_seq_len)
    print(f'  {name}: {len(seqs):,} sequences')
    return seqs


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss = total_tokens = 0
    for i in range(0, len(data) - batch_size, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size,  1:].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=80, temperature=0.0):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(ids[:, -MAX_SEQ_LEN:])
                logits_last = logits[0, -1].float()
                if temperature <= 0.01:
                    next_id = logits_last.argmax()
                else:
                    probs   = F.softmax(logits_last / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)[0]
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        gen = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:100])
    return results


def get_lr(step, total_steps, warmup_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_data, val_data, test_data, tokenizer,
          save_dir=CHECKPOINT_DIR, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    decay     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.1},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=LR, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    total_steps     = NUM_EPOCHS * steps_per_epoch
    warmup_steps    = max(1, int(WARMUP_FRAC * total_steps))
    lr_min          = LR * ETA_MIN_FRAC
    scaler          = torch.amp.GradScaler('cuda')

    best_val_loss = best_val_ppl = float('inf')
    best_epoch    = 0
    epoch_log     = []
    global_step   = 0
    t0            = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices   = torch.randperm(len(train_data))
        last_loss = 0.0
        optimizer.zero_grad()

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                i0    = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if i0 >= len(train_data): continue
                batch = train_data[indices[i0: i0 + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
                last_loss = loss.item() * GRAD_ACCUM

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            global_step += 1
            lr_now = get_lr(global_step, total_steps, warmup_steps, LR, lr_min)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

            if (acc_step + 1) % 200 == 0:
                print(f'  Step {acc_step+1}/{steps_per_epoch} | '
                      f'Loss {last_loss:.4f} | LR {lr_now:.2e}')

        val_loss = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl  = math.exp(min(val_loss, 20))
        elapsed  = time.time() - t0
        marker   = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {last_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')
        epoch_log.append({'epoch': epoch, 'train_loss': round(last_loss, 4),
                          'val_loss': round(val_loss, 4), 'val_ppl': round(val_ppl, 2),
                          'elapsed_s': round(elapsed, 1)})

        print('  ── Generation samples (greedy) ──')
        for p, g in zip(GEN_PROMPTS,
                        generate(model, tokenizer, GEN_PROMPTS, device, temperature=0.0)):
            print(f"    '{p}' → '{g[:80]}'")
        print('  ──')
        sys.stdout.flush()

    # ── Test eval ─────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n── Standard 85M TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f} ──')

    # ── Temperature sweep ─────────────────────────────────────────────────────
    print('\n── Temperature sweep (best checkpoint) ──')
    sweep = {}
    for label, temp in [('greedy', 0.0), ('T=0.5', 0.5), ('T=0.7', 0.7), ('T=1.0', 1.0)]:
        gens = generate(model, tokenizer, GEN_PROMPTS, device, temperature=temp)
        sweep[label] = gens
        print(f'\n[{label}]')
        for p, g in zip(GEN_PROMPTS, gens):
            print(f"  '{p}' → '{g[:80]}'")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_params = model.param_count()
    print('\n' + '=' * 72)
    print('  condM Scaling Series vs Standard Transformer')
    print('=' * 72)
    rows = [
        ('Standard 13M',           '21.6M',  64.073),
        ('condM 13M',              '13.98M', 54.529),
        ('Standard 27M',           '37.97M', 50.683),
        ('condM 27M',              '26.46M', 44.500),
        (f'Standard 85M (this)',   f'{n_params/1e6:.1f}M', test_ppl),
        ('condM 85M',              '88.3M',  36.042),
    ]
    print(f'  {"Model":<36} {"Params":>8}  {"PPL":>8}')
    print(f'  {"─"*36}  {"─"*8}  {"─"*8}')
    for name, params, ppl in rows:
        print(f'  {name:<36} {params:>8}  {ppl:>8.3f}')
    condm_gap = 36.042 - test_ppl
    param_pct  = (1 - 88.3 / (n_params / 1e6)) * 100
    print(f'\n  85M gap: condM vs standard = {condm_gap:+.3f} PPL '
          f'({param_pct:.0f}% fewer params for condM)')
    print('=' * 72)

    total_time = time.time() - t0
    return {
        'model':          'Standard Transformer 85M baseline',
        'test_ppl':       round(test_ppl, 4),
        'test_loss':      round(test_loss, 4),
        'best_val_ppl':   round(best_val_ppl, 4),
        'best_val_loss':  round(best_val_loss, 4),
        'best_epoch':     best_epoch,
        'n_params':       n_params,
        'total_time_s':   round(total_time, 1),
        'epoch_log':      epoch_log,
        'architecture': {
            'type':            'pre-norm causal transformer',
            'attention':       'standard causal SDPA (O(N^2))',
            'vocab_size':      VOCAB_SIZE,
            'embedding_dim':   EMBEDDING_DIM,
            'num_layers':      NUM_LAYERS,
            'num_heads':       NUM_HEADS,
            'd_head':          EMBEDDING_DIM // NUM_HEADS,
            'ffn_dim':         FFN_DIM,
            'max_seq_len':     MAX_SEQ_LEN,
            'tied_embeddings': False,
            'activation':      'gelu',
            'normalization':   'pre-norm LayerNorm',
        },
        'training': {
            'optimizer':        'AdamW', 'lr': LR, 'weight_decay': 0.1,
            'betas':            [0.9, 0.95], 'batch_size': BATCH_SIZE,
            'grad_accum':       GRAD_ACCUM, 'effective_batch': BATCH_SIZE * GRAD_ACCUM,
            'lr_schedule':      f'linear warmup {WARMUP_FRAC*100:.0f}% + cosine to {ETA_MIN_FRAC*100:.0f}%',
            'grad_clip':        1.0, 'epochs': NUM_EPOCHS,
            'tokenizer':        'condI BPE 32K',
            'dataset':          'OpenWebText 100K docs',
            'precision':        'bf16 AMP',
        },
        'comparison': {
            'condM_85m_ppl':    36.042,
            'condM_85m_params': 88_267_552,
            'gap_ppl':          round(36.042 - test_ppl, 4),
        },
        'temperature_sweep': sweep,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 72)
    print('  Standard Transformer Baseline — 85M Scale')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, d_head={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  F.scaled_dot_product_attention (Flash Attention on H100/H200)')
    print(f'  Untied out_proj — consistent with 13M and 27M baselines')
    print('=' * 72)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f'  GPU: {p.name}  ({p.total_memory/1e9:.1f} GB)')

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_root   = os.path.dirname(script_dir)
    ckpt_dir    = os.path.join(repo_root, CHECKPOINT_DIR)
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    tok_path = os.path.join(script_dir, 'results', '2048_condI_tokenizer.json')
    if not os.path.exists(tok_path):
        tok_path = os.path.join(script_dir, '2048_condI_tokenizer.json')
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f'condI tokenizer not found. Expected: {tok_path}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded condI BPE tokenizer from {tok_path}')

    splits     = load_data(NUM_DOCS)
    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model    = StandardTransformer85M().to(device)
    n_params = model.param_count()
    print(f'\nStandard 85M baseline: {n_params:,} parameters ({n_params/1e6:.2f}M)')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, d_head={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  vs condM 85M: 88,267,552 params (tied emb + DSQG layers)')
    print(f'  Δparams: {n_params - 88_267_552:+,} ({(n_params/88_267_552-1)*100:+.1f}%)')

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir=ckpt_dir, device=device)

    out = os.path.join(results_dir, '85m_standard_baseline_results.json')
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\nResults → {out}')


if __name__ == '__main__':
    main()
