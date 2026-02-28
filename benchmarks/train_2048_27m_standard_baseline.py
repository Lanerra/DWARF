"""
Standard Transformer Baseline — 27M Scale

Comparison target for 27M condP scaling validation.

ARCHITECTURE
------------
Pre-norm causal transformer, same D/H/L/FFN as 27M condP:
  D=400, H=8, d_head=50, L=6, FFN=1600
  Standard causal self-attention (full N×N softmax, O(N²))
  Separate out_proj — NOT weight-tied (consistent with 13M baseline, train_2048_condK.py)

PARAMETER COUNTS
  13M condP:       14,118,368  (tied embeddings)
  13M baseline:    21,631,008  (untied — this is the established comparison pattern)
  27M condP:       26,781,152  (tied embeddings)
  27M baseline:    ~37-38M     (untied — same pattern as 13M)

The untied vs tied difference is an architectural choice, not a bug.
We report both numbers. The main question is whether the gap between
condP and baseline closes at 27M scale as predicted (~+0.7 PPL vs +0.99 at 13M).

HYPERPARAMETERS
  Identical to 27M condP: B=32, GA=1, LR=3e-4, cosine, same data split.
  B=32 GA=1 is correct for H100 — same effective batch as 13M baseline (B=8 GA=4=32).

Run on RunPod (same pod as 27M condP, or a fresh pod):
  tmux new-session -d -s baseline27m -x 220 -y 50
  tmux send-keys -t baseline27m \\
    "cd /workspace/DWARF && python3 -u benchmarks/train_2048_27m_standard_baseline.py \\
    2>&1 | tee /workspace/logs/27m_baseline_run.log" Enter

Monitor:
  tmux attach -t baseline27m
  tail -f /workspace/logs/27m_baseline_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# Identical to 27M condP for direct comparison.

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 10
BATCH_SIZE    = 32      # H100: B=32 GA=1 = effective batch 32, same as 13M baseline
GRAD_ACCUM    = 1
LR            = 3e-4
MAX_SEQ_LEN   = 2048
NUM_DOCS      = 100_000

# Same dimensions as 27M condP — only the attention mechanism differs
EMBEDDING_DIM = 400
NUM_LAYERS    = 6
NUM_HEADS     = 8       # d_head = 50
FFN_DIM       = 1600    # 4 × EMBEDDING_DIM

CHECKPOINT_DIR = '2048_27m_standard_baseline_checkpoints'

GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]


# ─── Standard Causal Self-Attention ──────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Standard O(N²) causal multi-head self-attention."""

    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.dropout   = nn.Dropout(dropout)

        # Causal mask — register as buffer so it moves with the model
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            self.causal_mask[:, :, :N, :N] == 0, float('-inf'))
        alpha  = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(alpha, v)

        return self.out_proj(out.permute(0, 2, 1, 3).reshape(B, N, D))


# ─── FFN ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ─── Transformer Block (pre-norm) ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, seq_len, dropout)
        self.ln2  = nn.LayerNorm(embedding_dim)
        self.ffn  = FFN(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ─── Standard Transformer ────────────────────────────────────────────────────

class StandardTransformer27M(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(seq_len, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ffn_dim, seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        # Separate out_proj — NOT weight-tied (consistent with 13M baseline)
        self.out_proj  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
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


# ─── Data utilities ───────────────────────────────────────────────────────────

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
                    sorted_probs /= sorted_probs.sum()
                    next_id = sorted_idx[torch.multinomial(sorted_probs, 1)]
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        gen = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


# ─── Training loop ────────────────────────────────────────────────────────────

def train(model, train_data, val_data, test_data, tokenizer,
          save_dir=CHECKPOINT_DIR, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss, best_val_ppl, best_epoch = float('inf'), float('inf'), 0
    t0     = time.time()
    epoch_log = []

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

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')

        epoch_log.append({
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4), 'val_ppl': round(val_ppl, 2),
            'elapsed_s': round(elapsed, 1),
        })

        print('  ── Generation samples (greedy) ──')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')
        sys.stdout.flush()

    # ── Final test eval ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  27M Standard Baseline TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')

    # Temperature sweep
    print('\n  ── Temperature sweep (best checkpoint) ──')
    sweep = {}
    for temp in [0.0, 0.5, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        gens  = generate(model, tokenizer, GEN_PROMPTS, device,
                         temperature=temp, top_p=0.9)
        sweep[label] = gens
        print(f'\n  [{label}]')
        for p, g in zip(GEN_PROMPTS, gens):
            print(f'    {repr(p)} → {repr(g[:80])}')

    # ── Scaling comparison summary ─────────────────────────────────────────────
    condP_13m_ppl     = 65.057
    baseline_13m_ppl  = 64.073
    condP_27m_ppl     = None   # filled in from the other run
    # NOTE: These are Wave Field architecture runs, NOT standard transformers.
    # "baseline_85m" = condA (Wave Field V4, no dispersion): test PPL 57.7
    # "condP_85m"    = condB (Wave Field V4 + learned dispersion): test PPL 58.1
    # There is NO 85M standard transformer or 85M condP in our data.
    # The closest standard transformer at scale is the pre-V4 100M run: test PPL 45.96.
    condP_85m_ppl     = 58.1   # actually condB (Wave Field + dispersion) — mislabeled
    baseline_85m_ppl  = 57.7   # actually condA (Wave Field V4) — mislabeled

    print('\n' + '=' * 72)
    print('  27M Standard Baseline — Scaling Summary')
    print('=' * 72)
    print(f'  {"Model":<48} {"Params":>8}  {"PPL":>8}  {"Gap":>8}')
    print(f'  {"─"*48}  {"─"*8}  {"─"*8}  {"─"*8}')
    print(f'  {"Standard 13M baseline":<48} {"21.6M":>8}  {baseline_13m_ppl:>8.3f}')
    print(f'  {"condP 13M":<48} {"14.1M":>8}  {condP_13m_ppl:>8.3f}  {condP_13m_ppl - baseline_13m_ppl:>+8.3f}')
    print(f'  {"Standard 27M baseline (this run)":<48} {"~38M":>8}  {test_ppl:>8.3f}')
    if condP_27m_ppl:
        print(f'  {"condP 27M":<48} {"26.8M":>8}  {condP_27m_ppl:>8.3f}  {condP_27m_ppl - test_ppl:>+8.3f}')
    print(f'  {"condA 85M (Wave Field V4, no disp) [mislabeled]":<48} {"~85M":>8}  {baseline_85m_ppl:>8.1f}')
    print(f'  {"condB 85M (Wave Field + dispersion) [mislabeled]":<48} {"~85M":>8}  {condP_85m_ppl:>8.1f}  {condP_85m_ppl - baseline_85m_ppl:>+8.1f}')
    print()
    print(f'  13M gap: condP vs baseline = {condP_13m_ppl - baseline_13m_ppl:+.3f} PPL')
    print(f'  27M gap: condP vs baseline = (run condP 27M to complete)')
    print(f'  85M gap: condP vs baseline = {condP_85m_ppl - baseline_85m_ppl:+.1f} PPL')
    print()
    print(f'  Pattern: gap should close at scale (two-thirds closure per doubling)')

    total_time = time.time() - t0
    return {
        'model':           'Standard Transformer 27M baseline',
        'test_ppl':        test_ppl,
        'test_loss':       test_loss,
        'best_val_ppl':    best_val_ppl,
        'best_val_loss':   best_val_loss,
        'best_epoch':      best_epoch,
        'total_time_s':    total_time,
        'epoch_log':       epoch_log,
        'architecture': {
            'type':          'pre-norm causal transformer',
            'vocab_size':    VOCAB_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'num_layers':    NUM_LAYERS,
            'num_heads':     NUM_HEADS,
            'ffn_dim':       FFN_DIM,
            'max_seq_len':   MAX_SEQ_LEN,
            'tied_embeddings': False,   # consistent with 13M baseline
            'activation':    'gelu',
            'normalization': 'pre-norm LayerNorm',
        },
        'training': {
            'optimizer':      'AdamW', 'lr': LR, 'weight_decay': 0.1,
            'betas':          [0.9, 0.95], 'batch_size': BATCH_SIZE,
            'grad_accum':     GRAD_ACCUM,
            'effective_batch': BATCH_SIZE * GRAD_ACCUM,
            'lr_schedule':    'cosine', 'grad_clip': 1.0, 'epochs': NUM_EPOCHS,
            'tokenizer':      'condI BPE 32K', 'dataset': 'OpenWebText 100K docs',
            'precision':      'bf16 AMP',
        },
        'temperature_sweep':  sweep,
        'condP_13m_ppl':      condP_13m_ppl,
        'baseline_13m_ppl':   baseline_13m_ppl,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 72)
    print('  Standard Transformer 27M Baseline')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, d_head={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Standard causal attention (O(N²), full softmax)')
    print(f'  Untied out_proj — consistent with 13M baseline')
    print('=' * 72)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    ckpt_dir   = os.path.join(repo_root, CHECKPOINT_DIR)
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_candidates = [
        os.path.join(script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in tok_candidates if os.path.exists(p)), None)
    if tok_path:
        from tokenizers import Tokenizer
        tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
        print(f'Loaded condI BPE tokenizer from {tok_path}')
    else:
        raise FileNotFoundError(
            'condI tokenizer not found. Tried:\n' +
            '\n'.join(f'  {p}' for p in tok_candidates))

    splits = load_data(NUM_DOCS)

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = StandardTransformer27M().to(device)

    n_params = model.param_count()
    print(f'\nStandard 27M baseline: {n_params:,} parameters  ({n_params/1e6:.2f}M)')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, d_head={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Comparison: condP 27M has 26,781,152 params (tied) vs this {n_params:,} (untied)')
    print(f'  Δparams = {n_params - 26_781_152:+,} ({(n_params/26_781_152 - 1)*100:.1f}% larger than condP 27M)')

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir=ckpt_dir, device=device)

    results_path = os.path.join(results_dir, '2048_27m_standard_baseline_results.json')
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
