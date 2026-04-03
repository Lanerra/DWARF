"""
smol-DWARF SFT — Instruction Fine-Tuning on smol-smoltalk
==========================================================
Base model:  autoresearch/checkpoints/moonshot_58m_best.pt  (ep2, PPL=35.04, passkey=99.2%)
Dataset:     logs/smoltalk_50k.pt  (47,500 train / 2,500 val, from HuggingFaceTB/smol-smoltalk)
Tokenizer:   results/fineweb_tokenizer_32k.json  (32K BPE, EOS=0)
Device:      3090 (CUDA_VISIBLE_DEVICES=1)
Arch:        D=512, H=8, L=8, FFN=2048, J=24 se015, FA@L2, preIF@L1

Chat template (plain-text, no new special tokens needed):
  ### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n{response}<EOS>

Training:
  - Full fine-tune (no LoRA) — 45.6M params is fast enough
  - Loss only on assistant tokens (mask user/system turns)
  - Bucketed batch sampler: token budget 32K tokens/batch, ~37 seqs avg
    Sorts by length → minimal padding (~1% waste vs 42% with random shuffle)
  - Chunked cross-entropy: avoids materialising giant fp32 logits gradient
  - LR=1e-4, warmup 100 steps, cosine decay
  - GRAD_ACCUM=1 (each bucket batch is already ~32K tokens)
  - 3 epochs on 47,500 examples

git: see `git log -1`
"""

import os, sys, json, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [os.path.join(REPO_ROOT, 'kernels'), REPO_ROOT]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Kernel import (3090 = sm86) ────────────────────────────────────────────────
try:
    from dsqg_attention_v8_4090 import DSQGAttentionV8_4090 as DSQGAttentionV6, npci_rotate
    _kernel_label = 'v8_4090 (sm86)'
except ImportError:
    from dsqg_attention_v8 import DSQGAttentionV8 as DSQGAttentionV6, npci_rotate
    _kernel_label = 'v8_generic'

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBEDDING_DIM        = 512
NUM_LAYERS           = 8
NUM_HEADS            = 8
FFN_DIM              = 2048
MAX_SEQ_LEN          = 2048
FULL_ATTN_LAYER      = 2       # FA@L2, preIF@L1

OFFSETS = [1,2,3,4,5,6,7,8,9,10,13,15,16,21,23,28,48,64,96,192,384,512,768,1024]
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0
EMA_INIT             = 0.0208  # 1/δ_relay_min = 1/48

LR                   = 1e-4
WARMUP_STEPS         = 100
SCREEN_EPOCHS        = 3
# Token budget per physical batch — auto-sizes BS by sequence length.
# At mean=882 tok → ~37 seqs/batch; at max=2048 tok → ~16 seqs/batch.
# Peak VRAM budget: ~16 GB at worst case (seq=2048, BS=16) on 24 GB 3090.
TOKEN_BUDGET         = 20_000  # conservative for 3090 fragmentation safety
GRAD_ACCUM           = 1       # each bucket batch is already ~32K tokens
GRAD_CLIP            = 1.0
WEIGHT_DECAY         = 0.01

# Chunked cross-entropy chunk size (tokens per chunk)
# Avoids materialising 32K-vocab fp32 logits grad all at once.
CE_CHUNK             = 512

MAX_VAL_SEQS         = 500

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]
DATA_FILE    = 'logs/smoltalk_50k.pt'
BASE_CKPT    = 'autoresearch/checkpoints/moonshot_58m_best.pt'
SAVE_DIR     = 'autoresearch/checkpoints/smol_dwarf_sft'
RESULT_FILE  = 'autoresearch/smol_dwarf_sft_results.json'

# ── Chat template ──────────────────────────────────────────────────────────────
SYSTEM_PREFIX    = '### System:\n'
USER_PREFIX      = '### User:\n'
ASSISTANT_PREFIX = '### Assistant:\n'
TURN_SEP         = '\n\n'

def format_conversation(messages, eos_id, tokenizer):
    """
    Returns (token_ids: list[int], loss_mask: list[bool]).
    Loss is computed only on assistant tokens. Truncates to MAX_SEQ_LEN.
    """
    token_ids = []
    loss_mask = []

    msg_iter = iter(messages)
    first = next(msg_iter, None)
    if first is None:
        return [], []

    # Optional system turn
    if first['role'] == 'system':
        sys_toks = tokenizer.encode(SYSTEM_PREFIX + first['content'] + TURN_SEP).ids
        token_ids.extend(sys_toks)
        loss_mask.extend([False] * len(sys_toks))
        first = next(msg_iter, None)

    current = first
    while current is not None:
        nxt  = next(msg_iter, None)
        role = current['role']
        content = current['content']

        if role == 'user':
            toks = tokenizer.encode(USER_PREFIX + content + TURN_SEP).ids
            token_ids.extend(toks)
            loss_mask.extend([False] * len(toks))
        elif role == 'assistant':
            prefix_toks  = tokenizer.encode(ASSISTANT_PREFIX).ids
            content_toks = tokenizer.encode(content).ids
            token_ids.extend(prefix_toks)
            loss_mask.extend([False] * len(prefix_toks))
            token_ids.extend(content_toks)
            loss_mask.extend([True] * len(content_toks))
            token_ids.append(eos_id)
            loss_mask.append(True)
            if nxt is not None:
                sep_toks = tokenizer.encode(TURN_SEP).ids
                token_ids.extend(sep_toks)
                loss_mask.extend([False] * len(sep_toks))

        current = nxt

    token_ids = token_ids[:MAX_SEQ_LEN]
    loss_mask  = loss_mask[:MAX_SEQ_LEN]
    return token_ids, loss_mask


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, eos_id):
        self.data    = []
        self.lengths = []   # for BucketBatchSampler
        skipped = 0
        for ex in examples:
            ids, mask = format_conversation(ex['messages'], eos_id, tokenizer)
            if len(ids) < 4 or sum(mask) == 0:
                skipped += 1
                continue
            t_ids  = torch.tensor(ids,  dtype=torch.long)
            t_mask = torch.tensor(mask, dtype=torch.bool)
            self.data.append((t_ids, t_mask))
            self.lengths.append(len(ids))
        if skipped:
            print(f'  Skipped {skipped} examples (too short or no assistant tokens)', flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BucketBatchSampler(Sampler):
    """
    Yields batches of indices where all sequences in a batch are similar length.
    Target: TOKEN_BUDGET total tokens per batch (= max_len_in_batch × n_seqs).
    Shuffles batch order each epoch; within each bucket the sort order is stable.
    """
    def __init__(self, lengths, token_budget, shuffle=True):
        self.lengths      = lengths
        self.token_budget = token_budget
        self.shuffle      = shuffle
        self._epoch       = 0

    def _build_batches(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, batch, cur_max = [], [], 0
        for idx in order:
            L = self.lengths[idx]
            new_max = max(cur_max, L)
            if batch and (len(batch) + 1) * new_max > self.token_budget:
                batches.append(batch)
                batch, cur_max = [idx], L
            else:
                batch.append(idx)
                cur_max = new_max
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self.shuffle:
            rng = random.Random(42 + self._epoch)
            random.Random(42 + self._epoch).shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return len(self._build_batches())

    def set_epoch(self, epoch):
        self._epoch = epoch


def collate_fn(batch):
    """Pad batch to max length in batch (not global MAX_SEQ_LEN)."""
    ids_list, mask_list = zip(*batch)
    max_len = max(t.size(0) for t in ids_list)
    B = len(ids_list)
    padded_ids  = torch.zeros(B, max_len, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = ids.size(0)
        padded_ids[i, :L]  = ids
        padded_mask[i, :L] = mask
    return padded_ids, padded_mask


# ── Model ─────────────────────────────────────────────────────────────────────
class BPETokenizerWrapper:
    def __init__(self, tok):
        self._tok = tok
    def encode(self, text):
        return self._tok.encode(text)
    def decode(self, ids):
        return self._tok.decode(ids)
    def vocab_size(self):
        return self._tok.get_vocab_size()
    def token_to_id(self, t):
        return self._tok.token_to_id(t)


def _import_moonshot_model():
    import importlib.util
    script = os.path.join(REPO_ROOT, 'train', 'train_moonshot_58m_4090_bf16.py')
    spec   = importlib.util.spec_from_file_location('moonshot_train', script)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AutoresearchTransformerPhysics


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Chunked SFT loss ──────────────────────────────────────────────────────────
def sft_loss_chunked(logits, targets, loss_mask):
    """
    Masked cross-entropy, computed in token chunks to avoid OOM on 32K vocab.
    logits:    [B, T, V]  (bfloat16 OK — cast to float32 per chunk)
    targets:   [B, T]     (token IDs)
    loss_mask: [B, T]     (True = compute loss at this position)
    Returns scalar loss (mean over assistant tokens).
    """
    B, T, V = logits.shape
    # Shift: position t predicts token t+1
    shift_logits = logits[:, :-1]      # [B, T-1, V]
    shift_target = targets[:, 1:]      # [B, T-1]
    shift_mask   = loss_mask[:, 1:]    # [B, T-1]

    total_loss   = torch.zeros((), device=logits.device, dtype=torch.float32)
    total_tokens = shift_mask.float().sum()

    for c in range(0, T - 1, CE_CHUNK):
        chunk_logits  = shift_logits[:, c:c+CE_CHUNK].reshape(-1, V).float()
        chunk_targets = shift_target[:, c:c+CE_CHUNK].reshape(-1)
        chunk_mask    = shift_mask[:, c:c+CE_CHUNK].reshape(-1).float()
        loss_per_tok  = F.cross_entropy(chunk_logits, chunk_targets, reduction='none')
        total_loss    = total_loss + (loss_per_tok * chunk_mask).sum()

    return total_loss / total_tokens.clamp(min=1)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids, mask in val_loader:
        ids, mask = ids.to(device), mask.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ids)
        B, T, V = logits.shape
        shift_logits  = logits[:, :-1].reshape(-1, V).float()
        shift_targets = ids[:, 1:].reshape(-1)
        shift_mask    = mask[:, 1:].reshape(-1).float()
        loss_all = F.cross_entropy(shift_logits, shift_targets, reduction='none')
        total_loss   += (loss_all * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


# ── Generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, eos_id, prompt, max_new=200, temperature=0.7, device='cuda'):
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        if ids.size(1) >= MAX_SEQ_LEN:
            break
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ids)
        next_tok = torch.multinomial(torch.softmax(logits[0, -1] / temperature, dim=-1), 1)
        ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)
        if next_tok.item() == eos_id:
            break
    model.train()
    return tokenizer.decode(ids[0].tolist())


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    cc       = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f'GPU: {gpu_name}  (sm_{cc[0]}{cc[1]})', flush=True)
    print(f'Kernel: {_kernel_label}', flush=True)

    # Tokenizer
    tok_path = next((p for p in TOKENIZER_CANDIDATES
                     if os.path.exists(os.path.join(REPO_ROOT, p))), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(os.path.join(REPO_ROOT, tok_path)))
    eos_id = tokenizer.token_to_id('<|endoftext|>') or 0
    print(f'Tokenizer: {tok_path}  vocab={tokenizer.vocab_size()}  EOS={eos_id}', flush=True)

    # Dataset
    print(f'Loading dataset from {DATA_FILE}...', flush=True)
    raw      = torch.load(os.path.join(REPO_ROOT, DATA_FILE), weights_only=False)
    val_raw  = raw['val'][:MAX_VAL_SEQS]

    print(f'Tokenizing {len(raw["train"]):,} train / {len(val_raw):,} val examples...', flush=True)
    t0       = time.time()
    train_ds = SFTDataset(raw['train'], tokenizer, eos_id)
    val_ds   = SFTDataset(val_raw,      tokenizer, eos_id)
    print(f'  Done in {time.time()-t0:.1f}s  '
          f'train={len(train_ds):,}  val={len(val_ds):,}', flush=True)

    # Bucket sampler — minimal padding
    sampler      = BucketBatchSampler(train_ds.lengths, TOKEN_BUDGET, shuffle=True)
    n_batches    = len(sampler)
    mean_bs      = len(train_ds) / n_batches
    print(f'  BucketBatchSampler: {n_batches} batches/epoch  '
          f'mean_bs={mean_bs:.1f}  token_budget={TOKEN_BUDGET}', flush=True)

    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model
    AutoresearchTransformerPhysics = _import_moonshot_model()
    model = AutoresearchTransformerPhysics(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=None,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,} ({total_params/1e6:.1f}M)', flush=True)

    # Load base checkpoint
    print(f'Loading base checkpoint: {BASE_CKPT}', flush=True)
    ckpt  = torch.load(os.path.join(REPO_ROOT, BASE_CKPT), map_location='cpu', weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing: {missing[:5]}', flush=True)
    if unexpected:
        print(f'  Unexpected: {unexpected[:5]}', flush=True)
    print('  Checkpoint loaded.', flush=True)

    # Optimizer — scale_embed gets LR×15
    se_params    = [p for n, p in model.named_parameters() if 'scale_embed' in n]
    base_params  = [p for n, p in model.named_parameters() if 'scale_embed' not in n]
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': LR,                       'weight_decay': WEIGHT_DECAY},
        {'params': se_params,   'lr': LR * SCALE_EMBED_LR_MULT, 'weight_decay': 0.0},
    ], betas=(0.9, 0.95))

    total_steps = n_batches * SCREEN_EPOCHS // max(GRAD_ACCUM, 1)
    print(f'  LR={LR}  warmup={WARMUP_STEPS}  total_optim_steps={total_steps}', flush=True)
    print(f'  TOKEN_BUDGET={TOKEN_BUDGET}  GRAD_ACCUM={GRAD_ACCUM}', flush=True)

    best_val_ppl = float('inf')
    results      = []
    optim_step   = 0

    for epoch in range(1, SCREEN_EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        ep_start = time.time()
        optimizer.zero_grad()

        for batch_idx, (ids, mask) in enumerate(train_loader, 1):
            ids, mask = ids.to(device), mask.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(ids)

            loss = sft_loss_chunked(logits, ids, mask)
            if GRAD_ACCUM > 1:
                loss = loss / GRAD_ACCUM
            loss.backward()

            if batch_idx % GRAD_ACCUM == 0 or batch_idx == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr_now = get_lr(optim_step, total_steps, WARMUP_STEPS, LR)
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = lr_now * (SCALE_EMBED_LR_MULT if i == 1 else 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optim_step += 1

            if batch_idx % 100 == 0:
                raw_loss = loss.item() * (GRAD_ACCUM if GRAD_ACCUM > 1 else 1)
                elapsed  = time.time() - ep_start
                secs_per_step = elapsed / batch_idx
                eta_min  = secs_per_step * (n_batches - batch_idx) / 60
                print(f'  Step {batch_idx}/{n_batches} | Loss {raw_loss:.4f} '
                      f'| BS={ids.size(0)} seq={ids.size(1)} '
                      f'| ETA {eta_min:.0f}m', flush=True)

        # Epoch end
        val_loss, val_ppl = evaluate(model, val_loader, device)
        elapsed = time.time() - ep_start

        # Physics summary
        se_str = phys_str = ''
        for i, blk in enumerate(model.blocks):
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'ema_alpha'):
                alpha   = blk.attn.ema_alpha.data.mean().item()
                se_mean = blk.attn.scale_embed.data.abs().mean().item()
                se_max  = blk.attn.scale_embed.data.abs().max().item()
                se_str   = f'scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}'
                phys_str = f'b{i}: α={alpha:.4f}(w≈{1/max(alpha,1e-9):.0f}t)'

        star = ' *' if val_ppl < best_val_ppl else ''
        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{star} '
              f'| {elapsed/60:.1f}m', flush=True)
        if se_str:
            print(f'  {se_str}', flush=True)
            print(f'  Physics: {phys_str}', flush=True)

        # Sample generations
        test_prompts = [
            '### User:\nWhat is the capital of France?\n\n### Assistant:\n',
            '### User:\nExplain what a neural network is in simple terms.\n\n### Assistant:\n',
            '### User:\nWrite a haiku about the ocean.\n\n### Assistant:\n',
        ]
        print('  Samples:', flush=True)
        for p in test_prompts:
            out      = generate(model, tokenizer, eos_id, p, max_new=120, device=device)
            response = out[len(p):].strip()
            question = p.split('\n')[1].strip()[:60]
            print(f'    Q: {question}', flush=True)
            print(f'    A: {response[:160]}', flush=True)

        # Checkpoints
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch, 'val_ppl': val_ppl,
                'model_state_dict': model.state_dict(),
            }, os.path.join(REPO_ROOT, SAVE_DIR, 'smol_dwarf_sft_best.pt'))

        torch.save({
            'epoch': epoch, 'val_ppl': val_ppl,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optim_step':           optim_step,
        }, os.path.join(REPO_ROOT, SAVE_DIR, f'smol_dwarf_sft_ep{epoch}_resume.pt'))

        results.append({
            'epoch': epoch,
            'val_ppl': round(val_ppl, 3),
            'elapsed_s': round(elapsed, 1),
            'num_params_M': round(total_params / 1e6, 1),
        })
        with open(os.path.join(REPO_ROOT, RESULT_FILE), 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nDone. Best val PPL: {best_val_ppl:.2f}', flush=True)


if __name__ == '__main__':
    main()
