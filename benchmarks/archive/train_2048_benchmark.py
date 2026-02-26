"""
Wave Field V3.5 — 2048-Token Scaling Benchmark
================================================
Train both 6M and 100M models at 2048 seq_len on OpenWebText.
Fills in the missing points on the scaling curve:

  Params  | seqlen | Std PPL | Wave PPL | gap%
  --------|--------|---------|----------|------
  6-8M    |  512   |  91.4   |  170.7   | 87%   (WikiText-2, old)
  100M    |  512   |  46.0   |   72.4   | 57%   (OpenWebText)
  6M      | 2048   |   ???   |    ???   |  ?    ← THIS SCRIPT
  100M    | 2048   |   ???   |    ???   |  ?    ← THIS SCRIPT

Key fixes vs previous 100M run:
  - WaveFieldAttention now uses dynamic stride: stride = (G-1)/(N-1)
    computed at forward() time, not frozen at construction.  A single
    model generalises to any context length without aliasing or NaN.
  - field_size scaled 2x relative to max_seq_len (same ratio as 512→1024)
    so tokens are spread equally across the field at training time.

Memory notes (RTX 4090, 24 GB):
  - 6M  @ 2048: batch=8, grad_accum=4   (effective=32)
  - 100M@ 2048: batch=2, grad_accum=16  (effective=32)
  Standard Transformer attention at 2048 is O(n²) — heavier than Wave
  Field's O(n log n) — so std may OOM first; reduce batch_size if needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import gc

from src.wave_field_transformer import WaveFieldTransformer


# ======================================================================
# STANDARD TRANSFORMER
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=2050, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.positional_embedding(pos)
        x = self.dropout(x)
        causal_mask = self._causal_mask(N, input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.output_projection(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        return logits, loss


# ======================================================================
# TOKENIZER
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPEWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


# ======================================================================
# DATA
# ======================================================================

def load_openwebtext(max_docs=100000):
    from datasets import load_dataset
    print(f"Loading OpenWebText (up to {max_docs:,} docs)...")
    ds = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= max_docs:
            break
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if (i + 1) % 25000 == 0:
            print(f"  Loaded {i+1:,} docs...")
    n = len(texts)
    train_end = int(n * 0.95)
    val_end   = int(n * 0.975)
    print(f"  Total {n:,} docs | train {train_end:,} | val {val_end - train_end:,} | test {n - val_end:,}")
    return {'train': texts[:train_end], 'valid': texts[train_end:val_end], 'test': texts[val_end:]}


def encode_lines(lines, tok, max_seq_len):
    data = []
    for line in lines:
        ids = tok.encode(line)
        if len(ids) < 2:
            continue
        for s in range(0, len(ids) - 1, max_seq_len):
            chunk = ids[s:s + max_seq_len + 1]
            if len(chunk) >= 2:
                data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=True):
    indices = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        bx = [data[i][0] for i in batch_idx]
        by = [data[i][1] for i in batch_idx]
        ml = max(x.size(0) for x in bx)
        px = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(bx, by)):
            px[i, :x.size(0)] = x
            py[i, :y.size(0)] = y
        batches.append((px, py))
    return batches


# ======================================================================
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lr      = optimizer.param_groups[0]['lr']
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            p  = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


@torch.no_grad()
def evaluate(model, batches, vocab_size, device, use_amp=False):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


def train_model(model, train_data, val_data, tok, vocab_size, device,
                model_name, num_epochs=10, batch_size=8, grad_accum=4,
                peak_lr=3e-4, use_amp=True, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    effective_batch = batch_size * grad_accum
    print(f"\n  {model_name}: {params:,} params | batch {batch_size}×{grad_accum}={effective_batch} | lr {peak_lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                   weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    spe = math.ceil(len(train_data) / batch_size)
    scheduler = WarmupCosineScheduler(optimizer, spe * 1, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_vl, best_vp, best_va, best_ep = float('inf'), float('inf'), 0, 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb = 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(batches):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
                loss = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            tl += loss.item() * grad_accum
            nb += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 200 == 0:
                print(f"    Step {step+1}/{len(batches)} | Loss {tl/nb:.4f}", flush=True)

        al  = tl / max(nb, 1)
        et  = time.time() - et
        vb  = create_batches(val_data, batch_size, device, shuffle=False)
        vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp)

        if vl < best_vl:
            best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            mk = " * BEST"
        else:
            mk = ""
        print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} PPL {vp:.1f} Acc {va:.1f}% | {et:.0f}s{mk}")

    total = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))
    return {'model_name': model_name, 'params': params,
            'best_ppl': best_vp, 'best_acc': best_va, 'best_epoch': best_ep, 'total_time': total}


# ======================================================================
# MODEL CONFIGS
# ======================================================================

CONFIGS = {
    '6m': dict(
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        # Wave field: 2× seq_len so stride at training ≈ 2 (same ratio as 512→1024)
        field_size=4096,
        num_epochs=10,
        batch_size=8,
        grad_accum=4,      # effective = 32
        peak_lr=3e-4,
    ),
    '100m': dict(
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        ffn_dim=3072,
        field_size=4096,   # 2× seq_len (2048×2)
        num_epochs=10,
        batch_size=2,      # O(n²) standard attn is memory-hungry at 2048
        grad_accum=16,     # effective = 32
        peak_lr=6e-4,
    ),
}


# ======================================================================
# MAIN
# ======================================================================

def run_size(size_key, splits, tok, vocab_size, device, use_amp, max_seq_len=2048):
    cfg = CONFIGS[size_key]
    label = size_key.upper()
    print(f"\n{'='*70}")
    print(f"  {label} MODELS — seq_len={max_seq_len} — OpenWebText")
    print(f"{'='*70}")

    print(f"  Encoding data (max_seq_len={max_seq_len})...")
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data   = encode_lines(splits['valid'], tok, max_seq_len)
    test_data  = encode_lines(splits['test'],  tok, max_seq_len)
    print(f"  Train {len(train_data):,} | Val {len(val_data):,} | Test {len(test_data):,}")

    results = []

    # ---- Standard Transformer ----------------------------------------
    print(f"\n  --- {label} Standard Transformer (O(n²)) ---")
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=max_seq_len + 2,
        dropout=0.1,
    ).to(device)

    std_res = train_model(
        std_model, train_data, val_data, tok, vocab_size, device,
        f"Standard {label}",
        num_epochs=cfg['num_epochs'], batch_size=cfg['batch_size'],
        grad_accum=cfg['grad_accum'], peak_lr=cfg['peak_lr'],
        use_amp=use_amp, save_dir=f"2048_{size_key}_std_checkpoints",
    )

    test_batches = create_batches(test_data, cfg['batch_size'], device, shuffle=False)
    _, std_tp, std_ta = evaluate(std_model, test_batches, vocab_size, device, use_amp)
    std_res['test_ppl'] = std_tp
    std_res['test_acc'] = std_ta
    print(f"  {label} Std  TEST: PPL {std_tp:.1f} | Acc {std_ta:.1f}%")
    results.append(std_res)

    del std_model
    gc.collect(); torch.cuda.empty_cache()

    # ---- Wave Field --------------------------------------------------
    print(f"\n  --- {label} Wave Field V3.5 (O(n log n)) ---")
    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=max_seq_len + 2,
        dropout=0.1,
        use_checkpoint=True,
        interference_interval=3,
        device=device,
    ).to(device)

    wave_res = train_model(
        wave_model, train_data, val_data, tok, vocab_size, device,
        f"Wave {label}",
        num_epochs=cfg['num_epochs'], batch_size=cfg['batch_size'],
        grad_accum=cfg['grad_accum'], peak_lr=cfg['peak_lr'],
        use_amp=use_amp, save_dir=f"2048_{size_key}_wave_checkpoints",
    )

    test_batches = create_batches(test_data, cfg['batch_size'], device, shuffle=False)
    _, wave_tp, wave_ta = evaluate(wave_model, test_batches, vocab_size, device, use_amp)
    wave_res['test_ppl'] = wave_tp
    wave_res['test_acc'] = wave_ta
    print(f"  {label} Wave TEST: PPL {wave_tp:.1f} | Acc {wave_ta:.1f}%")
    results.append(wave_res)

    del wave_model
    gc.collect(); torch.cuda.empty_cache()

    return results


def print_summary(all_results):
    """Print scaling table with all data points."""
    print(f"\n{'='*75}")
    print("  SCALING CURVE — OpenWebText, 32K BPE")
    print(f"  {'Config':<18} {'Params':>10} {'seq':>6} {'Std PPL':>10} {'Wave PPL':>10} {'Gap %':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    # Historical reference points (different dataset, included for context)
    print(f"  {'[6-8M WikiText-2]':<18} {'~7M':>10} {'512':>6} {'91.4':>10} {'170.7':>10} {'87.0%':>8}  (old, diff dataset)")
    print(f"  {'[100M OpenWebText]':<18} {'~105M':>10} {'512':>6} {'46.0':>10} {'72.4':>10} {'57.5%':>8}  (prev run)")

    for size_key in ['6m', '100m']:
        key = f"{size_key}_2048"
        if key not in all_results:
            continue
        std, wave = all_results[key]
        std_p, wave_p = std['test_ppl'], wave['test_ppl']
        gap = (wave_p - std_p) / std_p * 100 if std_p > 0 else float('nan')
        label = f"~{std['params']//1_000_000}M @ 2048"
        print(f"  {label:<18} {std['params']:>10,} {'2048':>6} {std_p:>10.1f} {wave_p:>10.1f} {gap:>7.1f}%")

    print(f"{'='*75}")


def main():
    max_seq_len = 2048

    print("=" * 70)
    print("  WAVE FIELD V3.5 — 2048-TOKEN SCALING BENCHMARK")
    print(f"  Train 6M + 100M models at seq_len={max_seq_len} on OpenWebText")
    print("=" * 70)
    print("\n  Dynamic stride fix: wave field positions computed from actual N")
    print("  Field size = 4096 (2× seq_len, same ratio as 512→1024 baseline)")

    splits = load_openwebtext(max_docs=100000)

    bpe_vocab_size = 32000
    print(f"\nTraining BPE tokenizer (vocab={bpe_vocab_size}) on 50K docs...")
    raw_tok = train_bpe_tokenizer(splits['train'][:50000], vocab_size=bpe_vocab_size)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  BPE vocab: {vocab_size} tokens")

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  Device: {device} | AMP: {use_amp}")

    all_results = {}

    for size_key in ['6m', '100m']:
        res = run_size(size_key, splits, tok, vocab_size, device, use_amp, max_seq_len=max_seq_len)
        all_results[f"{size_key}_2048"] = res

    print_summary(all_results)

    # Flatten for JSON
    flat = {}
    for k, v in all_results.items():
        flat[k] = v
    with open("2048_benchmark_results.json", 'w') as f:
        json.dump(flat, f, indent=2)
    print("\n  Results saved → 2048_benchmark_results.json")


if __name__ == "__main__":
    main()
