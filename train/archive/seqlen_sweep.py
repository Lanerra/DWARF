"""
Sequence Length Sweep — Standard vs Wave Field (100M)
======================================================
Evaluates both trained checkpoints at varying sequence lengths:
  512, 1024, 2048, 4096

Purpose: does Wave Field's O(n log n) architecture confer any
advantage at longer contexts? Exploratory data point — both models
were only trained at 512 tokens, so this tests extrapolation.

Architectural constraints (noted in output):
  Standard:   Learned positional embeddings (max 514 trained).
              Positions beyond 511 are clamped → all extra tokens
              share position 511's embedding. Causal mask is full-length.

  Wave Field: Sinusoidal PE (extrapolates freely).
              BUT field_stride is fixed at training time:
              field_stride ≈ (1024-1)/(514-1) ≈ 1.99
              → tokens beyond position ~512 alias to field pos ~1022.
              Multiple tokens compete for the same field bucket.

Neither is truly "long context" evaluation. Both results reflect
extrapolation behaviour, not trained capability.

Evaluation method: sliding window, stride = seq_len (non-overlapping).
Tokens are drawn from a single concatenated OpenWebText test stream.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time

from src.wave_field_transformer import WaveFieldTransformer


# ====================================================================
# STANDARD TRANSFORMER — position-clamped for long-context eval
# ====================================================================

class StandardTransformer(nn.Module):
    """
    Identical architecture to training, with one modification:
    positions are clamped to [0, max_seq_len-1] so evaluation at
    longer sequence lengths doesn't raise IndexError.

    Tokens beyond position max_seq_len-1 all use the same
    positional embedding (position 511). Causal attention mask
    covers the full sequence.
    """
    def __init__(self, vocab_size, embedding_dim=768, num_layers=12,
                 num_heads=12, ffn_dim=3072, max_seq_len=514, dropout=0.1):
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        # Clamp: tokens beyond training length reuse the last trained position
        positions = positions.clamp(0, self.max_seq_len - 1)
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        x = self.dropout(x)
        causal_mask = self._generate_causal_mask(N, input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.output_projection(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ====================================================================
# TOKENIZER — train from same data and save for reuse
# ====================================================================

TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "100m_bpe_tokenizer.json",
)

def load_or_train_tokenizer(max_docs=100000, bpe_vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    if os.path.exists(TOKENIZER_PATH):
        print(f"  Loading saved tokenizer from {TOKENIZER_PATH}")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        return tokenizer

    print(f"  No saved tokenizer found. Training BPE (vocab={bpe_vocab_size}) from OpenWebText...")
    print(f"  (Will save to {TOKENIZER_PATH} for future runs)")
    from datasets import load_dataset

    ds = load_dataset("openwebtext", split="train", streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= 50000:  # same 50K docs used during training
            break
        text = item["text"].strip()
        if len(text) > 50:
            texts.append(text)
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,} docs loaded...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=bpe_vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"  Tokenizer saved.")
    return tokenizer


class BPEWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


# ====================================================================
# DATA — build a long token stream from test docs
# ====================================================================

def build_token_stream(max_docs=100000, tok=None, max_stream_tokens=2_000_000):
    """
    Load OpenWebText test split (same 2.5% held-out docs as training),
    tokenize all docs, concatenate into a single long stream.
    Cap at max_stream_tokens to keep evaluation fast.
    """
    from datasets import load_dataset
    print(f"  Loading OpenWebText to build test token stream...")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    # Replicate the same train/val/test split used in training:
    # 0–94999 = train, 95000–97499 = val, 97500–99999 = test
    all_tokens = []
    doc_count = 0
    for i, item in enumerate(ds):
        if i < 97500:          # skip train + val
            continue
        text = item["text"].strip()
        if len(text) > 50:
            ids = tok.encode(text)
            all_tokens.extend(ids)
            doc_count += 1
        if i >= max_docs - 1:
            break
        if len(all_tokens) >= max_stream_tokens:
            break

    print(f"  Test stream: {doc_count:,} docs, {len(all_tokens):,} tokens")
    return all_tokens


# ====================================================================
# EVALUATION
# ====================================================================

@torch.no_grad()
def evaluate_at_seqlen(model, token_stream, seq_len, vocab_size, device,
                       use_amp=True, batch_size=4, max_chunks=500):
    """
    Sliding window PPL at a given sequence length.
    Creates non-overlapping (input, label) pairs from the token stream.
    """
    model.eval()

    # Build chunks: each chunk is seq_len+1 tokens (input + label)
    chunks = []
    for start in range(0, len(token_stream) - seq_len, seq_len):
        chunk = token_stream[start : start + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        return None, None

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        B = len(batch_chunks)
        x = torch.tensor([c[:-1] for c in batch_chunks], dtype=torch.long, device=device)
        y = torch.tensor([c[1:]  for c in batch_chunks], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
                ignore_index=-100,
            )

        total_loss += loss.item()
        n_batches += 1
        mask = y != -100
        total_correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    acc = total_correct / max(total_tokens, 1) * 100
    return ppl, acc


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 70)
    print("  SEQUENCE LENGTH SWEEP — Standard vs Wave Field (100M params)")
    print("  Exploratory: both models trained at 512 tokens only")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n  Device: {device} | AMP: {use_amp}")

    # ----------------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------------
    print("\n[1/4] Tokenizer")
    raw_tok = load_or_train_tokenizer()
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  Vocab size: {vocab_size:,}")

    # ----------------------------------------------------------------
    # Token stream
    # ----------------------------------------------------------------
    print("\n[2/4] Building test token stream")
    token_stream = build_token_stream(max_docs=100000, tok=tok, max_stream_tokens=2_000_000)

    # ----------------------------------------------------------------
    # Load checkpoints
    # ----------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    std_ckpt = os.path.join(base_dir, "100m_std_checkpoints", "best.pt")
    wave_ckpt = os.path.join(base_dir, "100m_wave_checkpoints", "best.pt")

    if not os.path.exists(std_ckpt):
        print(f"ERROR: Standard checkpoint not found at {std_ckpt}")
        sys.exit(1)
    if not os.path.exists(wave_ckpt):
        print(f"ERROR: Wave Field checkpoint not found at {wave_ckpt}")
        sys.exit(1)

    print(f"\n[3/4] Loading checkpoints")

    embed_dim = 768
    num_layers = 12
    num_heads = 12
    ffn_dim = 3072
    field_size = 1024
    trained_seq_len = 512

    print(f"  Loading Standard Transformer...")
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=trained_seq_len + 2,
        dropout=0.1,
    ).to(device)
    std_model.load_state_dict(
        torch.load(std_ckpt, map_location=device, weights_only=True)
    )
    std_params = sum(p.numel() for p in std_model.parameters())
    print(f"  Standard: {std_params:,} params  ✓")

    print(f"  Loading Wave Field Transformer...")
    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        field_size=field_size,
        max_seq_len=trained_seq_len + 2,
        dropout=0.1,
        use_checkpoint=False,   # no checkpointing needed for eval
        interference_interval=3,
        device=device,
    ).to(device)
    wave_model.load_state_dict(
        torch.load(wave_ckpt, map_location=device, weights_only=True)
    )
    wave_params = sum(p.numel() for p in wave_model.parameters())
    print(f"  Wave Field: {wave_params:,} params  ✓")

    # ----------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------
    seq_lens = [512, 1024, 2048, 4096]
    batch_size = 4

    print(f"\n[4/4] Sequence length sweep")
    print(f"  Lengths: {seq_lens}")
    print(f"  Chunks per length: up to 500 (non-overlapping)")
    print()

    results = []

    for seq_len in seq_lens:
        print(f"  --- seq_len = {seq_len} ---")

        # Std note
        if seq_len > trained_seq_len:
            std_note = f"pos clamped at {trained_seq_len}"
        else:
            std_note = "normal"

        # Wave note
        field_stride = (field_size - 1) / max(trained_seq_len + 2 - 1, 1)
        alias_start = math.ceil((field_size - 2) / field_stride)
        if seq_len > alias_start:
            wave_note = f"field aliasing beyond pos {alias_start}"
        else:
            wave_note = "normal"

        t0 = time.time()
        std_ppl, std_acc = evaluate_at_seqlen(
            std_model, token_stream, seq_len, vocab_size, device,
            use_amp=use_amp, batch_size=batch_size,
        )
        std_time = time.time() - t0

        t0 = time.time()
        wave_ppl, wave_acc = evaluate_at_seqlen(
            wave_model, token_stream, seq_len, vocab_size, device,
            use_amp=use_amp, batch_size=batch_size,
        )
        wave_time = time.time() - t0

        if std_ppl is None or wave_ppl is None:
            print(f"    Not enough tokens for seq_len={seq_len}, skipping.")
            continue

        gap_pct = (wave_ppl - std_ppl) / std_ppl * 100 if wave_ppl > std_ppl else -(std_ppl - wave_ppl) / std_ppl * 100
        winner = "Standard" if std_ppl < wave_ppl else "Wave Field"

        print(f"    Standard:   PPL {std_ppl:6.2f}  Acc {std_acc:.1f}%  ({std_time:.1f}s)  [{std_note}]")
        print(f"    Wave Field: PPL {wave_ppl:6.2f}  Acc {wave_acc:.1f}%  ({wave_time:.1f}s)  [{wave_note}]")
        print(f"    Gap: {abs(gap_pct):.1f}%  Winner: {winner}")
        print()

        results.append({
            "seq_len": seq_len,
            "std_ppl": std_ppl,
            "std_acc": std_acc,
            "std_note": std_note,
            "wave_ppl": wave_ppl,
            "wave_acc": wave_acc,
            "wave_note": wave_note,
            "gap_pct": gap_pct,
            "winner": winner,
        })

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("=" * 70)
    print("  SEQUENCE LENGTH SWEEP RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Seq Len':>8}  {'Std PPL':>9}  {'Wave PPL':>9}  {'Gap':>8}  {'Winner':>12}  Notes")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*12}  -----")
    for r in results:
        gap_str = f"{r['gap_pct']:+.1f}%"
        notes = ""
        if r['std_note'] != "normal":
            notes += f"Std: {r['std_note']}; "
        if r['wave_note'] != "normal":
            notes += f"Wave: {r['wave_note']}"
        print(f"  {r['seq_len']:>8}  {r['std_ppl']:>9.2f}  {r['wave_ppl']:>9.2f}  {gap_str:>8}  {r['winner']:>12}  {notes}")

    print()
    print("  Notes:")
    print("  - Positive gap = Wave Field worse; negative gap = Wave Field better")
    print("  - Both models trained at 512 tokens; longer lengths are extrapolation")
    print("  - Std pos-clamping: tokens beyond pos 511 share position embedding")
    print(f"  - Wave field aliasing: tokens beyond pos ~{alias_start} stack in same field bucket")
    print()

    # Trend analysis
    if len(results) >= 2:
        first_gap = results[0]['gap_pct']
        last_gap = results[-1]['gap_pct']
        print("  Trend:")
        if last_gap < first_gap:
            print(f"    Gap narrows from {first_gap:+.1f}% at {results[0]['seq_len']} tokens")
            print(f"    to {last_gap:+.1f}% at {results[-1]['seq_len']} tokens.")
            print(f"    Wave Field degrades slower (or benefits more) at longer lengths.")
        elif last_gap > first_gap:
            print(f"    Gap widens from {first_gap:+.1f}% at {results[0]['seq_len']} tokens")
            print(f"    to {last_gap:+.1f}% at {results[-1]['seq_len']} tokens.")
            print(f"    Standard degrades slower at longer lengths.")
        else:
            print(f"    Gap roughly stable across sequence lengths.")

    # Save results
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "seqlen_sweep_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print()
    print("=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
