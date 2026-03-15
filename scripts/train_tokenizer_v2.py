#!/usr/bin/env python3
"""
Train BPE tokenizer on 1M FineWeb-Edu documents streamed from HuggingFace.
Produces V=32K and V=8K tokenizers.
"""

import os
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_DOCS = 1_000_000
VOCAB_SIZE_32K = 32_000
VOCAB_SIZE_8K = 8_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def stream_documents(num_docs: int) -> list[str]:
    """Stream and collect text from FineWeb-Edu documents."""
    print(f"Streaming {num_docs:,} documents from HuggingFaceFW/fineweb-edu...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    texts = []
    for i, example in enumerate(ds):
        if i >= num_docs:
            break
        texts.append(example["text"])
        if (i + 1) % 100_000 == 0:
            print(f"  Collected {i + 1:,} documents...")

    print(f"Collected {len(texts):,} documents total")
    return texts


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    """Train a BPE tokenizer on the given texts."""
    print(f"\nTraining BPE tokenizer with vocab_size={vocab_size:,}...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))

    return tokenizer


def count_token_frequencies(tokenizer: Tokenizer, texts: list[str]) -> Counter:
    """Count token frequencies across the corpus."""
    print("\nCounting token frequencies for V=8K derivation...")

    freq = Counter()
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text).ids
        freq.update(ids)
        if (i + 1) % 100_000 == 0:
            print(f"  Processed {i + 1:,} documents...")

    print(f"Found {len(freq):,} unique tokens in corpus")
    return freq


def derive_8k_tokenizer(tokenizer_32k: Tokenizer, freq: Counter) -> Tokenizer:
    """
    Derive a V=8K tokenizer by keeping the 8000 most frequent tokens.
    Special tokens are always kept.
    """
    print(f"\nDeriving V={VOCAB_SIZE_8K:,} tokenizer from V={VOCAB_SIZE_32K:,}...")

    vocab_32k = tokenizer_32k.get_vocab()

    special_token_ids = set(range(len(SPECIAL_TOKENS)))

    non_special_freq = {
        token_id: count
        for token_id, count in freq.items()
        if token_id not in special_token_ids
    }

    num_non_special = VOCAB_SIZE_8K - len(SPECIAL_TOKENS)
    top_tokens = sorted(non_special_freq.items(), key=lambda x: -x[1])[:num_non_special]
    top_token_ids = {tid for tid, _ in top_tokens}

    keep_ids = special_token_ids | top_token_ids

    id_to_token = {v: k for k, v in vocab_32k.items()}

    new_vocab = {}
    old_to_new = {}

    for i, special_token in enumerate(SPECIAL_TOKENS):
        new_vocab[special_token] = i
        old_to_new[i] = i

    new_id = len(SPECIAL_TOKENS)
    for old_id in sorted(keep_ids - special_token_ids):
        token = id_to_token[old_id]
        new_vocab[token] = new_id
        old_to_new[old_id] = new_id
        new_id += 1

    merges_32k = tokenizer_32k.model.get_state()["merges"]

    new_merges = []
    for merge in merges_32k:
        parts = merge.split(" ")
        if len(parts) != 2:
            continue
        part_a, part_b = parts
        merged = part_a + part_b

        if part_a in new_vocab and part_b in new_vocab and merged in new_vocab:
            new_merges.append(merge)

    tokenizer_8k = Tokenizer(models.BPE(
        vocab=new_vocab,
        merges=new_merges,
        unk_token="<unk>",
    ))
    tokenizer_8k.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_8k.decoder = decoders.ByteLevel()

    print(f"Derived tokenizer has {tokenizer_8k.get_vocab_size():,} tokens")
    print(f"Kept {len(new_merges):,} merges from original {len(merges_32k):,}")

    return tokenizer_8k


def verify_tokenizer(tokenizer: Tokenizer, name: str):
    """Verify tokenizer with test cases."""
    print(f"\n🔍 Verifying {name}...")

    test_cases = [
        "The President of the United States",
        "Mathematical reasoning requires careful attention to detail.",
        "Hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in test_cases:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        status = "✓" if decoded.strip() == text.strip() else "✗"
        print(f"  {status} [{len(ids):3d} tokens] {repr(text[:50])}")


def print_vocab_stats(tokenizer: Tokenizer, name: str):
    """Print vocabulary statistics."""
    vocab = tokenizer.get_vocab()

    print(f"\n📊 {name} Statistics:")
    print(f"  Vocabulary size: {len(vocab):,}")

    special = [t for t in SPECIAL_TOKENS if t in vocab]
    print(f"  Special tokens: {special}")

    single_char = sum(1 for t in vocab if len(t) == 1)
    byte_level = sum(1 for t in vocab if t.startswith("Ġ"))
    print(f"  Single-char tokens: {single_char:,}")
    print(f"  Word-start tokens (Ġ prefix): {byte_level:,}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    texts = stream_documents(NUM_DOCS)

    tokenizer_32k = train_bpe_tokenizer(texts, VOCAB_SIZE_32K)

    path_32k = RESULTS_DIR / "fineweb_v32k_v2_tokenizer.json"
    tokenizer_32k.save(str(path_32k))
    print(f"\n💾 Saved V=32K tokenizer to: {path_32k}")

    freq = count_token_frequencies(tokenizer_32k, texts)

    tokenizer_8k = derive_8k_tokenizer(tokenizer_32k, freq)

    path_8k = RESULTS_DIR / "fineweb_v8k_v2_tokenizer.json"
    tokenizer_8k.save(str(path_8k))
    print(f"💾 Saved V=8K tokenizer to: {path_8k}")

    verify_tokenizer(tokenizer_32k, "V=32K")
    verify_tokenizer(tokenizer_8k, "V=8K")

    print_vocab_stats(tokenizer_32k, "V=32K")
    print_vocab_stats(tokenizer_8k, "V=8K")

    print("\n✅ Tokenizer training complete!")


if __name__ == "__main__":
    main()
