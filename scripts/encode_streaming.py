#!/usr/bin/env python3
"""
Streaming encoder for FineWeb-Edu that builds encoded datasets without downloading the raw corpus.
Works on local machines and RunPod.
"""

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from tokenizers import Tokenizer


EOS_TOKEN_ID = 3
CHECKPOINT_INTERVAL = 50_000
NUM_WORKERS = 4


class BPETokenizerWrapper:
    def __init__(self, tok: Tokenizer):
        self.tokenizer = tok

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


_global_tokenizer: BPETokenizerWrapper | None = None


def init_worker(tokenizer_path: str):
    """Initialize tokenizer in worker process."""
    global _global_tokenizer
    raw_tok = Tokenizer.from_file(tokenizer_path)
    _global_tokenizer = BPETokenizerWrapper(raw_tok)


def encode_document(text: str) -> list[int]:
    """Encode a single document using the global tokenizer."""
    assert _global_tokenizer is not None
    ids = _global_tokenizer.encode(text)
    ids.append(EOS_TOKEN_ID)
    return ids


def stream_documents(num_docs: int, dataset_name: str = "sample-10BT") -> Iterator[str]:
    """Stream documents from HuggingFace FineWeb-Edu."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=dataset_name,
        split="train",
        streaming=True,
    )

    for i, example in enumerate(ds):
        if i >= num_docs:
            break
        yield example["text"]


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def save_checkpoint(
    checkpoint_path: Path,
    all_tokens: list[int],
    docs_processed: int,
    seqs_collected: int,
):
    """Save checkpoint for resuming."""
    checkpoint = {
        "tokens": all_tokens,
        "docs_processed": docs_processed,
        "seqs_collected": seqs_collected,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path) -> tuple[list[int], int, int] | None:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None

    print(f"📂 Found checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return (
        checkpoint["tokens"],
        checkpoint["docs_processed"],
        checkpoint["seqs_collected"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Streaming encoder for FineWeb-Edu datasets"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for .pt file",
    )
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=500_000,
        help="Maximum sequences to collect (default: 500,000)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.044,
        help="Validation fraction (default: 0.044 for ~95/5 split)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=2_000_000,
        help="Number of documents to stream through (default: 2,000,000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample-10BT",
        help="Dataset variant: 'sample-10BT' or 'CC-MAIN-2024-10' (default: sample-10BT)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint")

    raw_tok = Tokenizer.from_file(str(tokenizer_path))
    tokenizer = BPETokenizerWrapper(raw_tok)
    print(f"📖 Loaded tokenizer: {tokenizer.vocab_size():,} vocab")
    print(f"📊 Config: seq_len={args.seq_len}, max_seqs={args.max_seqs:,}, val_frac={args.val_frac}")

    all_tokens: list[int] = []
    docs_processed = 0
    seqs_collected = 0

    if args.resume:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data:
            all_tokens, docs_processed, seqs_collected = checkpoint_data
            print(f"🔄 Resuming from: {docs_processed:,} docs, {seqs_collected:,} seqs")

    start_time = time.time()
    last_checkpoint_seqs = seqs_collected

    print(f"\n🚀 Starting encoding (streaming {args.num_docs:,} docs, {NUM_WORKERS} workers)...")

    doc_batch: list[str] = []
    batch_size = NUM_WORKERS * 100

    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        doc_iter = stream_documents(args.num_docs - docs_processed, args.dataset)

        if docs_processed > 0:
            print(f"⏭️  Skipping first {docs_processed:,} documents...")

        for text in doc_iter:
            doc_batch.append(text)

            if len(doc_batch) >= batch_size:
                encoded_docs = pool.map(encode_document, doc_batch)

                for ids in encoded_docs:
                    all_tokens.extend(ids)
                    docs_processed += 1

                seqs_collected = len(all_tokens) // args.seq_len
                doc_batch = []

                if seqs_collected >= args.max_seqs:
                    break

                if seqs_collected - last_checkpoint_seqs >= CHECKPOINT_INTERVAL:
                    elapsed = time.time() - start_time
                    seqs_per_sec = seqs_collected / elapsed if elapsed > 0 else 0
                    remaining_seqs = args.max_seqs - seqs_collected
                    eta = remaining_seqs / seqs_per_sec if seqs_per_sec > 0 else 0

                    token_bytes = len(all_tokens) * 8
                    print(
                        f"  📈 {seqs_collected:,}/{args.max_seqs:,} seqs | "
                        f"{docs_processed:,} docs | "
                        f"{len(all_tokens):,} tokens | "
                        f"{format_size(token_bytes)} | "
                        f"ETA: {format_time(eta)}"
                    )

                    save_checkpoint(checkpoint_path, all_tokens, docs_processed, seqs_collected)
                    last_checkpoint_seqs = seqs_collected

        if doc_batch:
            encoded_docs = pool.map(encode_document, doc_batch)
            for ids in encoded_docs:
                all_tokens.extend(ids)
                docs_processed += 1

    seqs_collected = len(all_tokens) // args.seq_len
    n_total = seqs_collected * args.seq_len
    all_tokens = all_tokens[:n_total]

    n_val = int(seqs_collected * args.val_frac)
    n_train = seqs_collected - n_val

    print(f"\n📊 Final split: {n_train:,} train + {n_val:,} val = {seqs_collected:,} seqs")

    data = torch.tensor(all_tokens, dtype=torch.long).view(-1, args.seq_len)

    train_data = data[:n_train]
    val_data = data[n_train:]

    output_dict = {
        "train": train_data,
        "val": val_data,
    }

    torch.save(output_dict, output_path)

    file_size = output_path.stat().st_size
    elapsed = time.time() - start_time

    print(f"\n✅ Encoding complete!")
    print(f"  📁 Output: {output_path}")
    print(f"  📊 Train: {train_data.shape}")
    print(f"  📊 Val: {val_data.shape}")
    print(f"  💾 Size: {format_size(file_size)}")
    print(f"  ⏱️  Time: {format_time(elapsed)}")
    print(f"  📈 Docs processed: {docs_processed:,}")
    print(f"  📈 Tokens: {len(all_tokens):,}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  🗑️  Removed checkpoint")


if __name__ == "__main__":
    main()
