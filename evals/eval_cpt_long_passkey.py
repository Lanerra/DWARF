#!/usr/bin/env python3
"""
Eval long-range passkey retrieval for the D512/L10 Muon CPT checkpoint.

This is intentionally eval-only and prefix-only. The legacy padded passkey
audit is contaminated for this model family, and kernel-compatible right
padding can inflate 16k/32k probes into much larger allocations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_cpt8192_boundary.py"
DEFAULT_CHECKPOINT = (
    ROOT
    / "autoresearch"
    / "checkpoints"
    / "cpt8192_muon_boundary"
    / "d512_l10_muon_cpt8192_boundary_new_dataset_best.pt"
)
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "mixed_tokenizer_32k.json"

PASSKEY_WORDS = [
    "apple",
    "banana",
    "orange",
    "cherry",
    "grape",
    "lemon",
    "mango",
    "peach",
    "plum",
    "berry",
]
FILLER_SENTENCE = "the weather was mild and the air was still . "
INTRO_TEMPLATE = "the secret word is {word} ."
RETRIEVAL_CUE = "the secret word is"


class BPETokenizerWrapper:
    def __init__(self, tok: Tokenizer):
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()


def load_train_module():
    # Keep imported training module in eval-friendly mode.
    os.environ.setdefault("DWARF_TORCH_COMPILE", "0")
    os.environ.setdefault("DWARF_CKPT", "none")
    os.environ.setdefault("DWARF_BS", "1")
    os.environ.setdefault("DWARF_SKIP_OPT", "1")
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    spec = importlib.util.spec_from_file_location("d512_l10_cpt", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_token_ids(tokenizer: BPETokenizerWrapper, words: list[str]) -> dict[str, int]:
    cue_ids = tokenizer.encode(RETRIEVAL_CUE)
    out: dict[str, int] = {}
    for word in words:
        cue_plus_word = tokenizer.encode(f"{RETRIEVAL_CUE} {word}")
        if len(cue_plus_word) > len(cue_ids) and cue_plus_word[: len(cue_ids)] == cue_ids:
            out[word] = cue_plus_word[len(cue_ids)]
            continue
        encoded = tokenizer.encode(" " + word) or tokenizer.encode(word)
        if not encoded:
            raise ValueError(f"Could not encode passkey word {word!r}")
        out[word] = encoded[0]
    return out


def build_sequence(tokenizer: BPETokenizerWrapper, word: str, distance: int) -> list[int]:
    intro = tokenizer.encode(INTRO_TEMPLATE.format(word=word))
    filler_ids = tokenizer.encode(FILLER_SENTENCE)
    cue = tokenizer.encode(RETRIEVAL_CUE)
    filler: list[int] = []
    while len(filler) < distance:
        filler.extend(filler_ids)
    return intro + filler[:distance] + cue


def build_sequence_for_context_len(tokenizer: BPETokenizerWrapper, word: str, context_len: int) -> tuple[list[int], int]:
    intro = tokenizer.encode(INTRO_TEMPLATE.format(word=word))
    cue = tokenizer.encode(RETRIEVAL_CUE)
    distance = context_len - len(intro) - len(cue)
    if distance < 0:
        raise ValueError(f"context_len={context_len} is too short for passkey prompt")
    return build_sequence(tokenizer, word, distance), distance


def load_model(mod, checkpoint: Path, tokenizer: BPETokenizerWrapper, device: str):
    model = mod.TriadicJ96Dsr(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=mod.EMBEDDING_DIM,
        num_heads=mod.NUM_HEADS,
        ffn_dim=mod.FFN_DIM,
        seq_len=mod.MAX_SEQ_LEN,
        dsr_layer=mod.DSR_LAYER,
        scale_embed_init_val=mod.SCALE_EMBED_INIT_VAL,
        dropout=0.0,
        num_chunks=mod.NUM_CHUNKS,
        top_k_chunks=mod.TOP_K_CHUNKS,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") or "._orig_mod." in k for k in state):
        state = {k.replace("._orig_mod", "").replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)}", flush=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def eval_target(model, tokenizer, cand_ids, target_len: int, trials: int, device: str, context_lengths: bool):
    correct = 0
    total = 0
    seq_lens: list[int] = []
    words = PASSKEY_WORDS
    distances: list[int] = []
    per_word = {word: {"correct": 0, "total": 0} for word in words}

    for trial_idx in range(trials):
        target_word = words[trial_idx % len(words)]
        candidates = [target_word] + [w for w in words if w != target_word]
        if context_lengths:
            seq, distance = build_sequence_for_context_len(tokenizer, target_word, target_len)
        else:
            distance = target_len
            seq = build_sequence(tokenizer, target_word, distance)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.no_grad():
            logits = model(ids)[0, -1, :]
        scores = logits[[cand_ids[w] for w in candidates]]
        is_correct = int(scores.argmax().item() == 0)
        correct += is_correct
        per_word[target_word]["correct"] += is_correct
        per_word[target_word]["total"] += 1
        total += 1
        seq_lens.append(len(seq))
        distances.append(distance)
        del ids, logits, scores
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "mode": "context_length" if context_lengths else "filler_distance",
        "target": target_len,
        "min_distance": min(distances) if distances else None,
        "max_distance": max(distances) if distances else None,
        "min_seq_len": min(seq_lens) if seq_lens else None,
        "max_seq_len": max(seq_lens) if seq_lens else None,
        "per_word": per_word,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--distances", default="")
    parser.add_argument("--context-lengths", default="8192,16384,32768")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cpt_long_passkey_16k_32k.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This DWARF eval needs CUDA.")
    device = "cuda"

    print("=" * 72)
    print("  D512/L10 Muon CPT long passkey probe")
    print(f"  checkpoint: {args.checkpoint}")
    mode = "filler_distance" if args.distances else "context_length"
    targets_raw = args.distances if args.distances else args.context_lengths
    print(f"  mode:       {mode}")
    print(f"  targets:    {targets_raw}")
    print(f"  trials:     {args.trials}")
    print(f"  gpu:        {torch.cuda.get_device_name(0)}")
    print("=" * 72, flush=True)

    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(str(args.tokenizer)))
    mod = load_train_module()
    model = load_model(mod, args.checkpoint, tokenizer, device)
    cand_ids = candidate_token_ids(tokenizer, PASSKEY_WORDS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    results = {}
    context_lengths = not bool(args.distances)
    for target_len in [int(x) for x in targets_raw.split(",") if x.strip()]:
        label = "context_len" if context_lengths else "distance"
        print(f"\n  {label}={target_len}", flush=True)
        t0 = time.time()
        try:
            result = eval_target(model, tokenizer, cand_ids, target_len, args.trials, device, context_lengths)
            result["elapsed_sec"] = time.time() - t0
            result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"    acc={result['accuracy']:.1%} "
                f"({result['correct']}/{result['total']}) "
                f"distance={result['min_distance']}-{result['max_distance']} "
                f"seq_len={result['min_seq_len']}-{result['max_seq_len']} "
                f"peak={result['peak_vram_gb']:.2f}GB "
                f"time={result['elapsed_sec']:.1f}s",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            result = {"error": "cuda_oom", "message": str(exc), "elapsed_sec": time.time() - t0}
            print(f"    OOM after {result['elapsed_sec']:.1f}s", flush=True)
        except Exception as exc:
            result = {"error": type(exc).__name__, "message": str(exc), "elapsed_sec": time.time() - t0}
            print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
        results[str(target_len)] = result

    payload = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "trials": args.trials,
        "mode": mode,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
