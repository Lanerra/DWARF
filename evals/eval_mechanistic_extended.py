#!/usr/bin/env python3
"""
🔬 Extended Mechanistic Validation Script for DWARF Gen5 L=8 preIF

Three experiments on the borg_gen5_L8_preIF checkpoint:
  1. Multi-key scaling curve — map FA retrieval bandwidth vs key count/distance
  2. Offset-set control — test whether coprime offset geometry is load-bearing
  3. Post-FA linear probe — detect relay representational bump with linear probe

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 evals/eval_mechanistic_extended.py

Results: evals/logs/eval_mechanistic_extended_gen5.json
"""

import sys
import os
import importlib.util
import contextlib
import json
import subprocess
import time
import random
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kernels"))

CKPT = os.path.join(ROOT, "autoresearch/checkpoints/borg_gen5_L8_preIF_best.pt")
SCRIPT = os.path.join(ROOT, "train/train_borg_gen5_L8_preIF_bf16.py")
TOK_PATH = os.path.join(ROOT, "results/2048_condI_tokenizer.json")
OUT_FILE = os.path.join(ROOT, "evals/logs/eval_mechanistic_extended_gen5.json")

MAX_SEQ_LEN = 2048
D_MODEL = 512
NUM_LAYERS = 8
FULL_ATTN_LAYER = 2
NUM_TRIALS = 20

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

WORDS = ["apple", "banana", "orange", "cherry", "lemon", "plum"]
NUMBERS = ["seven", "three", "nine", "four", "eight", "two"]
COLORS = ["red", "blue", "green", "yellow", "purple", "white"]
SHAPES = ["circle", "square", "triangle", "star", "diamond", "oval"]

FILLER = "the weather was mild and the air was still . "


def load_model(device: str):
    """Load model from training script."""
    spec = importlib.util.spec_from_file_location("gen5", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.AutoresearchTransformerPhysics(
        vocab_size=32000, embedding_dim=512, num_layers=8, num_heads=8,
        ffn_dim=2048, seq_len=2048, full_attn_layer=2,
        interference_interval=2, scale_embed_init_val=0.1,
    )
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("._orig_mod", "").replace("_orig_mod.", ""): v
             for k, v in state.items() if "causal_mask" not in k}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_tokenizer():
    """Load BPE tokenizer."""
    from tokenizers import Tokenizer

    class BPETokenizerWrapper:
        def __init__(self, tok):
            self.tokenizer = tok

        def encode(self, text: str) -> List[int]:
            return self.tokenizer.encode(text).ids

        def decode(self, ids: List[int]) -> str:
            return self.tokenizer.decode(ids)

        def vocab_size(self) -> int:
            return self.tokenizer.get_vocab_size()

    return BPETokenizerWrapper(Tokenizer.from_file(TOK_PATH))


@contextlib.contextmanager
def zero_offset_rows(model, offset_indices: List[int]):
    """Temporarily zero pos_bias rows at specified indices across all DSQG layers."""
    saved = {}
    for li, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "pos_bias"):
            continue
        saved[li] = attn.pos_bias.data[offset_indices].clone()
        attn.pos_bias.data[offset_indices] = 0.0
    try:
        yield
    finally:
        for li, block in enumerate(model.blocks):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "pos_bias"):
                continue
            if li in saved:
                attn.pos_bias.data[offset_indices] = saved[li]


def experiment_1_multikey_scaling(device: str) -> Dict[str, Any]:
    """
    Experiment 1: Multi-key scaling curve

    Map FA retrieval bandwidth as a function of key count, distance, and similarity.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Multi-key Scaling Curve")
    print("=" * 70)

    tokenizer = load_tokenizer()
    model = load_model(device)
    filler_ids = tokenizer.encode(FILLER)

    results = {}

    def build_filler(length: int) -> List[int]:
        filler = []
        while len(filler) < length:
            filler.extend(filler_ids)
        return filler[:length]

    @torch.no_grad()
    def test_single_key(target: str, distance: int, intro_template: str,
                        cue_template: str, candidates: List[str]) -> bool:
        """Single key passkey trial."""
        intro_ids = tokenizer.encode(intro_template.format(key=target))
        cue_ids = tokenizer.encode(cue_template)
        filler = build_filler(distance)

        full_seq = intro_ids + filler + cue_ids
        if len(full_seq) >= MAX_SEQ_LEN:
            return None

        ids = torch.tensor([full_seq], dtype=torch.long, device=device)
        logits = model(ids)[:, -1, :]

        others = [c for c in candidates if c != target][:9]
        cand_list = [target] + others
        cand_ids = []
        for c in cand_list:
            enc = tokenizer.encode(" " + c) or tokenizer.encode(c)
            cand_ids.append(enc[0] if enc else 0)

        pred_idx = logits[0][cand_ids].argmax().item()
        return cand_list[pred_idx] == target

    @torch.no_grad()
    def test_multikey(keys: List[Tuple[str, str, str, int, List[str]]],
                      n_trials: int = NUM_TRIALS) -> Dict[str, Any]:
        """
        Test multiple keys planted in sequence.
        keys: list of (target, intro_template, cue_template, distance, candidates)
        Returns per-key accuracy and joint accuracy.
        """
        key_correct = {i: 0 for i in range(len(keys))}
        joint_correct = 0
        valid = 0

        for trial in range(n_trials):
            base_seq = []
            targets_info = []

            for key_idx, (candidates, intro_tmpl, cue_tmpl, dist, cand_list) in enumerate(keys):
                target = candidates[trial % len(candidates)]
                intro_ids = tokenizer.encode(intro_tmpl.format(key=target))
                filler = build_filler(dist)
                base_seq.extend(intro_ids)
                base_seq.extend(filler)
                targets_info.append((target, cue_tmpl, cand_list))

            if len(base_seq) + 10 >= MAX_SEQ_LEN:
                continue

            all_correct = True
            for key_idx, (target, cue_tmpl, cand_list) in enumerate(targets_info):
                cue_ids = tokenizer.encode(cue_tmpl)
                full_seq = base_seq + cue_ids
                if len(full_seq) >= MAX_SEQ_LEN:
                    continue

                ids = torch.tensor([full_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]

                others = [c for c in cand_list if c != target][:9]
                cl = [target] + others
                cids = []
                for c in cl:
                    enc = tokenizer.encode(" " + c) or tokenizer.encode(c)
                    cids.append(enc[0] if enc else 0)

                pred_idx = logits[0][cids].argmax().item()
                if cl[pred_idx] == target:
                    key_correct[key_idx] += 1
                else:
                    all_correct = False

            if all_correct:
                joint_correct += 1
            valid += 1

        return {
            "per_key_accuracy": {f"key_{i}": key_correct[i] / valid if valid > 0 else 0.0
                                 for i in range(len(keys))},
            "joint_accuracy": joint_correct / valid if valid > 0 else 0.0,
            "trials": valid,
        }

    print("\n  [n_keys=1] Single key baseline...")
    single_key_results = {}
    for d in [64, 256, 512]:
        correct, valid = 0, 0
        for i in range(NUM_TRIALS):
            target = WORDS[i % len(WORDS)]
            result = test_single_key(
                target, d,
                "the secret word is {key} .",
                "the secret word is",
                WORDS
            )
            if result is not None:
                valid += 1
                correct += int(result)
        single_key_results[f"d={d}"] = correct / valid if valid > 0 else 0.0
    results["n_keys_1_baseline"] = single_key_results
    print(f"    n_keys=1: {single_key_results}")

    print("\n  [n_keys=2] word+number, symmetric d=[256,512]...")
    r = test_multikey([
        (WORDS, "the secret word is {key} .", "the secret word is", 256, WORDS),
        (NUMBERS, "the magic number is {key} .", "the magic number is", 512, NUMBERS),
    ])
    results["n_keys_2_symmetric_256_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    print("\n  [n_keys=2] word+number, asymmetric d=[128,512]...")
    r = test_multikey([
        (WORDS, "the secret word is {key} .", "the secret word is", 128, WORDS),
        (NUMBERS, "the magic number is {key} .", "the magic number is", 512, NUMBERS),
    ])
    results["n_keys_2_asymmetric_128_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    print("\n  [n_keys=2] word+number, both far d=[512,512]...")
    r = test_multikey([
        (WORDS, "the secret word is {key} .", "the secret word is", 512, WORDS),
        (NUMBERS, "the magic number is {key} .", "the magic number is", 512, NUMBERS),
    ])
    results["n_keys_2_both_far_512_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    print("\n  [n_keys=3] word+number+color, d=[128,256,512]...")
    r = test_multikey([
        (WORDS, "the secret word is {key} .", "the secret word is", 128, WORDS),
        (NUMBERS, "the magic number is {key} .", "the magic number is", 256, NUMBERS),
        (COLORS, "the lucky color is {key} .", "the lucky color is", 512, COLORS),
    ])
    results["n_keys_3_d_128_256_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    print("\n  [n_keys=4] word+number+color+shape, d=[64,128,256,512]...")
    r = test_multikey([
        (WORDS, "the secret word is {key} .", "the secret word is", 64, WORDS),
        (NUMBERS, "the magic number is {key} .", "the magic number is", 128, NUMBERS),
        (COLORS, "the lucky color is {key} .", "the lucky color is", 256, COLORS),
        (SHAPES, "the lucky shape is {key} .", "the lucky shape is", 512, SHAPES),
    ])
    results["n_keys_4_d_64_128_256_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    print("\n  [n_keys=2] SIMILAR keys: word1+word2 (both from WORDS), d=[256,512]...")
    r = test_multikey([
        (WORDS[:3], "the secret word is {key} .", "the secret word is", 256, WORDS[:3]),
        (WORDS[3:], "the hidden word is {key} .", "the hidden word is", 512, WORDS[3:]),
    ])
    results["n_keys_2_similar_words_256_512"] = r
    print(f"    per_key: {r['per_key_accuracy']}, joint: {r['joint_accuracy']:.2f}")

    del model
    torch.cuda.empty_cache()

    return results


def experiment_2_offset_control(device: str) -> Dict[str, Any]:
    """
    Experiment 2: Offset-set control (relay geometry is load-bearing)

    Test whether the specific coprime offset geometry matters by zeroing
    subsets of pos_bias rows.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Offset-set Control (Relay Geometry)")
    print("=" * 70)

    tokenizer = load_tokenizer()
    filler_ids = tokenizer.encode(FILLER)

    ablation_configs = {
        "no_medium_range": list(range(10, 15)),
        "no_long_range": list(range(18, 24)),
        "gcd_redundant": [16, 18, 19, 20],
        "keep_only_dense": list(range(14, 24)),
    }

    test_distances = [64, 256, 512, 1024, 1536]
    results = {}

    def build_filler(length: int) -> List[int]:
        filler = []
        while len(filler) < length:
            filler.extend(filler_ids)
        return filler[:length]

    @torch.no_grad()
    def run_passkey_suite(model, distances: List[int], n_trials: int = NUM_TRIALS) -> Dict[int, float]:
        """Run passkey evaluation at multiple distances."""
        results_inner = {}
        for d in distances:
            correct, valid = 0, 0
            for i in range(n_trials):
                target = WORDS[i % len(WORDS)]
                intro_ids = tokenizer.encode(f"the secret word is {target} .")
                cue_ids = tokenizer.encode("the secret word is")
                filler = build_filler(d)

                full_seq = intro_ids + filler + cue_ids
                if len(full_seq) >= MAX_SEQ_LEN:
                    continue

                ids = torch.tensor([full_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]

                others = [w for w in WORDS if w != target]
                cand_list = [target] + others[:5]
                cand_ids = []
                for w in cand_list:
                    enc = tokenizer.encode(" " + w) or tokenizer.encode(w)
                    cand_ids.append(enc[0] if enc else 0)

                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == target:
                    correct += 1
                valid += 1
            results_inner[d] = correct / valid if valid > 0 else 0.0
        return results_inner

    print("\n  [baseline] Loading fresh model...")
    model = load_model(device)
    baseline = run_passkey_suite(model, test_distances)
    results["baseline"] = baseline
    print(f"    baseline: {baseline}")

    for ablation_name, indices in ablation_configs.items():
        offset_names = [OFFSETS[i] for i in indices if i < len(OFFSETS)]
        print(f"\n  [{ablation_name}] Zeroing offsets {offset_names} (indices {indices})...")

        with zero_offset_rows(model, indices):
            ablated = run_passkey_suite(model, test_distances)

        results[ablation_name] = {
            "accuracies": ablated,
            "zeroed_offsets": offset_names,
            "zeroed_indices": indices,
        }

        drops = {d: baseline[d] - ablated[d] for d in test_distances}
        print(f"    accuracies: {ablated}")
        print(f"    drops from baseline: {drops}")

    del model
    torch.cuda.empty_cache()

    print("\n  Summary of offset ablation effects:")
    print(f"  {'Ablation':<20} | " + " | ".join(f"d={d:4d}" for d in test_distances))
    print(f"  {'-'*20}-+-" + "-+-".join(["-" * 7 for _ in test_distances]))
    print(f"  {'baseline':<20} | " + " | ".join(f"{baseline[d]*100:5.1f}%" for d in test_distances))
    for ablation_name in ablation_configs.keys():
        accs = results[ablation_name]["accuracies"]
        print(f"  {ablation_name:<20} | " + " | ".join(f"{accs[d]*100:5.1f}%" for d in test_distances))

    return results


def experiment_3_linear_probe(device: str) -> Dict[str, Any]:
    """
    Experiment 3: Post-FA linear probe (relay representational bump)

    Use linear probes to detect codeword presence at various positions and layers.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Post-FA Linear Probe (Relay Representational Bump)")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression

    tokenizer = load_tokenizer()
    model = load_model(device)
    filler_ids = tokenizer.encode(FILLER)

    N_CODEWORDS = 16
    N_SEQUENCES = 10
    SEQ_LEN = 256

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    codewords = torch.randn(N_CODEWORDS, D_MODEL, device=device)
    codewords = codewords / codewords.norm(dim=-1, keepdim=True)

    def build_filler_seq(length: int) -> List[int]:
        seq = []
        while len(seq) < length:
            seq.extend(filler_ids)
        return seq[:length]

    results = {}

    print("\n  Generating sequences and collecting hidden states...")

    @torch.no_grad()
    def collect_features(inject_pos: int, measure_positions: List[int]
                        ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Collect (hidden_state, label) pairs for linear probe training.
        Returns: {f"L{layer}": {pos: (X, y)}} where X is [n_samples, D], y is [n_samples]
        """
        features = {f"L{li}": {pos: ([], []) for pos in measure_positions}
                    for li in range(NUM_LAYERS)}

        for cw_idx in range(N_CODEWORDS):
            codeword = codewords[cw_idx]

            for seq_idx in range(N_SEQUENCES):
                filler_seq = build_filler_seq(SEQ_LEN)
                input_ids = torch.tensor([filler_seq], dtype=torch.long, device=device)

                layer_outputs = {}

                def make_hook(layer_idx):
                    def hook(module, inp, out):
                        layer_outputs[layer_idx] = out.detach().cpu()
                    return hook

                handles = [model.blocks[li].register_forward_hook(make_hook(li))
                           for li in range(NUM_LAYERS)]

                B, N = input_ids.shape
                pos_ids = torch.arange(N, device=device).unsqueeze(0)
                emb_output = model.drop(model.embedding(input_ids) + model.pos_embed(pos_ids))
                emb_injected = emb_output.clone()
                emb_injected[0, inject_pos] = emb_injected[0, inject_pos] + codeword

                x_pos = emb_injected
                for block in model.blocks:
                    x_pos = block(x_pos)

                for h in handles:
                    h.remove()

                for li in range(NUM_LAYERS):
                    hidden = layer_outputs[li]
                    for pos in measure_positions:
                        if pos < hidden.shape[1]:
                            h_vec = hidden[0, pos].numpy().astype(np.float32)
                            features[f"L{li}"][pos][0].append(h_vec)
                            features[f"L{li}"][pos][1].append(1)

                layer_outputs.clear()
                handles = [model.blocks[li].register_forward_hook(make_hook(li))
                           for li in range(NUM_LAYERS)]

                x_neg = emb_output
                for block in model.blocks:
                    x_neg = block(x_neg)

                for h in handles:
                    h.remove()

                for li in range(NUM_LAYERS):
                    hidden = layer_outputs[li]
                    for pos in measure_positions:
                        if pos < hidden.shape[1]:
                            h_vec = hidden[0, pos].numpy().astype(np.float32)
                            features[f"L{li}"][pos][0].append(h_vec)
                            features[f"L{li}"][pos][1].append(0)

        for li in range(NUM_LAYERS):
            for pos in measure_positions:
                X_list, y_list = features[f"L{li}"][pos]
                features[f"L{li}"][pos] = (np.array(X_list), np.array(y_list))

        return features

    def train_and_eval_probe(features: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
                             measure_positions: List[int]) -> Dict[str, Dict[int, float]]:
        """Train linear probes and return per-layer, per-position accuracy."""
        probe_results = {f"L{li}": {} for li in range(NUM_LAYERS)}

        for li in range(NUM_LAYERS):
            for pos in measure_positions:
                X, y = features[f"L{li}"][pos]
                if len(X) < 10:
                    probe_results[f"L{li}"][pos] = 0.5
                    continue

                n_train = int(0.8 * len(X))
                indices = np.random.permutation(len(X))
                train_idx, test_idx = indices[:n_train], indices[n_train:]

                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    probe_results[f"L{li}"][pos] = 0.5
                    continue

                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
                probe_results[f"L{li}"][pos] = round(acc, 4)

        return probe_results

    print("\n  [inject_pos=0] Collecting features at positions [0, 8, 16, 32, 64, 128, 255]...")
    measure_positions_0 = [0, 8, 16, 32, 64, 128, 255]
    features_0 = collect_features(inject_pos=0, measure_positions=measure_positions_0)

    print("    Training probes...")
    probe_results_0 = train_and_eval_probe(features_0, measure_positions_0)
    results["inject_pos_0"] = probe_results_0

    print("\n  Probe accuracy (inject_pos=0):")
    print(f"  {'Layer':<8} | " + " | ".join(f"pos={p:3d}" for p in measure_positions_0))
    print(f"  {'-'*8}-+-" + "-+-".join(["-" * 8 for _ in measure_positions_0]))
    for li in range(NUM_LAYERS):
        row = f"  L{li:<6} | "
        row += " | ".join(f"{probe_results_0[f'L{li}'][p]*100:6.1f}%" for p in measure_positions_0)
        print(row)

    print("\n  [inject_pos=64] Collecting features at positions [64, 72, 80, 96, 128]...")
    measure_positions_64 = [64, 72, 80, 96, 128]
    features_64 = collect_features(inject_pos=64, measure_positions=measure_positions_64)

    print("    Training probes...")
    probe_results_64 = train_and_eval_probe(features_64, measure_positions_64)
    results["inject_pos_64"] = probe_results_64

    print("\n  Probe accuracy (inject_pos=64):")
    print(f"  {'Layer':<8} | " + " | ".join(f"pos={p:3d}" for p in measure_positions_64))
    print(f"  {'-'*8}-+-" + "-+-".join(["-" * 8 for _ in measure_positions_64]))
    for li in range(NUM_LAYERS):
        row = f"  L{li:<6} | "
        row += " | ".join(f"{probe_results_64[f'L{li}'][p]*100:6.1f}%" for p in measure_positions_64)
        print(row)

    print("\n  Interpretation:")
    pre_fa_layers = list(range(FULL_ATTN_LAYER))
    post_fa_layers = list(range(FULL_ATTN_LAYER, NUM_LAYERS))

    for inject_pos, positions, probe_res in [
        (0, measure_positions_0, probe_results_0),
        (64, measure_positions_64, probe_results_64),
    ]:
        downstream_positions = [p for p in positions if p > inject_pos]
        if not downstream_positions:
            continue

        print(f"\n    inject_pos={inject_pos}:")

        for pos in downstream_positions:
            pre_fa_accs = [probe_res[f"L{li}"].get(pos, 0.5) for li in pre_fa_layers]
            post_fa_accs = [probe_res[f"L{li}"].get(pos, 0.5) for li in post_fa_layers]

            pre_fa_mean = sum(pre_fa_accs) / len(pre_fa_accs) if pre_fa_accs else 0.5
            post_fa_mean = sum(post_fa_accs) / len(post_fa_accs) if post_fa_accs else 0.5
            bump = post_fa_mean - pre_fa_mean

            interpretation = "RELAY_BUMP" if bump > 0.05 else ("NO_BUMP" if bump < 0.01 else "MARGINAL")
            print(f"      pos={pos}: pre-FA mean={pre_fa_mean*100:.1f}%, post-FA mean={post_fa_mean*100:.1f}%, bump={bump*100:+.1f}% [{interpretation}]")

    del model
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("  🔬 Extended Mechanistic Validation: DWARF Gen5 L=8 preIF")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Output: {OUT_FILE}")

    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    all_results = {
        "model": "borg_gen5_L8_preIF",
        "checkpoint": CKPT,
        "full_attn_layer": FULL_ATTN_LAYER,
        "offsets": OFFSETS,
    }

    t0 = time.time()

    all_results["exp1_multikey_scaling"] = experiment_1_multikey_scaling(device)
    all_results["exp2_offset_control"] = experiment_2_offset_control(device)
    all_results["exp3_linear_probe"] = experiment_3_linear_probe(device)

    all_results["elapsed_s"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {OUT_FILE}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    exp1 = all_results["exp1_multikey_scaling"]
    print("\n  Exp 1 (Multi-key Scaling):")
    print(f"    Single key baseline: {exp1.get('n_keys_1_baseline', {})}")
    for key in ["n_keys_2_symmetric_256_512", "n_keys_3_d_128_256_512", "n_keys_4_d_64_128_256_512"]:
        if key in exp1:
            joint = exp1[key].get("joint_accuracy", 0)
            print(f"    {key}: joint={joint*100:.0f}%")

    exp2 = all_results["exp2_offset_control"]
    print("\n  Exp 2 (Offset Control) - Mean drop from baseline:")
    baseline = exp2.get("baseline", {})
    baseline_mean = sum(baseline.values()) / len(baseline) if baseline else 0
    for ablation in ["no_medium_range", "no_long_range", "gcd_redundant", "keep_only_dense"]:
        if ablation in exp2:
            accs = exp2[ablation].get("accuracies", {})
            if accs:
                ablation_mean = sum(accs.values()) / len(accs)
                drop = baseline_mean - ablation_mean
                print(f"    {ablation}: mean_drop={drop*100:+.1f}%")

    print(f"\n  Total elapsed: {all_results['elapsed_s']:.1f}s")

    subprocess.run(
        ["openclaw", "system", "event", "--text",
         "Done: extended mechanistic evals complete", "--mode", "now"],
        capture_output=True
    )


if __name__ == "__main__":
    main()
