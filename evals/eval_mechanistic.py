#!/usr/bin/env python3
"""
🔬 Mechanistic Validation Script for DWARF Gen5 L=8 preIF

Runs 4 experiments on the borg_gen5_L8_preIF checkpoint:
  1. Retrieval-disabled FA control — ablate the full attention layer (L2)
  2. Dummy-slot IF ablation — ablate the interference layer (L1)
  3. Multi-key passkey — multi-key retrieval with word, number, color
  4. Causal relay tracing — track codeword propagation through layers

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 evals/eval_mechanistic.py

Results: evals/logs/eval_mechanistic_gen5_L8_preIF.json
"""

import sys
import os
import importlib.util
import contextlib
import math
import json
import subprocess
import time
import random
from typing import Dict, List, Tuple, Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kernels"))

CKPT = os.path.join(ROOT, "autoresearch/checkpoints/borg_gen5_L8_preIF_best.pt")
SCRIPT = os.path.join(ROOT, "train/train_borg_gen5_L8_preIF_bf16.py")
TOK_PATH = os.path.join(ROOT, "results/2048_condI_tokenizer.json")
FULL_ATTN_LAYER = 2
IF_LAYER = 1
OUT_FILE = os.path.join(ROOT, "evals/logs/eval_mechanistic_gen5_L8_preIF.json")

MAX_SEQ_LEN = 2048
PASSKEY_DISTANCES = [64, 256, 512, 1024, 1536]
PASSKEY_TRIALS = 20
PASSKEY_WORDS = ["apple", "banana", "orange", "cherry", "grape",
                 "lemon", "mango", "peach", "plum", "berry"]
NUMBERS = ["seven", "three", "nine", "four", "eight", "two", "six", "five"]
COLORS = ["red", "blue", "green", "yellow", "purple", "white", "black", "brown"]
FILLER = "the weather was mild and the air was still . "

INTRO_TEMPLATE = "the secret word is {word} ."
RETRIEVAL_CUE = "the secret word is"
NUMBER_INTRO = "the magic number is {number} ."
NUMBER_CUE = "the magic number is"
COLOR_INTRO = "the lucky color is {color} ."
COLOR_CUE = "the lucky color is"


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


@torch.no_grad()
def passkey_trial(
    model: nn.Module,
    tokenizer,
    device: str,
    target_word: str,
    distance: int,
    intro_template: str = INTRO_TEMPLATE,
    retrieval_cue: str = RETRIEVAL_CUE,
    all_candidates: List[str] = None,
) -> bool:
    """Run a single passkey trial and return True if model predicts correctly."""
    if all_candidates is None:
        all_candidates = PASSKEY_WORDS

    filler_ids = tokenizer.encode(FILLER)
    cue_ids = tokenizer.encode(retrieval_cue)
    intro_ids = tokenizer.encode(intro_template.format(word=target_word))

    available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
    if distance > available:
        return None

    filler = []
    while len(filler) < distance:
        filler.extend(filler_ids)
    full_seq = intro_ids + filler[:distance] + cue_ids

    if len(full_seq) >= MAX_SEQ_LEN:
        return None

    ids = torch.tensor([full_seq], dtype=torch.long, device=device)
    logits = model(ids)[:, -1, :]

    others = [w for w in all_candidates if w != target_word][:9]
    cand_list = [target_word] + others
    cand_ids = []
    for w in cand_list:
        enc = tokenizer.encode(" " + w) or tokenizer.encode(w)
        cand_ids.append(enc[0] if enc else 0)

    pred_idx = logits[0][cand_ids].argmax().item()
    return cand_list[pred_idx] == target_word


@torch.no_grad()
def run_passkey_suite(
    model: nn.Module,
    tokenizer,
    device: str,
    distances: List[int] = None,
    n_trials: int = PASSKEY_TRIALS,
) -> Dict[int, float]:
    """Run passkey evaluation at multiple distances."""
    if distances is None:
        distances = PASSKEY_DISTANCES
    results = {}
    for d in distances:
        correct, valid = 0, 0
        for i in range(n_trials):
            target = PASSKEY_WORDS[i % len(PASSKEY_WORDS)]
            result = passkey_trial(model, tokenizer, device, target, d)
            if result is not None:
                valid += 1
                correct += int(result)
        results[d] = correct / valid if valid > 0 else 0.0
    return results


class HookManager:
    """Context manager for PyTorch hooks."""

    def __init__(self):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, handle: torch.utils.hooks.RemovableHandle):
        self.handles.append(handle)

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear()


def experiment_1_fa_control(device: str) -> Dict[str, Any]:
    """
    Experiment 1: Retrieval-disabled FA control

    Test passkey at various distances under 4 conditions:
    - baseline: normal model
    - random_K: replace K in FA qkv_proj with random values (same norm)
    - zero_FA_output: FA block returns zeros
    - local_FA: zero attention logits where |q_pos - k_pos| > 64
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Retrieval-disabled FA control")
    print("=" * 60)

    tokenizer = load_tokenizer()
    results = {}

    print("\n  [baseline] Loading fresh model...")
    model = load_model(device)
    results["baseline"] = run_passkey_suite(model, tokenizer, device)
    print(f"    baseline: {results['baseline']}")
    del model
    torch.cuda.empty_cache()

    print("\n  [random_K] Replacing K slice with random values...")
    model = load_model(device)
    fa_block = model.blocks[FULL_ATTN_LAYER]

    def hook_random_k(module, args, output):
        B, N, D3 = output.shape
        D = D3 // 3
        q, k, v = output.split(D, dim=-1)
        k_norm = k.norm(dim=-1, keepdim=True)
        k_rand = torch.randn_like(k)
        k_rand = k_rand / (k_rand.norm(dim=-1, keepdim=True) + 1e-8) * k_norm
        return torch.cat([q, k_rand, v], dim=-1)

    with HookManager() as hm:
        hm.register(fa_block.attn.qkv_proj.register_forward_hook(hook_random_k))
        results["random_K"] = run_passkey_suite(model, tokenizer, device)
    print(f"    random_K: {results['random_K']}")
    del model
    torch.cuda.empty_cache()

    print("\n  [zero_FA_output] FA block returns zeros...")
    model = load_model(device)
    fa_block = model.blocks[FULL_ATTN_LAYER]
    original_forward = fa_block.forward

    def zero_forward(x):
        return torch.zeros_like(x)

    fa_block.forward = zero_forward
    results["zero_FA_output"] = run_passkey_suite(model, tokenizer, device)
    fa_block.forward = original_forward
    print(f"    zero_FA_output: {results['zero_FA_output']}")
    del model
    torch.cuda.empty_cache()

    print("\n  [local_FA] Zero attention where |q_pos - k_pos| > 64...")
    model = load_model(device)
    fa_block = model.blocks[FULL_ATTN_LAYER]
    original_attn_forward = fa_block.attn.forward

    def local_attn_forward(x):
        B, N, D = x.shape
        H = fa_block.attn.num_heads
        HD = fa_block.attn.head_dim
        qkv = fa_block.attn.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scale = HD ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        pos_q = torch.arange(N, device=x.device).unsqueeze(1)
        pos_k = torch.arange(N, device=x.device).unsqueeze(0)
        local_mask = (pos_q - pos_k).abs() > 64
        attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(fa_block.attn.gate_proj(x))
        return fa_block.attn.out_proj(out * gate)

    fa_block.attn.forward = local_attn_forward
    results["local_FA"] = run_passkey_suite(model, tokenizer, device)
    fa_block.attn.forward = original_attn_forward
    print(f"    local_FA: {results['local_FA']}")
    del model
    torch.cuda.empty_cache()

    return results


def experiment_2_if_ablation(device: str) -> Dict[str, Any]:
    """
    Experiment 2: Dummy-slot IF ablation

    Test passkey at various distances under 4 conditions:
    - baseline: normal model
    - zero_IF_weights: zero all IF params (inter_norm, inter_gate, inter_k_proj, inter_v_proj)
    - identity_slot: hook inter_gate to return zeros (gates off EMA injection)
    - remove_IF_residual: IF block returns zeros
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Dummy-slot IF ablation")
    print("=" * 60)

    tokenizer = load_tokenizer()
    results = {}

    print("\n  [baseline] Loading fresh model...")
    model = load_model(device)
    results["baseline"] = run_passkey_suite(model, tokenizer, device)
    print(f"    baseline: {results['baseline']}")
    del model
    torch.cuda.empty_cache()

    print("\n  [zero_IF_weights] Zero all IF params...")
    model = load_model(device)
    if_block = model.blocks[IF_LAYER]

    saved_params = {}
    if hasattr(if_block, "inter_norm"):
        for name in ["inter_norm", "inter_gate", "inter_k_proj", "inter_v_proj"]:
            if hasattr(if_block, name):
                module = getattr(if_block, name)
                saved_params[name] = {n: p.clone() for n, p in module.named_parameters()}
                for p in module.parameters():
                    p.data.zero_()

    results["zero_IF_weights"] = run_passkey_suite(model, tokenizer, device)
    print(f"    zero_IF_weights: {results['zero_IF_weights']}")

    for name, params_dict in saved_params.items():
        module = getattr(if_block, name)
        state = module.state_dict()
        for n, p in params_dict.items():
            state[n] = p
        module.load_state_dict(state)

    del model
    torch.cuda.empty_cache()

    print("\n  [identity_slot] inter_gate returns zeros (EMA off)...")
    model = load_model(device)
    if_block = model.blocks[IF_LAYER]

    def hook_zero_gate(module, args, output):
        return torch.zeros_like(output)

    with HookManager() as hm:
        if hasattr(if_block, "inter_gate"):
            hm.register(if_block.inter_gate.register_forward_hook(hook_zero_gate))
        results["identity_slot"] = run_passkey_suite(model, tokenizer, device)
    print(f"    identity_slot: {results['identity_slot']}")
    del model
    torch.cuda.empty_cache()

    print("\n  [remove_IF_residual] IF block returns zeros...")
    model = load_model(device)
    if_block = model.blocks[IF_LAYER]
    original_forward = if_block.forward

    def zero_forward(x):
        return torch.zeros_like(x)

    if_block.forward = zero_forward
    results["remove_IF_residual"] = run_passkey_suite(model, tokenizer, device)
    if_block.forward = original_forward
    print(f"    remove_IF_residual: {results['remove_IF_residual']}")
    del model
    torch.cuda.empty_cache()

    return results


def experiment_3_multikey(device: str) -> Dict[str, Any]:
    """
    Experiment 3: Multi-key passkey

    Plant multiple key-value pairs, retrieve each at the end.
    Configs:
    - word_only_d256_d512: word at d=256, number at d=512
    - word_only_d512_d512: both at d=512
    - triple_d128_d256_d512: word + number + color
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Multi-key passkey")
    print("=" * 60)

    tokenizer = load_tokenizer()
    model = load_model(device)
    results = {}

    filler_ids = tokenizer.encode(FILLER)

    @torch.no_grad()
    def test_multikey_word_number(d1: int, d2: int, n_trials: int = PASSKEY_TRIALS) -> Dict[str, float]:
        word_correct, number_correct, valid = 0, 0, 0
        for i in range(n_trials):
            word = PASSKEY_WORDS[i % len(PASSKEY_WORDS)]
            number = NUMBERS[i % len(NUMBERS)]

            word_intro_ids = tokenizer.encode(f"the secret word is {word} . ")
            number_intro_ids = tokenizer.encode(f"the magic number is {number} . ")

            filler1 = []
            while len(filler1) < d1:
                filler1.extend(filler_ids)
            filler1 = filler1[:d1]

            filler2 = []
            while len(filler2) < d2:
                filler2.extend(filler_ids)
            filler2 = filler2[:d2]

            base_seq = word_intro_ids + filler1 + number_intro_ids + filler2

            word_cue_ids = tokenizer.encode("the secret word is")
            word_seq = base_seq + word_cue_ids
            if len(word_seq) < MAX_SEQ_LEN:
                ids = torch.tensor([word_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                others = [w for w in PASSKEY_WORDS if w != word][:9]
                cand_list = [word] + others
                cand_ids = []
                for w in cand_list:
                    enc = tokenizer.encode(" " + w) or tokenizer.encode(w)
                    cand_ids.append(enc[0] if enc else 0)
                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == word:
                    word_correct += 1

            number_cue_ids = tokenizer.encode("the magic number is")
            number_seq = base_seq + number_cue_ids
            if len(number_seq) < MAX_SEQ_LEN:
                ids = torch.tensor([number_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                others = [n for n in NUMBERS if n != number]
                cand_list = [number] + others[:7]
                cand_ids = []
                for n in cand_list:
                    enc = tokenizer.encode(" " + n) or tokenizer.encode(n)
                    cand_ids.append(enc[0] if enc else 0)
                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == number:
                    number_correct += 1

            valid += 1

        return {
            "word_accuracy": word_correct / valid if valid > 0 else 0.0,
            "number_accuracy": number_correct / valid if valid > 0 else 0.0,
        }

    @torch.no_grad()
    def test_triple(d1: int, d2: int, d3: int, n_trials: int = PASSKEY_TRIALS) -> Dict[str, float]:
        word_correct, number_correct, color_correct, valid = 0, 0, 0, 0
        for i in range(n_trials):
            word = PASSKEY_WORDS[i % len(PASSKEY_WORDS)]
            number = NUMBERS[i % len(NUMBERS)]
            color = COLORS[i % len(COLORS)]

            word_intro_ids = tokenizer.encode(f"the secret word is {word} . ")
            number_intro_ids = tokenizer.encode(f"the magic number is {number} . ")
            color_intro_ids = tokenizer.encode(f"the lucky color is {color} . ")

            filler1, filler2, filler3 = [], [], []
            for filler, d in [(filler1, d1), (filler2, d2), (filler3, d3)]:
                while len(filler) < d:
                    filler.extend(filler_ids)

            base_seq = (word_intro_ids + filler1[:d1] +
                       number_intro_ids + filler2[:d2] +
                       color_intro_ids + filler3[:d3])

            word_cue_ids = tokenizer.encode("the secret word is")
            word_seq = base_seq + word_cue_ids
            if len(word_seq) < MAX_SEQ_LEN:
                ids = torch.tensor([word_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                others = [w for w in PASSKEY_WORDS if w != word][:9]
                cand_list = [word] + others
                cand_ids = [tokenizer.encode(" " + w)[0] if tokenizer.encode(" " + w) else tokenizer.encode(w)[0] for w in cand_list]
                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == word:
                    word_correct += 1

            number_cue_ids = tokenizer.encode("the magic number is")
            number_seq = base_seq + number_cue_ids
            if len(number_seq) < MAX_SEQ_LEN:
                ids = torch.tensor([number_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                others = [n for n in NUMBERS if n != number][:7]
                cand_list = [number] + others
                cand_ids = [tokenizer.encode(" " + n)[0] if tokenizer.encode(" " + n) else tokenizer.encode(n)[0] for n in cand_list]
                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == number:
                    number_correct += 1

            color_cue_ids = tokenizer.encode("the lucky color is")
            color_seq = base_seq + color_cue_ids
            if len(color_seq) < MAX_SEQ_LEN:
                ids = torch.tensor([color_seq], dtype=torch.long, device=device)
                logits = model(ids)[:, -1, :]
                others = [c for c in COLORS if c != color][:7]
                cand_list = [color] + others
                cand_ids = [tokenizer.encode(" " + c)[0] if tokenizer.encode(" " + c) else tokenizer.encode(c)[0] for c in cand_list]
                pred_idx = logits[0][cand_ids].argmax().item()
                if cand_list[pred_idx] == color:
                    color_correct += 1

            valid += 1

        return {
            "word_accuracy": word_correct / valid if valid > 0 else 0.0,
            "number_accuracy": number_correct / valid if valid > 0 else 0.0,
            "color_accuracy": color_correct / valid if valid > 0 else 0.0,
        }

    print("\n  [word_only_d256_d512] word at d=256, number at d=512...")
    results["word_only_d256_d512"] = test_multikey_word_number(d1=256, d2=512)
    print(f"    word_only_d256_d512: {results['word_only_d256_d512']}")

    print("\n  [word_only_d512_d512] both at d=512...")
    results["word_only_d512_d512"] = test_multikey_word_number(d1=512, d2=512)
    print(f"    word_only_d512_d512: {results['word_only_d512_d512']}")

    print("\n  [triple_d128_d256_d512] word + number + color...")
    results["triple_d128_d256_d512"] = test_triple(d1=128, d2=256, d3=512)
    print(f"    triple_d128_d256_d512: {results['triple_d128_d256_d512']}")

    del model
    torch.cuda.empty_cache()

    return results


def experiment_4_relay_tracing(device: str) -> Dict[str, Any]:
    """
    Experiment 4: Causal relay tracing

    Inject random unit codewords at specific positions, track cosine similarity
    through layers.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Causal relay tracing")
    print("=" * 60)

    tokenizer = load_tokenizer()
    model = load_model(device)
    results = {}

    D = 512
    n_codewords = 8
    seq_len = 256
    filler_ids = tokenizer.encode(FILLER)

    codewords = torch.randn(n_codewords, D, device=device)
    codewords = codewords / codewords.norm(dim=-1, keepdim=True)

    def build_filler_seq(length: int) -> List[int]:
        seq = []
        while len(seq) < length:
            seq.extend(filler_ids)
        return seq[:length]

    @torch.no_grad()
    def trace_codeword(inject_pos: int, measure_positions: List[int]) -> Dict[str, List[float]]:
        """Inject codeword at inject_pos, measure cosine sim at measure_positions for each layer."""
        filler_seq = build_filler_seq(seq_len)
        input_ids = torch.tensor([filler_seq], dtype=torch.long, device=device)

        layer_outputs = {}

        def make_capture_hook(layer_idx):
            def hook(module, args, output):
                layer_outputs[layer_idx] = output.detach()
            return hook

        all_sims = {pos: {layer: [] for layer in range(len(model.blocks) + 1)} for pos in measure_positions}

        for cw_idx in range(n_codewords):
            codeword = codewords[cw_idx]
            layer_outputs.clear()

            handles = []

            def make_inject_hook(cw):
                def hook(module, args, output):
                    output = output.clone()
                    output[0, inject_pos] = output[0, inject_pos] + cw
                    return output

                return hook

            with torch.no_grad():
                B, N = input_ids.shape
                pos_ids = torch.arange(N, device=device).unsqueeze(0)
                emb_output = model.drop(model.embedding(input_ids) + model.pos_embed(pos_ids))

                emb_output[0, inject_pos] = emb_output[0, inject_pos] + codeword

                layer_outputs[-1] = emb_output.clone()

                x = emb_output
                for layer_idx, block in enumerate(model.blocks):
                    x = block(x)
                    layer_outputs[layer_idx] = x.clone()

            for pos in measure_positions:
                if pos < seq_len:
                    emb_hidden = layer_outputs[-1][0, pos]
                    sim = F.cosine_similarity(emb_hidden.unsqueeze(0), codeword.unsqueeze(0)).item()
                    if -1 not in all_sims[pos]:
                        all_sims[pos][-1] = []
                    all_sims[pos][-1].append(sim)

                    for layer_idx in range(len(model.blocks)):
                        hidden = layer_outputs[layer_idx][0, pos]
                        sim = F.cosine_similarity(hidden.unsqueeze(0), codeword.unsqueeze(0)).item()
                        all_sims[pos][layer_idx].append(sim)

        avg_sims = {}
        for pos in measure_positions:
            avg_sims[pos] = {}
            for layer_idx in list(range(-1, len(model.blocks))):
                if all_sims[pos][layer_idx]:
                    avg_sims[pos][f"L{layer_idx}"] = sum(all_sims[pos][layer_idx]) / len(all_sims[pos][layer_idx])

        return avg_sims

    print("\n  [inject_pos_0] Injecting at position 0, measuring at [0, 32, 64, 128, 255]...")
    positions_0 = [0, 32, 64, 128, 255]
    results["inject_pos_0"] = trace_codeword(inject_pos=0, measure_positions=positions_0)
    print(f"    inject_pos_0: done")
    for pos in positions_0:
        layers = sorted(results["inject_pos_0"][pos].keys(), key=lambda x: int(x[1:]) if x[1:].lstrip('-').isdigit() else -100)
        vals = [f"{results['inject_pos_0'][pos][l]:.3f}" for l in layers]
        print(f"      pos={pos}: {' '.join(vals)}")

    print("\n  [inject_pos_64] Injecting at position 64, measuring at [64, 96, 128, 192, 255]...")
    positions_64 = [64, 96, 128, 192, 255]
    results["inject_pos_64"] = trace_codeword(inject_pos=64, measure_positions=positions_64)
    print(f"    inject_pos_64: done")
    for pos in positions_64:
        layers = sorted(results["inject_pos_64"][pos].keys(), key=lambda x: int(x[1:]) if x[1:].lstrip('-').isdigit() else -100)
        vals = [f"{results['inject_pos_64'][pos][l]:.3f}" for l in layers]
        print(f"      pos={pos}: {' '.join(vals)}")

    del model
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 60)
    print("  🔬 Mechanistic Validation: DWARF Gen5 L=8 preIF")
    print("=" * 60)

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
        "if_layer": IF_LAYER,
    }

    t0 = time.time()

    all_results["exp1_fa_control"] = experiment_1_fa_control(device)
    all_results["exp2_if_ablation"] = experiment_2_if_ablation(device)
    all_results["exp3_multikey"] = experiment_3_multikey(device)
    all_results["exp4_relay_tracing"] = experiment_4_relay_tracing(device)

    all_results["elapsed_s"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {OUT_FILE}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    exp1 = all_results["exp1_fa_control"]
    print("\n  Exp 1 (FA Control) - mean passkey accuracy:")
    for cond in ["baseline", "random_K", "zero_FA_output", "local_FA"]:
        vals = list(exp1[cond].values())
        mean = sum(vals) / len(vals) if vals else 0
        print(f"    {cond}: {mean * 100:.1f}%")

    exp2 = all_results["exp2_if_ablation"]
    print("\n  Exp 2 (IF Ablation) - mean passkey accuracy:")
    for cond in ["baseline", "zero_IF_weights", "identity_slot", "remove_IF_residual"]:
        vals = list(exp2[cond].values())
        mean = sum(vals) / len(vals) if vals else 0
        print(f"    {cond}: {mean * 100:.1f}%")

    exp3 = all_results["exp3_multikey"]
    print("\n  Exp 3 (Multi-key) - accuracy by config:")
    for config in ["word_only_d256_d512", "word_only_d512_d512", "triple_d128_d256_d512"]:
        parts = [f"{k}={v * 100:.0f}%" for k, v in exp3[config].items()]
        print(f"    {config}: {', '.join(parts)}")

    print(f"\n  Total elapsed: {all_results['elapsed_s']:.1f}s")

    subprocess.run(
        ["openclaw", "system", "event", "--text",
         "Done: mechanistic evals complete (4 experiments)", "--mode", "now"],
        capture_output=True
    )


if __name__ == "__main__":
    main()
