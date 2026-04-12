"""
relay_path_tracer.py — DWARF Relay Path Offset Analysis Tool

For a given passkey prompt and checkpoint, identifies which DSQG offset values
are carrying the retrieval signal via:

1. Offset ablation sweep: Zero out each offset's attention contribution across
   all DSQG layers, measure passkey accuracy change to identify load-bearing
   vs redundant offsets.

2. Hop-chain reconstruction: For passkey at distance d from cue, identify the
   minimal 2-hop relay path (δ₁ + δ₂ ≈ d). Run targeted ablation to measure
   path criticality.

3. Per-layer signal flow: Compute cosine similarity between residual stream at
   passkey position and cue position at each layer, showing signal buildup.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/relay_path_tracer.py \
      --arch moonshot_58m \
      --checkpoint autoresearch/checkpoints/moonshot_58m_best.pt \
      --distance 512 \
      --out_dir tools/viz_output/ \
      --n_trials 10
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_registry import (
    MODEL_REGISTRY, build_model, get_tokenizer, get_offsets,
    SE015_OFFSETS, SE096_OFFSETS,
)

_PASSKEY_WORDS = ['apple', 'banana', 'orange', 'cherry', 'grape',
                  'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'


class OffsetAblationHook:
    """Hook that zeros out attention contributions from specific DSQG offsets."""

    def __init__(self, offsets_to_ablate: set, all_offsets: list):
        self.offsets_to_ablate = offsets_to_ablate
        self.all_offsets = all_offsets
        self.ablation_indices = [
            i for i, o in enumerate(all_offsets) if o in offsets_to_ablate
        ]
        self.enabled = True

    def get_hook(self):
        def hook_fn(module, inputs, output):
            if not self.enabled:
                return output
            return output
        return hook_fn


class DSQGAttentionAblator:
    """Wraps a DSQG model to allow offset-specific ablation during forward pass."""

    def __init__(self, model, offsets: list):
        self.model = model
        self.offsets = offsets
        self.ablated_offsets: set = set()
        self._original_forwards = {}
        self._dsqg_blocks = []

        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
                self._dsqg_blocks.append((i, block))

    def set_ablated_offsets(self, offsets_to_ablate: set):
        self.ablated_offsets = offsets_to_ablate

    def clear_ablation(self):
        self.ablated_offsets = set()


class ResidualStreamCapture:
    """Captures residual stream activations at each layer for signal flow analysis."""

    def __init__(self, model, full_attn_layer: int):
        self.model = model
        self.full_attn_layer = full_attn_layer
        self.activations = {}
        self.hooks = []

    def _make_embed_hook(self):
        def hook(module, inputs, output):
            self.activations['embeddings'] = output.detach().cpu()
        return hook

    def _make_block_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            key = f'after_block_{layer_idx}'
            self.activations[key] = output.detach().cpu()
        return hook

    def register_hooks(self):
        self.hooks.append(
            self.model.drop.register_forward_hook(self._make_embed_hook())
        )
        for i, block in enumerate(self.model.blocks):
            self.hooks.append(
                block.register_forward_hook(self._make_block_hook(i))
            )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.activations = {}


def load_model(arch: str, checkpoint_path: str, root: str, device: torch.device):
    """Load model from checkpoint via shared registry."""
    model, cfg, m = build_model(arch, device=device, checkpoint_path=checkpoint_path)
    full_attn_layer = cfg.get('_resolved_full_attn_layer')
    if full_attn_layer is None and not cfg.get('no_full_attn', False):
        full_attn_layer = getattr(m, 'FULL_ATTN_LAYER', len(model.blocks) - 1)
    return model, full_attn_layer, m


def load_tokenizer(m, root: str, arch: str):
    """Load tokenizer via shared registry, wrapped with BPETokenizerWrapper."""
    tok = get_tokenizer(arch)
    return m.BPETokenizerWrapper(tok)


def build_passkey_sequence(tokenizer, word: str, distance: int, max_seq_len: int = 2048):
    """Build passkey retrieval sequence with given distance between passkey and cue."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=word))
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)

    available = max_seq_len - 1 - len(intro_ids) - len(cue_ids) - 1
    actual_distance = min(distance, available)

    filler = []
    while len(filler) < actual_distance:
        filler.extend(filler_ids)
    filler = filler[:actual_distance]

    full_seq = intro_ids + filler + cue_ids

    passkey_start = len(tokenizer.encode('the secret word is '))
    passkey_pos = passkey_start
    cue_pos = len(full_seq) - 1

    return full_seq, passkey_pos, cue_pos, actual_distance


def dsqg_forward_with_ablation(
    model,
    ids_tensor: torch.Tensor,
    offsets: list,
    ablated_offsets: set,
    device: torch.device,
):
    """
    Run DSQG model forward pass with specific offsets ablated.

    Uses pure Python reference implementation of DSQG attention that allows
    offset-level masking. This is slower than the Triton kernel but necessary
    for offset-level ablation analysis.
    """
    B, N = ids_tensor.shape
    with torch.no_grad():
        pos = torch.arange(N, device=device).unsqueeze(0)
        tok_emb = model.embedding(ids_tensor)
        # Some archs (moonshot lineage) have pos_embed; d768_l24 does not
        if hasattr(model, 'pos_embed'):
            x = model.drop(tok_emb + model.pos_embed(pos))
        else:
            x = model.drop(tok_emb)

        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
                # For triadic archs, use the block's own offset list
                block_offsets = offsets
                if hasattr(block.attn, 'offsets_dev'):
                    block_offsets = block.attn.offsets_dev.tolist()
                x = _dsqg_block_forward_with_ablation(
                    block, x, block_offsets, ablated_offsets, device
                )
            else:
                x = block(x)

        logits = model.out(model.norm(x))
    return logits


def _dsqg_block_forward_with_ablation(
    block,
    x: torch.Tensor,
    offsets: list,
    ablated_offsets: set,
    device: torch.device,
):
    """Forward pass through DSQG block with offset ablation."""
    attn = block.attn
    B, N, D = x.shape
    H = attn.num_heads
    HD = D // H

    kv_inject = None
    if hasattr(block, 'interference') and block.interference:
        xi = block.inter_norm(x)
        from causal_ema_scan import causal_ema_scan
        pool = causal_ema_scan(xi, block.ema_factor.abs() + 0.00001, floor=0.00001)
        rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
        pool = pool / (rms + 1e-6)
        inter = torch.sigmoid(block.inter_gate(xi)) * pool
        k_delta = block.inter_k_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3)
        v_delta = block.inter_v_proj(inter).view(B, N, H, HD).permute(0, 2, 1, 3)
        kv_inject = (k_delta.contiguous(), v_delta.contiguous())

    h = block.norm1(x)
    qkv = attn.qkv_proj(h)
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3).float()
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3).float()
    v = v.view(B, N, H, HD).permute(0, 2, 1, 3).float()

    if kv_inject is not None:
        k = k + kv_inject[0].float()
        v = v + kv_inject[1].float()

    out = _dsqg_attention_with_ablation(
        q, k, v,
        attn.pos_bias.float(),
        attn.scale_embed.float(),
        offsets,
        ablated_offsets,
        device,
    )

    out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D).to(x.dtype)

    if hasattr(attn, 'gate_proj'):
        gate = torch.sigmoid(attn.gate_proj(h))
        attn_out = attn.out_proj(out_flat * gate)
    else:
        attn_out = attn.out_proj(out_flat)

    x = x + F.dropout(attn_out, p=0.0, training=False)
    x = x + block.ffn(block.norm2(x))
    return x


def _dsqg_attention_with_ablation(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    offsets: list,
    ablated_offsets: set,
    device: torch.device,
):
    """
    Pure Python DSQG attention with offset ablation support.

    q, k, v: [B, H, N, HD]
    pos_bias: [J, H]
    scale_embed: [J, HD]
    """
    B, H, N, HD = q.shape
    J = len(offsets)
    scale = 1.0 / math.sqrt(HD)
    INF = float('-inf')

    max_offset = max(offsets)
    k_padded = F.pad(k, (0, 0, max_offset, 0))
    v_padded = F.pad(v, (0, 0, max_offset, 0))

    scores = torch.full((B, H, N, J), INF, device=device, dtype=torch.float32)
    values_gather = torch.zeros((B, H, N, J, HD), device=device, dtype=torch.float32)

    for j_idx, delta in enumerate(offsets):
        if delta in ablated_offsets:
            continue

        k_j = k_padded[:, :, max_offset - delta:max_offset - delta + N, :]
        v_j = v_padded[:, :, max_offset - delta:max_offset - delta + N, :]

        qk = (q * k_j).sum(-1) * scale
        pb = pos_bias[j_idx].view(1, H, 1).expand(B, H, N)
        se = (q * scale_embed[j_idx].view(1, 1, 1, HD)).sum(-1) * scale

        raw = qk + pb + se

        valid_mask = torch.arange(N, device=device).view(1, 1, N) >= delta
        scores[:, :, :, j_idx] = torch.where(valid_mask, raw, torch.tensor(INF, device=device))
        values_gather[:, :, :, j_idx, :] = v_j

    alpha = torch.softmax(scores, dim=-1)
    alpha = torch.nan_to_num(alpha, nan=0.0)

    out = torch.einsum('bhnj,bhnjd->bhnd', alpha, values_gather)
    return out


@torch.inference_mode()
def evaluate_passkey_accuracy(
    model,
    tokenizer,
    device: torch.device,
    distance: int,
    n_trials: int,
    offsets: list,
    ablated_offsets: Optional[set] = None,
    use_ablation: bool = False,
):
    """Evaluate passkey retrieval accuracy with optional offset ablation."""
    model.eval()

    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        assert encoded, f'Could not encode passkey word: {word}'
        word_token_ids[word] = encoded[0]

    correct = 0
    total = 0

    for trial in range(n_trials):
        word = _PASSKEY_WORDS[trial % len(_PASSKEY_WORDS)]
        seq, passkey_pos, cue_pos, actual_d = build_passkey_sequence(
            tokenizer, word, distance
        )

        ids_tensor = torch.tensor([seq], dtype=torch.long, device=device)

        if use_ablation and ablated_offsets:
            logits = dsqg_forward_with_ablation(
                model, ids_tensor, offsets, ablated_offsets, device
            )
        else:
            logits = model(ids_tensor)

        cand_words = [word] + [w for w in _PASSKEY_WORDS if w != word][:9]
        cand_ids = [word_token_ids[w] for w in cand_words]
        cand_tensor = torch.tensor([cand_ids], device=device)

        next_logits = logits[0, cue_pos, :]
        cand_logits = next_logits[cand_ids]

        if cand_logits.argmax().item() == 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_offset_ablation_sweep(
    model,
    tokenizer,
    device: torch.device,
    distance: int,
    n_trials: int,
    offsets: list,
):
    """Run ablation sweep: zero each offset and measure accuracy drop."""
    print(f'\n🔬 Running offset ablation sweep (distance={distance}, n_trials={n_trials})')

    baseline_acc = evaluate_passkey_accuracy(
        model, tokenizer, device, distance, n_trials, offsets,
        ablated_offsets=None, use_ablation=False,
    )
    print(f'  Baseline accuracy: {baseline_acc * 100:.1f}%')

    results = {'baseline': baseline_acc, 'ablations': {}}

    for offset in offsets:
        acc = evaluate_passkey_accuracy(
            model, tokenizer, device, distance, n_trials, offsets,
            ablated_offsets={offset}, use_ablation=True,
        )
        drop = baseline_acc - acc
        results['ablations'][offset] = {
            'accuracy': acc,
            'drop': drop,
        }
        marker = '❌' if drop > 0.1 else ('⚠️' if drop > 0.05 else '✓')
        print(f'  δ={offset:4d}: acc={acc * 100:5.1f}% (drop={drop * 100:+5.1f}%) {marker}')

    return results


def find_hop_chains(distance: int, offsets: list):
    """Find all 2-hop relay paths where δ₁ + δ₂ ≈ distance."""
    chains = []
    offset_set = set(offsets)

    for d1 in offsets:
        d2_needed = distance - d1
        if d2_needed in offset_set and d2_needed > 0:
            chains.append((d1, d2_needed, d1 + d2_needed))
        for d2 in offsets:
            if d1 + d2 >= distance - 5 and d1 + d2 <= distance + 5:
                if (d1, d2, d1 + d2) not in chains:
                    chains.append((d1, d2, d1 + d2))

    chains.sort(key=lambda x: abs(x[2] - distance))
    return chains[:10]


def run_hop_chain_analysis(
    model,
    tokenizer,
    device: torch.device,
    distance: int,
    n_trials: int,
    offsets: list,
):
    """Analyze 2-hop relay paths for the given distance."""
    print(f'\n🔗 Running hop-chain analysis (distance={distance})')

    chains = find_hop_chains(distance, offsets)
    if not chains:
        print(f'  No viable 2-hop chains found for distance={distance}')
        return {'distance': distance, 'chains': []}

    print(f'  Found {len(chains)} potential 2-hop chains')

    baseline_acc = evaluate_passkey_accuracy(
        model, tokenizer, device, distance, n_trials, offsets,
        ablated_offsets=None, use_ablation=False,
    )
    print(f'  Baseline accuracy: {baseline_acc * 100:.1f}%')

    results = {
        'distance': distance,
        'baseline': baseline_acc,
        'chains': [],
    }

    for d1, d2, total in chains[:5]:
        acc_no_d1 = evaluate_passkey_accuracy(
            model, tokenizer, device, distance, n_trials, offsets,
            ablated_offsets={d1}, use_ablation=True,
        )
        acc_no_d2 = evaluate_passkey_accuracy(
            model, tokenizer, device, distance, n_trials, offsets,
            ablated_offsets={d2}, use_ablation=True,
        )
        acc_no_both = evaluate_passkey_accuracy(
            model, tokenizer, device, distance, n_trials, offsets,
            ablated_offsets={d1, d2}, use_ablation=True,
        )

        chain_result = {
            'path': f'δ={d1}+δ={d2}={total}',
            'd1': d1,
            'd2': d2,
            'total': total,
            'baseline': baseline_acc,
            'acc_no_d1': acc_no_d1,
            'acc_no_d2': acc_no_d2,
            'acc_no_both': acc_no_both,
            'drop_d1': baseline_acc - acc_no_d1,
            'drop_d2': baseline_acc - acc_no_d2,
            'drop_both': baseline_acc - acc_no_both,
        }
        results['chains'].append(chain_result)

        print(f'  Path δ={d1}+δ={d2}={total}:')
        print(f'    -δ{d1}: {acc_no_d1 * 100:5.1f}% (drop={chain_result["drop_d1"] * 100:+5.1f}%)')
        print(f'    -δ{d2}: {acc_no_d2 * 100:5.1f}% (drop={chain_result["drop_d2"] * 100:+5.1f}%)')
        print(f'    -both: {acc_no_both * 100:5.1f}% (drop={chain_result["drop_both"] * 100:+5.1f}%)')

    return results


def run_signal_flow_analysis(
    model,
    tokenizer,
    device: torch.device,
    distance: int,
    full_attn_layer: int,
):
    """Compute per-layer cosine similarity between passkey and cue positions."""
    print(f'\n📊 Running signal flow analysis (distance={distance})')

    word = _PASSKEY_WORDS[0]
    seq, passkey_pos, cue_pos, actual_d = build_passkey_sequence(
        tokenizer, word, distance
    )

    print(f'  Sequence length: {len(seq)}, passkey@{passkey_pos}, cue@{cue_pos}')

    ids_tensor = torch.tensor([seq], dtype=torch.long, device=device)

    capture = ResidualStreamCapture(model, full_attn_layer)
    capture.register_hooks()

    with torch.no_grad():
        _ = model(ids_tensor)

    capture.remove_hooks()

    num_layers = len(model.blocks)
    results = {
        'distance': distance,
        'passkey_pos': passkey_pos,
        'cue_pos': cue_pos,
        'full_attn_layer': full_attn_layer,
        'layers': [],
    }

    all_keys = ['embeddings'] + [f'after_block_{i}' for i in range(num_layers)]

    for key in all_keys:
        if key not in capture.activations:
            continue

        act = capture.activations[key]
        h_passkey = act[0, passkey_pos, :]
        h_cue = act[0, cue_pos, :]

        h_pk_norm = h_passkey / (h_passkey.norm() + 1e-8)
        h_cue_norm = h_cue / (h_cue.norm() + 1e-8)
        cos_sim = (h_pk_norm @ h_cue_norm).item()

        layer_name = 'embed' if key == 'embeddings' else f'L{int(key.split("_")[-1])}'
        layer_idx = -1 if key == 'embeddings' else int(key.split('_')[-1])

        is_fa = layer_idx == full_attn_layer
        marker = ' ← FA' if is_fa else ''

        results['layers'].append({
            'name': layer_name,
            'layer_idx': layer_idx,
            'cos_sim': cos_sim,
            'is_full_attn': is_fa,
        })

        print(f'  {layer_name:6s}: cos_sim={cos_sim:+.4f}{marker}')

    return results


def plot_ablation_results(results: dict, arch: str, distance: int, out_path: str):
    """Plot bar chart of accuracy drop per ablated offset."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ablations = results['ablations']
    offsets_sorted = sorted(ablations.keys(), key=lambda x: ablations[x]['drop'], reverse=True)
    drops = [ablations[o]['drop'] * 100 for o in offsets_sorted]

    x = np.arange(len(offsets_sorted))
    colors = ['red' if d > 10 else ('orange' if d > 5 else 'steelblue') for d in drops]

    bars = ax.bar(x, drops, color=colors, alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Critical (>10%)')
    ax.axhline(5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (>5%)')

    ax.set_xlabel('Offset δ (sorted by impact)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'DSQG Offset Ablation Impact — {arch} @ d={distance}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in offsets_sorted], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_signal_flow(results: dict, arch: str, out_path: str):
    """Plot layer-by-layer cosine similarity between passkey and cue."""
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = results['layers']
    names = [l['name'] for l in layers]
    cos_sims = [l['cos_sim'] for l in layers]
    is_fa = [l['is_full_attn'] for l in layers]

    x = np.arange(len(names))
    colors = ['red' if fa else 'steelblue' for fa in is_fa]

    bars = ax.bar(x, cos_sims, color=colors, alpha=0.8)

    ax.plot(x, cos_sims, 'k-o', linewidth=1.5, markersize=4, alpha=0.5)

    ax.axhline(0, color='gray', linewidth=0.5)

    fa_idx = [i for i, fa in enumerate(is_fa) if fa]
    if fa_idx:
        if fa_idx:
            ax.axvline(fa_idx[0], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Full Attention Layer')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity (passkey ↔ cue)', fontsize=12)
    ax.set_title(f'Signal Flow: Passkey→Cue Similarity — {arch} @ d={results["distance"]}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_hop_chains(results: dict, arch: str, out_path: str):
    """Plot 2-hop path analysis results."""
    fig, ax = plt.subplots(figsize=(12, 7))

    chains = results.get('chains', [])
    if not chains:
        ax.text(0.5, 0.5, 'No hop chains found', ha='center', va='center', fontsize=14)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    n_chains = len(chains)
    x = np.arange(n_chains)
    width = 0.25

    baseline_drops = [c['baseline'] * 100 for c in chains]
    drop_d1 = [c['drop_d1'] * 100 for c in chains]
    drop_d2 = [c['drop_d2'] * 100 for c in chains]
    drop_both = [c['drop_both'] * 100 for c in chains]

    ax.bar(x - width, drop_d1, width, label='-δ₁ only', color='steelblue', alpha=0.8)
    ax.bar(x, drop_d2, width, label='-δ₂ only', color='orange', alpha=0.8)
    ax.bar(x + width, drop_both, width, label='-both', color='red', alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Hop Chain', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'2-Hop Relay Path Analysis — {arch} @ d={results["distance"]}', fontsize=14)

    labels = [f'{c["d1"]}+{c["d2"]}={c["total"]}' for c in chains]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='DWARF Relay Path Tracer — offset ablation and signal flow analysis')
    parser.add_argument('--arch', required=True, choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--distance', type=int, default=512,
                        choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536],
                        help='Passkey distance for analysis')
    parser.add_argument('--out_dir', default='tools/viz_output',
                        help='Output directory for visualizations')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of passkey trials per ablation condition')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to run on')
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip offset ablation sweep')
    parser.add_argument('--skip_hops', action='store_true',
                        help='Skip hop-chain analysis')
    parser.add_argument('--skip_flow', action='store_true',
                        help='Skip signal flow analysis')

    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)

    out_subdir = os.path.join(args.out_dir, f'relay_{args.arch}')
    os.makedirs(out_subdir, exist_ok=True)

    print('=' * 70)
    print(f'  🔍 DWARF Relay Path Tracer — {args.arch}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Distance: {args.distance}')
    print(f'  Trials: {args.n_trials}')
    print('=' * 70)

    print('\nLoading model...')
    model, full_attn_layer, m = load_model(args.arch, args.checkpoint, root, device)

    print('Loading tokenizer...')
    tokenizer = load_tokenizer(m, root, args.arch)

    offsets = get_offsets(args.arch)
    print(f'  Offset set: J={len(offsets)} offsets')
    print(f'  Full attention layer: L{full_attn_layer}')

    all_results = {
        'arch': args.arch,
        'checkpoint': args.checkpoint,
        'distance': args.distance,
        'n_trials': args.n_trials,
        'offsets': offsets,
        'full_attn_layer': full_attn_layer,
    }

    if not args.skip_ablation:
        ablation_results = run_offset_ablation_sweep(
            model, tokenizer, device, args.distance, args.n_trials, offsets
        )
        all_results['ablation'] = ablation_results

        plot_ablation_results(
            ablation_results, args.arch, args.distance,
            os.path.join(out_subdir, f'relay_path_{args.arch}_d{args.distance}.png')
        )

    if not args.skip_hops:
        hop_results = run_hop_chain_analysis(
            model, tokenizer, device, args.distance, args.n_trials, offsets
        )
        all_results['hop_chains'] = hop_results

        plot_hop_chains(
            hop_results, args.arch,
            os.path.join(out_subdir, f'relay_hops_{args.arch}_d{args.distance}.png')
        )

    if not args.skip_flow:
        flow_results = run_signal_flow_analysis(
            model, tokenizer, device, args.distance, full_attn_layer
        )
        all_results['signal_flow'] = flow_results

        plot_signal_flow(
            flow_results, args.arch,
            os.path.join(out_subdir, f'relay_flow_{args.arch}_d{args.distance}.png')
        )

    json_path = os.path.join(out_subdir, f'relay_analysis_{args.arch}_d{args.distance}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Saved summary: {json_path}')

    print(f'\n{"=" * 70}')
    print(f'  Done. Results in {out_subdir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
