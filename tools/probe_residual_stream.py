"""
probe_residual_stream.py — General-purpose DWARF residual stream prober

A comprehensive tool for analyzing residual stream dynamics in DWARF checkpoints.
Works on any architecture. Key analyses:

  1. Logit lens — at each layer, project residual through norm + lm_head, compute
     entropy and top-k predictions. Shows where information emerges and entropy
     collapses (commitment) or rebounds (relay override).

  2. Orthogonality — cosine similarity between passkey, cue, and filler positions
     across layers. Shows subspace alignment/divergence.

  3. FA delta — ||h_after - h_before|| / ||h_before|| per token. Shows which tokens
     the FA layer modifies most.

  4. Cross-architecture comparison — side-by-side plots comparing two models.

Outputs:
  - logit_lens_{arch}_d{dist}.png     — entropy heatmap + top-1 labels
  - orthogonality_{arch}_d{dist}.png  — cosim trajectories across layers
  - fa_delta_{arch}_d{dist}.png       — per-token FA modification
  - probe_summary_{arch}_d{dist}.json — raw data
  - comparison_{arch1}_vs_{arch2}_d{dist}.png (if --compare)

Usage:
  # Single model
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/probe_residual_stream.py \\
      --arch d768_l16 \\
      --checkpoint autoresearch/checkpoints/d768_l16_fa4_ep3_resume.pt \\
      --distance 512

  # Cross-arch comparison
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/probe_residual_stream.py \\
      --arch d768_l16 \\
      --checkpoint autoresearch/checkpoints/d768_l16_fa4_ep3_resume.pt \\
      --compare_arch d768_l24 \\
      --compare_checkpoint autoresearch/checkpoints/d768_l24_fa6_ep1_resume.pt \\
      --distance 512
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F

from model_registry import MODEL_REGISTRY, build_model, get_tokenizer

_PASSKEY_WORDS = ['apple', 'banana', 'orange', 'cherry', 'grape',
                  'lemon', 'mango', 'peach', 'plum', 'berry',
                  'walnut', 'almond', 'cashew', 'pecan', 'hazel']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'


class ResidualStreamCapture:
    """Hook-based capture of residual stream at each layer boundary."""

    def __init__(self, model, full_attn_layer):
        self.model = model
        self.full_attn_layer = full_attn_layer
        self.activations = {}
        self.hooks = []

    def _make_embed_hook(self):
        def hook(module, input, output):
            self.activations['layer_0'] = output.detach().clone()
        return hook

    def _make_block_hook(self, layer_idx):
        def hook(module, input, output):
            key = f'layer_{layer_idx + 1}'
            self.activations[key] = output.detach().clone()
            if layer_idx == self.full_attn_layer:
                self.activations['before_fa'] = input[0].detach().clone()
                self.activations['after_fa'] = output.detach().clone()
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


def load_model(arch, checkpoint_path, root, device):
    """Load model from checkpoint via shared registry."""
    model, cfg, m = build_model(arch, device=device, checkpoint_path=checkpoint_path)
    full_attn_layer = cfg.get('_resolved_full_attn_layer') or getattr(m, 'FULL_ATTN_LAYER', len(model.blocks) - 1)
    val_ppl = cfg.get('_val_ppl', '?')
    return model, full_attn_layer, m, val_ppl


def load_tokenizer(m, root, arch=None):
    """Load tokenizer via shared registry, wrapped with BPETokenizerWrapper."""
    tok = get_tokenizer(arch)
    tok_key = MODEL_REGISTRY[arch].get('tokenizer', 'fineweb_32k')
    print(f'  Tokenizer: {tok_key}')
    return m.BPETokenizerWrapper(tok), tok


def build_passkey_sequence(tokenizer, word, distance, max_seq_len=2048):
    """Build a passkey retrieval sequence."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=word))
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)

    available = max_seq_len - 1 - len(intro_ids) - len(cue_ids) - 1
    if distance > available:
        distance = available

    filler = []
    while len(filler) < distance:
        filler.extend(filler_ids)
    filler = filler[:distance]

    full_seq = intro_ids + filler + cue_ids

    prefix_len = len(tokenizer.encode('the secret word is '))
    passkey_pos = prefix_len
    cue_start = len(intro_ids) + len(filler)
    cue_pos = len(full_seq) - 1

    return full_seq, passkey_pos, cue_pos, cue_start


def compute_logit_lens(model, activations, device):
    """Apply logit lens at each layer: norm → lm_head → softmax → entropy."""
    results = {}
    num_layers = len(model.blocks)

    for layer_idx in range(num_layers + 1):
        key = f'layer_{layer_idx}'
        if key not in activations:
            continue

        h = activations[key].to(device)
        normed = model.norm(h)
        logits = model.out(normed)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        top_vals, top_ids = torch.topk(probs, k=5, dim=-1)

        results[layer_idx] = {
            'entropy': entropy[0].detach().cpu().numpy(),
            'top5_ids': top_ids[0].detach().cpu().numpy(),
            'top5_probs': top_vals[0].detach().cpu().numpy(),
        }

    return results


def compute_orthogonality(activations, passkey_pos, cue_pos, filler_positions, num_layers):
    """Compute cosine similarities between passkey, cue, and filler positions."""
    results = {'passkey_cue': [], 'passkey_filler': [], 'cue_filler': [], 'layers': []}

    for layer_idx in range(num_layers + 1):
        key = f'layer_{layer_idx}'
        if key not in activations:
            continue

        h = activations[key][0]
        h_passkey = h[passkey_pos]
        h_cue = h[cue_pos]
        h_filler = h[filler_positions].mean(dim=0)

        def cosim(a, b):
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

        results['layers'].append(layer_idx)
        results['passkey_cue'].append(cosim(h_passkey, h_cue))
        results['passkey_filler'].append(cosim(h_passkey, h_filler))
        results['cue_filler'].append(cosim(h_cue, h_filler))

    return results


def compute_fa_delta(activations):
    """Compute ||h_after - h_before|| / ||h_before|| at FA layer."""
    if 'before_fa' not in activations or 'after_fa' not in activations:
        return None

    before = activations['before_fa'][0]
    after = activations['after_fa'][0]

    delta = (after - before).norm(dim=-1)
    input_norm = before.norm(dim=-1)

    relative_delta = delta / (input_norm + 1e-8)
    return relative_delta.cpu().numpy()


def plot_logit_lens(logit_lens_data, tokenizer, passkey_pos, cue_pos, passkey_word,
                    full_attn_layer, arch, distance, out_path):
    """Plot logit lens entropy heatmap with top-1 token labels."""
    layers = sorted(logit_lens_data.keys())
    if not layers:
        return

    seq_len = len(logit_lens_data[layers[0]]['entropy'])
    display_range = min(seq_len, 100)

    entropy_matrix = np.zeros((len(layers), display_range))
    for i, layer in enumerate(layers):
        entropy_matrix[i, :display_range] = logit_lens_data[layer]['entropy'][:display_range]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

    im = ax1.imshow(entropy_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    ax1.set_xlabel('Token Position', fontsize=11)
    ax1.set_ylabel('Layer', fontsize=11)
    ax1.set_title(f'Logit Lens Entropy — {arch} (distance={distance})\n'
                  f'Lower entropy = higher confidence; FA@L{full_attn_layer}', fontsize=12)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels([f'L{l}' for l in layers], fontsize=8)

    if passkey_pos < display_range:
        ax1.axvline(passkey_pos, color='lime', linestyle='--', linewidth=2, label='Passkey')
    if cue_pos < display_range:
        ax1.axvline(cue_pos, color='orange', linestyle='--', linewidth=2, label='Cue')
    ax1.axhline(full_attn_layer + 0.5, color='red', linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'FA@L{full_attn_layer}')
    ax1.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax1, label='Entropy (nats)', shrink=0.8)

    cue_entropy = [logit_lens_data[l]['entropy'][cue_pos] for l in layers]
    ax2.plot(layers, cue_entropy, 'o-', linewidth=2, markersize=6, color='darkorange', label='Cue pos entropy')
    ax2.axvline(full_attn_layer, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'FA@L{full_attn_layer}')
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Entropy at Cue (nats)', fontsize=11)
    ax2.set_title(f'Entropy trajectory at cue position — target word: "{passkey_word.upper()}"', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    target_token_id = tokenizer.encode(f' {passkey_word}')
    if target_token_id:
        target_id = target_token_id[0]
        first_top5_layer = None
        first_top1_layer = None
        for l in layers:
            top5 = logit_lens_data[l]['top5_ids'][cue_pos]
            if target_id in top5:
                if first_top5_layer is None:
                    first_top5_layer = l
                if top5[0] == target_id and first_top1_layer is None:
                    first_top1_layer = l

        info_text = f'Target "{passkey_word}" (id={target_id}): '
        if first_top5_layer is not None:
            info_text += f'enters top-5 @ L{first_top5_layer}'
        if first_top1_layer is not None:
            info_text += f', top-1 @ L{first_top1_layer}'
        ax2.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                     fontsize=9, color='green', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_orthogonality(ortho_data, full_attn_layer, arch, distance, out_path):
    """Plot cosine similarity trajectories across layers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = ortho_data['layers']
    ax.plot(layers, ortho_data['passkey_cue'], 'o-', linewidth=2, markersize=6,
            color='blue', label='Passkey ↔ Cue')
    ax.plot(layers, ortho_data['passkey_filler'], 's--', linewidth=2, markersize=5,
            color='gray', label='Passkey ↔ Filler (mean)')
    ax.plot(layers, ortho_data['cue_filler'], '^--', linewidth=2, markersize=5,
            color='purple', label='Cue ↔ Filler (mean)')

    ax.axvline(full_attn_layer, color='red', linestyle='-', linewidth=2, alpha=0.7,
               label=f'FA @ L{full_attn_layer}')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title(f'Residual Stream Orthogonality — {arch} (distance={distance})\n'
                 f'Passkey, Cue, and Filler position alignment across layers', fontsize=12)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_fa_delta(fa_delta, passkey_pos, cue_pos, cue_start, full_attn_layer,
                  arch, distance, out_path):
    """Plot FA delta (relative change) per token position."""
    if fa_delta is None:
        print(f'  Skipping FA delta plot (no FA layer data)')
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(fa_delta))
    ax.bar(x, fa_delta, width=1.0, alpha=0.7, color='steelblue')

    ax.axvline(passkey_pos, color='lime', linestyle='--', linewidth=2,
               label=f'Passkey pos={passkey_pos}')
    ax.axvline(cue_pos, color='orange', linestyle='--', linewidth=2,
               label=f'Cue pos={cue_pos}')
    if cue_start != cue_pos:
        ax.axvline(cue_start, color='yellow', linestyle=':', linewidth=1.5,
                   label=f'Cue start={cue_start}')

    ax.set_xlabel('Token Position', fontsize=11)
    ax.set_ylabel('||FA(h) - h|| / ||h||', fontsize=11)
    ax.set_title(f'FA Relative Delta @ L{full_attn_layer} — {arch} (distance={distance})\n'
                 f'Shows which tokens FA modifies most', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    if passkey_pos < len(fa_delta):
        pk_val = fa_delta[passkey_pos]
        ax.annotate(f'{pk_val:.3f}', xy=(passkey_pos, pk_val),
                    xytext=(passkey_pos + 5, pk_val + 0.02),
                    fontsize=9, color='lime')
    if cue_pos < len(fa_delta):
        cue_val = fa_delta[cue_pos]
        ax.annotate(f'{cue_val:.3f}', xy=(cue_pos, cue_val),
                    xytext=(cue_pos - 20, cue_val + 0.02),
                    fontsize=9, color='orange')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_comparison(data_a, data_b, arch_a, arch_b, distance, out_path):
    """Create 2×3 comparison grid for two architectures."""
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    for col, (data, arch) in enumerate([(data_a, arch_a), (data_b, arch_b)]):
        logit_lens = data['logit_lens']
        ortho = data['orthogonality']
        fa_delta = data['fa_delta']
        fa_layer = data['full_attn_layer']
        cue_pos = data['cue_pos']
        passkey_pos = data['passkey_pos']

        ax1 = fig.add_subplot(gs[0, col])
        layers = sorted(logit_lens.keys())
        cue_entropy = [logit_lens[l]['entropy'][cue_pos] for l in layers]
        ax1.plot(layers, cue_entropy, 'o-', linewidth=2, markersize=6, color='darkorange')
        ax1.axvline(fa_layer, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Layer', fontsize=10)
        ax1.set_ylabel('Entropy at Cue (nats)', fontsize=10)
        ax1.set_title(f'{arch}\nLogit Lens: Cue Position Entropy', fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, col])
        ax2.plot(ortho['layers'], ortho['passkey_cue'], 'o-', linewidth=2, markersize=5,
                 color='blue', label='Passkey↔Cue')
        ax2.plot(ortho['layers'], ortho['passkey_filler'], 's--', linewidth=1.5, markersize=4,
                 color='gray', label='Passkey↔Filler')
        ax2.plot(ortho['layers'], ortho['cue_filler'], '^--', linewidth=1.5, markersize=4,
                 color='purple', label='Cue↔Filler')
        ax2.axvline(fa_layer, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Layer', fontsize=10)
        ax2.set_ylabel('Cosine Similarity', fontsize=10)
        ax2.set_title(f'{arch}\nOrthogonality Across Layers', fontsize=11)
        ax2.set_ylim(-0.3, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        ax3 = fig.add_subplot(gs[2, col])
        if fa_delta is not None:
            x = np.arange(len(fa_delta))
            ax3.bar(x, fa_delta, width=1.0, alpha=0.7, color='steelblue')
            ax3.axvline(passkey_pos, color='lime', linestyle='--', linewidth=2)
            ax3.axvline(cue_pos, color='orange', linestyle='--', linewidth=2)
        ax3.set_xlabel('Token Position', fontsize=10)
        ax3.set_ylabel('||FA(h) - h|| / ||h||', fontsize=10)
        ax3.set_title(f'{arch}\nFA Delta @ L{fa_layer}', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Cross-Architecture Comparison: {arch_a} vs {arch_b}\n'
                 f'Passkey distance = {distance}', fontsize=14, fontweight='bold')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def run_probe(arch, checkpoint_path, distance, out_dir, root, device, passkey_word=None):
    """Run all probes for a single model."""
    print(f'\n{"="*60}')
    print(f'  Probing {arch}')
    print(f'  Checkpoint: {checkpoint_path}')
    print(f'  Distance: {distance}')
    print(f'{"="*60}\n')

    model, full_attn_layer, m, val_ppl = load_model(arch, checkpoint_path, root, device)
    tokenizer_wrapper, raw_tokenizer = load_tokenizer(m, root, arch=arch)
    num_layers = len(model.blocks)

    print(f'  Model: {num_layers} layers, FA @ L{full_attn_layer}')

    if passkey_word is None:
        passkey_word = _PASSKEY_WORDS[hash(checkpoint_path) % len(_PASSKEY_WORDS)]

    ids, passkey_pos, cue_pos, cue_start = build_passkey_sequence(
        tokenizer_wrapper, passkey_word, distance)

    print(f'  Sequence: len={len(ids)}, passkey_pos={passkey_pos}, cue_pos={cue_pos}')
    print(f'  Passkey word: "{passkey_word}"')

    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    capture = ResidualStreamCapture(model, full_attn_layer)
    capture.register_hooks()

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(ids_tensor)

    print('\n  Computing logit lens...')
    logit_lens = compute_logit_lens(model, capture.activations, device)

    filler_start = len(tokenizer_wrapper.encode(_INTRO_TEMPLATE.format(word=passkey_word)))
    filler_end = cue_start
    filler_positions = list(range(filler_start, min(filler_end, len(ids))))
    if len(filler_positions) < 10:
        filler_positions = list(range(max(0, cue_start - 50), cue_start))

    print('  Computing orthogonality...')
    ortho = compute_orthogonality(
        capture.activations, passkey_pos, cue_pos, filler_positions, num_layers)

    print('  Computing FA delta...')
    fa_delta = compute_fa_delta(capture.activations)

    capture.remove_hooks()

    os.makedirs(out_dir, exist_ok=True)
    prefix = f'{arch}_d{distance}'

    print('\n  Generating plots...')
    plot_logit_lens(logit_lens, tokenizer_wrapper, passkey_pos, cue_pos, passkey_word,
                    full_attn_layer, arch, distance,
                    os.path.join(out_dir, f'logit_lens_{prefix}.png'))

    plot_orthogonality(ortho, full_attn_layer, arch, distance,
                       os.path.join(out_dir, f'orthogonality_{prefix}.png'))

    plot_fa_delta(fa_delta, passkey_pos, cue_pos, cue_start, full_attn_layer,
                  arch, distance,
                  os.path.join(out_dir, f'fa_delta_{prefix}.png'))

    summary = {
        'arch': arch,
        'checkpoint': checkpoint_path,
        'val_ppl': val_ppl,
        'distance': distance,
        'passkey_word': passkey_word,
        'passkey_pos': passkey_pos,
        'cue_pos': cue_pos,
        'seq_len': len(ids),
        'num_layers': num_layers,
        'full_attn_layer': full_attn_layer,
        'logit_lens_entropy_at_cue': {
            l: float(logit_lens[l]['entropy'][cue_pos]) for l in logit_lens
        },
        'orthogonality': {
            'layers': ortho['layers'],
            'passkey_cue': ortho['passkey_cue'],
            'passkey_filler': ortho['passkey_filler'],
            'cue_filler': ortho['cue_filler'],
        },
        'fa_delta_at_passkey': float(fa_delta[passkey_pos]) if fa_delta is not None else None,
        'fa_delta_at_cue': float(fa_delta[cue_pos]) if fa_delta is not None else None,
        'fa_delta_mean_filler': float(np.mean([fa_delta[i] for i in filler_positions if i < len(fa_delta)])) if fa_delta is not None else None,
    }

    summary_path = os.path.join(out_dir, f'probe_summary_{prefix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: {summary_path}')

    return {
        'logit_lens': logit_lens,
        'orthogonality': ortho,
        'fa_delta': fa_delta,
        'full_attn_layer': full_attn_layer,
        'passkey_pos': passkey_pos,
        'cue_pos': cue_pos,
        'cue_start': cue_start,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description='General-purpose DWARF residual stream prober')
    parser.add_argument('--arch', required=True, choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--distance', type=int, default=512,
                        help='Passkey distance (tokens between passkey and cue)')
    parser.add_argument('--out_dir', default='tools/viz_output',
                        help='Output directory for visualizations')
    parser.add_argument('--device', default='cuda',
                        help='Device to run on')
    parser.add_argument('--passkey_word', default=None,
                        help='Specific passkey word to use (default: auto-select)')

    parser.add_argument('--compare_arch', default=None, choices=list(MODEL_REGISTRY.keys()),
                        help='Second architecture for comparison')
    parser.add_argument('--compare_checkpoint', default=None,
                        help='Checkpoint for second model')

    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)

    passkey_word = args.passkey_word if args.passkey_word else 'banana'

    data_a = run_probe(
        args.arch, args.checkpoint, args.distance,
        os.path.join(root, args.out_dir), root, device, passkey_word)

    if args.compare_arch and args.compare_checkpoint:
        data_b = run_probe(
            args.compare_arch, args.compare_checkpoint, args.distance,
            os.path.join(root, args.out_dir), root, device, passkey_word)

        comparison_path = os.path.join(
            root, args.out_dir,
            f'comparison_{args.arch}_vs_{args.compare_arch}_d{args.distance}.png')
        plot_comparison(data_a, data_b, args.arch, args.compare_arch,
                        args.distance, comparison_path)

    print(f'\nDone. Output in {args.out_dir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
