"""
visualize_residual_stream.py — Residual Stream Visualization for Hybrid DSQG Models

Visualizes what the residual stream looks like at each layer boundary in a hybrid
DSQG model, with special focus on what the signal looks like before and after
the FullAttn block.

This answers: "What does the frozen FA actually receive, and what does it produce?"

Visualizations produced:
  1. norm_progression.png   — Mean ||h_t|| across token positions at each layer
  2. cosim_heatmap_*.png    — cos_sim(h[retrieval_cue], h[t]) before/after FA
  3. pca_layer*.png         — 2D PCA of residual stream at key layers
  4. fa_delta.png           — ||FA(h_t) - h_t|| / ||h_t|| per token position
  5. summary.png            — Combined 4-panel visualization

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/visualize_residual_stream.py \
      --arch borg_midattn \
      --checkpoint autoresearch/checkpoints/borg_midattn_best.pt \
      --out tools/viz_output/borg_midattn_residual \
      --passkey_distance 256

  # Compare multiple models
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/visualize_residual_stream.py \
      --arch j26d_int2_physics \
      --checkpoint autoresearch/checkpoints/99437df_j26d_int2_physics_best.pt \
      --out tools/viz_output/j26d_residual \
      --passkey_distance 512 \
      --n_sequences 10
"""

import argparse
import importlib.util
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F


def pca_transform(data, n_components=2):
    """Simple PCA using numpy SVD (no sklearn dependency)."""
    data = np.array(data)
    mean = data.mean(axis=0)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = U[:, :n_components] * S[:n_components]
    total_var = np.sum(S ** 2)
    explained_variance = (S[:n_components] ** 2) / total_var
    return projected, explained_variance

_TRAIN_SCRIPTS = {
    'condu':    'train/train_2048_condU.py',
    'd41s3':    'train/train_2048_14m_d41s3.py',
    'd41s5':    'train/train_2048_14m_d41s5.py',
    'd41_35m':  'train/train_2048_35m_d41.py',
    'condx_v2': 'train/train_2048_35m_condX_v2_bf16.py',
    'condm_85m': 'train/train_2048_85m_condM.py',
    'condu_v5': 'train/train_2048_condU_v5.py',
    'condv':    'train/train_2048_condV.py',
    'condw':    'train/train_2048_condW.py',
    'std_85m':  'train/train_2048_85m_standard_baseline.py',
    'std_13m':  'train/train_2048_85m_standard_baseline.py',
    'j24d_int2_physics': 'train/train_j24d_int2_physics_bf16.py',
    'j26d_int2_physics': 'train/train_j26d_int2_physics_bf16.py',
    'j20d_v10_L8':  'train/train_j20d_v10_L8_bf16.py',
    'j20d_v10_L10': 'train/train_j20d_v10_L10_bf16.py',
    'j20d_v10_L12': 'train/train_j20d_v10_L12_bf16.py',
    'j20d_v10_L32': 'train/train_j20d_v10_L32_bf16.py',
    'curve_27m': 'train/train_curve_27m_bf16.py',
    'borg_adapt_warmstart': 'train/train_borg_adapt_13m_bf16.py',
    'borg_midattn': 'train/train_borg_midattn_bf16.py',
    'borg_lastattn': 'train/train_borg_lastattn_bf16.py',
}

_MODEL_CLASSES = {
    'condu':    'CondMTransformer',
    'd41s3':    'CondMTransformer',
    'd41s5':    'CondMTransformer',
    'd41_35m':  'CondMTransformer',
    'condx_v2': 'CondXTransformer',
    'condm_85m': 'CondMTransformer',
    'condu_v5': 'CondUV5Transformer',
    'condv':    'CondMTransformer',
    'condw':    'CondWTransformer',
    'std_85m':  'StandardTransformer85M',
    'std_13m':  'StandardTransformer85M',
    'j24d_int2_physics': 'AutoresearchTransformerPhysics',
    'j26d_int2_physics': 'AutoresearchTransformerPhysics',
    'j20d_v10_L8':  'AutoresearchTransformerPhysics',
    'j20d_v10_L10': 'AutoresearchTransformerPhysics',
    'j20d_v10_L12': 'AutoresearchTransformerPhysics',
    'j20d_v10_L32': 'AutoresearchTransformerPhysics',
    'curve_27m': 'CurveTransformer',
    'borg_adapt_warmstart': 'AutoresearchTransformerPhysics',
    'borg_midattn': 'AutoresearchTransformerPhysics',
    'borg_lastattn': 'AutoresearchTransformerPhysics',
}

_ARCH_CONFIGS = {
    'borg_midattn': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 5, 'full_attn_layer': 2, 'interference_interval': 2,
    },
    'borg_lastattn': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 6, 'full_attn_layer': 5, 'interference_interval': 2,
    },
    'borg_adapt_warmstart': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 3, 'full_attn_layer': 0, 'interference_interval': 2,
    },
    'j26d_int2_physics': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 6, 'full_attn_layer': 5, 'interference_interval': 2,
    },
    'j20d_v10_L8': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 8, 'full_attn_layer': 7, 'interference_interval': 2,
    },
    'j20d_v10_L10': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 10, 'full_attn_layer': 9, 'interference_interval': 2,
    },
    'j20d_v10_L12': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 2048,
        'num_layers': 12, 'full_attn_layer': 11, 'interference_interval': 2,
    },
    'curve_27m': {
        'embedding_dim': 512, 'num_heads': 8, 'ffn_dim': 768,
        'num_layers': 6, 'full_attn_layer': 5, 'interference_interval': 2,
    },
}

_IS_STANDARD = {'std_85m', 'std_13m'}

_PASSKEY_WORDS = ['apple', 'banana', 'orange', 'cherry', 'grape',
                  'mango', 'peach', 'plum', 'kiwi', 'melon']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'


class ResidualStreamCapture:
    """Captures residual stream activations at each layer boundary via hooks."""

    def __init__(self, model, full_attn_layer):
        self.model = model
        self.full_attn_layer = full_attn_layer
        self.activations = {}
        self.hooks = []

    def _make_embed_hook(self):
        def hook(module, input, output):
            self.activations['embeddings'] = output.detach().cpu()
        return hook

    def _make_block_hook(self, layer_idx):
        def hook(module, input, output):
            key = f'after_block_{layer_idx}'
            self.activations[key] = output.detach().cpu()
            if layer_idx == self.full_attn_layer:
                self.activations[f'before_fa'] = input[0].detach().cpu()
                self.activations[f'after_fa'] = output.detach().cpu()
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
    """Load model from checkpoint using the train script pattern."""
    spec = importlib.util.spec_from_file_location(
        f'train_script_{arch}',
        os.path.join(root, _TRAIN_SCRIPTS[arch]),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class_name = _MODEL_CLASSES[arch]
    cls = getattr(m, class_name)

    cfg = _ARCH_CONFIGS.get(arch, {})
    D = cfg.get('embedding_dim', getattr(m, 'EMBEDDING_DIM', 512))
    L = cfg.get('num_layers', getattr(m, 'NUM_LAYERS', 6))
    H = cfg.get('num_heads', getattr(m, 'NUM_HEADS', 8))
    F = cfg.get('ffn_dim', getattr(m, 'FFN_DIM', 2048))
    fa = cfg.get('full_attn_layer', getattr(m, 'FULL_ATTN_LAYER', L - 1))
    iv = cfg.get('interference_interval', getattr(m, 'INTERFERENCE', 2))
    vs = getattr(m, 'VOCAB_SIZE', 32000)

    if arch in _IS_STANDARD:
        model = cls(vocab_size=vs, embedding_dim=D, num_layers=L,
                    num_heads=H, ffn_dim=F, seq_len=2048)
    else:
        model = cls(vocab_size=vs, embedding_dim=D, num_layers=L,
                    num_heads=H, ffn_dim=F, seq_len=2048,
                    full_attn_layer=fa, interference_interval=iv)

    abs_ckpt = os.path.join(root, checkpoint_path)
    ck = torch.load(abs_ckpt, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)

    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    val_ppl = ck.get('val_ppl', '?')
    print(f'  Loaded: {checkpoint_path}  (val_ppl={val_ppl})')

    full_attn_layer = fa if not arch in _IS_STANDARD else L - 1
    return model, full_attn_layer, m


def load_tokenizer(m, root):
    """Load tokenizer from the train script module."""
    tokenizer_path = os.path.join(root, 'results/2048_condI_tokenizer.json')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    return m.BPETokenizerWrapper(tok)


def build_passkey_sequence(tokenizer, word, distance, max_seq_len=2048):
    """Build a passkey retrieval sequence with given distance.

    Returns:
        ids: token IDs
        passkey_pos: position of the passkey word (first token)
        cue_pos: position of the retrieval cue (last token before prediction)
    """
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

    passkey_start = len(tokenizer.encode('the secret word is '))
    passkey_pos = passkey_start
    cue_pos = len(full_seq) - 1

    return full_seq, passkey_pos, cue_pos


def compute_norm_stats(activations):
    """Compute mean and std of ||h_t|| for each activation tensor."""
    stats = {}
    for key, act in activations.items():
        norms = act.norm(dim=-1)
        stats[key] = {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
            'per_token_mean': norms.mean(dim=0).numpy(),
        }
    return stats


def compute_cosine_similarity_to_cue(activations, cue_pos):
    """Compute cos_sim(h[cue_pos], h[t]) for all t."""
    similarities = {}
    for key, act in activations.items():
        h_cue = act[0, cue_pos, :]
        h_all = act[0, :, :]

        h_cue_norm = h_cue / (h_cue.norm() + 1e-8)
        h_all_norm = h_all / (h_all.norm(dim=-1, keepdim=True) + 1e-8)

        cos_sim = (h_all_norm @ h_cue_norm).numpy()
        similarities[key] = cos_sim

    return similarities


def compute_fa_delta(activations):
    """Compute ||FA(h_t) - h_t|| / ||h_t|| for the FA layer."""
    if 'before_fa' not in activations or 'after_fa' not in activations:
        return None

    before = activations['before_fa'][0]
    after = activations['after_fa'][0]

    delta = (after - before).norm(dim=-1)
    input_norm = before.norm(dim=-1)

    relative_delta = delta / (input_norm + 1e-8)
    return relative_delta.numpy()


def compute_pca(activations, n_components=2):
    """Project activations to 2D via PCA."""
    pca_results = {}
    for key, act in activations.items():
        h = act[0].numpy()
        projected, explained_variance = pca_transform(h, n_components)
        pca_results[key] = {
            'projected': projected,
            'explained_variance': explained_variance,
        }
    return pca_results


def plot_norm_progression(norm_stats, full_attn_layer, num_layers, out_path):
    """Plot mean ||h_t|| across layers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = ['embeddings'] + [f'after_block_{i}' for i in range(num_layers)]
    means = []
    stds = []
    x_labels = []

    for i, key in enumerate(layers):
        if key in norm_stats:
            means.append(norm_stats[key]['mean'])
            stds.append(norm_stats[key]['std'])
            x_labels.append('embed' if key == 'embeddings' else f'L{i-1}')

    x = np.arange(len(means))
    ax.plot(x, means, 'b-o', linewidth=2, markersize=8, label='Mean ||h||')
    ax.fill_between(x,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.3, color='blue', label='±1 std')

    fa_x = full_attn_layer + 1
    if fa_x < len(x):
        ax.axvline(fa_x, color='red', linestyle='--', linewidth=2,
                   label=f'FullAttn (L{full_attn_layer})')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('||h_t|| (norm)', fontsize=12)
    ax.set_title('Residual Stream Norm Progression Across Layers', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_cosine_heatmap(similarities, passkey_pos, cue_pos, key, out_path, title_suffix=''):
    """Plot cosine similarity heatmap."""
    fig, ax = plt.subplots(figsize=(14, 4))

    cos_sim = similarities[key]
    N = len(cos_sim)

    im = ax.imshow(cos_sim.reshape(1, -1), aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)

    ax.axvline(passkey_pos, color='green', linestyle='--', linewidth=2,
               label=f'Passkey pos={passkey_pos}')
    ax.axvline(cue_pos, color='orange', linestyle='--', linewidth=2,
               label=f'Cue pos={cue_pos}')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Cosine Similarity to Retrieval Cue ({key}){title_suffix}', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.colorbar(im, ax=ax, label='cos_sim', shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_pca_scatter(pca_result, passkey_pos, cue_pos, key, out_path, title_suffix=''):
    """Plot PCA scatter of residual stream."""
    fig, ax = plt.subplots(figsize=(10, 8))

    projected = pca_result['projected']
    N = len(projected)

    colors = np.arange(N)
    scatter = ax.scatter(projected[:, 0], projected[:, 1],
                         c=colors, cmap='coolwarm', alpha=0.6, s=20)

    ax.scatter(projected[passkey_pos, 0], projected[passkey_pos, 1],
               c='green', s=200, marker='*', edgecolors='black',
               linewidths=1.5, label=f'Passkey (pos={passkey_pos})', zorder=5)

    ax.scatter(projected[cue_pos, 0], projected[cue_pos, 1],
               c='orange', s=200, marker='s', edgecolors='black',
               linewidths=1.5, label=f'Cue (pos={cue_pos})', zorder=5)

    var_explained = pca_result['explained_variance']
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} var)', fontsize=12)
    ax.set_title(f'PCA of Residual Stream ({key}){title_suffix}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, label='Token position')
    cbar.ax.set_ylabel('Token position (blue=early, red=late)', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_fa_delta(fa_delta, passkey_pos, cue_pos, out_path):
    """Plot FA delta (relative change) per token position."""
    if fa_delta is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(fa_delta))
    ax.bar(x, fa_delta, width=1.0, alpha=0.7, color='steelblue')

    ax.axvline(passkey_pos, color='green', linestyle='--', linewidth=2,
               label=f'Passkey pos={passkey_pos}')
    ax.axvline(cue_pos, color='orange', linestyle='--', linewidth=2,
               label=f'Cue pos={cue_pos}')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('||FA(h) - h|| / ||h||', fontsize=12)
    ax.set_title('FullAttn Relative Delta Per Token Position', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    if passkey_pos < len(fa_delta):
        pk_val = fa_delta[passkey_pos]
        ax.annotate(f'{pk_val:.2f}', xy=(passkey_pos, pk_val),
                    xytext=(passkey_pos + 10, pk_val + 0.05),
                    fontsize=9, color='green')
    if cue_pos < len(fa_delta):
        cue_val = fa_delta[cue_pos]
        ax.annotate(f'{cue_val:.2f}', xy=(cue_pos, cue_val),
                    xytext=(cue_pos - 30, cue_val + 0.05),
                    fontsize=9, color='orange')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_summary(norm_stats, pca_results, fa_delta, similarities,
                 full_attn_layer, num_layers, passkey_pos, cue_pos, out_path):
    """Create a 4-panel summary visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    layers = ['embeddings'] + [f'after_block_{i}' for i in range(num_layers)]
    means = []
    stds = []
    for key in layers:
        if key in norm_stats:
            means.append(norm_stats[key]['mean'])
            stds.append(norm_stats[key]['std'])
    x = np.arange(len(means))
    ax1.plot(x, means, 'b-o', linewidth=2, markersize=6)
    ax1.fill_between(x, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.3)
    fa_x = full_attn_layer + 1
    if fa_x < len(x):
        ax1.axvline(fa_x, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('||h||')
    ax1.set_title('Norm Progression')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if fa_delta is not None:
        ax2.bar(np.arange(len(fa_delta)), fa_delta, width=1.0, alpha=0.7)
        ax2.axvline(passkey_pos, color='green', linestyle='--', linewidth=2)
        ax2.axvline(cue_pos, color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('||FA(h) - h|| / ||h||')
    ax2.set_title('FA Delta by Position')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 0])
    before_key = 'before_fa'
    if before_key in pca_results:
        proj = pca_results[before_key]['projected']
        colors = np.arange(len(proj))
        ax3.scatter(proj[:, 0], proj[:, 1], c=colors, cmap='coolwarm', alpha=0.5, s=10)
        ax3.scatter(proj[passkey_pos, 0], proj[passkey_pos, 1],
                   c='green', s=100, marker='*', edgecolors='black', zorder=5)
        ax3.scatter(proj[cue_pos, 0], proj[cue_pos, 1],
                   c='orange', s=100, marker='s', edgecolors='black', zorder=5)
    ax3.set_title('PCA Before FA')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    after_key = 'after_fa'
    if after_key in pca_results:
        proj = pca_results[after_key]['projected']
        colors = np.arange(len(proj))
        ax4.scatter(proj[:, 0], proj[:, 1], c=colors, cmap='coolwarm', alpha=0.5, s=10)
        ax4.scatter(proj[passkey_pos, 0], proj[passkey_pos, 1],
                   c='green', s=100, marker='*', edgecolors='black', zorder=5)
        ax4.scatter(proj[cue_pos, 0], proj[cue_pos, 1],
                   c='orange', s=100, marker='s', edgecolors='black', zorder=5)
    ax4.set_title('PCA After FA')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Residual Stream Summary (FA at L{full_attn_layer}, '
                 f'passkey@{passkey_pos}, cue@{cue_pos})', fontsize=14, y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize residual stream in hybrid DSQG models')
    parser.add_argument('--arch', required=True, choices=list(_TRAIN_SCRIPTS.keys()),
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--out', required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--passkey_distance', type=int, default=256,
                        help='Distance for passkey retrieval test')
    parser.add_argument('--n_sequences', type=int, default=5,
                        help='Number of passkey sequences to analyze')
    parser.add_argument('--device', default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)

    os.makedirs(args.out, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Residual Stream Visualization — {args.arch}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Passkey distance: {args.passkey_distance}')
    print(f'{"="*60}\n')

    print('Loading model...')
    model, full_attn_layer, m = load_model(
        args.arch, args.checkpoint, root, device)

    print('Loading tokenizer...')
    tokenizer = load_tokenizer(m, root)

    num_layers = len(model.blocks)
    print(f'  Model has {num_layers} layers, FA at layer {full_attn_layer}')

    capture = ResidualStreamCapture(model, full_attn_layer)
    capture.register_hooks()

    all_norm_stats = []
    all_similarities = []
    all_fa_deltas = []
    all_pca_results = []

    for seq_idx in range(args.n_sequences):
        word = _PASSKEY_WORDS[seq_idx % len(_PASSKEY_WORDS)]

        print(f'\nSequence {seq_idx + 1}/{args.n_sequences}: word="{word}"')

        ids, passkey_pos, cue_pos = build_passkey_sequence(
            tokenizer, word, args.passkey_distance)

        print(f'  Sequence length: {len(ids)}, passkey_pos={passkey_pos}, cue_pos={cue_pos}')

        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        capture.clear()
        with torch.no_grad():
            _ = model(ids_tensor)

        norm_stats = compute_norm_stats(capture.activations)
        all_norm_stats.append(norm_stats)

        similarities = compute_cosine_similarity_to_cue(
            capture.activations, cue_pos)
        all_similarities.append(similarities)

        fa_delta = compute_fa_delta(capture.activations)
        if fa_delta is not None:
            all_fa_deltas.append(fa_delta)

        pca_results = compute_pca(capture.activations)
        all_pca_results.append({
            'pca': pca_results,
            'passkey_pos': passkey_pos,
            'cue_pos': cue_pos,
            'word': word,
        })

        if seq_idx == 0:
            plot_norm_progression(
                norm_stats, full_attn_layer, num_layers,
                os.path.join(args.out, 'norm_progression.png')
            )

            for key in ['before_fa', 'after_fa']:
                if key in similarities:
                    plot_cosine_heatmap(
                        similarities, passkey_pos, cue_pos, key,
                        os.path.join(args.out, f'cosim_{key}.png')
                    )

            for key in ['before_fa', 'after_fa']:
                if key in pca_results:
                    plot_pca_scatter(
                        pca_results[key], passkey_pos, cue_pos, key,
                        os.path.join(args.out, f'pca_{key}.png')
                    )

            plot_fa_delta(
                fa_delta, passkey_pos, cue_pos,
                os.path.join(args.out, 'fa_delta.png')
            )

            plot_summary(
                norm_stats, pca_results, fa_delta, similarities,
                full_attn_layer, num_layers, passkey_pos, cue_pos,
                os.path.join(args.out, 'summary.png')
            )

    capture.remove_hooks()

    print('\nComputing aggregate statistics...')

    agg_norm_stats = {}
    for key in all_norm_stats[0].keys():
        means = [ns[key]['mean'] for ns in all_norm_stats if key in ns]
        stds = [ns[key]['std'] for ns in all_norm_stats if key in ns]
        agg_norm_stats[key] = {
            'mean': float(np.mean(means)),
            'std': float(np.mean(stds)),
        }

    avg_fa_delta_at_passkey = []
    avg_fa_delta_at_cue = []
    for i, fd in enumerate(all_fa_deltas):
        pp = all_pca_results[i]['passkey_pos']
        cp = all_pca_results[i]['cue_pos']
        if pp < len(fd):
            avg_fa_delta_at_passkey.append(fd[pp])
        if cp < len(fd):
            avg_fa_delta_at_cue.append(fd[cp])

    summary_data = {
        'arch': args.arch,
        'checkpoint': args.checkpoint,
        'passkey_distance': args.passkey_distance,
        'n_sequences': args.n_sequences,
        'full_attn_layer': full_attn_layer,
        'num_layers': num_layers,
        'norm_progression': agg_norm_stats,
        'avg_fa_delta_at_passkey': float(np.mean(avg_fa_delta_at_passkey)) if avg_fa_delta_at_passkey else None,
        'avg_fa_delta_at_cue': float(np.mean(avg_fa_delta_at_cue)) if avg_fa_delta_at_cue else None,
    }

    summary_path = os.path.join(args.out, 'residual_stream_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f'  Saved: {summary_path}')

    print(f'\nDone. Visualizations saved to {args.out}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
