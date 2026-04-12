"""
visualize_attention.py — DSQG vs Full Attention visualization.

Runs text through a condU checkpoint and generates:
  1. DSQG attention combs — per-head heatmaps (position × offset)
  2. Full causal attention — standard position × position heatmap
  3. pos_bias heatmap — learned frequency prior [44 offsets × H heads]
  4. scale_embed PCA — what Q directions each offset responds to
  5. IF gain trajectory — per-head amplification values

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/visualize_attention.py \
      --checkpoint checkpoints/condU/best.pt \
      --text "The quick brown fox jumps over the lazy dog. In the beginning was the word." \
      --out_dir tools/viz_output/

  # Passkey-style sequence (shows long-range retrieval)
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/visualize_attention.py \
      --checkpoint checkpoints/condU/best.pt \
      --passkey --out_dir tools/viz_output/
"""

import sys, os, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from model_registry import (
    MODEL_REGISTRY, build_model, load_checkpoint, get_tokenizer, get_offsets,
    J44_OFFSETS, SE015_OFFSETS, SE096_OFFSETS, REPO,
)
from kernels.topk_sparse_attn import TopKSparseAttention


def detect_arch_from_results(results_json_path: str) -> str | None:
    """Inspect a results JSON for an 'experiment' key and return matching arch name."""
    try:
        with open(results_json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    experiment = data.get('experiment', '')
    lowered = experiment.lower().replace(' ', '_')
    if 'd41_35m' in lowered or 'd41 35m' in experiment.lower():
        return 'd41_35m'
    if 'd41s3' in lowered:
        return 'd41s3'
    if 'd41s5' in lowered:
        return 'd41s5'
    if 'condu' in lowered:
        return 'condu'
    return None


# ---------------------------------------------------------------------------
# Pure Python DSQG forward — computes attention weights (not just output)
# ---------------------------------------------------------------------------

def dsqg_attention_weights(q, k, pos_bias, scale_embed, offsets):
    """
    Compute DSQG attention weight matrix over the given offset set.

    Args:
        q:           [B, H, N, HD]   query vectors
        k:           [B, H, N, HD]   key vectors
        pos_bias:    [J, H]          learned frequency prior
        scale_embed: [J, HD]         Q-weighted matched filter
        offsets:     list[int]       offset values

    Returns:
        alpha: [B, H, N, J]   attention weights (softmax over offsets)
        score: [B, H, N, J]   raw scores (pre-softmax)
    """
    B, H, N, HD = q.shape
    J = len(offsets)
    device = q.device
    INF = float('-inf')

    max_offset = offsets[-1]
    k_padded = F.pad(k, (0, 0, max_offset, 0))

    scores = torch.full((B, H, N, J), INF, device=device)

    for j_idx, delta in enumerate(offsets):
        if delta == 0:
            valid_n = slice(None)
            k_j = k_padded[:, :, max_offset:, :]
        else:
            k_j = k_padded[:, :, max_offset - delta: max_offset - delta + N, :]
            valid_n = torch.arange(N, device=device) >= delta

        qk = (q * k_j).sum(-1) / math.sqrt(HD)
        pb = pos_bias[j_idx].view(1, H, 1).expand(B, H, N)
        se = (q * scale_embed[j_idx].view(1, 1, 1, HD)).sum(-1) / math.sqrt(HD)
        raw = qk + pb + se

        if isinstance(valid_n, slice):
            scores[:, :, :, j_idx] = raw
        else:
            valid_mask = valid_n.view(1, 1, N)
            scores[:, :, :, j_idx] = torch.where(valid_mask, raw, torch.tensor(INF, device=device))

    alpha = torch.softmax(scores, dim=-1)
    alpha = torch.nan_to_num(alpha, nan=0.0)

    return alpha, scores


def capture_full_attention(model, x, full_layer_idx):
    """Hook into full attention layer to capture attention weights.

    Handles two projection styles:
      - Standard: combined qkv_proj (FullCausalAttention)
      - Bypass:   separate q_proj + kv_proj (FullCausalAttentionBypass)
    """
    weights_store = {}

    def hook_fn(module, inp, out):
        with torch.no_grad():
            h_inp = inp[0]                                   # [B, N, D]
            B, N, D = h_inp.shape
            H = module.num_heads
            HD = D // H
            scale = math.sqrt(HD)

            if hasattr(module, 'qkv_proj'):
                qkv = module.qkv_proj(h_inp)
                q, k, _ = qkv.split(D, dim=-1)
            elif hasattr(module, 'q_proj') and hasattr(module, 'kv_proj'):
                q = module.q_proj(h_inp)
                k, _ = module.kv_proj(h_inp).split(D, dim=-1)
            else:
                return

            q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
            k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
            mask = torch.triu(torch.full((N, N), float('-inf'), device=h_inp.device), diagonal=1)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale + mask
            attn_w = torch.softmax(attn, dim=-1)
            weights_store['full_attn'] = attn_w.detach().cpu()

    handle = model.blocks[full_layer_idx].attn.register_forward_hook(hook_fn)
    return weights_store, handle


def capture_topk_attention(model, topk_layer_idx, topk_k):
    """Hook into TopK attention layer to capture sparse attention weights.

    Returns the full softmax-after-topk-mask weights [B, H, N, N].
    """
    weights_store = {}

    def hook_fn(module, inp, out):
        with torch.no_grad():
            h_inp = inp[0]
            B, N, D = h_inp.shape
            H, hd = module.H, module.hd

            def reshape(t):
                return t.reshape(B, N, H, hd).transpose(1, 2)

            Q = reshape(module.W_q(h_inp))
            K = reshape(module.W_k(h_inp))
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)

            causal_mask = torch.triu(
                torch.ones(N, N, device=h_inp.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask[None, None], float('-inf'))

            if topk_k < N:
                topk_vals, _ = scores.topk(topk_k, dim=-1)
                threshold = topk_vals[..., -1:]
                scores = scores.masked_fill(scores < threshold, float('-inf'))

            attn_w = torch.softmax(scores, dim=-1)
            weights_store['topk_attn'] = attn_w.detach().cpu()

    handle = model.blocks[topk_layer_idx].attn.register_forward_hook(hook_fn)
    return weights_store, handle


def plot_topk_attention(attn_w, topk_k, title_prefix, out_path):
    """Plot TopK sparse attention matrix — H subplots showing sparsity pattern."""
    B, H, N, _ = attn_w.shape
    N_show = min(N, 256)
    ncols = min(H, 4)
    nrows = math.ceil(H / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if H == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for h in range(H):
        ax = axes[h]
        img = attn_w[0, h, :N_show, :N_show].numpy()
        im = ax.imshow(img, aspect='auto', origin='upper',
                       cmap='hot', vmin=0, vmax=img.max() * 0.95 + 1e-8)
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xlabel('Key position', fontsize=8)
        ax.set_ylabel('Query position', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6)

    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f'{title_prefix}\nTopK Sparse Attention (k={topk_k})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def offset_labels(offsets, max_labels=20):
    """Nice labels for the offset axis — don't crowd."""
    step = max(1, len(offsets) // max_labels)
    labels = []
    for i, d in enumerate(offsets):
        if i % step == 0 or d in [0, 1, 32, 48, 256, 1024, 1536]:
            labels.append((i, str(d)))
    return labels


def plot_dsqg_combs(alpha, offsets, title_prefix, out_path, token_texts=None):
    """
    Plot DSQG attention comb — one subplot per head.
    alpha: [H, N, 44]
    """
    H, N, J = alpha.shape
    ncols = min(H, 4)
    nrows = math.ceil(H / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if H == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    # Show a window where all offset groups have valid targets.
    # For triadic J=96, GROUP_C offsets go up to ~2048; we want positions
    # where even the longest offsets have fired. Use the last 256 positions
    # of the sequence (or all if N<=256).
    N_show = min(N, 256)
    start = max(0, N - N_show)  # last N_show positions
    alpha_show = alpha[:, start:start + N_show, :]  # [H, N_show, J]

    label_positions, label_texts = zip(*offset_labels(offsets))

    for h in range(H):
        ax = axes[h]
        img = alpha_show[h].T.numpy()   # [J, N_show]
        im = ax.imshow(img, aspect='auto', origin='lower',
                       cmap='hot', vmin=0, vmax=img.max() * 0.9 + 1e-8)
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xlabel(f'Token position (t={start}–{start+N_show-1})', fontsize=8)
        ax.set_ylabel('Offset δ', fontsize=8)
        ax.set_yticks(list(label_positions))
        ax.set_yticklabels(list(label_texts), fontsize=7)

        # Dividing line between dense local and long-range (use last offset ≤32, else largest <64)
        _bnd = 32 if 32 in offsets else max((o for o in offsets if o < 64), default=offsets[-1])
        dense_last = offsets.index(_bnd)
        ax.axhline(dense_last + 0.5, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.6)

    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f'{title_prefix}\nDSQG Attention Weights (position × offset)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_full_attention(attn_w, title_prefix, out_path):
    """
    Plot full causal attention matrix — H subplots.
    attn_w: [B, H, N, N]
    """
    B, H, N, _ = attn_w.shape
    N_show = min(N, 256)
    ncols = min(H, 4)
    nrows = math.ceil(H / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if H == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for h in range(H):
        ax = axes[h]
        img = attn_w[0, h, :N_show, :N_show].numpy()
        im = ax.imshow(img, aspect='auto', origin='upper',
                       cmap='Blues', vmin=0, vmax=img.max() * 0.95 + 1e-8)
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xlabel('Key position', fontsize=8)
        ax.set_ylabel('Query position', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6)

    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f'{title_prefix}\nFull Causal Attention Weights (position × position)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_pos_bias(pos_bias, offsets, out_path):
    """pos_bias [44, H] — learned frequency prior per head."""
    pb = pos_bias.numpy()  # [44, H]
    H = pb.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, H * 1.2), 7))
    im = ax.imshow(pb, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-abs(pb).max(), vmax=abs(pb).max())
    ax.set_xlabel('Head', fontsize=10)
    ax.set_ylabel('Offset δ', fontsize=10)
    ax.set_xticks(range(H))
    ax.set_xticklabels([f'h{h}' for h in range(H)], fontsize=8)

    label_positions, label_texts = zip(*offset_labels(offsets))
    ax.set_yticks(list(label_positions))
    ax.set_yticklabels(list(label_texts), fontsize=8)

    _bnd2 = 32 if 32 in offsets else max((o for o in offsets if o < 64), default=offsets[-1])
    dense_last = offsets.index(_bnd2)
    ax.axhline(dense_last + 0.5, color='black', linewidth=1.2, linestyle='--', alpha=0.6,
               label='Dense/dyadic boundary')
    ax.legend(fontsize=8, loc='upper right')

    plt.colorbar(im, ax=ax, shrink=0.8, label='pos_bias value')
    ax.set_title('Learned pos_bias [44 offsets × H heads]\nRed=positive (attend here), Blue=negative (avoid)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_if_gains(if_gains_per_layer, out_path):
    """Plot IF gains [layers × H] as grouped bars."""
    layers = list(if_gains_per_layer.keys())
    gains_arr = np.array([if_gains_per_layer[l] for l in layers])  # [L, H]
    H = gains_arr.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, H * 1.5), 5))
    x = np.arange(H)
    width = 0.8 / len(layers)
    for i, l in enumerate(layers):
        bars = ax.bar(x + i * width, gains_arr[i], width, label=f'Layer {l}', alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='init (1.0)')
    ax.set_xlabel('Head', fontsize=10)
    ax.set_ylabel('IF gain', fontsize=10)
    ax.set_xticks(x + width * len(layers) / 2)
    ax.set_xticklabels([f'h{h}' for h in range(H)], fontsize=9)
    ax.set_title('Per-head IF Amplifier Gains\nAbove 1.0 = amplifying (global heads), Below = attenuating (local heads)',
                 fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_side_by_side(dsqg_alpha, full_attn, offsets, out_path):
    """
    The money shot: DSQG comb vs full attention, single head each, side by side.
    Shows the structural difference visually.
    """
    N_show = min(dsqg_alpha.shape[1], 256)
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig, wspace=0.35)

    # DSQG — mean over heads, position × offset
    dsqg_mean = dsqg_alpha[:, :N_show, :].mean(0).T.numpy()  # [44, N_show]
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(dsqg_mean, aspect='auto', origin='lower', cmap='hot',
                     vmin=0, vmax=dsqg_mean.max() * 0.9 + 1e-8)
    ax1.set_title('DSQG Attention\n(position × offset — the COMB)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Token position', fontsize=10)
    ax1.set_ylabel('Attention offset δ', fontsize=10)
    label_positions, label_texts = zip(*offset_labels(offsets))
    ax1.set_yticks(list(label_positions))
    ax1.set_yticklabels(list(label_texts), fontsize=8)
    # Find boundary between dense-local and long-range offsets.
    # Use the last offset ≤ 32 if 32 is present, otherwise the largest offset < 64.
    _boundary = 32 if 32 in offsets else max((o for o in offsets if o < 64), default=offsets[-1])
    dense_last = offsets.index(_boundary)
    ax1.axhline(dense_last + 0.5, color='cyan', linewidth=1, linestyle='--', alpha=0.8)
    ax1.text(N_show * 0.02, dense_last - 2, 'dense\nlocal', color='cyan', fontsize=7)
    ax1.text(N_show * 0.02, dense_last + 2, 'dyadic\nlong', color='cyan', fontsize=7)

    # Full attention — mean over heads, position × position
    full_mean = full_attn[0, :, :N_show, :N_show].mean(0).numpy()  # [N_show, N_show]
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(full_mean, aspect='auto', origin='upper', cmap='Blues',
                     vmin=0, vmax=full_mean.max() * 0.95 + 1e-8)
    ax2.set_title('Full Causal Attention\n(position × position — the CLOUD)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key position', fontsize=10)
    ax2.set_ylabel('Query position', fontsize=10)

    # Shared colorbar
    cax = fig.add_subplot(gs[2])
    fig.colorbar(im1, cax=cax, label='Attention weight')

    fig.suptitle('DWARF DSQG vs Standard Full Attention — Structural Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_and_extract(checkpoint_path, arch, ids_tensor, device, root):
    """
    Load a checkpoint for the given arch, run inference, and return all
    extracted visualization data.

    Returns dict with keys: dsqg_alphas, dsqg_layers, full_attn,
    if_gains, pos_bias, checkpoint_name, offsets.
    """
    cfg = MODEL_REGISTRY[arch]
    offsets = get_offsets(arch)

    model, run_cfg, m = build_model(arch, device=device, checkpoint_path=checkpoint_path)
    val_ppl = run_cfg.get('_val_ppl', '?')

    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if '/' in checkpoint_path:
        ck_dir = os.path.basename(os.path.dirname(checkpoint_path))
        checkpoint_name = f'{ck_dir}_{checkpoint_name}'

    N = ids_tensor.shape[1]

    # ── Standard transformer path (no DSQG) ──────────────────────────────────
    if cfg.get('is_standard', False):
        last_block_idx = len(model.blocks) - 1
        full_weights_store = {}
        def _std_hook(module, inp, out):
            with torch.no_grad():
                h_inp = inp[0]
                B2, N2, D2 = h_inp.shape
                q2, k2, _ = module.qkv_proj(h_inp).split(D2, dim=-1)
                H2 = module.num_heads; HD2 = D2 // H2
                q2 = q2.view(B2, N2, H2, HD2).permute(0, 2, 1, 3)
                k2 = k2.view(B2, N2, H2, HD2).permute(0, 2, 1, 3)
                scale2 = math.sqrt(HD2)
                mask2 = torch.triu(torch.full((N2, N2), float('-inf'), device=h_inp.device), diagonal=1)
                attn2 = torch.matmul(q2, k2.transpose(-2, -1)) / scale2 + mask2
                full_weights_store['full_attn'] = torch.softmax(attn2, dim=-1).detach().cpu()
        handle = model.blocks[last_block_idx].attn.register_forward_hook(_std_hook)
        with torch.no_grad():
            _ = model(ids_tensor)
        handle.remove()
        print(f'  Standard transformer: captured full attention on layer {last_block_idx}')
        return {
            'dsqg_alphas': {}, 'dsqg_layers': [], 'if_gains': {}, 'pos_bias': None,
            'full_attn': full_weights_store.get('full_attn'),
            'checkpoint_name': checkpoint_name, 'offsets': [], 'val_ppl': val_ppl,
        }

    # ── DSQG path (all other archs) ───────────────────────────────────────────
    is_topk = cfg.get('is_topk', False)

    with torch.no_grad():
        pos = torch.arange(N, device=device).unsqueeze(0)

        full_weights_store = {}
        topk_weights_store = {}
        hook_handle = None
        topk_hook_handle = None

        if is_topk:
            full_layer_idx = model.full_attn_layer
            topk_k = cfg.get('topk_k', 64)
            topk_weights_store, topk_hook_handle = capture_topk_attention(
                model, full_layer_idx, topk_k)
        elif not cfg.get('no_full_attn', False):
            if hasattr(model, 'full_attn_layers'):
                full_layer_idx = model.full_attn_layers[-1]
            else:
                full_layer_idx = model.full_attn_layer
            full_weights_store, hook_handle = capture_full_attention(model, None, full_layer_idx)

        _ = model(ids_tensor)
        if hook_handle is not None:
            hook_handle.remove()
        if topk_hook_handle is not None:
            topk_hook_handle.remove()

        if hasattr(model, 'pos_embed'):
            x_running = model.embedding(ids_tensor) + model.pos_embed(pos)
        else:
            x_running = model.embedding(ids_tensor)
        x_running = model.drop(x_running)

        dsqg_alphas = {}
        dsqg_layers = []
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
                attn = block.attn
                qkv = attn.qkv_proj(x_running)
                D = x_running.shape[-1]
                q, k, v = qkv.split(D, dim=-1)
                H = attn.num_heads
                HD = D // H
                q = q.view(*x_running.shape[:2], H, HD).permute(0, 2, 1, 3)
                k = k.view(*x_running.shape[:2], H, HD).permute(0, 2, 1, 3)

                # Per-layer offsets for triadic; global offsets otherwise
                layer_offsets = attn.offsets_dev.tolist() if hasattr(attn, 'offsets_dev') else offsets

                if cfg.get('no_scale_embed', False) or not hasattr(attn, 'scale_embed'):
                    J = len(layer_offsets)
                    se = torch.zeros(J, HD)
                else:
                    se = attn.scale_embed.float().cpu()

                alpha, _ = dsqg_attention_weights(
                    q.float().cpu(), k.float().cpu(),
                    attn.pos_bias.float().cpu(),
                    se,
                    layer_offsets,
                )

                dsqg_alphas[i] = alpha[0].cpu()
                dsqg_layers.append(i)
                print(f'  Layer {i} DSQG: alpha shape {alpha.shape}, '
                      f'max={alpha.max():.4f}, mean={alpha.mean():.4f}')

            x_running = block(x_running)

    if_gains = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn') and hasattr(block.attn, 'if_gain'):
            if_gains[i] = block.attn.if_gain.detach().cpu().tolist()

    pos_bias = None
    for block in model.blocks:
        if hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias'):
            pos_bias = block.attn.pos_bias.detach().cpu()
            break

    return {
        'dsqg_alphas': dsqg_alphas,
        'dsqg_layers': dsqg_layers,
        'full_attn': full_weights_store.get('full_attn'),
        'topk_attn': topk_weights_store.get('topk_attn') if is_topk else None,
        'is_topk': is_topk,
        'topk_k': cfg.get('topk_k', 64) if is_topk else None,
        'if_gains': if_gains,
        'pos_bias': pos_bias,
        'checkpoint_name': checkpoint_name,
        'offsets': offsets,
        'val_ppl': val_ppl,
    }


def plot_compare_checkpoints(data_a, data_b, label_a, label_b, out_path):
    """
    2-column comparison: left = checkpoint A, right = checkpoint B.
    Each column shows DSQG attention (mean heads), pos_bias, and IF gains.
    """
    rows = 3
    fig, axes = plt.subplots(rows, 2, figsize=(20, rows * 5))

    for col, (data, label) in enumerate([(data_a, label_a), (data_b, label_b)]):
        offsets = data['offsets']
        label_positions, label_texts = zip(*offset_labels(offsets))

        # Row 0: DSQG attention (mean over heads, first DSQG layer)
        ax = axes[0][col]
        first_layer = data['dsqg_layers'][0]
        dsqg_mean = data['dsqg_alphas'][first_layer]
        N_show = min(dsqg_mean.shape[1], 256)
        img = dsqg_mean[:, :N_show, :].mean(0).T.numpy()
        im = ax.imshow(img, aspect='auto', origin='lower', cmap='hot',
                       vmin=0, vmax=img.max() * 0.9 + 1e-8)
        ax.set_title(f'{label}\nDSQG Attention (layer {first_layer}, mean heads)', fontsize=11)
        ax.set_xlabel('Token position', fontsize=9)
        ax.set_ylabel('Offset δ', fontsize=9)
        ax.set_yticks(list(label_positions))
        ax.set_yticklabels(list(label_texts), fontsize=7)
        if 32 in offsets:
            dense_last = offsets.index(32)
            ax.axhline(dense_last + 0.5, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.6)

        # Row 1: pos_bias
        ax = axes[1][col]
        pb = data['pos_bias'].numpy()
        H = pb.shape[1]
        im = ax.imshow(pb, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-abs(pb).max(), vmax=abs(pb).max())
        ax.set_title(f'{label}\npos_bias [{pb.shape[0]} offsets × {H} heads]', fontsize=11)
        ax.set_xlabel('Head', fontsize=9)
        ax.set_ylabel('Offset δ', fontsize=9)
        ax.set_xticks(range(H))
        ax.set_xticklabels([f'h{h}' for h in range(H)], fontsize=8)
        ax.set_yticks(list(label_positions))
        ax.set_yticklabels(list(label_texts), fontsize=7)
        if 32 in offsets:
            dense_last = offsets.index(32)
            ax.axhline(dense_last + 0.5, color='black', linewidth=1, linestyle='--', alpha=0.6)
        plt.colorbar(im, ax=ax, shrink=0.6)

        # Row 2: IF gains
        ax = axes[2][col]
        if data['if_gains']:
            layers = list(data['if_gains'].keys())
            gains_arr = np.array([data['if_gains'][l] for l in layers])
            H = gains_arr.shape[1]
            x = np.arange(H)
            width = 0.8 / len(layers)
            for i, l in enumerate(layers):
                ax.bar(x + i * width, gains_arr[i], width, label=f'Layer {l}', alpha=0.8)
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='init (1.0)')
            ax.set_xlabel('Head', fontsize=9)
            ax.set_ylabel('IF gain', fontsize=9)
            ax.set_xticks(x + width * len(layers) / 2)
            ax.set_xticklabels([f'h{h}' for h in range(H)], fontsize=8)
            ax.legend(fontsize=7)
        ax.set_title(f'{label}\nIF Amplifier Gains', fontsize=11)

    fig.suptitle('Checkpoint Comparison: A vs B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def build_passkey_text(tokenizer, n_tokens=2000, key_pos=512, key_val=42):
    """Build a passkey-retrieval sequence for visualization.

    key_pos: approximate token position for the secret number insertion.
    We build ~2000 tokens so all triadic groups (GROUP_A local, GROUP_B mid
    δ≥48, GROUP_C long δ≥384) have valid KV targets visible in the window.
    """
    filler = "The meeting was scheduled for the following week. All participants were expected to attend and review the materials beforehand. "
    # Encode filler to measure tokens per character
    filler_ids = tokenizer.encode(filler)
    chars_per_tok = len(filler) / max(len(filler_ids), 1)
    target_chars = int(n_tokens * chars_per_tok)
    key_char_pos = int(key_pos * chars_per_tok)
    filler_long = (filler * ((target_chars // len(filler)) + 2))[:target_chars]
    parts = [
        filler_long[:key_char_pos],
        f" REMEMBER THE NUMBER {key_val}. ",
        filler_long[key_char_pos:],
        f" What was the number? The number was {key_val}.",
    ]
    return ''.join(parts)


def _resolve_arch(explicit_arch, results_path, fallback_arch, root):
    """Resolve architecture: explicit flag > auto-detect from results > fallback."""
    if explicit_arch is not None:
        return explicit_arch
    if results_path is not None:
        full_path = os.path.join(root, results_path) if not os.path.isabs(results_path) else results_path
        detected = detect_arch_from_results(full_path)
        if detected is not None:
            print(f'  Auto-detected arch={detected} from {results_path}')
            return detected
    return fallback_arch


def _run_compare(args, root):
    """Execute --compare mode: load two checkpoints and render side-by-side."""
    os.makedirs(os.path.join(root, args.out_dir), exist_ok=True)
    device = torch.device(args.device)

    arch_a = _resolve_arch(args.arch_a, args.results_a, args.arch, root)
    arch_b = _resolve_arch(args.arch_b, args.results_b, args.arch, root)

    import importlib.util
    from model_registry import _import_train_script
    m = _import_train_script(MODEL_REGISTRY[arch_a]['train_script'])
    tok = get_tokenizer(arch_a)
    tokenizer = m.BPETokenizerWrapper(tok)

    if args.passkey:
        text = build_passkey_text(tokenizer)
    elif args.text:
        text = args.text
    else:
        text = (
            "In the distant future, humanity had spread across the stars. "
            "The great archive of Alexandria once held the sum of ancient knowledge. "
            "Scientists discovered that the key to everything was hidden in plain sight. "
            "Remember the passkey: seven four two. After many years of searching, "
            "they finally found the answer. The passkey was seven four two. "
        ) * 4

    ids = tokenizer.encode(text)
    ids_tensor = torch.tensor([ids[:2047]], dtype=torch.long, device=device)
    N = ids_tensor.shape[1]
    print(f'  Sequence length: {N} tokens')

    print(f'\n📊 Loading checkpoint A: {args.compare[0]}  (arch={arch_a})')
    data_a = load_and_extract(args.compare[0], arch_a, ids_tensor, device, root)

    print(f'\n📊 Loading checkpoint B: {args.compare[1]}  (arch={arch_b})')
    data_b = load_and_extract(args.compare[1], arch_b, ids_tensor, device, root)

    out_dir = os.path.join(root, args.out_dir)
    label_a = f'{arch_a} / {data_a["checkpoint_name"]} (PPL={data_a["val_ppl"]})'
    label_b = f'{arch_b} / {data_b["checkpoint_name"]} (PPL={data_b["val_ppl"]})'

    name_a = data_a['checkpoint_name']
    name_b = data_b['checkpoint_name']
    out_path = os.path.join(out_dir, f'compare_{name_a}_vs_{name_b}.jpg')

    plot_compare_checkpoints(data_a, data_b, label_a, label_b, out_path)
    print(f'\nDone. Comparison saved to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='condu', choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--checkpoint', default=None)
    checkpoint_group.add_argument('--compare', nargs=2, metavar=('CHECKPOINT_A', 'CHECKPOINT_B'),
                                  help='Compare two checkpoints side by side')

    parser.add_argument('--arch_a', default=None, choices=list(MODEL_REGISTRY.keys()),
                        help='Architecture for checkpoint A (compare mode)')
    parser.add_argument('--arch_b', default=None, choices=list(MODEL_REGISTRY.keys()),
                        help='Architecture for checkpoint B (compare mode)')
    parser.add_argument('--results_a', default=None,
                        help='Results JSON for checkpoint A (auto-detect arch)')
    parser.add_argument('--results_b', default=None,
                        help='Results JSON for checkpoint B (auto-detect arch)')

    parser.add_argument('--text', default=None,
                        help='Text to run through the model')
    parser.add_argument('--passkey', action='store_true',
                        help='Use passkey-style text instead')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_dir', default='tools/viz_output')
    parser.add_argument('--layer', type=int, default=None,
                        help='Which DSQG layer to visualize (default: all)')
    args = parser.parse_args()

    if args.checkpoint is None and args.compare is None:
        args.checkpoint = 'checkpoints/condU/best.pt'

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.compare:
        _run_compare(args, root)
        return

    arch_offsets = get_offsets(args.arch)

    os.makedirs(os.path.join(root, args.out_dir), exist_ok=True)
    device = torch.device(args.device)

    print('Loading model...')
    from model_registry import _import_train_script
    m = _import_train_script(MODEL_REGISTRY[args.arch]['train_script'])
    model, run_cfg, _m = build_model(args.arch, device=device, checkpoint_path=args.checkpoint)

    tok = get_tokenizer(args.arch)
    tokenizer = m.BPETokenizerWrapper(tok)

    if args.passkey:
        text = build_passkey_text(tokenizer)
    elif args.text:
        text = args.text
    else:
        text = (
            "In the distant future, humanity had spread across the stars. "
            "The great archive of Alexandria once held the sum of ancient knowledge. "
            "Scientists discovered that the key to everything was hidden in plain sight. "
            "Remember the passkey: seven four two. After many years of searching, "
            "they finally found the answer. The passkey was seven four two. "
        ) * 4

    ids = tokenizer.encode(text)
    ids_t = torch.tensor([ids[:2047]], dtype=torch.long, device=device)
    N = ids_t.shape[1]
    print(f'  Sequence length: {N} tokens')

    print('Extracting attention weights...')

    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if '/' in args.checkpoint:
        ck_dir = os.path.basename(os.path.dirname(args.checkpoint))
        checkpoint_name = f'{ck_dir}_{checkpoint_name}'

    extracted = load_and_extract(args.checkpoint, args.arch, ids_t, device, root)
    dsqg_alphas = extracted['dsqg_alphas']
    dsqg_layers = extracted['dsqg_layers']
    full_attn_store = extracted['full_attn']
    topk_attn_store = extracted['topk_attn']
    if_gains_per_layer = extracted['if_gains']
    pos_bias_l0 = extracted['pos_bias']
    full_weights_store = {'full_attn': full_attn_store}
    full_layer_idx = getattr(model, 'full_attn_layer', len(model.blocks) - 1)

    out_dir = os.path.join(root, args.out_dir)
    print(f'\nGenerating visualizations in {out_dir}/')

    if topk_attn_store is not None:
        topk_k = extracted['topk_k']
        plot_topk_attention(
            topk_attn_store,
            topk_k,
            f'Layer {full_layer_idx} (TopK Sparse)',
            os.path.join(out_dir, f'{checkpoint_name}_topk_attn.jpg')
        )

    if dsqg_layers and full_weights_store.get('full_attn') is not None:
        mean_dsqg = dsqg_alphas[dsqg_layers[0]]
        plot_side_by_side(
            mean_dsqg,
            full_weights_store['full_attn'],
            arch_offsets,
            os.path.join(out_dir, f'{checkpoint_name}_comparison.jpg')
        )

    if dsqg_layers:
        layer_choice = args.layer if args.layer is not None else dsqg_layers
        if isinstance(layer_choice, int):
            layer_choice = [layer_choice]

        for layer_idx in layer_choice:
            if layer_idx in dsqg_alphas:
                plot_dsqg_combs(
                    dsqg_alphas[layer_idx],
                    arch_offsets,
                    f'Layer {layer_idx}',
                    os.path.join(out_dir, f'{checkpoint_name}_dsqg_layer{layer_idx}.jpg')
                )

    if full_weights_store.get('full_attn') is not None:
        plot_full_attention(
            full_weights_store['full_attn'],
            f'Layer {full_layer_idx} (Full Causal)',
            os.path.join(out_dir, f'{checkpoint_name}_full_attn.jpg')
        )

    if pos_bias_l0 is not None and arch_offsets:
        plot_pos_bias(
            pos_bias_l0,
            arch_offsets,
            os.path.join(out_dir, f'{checkpoint_name}_pos_bias.jpg')
        )

    if if_gains_per_layer:
        plot_if_gains(
            if_gains_per_layer,
            os.path.join(out_dir, f'{checkpoint_name}_if_gains.jpg')
        )

    summary = {
        'checkpoint': args.checkpoint,
        'n_tokens': N,
        'dsqg_layers': dsqg_layers,
        'if_gains': if_gains_per_layer,
        'pos_bias_l0_per_head_mean': pos_bias_l0.mean(0).tolist() if pos_bias_l0 is not None else [],
        'offsets': arch_offsets,
    }
    with open(os.path.join(out_dir, f'{checkpoint_name}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDone. Files in {out_dir}/')
    print('Key file: *_comparison.png — the DSQG comb vs full attention side-by-side')


if __name__ == '__main__':
    main()
