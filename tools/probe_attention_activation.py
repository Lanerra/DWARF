"""
probe_attention_activation.py — Layer-wise Attention Activation Probing

For a given checkpoint and passkey distance, probes which positions each
attention layer attends to. Specifically answers: does the FA layer attend
to the passkey position while DSQG layers do not?

Outputs per distance:
  - attention_activation_d{dist}.json  — per-layer attention ratios
  - attention_activation_d{dist}.png   — heatmap (layers x key positions)
  - attention_activation_summary.txt   — human-readable FA vs DSQG comparison

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 tools/probe_attention_activation.py \
      --model triadic_l25_d768 \
      --checkpoint autoresearch/checkpoints/d768_l25_triadic_j96_ep3_full_attn.pt \
      --distance 512 \
      --distance 1536 \
      --output tools/viz_output/attention_activation/
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, build_model, get_tokenizer, get_offsets

_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_PASSKEY_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'
_PASSKEY_WORD = 'banana'


def build_passkey_sequence(tokenizer, distance, max_seq_length=2048):
    """Build a passkey retrieval sequence with the passkey at ~distance tokens from the cue."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    intro_ids = tokenizer.encode(_PASSKEY_TEMPLATE.format(word=_PASSKEY_WORD))
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)

    available = max_seq_length - len(intro_ids) - len(cue_ids) - 1
    actual_distance = min(distance, available)

    filler = []
    while len(filler) < actual_distance:
        filler.extend(filler_ids)
    filler = filler[:actual_distance]

    full_sequence = intro_ids + filler + cue_ids

    prefix_length = len(tokenizer.encode('the secret word is '))
    passkey_position = prefix_length
    cue_position = len(full_sequence) - 1

    return full_sequence, passkey_position, cue_position


def compute_fa_attention_weights(module, hidden_input):
    """Manually compute full causal attention weights from a FullCausalAttention module."""
    B, N, D = hidden_input.shape
    H = module.num_heads
    HD = module.head_dim
    scale = math.sqrt(HD)

    qkv = module.qkv_proj(hidden_input)
    q, k, _ = qkv.split(D, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3)

    causal_mask = torch.triu(
        torch.full((N, N), float('-inf'), device=hidden_input.device), diagonal=1
    )
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale + causal_mask
    return torch.softmax(scores, dim=-1).detach().cpu()


def compute_dsqg_offset_attention(module, hidden_input, layer_offsets):
    """Compute DSQG attention weights [B, H, N, J] over offsets, then scatter to absolute positions."""
    B, N, D = hidden_input.shape
    H = module.num_heads
    HD = module.head_dim
    J = len(layer_offsets)
    scale = 1.0 / math.sqrt(HD)

    qkv = module.qkv_proj(hidden_input)
    q, k, _ = qkv.split(D, dim=-1)
    q = q.view(B, N, H, HD).permute(0, 2, 1, 3).float()
    k = k.view(B, N, H, HD).permute(0, 2, 1, 3).float()

    pos_bias = module.pos_bias.float()
    scale_embed = module.scale_embed.float()

    max_offset = max(layer_offsets)
    k_padded = F.pad(k, (0, 0, max_offset, 0))

    scores = torch.full((B, H, N, J), float('-inf'), device=hidden_input.device)

    for j_index, delta in enumerate(layer_offsets):
        k_shifted = k_padded[:, :, max_offset - delta:max_offset - delta + N, :]
        qk = (q * k_shifted).sum(-1) * scale
        pb = pos_bias[j_index].view(1, H, 1).expand(B, H, N)
        se = (q * scale_embed[j_index].view(1, 1, 1, HD)).sum(-1) * scale
        raw = qk + pb + se

        valid_mask = torch.arange(N, device=hidden_input.device).view(1, 1, N) >= delta
        scores[:, :, :, j_index] = torch.where(
            valid_mask, raw, torch.tensor(float('-inf'), device=hidden_input.device)
        )

    alpha = torch.softmax(scores, dim=-1)
    alpha = torch.nan_to_num(alpha, nan=0.0)

    position_weights = torch.zeros(B, H, N, N, device='cpu')
    alpha_cpu = alpha.detach().cpu()
    for j_index, delta in enumerate(layer_offsets):
        for query_position in range(delta, N):
            key_position = query_position - delta
            position_weights[:, :, query_position, key_position] += alpha_cpu[:, :, query_position, j_index]

    return position_weights


def probe_single_distance(model, tokenizer, device, full_attn_layer, distance, arch):
    """Run attention activation probing for a single passkey distance."""
    sequence, passkey_position, cue_position = build_passkey_sequence(tokenizer, distance)
    ids_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
    N = len(sequence)

    print(f'  Sequence length: {N}, passkey@{passkey_position}, cue@{cue_position}')

    hidden_states = {}

    def make_embed_hook():
        def hook(module, inputs, output):
            hidden_states['embed'] = output.detach()
        return hook

    def make_block_hook(layer_index):
        def hook(module, inputs, output):
            hidden_states[f'input_{layer_index}'] = inputs[0].detach()
            hidden_states[f'output_{layer_index}'] = output.detach()
        return hook

    hooks = []
    hooks.append(model.drop.register_forward_hook(make_embed_hook()))
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_block_hook(i)))

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(ids_tensor)

    for h in hooks:
        h.remove()

    number_of_layers = len(model.blocks)
    layer_results = []

    for layer_index in range(number_of_layers):
        block = model.blocks[layer_index]
        layer_input = hidden_states.get(f'input_{layer_index}')
        assert layer_input is not None, f'Missing input for layer {layer_index}'

        is_fa = layer_index == full_attn_layer
        has_pos_bias = hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias')

        if is_fa and not has_pos_bias:
            layer_type = 'FA'
            with torch.no_grad():
                attention_weights = compute_fa_attention_weights(
                    block.attn, block.norm1(layer_input.float())
                )
        elif has_pos_bias:
            layer_offsets = block.attn.offsets_dev.tolist()
            group_label = getattr(block, 'group_label', '?')
            layer_type = f'DSQG-{group_label}'
            with torch.no_grad():
                attention_weights = compute_dsqg_offset_attention(
                    block.attn, block.norm1(layer_input.float()), layer_offsets
                )
        else:
            layer_results.append({
                'layer': layer_index,
                'type': 'unknown',
                'passkey_attention': 0.0,
                'baseline_attention': 0.0,
                'ratio': 0.0,
            })
            continue

        query_weights = attention_weights[0, :, cue_position, :]
        passkey_attention = query_weights[:, passkey_position].mean().item()
        baseline_attention = query_weights.mean().item()
        ratio = passkey_attention / (baseline_attention + 1e-10)

        head_passkey_attentions = query_weights[:, passkey_position].tolist()

        layer_results.append({
            'layer': layer_index,
            'type': layer_type,
            'passkey_attention': passkey_attention,
            'baseline_attention': baseline_attention,
            'ratio': ratio,
            'per_head_passkey_attention': head_passkey_attentions,
        })

        marker = '***' if ratio > 5.0 else ''
        print(f'  L{layer_index:2d} [{layer_type:8s}]: '
              f'passkey={passkey_attention:.6f}, baseline={baseline_attention:.6f}, '
              f'ratio={ratio:8.2f}x {marker}')

    query_heatmap = np.zeros((number_of_layers, N))
    for result in layer_results:
        layer_index = result['layer']
        block = model.blocks[layer_index]
        layer_input = hidden_states.get(f'input_{layer_index}')
        if layer_input is None:
            continue

        is_fa = layer_index == full_attn_layer
        has_pos_bias = hasattr(block, 'attn') and hasattr(block.attn, 'pos_bias')

        if is_fa and not has_pos_bias:
            with torch.no_grad():
                weights = compute_fa_attention_weights(
                    block.attn, block.norm1(layer_input.float())
                )
        elif has_pos_bias:
            layer_offsets = block.attn.offsets_dev.tolist()
            with torch.no_grad():
                weights = compute_dsqg_offset_attention(
                    block.attn, block.norm1(layer_input.float()), layer_offsets
                )
        else:
            continue

        query_heatmap[layer_index, :] = weights[0, :, cue_position, :].mean(dim=0).numpy()

    return {
        'distance': distance,
        'passkey_position': passkey_position,
        'cue_position': cue_position,
        'sequence_length': N,
        'full_attn_layer': full_attn_layer,
        'layers': layer_results,
        'query_heatmap': query_heatmap,
    }


def plot_attention_activation(probe_data, output_path):
    """Plot heatmap of attention from the cue query to all key positions, per layer."""
    heatmap = probe_data['query_heatmap']
    number_of_layers, N = heatmap.shape
    passkey_position = probe_data['passkey_position']
    cue_position = probe_data['cue_position']
    full_attn_layer = probe_data['full_attn_layer']
    distance = probe_data['distance']

    fig, (ax_heat, ax_ratio) = plt.subplots(
        2, 1, figsize=(18, 10), height_ratios=[3, 1],
        gridspec_kw={'hspace': 0.3}
    )

    im = ax_heat.imshow(
        heatmap, aspect='auto', origin='lower', cmap='hot',
        vmin=0, vmax=np.percentile(heatmap[heatmap > 0], 99) + 1e-8,
    )
    ax_heat.axvline(passkey_position, color='lime', linestyle='--', linewidth=2, label='Passkey')
    ax_heat.axvline(cue_position, color='cyan', linestyle='--', linewidth=2, label='Cue')
    ax_heat.axhline(full_attn_layer, color='white', linestyle='-', linewidth=2, alpha=0.8,
                    label=f'FA @ L{full_attn_layer}')

    layer_labels = []
    for result in probe_data['layers']:
        layer_labels.append(f'L{result["layer"]} {result["type"]}')
    ax_heat.set_yticks(range(number_of_layers))
    ax_heat.set_yticklabels(layer_labels, fontsize=7)
    ax_heat.set_xlabel('Key Position', fontsize=11)
    ax_heat.set_ylabel('Layer', fontsize=11)
    ax_heat.set_title(
        f'Attention Activation: Cue Query -> All Key Positions (d={distance})\n'
        f'Passkey @ pos {passkey_position}, Cue @ pos {cue_position}',
        fontsize=13,
    )
    ax_heat.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax_heat, label='Attention Weight (mean over heads)', shrink=0.8)

    layers_plotted = []
    ratios = []
    colors = []
    for result in probe_data['layers']:
        layers_plotted.append(result['layer'])
        ratios.append(result['ratio'])
        if result['type'] == 'FA':
            colors.append('red')
        elif 'DSQG' in result['type']:
            colors.append('steelblue')
        else:
            colors.append('gray')

    ax_ratio.bar(layers_plotted, ratios, color=colors, alpha=0.8)
    ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1x baseline')
    ax_ratio.set_xlabel('Layer', fontsize=11)
    ax_ratio.set_ylabel('Passkey Attention Ratio\n(vs baseline)', fontsize=10)
    ax_ratio.set_title('Per-Layer Passkey Attention Ratio (red=FA, blue=DSQG)', fontsize=11)
    ax_ratio.set_yscale('symlog', linthresh=1.0)
    ax_ratio.legend(fontsize=9)
    ax_ratio.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def write_summary(all_results, output_directory):
    """Write a human-readable summary comparing FA vs DSQG passkey attention."""
    summary_path = os.path.join(output_directory, 'attention_activation_summary.txt')
    lines = ['DWARF Attention Activation Probe — Summary', '=' * 60, '']

    for probe_data in all_results:
        distance = probe_data['distance']
        lines.append(f'Distance = {distance} tokens')
        lines.append(f'  Passkey @ position {probe_data["passkey_position"]}, '
                     f'Cue @ position {probe_data["cue_position"]}')
        lines.append('')

        fa_results = [r for r in probe_data['layers'] if r['type'] == 'FA']
        dsqg_results = [r for r in probe_data['layers'] if 'DSQG' in r['type']]

        if fa_results:
            fa = fa_results[0]
            lines.append(f'  FA layer (L{fa["layer"]}):')
            lines.append(f'    Passkey attention:  {fa["passkey_attention"]:.6f}')
            lines.append(f'    Baseline attention: {fa["baseline_attention"]:.6f}')
            lines.append(f'    Ratio:              {fa["ratio"]:.2f}x baseline')
            lines.append('')

        if dsqg_results:
            ratios = [r['ratio'] for r in dsqg_results]
            attentions = [r['passkey_attention'] for r in dsqg_results]
            best_dsqg = max(dsqg_results, key=lambda r: r['ratio'])
            lines.append(f'  DSQG layers ({len(dsqg_results)} layers):')
            lines.append(f'    Mean passkey ratio:    {np.mean(ratios):.2f}x')
            lines.append(f'    Max passkey ratio:     {max(ratios):.2f}x (L{best_dsqg["layer"]})')
            lines.append(f'    Mean passkey attention: {np.mean(attentions):.6f}')
            lines.append('')

        if fa_results and dsqg_results:
            fa_ratio = fa_results[0]['ratio']
            dsqg_mean_ratio = np.mean(ratios)
            advantage = fa_ratio / (dsqg_mean_ratio + 1e-10)
            lines.append(f'  FA shows {fa_ratio:.1f}x baseline attention to passkey; '
                         f'DSQG layers average {dsqg_mean_ratio:.1f}x')
            lines.append(f'  FA advantage over DSQG: {advantage:.1f}x')
            lines.append('')

        lines.append('-' * 60)
        lines.append('')

    summary_text = '\n'.join(lines)
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f'  Saved: {summary_path}')
    print()
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description='Layer-wise attention activation probing for DWARF passkey retrieval')
    parser.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture name from registry')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--distance', type=int, action='append', required=True,
                        help='Passkey distance(s) to probe (can specify multiple)')
    parser.add_argument('--output', default='tools/viz_output/attention_activation/',
                        help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)
    output_directory = os.path.join(root, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_directory, exist_ok=True)

    print('=' * 60)
    print(f'  Attention Activation Probe')
    print(f'  Model: {args.model}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Distances: {args.distance}')
    print('=' * 60)

    print('\nLoading model...')
    model, cfg, train_module = build_model(args.model, device=device, checkpoint_path=args.checkpoint)
    full_attn_layer = cfg.get('_resolved_full_attn_layer')
    if full_attn_layer is None:
        full_attn_layer = getattr(model, 'full_attn_layer', len(model.blocks) - 1)

    tok = get_tokenizer(args.model)
    tokenizer = train_module.BPETokenizerWrapper(tok)

    number_of_layers = len(model.blocks)
    print(f'  {number_of_layers} layers, FA @ L{full_attn_layer}')

    all_results = []

    for distance in args.distance:
        print(f'\nProbing distance={distance}...')
        probe_data = probe_single_distance(
            model, tokenizer, device, full_attn_layer, distance, args.model
        )

        json_path = os.path.join(output_directory, f'attention_activation_d{distance}.json')
        serializable = {
            'distance': probe_data['distance'],
            'passkey_position': probe_data['passkey_position'],
            'cue_position': probe_data['cue_position'],
            'sequence_length': probe_data['sequence_length'],
            'full_attn_layer': probe_data['full_attn_layer'],
            'layers': probe_data['layers'],
        }
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f'  Saved: {json_path}')

        plot_attention_activation(
            probe_data,
            os.path.join(output_directory, f'attention_activation_d{distance}.png'),
        )

        all_results.append(probe_data)

    write_summary(all_results, output_directory)

    print(f'\nDone. Results in {output_directory}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
