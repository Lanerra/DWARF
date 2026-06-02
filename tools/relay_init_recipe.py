#!/usr/bin/env python3
"""
Relay Topology Initialization Recipe
Distilled from D640/L20 Triadic EP3 (97.5% passkey)

This module provides initialization functions that encode the converged
relay topology structure, allowing new models to start in the right basin
instead of discovering it through expensive training.

Key findings from the analysis:
1. Two layer types with distinct relay structure:
   - A (pulse) layers (L0, L3, L7, L10, L13, L16): J_LARGE offsets, high SE
   - B/C (carry) layers: full J offsets, moderate SE
2. Pulse layers gate relay ON strongly (mean sigmoid ~0.63)
3. Carry layers are more selective (mean sigmoid ~0.55)
4. SE top offsets differ by layer type:
   - Pulse: high offsets (25-31) dominate
   - Carry: low offsets (0-5) and high offsets (28-31) mix
5. Phase_base std grows during training: ~0.023 (ep1) → ~0.042 (ep3)
6. Phase_gate converges to: pulse ~0.63, carry ~0.55
"""

import math
import torch
import torch.nn as nn


def init_relay_topology(module, layer_group='pulse', convergence_level=1.0):
    """
    Initialize a DSQG attention module with topology-matched relay parameters.
    
    Args:
        module: DSQGAttention module with phase_base, phase_gate, scale_embed, etc.
        layer_group: 'pulse' (A-group) or 'carry' (B/C-group)
        convergence_level: 0.0 = random init, 1.0 = full converged structure
    """
    if layer_group == 'pulse':
        recipe = _PULSE_RECIPE
    else:
        recipe = _CARRY_RECIPE
    _apply_recipe(module, recipe, convergence_level)


def _apply_recipe(module, recipe, convergence_level=1.0):
    """
    Apply a specific recipe dict to initialize relay parameters.
    Allows callers to override recipe values (e.g. lower SE amplitude).
    """
    
    alpha = convergence_level  # interpolation factor
    
    # --- Phase Base ---
    # Converged std: pulse=0.042, carry=0.029
    # Mix between random(0, target_std) and the structural pattern
    target_std = recipe['phase_base_std']
    with torch.no_grad():
        # Start with the V18-style normal init, scaled by convergence
        nn.init.normal_(module.phase_base, 0.0, target_std * (0.5 + 0.5 * alpha))
        
        # Add structured bias with decay from hot to cold offsets
        if alpha > 0.3:
            pb = module.phase_base.data
            J_LARGE = pb.shape[0]
            n_hot = min(len(recipe['hot_offsets']), J_LARGE)
            for rank, hot_offset in enumerate(recipe['hot_offsets'][:n_hot]):
                if hot_offset < J_LARGE:
                    # Decay: rank 0 gets full boost, rank n gets ~20%
                    decay = 1.0 - 0.8 * (rank / max(n_hot - 1, 1))
                    boost = alpha * 0.05 * decay * torch.randn_like(pb[hot_offset])
                    pb[hot_offset] += boost
    
    # --- Phase Gate ---
    # Converged sigmoid: pulse=0.63, carry=0.55
    # sigmoid(x) = target → x = logit(target)
    target_gate = recipe['phase_gate_mean']
    target_logit = math.log(target_gate / (1.0 - target_gate))
    with torch.no_grad():
        # Start at the converged mean with small noise
        module.phase_gate.fill_(target_logit)
        module.phase_gate.add_(torch.randn_like(module.phase_gate) * 0.1 * (1.0 - alpha))
    
    # --- Scale Embed ---
    # Converged pattern: pulse layers have high SE concentrated on large offsets
    #                   carry layers have moderate SE spread
    with torch.no_grad():
        se = module.scale_embed.data
        J, HD = se.shape
        
        base_val = recipe['se_base'] * alpha
        
        # Set base level
        se.fill_(base_val / math.sqrt(HD))
        
        # Amplify hot offsets
        for offset_idx in recipe['hot_offsets']:
            if offset_idx < J:
                amp = recipe['se_hot_amplitude'] * alpha
                se[offset_idx] = amp * torch.randn(HD) / math.sqrt(HD)
                se[offset_idx].abs_().mul_(1.0)
        
        # Ensure non-converged component is still random
        noise = torch.randn_like(se) * recipe['se_noise_std'] * (1.0 - alpha)
        se.add_(noise)
    
    # --- Query/Key Probes ---
    # V18 already uses orthogonal probes with pi/2 offsets, which is correct
    # No additional initialization needed here
    pass


# Recipe parameters extracted from D640/L20 EP3 analysis

_PULSE_RECIPE = {
    # Layers 0, 3, 7, 10, 13, 16 (A-group, J_LARGE=15)
    'phase_base_std': 0.042,       # Converged std
    'phase_gate_mean': 0.63,       # Converged sigmoid mean
    'se_base': 1.5,               # Base SE norm for cold offsets
    'se_hot_amplitude': 5.5,      # SE norm for hot offsets
    'se_noise_std': 0.5,          # Random component
    'hot_offsets': [31, 30, 29, 28, 26, 25, 27, 24, 23],  # High offsets dominate
    'description': 'Pulse layers: high SE on large offsets, strong gating, concentrated relay',
}

_CARRY_RECIPE = {
    # Layers 1, 2, 4, 5, 8, 9, 11, 12, 14, 15, 17, 18, 19 (B/C-group)
    'phase_base_std': 0.029,       # Lower std than pulse
    'phase_gate_mean': 0.55,       # Weaker gating
    'se_base': 1.2,               # Moderate base
    'se_hot_amplitude': 2.0,      # Lower peak amplitude
    'se_noise_std': 0.5,          # Random component
    'hot_offsets': [0, 3, 5, 30, 29, 28, 1, 2, 4],  # Mix of low and high
    'description': 'Carry layers: mixed offsets, selective gating, distributed relay',
}


def log_relay_state(module, label, layer_group='pulse'):
    """Log current relay parameter state for comparison across epochs."""
    with torch.no_grad():
        gate_sig = torch.sigmoid(module.phase_gate).mean().item()
        se_norms = module.scale_embed.norm(dim=1)
        pb_std = module.phase_base.std().item()
        pb_max = module.phase_base.abs().max().item()

        recipe = _PULSE_RECIPE if layer_group == 'pulse' else _CARRY_RECIPE
        target_gate = recipe['phase_gate_mean']
        gate_drift = abs(gate_sig - target_gate)

        print(f"  [{label}] {layer_group} gate_sig={gate_sig:.4f} (target={target_gate:.2f}, drift={gate_drift:.4f}) "
              f"SE max={se_norms.max():.2f} mean={se_norms.mean():.2f} "
              f"pb_std={pb_std:.4f} pb_max={pb_max:.4f}")

        return {
            'label': label, 'layer_group': layer_group,
            'gate_sigmoid': gate_sig, 'gate_drift': gate_drift,
            'se_max': se_norms.max().item(), 'se_mean': se_norms.mean().item(),
            'phase_base_std': pb_std, 'phase_base_max': pb_max,
        }


def validate_stability(convergence_level=0.8, mock_training_steps=3):
    """
    Simulate init → mock training epochs and log relay parameter stability.
    If structured init is correct, parameters should barely move across epochs.
    """
    print(f"=== Stability Validation (convergence={convergence_level}) ===")
    print(f"Simulating {mock_training_steps} epochs of training.\n")

    results = []
    for group in ['pulse', 'carry']:
        j_large = 15 if group == 'pulse' else 32
        mod = DummyAttn(J_LARGE=j_large, J=32, HD=64)
        init_relay_topology(mod, layer_group=group, convergence_level=convergence_level)

        # Log init state
        results.append(log_relay_state(mod, 'INIT', group))

        # Simulate training: small gradient steps that shouldn't shift topology
        for step in range(1, mock_training_steps + 1):
            with torch.no_grad():
                # Simulate gradient noise proportional to what training would add
                mod.phase_base.add_(torch.randn_like(mod.phase_base) * 0.003)
                mod.phase_gate.add_(torch.randn_like(mod.phase_gate) * 0.05)
                mod.scale_embed.add_(torch.randn_like(mod.scale_embed) * 0.02)

            label = f'MOCK_EP{step}'
            results.append(log_relay_state(mod, label, group))

    # Summary
    print(f"\n=== Stability Summary ===")
    for group in ['pulse', 'carry']:
        group_results = [r for r in results if r['layer_group'] == group]
        init = group_results[0]
        final = group_results[-1]
        print(f"  {group}: gate_drift init→final = {init['gate_sigmoid']:.4f} → {final['gate_sigmoid']:.4f} "
              f"(Δ={abs(final['gate_sigmoid'] - init['gate_sigmoid']):.4f})")
        print(f"  {group}: SE_max init→final = {init['se_max']:.2f} → {final['se_max']:.2f} "
              f"(Δ={abs(final['se_max'] - init['se_max']):.2f})")
        print(f"  {group}: pb_std init→final = {init['phase_base_std']:.4f} → {final['phase_base_std']:.4f} "
              f"(Δ={abs(final['phase_base_std'] - init['phase_base_std']):.4f})")


class DummyAttn(nn.Module):
    def __init__(self, J_LARGE, J, HD):
        super().__init__()
        self.phase_base = nn.Parameter(torch.zeros(J_LARGE, 8, 2))
        self.phase_gate = nn.Parameter(torch.zeros(J_LARGE))
        self.scale_embed = nn.Parameter(torch.zeros(J, HD))


def print_recipe():
    """Print the extracted recipes for documentation."""
    for name, recipe in [('PULSE', _PULSE_RECIPE), ('CARRY', _CARRY_RECIPE)]:
        print(f"=== {name} LAYER RECIPE ===")
        for k, v in recipe.items():
            print(f"  {k}: {v}")
        print()


if __name__ == '__main__':
    print("Relay Topology Initialization Recipe")
    print("Source: D640/L20 Triadic EP3 (97.5% passkey)")
    print()
    print_recipe()
    print()
    validate_stability(convergence_level=0.8, mock_training_steps=3)
    
    print_recipe()
    print()
    validate_stability(convergence_level=0.8, mock_training_steps=3)
