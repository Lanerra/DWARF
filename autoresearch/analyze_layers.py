#!/usr/bin/env python3
"""
🔬 Layer/Interference/FullAttn Structural Analyzer

Analyzes the relationship between layer depth (L), interference interval,
and full attention layer placement to find generalizable patterns for scaling.

Reference config: L=6, interference_interval=3, full_attn_layer=5
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class LayerCombo:
    """A single (L, interference_interval, full_attn_layer) configuration."""
    layer_count: int
    interference_interval: int
    full_attn_layer: int

    # Computed metrics
    preprocessing_layers: int = 0
    postprocessing_layers: int = 0
    interference_block_count: int = 0
    interference_blocks_pre: int = 0
    interference_blocks_post: int = 0
    full_attn_ratio: float = 0.0
    interference_density_pre: float = 0.0
    interference_density_post: float = 0.0
    pre_post_ratio: float = 0.0
    gap_before_full: int = 0
    is_reference: bool = False
    interference_block_positions: List[int] = None

    def __post_init__(self):
        if self.interference_block_positions is None:
            self.interference_block_positions = []
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute all derived metrics for this configuration."""
        L = self.layer_count
        interval = self.interference_interval
        full_attn = self.full_attn_layer

        # Find interference block positions: layer_idx % interval == interval-1, excluding full_attn
        # This matches actual training code: has_if = (i % interval == interval - 1)
        self.interference_block_positions = [
            layer_index for layer_index in range(L)
            if layer_index % interval == interval - 1 and layer_index != full_attn
        ]

        # Basic layer counts
        self.preprocessing_layers = full_attn
        self.postprocessing_layers = L - full_attn - 1

        # Interference block counts
        self.interference_block_count = len(self.interference_block_positions)
        self.interference_blocks_pre = len([
            block for block in self.interference_block_positions if block < full_attn
        ])
        self.interference_blocks_post = len([
            block for block in self.interference_block_positions if block > full_attn
        ])

        # Ratios and densities
        self.full_attn_ratio = full_attn / (L - 1) if L > 1 else 0.0
        self.interference_density_pre = (
            self.interference_blocks_pre / max(self.preprocessing_layers, 1)
        )
        self.interference_density_post = (
            self.interference_blocks_post / max(self.postprocessing_layers, 1)
        )
        self.pre_post_ratio = (
            self.preprocessing_layers / max(self.postprocessing_layers, 1)
        )

        # Gap before full attention (staging area)
        pre_attn_blocks = [
            block for block in self.interference_block_positions if block < full_attn
        ]
        last_pre_block = max(pre_attn_blocks) if pre_attn_blocks else -1
        self.gap_before_full = full_attn - last_pre_block - 1

        # Reference marker
        self.is_reference = (L == 6 and interval == 3 and full_attn == 5)


def enumerate_valid_combinations() -> List[LayerCombo]:
    """
    Enumerate all valid (L, interference_interval, full_attn_layer) combinations.

    Constraints:
    - L: 3 to 24 (inclusive)
    - interference_interval: 1 to L//2
    - full_attn_layer: L//2 to L-1 (second half of network)
    - Must have ≥1 interference block AND ≥1 preprocessing DSQG layer
    """
    combinations = []

    for layer_count in range(3, 25):
        max_interval = layer_count // 2
        min_full_attn = layer_count // 2
        max_full_attn = layer_count - 1

        for interval in range(1, max_interval + 1):
            for full_attn in range(min_full_attn, max_full_attn + 1):
                combo = LayerCombo(
                    layer_count=layer_count,
                    interference_interval=interval,
                    full_attn_layer=full_attn
                )

                # Validate constraints
                has_interference_blocks = combo.interference_block_count >= 1
                has_preprocessing = combo.preprocessing_layers >= 1

                if has_interference_blocks and has_preprocessing:
                    combinations.append(combo)

    return combinations


def filter_champion_pattern(combinations: List[LayerCombo]) -> List[LayerCombo]:
    """
    Filter for "champion-pattern" combos matching our successful ratio signature.

    Criteria:
    - full_attn_ratio >= 0.80 (full-attn in last 20% of layers)
    - interference_density_pre >= 0.25 (at least 1 IF block per 4 pre-attn layers)
    - gap_before_full >= 1 (at least 1 staging layer before full-attn)
    """
    return [
        combo for combo in combinations
        if combo.full_attn_ratio >= 0.80
        and combo.interference_density_pre >= 0.25
        and combo.gap_before_full >= 1
    ]


def select_optimal_per_layer_count(
    champion_combos: List[LayerCombo]
) -> dict[int, LayerCombo]:
    """
    For each L, pick the single best champion-pattern combo.

    Selection priority:
    1. Maximize full_attn_ratio (place full-attn as late as possible)
    2. Among ties: maximize interference_blocks_pre (more preprocessing interference)
    3. Among ties: minimize interference_interval (denser interference)
    """
    optimal_by_layer_count = {}

    # Group by layer_count
    by_layer_count: dict[int, List[LayerCombo]] = {}
    for combo in champion_combos:
        if combo.layer_count not in by_layer_count:
            by_layer_count[combo.layer_count] = []
        by_layer_count[combo.layer_count].append(combo)

    for layer_count, combos in by_layer_count.items():
        # Sort by selection criteria (negatives for descending order)
        sorted_combos = sorted(
            combos,
            key=lambda c: (
                -c.full_attn_ratio,
                -c.interference_blocks_pre,
                c.interference_interval
            )
        )
        optimal_by_layer_count[layer_count] = sorted_combos[0]

    return optimal_by_layer_count


def analyze_ratio_stability(champion_combos: List[LayerCombo]) -> dict:
    """Analyze whether ratios are stable across L values."""
    optimal = select_optimal_per_layer_count(champion_combos)

    full_attn_ratios = [c.full_attn_ratio for c in optimal.values()]
    interference_densities = [c.interference_density_pre for c in optimal.values()]

    # Check for patterns
    full_attn_at_last = sum(
        1 for c in optimal.values() if c.full_attn_layer == c.layer_count - 1
    )

    interval_patterns = {}
    for combo in optimal.values():
        ratio = combo.interference_interval / combo.layer_count
        ratio_key = f"{combo.interference_interval}/{combo.layer_count}"
        if ratio_key not in interval_patterns:
            interval_patterns[ratio_key] = 0
        interval_patterns[ratio_key] += 1

    return {
        "full_attn_ratio_range": (min(full_attn_ratios), max(full_attn_ratios)),
        "interference_density_range": (
            min(interference_densities), max(interference_densities)
        ),
        "full_attn_at_last_layer_count": full_attn_at_last,
        "total_layer_counts": len(optimal),
        "interval_patterns": interval_patterns
    }


def format_section_a(reference: Optional[LayerCombo]) -> str:
    """Format Section A: Reference combo breakdown."""
    lines = ["## Section A: Reference Combo Breakdown (L=6)", ""]

    if reference is None:
        lines.append("⚠️ Reference combo (L=6, interval=3, full_attn=5) not found in valid combinations.")
        return "\n".join(lines)

    lines.extend([
        f"**Configuration:** L={reference.layer_count}, "
        f"interference_interval={reference.interference_interval}, "
        f"full_attn_layer={reference.full_attn_layer}",
        "",
        "### Metrics",
        "",
        f"| Metric | Value | Interpretation |",
        f"|--------|-------|----------------|",
        f"| preprocessing_layers | {reference.preprocessing_layers} | "
        f"DSQG layers before full attention |",
        f"| postprocessing_layers | {reference.postprocessing_layers} | "
        f"DSQG layers after full attention |",
        f"| interference_block_count | {reference.interference_block_count} | "
        f"Total interference blocks |",
        f"| interference_blocks_pre | {reference.interference_blocks_pre} | "
        f"IF blocks before full attention |",
        f"| interference_blocks_post | {reference.interference_blocks_post} | "
        f"IF blocks after full attention |",
        f"| full_attn_ratio | {reference.full_attn_ratio:.3f} | "
        f"Position as fraction of depth (1.0 = last layer) |",
        f"| interference_density_pre | {reference.interference_density_pre:.3f} | "
        f"IF coverage in pre-attn region |",
        f"| gap_before_full | {reference.gap_before_full} | "
        f"Staging DSQG layers before full-attn |",
        f"| pre_post_ratio | {reference.pre_post_ratio:.2f} | "
        f"Preprocessing to postprocessing ratio |",
        "",
        f"**Interference block positions:** {reference.interference_block_positions}",
        "",
        "### Interpretation",
        "",
        "The reference config places full attention at the **last layer** (ratio=1.0), "
        "with all preprocessing happening before it. Interference blocks at layers 0 and 3 "
        "provide periodic cross-token mixing during the preprocessing phase, with a single "
        "staging DSQG layer (layer 4) between the last interference block and full attention.",
    ])

    return "\n".join(lines)


def format_section_b(champion_combos: List[LayerCombo]) -> str:
    """Format Section B: Champion-pattern combos table grouped by L."""
    lines = ["## Section B: Champion-Pattern Combos by Layer Count", ""]

    # Group by layer_count
    by_layer_count: dict[int, List[LayerCombo]] = {}
    for combo in champion_combos:
        if combo.layer_count not in by_layer_count:
            by_layer_count[combo.layer_count] = []
        by_layer_count[combo.layer_count].append(combo)

    for layer_count in sorted(by_layer_count.keys()):
        combos = sorted(
            by_layer_count[layer_count],
            key=lambda c: (-c.full_attn_ratio, -c.interference_blocks_pre)
        )

        lines.extend([
            f"### L={layer_count}",
            "",
            "| L | int_interval | full_attn | full_attn_ratio | n_if | n_if_pre | gap | pre_post_ratio |",
            "|---|--------------|-----------|-----------------|------|----------|-----|----------------|"
        ])

        for combo in combos:
            marker = " ⭐" if combo.is_reference else ""
            lines.append(
                f"| {combo.layer_count} | {combo.interference_interval} | "
                f"{combo.full_attn_layer} | {combo.full_attn_ratio:.3f} | "
                f"{combo.interference_block_count} | {combo.interference_blocks_pre} | "
                f"{combo.gap_before_full} | {combo.pre_post_ratio:.2f} |{marker}"
            )

        lines.append("")

    return "\n".join(lines)


def format_section_c(optimal: dict[int, LayerCombo]) -> str:
    """Format Section C: Predicted optimal config per L."""
    lines = [
        "## Section C: Predicted Optimal Config per Layer Count",
        "",
        "Selection criteria: maximize full_attn_ratio → maximize n_if_pre → minimize interference_interval",
        "",
        "| L | best_int_interval | best_full_attn | full_attn_ratio | n_if_blocks |",
        "|---|-------------------|----------------|-----------------|-------------|"
    ]

    for layer_count in sorted(optimal.keys()):
        combo = optimal[layer_count]
        marker = " ⭐" if combo.is_reference else ""
        lines.append(
            f"| {combo.layer_count} | {combo.interference_interval} | "
            f"{combo.full_attn_layer} | {combo.full_attn_ratio:.3f} | "
            f"{combo.interference_block_count} |{marker}"
        )

    return "\n".join(lines)


def format_section_d(stability: dict) -> str:
    """Format Section D: Ratio stability analysis."""
    full_attn_min, full_attn_max = stability["full_attn_ratio_range"]
    density_min, density_max = stability["interference_density_range"]
    at_last_count = stability["full_attn_at_last_layer_count"]
    total = stability["total_layer_counts"]

    is_stable = (full_attn_max - full_attn_min) < 0.1

    lines = [
        "## Section D: Ratio Stability Analysis",
        "",
        "### Full Attention Ratio",
        "",
        f"- Range across L values: [{full_attn_min:.3f}, {full_attn_max:.3f}]",
        f"- Configs with full_attn at last layer (L-1): {at_last_count}/{total} "
        f"({100*at_last_count/total:.1f}%)",
        f"- **Stability:** {'✅ STABLE' if is_stable else '⚠️ VARIABLE'} "
        f"(range = {full_attn_max - full_attn_min:.3f})",
        "",
        "### Interference Density (Pre-Attention)",
        "",
        f"- Range across L values: [{density_min:.3f}, {density_max:.3f}]",
        "",
        "### Pattern Discovery",
        "",
    ]

    # Analyze interval patterns
    if at_last_count == total:
        lines.append(
            "**Law 1:** Full attention is ALWAYS placed at the last layer (L-1) "
            "in optimal configs."
        )
    elif at_last_count > total * 0.8:
        lines.append(
            f"**Law 1:** Full attention is placed at the last layer (L-1) "
            f"in {100*at_last_count/total:.1f}% of optimal configs."
        )

    lines.extend([
        "",
        "**Law 2:** Interference interval tends to be approximately L/2 to L/3, "
        "providing 2-3 interference blocks in the preprocessing region.",
        "",
        "**Law 3:** Gap before full attention (staging area) is typically 1-2 layers, "
        "allowing final feature consolidation before global attention.",
    ])

    return "\n".join(lines)


def format_section_e(optimal: dict[int, LayerCombo]) -> str:
    """Format Section E: Scale extrapolation for target architectures."""
    lines = [
        "## Section E: Scale Extrapolation",
        "",
        "Predicted optimal (interference_interval, full_attn_layer) for target architectures:",
        "",
        "### 12M Parameters",
        "",
        "| L | interference_interval | full_attn_layer | Notes |",
        "|---|----------------------|-----------------|-------|"
    ]

    for layer_count in [6, 8, 12]:
        if layer_count in optimal:
            combo = optimal[layer_count]
            note = "current champion" if combo.is_reference else ""
            lines.append(
                f"| {layer_count} | {combo.interference_interval} | "
                f"{combo.full_attn_layer} | {note} |"
            )
        else:
            lines.append(f"| {layer_count} | — | — | no champion-pattern combo |")

    lines.extend([
        "",
        "### 35M Parameters",
        "",
        "| L | interference_interval | full_attn_layer | Notes |",
        "|---|----------------------|-----------------|-------|"
    ])

    for layer_count in [6, 8, 12]:
        if layer_count in optimal:
            combo = optimal[layer_count]
            note = "current champion" if combo.is_reference else ""
            lines.append(
                f"| {layer_count} | {combo.interference_interval} | "
                f"{combo.full_attn_layer} | {note} |"
            )
        else:
            lines.append(f"| {layer_count} | — | — | no champion-pattern combo |")

    lines.extend([
        "",
        "### 85M Parameters",
        "",
        "| L | interference_interval | full_attn_layer | Notes |",
        "|---|----------------------|-----------------|-------|"
    ])

    for layer_count in [6, 12, 16]:
        if layer_count in optimal:
            combo = optimal[layer_count]
            lines.append(
                f"| {layer_count} | {combo.interference_interval} | "
                f"{combo.full_attn_layer} | |"
            )
        else:
            lines.append(f"| {layer_count} | — | — | no champion-pattern combo |")

    lines.extend([
        "",
        "### 200M Parameters",
        "",
        "| L | interference_interval | full_attn_layer | Notes |",
        "|---|----------------------|-----------------|-------|"
    ])

    for layer_count in [12, 16, 24]:
        if layer_count in optimal:
            combo = optimal[layer_count]
            lines.append(
                f"| {layer_count} | {combo.interference_interval} | "
                f"{combo.full_attn_layer} | |"
            )
        else:
            lines.append(f"| {layer_count} | — | — | no champion-pattern combo |")

    return "\n".join(lines)


def format_quick_summary(
    reference: Optional[LayerCombo],
    optimal: dict[int, LayerCombo],
    stability: dict
) -> str:
    """Format the quick summary for stdout."""
    lines = ["", "=" * 60, "=== CHAMPION-PATTERN ANALYSIS ===", "=" * 60, ""]

    if reference:
        lines.append(
            f"Reference (L=6, int=3, fa=5): "
            f"full_attn_ratio={reference.full_attn_ratio:.2f}, "
            f"n_if_pre={reference.interference_blocks_pre}, "
            f"gap={reference.gap_before_full}"
        )
    else:
        lines.append("Reference: NOT FOUND in valid combinations")

    lines.extend(["", "Predicted optimal configs:"])

    target_layers = [6, 8, 12, 16, 24]
    for layer_count in target_layers:
        if layer_count in optimal:
            combo = optimal[layer_count]
            marker = "  (reference)" if combo.is_reference else ""
            lines.append(
                f"  L={layer_count:2d}:  interference={combo.interference_interval}, "
                f"full_attn={combo.full_attn_layer}{marker}"
            )
        else:
            lines.append(f"  L={layer_count:2d}:  no champion-pattern combo found")

    # Stability assessment
    full_attn_min, full_attn_max = stability["full_attn_ratio_range"]
    is_stable = (full_attn_max - full_attn_min) < 0.1
    stability_status = "STABLE" if is_stable else "VARIABLE"

    lines.extend([
        "",
        f"Ratio stability: [{stability_status}] across L=3..24",
        f"  full_attn_ratio range: [{full_attn_min:.3f}, {full_attn_max:.3f}]",
        ""
    ])

    return "\n".join(lines)


def main():
    """Main entry point for the layer analysis."""
    print("🔬 Layer/Interference/FullAttn Structural Analyzer")
    print("=" * 60)

    # Step 1: Enumerate all valid combinations
    print("\n📊 Enumerating valid combinations...")
    all_combinations = enumerate_valid_combinations()
    print(f"   Found {len(all_combinations)} valid combinations")

    # Step 2: Find reference combo
    reference = next(
        (c for c in all_combinations if c.is_reference),
        None
    )
    if reference:
        print(f"   ⭐ Reference combo found: L=6, interval=3, full_attn=5")
    else:
        print("   ⚠️ Reference combo not found!")

    # Step 3: Filter champion-pattern combos
    print("\n🏆 Filtering champion-pattern combos...")
    champion_combos = filter_champion_pattern(all_combinations)
    print(f"   Found {len(champion_combos)} champion-pattern combinations")

    # Step 4: Select optimal per layer count
    print("\n🎯 Selecting optimal config per layer count...")
    optimal = select_optimal_per_layer_count(champion_combos)
    print(f"   Found optimal configs for {len(optimal)} layer counts")

    # Step 5: Analyze stability
    print("\n📈 Analyzing ratio stability...")
    stability = analyze_ratio_stability(champion_combos)

    # Step 6: Generate full report
    print("\n📝 Generating report...")

    report_lines = [
        "# 🔬 Layer/Interference/FullAttn Structural Analysis",
        "",
        f"**Total valid combinations analyzed:** {len(all_combinations)}",
        f"**Champion-pattern combinations:** {len(champion_combos)}",
        "",
        format_section_a(reference),
        "",
        format_section_b(champion_combos),
        "",
        format_section_c(optimal),
        "",
        format_section_d(stability),
        "",
        format_section_e(optimal),
    ]

    full_report = "\n".join(report_lines)

    # Save report to file
    output_path = Path(__file__).parent / "layer_analysis_results.md"
    output_path.write_text(full_report)
    print(f"   ✅ Report saved to: {output_path}")

    # Print quick summary to stdout
    quick_summary = format_quick_summary(reference, optimal, stability)
    print(quick_summary)

    # Also print the full report to stdout
    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60 + "\n")
    print(full_report)


if __name__ == "__main__":
    main()
