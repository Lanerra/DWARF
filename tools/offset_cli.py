#!/usr/bin/env python3
"""⚡ Interactive CLI for exploring DSQG sparse offset configurations.

Usage:
    python tools/offset_cli.py rank --dense 48 --n-sparse 3 --top 10
    python tools/offset_cli.py score --dense 48 --sparse 96,128,384
    python tools/offset_cli.py compare --dense 48 --sets "96,128,384;64,128,384"
    python tools/offset_cli.py sweep --dense-range 41,48,56,64 --n-sparse 3 --top 5
"""

import argparse
import math
from itertools import combinations

from rich.console import Console
from rich.table import Table

SPARSE_POOL = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
DEFAULT_TARGETS = [64, 128, 256, 512, 1024, 1536]
MAX_LAG = 2048
MAX_HOPS = 3
LEARNABLE_THRESHOLD = 1e-8

console = Console()


# --- Scoring ---


def build_all_offsets(dense_width: int, sparse_offsets: list[int]) -> list[int]:
    dense = list(range(dense_width + 1))
    return sorted(set(dense + sparse_offsets))


def path_counts_table(
    all_offsets: list[int], max_hops: int = MAX_HOPS
) -> list[list[int]]:
    """Compute counts[k][d] = number of ordered k-hop paths reaching distance d."""
    counts = [[0] * (MAX_LAG + 1) for _ in range(max_hops + 1)]
    for delta in all_offsets:
        if 0 < delta <= MAX_LAG:
            counts[1][delta] += 1
    for k in range(2, max_hops + 1):
        for lag in range(1, MAX_LAG + 1):
            total = 0
            for delta in all_offsets:
                if 0 < delta < lag:
                    total += counts[k - 1][lag - delta]
            counts[k][lag] = total
    return counts


def path_count(
    target_distance: int, all_offsets: list[int], max_hops: int = MAX_HOPS
) -> int:
    """Count ways to reach target_distance using up to max_hops offsets."""
    counts = path_counts_table(all_offsets, max_hops)
    return sum(counts[k][target_distance] for k in range(1, max_hops + 1))


def coverage_score(
    all_offsets: list[int],
    target_distances: list[int] | None = None,
    max_hops: int = MAX_HOPS,
) -> tuple[float, list[list[int]]]:
    """Weighted sum of path_count(d) for target distances, weight = 1/d.

    Returns (score, hop_detail) where hop_detail[i] = [1-hop, 2-hop, 3-hop, total].
    """
    if target_distances is None:
        target_distances = DEFAULT_TARGETS
    counts = path_counts_table(all_offsets, max_hops)
    score = 0.0
    hop_detail = []
    for distance in target_distances:
        per_hop = [counts[k][distance] for k in range(1, max_hops + 1)]
        total = sum(per_hop)
        hop_detail.append(per_hop + [total])
        score += total / distance
    return score, hop_detail


def softmax_weights(
    sparse_offsets: list[int],
    dense_width: int,
    alphas: list[float] | None = None,
) -> tuple[float, list[list[float]], list[float]]:
    """Min softmax weight across all sparse offsets and all heads.

    Returns (min_weight, weights_table, alphas).
    weights_table[alpha_index][sparse_index] = softmax weight.
    """
    if alphas is None:
        alphas = [0.2 + i * 1.8 / 7 for i in range(8)]
    all_offsets = list(range(dense_width + 1)) + list(sparse_offsets)
    sparse_start = dense_width + 1

    minimum = float("inf")
    weights_table = []
    for alpha in alphas:
        log_values = [-math.log(1 + d) * alpha for d in all_offsets]
        max_log = max(log_values)
        exp_values = [math.exp(lv - max_log) for lv in log_values]
        denominator = sum(exp_values)
        row = []
        for j in range(len(sparse_offsets)):
            weight = exp_values[sparse_start + j] / denominator
            row.append(weight)
            minimum = min(minimum, weight)
        weights_table.append(row)

    return minimum, weights_table, alphas


def compute_scores(sparse_offsets: list[int], dense_width: int) -> dict:
    """Compute all scoring metrics for a sparse offset set."""
    all_offsets = build_all_offsets(dense_width, sparse_offsets)
    coverage, hop_detail = coverage_score(all_offsets)
    min_weight, weights_table, alphas = softmax_weights(sparse_offsets, dense_width)
    learnable = min_weight > LEARNABLE_THRESHOLD
    learnability_bonus = 1.0 if learnable else 0.1
    composite = coverage * learnability_bonus
    return {
        "all_offsets": all_offsets,
        "sparse_offsets": sparse_offsets,
        "dense_width": dense_width,
        "coverage": coverage,
        "hop_detail": hop_detail,
        "min_weight": min_weight,
        "weights_table": weights_table,
        "alphas": alphas,
        "learnable": learnable,
        "learnability_bonus": learnability_bonus,
        "composite": composite,
    }


def rank_combinations(
    dense_width: int, n_sparse: int, pool: list[int]
) -> list[tuple[float, float, float, bool, list[int]]]:
    """Rank all C(pool, n_sparse) combinations by composite score.

    Returns sorted list of (composite, coverage, min_weight, learnable, sparse_offsets).
    """
    results = []
    for combination in combinations(pool, n_sparse):
        sparse = list(combination)
        all_offsets = build_all_offsets(dense_width, sparse)
        coverage, _ = coverage_score(all_offsets)
        min_weight, _, _ = softmax_weights(sparse, dense_width)
        learnable = min_weight > LEARNABLE_THRESHOLD
        bonus = 1.0 if learnable else 0.1
        composite = coverage * bonus
        results.append((composite, coverage, min_weight, learnable, sparse))
    results.sort(key=lambda r: r[0], reverse=True)
    return results


# --- Display ---


def display_score(sparse_offsets: list[int], dense_width: int) -> None:
    scores = compute_scores(sparse_offsets, dense_width)

    console.print()
    console.print(
        f"[bold]⚡ Offset set: dense_width={dense_width}, "
        f"sparse={sparse_offsets}[/bold]"
    )
    console.print(
        f"All offsets ({len(scores['all_offsets'])}): {scores['all_offsets']}"
    )
    console.print()

    table = Table(title="🔗 Path counts to target distances")
    table.add_column("Distance", justify="right", style="cyan")
    for k in range(1, MAX_HOPS + 1):
        table.add_column(f"{k}-hop", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Weighted (1/d)", justify="right", style="green")

    for i, distance in enumerate(DEFAULT_TARGETS):
        detail = scores["hop_detail"][i]
        total = detail[-1]
        weighted = total / distance
        row = [str(distance)]
        for k in range(MAX_HOPS):
            row.append(f"{detail[k]:,}")
        row.append(f"{total:,}")
        row.append(f"{weighted:,.1f}")
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold green]📊 Coverage score: {scores['coverage']:,.1f}[/bold green]"
    )
    console.print()

    sw_table = Table(title="🎯 Softmax weights per head (\u03b1) \u00d7 sparse offset")
    sw_table.add_column("\u03b1 (head)", justify="right", style="cyan")
    for delta in sparse_offsets:
        sw_table.add_column(f"\u03b4={delta}", justify="right")

    for i, alpha in enumerate(scores["alphas"]):
        row = [f"{alpha:.3f}"]
        for j in range(len(sparse_offsets)):
            weight = scores["weights_table"][i][j]
            style = "green" if weight > LEARNABLE_THRESHOLD else "red"
            row.append(f"[{style}]{weight:.2e}[/{style}]")
        sw_table.add_row(*row)

    console.print(sw_table)

    status = (
        "[bold green]LEARNABLE[/bold green]"
        if scores["learnable"]
        else "[bold red]NOT LEARNABLE[/bold red]"
    )
    console.print(f"\nMin softmax weight: {scores['min_weight']:.2e}  \u2190 {status}")
    console.print(
        f"[bold]🏆 Composite rank score: {scores['composite']:,.1f}[/bold]"
    )
    console.print()


def _make_rank_table(
    results: list[tuple[float, float, float, bool, list[int]]],
    title: str,
    top: int,
) -> Table:
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Sparse offsets", style="cyan")
    table.add_column("Coverage", justify="right")
    table.add_column("Min softmax wt", justify="right")
    table.add_column("Learnable", justify="center")
    table.add_column("Composite", justify="right", style="bold green")

    for rank, (composite, coverage, min_weight, learnable, sparse) in enumerate(
        results[:top], 1
    ):
        learn_mark = "[green]\u2713[/green]" if learnable else "[red]\u2717[/red]"
        table.add_row(
            str(rank),
            str(sparse),
            f"{coverage:,.1f}",
            f"{min_weight:.2e}",
            learn_mark,
            f"{composite:,.1f}",
        )
    return table


def display_rank(
    dense_width: int, n_sparse: int, pool: list[int], top: int
) -> None:
    results = rank_combinations(dense_width, n_sparse, pool)
    total_combos = len(results)

    console.print()
    console.print(
        f"[bold]⚡ Top {top} of {total_combos} sparse sets "
        f"(dense_width={dense_width}, n_sparse={n_sparse})[/bold]"
    )
    console.print(f"Pool: {pool}")
    console.print()

    table = _make_rank_table(
        results, f"🏆 Rank table (dense_width={dense_width})", top
    )
    console.print(table)
    console.print()


def display_compare(dense_width: int, sets: list[list[int]]) -> None:
    console.print()
    console.print(
        f"[bold]⚡ Comparing {len(sets)} sparse sets "
        f"(dense_width={dense_width})[/bold]"
    )
    console.print()

    all_scores = [compute_scores(s, dense_width) for s in sets]

    table = Table(
        title=f"📊 Side-by-side comparison (dense_width={dense_width})"
    )
    table.add_column("Metric", style="cyan")
    for s in sets:
        table.add_column(str(s), justify="right")

    table.add_row(
        "Total offsets",
        *[str(len(sc["all_offsets"])) for sc in all_scores],
    )
    for i, distance in enumerate(DEFAULT_TARGETS):
        table.add_row(
            f"Paths \u2192 d={distance}",
            *[f"{sc['hop_detail'][i][-1]:,}" for sc in all_scores],
        )
    table.add_row(
        "Coverage score",
        *[f"{sc['coverage']:,.1f}" for sc in all_scores],
    )
    table.add_row(
        "Min softmax weight",
        *[f"{sc['min_weight']:.2e}" for sc in all_scores],
    )
    table.add_row(
        "Learnable",
        *[
            "[green]\u2713[/green]" if sc["learnable"] else "[red]\u2717[/red]"
            for sc in all_scores
        ],
    )
    table.add_row(
        "Composite rank",
        *[f"[bold]{sc['composite']:,.1f}[/bold]" for sc in all_scores],
    )

    console.print(table)
    console.print()


def display_sweep(
    dense_range: list[int], n_sparse: int, pool: list[int], top: int
) -> None:
    console.print()
    console.print(
        f"[bold]⚡ Sweep across dense widths: {dense_range} "
        f"(n_sparse={n_sparse}, top={top})[/bold]"
    )
    console.print()

    winners = []
    for dense_width in dense_range:
        results = rank_combinations(dense_width, n_sparse, pool)
        table = _make_rank_table(results, f"dense_width={dense_width}", top)
        console.print(table)
        console.print()
        if results:
            composite, coverage, min_weight, learnable, sparse = results[0]
            winners.append(
                (dense_width, sparse, composite, coverage, min_weight, learnable)
            )

    if len(winners) > 1:
        summary = Table(title="🏆 Best sparse set per dense width")
        summary.add_column("Dense width", justify="right", style="cyan")
        summary.add_column("Best sparse set", style="bold")
        summary.add_column("Composite", justify="right", style="green")
        summary.add_column("Coverage", justify="right")
        summary.add_column("Min softmax wt", justify="right")
        summary.add_column("Learnable", justify="center")
        for (
            dense_width,
            sparse,
            composite,
            coverage,
            min_weight,
            learnable,
        ) in winners:
            learn_mark = (
                "[green]\u2713[/green]" if learnable else "[red]\u2717[/red]"
            )
            summary.add_row(
                str(dense_width),
                str(sparse),
                f"{composite:,.1f}",
                f"{coverage:,.1f}",
                f"{min_weight:.2e}",
                learn_mark,
            )
        console.print(summary)
        console.print()


# --- CLI ---


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="⚡ DSQG sparse offset configuration explorer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser(
        "rank", help="Rank all C(pool, n_sparse) combinations"
    )
    rank_parser.add_argument(
        "--dense", type=int, required=True, help="Dense width"
    )
    rank_parser.add_argument(
        "--n-sparse", type=int, required=True, help="Number of sparse offsets"
    )
    rank_parser.add_argument(
        "--pool", type=str, default=None, help="Sparse pool (comma-separated)"
    )
    rank_parser.add_argument(
        "--top", type=int, default=10, help="Show top N results"
    )

    score_parser = subparsers.add_parser(
        "score", help="Detailed score for a specific sparse set"
    )
    score_parser.add_argument(
        "--dense", type=int, required=True, help="Dense width"
    )
    score_parser.add_argument(
        "--sparse", type=str, required=True, help="Sparse offsets (comma-separated)"
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Side-by-side comparison of sparse sets"
    )
    compare_parser.add_argument(
        "--dense", type=int, required=True, help="Dense width"
    )
    compare_parser.add_argument(
        "--sets", type=str, required=True, help='Semicolon-separated sparse sets'
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="Sweep across dense widths"
    )
    sweep_parser.add_argument(
        "--dense-range", type=str, required=True, help="Dense widths (comma-separated)"
    )
    sweep_parser.add_argument(
        "--n-sparse", type=int, required=True, help="Number of sparse offsets"
    )
    sweep_parser.add_argument(
        "--pool", type=str, default=None, help="Sparse pool (comma-separated)"
    )
    sweep_parser.add_argument(
        "--top", type=int, default=5, help="Show top N per dense width"
    )

    args = parser.parse_args()

    if args.command == "rank":
        pool = parse_int_list(args.pool) if args.pool else SPARSE_POOL
        display_rank(args.dense, args.n_sparse, pool, args.top)

    elif args.command == "score":
        sparse = parse_int_list(args.sparse)
        display_score(sparse, args.dense)

    elif args.command == "compare":
        sets = [parse_int_list(s) for s in args.sets.split(";")]
        display_compare(args.dense, sets)

    elif args.command == "sweep":
        dense_range = parse_int_list(args.dense_range)
        pool = parse_int_list(args.pool) if args.pool else SPARSE_POOL
        display_sweep(dense_range, args.n_sparse, pool, args.top)


if __name__ == "__main__":
    main()
