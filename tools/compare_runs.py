#!/usr/bin/env python3
"""Rich TUI dashboard comparing all DWARF training runs."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


KNOWN_PARAMS = {
    "d41s3": 13_800_000,
    "d41s5": 13_800_000,
    "d41s3_seinit": 13_800_000,
    "condX": 13_800_000,
    "condX_v2": 13_800_000,
    "d41_35m": 38_732_618,
}

REFERENCES = {
    "I3G0": {"test_ppl": 52.948, "passkey": 0.533},
    "condU": {"test_ppl": 52.237, "passkey": 0.383},
    "condM": {"test_ppl": 54.529, "passkey": 0.833},
}

PASSKEY_DISTANCES = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "1536"]

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values):
    if not values:
        return ""
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    if spread == 0:
        return SPARKLINE_CHARS[4] * len(values)
    result = []
    for value in values:
        index = int((value - minimum) / spread * (len(SPARKLINE_CHARS) - 1))
        result.append(SPARKLINE_CHARS[index])
    return "".join(result)


def format_params(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


def load_runs(logs_directory):
    runs = []
    for path in sorted(Path(logs_directory).glob("*_results.json")):
        with open(path) as file_handle:
            data = json.load(file_handle)
        if "per_epoch" not in data:
            continue
        run_name = path.stem.replace("_results", "")
        data["_run_name"] = run_name
        data["_path"] = str(path)
        runs.append(data)
    return runs


def ppl_color(test_ppl):
    if test_ppl is None:
        return "dim"
    if test_ppl < REFERENCES["I3G0"]["test_ppl"]:
        return "green"
    if test_ppl < REFERENCES["condM"]["test_ppl"]:
        return "yellow"
    return "red"


def passkey_color(passkey_mean):
    if passkey_mean is None:
        return "dim"
    if passkey_mean >= 0.70:
        return "green"
    if passkey_mean >= 0.50:
        return "yellow"
    return "red"


def passkey_cell_color(value):
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


def best_val_ppl(run):
    epochs = run.get("per_epoch", [])
    if not epochs:
        return None
    return min(epoch.get("val_ppl", float("inf")) for epoch in epochs)


def run_status(run):
    if run.get("final_test_ppl") is not None:
        return "COMPLETE"
    epochs = run.get("per_epoch", [])
    current = len(epochs)
    return f"TRAINING (ep {current}/10)"


def build_table(runs):
    table = Table(
        title="🔬 DWARF Training Run Comparison",
        show_lines=True,
        title_style="bold bright_blue",
    )
    table.add_column("Run Name", style="bold")
    table.add_column("Test PPL", justify="right")
    table.add_column("Best Val PPL", justify="right")
    table.add_column("Passkey%", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Epochs", justify="center")
    table.add_column("Status", justify="center")

    sorted_runs = sorted(
        runs,
        key=lambda run: run.get("final_test_ppl") or float("inf"),
    )

    for run in sorted_runs:
        name = run["_run_name"]
        test_ppl = run.get("final_test_ppl")
        best_val = best_val_ppl(run)
        passkey_mean = run.get("final_passkey_mean")
        params = run.get("params") or KNOWN_PARAMS.get(name)
        epoch_count = len(run.get("per_epoch", []))
        status = run_status(run)

        color = ppl_color(test_ppl)
        ppl_text = Text(f"{test_ppl:.3f}" if test_ppl is not None else "—", style=color)
        val_text = Text(f"{best_val:.3f}" if best_val is not None else "—", style=color)

        passkey_style = passkey_color(passkey_mean)
        passkey_text = Text(
            f"{passkey_mean * 100:.1f}%" if passkey_mean is not None else "—",
            style=passkey_style,
        )

        params_text = format_params(params) if params else "—"
        status_style = "green" if status == "COMPLETE" else "yellow"

        table.add_row(
            name,
            ppl_text,
            val_text,
            passkey_text,
            params_text,
            str(epoch_count),
            Text(status, style=status_style),
        )

    table.add_section()
    for reference_name, reference_values in REFERENCES.items():
        ppl_value = reference_values["test_ppl"]
        passkey_value = reference_values["passkey"]
        table.add_row(
            Text(f"⟶ {reference_name} (ref)", style="dim"),
            Text(f"{ppl_value:.3f}", style="dim"),
            Text("—", style="dim"),
            Text(f"{passkey_value * 100:.1f}%", style="dim"),
            Text("—", style="dim"),
            Text("—", style="dim"),
            Text("REF", style="dim"),
        )

    return table


def passkey_block(value):
    if value >= 0.8:
        return ("█", "green")
    if value >= 0.5:
        return ("▓", "yellow")
    if value > 0.0:
        return ("░", "red")
    return ("·", "dim")


def build_detail_table(run):
    name = run["_run_name"]
    epochs = run.get("per_epoch", [])

    table = Table(
        title=f"🔬 {name} — Epoch Detail",
        show_lines=True,
        title_style="bold bright_blue",
    )
    table.add_column("Ep", justify="center", width=3)
    table.add_column("Val PPL", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("%C", justify="right")
    table.add_column("PK%", justify="right")
    table.add_column("1 2 4 8 . . . . . . 1k.", no_wrap=True)
    table.add_column("ScEmb", justify="right")
    table.add_column("EMA", justify="right")
    table.add_column("KdV", justify="right")

    val_ppls = [epoch.get("val_ppl", 0) for epoch in epochs]
    best_epoch_ppl = min(val_ppls) if val_ppls else float("inf")

    for epoch in epochs:
        epoch_number = epoch.get("epoch", "?")
        val_ppl = epoch.get("val_ppl")
        train_loss = epoch.get("train_loss")
        chinchilla = epoch.get("chinchilla_pct")
        passkey_mean = epoch.get("passkey_mean")
        passkey_by_distance = epoch.get("passkey_by_d", {})
        scale_embed = epoch.get("scale_embed_abs_mean") or epoch.get("scale_embed_abs_max")
        ema_factors = epoch.get("ema_factors", [])
        kdv_alphas = epoch.get("kdv_alphas", [])

        is_best = val_ppl is not None and val_ppl == best_epoch_ppl
        ppl_style = "bold green" if is_best else ""
        ppl_text = Text(f"{val_ppl:.2f}" if val_ppl is not None else "—", style=ppl_style)

        passkey_grid = Text()
        for index, distance in enumerate(PASSKEY_DISTANCES):
            value = passkey_by_distance.get(distance, 0.0)
            character, color = passkey_block(value)
            passkey_grid.append(character, style=color)
            if index < len(PASSKEY_DISTANCES) - 1:
                passkey_grid.append(" ")

        passkey_text = Text(
            f"{passkey_mean * 100:.0f}%" if passkey_mean is not None else "—",
            style=passkey_color(passkey_mean),
        )

        ema_text = f"{ema_factors[0]:.5f}" if ema_factors else "—"
        kdv_text = f"{kdv_alphas[0]:.5f}" if kdv_alphas else "—"
        scale_text = f"{scale_embed:.3f}" if scale_embed is not None else "—"

        table.add_row(
            str(epoch_number),
            ppl_text,
            f"{train_loss:.3f}" if train_loss is not None else "—",
            f"{chinchilla:.0f}" if chinchilla is not None else "—",
            passkey_text,
            passkey_grid,
            scale_text,
            ema_text,
            kdv_text,
        )

    table.caption = Text()
    table.caption.append("Val PPL: ")
    table.caption.append(sparkline(val_ppls))
    passkey_means = [epoch.get("passkey_mean", 0) for epoch in epochs]
    table.caption.append("  Passkey: ")
    table.caption.append(sparkline(passkey_means))
    scale_values = [
        epoch.get("scale_embed_abs_mean") or epoch.get("scale_embed_abs_max", 0)
        for epoch in epochs
    ]
    table.caption.append("  Scale Embed: ")
    table.caption.append(sparkline(scale_values))
    ema_values = [epoch.get("ema_factors", [0])[0] for epoch in epochs]
    table.caption.append("  EMA: ")
    table.caption.append(sparkline(ema_values))

    return table


def find_run(runs, run_name):
    for run in runs:
        if run["_run_name"] == run_name:
            return run
    available = [run["_run_name"] for run in runs]
    print(f"Run '{run_name}' not found. Available: {', '.join(available)}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="DWARF training run comparison dashboard")
    parser.add_argument(
        "--mode",
        choices=["table", "detail", "live"],
        default="table",
        help="Display mode (default: table)",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run name for --mode detail",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="benchmarks/logs/",
        help="Directory containing *_results.json files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump parsed data as formatted JSON instead of TUI",
    )
    arguments = parser.parse_args()

    runs = load_runs(arguments.logs)
    if not runs:
        print(f"No training run results found in {arguments.logs}", file=sys.stderr)
        sys.exit(1)

    if arguments.json:
        output = []
        for run in runs:
            cleaned = {key: value for key, value in run.items() if not key.startswith("_")}
            cleaned["_run_name"] = run["_run_name"]
            output.append(cleaned)
        json.dump(output, sys.stdout, indent=2)
        print()
        return

    console = Console()

    if arguments.mode == "detail":
        if not arguments.run:
            available = [run["_run_name"] for run in runs]
            print(
                f"--run required for detail mode. Available: {', '.join(available)}",
                file=sys.stderr,
            )
            sys.exit(1)
        run = find_run(runs, arguments.run)
        console.print(build_detail_table(run))
        return

    if arguments.mode == "live":
        import time

        with Live(build_table(runs), console=console, refresh_per_second=1 / 30) as live:
            while True:
                time.sleep(30)
                runs = load_runs(arguments.logs)
                live.update(build_table(runs))

    console.print(build_table(runs))


if __name__ == "__main__":
    main()
