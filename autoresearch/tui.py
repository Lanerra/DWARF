#!/usr/bin/env python3
"""🌊 DWARF Autoresearch Monitor — real-time TUI for the math autoresearch loop."""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "autoresearch" / "math_history.jsonl"
LOOP_LOG_PATH = BASE_DIR / "autoresearch" / "math_loop_stdout.log"
MODULES_DIR = BASE_DIR / "autoresearch" / "modules"
TRAINING_LOG_PATH = BASE_DIR / "benchmarks" / "logs" / "d48_optimal_14m.log"


def read_tail_lines(filepath: Path, count: int) -> list[str]:
    """Read the last `count` lines from a file efficiently."""
    try:
        with open(filepath, "rb") as file:
            file.seek(0, 2)
            size = file.tell()
            if size == 0:
                return []
            buffer = min(size, count * 1024)
            file.seek(max(0, size - buffer))
            lines = file.read().decode("utf-8", errors="replace").splitlines()
            return lines[-count:]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def load_history() -> list[dict]:
    """Load all entries from math_history.jsonl."""
    entries = []
    try:
        with open(HISTORY_PATH) as file:
            for line in file:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return entries


def determine_mode_label(entry: dict) -> str:
    search_mode = entry.get("config", {}).get("search_mode", "")
    mode = entry.get("mode", "")
    if search_mode == "wavelet_filter":
        if mode == "explanation":
            return "WAVELET SEARCH"
        return "WAVELET SEARCH"
    return "EXPLORATION"


def make_header(history: list[dict]) -> Panel:
    try:
        now = datetime.now().strftime("%H:%M:%S")
        if not history:
            text = Text()
            text.append("  🌊 DWARF Autoresearch Monitor", style="bold cyan")
            text.append(f"   [last updated: {now}]", style="dim")
            text.append("  No history yet", style="dim yellow")
            return Panel(text, style="bold blue")

        latest = history[-1]
        iteration = latest.get("iteration", "?")
        mode_label = determine_mode_label(latest)
        exploration_count = sum(
            1 for entry in history if entry.get("mode") == "explanation"
        )
        best_score = max(entry.get("composite_score", 0.0) for entry in history)

        text = Text()
        text.append("  🌊 DWARF Autoresearch Monitor", style="bold cyan")
        text.append(f"   [last updated: {now}]", style="dim")
        text.append(f"  iter={iteration}", style="bold white")
        text.append("\n")
        text.append(f"  Mode: {mode_label}", style="bold magenta")
        text.append(f" | expl_iter={exploration_count}", style="white")
        text.append(f" | best_score={best_score:.4f}", style="bold green")
        return Panel(text, style="bold blue")
    except Exception as exception:
        return Panel(f"[red]Header error: {exception}[/red]", style="red")


def make_best_config(history: list[dict]) -> Panel:
    try:
        if not history:
            return Panel("No history yet", title="🏆 BEST CONFIG", border_style="green")

        best_entry = max(history, key=lambda entry: entry.get("composite_score", 0.0))
        score = best_entry.get("composite_score", 0.0)
        config = best_entry.get("config", {})
        search_mode = config.get("search_mode", "unknown")

        lines = Text()
        lines.append(f"score: {score:.4f}\n", style="bold green")
        lines.append(f"mode: {search_mode}\n", style="yellow")

        wavelet = config.get("wavelet")
        if wavelet:
            taps = wavelet.get("filter_taps", "?")
            angles = wavelet.get("lattice_angles", [])
            lines.append(f"taps: {taps}\n", style="white")
            lines.append(f"θ: {angles}\n", style="white")
            coefficients = best_entry.get("results", {}).get("filter_coefficients", [])
            if coefficients:
                formatted = [f"{coefficient:.3f}" for coefficient in coefficients]
                lines.append("coeff: [", style="dim")
                for index, formatted_value in enumerate(formatted):
                    if index > 0:
                        lines.append(", ", style="dim")
                    if index == 4:
                        lines.append("\n        ", style="dim")
                    lines.append(formatted_value, style="cyan")
                lines.append("]\n", style="dim")
        else:
            parameters = config.get("parameters", {})
            for key, value in list(parameters.items())[:6]:
                lines.append(f"{key}: {value}\n", style="white")

        return Panel(lines, title="🏆 BEST CONFIG", border_style="green")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="🏆 BEST CONFIG", border_style="red")


def make_recent_iterations(history: list[dict]) -> Panel:
    try:
        if not history:
            return Panel("No history yet", title="📊 RECENT ITERATIONS", border_style="cyan")

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Iter", style="dim", width=5, justify="right")
        table.add_column("Mode", width=5)
        table.add_column("Score", width=8, justify="right")
        table.add_column("Search Mode", width=22)
        table.add_column("Description", width=30)

        recent = history[-10:]
        recent.reverse()

        for entry in recent:
            iteration = str(entry.get("iteration", "?"))
            search_mode = entry.get("config", {}).get("search_mode", "")
            is_wavelet = search_mode == "wavelet_filter"
            mode_text = "SRCH" if is_wavelet else "EXPL"
            score = entry.get("composite_score", 0.0)
            is_best = entry.get("is_best", False)
            description = entry.get("config", {}).get("description", "")[:30]
            truncated_mode = search_mode[:22]

            if is_best:
                style = "bold green"
            elif score > 0:
                style = "yellow"
            else:
                style = "dim"

            table.add_row(
                iteration,
                mode_text,
                f"{score:.4f}",
                truncated_mode,
                description,
                style=style,
            )

        return Panel(table, title="📊 RECENT ITERATIONS (last 10)", border_style="cyan")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="📊 RECENT ITERATIONS", border_style="red")


def make_modules_panel() -> Panel:
    try:
        modules = {}

        if MODULES_DIR.is_dir():
            for subdir in sorted(MODULES_DIR.iterdir()):
                if not subdir.is_dir():
                    continue
                status_file = subdir / "status.json"
                try:
                    with open(status_file) as file:
                        data = json.load(file)
                    modules[subdir.name] = data.get("status", "unknown")
                except FileNotFoundError:
                    modules[subdir.name] = "unknown"
                except Exception:
                    modules[subdir.name] = "unknown"

        # wavelet_filter is always-ready built-in (may lack status.json)
        modules["wavelet_filter"] = "ready"

        status_emoji = {
            "ready": "✅",
            "building": "🔨",
            "failed": "❌",
            "unknown": "❓",
        }
        status_order = {"ready": 0, "building": 1, "failed": 2, "unknown": 3}

        sorted_modules = sorted(
            modules.items(), key=lambda item: (status_order.get(item[1], 3), item[0])
        )

        lines = Text()
        for name, status in sorted_modules:
            emoji = status_emoji.get(status, "❓")
            display_name = name[:24]
            lines.append(f"{emoji} {display_name}\n", style="white" if status == "ready" else "dim")

        count = len(sorted_modules)
        return Panel(lines, title=f"📦 MODULES ({count})", border_style="yellow")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="📦 MODULES", border_style="red")


def make_latest_metrics(history: list[dict]) -> Panel:
    try:
        if not history:
            return Panel("No history yet", title="📈 LATEST METRICS", border_style="magenta")

        # Walk backwards to find the most recent entry with non-empty extra_analysis
        extra_analysis = {}
        for entry in reversed(history):
            candidate = entry.get("extra_analysis", {})
            if candidate:
                extra_analysis = candidate
                break

        if not extra_analysis:
            return Panel("No module metrics yet", title="📈 LATEST METRICS", border_style="magenta")

        # Group by module name (keys are "module.metric_name")
        grouped = defaultdict(list)
        for key, value in extra_analysis.items():
            parts = key.split(".", 1)
            if len(parts) != 2:
                continue
            module_name, metric_name = parts
            # Skip non-scalar values
            if isinstance(value, (int, float)):
                grouped[module_name].append((metric_name, value))

        lines = Text()
        for module_name in sorted(grouped.keys()):
            metrics = grouped[module_name]
            # Sort by absolute value descending, take top 4
            metrics.sort(key=lambda pair: abs(pair[1]), reverse=True)
            metrics = metrics[:4]

            lines.append(f"[{module_name}]\n", style="bold cyan")
            for metric_name, value in metrics:
                display_name = metric_name[:36]
                lines.append(f"  {display_name}: ", style="dim")
                lines.append(f"{value:.4f}\n", style="white")

        return Panel(lines, title="📈 LATEST METRICS (most recent eval)", border_style="magenta")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="📈 LATEST METRICS", border_style="red")


def make_training_panel() -> Panel:
    try:
        lines = read_tail_lines(TRAINING_LOG_PATH, 5)
        if not lines:
            content = Text("d48_optimal_14m: not started", style="dim yellow")
            return Panel(content, title="🚂 TRAINING: d48_optimal_14m", border_style="blue")

        text = Text()
        for line in lines:
            if "val_ppl" in line or "ep=" in line:
                text.append(line + "\n", style="bold yellow")
            else:
                text.append(line + "\n", style="white")

        return Panel(text, title="🚂 TRAINING: d48_optimal_14m", border_style="blue")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="🚂 TRAINING", border_style="red")


def make_loop_log() -> Panel:
    try:
        lines = read_tail_lines(LOOP_LOG_PATH, 8)
        if not lines:
            content = Text("No loop output yet", style="dim")
            return Panel(content, title="🔄 LOOP LOG", border_style="dim")

        text = Text()
        for line in lines:
            text.append(line + "\n", style="dim")

        return Panel(text, title="🔄 LOOP LOG (math_loop_stdout.log)", border_style="dim")
    except Exception as exception:
        return Panel(f"[red]Error: {exception}[/red]", title="🔄 LOOP LOG", border_style="red")


def build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="middle", ratio=3),
        Layout(name="training", size=8),
        Layout(name="loop_log", size=11),
    )
    layout["middle"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="best_config", ratio=1),
        Layout(name="modules", ratio=1),
    )
    layout["right"].split_column(
        Layout(name="recent_iterations", ratio=1),
        Layout(name="latest_metrics", ratio=1),
    )
    return layout


def refresh(layout: Layout) -> None:
    history = load_history()
    layout["header"].update(make_header(history))
    layout["best_config"].update(make_best_config(history))
    layout["recent_iterations"].update(make_recent_iterations(history))
    layout["modules"].update(make_modules_panel())
    layout["latest_metrics"].update(make_latest_metrics(history))
    layout["training"].update(make_training_panel())
    layout["loop_log"].update(make_loop_log())


def main() -> None:
    console = Console()
    layout = build_layout()

    refresh(layout)

    try:
        with Live(layout, console=console, refresh_per_second=1, screen=True):
            while True:
                refresh(layout)
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
