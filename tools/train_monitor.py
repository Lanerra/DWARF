#!/usr/bin/env python3
"""Real-time TUI dashboard for monitoring DWARF LLM training runs."""

import re
import sys
import time
from collections import deque
from pathlib import Path

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

STEP_LOSS_PATTERN = re.compile(
    r"Step\s+(\d+)/(\d+)\s+\|\s+Loss\s+([\d.]+)"
)

GRAD_NORM_PATTERN = re.compile(
    r"\[step\s+(\d+)\]\s+gn=(.*?)\s+pv=(.*)"
)

EPOCH_PATTERN = re.compile(
    r"Ep\s+(\d+)/(\d+)\s+\|\s+Train\s+([\d.]+)\s+\|\s+Val\s+([\d.]+)\s+"
    r"PPL\s+([\d.]+)\s*(\*\s*BEST)?\s*\|\s*(\d+)s\s+\((\d+)%C\)"
)

DSQG_POSBIAS_PATTERN = re.compile(
    r"DSQG pos-bias:\s+\|mean\|=([\d.]+)\s+\|max\|=([\d.]+)\s+"
    r"most-local=(\S+)\s+most-global=(\S+)"
)

SCALE_EMBED_PATTERN = re.compile(
    r"scale_embed:\s+\|mean\|=([\d.]+)\s+\|max\|=([\d.]+)"
)

IF_GAINS_PATTERN = re.compile(r"IF gains:\s+(.*)")

EMA_FACTORS_PATTERN = re.compile(r"EMA factors:\s+(.*)")

KDV_ALPHAS_PATTERN = re.compile(r"KdV alphas:\s+(.*)")

PASSKEY_SUMMARY_PATTERN = re.compile(
    r"mean=([\d.]+)%\s+\((\d+)/(\d+)\s+distances\s+>50%\)"
)

PASSKEY_DISTANCES_PATTERN = re.compile(r"d=(\d+):([\d.]+)%")


def parse_key_value_pairs(text):
    pairs = {}
    for token in text.strip().split():
        if ":" in token:
            key, value = token.split(":", 1)
            pairs[key] = value
    return pairs


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


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_eta(seconds):
    if seconds <= 0:
        return "--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class TrainingState:
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.steps_per_epoch = 0
        self.latest_step_loss = None
        self.recent_losses = deque(maxlen=10)
        self.train_loss = None
        self.val_ppl = None
        self.best_ppl = None
        self.best_ppl_epoch = None
        self.chinchilla_percent = None
        self.grad_norms = {}
        self.param_values = {}
        self.grad_norm_step = None
        self.pos_bias_mean = None
        self.pos_bias_max = None
        self.pos_bias_most_local = None
        self.pos_bias_most_global = None
        self.scale_embed_mean = None
        self.scale_embed_max = None
        self.if_gains = {}
        self.ema_factors_raw = ""
        self.kdv_alphas_raw = ""
        self.passkey_mean = None
        self.passkey_above_50 = None
        self.passkey_total = None
        self.passkey_distances = {}
        self.epoch_history = []
        self.recent_lines = deque(maxlen=6)
        self.start_time = time.time()

    @property
    def total_steps(self):
        if self.steps_per_epoch > 0 and self.total_epochs > 0:
            return self.total_epochs * self.steps_per_epoch
        return 0

    @property
    def global_step(self):
        if self.steps_per_epoch > 0 and self.current_epoch > 0:
            return (self.current_epoch - 1) * self.steps_per_epoch + self.current_step
        return self.current_step

    @property
    def progress_fraction(self):
        total = self.total_steps
        if total == 0:
            return 0.0
        return min(self.global_step / total, 1.0)

    @property
    def eta_seconds(self):
        elapsed = time.time() - self.start_time
        fraction = self.progress_fraction
        if fraction <= 0:
            return 0
        return elapsed / fraction * (1 - fraction)

    @property
    def grad_norm_status(self):
        if not self.grad_norms:
            return "dim"
        values = list(self.grad_norms.values())
        if any(v == "inf" for v in values):
            return "red"
        try:
            if any(float(v) > 5.0 for v in values):
                return "yellow"
        except ValueError:
            pass
        return "green"

    def process_line(self, line):
        stripped = line.rstrip("\n")
        if stripped.strip():
            self.recent_lines.append(stripped)

        match = STEP_LOSS_PATTERN.search(stripped)
        if match:
            self.current_step = int(match.group(1))
            self.steps_per_epoch = int(match.group(2))
            self.latest_step_loss = float(match.group(3))
            self.recent_losses.append(self.latest_step_loss)
            return

        match = GRAD_NORM_PATTERN.search(stripped)
        if match:
            self.grad_norm_step = int(match.group(1))
            self.grad_norms = parse_key_value_pairs(match.group(2))
            self.param_values = parse_key_value_pairs(match.group(3))
            return

        match = EPOCH_PATTERN.search(stripped)
        if match:
            self.current_epoch = int(match.group(1))
            self.total_epochs = int(match.group(2))
            self.train_loss = float(match.group(3))
            val_loss = float(match.group(4))
            self.val_ppl = float(match.group(5))
            is_best = match.group(6) is not None
            epoch_seconds = int(match.group(7))
            self.chinchilla_percent = int(match.group(8))
            if is_best or self.best_ppl is None or self.val_ppl < self.best_ppl:
                self.best_ppl = self.val_ppl
                self.best_ppl_epoch = self.current_epoch
            self.epoch_history.append({
                "epoch": self.current_epoch,
                "train_loss": self.train_loss,
                "val_loss": val_loss,
                "val_ppl": self.val_ppl,
                "is_best": is_best,
                "seconds": epoch_seconds,
            })
            return

        match = DSQG_POSBIAS_PATTERN.search(stripped)
        if match:
            self.pos_bias_mean = float(match.group(1))
            self.pos_bias_max = float(match.group(2))
            self.pos_bias_most_local = match.group(3)
            self.pos_bias_most_global = match.group(4)
            return

        match = SCALE_EMBED_PATTERN.search(stripped)
        if match:
            self.scale_embed_mean = float(match.group(1))
            self.scale_embed_max = float(match.group(2))
            return

        match = IF_GAINS_PATTERN.search(stripped)
        if match:
            self.if_gains = parse_key_value_pairs(match.group(1))
            return

        match = EMA_FACTORS_PATTERN.search(stripped)
        if match:
            self.ema_factors_raw = match.group(1).strip()
            return

        match = KDV_ALPHAS_PATTERN.search(stripped)
        if match:
            self.kdv_alphas_raw = match.group(1).strip()
            return

        match = PASSKEY_SUMMARY_PATTERN.search(stripped)
        if match:
            self.passkey_mean = float(match.group(1))
            self.passkey_above_50 = int(match.group(2))
            self.passkey_total = int(match.group(3))
            return

        distance_matches = PASSKEY_DISTANCES_PATTERN.findall(stripped)
        if distance_matches:
            for distance, percentage in distance_matches:
                self.passkey_distances[int(distance)] = float(percentage)


def build_progress_bar(fraction, width=30):
    filled = int(fraction * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def render_header(state):
    elapsed = format_duration(time.time() - state.start_time)
    epoch_text = f"Ep {state.current_epoch}/{state.total_epochs}" if state.total_epochs else "Ep --/--"
    step_text = (
        f"Step {state.current_step}/{state.steps_per_epoch}"
        if state.steps_per_epoch
        else "Step --/--"
    )
    percentage = state.progress_fraction * 100
    bar = build_progress_bar(state.progress_fraction)
    eta = format_eta(state.eta_seconds)
    chinchilla = f"{state.chinchilla_percent}%C" if state.chinchilla_percent else ""

    line = f"{epoch_text}  {step_text} [{bar}] {percentage:.1f}%  ETA: {eta}  {chinchilla}"
    return Panel(
        Text(line),
        title="🔬 DWARF Training Monitor",
        subtitle=f"elapsed: {elapsed}",
        border_style="bright_blue",
    )


def render_loss(state):
    lines = []
    if state.latest_step_loss is not None:
        spark = sparkline(list(state.recent_losses))
        lines.append(f"Step:   {state.latest_step_loss:.4f}  {spark}")
    if state.train_loss is not None:
        lines.append(f"Train:  {state.train_loss:.4f}")
    if state.val_ppl is not None:
        lines.append(f"Val PPL: {state.val_ppl:.1f}")
    if state.best_ppl is not None:
        lines.append(f"Best:   {state.best_ppl:.1f} (ep {state.best_ppl_epoch})")
    if not lines:
        lines.append("Waiting for data...")
    return Panel("\n".join(lines), title="📉 Loss", border_style="cyan")


def render_parameters(state):
    lines = []
    if state.pos_bias_mean is not None:
        lines.append(
            f"pos_bias  |mean|: {state.pos_bias_mean:.3f}  "
            f"|max|: {state.pos_bias_max:.3f}"
        )
    if state.scale_embed_mean is not None:
        lines.append(
            f"scale_emb |mean|: {state.scale_embed_mean:.3f}  "
            f"|max|: {state.scale_embed_max:.3f}"
        )
    if state.ema_factors_raw:
        lines.append(f"EMA: {state.ema_factors_raw}")
    if state.kdv_alphas_raw:
        lines.append(f"KdV: {state.kdv_alphas_raw}")
    if state.if_gains:
        gains_text = "  ".join(f"{k}:{v}" for k, v in state.if_gains.items())
        lines.append(f"IF gains: {gains_text}")
    if not lines:
        lines.append("Waiting for data...")
    return Panel("\n".join(lines), title="⚙ Parameters", border_style="cyan")


def render_grad_norms(state):
    status = state.grad_norm_status
    if status == "dim":
        border = "dim"
        status_text = "No data yet"
    elif status == "red":
        border = "red"
        status_text = "⚠ INF DETECTED"
    elif status == "yellow":
        border = "yellow"
        status_text = "⚠ High grad norms (>5.0)"
    else:
        border = "green"
        status_text = "✓ OK"

    lines = []
    if state.grad_norms:
        step_label = f"[step {state.grad_norm_step}] " if state.grad_norm_step else ""
        norms_text = "  ".join(f"{k}:{v}" for k, v in state.grad_norms.items())
        lines.append(f"{step_label}{norms_text}")
    lines.append(f"Status: {status_text}")
    return Panel("\n".join(lines), title="📊 Grad norms", border_style=border)


def render_passkey(state):
    if not state.passkey_distances:
        return Panel("No passkey data yet", title="🔑 Passkey", border_style="dim")

    sorted_distances = sorted(state.passkey_distances.keys())
    header_row = []
    value_row = []
    for distance in sorted_distances:
        percentage = state.passkey_distances[distance]
        distance_label = f"d={distance}"
        header_row.append(f"{distance_label:>6}")
        if percentage >= 80:
            color = "green"
        elif percentage >= 50:
            color = "yellow"
        else:
            color = "red"
        value_row.append(f"[{color}]{percentage:5.0f}%[/{color}]")

    summary = ""
    if state.passkey_mean is not None:
        summary = (
            f"  mean={state.passkey_mean:.1f}%  "
            f"({state.passkey_above_50}/{state.passkey_total} distances >50%)"
        )

    text = Text.from_markup(
        " ".join(header_row) + "\n" + " ".join(value_row) + "\n" + summary
    )
    return Panel(text, title="🔑 Passkey", border_style="cyan")


def render_val_history(state):
    if not state.epoch_history:
        return Panel("No epoch data yet", title="📈 Val PPL history", border_style="dim")

    entries = []
    for entry in state.epoch_history:
        marker = "*" if entry["is_best"] else ""
        entries.append(f"Ep{entry['epoch']}: {entry['val_ppl']:.1f}{marker}")
    return Panel("  ".join(entries), title="📈 Val PPL history", border_style="cyan")


def render_recent_log(state):
    if not state.recent_lines:
        return Panel("Waiting for log output...", title="📜 Recent log", border_style="dim")
    return Panel(
        "\n".join(state.recent_lines),
        title="📜 Recent log",
        border_style="dim",
    )


def build_layout(state):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", size=7),
        Layout(name="grad_norms", size=5),
        Layout(name="passkey", size=6),
        Layout(name="val_history", size=3),
        Layout(name="recent_log", size=9),
    )
    layout["header"].update(render_header(state))
    layout["middle"].split_row(
        Layout(name="loss", ratio=1),
        Layout(name="parameters", ratio=2),
    )
    layout["middle"]["loss"].update(render_loss(state))
    layout["middle"]["parameters"].update(render_parameters(state))
    layout["grad_norms"].update(render_grad_norms(state))
    layout["passkey"].update(render_passkey(state))
    layout["val_history"].update(render_val_history(state))
    layout["recent_log"].update(render_recent_log(state))
    return layout


def tail_file(filepath):
    with open(filepath, "r") as file_handle:
        file_handle.seek(0, 2)
        while True:
            line = file_handle.readline()
            if line:
                yield line
            else:
                time.sleep(0.5)


def read_stdin():
    for line in sys.stdin:
        yield line


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile>  or  ... | {sys.argv[0]} -")
        sys.exit(1)

    source = sys.argv[1]
    if source == "-":
        line_source = read_stdin()
    else:
        path = Path(source)
        if not path.exists():
            print(f"File not found: {source}")
            sys.exit(1)
        line_source = tail_file(str(path))

    state = TrainingState()

    try:
        with Live(build_layout(state), refresh_per_second=2, screen=True) as live:
            for line in line_source:
                state.process_line(line)
                live.update(build_layout(state))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
