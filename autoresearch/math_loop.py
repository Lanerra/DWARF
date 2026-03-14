import json
import subprocess
import time
import os
import sys
import traceback
from datetime import datetime

RUST_BINARY = "exploration/math_autoresearch/target/release/math_autoresearch"
CONFIG_FILE = "exploration/math_autoresearch/current_config.json"
RESULTS_FILE = "exploration/math_autoresearch/current_results.json"
HISTORY_FILE = "autoresearch/math_history.jsonl"
SLEEP_SECONDS = 4
MAX_HISTORY_IN_PROMPT = 15
PROJECT_ROOT = "/home/dlewis3/Desktop/AI/DWARF"

PROMPT_TEMPLATE = """You are searching for the optimal wavelet filter for DSQG attention (sparse dyadic attention).
DSQG has 3 offset zones: dense \u03b4=1..48, mid \u03b4=96,128, long \u03b4=384.
We need a wavelet that best represents these multi-scale patterns.

Current best score: {best_score:.4f} (D4 baseline: ~0.834)
Best config: {best_config}

Last {history_count} results (most recent first):
{history_summary}

Metrics explanation:
- sparse_approx_mse: lower = better K-sparse approximation
- scale_separation: higher = better alignment with DSQG's 3 zones
- dsqg_coverage: higher = better localization per zone
- leakage: lower = less cross-zone bleeding (high values are bad)
- gradient_flow: higher = easier to train through
- composite_score: the single number to maximize

Search space:
- filter_taps: 4, 6, or 8 (more taps = more localization, slower)
- lattice_angles: list of (filter_taps/2 - 1) angles in radians, range [0, \u03c0]
  - 4-tap: 1 angle. D4 = [1.0472]. Try near \u03c0/4=0.785, \u03c0/2=1.571, 2\u03c0/3=2.094
  - 6-tap: 2 angles. More localization at cost of longer filter.
  - 8-tap: 3 angles. Maximum localization.
- levels_with_gelu: which DWT levels get GELU nonlinearity [0,2,4,6] = alternating
  - Options: [0,2,4,6], [6,7], [5,6,7], [0,2,4,5,6,7], [4,5,6,7]
- block_structure: "full" (cross-head mixing) or "per_head" (independent)

Based on the history, propose the NEXT configuration to try.
Be strategic \u2014 use the pattern of results to guide the search.
Output ONLY valid JSON matching this schema:
{{
  "search_mode": "wavelet_filter",
  "description": "one sentence explaining the hypothesis",
  "wavelet": {{
    "filter_taps": <4|6|8>,
    "lattice_angles": [<angle_radians>, ...],
    "levels_with_gelu": [<level_ints>],
    "block_structure": "<full|per_head>"
  }}
}}"""

D4_CONFIG = {
    "search_mode": "wavelet_filter",
    "description": "D4 Daubechies baseline (\u03b8=\u03c0/3)",
    "wavelet": {
        "filter_taps": 4,
        "lattice_angles": [1.0472],
        "levels_with_gelu": [0, 2, 4, 6],
        "block_structure": "full",
    },
}


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    entries = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_history_summary(history):
    recent = list(reversed(history[-MAX_HISTORY_IN_PROMPT:]))
    lines = []
    for entry in recent:
        config = entry.get("config", {})
        results = entry.get("results", {})
        metrics = results.get("metrics", {})
        description = config.get("description", "")[:60]
        score = entry.get("composite_score", 0)
        delta = results.get("vs_d4_delta", 0)
        is_best = entry.get("is_best", False)
        taps = config.get("wavelet", {}).get("filter_taps", "?")
        angles = config.get("wavelet", {}).get("lattice_angles", [])
        gelu = config.get("wavelet", {}).get("levels_with_gelu", [])
        lines.append(
            f"  score={score:.4f} (vs D4: {delta:+.4f})"
            f"{'*BEST*' if is_best else ''} "
            f"taps={taps} angles={angles} gelu={gelu} "
            f"mse={metrics.get('sparse_approx_mse', '?'):.4e} "
            f"sep={metrics.get('scale_separation', '?'):.4f} "
            f"leak={metrics.get('leakage', '?'):.3f} "
            f"| {description}"
        )
    return "\n".join(lines) if lines else "  (no history yet)"


def run_evaluation(config):
    with open(os.path.join(PROJECT_ROOT, CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2)

    result = subprocess.run(
        [os.path.join(PROJECT_ROOT, RUST_BINARY)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Rust evaluator failed (exit {result.returncode}): {result.stderr}"
        )

    with open(os.path.join(PROJECT_ROOT, RESULTS_FILE)) as f:
        return json.load(f)


def propose_next_config(history, best_score, best_config):
    import subprocess

    history_summary = format_history_summary(history)
    best_config_str = json.dumps(best_config, indent=2) if best_config else "None yet"

    prompt = PROMPT_TEMPLATE.format(
        best_score=best_score,
        best_config=best_config_str,
        history_count=min(len(history), MAX_HISTORY_IN_PROMPT),
        history_summary=history_summary,
    )

    for attempt in range(2):
        result = subprocess.run(
            ["claude", "--print", "--model", "claude-haiku-4-5", prompt],
            capture_output=True, text=True, timeout=60,
        )
        response_text = result.stdout.strip()
        if not response_text:
            raise RuntimeError(f"claude --print returned no output: {result.stderr[:200]}")

        # Mimic the response object interface used below
        class _Resp:
            pass
        response = _Resp()
        class _Content:
            text = response_text
        response.content = [_Content()]

        response_text = response.content[0].text.strip()

        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_text = response_text[start:end]
        else:
            json_text = response_text

        try:
            config = json.loads(json_text)
            assert config.get("search_mode") == "wavelet_filter"
            wavelet = config["wavelet"]
            assert wavelet["filter_taps"] in (4, 6, 8)
            expected_angles = wavelet["filter_taps"] // 2 - 1
            assert len(wavelet["lattice_angles"]) == expected_angles
            assert isinstance(wavelet["levels_with_gelu"], list)
            assert wavelet["block_structure"] in ("full", "per_head")
            return config
        except (json.JSONDecodeError, KeyError, AssertionError) as parse_error:
            if attempt == 0:
                prompt += (
                    f"\n\nYour previous response was invalid JSON or had wrong schema: "
                    f"{parse_error}. Please output ONLY valid JSON."
                )
                continue
            raise ValueError(
                f"LLM produced invalid config after 2 attempts: {response_text[:200]}"
            )


def main():
    os.chdir(PROJECT_ROOT)

    print(f"{'='*60}")
    print(f"\U0001f30a Math Autoresearch Loop")
    print(f"{'='*60}")

    rust_path = os.path.join(PROJECT_ROOT, RUST_BINARY)
    if not os.path.exists(rust_path):
        print("Compiling Rust evaluator...")
        env = {**os.environ, "PATH": os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"]}
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=os.path.join(PROJECT_ROOT, "exploration/math_autoresearch"),
            env=env,
            check=True,
        )
        print("Compilation complete.")

    history = load_history()
    best_score = 0.0
    best_config = None
    for entry in history:
        score = entry.get("composite_score", 0)
        if score > best_score:
            best_score = score
            best_config = entry.get("config")
    iteration = len(history)

    print(f"Loaded {len(history)} previous entries. Best score: {best_score:.4f}")
    print(f"Starting at iteration {iteration}")
    print()

    while True:
        try:
            history = load_history()

            if not history:
                config = D4_CONFIG
                print(f"[{iteration}] Seeding with D4 baseline...")
            else:
                print(f"[{iteration}] Proposing next config via LLM...")
                config = propose_next_config(history, best_score, best_config)

            print(f"  Config: {config.get('description', '')[:70]}")
            print(f"  Taps={config['wavelet']['filter_taps']} Angles={config['wavelet']['lattice_angles']}")

            results = run_evaluation(config)
            score = results["composite_score"]
            is_best = score > best_score

            if is_best:
                best_score = score
                best_config = config

            entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "results": results,
                "composite_score": score,
                "is_best": is_best,
            }

            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")

            marker = " *** BEST ***" if is_best else ""
            delta = results.get("vs_d4_delta", 0)
            print(
                f"  score={score:.4f} (vs D4: {delta:+.4f}){marker}"
            )
            print()

            iteration += 1
            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as loop_error:
            print(f"  Error: {loop_error}")
            traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main()
