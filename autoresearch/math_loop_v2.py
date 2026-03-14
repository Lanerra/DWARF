import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime

PROJECT_ROOT = "/home/dlewis3/Desktop/AI/DWARF"
RUST_BINARY = "exploration/math_autoresearch/target/release/math_autoresearch"
CONFIG_FILE = "exploration/math_autoresearch/current_config.json"
RESULTS_FILE = "exploration/math_autoresearch/current_results.json"
HISTORY_FILE = "autoresearch/math_history.jsonl"
MODULES_DIR = "autoresearch/modules"
SLEEP_SECONDS = 4
MAX_HISTORY_IN_PROMPT = 15
GAP_CHECK_EVERY = 15
STUCK_THRESHOLD = 8
CONVERGENCE_THRESHOLD = 15  # once stuck this long, enter explanation-only mode
EXPLANATION_SLEEP = 30      # longer sleep in explanation mode (gap detection only)
CLAUDE_CMD = "claude"
FINDINGS_FILE = "autoresearch/findings.md"
FINDINGS_UPDATE_EVERY = 40
EVALUATOR_BUILD_THRESHOLD = 2
MAX_CONCURRENT_EVALUATOR_BUILDS = 2

FINDINGS_SYNTHESIS_PROMPT = """You are maintaining a research findings log for DSQG attention autoresearch.

Current established findings:
{current_findings}

New module measurements just collected (iter {iteration}):
{metrics_dump}

Identify findings that are NEW — not already captured in the current findings above.
A finding must be:
- Specific and quantified (include the actual metric value)
- An insight about DSQG structure or optimization direction
- Something that would change what experiments to run

For each new finding, write exactly one bullet in this format:
- CATEGORY finding_description (metric_name=value)

Valid categories: STRUCTURAL, FREQUENCY, COHERENCE, OPTIMIZATION, RELAY, SCALING

If there are no genuinely new findings, respond with exactly: NO_NEW_FINDINGS

Bullet points only. No preamble. No explanation."""

CONFIG_PROMPT = """You are an autonomous research agent for DWARF/DSQG attention — J16D architecture.

CURRENT ARCHITECTURE (J16D — Mar 13 2026):
- J=16 relay offsets: [1,3,4,13,15,21,23,28,48,64,96,192,384,512,768,1024]
- L=6: L0-DSQG + L1-L3-MLP + L4-DSQG-staging + L5-full-attention.
- Scales: 14M (D=256, H=8, d_h=32), 35M (D=512, H=8, d_h=64), 100M (D=1024, H=8, d_h=128).
- Rank bottleneck ELIMINATED: d_h > N_OFFSETS=16 for ALL scales (14M: 32>16, 35M: 64>16).
  SNR: 14M went from -1.3dB (J=44) to +3.3dB (J=16). This is why J=16 works at 14M.
- Two-regime structure confirmed: LOCAL δ=1..128 vs DISTAL δ=384. Head split: 7 local + 1 distal.
- θ (wavelet/phase angle) is a FREE trainable parameter — let it find equilibrium during training.
  Do NOT propose fixing θ or running wavelet_filter search modes. Do NOT use d49 geometry.

HARD CONSTRAINTS (never violate in any proposed config):
- delta=3 AND delta=4 MUST always be in the offset set (coprime, gcd=1, Frobenius=5).
  Ablation: -56.6pp passkey removing δ=3, -46.6pp removing δ=4.
- offsets 128 and 384 must be kept together (co-dependent pair).

WHAT IS SOLVED (do not re-investigate):
- Two-regime local/distal structure (confirmed, score=1.0).
- Head specialization: 7 local + 1 distal is optimal.
- d49 geometry (OBSOLETE — do not use).
- Capped dataset (42% Chinchilla): undertrained baseline — superseded by fulldata.

CURRENT EMPIRICAL RESULTS (as of Mar 13 2026):
- J16D fulldata 35M ep3: PPL=44.04, passkey=75.8%, ar_score=+57.53 (no physics)
- J16D fulldata + physics 35M ep2: PPL=45.38, passkey=76.7%, d=1536=80% (PHYSICS IN PROGRESS)
  - Physics improves both PPL AND passkey simultaneously. EMA collapses to 8t but KdV+AGC help.
  - d=1536 jumped 30%→80% between ep1 and ep2 — KdV soliton enabling multi-hop relay.
- ar_score formula: (passkey_ep3 - 18.3) - max(0, ppl_ep3 - 61.75) * 0.5
  (positive = beats pure J16D fulldata champion; 0 = ties it)

ESTABLISHED FINDINGS FROM MEASUREMENT:
{findings}

OPEN RESEARCH QUESTIONS (focus here):

  1. optimal_j16d_offset_set
     J=16 offsets must cover passkey distances {{{{1,2,4,8,16,32,64,128,256,512,1024,1536}}}} via
     1-2 hop paths. Dense window 1..d_w provides short-range direct coverage. Sparse offsets
     provide single-hop reach to {{{{64,128,...,1536}}}}. Multi-hop: dense+sparse or sparse+sparse.
     Candidate: d10s6 with sparse={{{{64,128,192,256,384,512}}}}. Is this optimal, or does
     {{{{64,128,384,512,768,1536}}}} cover more passkey distances? Analyze path coverage.

  2. scale_embed_threshold_j_dependence
     Old τ=0.74 was J=44. J16D: scale_embed|max|=2.06 by ep2, passkey=10% at ep1.
     Back-projection: J16D crosses τ≈0.45 before ep1. Is this because:
     (a) fewer offsets → less gradient splitting → each offset gets more gradient?
     (b) signal-dominated regime (d_h > N) → cleaner Q·K → lower threshold needed?
     Predict: what scale_embed threshold τ is needed for J=16 vs J=44 relay chain formation?

  3. lm_head_lr_mult_optimization
     arXiv:2603.10145: gradient rank ≤ 2D through tied LM head. Suppression=(V-1-2D)/(V-1).
     14M (D=256,V=32k): suppression=98.4%, analytical lr_mult=63.9.
     35M (D=512,V=32k): suppression=96.8%, analytical lr_mult=31.9.
     PUZZLE: empirical 14M=10, 35M=15 (opposite of theory). Is lr_mult=15 at 35M suboptimal?
     Predict: optimal lr_mult per scale, and why empirical ordering is reversed.

  4. vocab_size_gradient_tradeoff
     V=8k at 14M: suppression drops to 93.8%, lr_mult=16.1, gradient propagation 4× faster.
     Predicted passkey emergence ep1 vs ep3 for V=32k. Trade-off: smaller V means smaller
     embedding table (fewer params). At 14M where every parameter counts, is V=8k better?

  5. Any dimension you identify that follows from J16D physics

Available analysis modules (each returns scores you can use to evaluate a config):
{modules_summary}

Recent history (last {history_count} entries):
{history_summary}

Current best score: {best_score:.4f}

TASK: Propose ONE experiment on any open question above (or a new J16D-relevant dimension).
Focus on mathematical analysis that can be evaluated analytically in milliseconds.
Do NOT propose wavelet_filter experiments. Do NOT use d49 geometry.

Respond with JSON only:
{{"search_mode": "<dimension_name>", "parameters": {{...}}, "description": "hypothesis (1-2 sentences)"}}

Use snake_case for search_mode. Keep description to 1-2 sentences."""

GAP_PROMPT = """You are analyzing DWARF/DSQG attention research to find architectural gaps.

DSQG: sparse dyadic attention — LOCAL regime (δ=1..128: dense δ=1..48 + far-local δ=96,128)
and DISTAL regime (δ=384, anti-phase). 6 layers D=256 H=8. Wavelet filter CONVERGED (θ=1.1377).

Research completed {total_iterations} iterations across multiple dimensions.
Stuck for {stuck_count} iterations — no composite score improvement.

Available analysis modules (what we can currently measure analytically):
{modules_summary}

Recent module outputs for best config (θ=1.1377):
{recent_metrics_summary}

ESTABLISHED FINDINGS (do NOT re-investigate these — they are settled):
{findings}

RESEARCH AGENDA — pick a gap that the current measurements do NOT explain:
A) What determines the optimal POSITION of the standing wave node (why δ=384, not δ=256 or δ=512)?
   Is there a harmonic relationship between θ=1.1377 and log(384)?
B) The 21% unused frequency bridges at the local→distal boundary — what gate mechanism would
   let information cross this node controllably without breaking the resonance?
C) Is interference pooling optimally placed relative to the local/distal regime boundary?
   Layer 5 (current) vs layer 3 (midpoint) vs layering one per regime.
D) Head allocation: H=8 heads for a 2-regime system — should heads explicitly split local/distal?
E) A measurement gap you identify from the module outputs above that none of A-D covers

Respond with ONE of:
{{
  "has_gap": true,
  "gap_description": "one sentence describing what we can't explain",
  "module_name": "snake_case_under_30_chars",
  "hypothesis": "specific testable claim",
  "metrics_to_compute": [{{"name": "metric", "description": "...", "higher_is_better": true}}],
  "rust_spec": "inputs: wavelet config JSON + DSQG constants. Algorithm: [step by step]. Outputs: JSON metrics."
}}
or {{"has_gap": false, "reason": "..."}}"""

MODULE_BRIEF_TEMPLATE = """# Brief: Build the `{name}` analysis module

## Goal
Create a Rust binary at `exploration/{name}/target/release/{name}` that computes
new analysis metrics for DSQG wavelet filter research.

## Hypothesis
{hypothesis}

## Module Interface
The binary must accept exactly 2 command-line arguments:
1. Input JSON path — contains the wavelet config (same format as `exploration/math_autoresearch/current_config.json`)
2. Output JSON path — write a JSON object mapping metric names to float values

Example input:
```json
{{
  "search_mode": "wavelet_filter",
  "wavelet": {{
    "filter_taps": 4,
    "lattice_angles": [1.1377],
    "levels_with_gelu": [5, 6, 7],
    "block_structure": "full"
  }}
}}
```

Example output:
```json
{{
{metrics_example}
}}
```

## Metrics to Compute
{metrics_description}

## Technical Specification
{rust_spec}

## DSQG Constants
- Dense zone: δ = 1..48 (48 consecutive offsets)
- Mid zone: δ = 96, 128 (2 sparse mid-range offsets)
- Long zone: δ = 384 (1 long-range offset)
- Model: 11M params, D=256, H=8, 6 DSQG layers

## Rust Project Setup
1. Create a new Rust project: `cargo init exploration/{name}`
2. Reference existing crates in `exploration/` for patterns:
   - `exploration/wavelet_ffn_sweep/` — wavelet coefficient generation from lattice angles
   - `exploration/film_hyper_sweep/` — parameter sweep patterns
   - `exploration/math_autoresearch/` — the main evaluator (shows how to read wavelet configs)
3. Use only standard dependencies: serde, serde_json, and optionally num-complex, rustfft
4. Build with: `cd exploration/{name} && PATH="$HOME/.cargo/bin:$PATH" cargo build --release`

## Done Criteria
- Binary exists at `exploration/{name}/target/release/{name}`
- Running `exploration/{name}/target/release/{name} <input.json> <output.json>` produces valid JSON output
- All metric values are f64 floats
- Higher values = better (convention), unless noted otherwise in the metric description
"""


EXPLORATION_EVALUATOR_PROMPT = """\
Build a Rust scoring binary at `exploration/{mode}/` for the DSQG autoresearcher.

## What to build
A Rust binary that SCORES configurations along the '{mode}' dimension analytically,
without any GPU training. The binary receives a config JSON and returns a score 0.0-1.0.

## DSQG physics context (use these constants)
Optimal wavelet: θ=1.1377, 4-tap, coefficients=[0.5261, 0.8228, 0.1810, -0.1157]
Two-regime structure: local δ=1..128 (phase-continuous), distal δ=384 (anti-phase)
Key targets:
  - zone_boundary_phase_gradient target = 1.1125 (from tap_zone_phase_bridge)
  - current boundary gradient = 14.2110 (12.75× too steep)
  - cross_zone_phase_coherence_potential = 0.9998 (achievable)
  - tap_phase_gradient_continuity_score = 0.0000 (currently zero)
  - boundary_tap_resonance_factor = 19.0 (boundaries at resonance peaks, should be troughs)
  - inter_zone_log_gap_resonance = 0.9000

Current findings (for scoring context):
{findings}

## Dimension to score
Name: {mode}
Research question: {description}

Example config this binary receives:
```json
{{
  "search_mode": "{mode}",
  "parameters": {params_example},
  "description": "..."
}}
```

## Binary interface
- Args: `<config_path> <results_path>`
- Reads config JSON from config_path
- Writes results JSON to results_path:
  ```json
  {{
    "composite_score": 0.75,
    "metrics": {{"metric_a": 0.8, "metric_b": 0.7, ...}},
    "best_value": <whatever parameter value scored best>,
    "rationale": "one sentence"
  }}
  ```
- Exit 0 on success, non-zero on failure

## Scoring guidance
Design a composite score (0.0=worst, 1.0=best) that measures how well the proposed
parameter values align with the DSQG physics constraints above. Use the wavelet
filter coefficients to compute phase properties analytically. The score should
differentiate meaningfully between different parameter values — don't return 0.5 always.

## Files to create
- `exploration/{mode}/Cargo.toml`
- `exploration/{mode}/src/main.rs`

Run `PATH="$HOME/.cargo/bin:$PATH" cargo build --release` inside the crate directory.

When finished: `openclaw system event --text "Done: evaluator scorer built for {mode}" --mode now`
"""


def load_history():
    history_path = os.path.join(PROJECT_ROOT, HISTORY_FILE)
    if not os.path.exists(history_path):
        return []
    entries = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_history_summary(history, wavelet_only=False):
    if wavelet_only:
        filtered = [e for e in history
                    if e.get("config", {}).get("search_mode", "wavelet_filter") == "wavelet_filter"]
        recent = list(reversed(filtered[-MAX_HISTORY_IN_PROMPT:]))
    else:
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
        mode = config.get("search_mode", "wavelet_filter")
        if mode == "wavelet_filter":
            mse = metrics.get("sparse_approx_mse", None)
            sep = metrics.get("scale_separation", None)
            leak = metrics.get("leakage", None)
            detail = (
                f"taps={taps} angles={angles} gelu={gelu} "
                f"mse={mse:.4e} sep={sep:.4f} leak={leak:.3f}"
                if mse is not None else f"taps={taps} angles={angles}"
            )
        else:
            params = config.get("parameters", {})
            detail = f"mode={mode} params={json.dumps(params)[:60]}"
        lines.append(
            f"  score={score:.4f} (vs D4: {delta:+.4f})"
            f"{'*BEST*' if is_best else ''} "
            f"{detail} | {description}"
        )
    return "\n".join(lines) if lines else "  (no history yet)"


def load_available_modules():
    modules = {
        "wavelet_filter": {
            "description": "Built-in wavelet filter evaluator (composite score from MSE, scale separation, leakage, gradient flow, DSQG coverage)",
            "binary_path": os.path.join(PROJECT_ROOT, RUST_BINARY),
            "ready": True,
            "metrics": [
                "sparse_approx_mse", "scale_separation", "leakage",
                "reconstruction_quality", "gradient_flow",
                "dsqg_dense_coverage", "dsqg_mid_coverage", "dsqg_long_coverage",
            ],
        }
    }

    modules_path = os.path.join(PROJECT_ROOT, MODULES_DIR)
    if not os.path.isdir(modules_path):
        return modules

    for name in os.listdir(modules_path):
        if name == "wavelet_filter":
            continue
        module_dir = os.path.join(modules_path, name)
        if not os.path.isdir(module_dir):
            continue

        status_path = os.path.join(module_dir, "status.json")
        spec_path = os.path.join(module_dir, "spec.md")
        binary_path = os.path.join(PROJECT_ROOT, "exploration", name, "target", "release", name)

        description = name
        if os.path.exists(spec_path):
            with open(spec_path) as f:
                first_lines = f.read(500)
                for line in first_lines.splitlines():
                    if line.startswith("# "):
                        description = line[2:].strip()
                        break

        status = "unknown"
        metrics = []
        if os.path.exists(status_path):
            with open(status_path) as f:
                status_data = json.load(f)
                status = status_data.get("status", "unknown")
                metrics = status_data.get("metrics", [])

        binary_exists = os.path.exists(binary_path)
        ready = binary_exists and status != "failed"

        modules[name] = {
            "description": description,
            "binary_path": binary_path if binary_exists else None,
            "ready": ready,
            "metrics": metrics,
        }

    return modules


def format_modules_summary(modules):
    lines = []
    for name, info in sorted(modules.items()):
        state = "ready" if info["ready"] else "building"
        lines.append(f"  {name} ({state}): {info['description']}")
    return "\n".join(lines)


def load_findings():
    findings_path = os.path.join(PROJECT_ROOT, FINDINGS_FILE)
    if os.path.exists(findings_path):
        with open(findings_path) as f:
            return f.read()
    return "No findings yet."


def synthesize_findings(extra_analysis, iteration):
    current_findings = load_findings()
    metrics_lines = []
    for key, value in sorted(extra_analysis.items()):
        if isinstance(value, (int, float)):
            metrics_lines.append(f"  {key} = {value}")
    metrics_dump = "\n".join(metrics_lines) if metrics_lines else "(no numeric metrics)"

    prompt = FINDINGS_SYNTHESIS_PROMPT.format(
        current_findings=current_findings,
        iteration=iteration,
        metrics_dump=metrics_dump,
    )

    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--model", "claude-haiku-4-5", "--print", prompt],
            capture_output=True, text=True, timeout=60,
            cwd=PROJECT_ROOT,
        )
        response = result.stdout.strip()
        if not response or "NO_NEW_FINDINGS" in response:
            print(f"  📋 Findings synthesis: no new findings at iter {iteration}")
            return

        findings_path = os.path.join(PROJECT_ROOT, FINDINGS_FILE)
        updated = re.sub(
            r"<!-- Last updated: iter \d+ -->",
            f"<!-- Last updated: iter {iteration} -->",
            current_findings,
        )
        updated += f"\n## Auto-discovered (iter {iteration})\n{response}\n"
        with open(findings_path, "w") as f:
            f.write(updated)
        print(f"  📋 Findings updated at iter {iteration}")

    except subprocess.TimeoutExpired:
        print("  ⚠ Findings synthesis timed out")
    except Exception as exception:
        print(f"  ⚠ Findings synthesis error: {exception}")


# Modules that reliably time out or are otherwise broken — skip entirely.
DISABLED_MODULES = {"intermediate_regime_bridging_potential"}


def evaluate_extra_modules(config, modules):
    combined = {}
    search_mode = config.get("search_mode", "wavelet_filter")
    is_wavelet = search_mode == "wavelet_filter"

    for name, info in modules.items():
        if name == "wavelet_filter":
            continue
        if name in DISABLED_MODULES:
            continue
        if not info["ready"]:
            continue
        # Wavelet-specific modules only run on wavelet configs
        wavelet_module = any(kw in name for kw in (
            "wavelet", "zone", "tap", "offset", "dilation", "frequency",
            "inter_zone", "envelope", "spectral", "leakage", "bridge", "frame"
        ))
        if wavelet_module and not is_wavelet:
            continue

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix=f"{name}_in_"
            ) as config_file:
                json.dump(config, config_file)
                config_path = config_file.name

            results_path = config_path.replace("_in_", "_out_")

            result = subprocess.run(
                [info["binary_path"], config_path, results_path],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(results_path):
                with open(results_path) as f:
                    module_results = json.load(f)
                for metric_name, value in module_results.items():
                    combined[f"{name}.{metric_name}"] = value
            else:
                if is_wavelet:  # only warn for wavelet configs; non-wavelet incompatibility is expected
                    print(f"  ⚠ Module {name} failed (exit {result.returncode}): {result.stderr[:100]}")
        except Exception as exception:
            print(f"  ⚠ Module {name} error: {exception}")
        finally:
            for path in [config_path, results_path]:
                if os.path.exists(path):
                    os.unlink(path)

    return combined


def compute_extended_score(base_score, extra_analysis):
    adjustments = {}

    phase_uniformity_key = "inter_zone_phase_alignment.phase_gradient_uniformity"
    if phase_uniformity_key in extra_analysis:
        uniformity = extra_analysis[phase_uniformity_key]
        adjustments["phase_uniformity"] = (uniformity - 0.5) * 0.10

    coherence_key = "tap_zone_phase_bridge.cross_zone_phase_coherence_potential"
    if coherence_key in extra_analysis:
        potential = extra_analysis[coherence_key]
        adjustments["coherence_potential"] = (potential - 0.5) * 0.10

    resonance_key = "multi_zone_tap_phase_continuity.boundary_tap_resonance_factor"
    if resonance_key in extra_analysis:
        resonance = extra_analysis[resonance_key]
        penalty = -min(0.05, (resonance - 1.0) / 18.0 * 0.05)
        adjustments["boundary_resonance"] = penalty

    coverage_key = "zone_frame_completeness_analysis.interzone_frequency_coverage_gap"
    if coverage_key in extra_analysis:
        gap = extra_analysis[coverage_key]
        adjustments["coverage_gap"] = -min(0.03, gap / 256.0 * 0.03)

    total_adjustment = sum(adjustments.values())
    extended_score = base_score + total_adjustment
    return extended_score, adjustments


def run_rust_binary(config):
    """Evaluate config via the appropriate Rust binary, routing by search_mode."""
    mode = config.get("search_mode", "wavelet_filter")

    config_path = os.path.join(PROJECT_ROOT, CONFIG_FILE)
    results_path = os.path.join(PROJECT_ROOT, RESULTS_FILE)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if mode == "wavelet_filter":
        if not isinstance(config.get("wavelet"), dict):
            return {"composite_score": 0.0, "search_mode": mode,
                    "note": "malformed_wavelet_config", "vs_d4_delta": -0.8426}
        binary = os.path.join(PROJECT_ROOT, RUST_BINARY)
    else:
        # Look for a module binary matching the search_mode name
        binary = os.path.join(PROJECT_ROOT, "exploration", mode, "target", "release", mode)
        if not os.path.exists(binary):
            # No evaluator yet — return a placeholder score of 0 and let gap
            # detection know to build one
            print(f"  ℹ No evaluator for search_mode '{mode}' yet — score=0, will trigger build")
            return {"composite_score": 0.0, "search_mode": mode,
                    "note": "no_evaluator", "vs_d4_delta": -0.8426}

    result = subprocess.run(
        [binary] if mode == "wavelet_filter" else [binary, config_path, results_path],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )

    if result.returncode != 0:
        if mode == "wavelet_filter":
            raise RuntimeError(
                f"Rust evaluator failed (exit {result.returncode}): {result.stderr}"
            )
        else:
            # Module binary failed on this config (likely expects wavelet fields) — treat as no evaluator
            print(f"  ⚠ Module evaluator '{mode}' failed (exit {result.returncode}) — score=0")
            return {"composite_score": 0.0, "search_mode": mode,
                    "note": "evaluator_failed", "vs_d4_delta": -0.8426}

    with open(results_path) as f:
        return json.load(f)


def propose_next_config(history, best_score, best_config, modules, force_new_dimension=False, wavelet_override=False):
    wavelet_only = not force_new_dimension
    history_summary = format_history_summary(history, wavelet_only=wavelet_only)
    best_config_string = json.dumps(best_config, indent=2) if best_config else "None yet"
    modules_summary = format_modules_summary(modules)

    findings = load_findings()
    prompt = CONFIG_PROMPT.format(
        best_score=best_score,
        best_config=best_config_string,
        history_count=min(
            len([e for e in history if e.get("config",{}).get("search_mode","wavelet_filter") == "wavelet_filter"])
            if wavelet_only else len(history),
            MAX_HISTORY_IN_PROMPT),
        history_summary=history_summary,
        modules_summary=modules_summary,
        findings=findings,
    )

    if wavelet_override:
        prompt += (
            "\n\nCRITICAL: Return ONLY this exact JSON schema (wavelet_filter, 4-tap, exactly 1 angle):\n"
            '{"search_mode": "wavelet_filter", "description": "...", '
            '"wavelet": {"filter_taps": 4, "lattice_angles": [<ONE float>], '
            '"levels_with_gelu": [], "block_structure": "full"}}\n'
            "The geometry has changed (new bridge offsets 145-340 added). Re-search for optimal θ. "
            "Current baseline: θ=1.1377 scores 0.8328 on new geometry. "
            "Propose ONE specific θ value to explore (e.g. 1.05, 1.20, 0.95, etc.)."
        )

    if force_new_dimension:
        known_modes = get_known_search_modes(history)
        known_modes_str = "\n".join(
            f"  - {m} (proposed {c}×)" for m, c in known_modes[:30]
        )
        prompt += (
            "\n\nIMPORTANT OVERRIDE: The wavelet filter dimension is FULLY SOLVED and "
            "MUST NOT be proposed. You MUST propose an experiment in a completely different "
            "DSQG architectural dimension. Use search_mode != 'wavelet_filter'.\n\n"
            "KNOWN DIMENSIONS (use EXACTLY one of these names if proposing something similar — "
            "do NOT invent slight variations like 'distal_offset_harmonic_resonance_v2' or "
            "'offset_zone_spacing_trough_alignment' if 'distal_offset_harmonic_resonance' or "
            "'offset_zone_spacing' already exists below):\n"
            f"{known_modes_str}\n\n"
            "If your idea is genuinely new and not covered by any existing name above, "
            "use a new concise snake_case name. Otherwise reuse an existing name exactly."
        )

    for attempt in range(2):
        result = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", "claude-haiku-4-5", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        response_text = result.stdout.strip()
        if not response_text:
            raise RuntimeError(f"claude --print returned no output: {result.stderr[:200]}")

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
            assert "search_mode" in config, "missing search_mode"
            mode = config["search_mode"]
            if mode == "wavelet_filter":
                # Strict validation for the known wavelet schema
                wavelet = config["wavelet"]
                assert wavelet["filter_taps"] in (4, 6, 8)
                expected_angles = wavelet["filter_taps"] // 2 - 1
                assert len(wavelet["lattice_angles"]) == expected_angles
                # Normalize gelu key
                if "gelu_levels" in wavelet:
                    wavelet["levels_with_gelu"] = wavelet.pop("gelu_levels")
                wavelet.setdefault("levels_with_gelu", [])
                wavelet.setdefault("block_structure", "full")
            else:
                # New search mode — accept any parameters dict, just needs description
                config.setdefault("parameters", {})
                config.setdefault("description", f"Explore {mode}")
                # Snap to existing name if semantically close (deduplication)
                existing_modes = [m for m, _ in get_known_search_modes(history)]
                normalized = normalize_search_mode(mode, existing_modes)
                if normalized != mode:
                    print(f"  🔀 Dedup: '{mode}' → '{normalized}'")
                    config["search_mode"] = normalized
            return config
        except (json.JSONDecodeError, KeyError, AssertionError, TypeError, ValueError) as parse_error:
            if attempt == 0:
                prompt += (
                    f"\n\nYour previous response was invalid JSON or had wrong schema: "
                    f"{parse_error}. Please output ONLY valid JSON."
                )
                continue
            raise ValueError(
                f"LLM produced invalid config after 2 attempts: {response_text[:200]}"
            )


def detect_explanation_gap(history, best_score, best_config, stuck_count, modules):
    modules_summary = format_modules_summary(modules)

    recent = history[-10:]
    recent_lines = []
    for entry in recent:
        metrics = entry.get("results", {}).get("metrics", {})
        extra = entry.get("extra_analysis", {})
        score = entry.get("composite_score", 0)
        config = entry.get("config", {})
        taps = config.get("wavelet", {}).get("filter_taps", "?")
        angles = config.get("wavelet", {}).get("lattice_angles", [])
        metric_parts = [f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, (int, float))]
        extra_parts = [f"{key}={value:.4f}" for key, value in extra.items() if isinstance(value, (int, float))]
        all_parts = metric_parts + extra_parts
        recent_lines.append(
            f"  score={score:.4f} taps={taps} angles={angles} | {', '.join(all_parts)}"
        )
    recent_metrics_summary = "\n".join(recent_lines)

    findings = load_findings()
    prompt = GAP_PROMPT.format(
        total_iterations=len(history),
        stuck_count=stuck_count,
        modules_summary=modules_summary,
        recent_metrics_summary=recent_metrics_summary,
        findings=findings,
    )

    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", "claude-haiku-4-5", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        response_text = result.stdout.strip()
        if not response_text:
            print("  ⚠ Gap detection: claude returned empty output")
            return None

        # Strip markdown code fences and locate the outermost JSON object
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start < 0 or end <= start:
            print(f"  ⚠ Gap detection: no JSON found in response (len={len(response_text)})")
            return None

        json_text = cleaned[start:end]
        try:
            gap_data = json.loads(json_text)
        except json.JSONDecodeError:
            json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
            gap_data = json.loads(json_text)

        if gap_data.get("has_gap"):
            required_keys = ["module_name", "hypothesis", "metrics_to_compute", "rust_spec"]
            for key in required_keys:
                if key not in gap_data:
                    print(f"  ⚠ Gap detection: missing key '{key}' in response")
                    return None
            return gap_data

        print(f"  ℹ Gap detection: no gap found — {gap_data.get('reason', 'no reason given')}")
        return None

    except json.JSONDecodeError as exception:
        print(f"  ⚠ Gap detection parse error: {exception}")
        return None
    except subprocess.TimeoutExpired:
        print("  ⚠ Gap detection: claude timed out")
        return None
    except Exception as exception:
        print(f"  ⚠ Gap detection error: {exception}")
        return None


def build_module(gap_spec, pending_builds):
    name = gap_spec["module_name"]
    hypothesis = gap_spec["hypothesis"]
    metrics = gap_spec["metrics_to_compute"]
    rust_spec = gap_spec["rust_spec"]

    module_dir = os.path.join(PROJECT_ROOT, MODULES_DIR, name)
    os.makedirs(module_dir, exist_ok=True)

    spec_content = f"# Module: {name}\n\n## Hypothesis\n{hypothesis}\n\n## Metrics\n"
    for metric in metrics:
        direction = "higher=better" if metric.get("higher_is_better", True) else "lower=better"
        spec_content += f"- {metric['name']} ({direction}): {metric['description']}\n"
    spec_content += f"\n## Rust Specification\n{rust_spec}\n"

    with open(os.path.join(module_dir, "spec.md"), "w") as f:
        f.write(spec_content)

    metrics_example = ",\n".join(
        f'  "{m["name"]}": 0.0' for m in metrics
    )
    metrics_description = "\n".join(
        f"- **{m['name']}**: {m['description']} ({'higher=better' if m.get('higher_is_better', True) else 'lower=better'})"
        for m in metrics
    )

    brief_content = MODULE_BRIEF_TEMPLATE.format(
        name=name,
        hypothesis=hypothesis,
        metrics_example=metrics_example,
        metrics_description=metrics_description,
        rust_spec=rust_spec,
    )

    brief_path = os.path.join(module_dir, "brief.md")
    with open(brief_path, "w") as f:
        f.write(brief_content)

    status = {
        "status": "building",
        "started_at": datetime.now().isoformat(),
        "metrics": [m["name"] for m in metrics],
    }
    with open(os.path.join(module_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=2)

    log_path = os.path.join(PROJECT_ROOT, "exploration", f"{name}_module_build.log")
    with open(brief_path) as brief_file:
        brief_text = brief_file.read()

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [CLAUDE_CMD, "--permission-mode", "bypassPermissions", "--print", brief_text],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    pending_builds[name] = {"proc": proc, "log": log_path, "log_file": log_file}
    print(f"  🔧 [module] Spawning Claude Code to build '{name}': {hypothesis[:80]}")


def poll_pending_builds(pending_builds):
    newly_built = []
    finished = []

    for name, build_info in pending_builds.items():
        proc = build_info["proc"]
        if proc.poll() is not None:
            build_info["log_file"].close()
            binary_path = os.path.join(
                PROJECT_ROOT, "exploration", name, "target", "release", name
            )
            status_path = os.path.join(PROJECT_ROOT, MODULES_DIR, name, "status.json")

            if os.path.exists(binary_path):
                status = {"status": "ready", "completed_at": datetime.now().isoformat()}
                with open(status_path, "w") as f:
                    json.dump(status, f, indent=2)
                newly_built.append(name)
                print(f"  ✅ Module '{name}' build complete — binary ready")
            else:
                status = {"status": "failed", "failed_at": datetime.now().isoformat()}
                with open(status_path, "w") as f:
                    json.dump(status, f, indent=2)
                print(f"  ❌ Module '{name}' build failed — no binary produced (see {build_info['log']})")

            finished.append(name)

    for name in finished:
        del pending_builds[name]

    return newly_built


def get_known_search_modes(history):
    """Return list of (mode, count) sorted by count desc, excluding wavelet_filter."""
    counts = {}
    for entry in history:
        mode = entry.get("config", {}).get("search_mode", "wavelet_filter")
        if mode != "wavelet_filter":
            counts[mode] = counts.get(mode, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])


_DEDUP_STOP = {
    "to", "with", "the", "and", "of", "for", "in", "at", "by", "or",
    "a", "an", "optimization", "analysis", "test", "via", "based",
    "using", "exploration", "sweep",
}


def normalize_search_mode(proposed, existing_modes):
    """Snap a proposed search_mode name to an existing one if ≥2 significant words overlap."""
    def sig(name):
        return set(w for w in name.split("_") if w not in _DEDUP_STOP and len(w) > 2)

    proposed_words = sig(proposed)
    best_match, best_overlap = None, 1  # require >1 to match
    for existing in existing_modes:
        overlap = len(proposed_words & sig(existing))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = existing
    return best_match if best_match else proposed


def get_backlog_search_modes(history, pending_evaluator_builds, min_count=None):
    if min_count is None:
        min_count = EVALUATOR_BUILD_THRESHOLD
    counts = {}
    for entry in history:
        config = entry.get("config", {})
        mode = config.get("search_mode", "wavelet_filter")
        if mode == "wavelet_filter":
            continue

        score = entry.get("composite_score", 0.0)
        if score != 0.0:
            continue

        note = entry.get("results", {}).get("note", "")
        if note not in ("no_evaluator", "evaluator_failed", ""):
            continue

        if mode not in counts:
            counts[mode] = {
                "search_mode": mode,
                "count": 0,
                "description": "",
                "parameters": {},
            }
        counts[mode]["count"] += 1
        counts[mode]["description"] = config.get("description", "")
        counts[mode]["parameters"] = config.get("parameters", {})

    backlog = []
    for mode, info in counts.items():
        if info["count"] < min_count:
            continue
        binary_path = os.path.join(
            PROJECT_ROOT, "exploration", mode, "target", "release", mode
        )
        if os.path.exists(binary_path):
            continue
        if mode in pending_evaluator_builds:
            continue
        backlog.append(info)

    backlog.sort(key=lambda item: item["count"], reverse=True)
    return backlog


def build_exploration_evaluator(mode_info, findings, pending_evaluator_builds):
    mode = mode_info["search_mode"]
    description = mode_info["description"]
    parameters = mode_info["parameters"]
    count = mode_info["count"]

    params_example = json.dumps(parameters, indent=2)
    prompt = EXPLORATION_EVALUATOR_PROMPT.format(
        mode=mode,
        description=description,
        params_example=params_example,
        findings=findings,
    )

    briefs_dir = os.path.join(PROJECT_ROOT, "exploration", "briefs")
    os.makedirs(briefs_dir, exist_ok=True)
    log_path = os.path.join(briefs_dir, f"build_{mode}.log")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [CLAUDE_CMD, "--permission-mode", "bypassPermissions", "--print", prompt],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    pending_evaluator_builds[mode] = {"proc": proc, "log": log_path, "log_file": log_file}
    print(f"  🔨 [evaluator] Building scorer for '{mode}' (proposed {count}× without score)")


def poll_pending_evaluator_builds(pending_evaluator_builds):
    newly_built = []
    finished = []

    for mode, build_info in pending_evaluator_builds.items():
        proc = build_info["proc"]
        if proc.poll() is not None:
            build_info["log_file"].close()
            binary_path = os.path.join(
                PROJECT_ROOT, "exploration", mode, "target", "release", mode
            )
            if os.path.exists(binary_path):
                newly_built.append(mode)
                print(f"  ✅ Evaluator '{mode}' build complete — binary ready")
            else:
                print(f"  ❌ Evaluator '{mode}' build failed — no binary produced (see {build_info['log']})")
            finished.append(mode)

    for mode in finished:
        del pending_evaluator_builds[mode]

    return newly_built


def main():
    os.chdir(PROJECT_ROOT)

    history = load_history()
    modules = load_available_modules()

    # Geometry reset: if flag file exists, restart wavelet search from scratch
    reset_flag = os.path.join(PROJECT_ROOT, "autoresearch/wavelet_reset.flag")
    geometry_reset = os.path.exists(reset_flag)
    if geometry_reset:
        os.remove(reset_flag)
        print("🔄 Geometry reset: restarting wavelet θ search with new offset set")

    best_score = 0.0
    best_config = None
    for entry in history:
        if geometry_reset:
            break  # ignore all prior history for scoring purposes
        # Only wavelet_filter entries count toward the wavelet best
        if entry.get("config", {}).get("search_mode", "wavelet_filter") != "wavelet_filter":
            continue
        score = entry.get("composite_score", 0)
        if score > best_score:
            best_score = score
            best_config = entry.get("config")

    iteration = len(history)

    # Wavelet search is CLOSED — always start in exploration/explanation mode.
    # Infer stuck_count from history but floor at CONVERGENCE_THRESHOLD so we
    # never fall back to the wavelet pre-convergence path.
    stuck_count = CONVERGENCE_THRESHOLD
    if not geometry_reset:
        for entry in reversed(history):
            if entry.get("is_best", False):
                break
            stuck_count += 1

    # Count explanation-mode iterations already in history (for exploration probe offset)
    explanation_iteration = sum(1 for e in history if e.get("mode") == "explanation")
    print(f"Inferred stuck_count={stuck_count}, explanation_iteration={explanation_iteration} (converged={stuck_count >= CONVERGENCE_THRESHOLD})")

    pending_builds = {}
    last_findings_update = 0.0

    print(f"{'=' * 60}")
    print(f"🌊 Math Autoresearch Loop v2")
    print(f"{'=' * 60}")
    print(f"History: {len(history)} entries | Best score: {best_score:.4f}")
    print(f"Starting at iteration {iteration}")
    print(f"Available modules:")
    print(format_modules_summary(modules))
    print()

    # --- Backlog queue ---
    pending_evaluator_builds = {}
    # Seed last_proposal_per_mode from history so retroactive re-runs work across restarts
    last_proposal_per_mode = {}
    for _entry in history:
        _mode = _entry.get("config", {}).get("search_mode", "wavelet_filter")
        if _mode != "wavelet_filter":
            last_proposal_per_mode[_mode] = _entry.get("config", {})

    backlog = get_backlog_search_modes(history, pending_evaluator_builds)
    if backlog:
        print(f"📋 Evaluator backlog: {len(backlog)} search_modes with {EVALUATOR_BUILD_THRESHOLD}+ proposals, no binary yet:")
        for item in backlog[:10]:
            print(f"    [{item['count']}×] {item['search_mode']}")
            print(f"         {item['description'][:80]}")
        print()
        findings = load_findings()
        for item in backlog[:MAX_CONCURRENT_EVALUATOR_BUILDS]:
            build_exploration_evaluator(item, findings, pending_evaluator_builds)

    while True:
        try:
            newly_built = poll_pending_builds(pending_builds)
            trigger_findings_synthesis = False
            if newly_built:
                modules = load_available_modules()
                print(f"  🔄 Reloaded modules: {', '.join(newly_built)} now available")
                trigger_findings_synthesis = True

            newly_built_evaluators = poll_pending_evaluator_builds(pending_evaluator_builds)
            if newly_built_evaluators:
                print(f"  ✅ Evaluator(s) ready: {', '.join(newly_built_evaluators)}")
                # Retroactively score the last proposal for each newly-built evaluator
                for _mode in newly_built_evaluators:
                    if _mode not in last_proposal_per_mode:
                        print(f"  ℹ No cached proposal for '{_mode}' — will score on next proposal")
                        continue
                    _config = last_proposal_per_mode[_mode]
                    print(f"  🔁 Retroactive eval: '{_mode}'...")
                    try:
                        _result = run_rust_binary(_config)
                        _score = _result.get("composite_score", 0.0)
                        _entry = {
                            "iteration": iteration,
                            "config": _config,
                            "composite_score": _score,
                            "results": _result,
                            "note": "retroactive_eval",
                        }
                        with open(os.path.join(PROJECT_ROOT, HISTORY_FILE), "a") as _f:
                            _f.write(json.dumps(_entry) + "\n")
                        print(f"  📊 Retroactive score '{_mode}': {_score:.4f}")
                    except Exception as _e:
                        print(f"  ⚠ Retroactive eval failed for '{_mode}': {_e}")

            history = load_history()
            converged = stuck_count >= CONVERGENCE_THRESHOLD

            # Universal throttle: applies regardless of convergence state.
            # Don't pile up unevaluated proposals while evaluator builds are pending.
            scored_modes_univ = {
                e.get("config", {}).get("search_mode", "")
                for e in history[-40:]
                if e.get("composite_score", 0.0) != 0.0
            }
            unevaluated_univ = sum(
                1 for e in history[-20:]
                if e.get("composite_score", 0.0) == 0.0
                and e.get("note", "") != "stub"
                and e.get("config", {}).get("search_mode", "") not in scored_modes_univ
            )
            if unevaluated_univ >= 2 or (pending_evaluator_builds and unevaluated_univ >= 1):
                if not pending_evaluator_builds:
                    forced_backlog = get_backlog_search_modes(history, pending_evaluator_builds, min_count=1)
                    if not forced_backlog:
                        # Hard deadlock: stale binaries exist but score=0. Auto-delete them.
                        from collections import Counter as _CUniv
                        _zero_counts = _CUniv(
                            e.get("config", {}).get("search_mode", "")
                            for e in history[-30:]
                            if e.get("composite_score", 0.0) == 0.0
                            and e.get("note", "") != "stub"
                            and e.get("config", {}).get("search_mode", "") not in scored_modes_univ
                        )
                        for _mode, _cnt in _zero_counts.items():
                            if _cnt >= 2:
                                _bin = os.path.join(PROJECT_ROOT, "exploration", _mode, "target", "release", _mode)
                                if os.path.exists(_bin):
                                    os.remove(_bin)
                                    print(f"  🗑 Auto-removed stale binary '{_mode}' ({_cnt}× score=0) — will rebuild")
                        forced_backlog = get_backlog_search_modes(history, pending_evaluator_builds, min_count=1)
                    for item in forced_backlog:
                        if len(pending_evaluator_builds) < MAX_CONCURRENT_EVALUATOR_BUILDS:
                            build_exploration_evaluator(item, load_findings(), pending_evaluator_builds)
                print(f"  ⏸ {len(pending_evaluator_builds)} build(s) pending, {unevaluated_univ} unevaluated — waiting 15s")
                time.sleep(15)
                continue

            if converged:
                # Throttle: don't pile up unevaluated proposals while builds are pending.
                # Allow at most 1 unevaluated mode in flight; otherwise wait for builds.
                # Build a set of modes that have received a real score (including retroactive evals)
                # so old score=0.0 entries don't keep triggering the throttle after eval completes.
                scored_modes = {
                    e.get("config", {}).get("search_mode", "")
                    for e in history[-40:]
                    if e.get("composite_score", 0.0) != 0.0
                }
                unevaluated = sum(
                    1 for e in history[-20:]
                    if e.get("composite_score", 0.0) == 0.0
                    and e.get("config", {}).get("search_mode", "") not in ("wavelet_filter", "j16d_architecture_seed", "lm_head_gradient_bottleneck_seed")
                    and e.get("note", "") != "stub"
                    and e.get("config", {}).get("search_mode", "") not in scored_modes
                )
                if unevaluated >= 2 or (pending_evaluator_builds and unevaluated >= 1):
                    if not pending_evaluator_builds:
                        forced_backlog = get_backlog_search_modes(history, pending_evaluator_builds, min_count=1)
                        if not forced_backlog:
                            # Hard deadlock: binaries exist but return score=0 (stale schema).
                            # Auto-delete any binary that has appeared 2+ times with score=0
                            # and isn't in scored_modes — forces a fresh rebuild next cycle.
                            from collections import Counter as _Counter
                            zero_counts = _Counter(
                                e.get("config", {}).get("search_mode", "")
                                for e in history[-30:]
                                if e.get("composite_score", 0.0) == 0.0
                                and e.get("note", "") != "stub"
                                and e.get("config", {}).get("search_mode", "") not in scored_modes
                            )
                            for _mode, _cnt in zero_counts.items():
                                if _cnt >= 2:
                                    _bin = os.path.join(
                                        PROJECT_ROOT, "exploration", _mode,
                                        "target", "release", _mode)
                                    if os.path.exists(_bin):
                                        os.remove(_bin)
                                        print(f"  🗑 Auto-removed stale binary '{_mode}' ({_cnt}× score=0) — will rebuild")
                            # Retry backlog after clearing stale binaries
                            forced_backlog = get_backlog_search_modes(history, pending_evaluator_builds, min_count=1)
                        for item in forced_backlog:
                            if len(pending_evaluator_builds) < MAX_CONCURRENT_EVALUATOR_BUILDS:
                                build_exploration_evaluator(item, load_findings(), pending_evaluator_builds)
                    print(f"  ⏸ {len(pending_evaluator_builds)} build(s) pending, {unevaluated} unevaluated — waiting 15s")
                    time.sleep(15)
                    continue

                # Wavelet search done — always explore a new DSQG dimension.
                # If no evaluator exists yet, score=0 and gap detection builds one.
                explanation_iteration += 1
                print(f"[{iteration}] EXPLORATION MODE (expl_iter={explanation_iteration}) — new DSQG dimension")
                config = propose_next_config(
                    history, best_score, best_config, modules,
                    force_new_dimension=True
                )
                _mode = config.get("search_mode", "wavelet_filter")
                if _mode != "wavelet_filter":
                    last_proposal_per_mode[_mode] = config
                print(f"  search_mode={_mode} | {config.get('description','')[:65]}")

                # Trigger build immediately on new score=0, not just every 5 iters
                if True:
                    backlog = get_backlog_search_modes(history, pending_evaluator_builds)
                    for item in backlog:
                        if len(pending_evaluator_builds) < MAX_CONCURRENT_EVALUATOR_BUILDS:
                            build_exploration_evaluator(item, load_findings(), pending_evaluator_builds)
            else:
                print(f"[{iteration}] Proposing next config via LLM...")
                config = propose_next_config(history, best_score, best_config, modules)
                # Guard: if LLM returned an exploration config in wavelet mode, force retry
                if config.get("search_mode", "wavelet_filter") != "wavelet_filter":
                    print(f"  ⚠ LLM returned exploration config in wavelet mode — retrying with explicit instruction")
                    config = propose_next_config(
                        history, best_score, best_config, modules,
                        wavelet_override=True,
                    )
                taps = config["wavelet"]["filter_taps"]
                angles = config["wavelet"]["lattice_angles"]
                print(f"  Config: {config.get('description', '')[:70]}")
                print(f"  Taps={taps} Angles={angles}")

            result = run_rust_binary(config)
            score = result["composite_score"]

            extra_analysis = evaluate_extra_modules(config, modules)

            search_mode = config.get("search_mode", "wavelet_filter")
            extended_score, score_adjustments = compute_extended_score(score, extra_analysis)
            use_extended = search_mode != "wavelet_filter" or bool(score_adjustments)

            if not converged:
                comparison_score = extended_score if use_extended else score
                is_best = comparison_score > best_score
                if is_best:
                    best_score = comparison_score
                    best_config = config
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                is_best = False

            entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "results": result,
                "composite_score": score,
                "is_best": is_best,
                "extra_analysis": extra_analysis,
                "mode": "explanation" if converged else "search",
            }
            if use_extended:
                entry["extended_score"] = extended_score
                entry["score_adjustments"] = score_adjustments

            with open(os.path.join(PROJECT_ROOT, HISTORY_FILE), "a") as f:
                f.write(json.dumps(entry) + "\n")

            extra_names = [k for k in extra_analysis if not k.endswith("per_offset_errors")]
            extra_tag = f" +modules[{len(extra_names)}]" if extra_names else ""
            ext_tag = f" ext={extended_score:.4f}({sum(score_adjustments.values()):+.4f})" if score_adjustments else ""
            if converged:
                print(f"  score={score:.4f}{ext_tag} best={best_score:.4f}{extra_tag}")
            else:
                delta = result.get("vs_d4_delta", 0)
                marker = " *** BEST ***" if is_best else ""
                print(f"  score={score:.4f}{ext_tag} (vs D4: {delta:+.4f}) best={best_score:.4f} stuck={stuck_count}{marker}{extra_tag}")
            if score_adjustments:
                adj_parts = [f"{key}={value:+.4f}" for key, value in score_adjustments.items()]
                print(f"  📊 Extended score adjustments: {', '.join(adj_parts)}")
            print()

            # Gap detection: every iteration in explanation mode, every N in search mode
            should_check_gap = (
                (converged or iteration % GAP_CHECK_EVERY == 0 or stuck_count >= STUCK_THRESHOLD)
                and len(pending_builds) == 0
                and (converged or stuck_count > 0)
            )
            if should_check_gap:
                print(f"  🔍 Running gap detection (stuck={stuck_count}, iter={iteration})...")
                gap = detect_explanation_gap(
                    history, best_score, best_config, stuck_count, modules
                )
                if gap:
                    print(f"  💡 Gap detected: {gap['gap_description'][:80]}")
                    build_module(gap, pending_builds)

            should_update_findings = (
                trigger_findings_synthesis
                or iteration % FINDINGS_UPDATE_EVERY == 0
            ) and (time.time() - last_findings_update > 300)
            if should_update_findings and extra_analysis:
                synthesize_findings(extra_analysis, iteration)
                last_findings_update = time.time()

            time.sleep(EXPLANATION_SLEEP if converged else SLEEP_SECONDS)
            iteration += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
            for name, build_info in pending_builds.items():
                build_info["proc"].terminate()
                build_info["log_file"].close()
                print(f"  Terminated pending build: {name}")
            for mode, build_info in pending_evaluator_builds.items():
                build_info["proc"].terminate()
                build_info["log_file"].close()
                print(f"  Terminated pending evaluator build: {mode}")
            break
        except Exception as loop_error:
            print(f"  ⚠ Error: {loop_error}")
            traceback.print_exc()
            time.sleep(10)
            iteration += 1


if __name__ == "__main__":
    main()
