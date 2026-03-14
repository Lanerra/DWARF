"""
math_summary.py — DWARF autoresearch loop summary (modern)

Shows: mode, top-5 synthesis signals, best wavelet config, exploration scores,
module registry. No more walls of raw metrics.

Usage:
    python3 autoresearch/math_summary.py
    python3 autoresearch/math_summary.py --metrics   # also dump raw module metrics
"""
import json
import os
import sys
import argparse

PROJECT_ROOT = "/home/dlewis3/Desktop/AI/DWARF"
HISTORY_FILE = "autoresearch/math_history.jsonl"
MODULES_DIR  = "autoresearch/modules"
CONVERGENCE_THRESHOLD = 15

# ── Priority rules for the synthesis layer ────────────────────────────────────
# Each rule: (module_prefix, metric_name, importance, interpretation_fn)
# importance: 1=critical, 2=high, 3=medium
# interpretation_fn: (value, all_module_metrics) → str
SIGNAL_RULES = [
    # Two-θ / regime decoupling
    ("regime_decoupled_wavelet_potential", "regime_scale_decoupling_ratio", 1,
     lambda v, m: f"+{(v-1):.1%} gain from two-θ design (local={m.get('optimal_theta_local','?'):.2f}, distal={m.get('optimal_theta_distal','?'):.2f})  → d50 candidate"),
    # Head specialization
    ("head_regime_specialization_asymmetry", "head_distal_specialist_count", 1,
     lambda v, m: (f"Distal specialists={v:.0f} (training never grows one — hard enforcement critical)"
                   if v < 0.5 else f"Distal specialists={v:.0f} (self-organising ✓)") +
                  f"  local={m.get('head_local_specialist_count','?'):.0f}"),
    # Phase discontinuity at regime boundary
    ("wavelet_zone_phase_continuity", "zone_boundary_phase_discontinuity", 2,
     lambda v, m: f"Boundary discontinuity={v:.1f} (bridges reduce gradient, NOT phase relationship)"),
    # Phase bridge capacity vs. actual continuity gap
    ("tap_zone_phase_bridge", "tap_phase_gradient_continuity_score", 2,
     lambda v, m: (f"tap_phase_gradient_continuity={v:.4f} — " +
                   ("ACHIEVED ✓" if v > 0.5 else
                    f"ZERO despite phase_bridge_capacity={m.get('cross_zone_phase_coherence_potential','?'):.4f}"))),
    # Bridge offset validity
    ("offset_regime_bridge_phase_stability", "cross_regime_offset_bridge_set", 2,
     lambda v, m: (f"Valid bridge offsets in current set: {v:.0f} " +
                   ("✓" if v > 0 else "→ structural impossibility without two-θ"))),
    # Frequency coverage gap (stale check)
    ("zone_frame_completeness_analysis", "interzone_frequency_coverage_gap", 2,
     lambda v, m: f"Interzone freq gap={v:.0f} " +
                  ("(bridge offsets in geometry but module uses old offsets — stale)" if v > 100 else "✓ covered")),
    # Dense zone saturation
    ("dense_zone_saturation_diagnosis", "predicted_under_covered_scales", 3,
     lambda v, m: f"Predicted under-covered scales={v:.0f} (coverage_hole_r²={m.get('coverage_hole_predictability_r_squared','?'):.3f})"),
    # Boundary gradient mismatch
    ("phase_transfer_boundary_conductance", "boundary_gradient_mismatch", 2,
     lambda v, m: f"Boundary gradient mismatch={v:.2f} " +
                  ("(below 1.5 ✓)" if v < 1.5 else f"(target 1.0 → {v:.1f}× too high)")),
    # Interference layer placement
    ("interference_pooling_layer_capacity", "optimal_pooling_layer", 3,
     lambda v, m: f"Optimal interference pooling at layer {v:.0f} (current: every 3rd — pure DSQG N/A)"),
    # Zone spectral retention
    ("zone_spectral_leakage", "zone_frequency_utilization_efficiency", 3,
     lambda v, m: f"Frequency utilisation={v:.4f} (leakage_ratio={m.get('leakage_to_coverage_ratio','?'):.4f})"),
]


def load_history():
    path = os.path.join(PROJECT_ROOT, HISTORY_FILE)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def load_modules():
    modules = {}
    modules_path = os.path.join(PROJECT_ROOT, MODULES_DIR)
    if not os.path.isdir(modules_path):
        return modules
    for name in sorted(os.listdir(modules_path)):
        module_dir = os.path.join(modules_path, name)
        if not os.path.isdir(module_dir):
            continue
        status_path = os.path.join(module_dir, "status.json")
        binary_path = os.path.join(PROJECT_ROOT, "exploration", name, "target", "release", name)
        status = "unknown"
        if os.path.exists(status_path):
            try:
                status = json.load(open(status_path)).get("status", "unknown")
            except Exception:
                pass
        modules[name] = {
            "status": status,
            "binary": os.path.exists(binary_path),
            "ready": os.path.exists(binary_path) and status != "failed",
        }
    return modules


def flatten_extra_analysis(ea):
    """Flatten extra_analysis dict into (module, metric, value) triples."""
    flat = {}  # module -> {metric: value}
    for k, v in ea.items():
        if "per_offset_errors" in k or isinstance(v, list):
            continue
        mod, _, metric = k.partition(".")
        flat.setdefault(mod, {})[metric] = v
    return flat


def detect_signals(extra_analysis, top_n=5):
    """Apply SIGNAL_RULES + generic interest scoring. Returns top_n signal strings."""
    by_module = flatten_extra_analysis(extra_analysis)
    signals = []  # (importance, module, text)

    # Apply named rules first
    for (mod_prefix, metric, importance, interp_fn) in SIGNAL_RULES:
        mod_data = by_module.get(mod_prefix, {})
        if metric not in mod_data:
            continue
        val = mod_data[metric]
        if not isinstance(val, (int, float)):
            continue
        try:
            text = interp_fn(float(val), {k: v for k, v in mod_data.items()
                                          if isinstance(v, (int, float))})
            signals.append((importance, mod_prefix, metric, text))
        except Exception:
            pass

    # Generic: find extreme/interesting values not already covered
    covered = {(r[0], r[1]) for r in signals}
    for mod_name, metrics in by_module.items():
        for mname, val in metrics.items():
            if (mod_name, mname) in covered:
                continue
            if not isinstance(val, (int, float)):
                continue
            score = 0
            # Near 0 or 1 (boundary values are interesting)
            if abs(val) < 1e-4:
                score += 3
            elif val > 0.999:
                score += 2
            # Keywords in metric name
            for kw in ("potential", "gain", "optimal", "bottleneck", "critical", "recovery"):
                if kw in mname:
                    score += 2
            # Very large magnitude
            if abs(val) > 1000:
                score += 1
            if score >= 3:
                signals.append((3, mod_name, mname, f"{mname}={val:.4g}"))

    # Sort by importance (lower = higher priority), deduplicate module
    signals.sort(key=lambda x: (x[0], x[1]))
    seen_mods = set()
    results = []
    for imp, mod, metric, text in signals:
        if len(results) >= top_n:
            break
        if mod in seen_mods:
            continue
        seen_mods.add(mod)
        icon = "🔑" if imp == 1 else ("⚠️ " if imp == 2 else "📐")
        results.append(f"  {icon} [{mod}]\n      {text}")
    return results


def fmt_metric(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="store_true", help="Also dump raw module metrics")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    history = load_history()
    modules = load_modules()

    if not history:
        print("No history found.")
        return

    # ── Categorise entries ────────────────────────────────────────────────
    wavelet_entries = [e for e in history
                       if e.get("config", {}).get("search_mode", "wavelet_filter") == "wavelet_filter"
                       and e.get("note", "") not in ("convergence_transition", "d49_sweep", "d49_sweep_fine")]
    sweep_entries   = [e for e in history
                       if e.get("note", "").startswith("d49_sweep")]
    expl_entries    = [e for e in history
                       if e.get("config", {}).get("search_mode", "wavelet_filter") != "wavelet_filter"
                       and e.get("note", "") not in ("convergence_transition",)]

    stuck_count = 0
    for e in reversed(history):
        if e.get("is_best", False):
            break
        stuck_count += 1
    converged = stuck_count >= CONVERGENCE_THRESHOLD

    # ── Best wavelet config (wavelet entries only) ────────────────────────
    all_scored_wavelet = [e for e in wavelet_entries + sweep_entries if e.get("composite_score", 0) > 0]
    best_wav = max(all_scored_wavelet, key=lambda e: e.get("composite_score", 0)) if all_scored_wavelet else None
    best_wav_score = best_wav.get("composite_score", 0) if best_wav else 0

    # ── Best exploration config ───────────────────────────────────────────
    expl_scored = [e for e in expl_entries if e.get("composite_score", 0) > 0]
    best_expl   = max(expl_scored, key=lambda e: e.get("composite_score", 0)) if expl_scored else None

    print(f"\n🌊 DWARF Math Autoresearch Summary")
    print(f"{'='*70}")
    iter_total = len(history)
    print(f"Iterations: {iter_total} total  |  wavelet: {len(wavelet_entries)+len(sweep_entries)}  |  exploration: {len(expl_entries)}")
    mode_str = "🔬 EXPLORATION (wavelet converged)" if converged else "🔍 WAVELET SEARCH"
    print(f"Mode: {mode_str}  |  stuck_count: {stuck_count}/{CONVERGENCE_THRESHOLD}")
    print()

    # ── TOP 5 SIGNALS ─────────────────────────────────────────────────────
    rich_entries = [e for e in history if e.get("extra_analysis")]
    if rich_entries:
        latest_rich = rich_entries[-1]
        ea = latest_rich.get("extra_analysis", {})
        signals = detect_signals(ea, top_n=5)
        if signals:
            print(f"🎯 TOP 5 SYNTHESIS SIGNALS  (iter {latest_rich.get('iteration','?')}, θ={latest_rich.get('config',{}).get('wavelet',{}).get('lattice_angles',[])})")
            print(f"{'-'*70}")
            for s in signals:
                print(s)
            print()

    # ── Best wavelet config ───────────────────────────────────────────────
    if best_wav:
        wav  = best_wav.get("config", {}).get("wavelet", {})
        note = best_wav.get("note", "")
        note_tag = f"  [{note}]" if note else ""
        print(f"🏆 Best wavelet  score={best_wav_score:.4f}{note_tag}")
        print(f"   θ={wav.get('lattice_angles',[])}  taps={wav.get('filter_taps','?')}  block={wav.get('block_structure','?')}")
        coeffs = best_wav.get("results", {}).get("filter_coefficients", [])
        if coeffs:
            print(f"   coefficients: {[round(c, 4) for c in coeffs]}")
        core_keys = ["sparse_approx_mse", "scale_separation", "leakage"]
        bm = best_wav.get("results", {}).get("metrics", best_wav.get("results", {}))
        shown = {k: bm[k] for k in core_keys if k in bm}
        if shown:
            print("   " + "  ".join(f"{k}={fmt_metric(v)}" for k, v in shown.items()))
        print()

    # ── Best exploration config ───────────────────────────────────────────
    if best_expl:
        ec   = best_expl.get("config", {})
        mode = ec.get("search_mode", "?")
        params = json.dumps(ec.get("parameters", {}))[:80]
        desc  = ec.get("description", "")[:70]
        print(f"🔬 Best exploration  score={best_expl.get('composite_score',0):.4f}  mode={mode}")
        print(f"   params: {params}")
        print(f"   desc:   {desc}")
        print()

    # ── Improvement timeline ──────────────────────────────────────────────
    improvements = [e for e in wavelet_entries + sweep_entries if e.get("is_best", False)]
    if improvements:
        print(f"📈 Wavelet improvement timeline ({len(improvements)} steps):")
        for e in improvements[-8:]:  # last 8
            wav = e.get("config", {}).get("wavelet", {})
            ang = wav.get("lattice_angles", [])
            note = f" [{e['note']}]" if e.get("note") else ""
            print(f"   [{e.get('iteration','?'):>4}] {e.get('composite_score',0):.4f}  θ={ang}{note}")
        print()

    # ── Top exploration scores ────────────────────────────────────────────
    if expl_scored:
        top_expl = sorted(expl_scored, key=lambda e: e.get("composite_score", 0), reverse=True)[:8]
        print(f"🔬 Top exploration configs ({len(expl_scored)} scored):")
        print(f"  {'Score':<8} {'Mode':<45} {'Desc':<35}")
        print(f"  {'-'*8} {'-'*45} {'-'*35}")
        for e in top_expl:
            ec   = e.get("config", {})
            mode = ec.get("search_mode", "?")[:44]
            desc = ec.get("description", "")[:34]
            print(f"  {e.get('composite_score',0):<8.4f} {mode:<45} {desc:<35}")
        print()

    # ── Module registry ───────────────────────────────────────────────────
    ready    = [n for n, i in modules.items() if i["ready"]]
    building = [n for n, i in modules.items() if not i["ready"] and i["status"] == "building"]
    broken   = [n for n, i in modules.items() if not i["ready"] and i["status"] not in ("building",)]
    print(f"🔧 Modules: {len(ready)} ready, {len(building)} building, {len(broken)} broken")
    if building:
        print(f"   building: {', '.join(building)}")
    if broken:
        print(f"   broken:   {', '.join(broken)}")
    print()

    # ── Raw module metrics (opt-in) ────────────────────────────────────────
    if args.metrics and rich_entries:
        latest_rich = rich_entries[-1]
        ea = latest_rich.get("extra_analysis", {})
        by_module = flatten_extra_analysis(ea)
        print(f"📊 Raw module metrics (iter {latest_rich.get('iteration','?')}):")
        for mod_name, metrics in sorted(by_module.items()):
            print(f"  [{mod_name}]")
            for metric, val in sorted(metrics.items()):
                print(f"    {metric}: {fmt_metric(val)}")
        print()


if __name__ == "__main__":
    main()
