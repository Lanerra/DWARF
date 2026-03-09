//! Run the complete DWARF exploration sweep suite.
//!
//! Executes all six exploratory modules in sequence, saves JSON results to
//! `verification/output/`, and prints a summary of key findings.
//!
//! Usage:
//!   cd verification
//!   PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example run_all_sweeps
//!
//! Expected runtime: ~5–15 seconds (all Rayon-parallel, mostly CPU-microseconds)
//!
//! Output files:
//!   output/offset_space_2d.json            — 2D offset set sweep (768 configs)
//!   output/offset_space_budget_44.json     — budget-constrained J=44 sweep
//!   output/layer_placement_single.json     — single-injection layer placement
//!   output/scale_embed_dynamics.json       — phase transition prediction (condX-v2 35M)
//!   output/gate_equilibrium.json           — bypass gate scaling law predictions
//!   output/knob_interactions.json          — 9×9 knob coupling matrix

use std::{fs, time::Instant};
use wave_field_verification::{
    offset_space_explorer,
    layer_placement_explorer,
    scale_embed_dynamics,
    gate_equilibrium,
    knob_interactions,
};

fn main() {
    let output_dir = "output";
    fs::create_dir_all(output_dir).expect("Failed to create output/ directory");

    let total_start = Instant::now();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       DWARF Verification Sweep Suite — Full Run          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Output directory: {output_dir}/");
    println!();

    // ── 1. Offset Space Explorer ─────────────────────────────────────────────
    let t = Instant::now();
    println!("━━━ [1/5] Offset Space Explorer ━━━━━━━━━━━━━━━━━━━━━━━━━━");
    offset_space_explorer::run_all(Some(output_dir));
    println!("  Completed in {:.2}s\n", t.elapsed().as_secs_f64());

    // ── 2. Layer Placement Explorer ──────────────────────────────────────────
    let t = Instant::now();
    println!("━━━ [2/5] Layer Placement Explorer ━━━━━━━━━━━━━━━━━━━━━━━");
    layer_placement_explorer::run_all(Some(output_dir));
    println!("  Completed in {:.2}s\n", t.elapsed().as_secs_f64());

    // ── 3. Scale_embed Dynamics ──────────────────────────────────────────────
    let t = Instant::now();
    println!("━━━ [3/5] Scale_embed Phase Transition Dynamics ━━━━━━━━━━");
    scale_embed_dynamics::run_all(Some(output_dir));
    println!("  Completed in {:.2}s\n", t.elapsed().as_secs_f64());

    // ── 4. Gate Equilibrium Predictor ────────────────────────────────────────
    let t = Instant::now();
    println!("━━━ [4/5] Bypass Gate Equilibrium Predictor ━━━━━━━━━━━━━");
    gate_equilibrium::run_all(Some(output_dir));
    println!("  Completed in {:.2}s\n", t.elapsed().as_secs_f64());

    // ── 5. Knob Interaction Matrix ───────────────────────────────────────────
    let t = Instant::now();
    println!("━━━ [5/5] Knob Interaction Matrix ━━━━━━━━━━━━━━━━━━━━━━━");
    knob_interactions::run_all(Some(output_dir));
    println!("  Completed in {:.2}s\n", t.elapsed().as_secs_f64());

    // ── Summary ──────────────────────────────────────────────────────────────
    let total = total_start.elapsed().as_secs_f64();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  All sweeps complete in {total:.2}s                          ");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Output files:");
    for entry in fs::read_dir(output_dir).unwrap() {
        let e = entry.unwrap();
        let meta = e.metadata().unwrap();
        println!("  {:>8} bytes  {}", meta.len(), e.file_name().to_string_lossy());
    }
}
