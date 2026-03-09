//! Offset space exploration example.
//!
//! Runs all three offset set sweeps (2D free, budget-constrained, dense-only ablation)
//! and saves JSON results to `verification/output/`.
//!
//! Usage:
//!   cd verification
//!   PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example explore_offsets
//!
//! Output files:
//!   output/offset_space_2d.json             — full 2D sweep (780 configs)
//!   output/offset_space_budget_44.json      — budget-constrained J=44 sweep

use std::fs;
use wave_field_verification::offset_space_explorer;

fn main() {
    let output_dir = "output";
    fs::create_dir_all(output_dir).expect("Failed to create output/ directory");

    println!("DWARF Offset Space Explorer");
    println!("===========================");
    println!("Output directory: {output_dir}/");
    println!();

    offset_space_explorer::run_all(Some(output_dir));

    println!("\nDone. Results saved to {output_dir}/");
}
