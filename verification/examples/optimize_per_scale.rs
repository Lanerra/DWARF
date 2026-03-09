//! Per-scale DWARF knob optimizer.
//!
//! Sweeps all analytically tractable knob combinations across 14M / 35M / 85M,
//! then prints a unified recommendation table with confidence levels.
//!
//! Usage:
//!   cd verification
//!   PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example optimize_per_scale

use std::{fs, time::Instant};
use wave_field_verification::scale_optimizer;

fn main() {
    let output_dir = "output";
    fs::create_dir_all(output_dir).expect("Failed to create output/");

    let t = Instant::now();
    scale_optimizer::run_all(Some(output_dir));
    println!("Completed in {:.3}s", t.elapsed().as_secs_f64());
}
