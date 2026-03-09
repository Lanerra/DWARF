//! Find the optimal sparse tier selection for a given dense window width.
//!
//! Sweeps all C(|POOL|, k) combinations of sparse tiers and ranks them by
//! coverage score. Useful for planning ablation experiments with constrained
//! offset budgets.
//!
//! Usage:
//!   cd verification
//!   PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example best_sparse_tiers

use wave_field_verification::offset_space_explorer::{
    build_offset_set, compute_metrics, offsets_condu, SPARSE_POOL,
};
use wave_field_verification::offset_optimizer::PASSKEY_DISTANCES;

/// Enumerate all k-combinations of indices 0..n (no repetition, lexicographic order).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if k > n  { return vec![]; }
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    // Initialize to first combination
    for i in 0..k { combo[i] = i; }
    loop {
        result.push(combo.clone());
        // Find rightmost element that can be incremented
        let mut i = k as isize - 1;
        while i >= 0 && combo[i as usize] == n - k + i as usize {
            i -= 1;
        }
        if i < 0 { break; }
        combo[i as usize] += 1;
        for j in (i + 1) as usize..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
    result
}

fn main() {
    let pool = SPARSE_POOL;

    // --- Target configurations ---
    let test_cases: &[(usize, usize, &str)] = &[
        (41, 3, "dense=41 + 3 sparse (J=44)"),
        (41, 4, "dense=41 + 4 sparse (J=45)"),
        (38, 6, "dense=38 + 6 sparse (J=44)"),
        (35, 8, "dense=35 + 8 sparse (J=43)"),
        (32, 11, "condU: dense=32 + all sparse (J=43)"),
    ];

    // Reference: condU
    let condu = offsets_condu();
    let condu_m = compute_metrics(&condu);
    println!("condU reference: J={}, coverage={:.0}, reliable_depth={}",
        condu.len(), condu_m.coverage_score, condu_m.reliable_retrieval_depth);
    println!();

    for &(dense_w, k, label) in test_cases {
        println!("=== {} ===", label);
        let n = pool.len();

        let combos = combinations(n, k);
        println!("  Evaluating {} combinations...", combos.len());

        let mut results: Vec<(Vec<usize>, f64, usize, bool)> = combos
            .iter()
            .map(|indices| {
                let tiers: Vec<usize> = indices.iter().map(|&i| pool[i]).collect();
                let offsets = build_offset_set(dense_w, &tiers);
                let m = compute_metrics(&offsets);
                let reaches_1536 = m.reliable_retrieval_depth >= 1536;
                (tiers, m.coverage_score, m.reliable_retrieval_depth, reaches_1536)
            })
            .collect();

        // Sort by coverage score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Show top-5 overall
        println!("  Top-5 by coverage score:");
        println!("  {:<35} {:>12}  {:>14}  {:>12}",
            "sparse tiers", "coverage", "reliable_depth", "covers_1536");
        println!("  {}", "-".repeat(80));
        for (tiers, score, depth, covers) in results.iter().take(5) {
            println!("  {:<35} {:>12.0}  {:>14}  {:>12}",
                format!("{:?}", tiers), score, depth,
                if *covers { "YES" } else { "NO" });
        }

        // Show top-5 that reliably reach d=1536
        let reaching_1536: Vec<_> = results.iter().filter(|(_, _, _, c)| *c).collect();
        println!("\n  Top-5 that reach d=1536 reliably:");
        println!("  {:<35} {:>12}  {:>14}",
            "sparse tiers", "coverage", "reliable_depth");
        println!("  {}", "-".repeat(65));
        for (tiers, score, depth, _) in reaching_1536.iter().take(5) {
            println!("  {:<35} {:>12.0}  {:>14}",
                format!("{:?}", tiers), score, depth);
        }

        // Show per-distance breakdown for the best 1536-reaching config
        if let Some((best_tiers, best_score, _, _)) = reaching_1536.first() {
            println!("\n  Best 1536-reaching config: {:?}", best_tiers);
            let offsets = build_offset_set(dense_w, best_tiers);
            let m = compute_metrics(&offsets);
            println!("  Total offsets: {}", offsets.len());
            println!("  Coverage score: {:.0} ({:.1}× condU)",
                best_score, best_score / condu_m.coverage_score);
            println!("  Path counts per distance:");
            for (&d, &paths) in PASSKEY_DISTANCES.iter().zip(m.paths_by_distance.iter()) {
                let condu_paths = condu_m.paths_by_distance[
                    PASSKEY_DISTANCES.iter().position(|&x| x == d).unwrap()];
                let ratio = if condu_paths > 0 { paths as f64 / condu_paths as f64 } else { 0.0 };
                println!("    d={:<6} {:>12} paths  ({:.2}× condU, {} paths)",
                    d, paths, ratio, condu_paths);
            }
        }

        // Count how many combinations reach d=1536
        println!("\n  {}/{} combinations reach reliable d=1536",
            reaching_1536.len(), combos.len());
        println!();
    }
}
