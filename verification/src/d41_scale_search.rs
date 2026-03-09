//! Optimal sparse set search for d41s3 at 35M scale.
//!
//! d41s3 used dense_w=41, sparse=[48,128,384] and proved optimal at 13M.
//! At 35M (D=512, vs 13M's D=256), the architecture has more capacity and
//! a wider effective dense window may be appropriate.
//!
//! This module searches for the optimal 3-sparse set at several candidate
//! dense widths (41, 48, 56, 64, 80, 96) and answers:
//! 1. Does d41s3's [48,128,384] remain optimal as dense_w increases?
//! 2. What is the best sparse set for each candidate 35M architecture?
//! 3. At what dense_w does a different sparse set beat [48,128,384]?

use crate::offset_optimizer::{path_counts, path_score, NUM_LAYERS, PASSKEY_DISTANCES};
use crate::offset_space_explorer::{build_offset_set, SPARSE_POOL};

const HOP_DISCOUNT: f64 = 0.75;

fn score_offset_set(offsets: &[usize]) -> (f64, usize) {
    let counts = path_counts(offsets, NUM_LAYERS);
    let mut coverage = 0.0f64;
    let mut reliable_depth = 0usize;
    for &d in PASSKEY_DISTANCES {
        let s = path_score(&counts, d, HOP_DISCOUNT);
        coverage += s;
        let total: u64 = (1..=NUM_LAYERS).map(|k| counts[k][d]).sum();
        if total > 0 { reliable_depth = reliable_depth.max(d); }
    }
    (coverage, reliable_depth)
}

fn top5_3sparse(dense_w: usize) -> Vec<(f64, usize, [usize; 3])> {
    let n = SPARSE_POOL.len();
    let mut results: Vec<(f64, usize, [usize; 3])> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                let sparse = [SPARSE_POOL[i], SPARSE_POOL[j], SPARSE_POOL[k]];
                let offsets = build_offset_set(dense_w, &sparse);
                let (score, depth) = score_offset_set(&offsets);
                results.push((score, depth, sparse));
            }
        }
    }
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    results.truncate(5);
    results
}

fn rank_d41s3_sparse(dense_w: usize) -> (f64, usize, usize) {
    // Returns (score, reliable_depth, rank/165)
    let target = [48usize, 128, 384];
    let offsets = build_offset_set(dense_w, &target);
    let (my_score, my_depth) = score_offset_set(&offsets);

    let n = SPARSE_POOL.len();
    let mut rank = 1usize;
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                if [SPARSE_POOL[i], SPARSE_POOL[j], SPARSE_POOL[k]] == target { continue; }
                let of = build_offset_set(dense_w, &[SPARSE_POOL[i], SPARSE_POOL[j], SPARSE_POOL[k]]);
                let (s, _) = score_offset_set(&of);
                if s > my_score { rank += 1; }
            }
        }
    }
    (my_score, my_depth, rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_sparse_by_dense_width() {
        println!("\n=== Optimal 3-sparse set at each candidate dense width ===");
        println!("  (path-count coverage score, top-5 per dense_w)");

        for &dense_w in &[41usize, 48, 56, 64, 80, 96] {
            let top = top5_3sparse(dense_w);
            println!("\n  dense_w={dense_w}:");
            println!("    {:>12}  {:>14}  sparse", "coverage", "reliable_depth");
            for (i, (score, depth, sparse)) in top.iter().enumerate() {
                let tag = if i == 0 { " ← best" } else { "" };
                println!("    {:>12.1}  {:>14}  {:?}{tag}", score, depth, sparse);
            }
        }
    }

    #[test]
    fn d41s3_sparse_rank_across_widths() {
        println!("\n=== d41s3 sparse [48,128,384] rank as dense_w increases ===");
        println!("  (does it remain #1 as the dense window grows?)");
        println!();
        println!("  {:>10}  {:>12}  {:>16}  {:>10}", "dense_w", "coverage", "reliable_depth", "rank/165");

        for &dense_w in &[41usize, 48, 56, 64, 80, 96] {
            let (score, depth, rank) = rank_d41s3_sparse(dense_w);
            let marker = if rank == 1 { " ✓ still #1" } else { &format!(" ← drops to #{rank}") };
            println!("  {:>10}  {:>12.1}  {:>16}  {:>7}/{}{marker}",
                     dense_w, score, depth, rank, 165);
        }
    }

    #[test]
    fn compare_d41s3_vs_wider_dense_tradeoff() {
        println!("\n=== Adding dense width vs keeping sparse=[48,128,384] ===");
        println!("  How much does widening the dense window buy vs the optimal sparse?");
        println!();

        // Baseline: d41s3 config (dense=41, sparse=[48,128,384], J=45)
        let base_offsets = build_offset_set(41, &[48, 128, 384]);
        let (base_score, base_depth) = score_offset_set(&base_offsets);
        println!("  Baseline d41s3 (dense=41, sparse=[48,128,384]): coverage={base_score:.1}, depth={base_depth}");
        println!();

        println!("  {:>10}  {:>10}  {:>10}  {:>16}  {:>12}",
                 "config", "dense_w", "J", "coverage", "vs_baseline%");
        for &dense_w in &[48usize, 56, 64, 80] {
            // Option A: just widen dense, keep same sparse
            let a = build_offset_set(dense_w, &[48, 128, 384]);
            let (sa, da) = score_offset_set(&a);
            let pct_a = (sa - base_score) / base_score * 100.0;
            println!("  {:>10}  {:>10}  {:>10}  {:>16.1}  {:>11.1}%  depth={da}",
                     "wider_dense", dense_w, a.len(), sa, pct_a);

            // Option B: widen dense + use best sparse for that width
            let top = top5_3sparse(dense_w);
            if let Some((sb, db, sparse_b)) = top.first() {
                let pct_b = (sb - base_score) / base_score * 100.0;
                println!("  {:>10}  {:>10}  {:>10}  {:>16.1}  {:>11.1}%  depth={db}  sparse={sparse_b:?}",
                         "best_combo", dense_w, build_offset_set(dense_w, sparse_b).len(), sb, pct_b);
            }
            println!();
        }
    }
}
