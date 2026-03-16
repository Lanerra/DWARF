//! Frobenius-Optimal Offset Set Search for DWARF Relay Chains
//!
//! Deep Research finding (March 15, 2026):
//!   No prior work connects the Frobenius coin problem (Chicken McNugget theorem)
//!   to attention mechanism design. DWARF's coprime relay chains are the first
//!   explicit application. This tool formalizes the coverage theory and searches
//!   for offset sets with J < 24 that achieve the same relay chain coverage.
//!
//! ## Mathematical Background
//!
//! Given a set S of positive integers (offsets), the **numerical semigroup** N(S)
//! is the set of all non-negative integers representable as Σ aᵢ·δᵢ with aᵢ ≥ 0.
//! The **Frobenius number** g(S) is the largest integer NOT in N(S).
//!
//! For two coprime integers (p, q): g(p,q) = p·q - p - q (Sylvester-Frobenius theorem)
//!   e.g. g(3,4) = 12 - 3 - 4 = 5  →  every d > 5 reachable via 3a + 4b
//!
//! For DWARF's relay chains, we need coverage of d = 1..MAX_D within max_hops hops.
//! "Coverage" here is multi-hop: we can chain hops across layers.
//!
//! ## Coverage Criterion
//!
//! A distance d is **k-hop reachable** from offset set S if:
//!   d = Σᵢ δᵢ   where each δᵢ ∈ S  and  k = number of hops
//!
//! With L=6 layers in DWARF and max_hops=2 per layer, effective max hops = 12.
//! In practice, with proper offsets, most distances are 2-hop reachable.
//!
//! ## Search Strategy
//!
//! 1. Start from the J=24 relay-optimal set as the known good baseline
//! 2. For each J from 8..23, search for the minimum set achieving:
//!    - Full coverage of d=1..1536 within max_hops=2
//!    - Redundancy score ≥ threshold (robustness, not just coverage)
//!    - Coprime pair count ≥ 2 (multiple coprime pairs for distributed redundancy)
//! 3. Score each candidate by:
//!    - Primary: redundancy_score = Σ_d log(1 + paths_to_d)
//!    - Secondary: critical_offset_risk = max single-offset removal damage
//!    - Tertiary: J (smaller is better, given equal coverage)
//!
//! ## Key Insight from Deep Research
//!
//! For 3+ generators, Kannan's algorithm computes the Frobenius number via
//! shortest-path in a directed graph. With generators {3, 4, k} for any k,
//! the Frobenius number is at most 5 (already covered by {3,4} alone).
//! Adding a third generator from {5,7,11,13...} adds redundancy without
//! changing coverage boundary — it creates alternative relay paths.
//!
//! The optimal search therefore minimizes J by finding the MINIMUM set of
//! offsets whose 2-hop closure covers d=1..1536 densely.

use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;

// ─── Constants ────────────────────────────────────────────────────────────────
const MAX_D: usize = 1536;
const MAX_HOPS: usize = 2;      // max relay hops (from Rust analyze_j.py finding)
const MIN_PATHS: u32 = 1;       // minimum paths required for coverage
const MIN_REDUNDANCY_PATHS: u32 = 2; // for "robust" coverage
const PASSKEY_DISTANCES: &[usize] = &[
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536
];

// J=24 baseline (relay-optimal, current production set)
const J24D_OFFSETS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
    48, 64, 96, 192, 384, 512, 768, 1024
];

// ─── Coverage analysis ────────────────────────────────────────────────────────

/// Count the number of distinct k-hop relay paths reaching distance d.
/// A k-hop path is a sequence (δ₁, δ₂, ..., δₖ) where each δᵢ ∈ offsets
/// and Σ δᵢ = d.
fn count_paths(offsets: &[usize], max_hops: usize, max_d: usize) -> Vec<u32> {
    // dp[d] = number of distinct paths reaching distance d within max_hops
    let mut dp = vec![0u32; max_d + 1];
    dp[0] = 1; // base: zero hops, distance 0

    for _hop in 0..max_hops {
        let mut new_dp = vec![0u32; max_d + 1];
        for d in 0..=max_d {
            if dp[d] == 0 { continue; }
            for &off in offsets {
                let nd = d + off;
                if nd <= max_d {
                    new_dp[nd] = new_dp[nd].saturating_add(dp[d]);
                }
            }
        }
        // accumulate: dp[d] = paths reachable in exactly 1..k hops
        for d in 1..=max_d {
            dp[d] = dp[d].saturating_add(new_dp[d]);
        }
        // Next hop starts from positions reachable in previous hops
        for d in 0..=max_d {
            if new_dp[d] > 0 { dp[d] = dp[d].max(new_dp[d]); }
        }
    }

    // Recompute correctly: dp[d] = total paths reaching d in 1..max_hops hops
    let mut result = vec![0u32; max_d + 1];
    let mut current = vec![0u32; max_d + 1];
    current[0] = 1;

    for _hop in 0..max_hops {
        let mut next = vec![0u32; max_d + 1];
        for d in 0..=max_d {
            if current[d] == 0 { continue; }
            for &off in offsets {
                let nd = d + off;
                if nd <= max_d {
                    next[nd] = next[nd].saturating_add(current[d]);
                    result[nd] = result[nd].saturating_add(current[d]);
                }
            }
        }
        current = next;
    }
    result
}

/// Coverage metrics for an offset set.
#[derive(Debug, Clone)]
struct CoverageMetrics {
    /// Number of passkey distances covered (paths ≥ 1)
    passkey_coverage: usize,
    /// Number of passkey distances with redundant paths (paths ≥ 2)
    passkey_redundancy: usize,
    /// Total paths across all d=1..MAX_D (log-summed)
    redundancy_score: f64,
    /// Minimum paths to any passkey distance (weakest link)
    min_passkey_paths: u32,
    /// Number of coprime pairs in the offset set
    coprime_pairs: usize,
    /// Frobenius number (largest d NOT covered in 1..MAX_D)
    frobenius_number: usize,
}

fn compute_coverage(offsets: &[usize]) -> CoverageMetrics {
    let paths = count_paths(offsets, MAX_HOPS, MAX_D);

    let passkey_coverage = PASSKEY_DISTANCES.iter()
        .filter(|&&d| paths[d] >= MIN_PATHS)
        .count();

    let passkey_redundancy = PASSKEY_DISTANCES.iter()
        .filter(|&&d| paths[d] >= MIN_REDUNDANCY_PATHS)
        .count();

    let redundancy_score: f64 = paths[1..].iter()
        .map(|&p| (1.0 + p as f64).ln())
        .sum();

    let min_passkey_paths = PASSKEY_DISTANCES.iter()
        .map(|&d| paths[d])
        .min()
        .unwrap_or(0);

    // Count coprime pairs (gcd == 1)
    let mut coprime_pairs = 0;
    for i in 0..offsets.len() {
        for j in (i+1)..offsets.len() {
            if gcd(offsets[i], offsets[j]) == 1 {
                coprime_pairs += 1;
            }
        }
    }

    // Frobenius number: largest d ≤ MAX_D with zero paths
    let frobenius_number = paths[1..=MAX_D].iter().enumerate()
        .rev()
        .find(|(_, &p)| p == 0)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    CoverageMetrics {
        passkey_coverage,
        passkey_redundancy,
        redundancy_score,
        min_passkey_paths,
        coprime_pairs,
        frobenius_number,
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Composite score for comparing offset sets (higher = better).
/// Primary: full passkey coverage. Secondary: redundancy. Tertiary: J (fewer = better).
fn score(metrics: &CoverageMetrics, j: usize) -> f64 {
    let coverage_term = metrics.passkey_coverage as f64 * 1000.0;
    let redundancy_term = metrics.passkey_redundancy as f64 * 100.0;
    let path_term = metrics.redundancy_score;
    let j_penalty = j as f64 * 10.0; // prefer fewer offsets
    let frobenius_penalty = metrics.frobenius_number as f64 * 5.0;
    coverage_term + redundancy_term + path_term - j_penalty - frobenius_penalty
}

// ─── Greedy search ────────────────────────────────────────────────────────────

/// Build a minimum-J offset set by greedy addition.
/// Start with {1} (mandatory), greedily add the offset that most improves score
/// until full passkey coverage is achieved with redundancy.
fn greedy_build(max_j: usize, candidate_pool: &[usize]) -> (Vec<usize>, CoverageMetrics) {
    let mut current: Vec<usize> = vec![1]; // δ=1 is always mandatory
    let mut best_metrics = compute_coverage(&current);

    while current.len() < max_j {
        if best_metrics.passkey_coverage == PASSKEY_DISTANCES.len()
            && best_metrics.passkey_redundancy == PASSKEY_DISTANCES.len() {
            break; // Full redundant coverage achieved
        }

        // Find the offset to add that gives the best score improvement
        let best_addition = candidate_pool.iter()
            .filter(|&&off| !current.contains(&off))
            .map(|&off| {
                let mut trial = current.clone();
                trial.push(off);
                trial.sort();
                let m = compute_coverage(&trial);
                let s = score(&m, trial.len());
                (off, s, m)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match best_addition {
            Some((off, _, metrics)) => {
                current.push(off);
                current.sort();
                best_metrics = metrics;
            }
            None => break,
        }
    }

    (current, best_metrics)
}

/// Refinement: given a set, try removing each offset and check if coverage holds.
/// Returns the minimal subset that maintains full passkey coverage.
fn minimize(offsets: &[usize]) -> (Vec<usize>, CoverageMetrics) {
    let mut current = offsets.to_vec();
    loop {
        let mut improved = false;
        for i in 0..current.len() {
            if current[i] == 1 { continue; } // never remove δ=1
            let mut trial: Vec<usize> = current.iter().enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();
            trial.sort();
            let m = compute_coverage(&trial);
            if m.passkey_coverage == PASSKEY_DISTANCES.len()
                && m.passkey_redundancy >= PASSKEY_DISTANCES.len() - 2 {
                current = trial;
                improved = true;
                break;
            }
        }
        if !improved { break; }
    }
    let m = compute_coverage(&current);
    (current, m)
}

// ─── Main analysis ────────────────────────────────────────────────────────────

pub fn run() {
    println!("================================================================");
    println!("  Frobenius-Optimal Offset Search for DWARF Relay Chains");
    println!("  Target: J < 24, full passkey coverage d=1..1536, max_hops={}", MAX_HOPS);
    println!("================================================================\n");

    // 1. Baseline: J=24 production set
    println!("── Baseline: J=24 production set ──────────────────────────────");
    let j24_metrics = compute_coverage(J24D_OFFSETS);
    println!("  Offsets: {:?}", J24D_OFFSETS);
    println!("  J = {}", J24D_OFFSETS.len());
    println!("  Passkey coverage:   {}/{}", j24_metrics.passkey_coverage, PASSKEY_DISTANCES.len());
    println!("  Passkey redundancy: {}/{}", j24_metrics.passkey_redundancy, PASSKEY_DISTANCES.len());
    println!("  Redundancy score:   {:.2}", j24_metrics.redundancy_score);
    println!("  Min passkey paths:  {}", j24_metrics.min_passkey_paths);
    println!("  Coprime pairs:      {}", j24_metrics.coprime_pairs);
    println!("  Frobenius number:   {}", j24_metrics.frobenius_number);
    println!("  Score:              {:.2}", score(&j24_metrics, J24D_OFFSETS.len()));
    println!();

    // 2. Candidate pool: integers 1..1536 that are "interesting"
    //    Must include: small integers 1..28 (local density)
    //                  long-range anchors at geometric/dyadic positions
    //                  key coprime-generating integers
    let mut candidate_pool: Vec<usize> = (1..=28).collect();
    // Dyadic anchors
    for &x in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] {
        candidate_pool.push(x);
    }
    // Additional candidates that might improve Frobenius coverage
    for &x in &[11usize, 12, 17, 19, 20, 31, 33, 36, 40, 43, 47, 50] {
        candidate_pool.push(x);
    }
    candidate_pool.sort();
    candidate_pool.dedup();

    // 3. Greedy search for J < 24
    println!("── Greedy search: minimum J for full passkey coverage ──────────");
    let (greedy_offsets, greedy_metrics) = greedy_build(23, &candidate_pool);
    println!("  Greedy result (J={}):", greedy_offsets.len());
    println!("  Offsets: {:?}", greedy_offsets);
    println!("  Passkey coverage:   {}/{}", greedy_metrics.passkey_coverage, PASSKEY_DISTANCES.len());
    println!("  Passkey redundancy: {}/{}", greedy_metrics.passkey_redundancy, PASSKEY_DISTANCES.len());
    println!("  Redundancy score:   {:.2}", greedy_metrics.redundancy_score);
    println!("  Min passkey paths:  {}", greedy_metrics.min_passkey_paths);
    println!("  Coprime pairs:      {}", greedy_metrics.coprime_pairs);
    println!("  Frobenius number:   {}", greedy_metrics.frobenius_number);
    println!("  Score:              {:.2}", score(&greedy_metrics, greedy_offsets.len()));
    println!();

    // 4. Minimize: try to remove offsets from greedy result
    println!("── Minimization: removing redundant offsets ─────────────────────");
    let (min_offsets, min_metrics) = minimize(&greedy_offsets);
    println!("  Minimal result (J={}):", min_offsets.len());
    println!("  Offsets: {:?}", min_offsets);
    println!("  Passkey coverage:   {}/{}", min_metrics.passkey_coverage, PASSKEY_DISTANCES.len());
    println!("  Passkey redundancy: {}/{}", min_metrics.passkey_redundancy, PASSKEY_DISTANCES.len());
    println!("  Redundancy score:   {:.2}", min_metrics.redundancy_score);
    println!("  Min passkey paths:  {}", min_metrics.min_passkey_paths);
    println!("  Coprime pairs:      {}", min_metrics.coprime_pairs);
    println!("  Frobenius number:   {}", min_metrics.frobenius_number);
    println!();

    // 5. Per-distance breakdown for comparison
    println!("── Per-passkey-distance path counts ─────────────────────────────");
    println!("  {:>6}  {:>10}  {:>10}  {:>10}",
        "d", "J24D paths", "Greedy paths", "Minimal paths");
    let j24_paths  = count_paths(J24D_OFFSETS, MAX_HOPS, MAX_D);
    let grdy_paths = count_paths(&greedy_offsets, MAX_HOPS, MAX_D);
    let mini_paths = count_paths(&min_offsets, MAX_HOPS, MAX_D);
    for &d in PASSKEY_DISTANCES {
        println!("  {:>6}  {:>10}  {:>10}  {:>10}",
            d, j24_paths[d], grdy_paths[d], mini_paths[d]);
    }
    println!();

    // 6. Frobenius-motivated structural analysis
    println!("── Frobenius analysis of minimal set ────────────────────────────");
    println!("  Coprime pair analysis:");
    for i in 0..min_offsets.len() {
        for j in (i+1)..min_offsets.len() {
            let a = min_offsets[i];
            let b = min_offsets[j];
            if gcd(a, b) == 1 {
                let frob = if a >= 2 && b >= 2 {
                    a * b - a - b  // Sylvester-Frobenius for two generators
                } else { 0 };
                println!("    ({}, {}): gcd=1, Frobenius_number={}", a, b, frob);
            }
        }
    }
    println!();

    // 7. Sensitivity: what happens if each offset in the minimal set is removed?
    println!("── Single-offset removal sensitivity (minimal set) ──────────────");
    println!("  {:>6}  {:>10}  {:>14}  {:>14}",
        "offset", "pk_cov", "pk_redund", "redund_score");
    for &off in &min_offsets {
        if off == 1 { continue; } // never remove δ=1, skip it
        let trial: Vec<usize> = min_offsets.iter().filter(|&&o| o != off).copied().collect();
        let m = compute_coverage(&trial);
        println!("  {:>6}  {:>10}  {:>14}  {:>14.1}",
            off, m.passkey_coverage, m.passkey_redundancy, m.redundancy_score);
    }

    println!("\n================================================================");
    println!("  Summary:");
    println!("  J24D baseline:  J={}  score={:.0}  Frobenius={}",
        J24D_OFFSETS.len(),
        score(&j24_metrics, J24D_OFFSETS.len()),
        j24_metrics.frobenius_number);
    println!("  Greedy result:  J={}  score={:.0}  Frobenius={}",
        greedy_offsets.len(),
        score(&greedy_metrics, greedy_offsets.len()),
        greedy_metrics.frobenius_number);
    println!("  Minimal result: J={}  score={:.0}  Frobenius={}",
        min_offsets.len(),
        score(&min_metrics, min_offsets.len()),
        min_metrics.frobenius_number);
    if min_offsets.len() < J24D_OFFSETS.len() {
        println!("\n  ✓ Found J={} set achieving comparable coverage to J=24",
            min_offsets.len());
        println!("    Saved {} offsets ({:.0}% reduction)",
            J24D_OFFSETS.len() - min_offsets.len(),
            100.0 * (J24D_OFFSETS.len() - min_offsets.len()) as f64 / J24D_OFFSETS.len() as f64);
    } else {
        println!("\n  J=24 appears near-minimal for full redundant coverage.");
        println!("  Greedy + minimize did not find a smaller set.");
        println!("  Frobenius-optimal insight: add 3rd coprime generator for J+1 redundancy.");
    }
    println!("================================================================");
}
