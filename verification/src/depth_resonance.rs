//! Depth Resonance Analysis
//!
//! Tests the hypothesis that relay chain quality is resonant at powers-of-2 depths
//! (L=2,4,8,16) and degraded at non-power-of-2 depths (L=10,12).
//!
//! ## Approach
//!
//! 1. **Relay frontier by depth** — for each L, compute how far a signal can
//!    propagate in L hops using J=26 offsets. Powers-of-2 might show qualitatively
//!    different coverage curves.
//!
//! 2. **Interference metric** — model the relay chain as a discrete signal
//!    processing pipeline. At each layer, the signal is a weighted sum of
//!    contributions from J offset positions. Compute the "interference pattern"
//!    as a function of L. If constructive interference peaks at L=2^k, the
//!    resonance hypothesis holds.
//!
//! 3. **Phase accumulation** — relay via offset δ introduces a "phase shift"
//!    proportional to log(δ). After L layers of relay, total phase accumulation
//!    = Σ log(δ_i). If this sums to a multiple of 2π at powers-of-2 depths,
//!    constructive interference occurs.
//!
//! 4. **Relay graph completeness** — is the relay graph "closed" (every distance
//!    reachable) at power-of-2 depths but not at L=10,12?
//!
//! ## Observable prediction
//!
//! If resonance is real:
//!   - Frontier coverage should show step-function jumps at L=1,2,4,8,16
//!   - L=10,12 frontiers should be "between" L=8 and L=16 with partial coverage
//!   - Interference amplitude should peak at L=2^k

use std::collections::{HashMap, HashSet};

/// J=26 offset set (J26D)
const J26_OFFSETS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 21, 23, 28, 32, 48,
    64, 96, 192, 384, 512, 768, 1024,
];

/// J=20 offset set (V10, used in depth scaling experiments)
const J20_OFFSETS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96,
    192, 384, 512, 768, 1024,
];

const MAX_DISTANCE: usize = 2048;

/// Compute the relay frontier after `num_layers` hops.
/// Frontier = set of distances reachable by chaining <= num_layers offsets.
fn relay_frontier(offsets: &[usize], num_layers: usize) -> HashSet<usize> {
    let mut reachable: HashSet<usize> = HashSet::new();
    reachable.insert(0); // distance 0 always reachable

    for _ in 0..num_layers {
        let current: Vec<usize> = reachable.iter().cloned().collect();
        for &base in &current {
            for &delta in offsets {
                let new_dist = base + delta;
                if new_dist <= MAX_DISTANCE {
                    reachable.insert(new_dist);
                }
            }
        }
    }
    reachable
}

/// Compute frontier size (number of reachable distances) per layer depth.
fn frontier_by_depth(offsets: &[usize], max_layers: usize) -> Vec<(usize, usize, usize)> {
    let mut results = Vec::new();
    for l in 1..=max_layers {
        let frontier = relay_frontier(offsets, l);
        let max_dist = *frontier.iter().max().unwrap_or(&0);
        let coverage = frontier.len();
        results.push((l, coverage, max_dist));
    }
    results
}

/// Compute coverage *gaps* — spans of distances not reachable at depth L.
fn coverage_gaps(offsets: &[usize], num_layers: usize) -> Vec<(usize, usize)> {
    let frontier = relay_frontier(offsets, num_layers);
    let max_dist = *frontier.iter().max().unwrap_or(&0);
    let mut gaps = Vec::new();
    let mut gap_start = None;

    for d in 1..=max_dist {
        if !frontier.contains(&d) {
            if gap_start.is_none() {
                gap_start = Some(d);
            }
        } else if let Some(start) = gap_start {
            gaps.push((start, d - 1));
            gap_start = None;
        }
    }
    gaps
}

/// Interference metric: model relay as repeated convolution.
/// Each layer applies a "relay filter" h[δ] = 1 for δ in offsets, 0 elsewhere.
/// After L layers, the total response is h^L (L-fold convolution).
/// Measure: sum of squared amplitudes (energy) of the response at key distances.
fn relay_energy(offsets: &[usize], num_layers: usize) -> f64 {
    // Start with impulse at position 0
    let mut signal: HashMap<usize, f64> = HashMap::new();
    signal.insert(0, 1.0);

    let norm = offsets.len() as f64;

    for _ in 0..num_layers {
        let mut new_signal: HashMap<usize, f64> = HashMap::new();
        for (&pos, &amp) in &signal {
            for &delta in offsets {
                let new_pos = pos + delta;
                if new_pos <= MAX_DISTANCE {
                    *new_signal.entry(new_pos).or_insert(0.0) += amp / norm;
                }
            }
        }
        signal = new_signal;
    }

    // Energy = sum of squared amplitudes
    signal.values().map(|&a| a * a).sum()
}

/// Phase accumulation model.
/// Each relay hop through offset δ accumulates phase φ(δ) = 2π * log2(δ) / log2(max_δ).
/// After L hops, measure constructive interference: |Σ exp(i * total_phase)|^2 / N^2.
/// Peaks at L where phase wraps to multiples of 2π.
fn phase_coherence(offsets: &[usize], num_layers: usize) -> f64 {
    let max_delta = *offsets.iter().max().unwrap_or(&1) as f64;
    let log_max = max_delta.log2();

    // For each path of length num_layers, compute total phase
    // Approximate: use mean phase per layer × num_layers
    let mean_phase: f64 = offsets.iter()
        .map(|&d| 2.0 * std::f64::consts::PI * (d as f64).log2() / log_max)
        .sum::<f64>() / offsets.len() as f64;

    let total_phase = mean_phase * num_layers as f64;

    // Constructive interference when total_phase ≈ 2πk
    // Measure: cos²(total_phase / 2) → 1.0 at resonance, 0.0 at anti-resonance
    (total_phase / 2.0).cos().powi(2)
}

/// Relay chain "completeness ratio" — fraction of distances 1..MAX_DISTANCE reachable.
fn completeness(offsets: &[usize], num_layers: usize) -> f64 {
    let frontier = relay_frontier(offsets, num_layers);
    // Count reachable distances up to max(frontier)
    let max_dist = *frontier.iter().max().unwrap_or(&0);
    if max_dist == 0 { return 0.0; }
    frontier.len() as f64 / max_dist as f64
}

fn is_power_of_2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEPTH RESONANCE ANALYSIS");
    println!("  Hypothesis: relay chain quality is resonant at L = powers of 2");
    println!("═══════════════════════════════════════════════════════════════");

    let max_layers = 20;

    // ── Analysis 1: Frontier coverage by depth ───────────────────────────────
    println!("\n── Relay Frontier Coverage by Depth (J=26) ──────────────────");
    println!("{:>4}  {:>8}  {:>10}  {:>11}  {:>8}  {:>6}",
             "L", "Coverage", "MaxDist", "Completeness", "POW2?", "Energy");
    println!("{}", "─".repeat(60));

    let depths_to_test = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20];

    for &l in &depths_to_test {
        let frontier = relay_frontier(J26_OFFSETS, l);
        let max_dist = *frontier.iter().max().unwrap_or(&0);
        let coverage = frontier.len();
        let comp = completeness(J26_OFFSETS, l);
        let energy = relay_energy(J26_OFFSETS, l);
        let pow2 = if is_power_of_2(l) { "★ POW2" } else { "" };

        println!("{:>4}  {:>8}  {:>10}  {:>11.4}  {:>8}  {:>8.4e}",
                 l, coverage, max_dist, comp, pow2, energy);
    }

    // ── Analysis 2: Phase coherence ──────────────────────────────────────────
    println!("\n── Phase Coherence by Depth ──────────────────────────────────");
    println!("(1.0 = fully constructive, 0.0 = fully destructive)");
    println!("{:>4}  {:>12}  {:>12}  {:>8}",
             "L", "Coherence_J26", "Coherence_J20", "POW2?");
    println!("{}", "─".repeat(45));

    for &l in &depths_to_test {
        let c26 = phase_coherence(J26_OFFSETS, l);
        let c20 = phase_coherence(J20_OFFSETS, l);
        let pow2 = if is_power_of_2(l) { "★ POW2" } else { "" };
        println!("{:>4}  {:>12.6}  {:>12.6}  {:>8}", l, c26, c20, pow2);
    }

    // ── Analysis 3: Coverage gaps at key depths ──────────────────────────────
    println!("\n── Coverage Gaps at Key Depths (J=26) ───────────────────────");
    for &l in &[6, 7, 8, 9, 10, 12, 16] {
        let gaps = coverage_gaps(J26_OFFSETS, l);
        let total_gap: usize = gaps.iter().map(|(s, e)| e - s + 1).sum();
        let pow2 = if is_power_of_2(l) { " ★" } else { "" };
        println!("  L={:>2}{}: {} gaps, {} uncovered distances (gaps: {:?})",
                 l, pow2, gaps.len(), total_gap,
                 &gaps[..gaps.len().min(3)]);
    }

    // ── Analysis 4: Delta between consecutive depths ─────────────────────────
    println!("\n── Marginal Coverage Gain per Additional Layer (J=26) ────────");
    println!("{:>4} → {:>4}  {:>12}  {:>10}  {:>8}",
             "L", "L+1", "NewDistances", "MaxDistGain", "POW2→?");
    println!("{}", "─".repeat(55));

    let mut prev_frontier = relay_frontier(J26_OFFSETS, 1);
    let mut prev_max = *prev_frontier.iter().max().unwrap_or(&0);

    for l in 2..=18 {
        let frontier = relay_frontier(J26_OFFSETS, l);
        let max_dist = *frontier.iter().max().unwrap_or(&0);
        let new_dists = frontier.len() - prev_frontier.len();
        let max_gain = max_dist.saturating_sub(prev_max);
        let label = if is_power_of_2(l) { "★ POW2" } else { "" };
        println!("{:>4} → {:>4}  {:>12}  {:>10}  {:>8}",
                 l-1, l, new_dists, max_gain, label);
        prev_frontier = frontier;
        prev_max = max_dist;
    }

    // ── Analysis 5: Empirical comparison with training data ──────────────────
    println!("\n── Empirical Passkey vs Geometric Prediction (J=20) ─────────");
    println!("Known results from depth scaling experiments:");
    println!("  L=6  (J26D): passkey=94.2%  ar_score=75.87  [reference]");
    println!("  L=8  (J20D): passkey=93.3%  ar_score=75.03  [best L>6]");
    println!("  L=10 (J20D): passkey=87.5%  ar_score=69.20  [degraded]");
    println!("  L=12 (J20D): passkey=87.5%  ar_score=69.20  [plateau]");
    println!("  L=16 (pred): passkey=???    ar_score=???    [pending]");
    println!();

    // Compute frontier completeness at each empirically tested depth for J20
    for &(l, emp_passkey, emp_ar) in &[(6usize, 94.2f64, 75.87f64), (8, 93.3, 75.03), (10, 87.5, 69.20), (12, 87.5, 69.20)] {
        let comp = completeness(J20_OFFSETS, l - 1); // L-1 DSQG blocks do the relay
        let coh = phase_coherence(J20_OFFSETS, l - 1);
        let pow2 = if is_power_of_2(l) { " ★" } else { "" };
        println!("  L={:>2}{}: completeness={:.4}  coherence={:.4}  empirical_passkey={:.1}%  ar={:.2}",
                 l, pow2, comp, coh, emp_passkey, emp_ar);
    }

    println!();

    // ── Resonance verdict ────────────────────────────────────────────────────
    println!("── Resonance Hypothesis Assessment ──────────────────────────");

    // Check if powers-of-2 show higher marginal coverage gains
    let mut pow2_gains = Vec::new();
    let mut non_pow2_gains = Vec::new();

    let mut prev = relay_frontier(J26_OFFSETS, 1).len();
    for l in 2..=16 {
        let curr = relay_frontier(J26_OFFSETS, l).len();
        let gain = curr - prev;
        if is_power_of_2(l) {
            pow2_gains.push((l, gain));
        } else {
            non_pow2_gains.push((l, gain));
        }
        prev = curr;
    }

    let pow2_mean: f64 = pow2_gains.iter().map(|&(_, g)| g as f64).sum::<f64>()
        / pow2_gains.len() as f64;
    let non_pow2_mean: f64 = non_pow2_gains.iter().map(|&(_, g)| g as f64).sum::<f64>()
        / non_pow2_gains.len() as f64;

    println!("  Mean marginal coverage gain:");
    println!("    Powers of 2 (L=2,4,8,16):     {:.1} new distances/layer", pow2_mean);
    println!("    Non-powers of 2 (L=3,5,6,..): {:.1} new distances/layer", non_pow2_mean);

    if pow2_mean > non_pow2_mean * 1.2 {
        println!("\n  ★ SUPPORTS resonance hypothesis: powers-of-2 show significantly");
        println!("    higher marginal coverage gain (>{:.0}% above average)", (pow2_mean/non_pow2_mean - 1.0)*100.0);
    } else if pow2_mean > non_pow2_mean {
        println!("\n  ~ WEAK support: powers-of-2 show slightly higher marginal gain");
        println!("    ({:.1}% above average — may not be significant)", (pow2_mean/non_pow2_mean - 1.0)*100.0);
    } else {
        println!("\n  ✗ DOES NOT SUPPORT resonance hypothesis from coverage alone.");
        println!("    Coverage gains are not systematically higher at powers of 2.");
        println!("    If resonance exists, it operates through a different mechanism.");
    }

    println!("\n  Note: Phase coherence analysis tests a different mechanism —");
    println!("  constructive interference in the relay signal, not just coverage.");
    println!("  High coherence at L=8,16 with low coherence at L=10,12 would");
    println!("  support resonance even if coverage gains are uniform.");

    println!("\n═══════════════════════════════════════════════════════════════");
}
