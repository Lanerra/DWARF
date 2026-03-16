//! Co-optimized Offset Set Search: Coverage + PPL Quality + J Budget
//!
//! Extends frobenius_offset_search.rs with an empirically-grounded PPL proxy
//! derived from the j24d_int2 ablation probe (March 15, 2026).
//!
//! ## The Missing Dimension in Pure Coverage Search
//!
//! A pure Frobenius coverage search found J=11 suffices for 12/12 passkey
//! coverage. But coverage ≠ quality:
//!   - Dense local offsets (δ=1..28) are critical for PPL — each one
//!     contributes to the model's ability to learn local language patterns
//!   - Long-range offsets (δ=64..1024) primarily drive passkey coverage
//!   - The J=11 minimal set has no local density → expect poor PPL
//!
//! ## Scoring Function Design
//!
//! score(S) = w_passkey * passkey_score(S)
//!           + w_ppl    * ppl_proxy(S)
//!           - w_j      * |S|              (penalty: fewer offsets preferred)
//!           + w_coprime * coprime_bonus(S) (bonus: multiple coprime pairs)
//!
//! where:
//!   passkey_score(S) = Σ_d log(1 + paths_to_d(S))  over passkey distances
//!   ppl_proxy(S)     = Σ_{δ ∈ S} ppl_value[δ]      (empirical contribution)
//!   coprime_bonus(S) = number of coprime pairs (gcd=1) in S
//!
//! ## PPL Proxy Construction
//!
//! From the j24d ablation (knockout of each offset from full J=24 set):
//!   passkey_drop[δ] = passkey_pp loss when δ is removed
//!
//! We use passkey_drop as a proxy for each offset's importance.
//! For PPL, we use empirical estimates from multiple runs:
//!   - δ=1..10: high PPL value (local context, highest gradient signal)
//!   - δ=11..28: medium PPL value (mid-local, diminishing returns)
//!   - δ=48..96: low-medium (transition zone)
//!   - δ=192..1024: low PPL value but critical for passkey coverage
//!
//! ## Multi-Objective Pareto Search
//!
//! For each J budget (8..23), find the Pareto-optimal sets balancing:
//!   - Passkey coverage (primary constraint: must be 12/12)
//!   - PPL proxy (maximize given coverage constraint)
//!   - Coprime pair count (robustness, secondary)
//!
//! Returns: for each J, the best set and the Pareto frontier.

use rayon::prelude::*;

const MAX_D: usize = 1536;
const MAX_HOPS: usize = 2;
const PASSKEY_DISTANCES: &[usize] = &[
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536
];

// J=24 baseline
const J24D_OFFSETS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28,
    48, 64, 96, 192, 384, 512, 768, 1024
];

// ─── Empirical PPL proxy weights ─────────────────────────────────────────────
// Estimated PPL contribution of each offset, derived from:
//   1. j24d ablation passkey_drop values
//   2. General knowledge: local offsets dominate PPL, long-range dominate passkey
//   3. d41_35m finding: dense [0..48] drives best PPL (32.136), sparse drives passkey
//
// Scale: higher = more important for PPL quality.
// These are relative, not absolute PPL points.

fn ppl_proxy_weight(offset: usize) -> f64 {
    match offset {
        // δ=1: most critical for PPL — local adjacency is the foundation of LM
        1  => 10.0,
        // δ=2..4: very high — coprime pair (3,4) most critical for relay + local LM
        2  => 6.0,
        3  => 8.0,   // critical for relay chain (gcd(3,4)=1)
        4  => 8.0,   // most critical single offset for passkey (−61.7pp ablation)
        // δ=5..10: high local density contribution
        5  => 5.0,
        6  => 4.5,
        7  => 5.0,   // secondary coprime pair with 8 (gcd(7,8)=1)
        8  => 4.5,
        9  => 3.5,
        10 => 3.0,
        // δ=11..20: medium, fills coprime gaps
        11 => 3.5,   // secondary coprime pair with 13 (gcd(11,13)=1) — new in j26d
        12 => 2.0,
        13 => 4.0,   // key offset in j24d
        14 => 1.5,
        15 => 4.0,
        16 => 3.0,
        17 => 1.5,
        18 => 1.5,
        19 => 1.5,
        20 => 1.5,
        // δ=21..28: medium-low, tail of local density
        21 => 3.0,
        22 => 1.0,
        23 => 2.5,
        24 => 1.0,
        25 => 1.0,
        26 => 1.0,
        27 => 1.0,
        28 => 2.5,
        // δ=32: closes mid-range gap (j26d addition)
        32 => 2.0,
        // δ=48..128: transition zone (d41 has these, condU doesn't)
        48 => 2.5,
        64 => 2.0,
        96 => 2.0,
        128 => 1.5,
        // δ=192..1024: primarily passkey coverage, minimal PPL
        192  => 1.0,
        256  => 1.0,
        384  => 0.8,
        512  => 0.8,
        768  => 0.5,
        1024 => 0.5,
        1536 => 0.3,
        // Unknown offsets: small contribution
        x if x < 48  => 1.0,
        x if x < 192 => 0.8,
        _             => 0.3,
    }
}

// ─── Passkey importance weights ───────────────────────────────────────────────
// From j24d ablation: passkey_drop when offset removed from full J=24 set.
// Higher = more critical for passkey.
fn passkey_criticality(offset: usize) -> f64 {
    match offset {
        4    => 61.7,
        3    => 31.7,
        7    => 25.9,
        1    => 20.0,
        13   => 20.0,
        15   => 21.7,
        8    => 18.4,
        10   => 18.4,
        6    => 17.5,
        9    => 17.5,
        23   => 17.5,
        48   => 15.9,
        5    => 14.2,
        28   => 14.2,
        21   => 15.0,
        96   => 15.0,
        16   => 10.9,
        64   => 10.9,
        192  => 10.0,
        512  => 9.2,
        384  => 7.5,
        1024 => 4.2,
        768  => 3.4,
        2    => 1.7,
        // Estimated for offsets not in j24d ablation:
        // Nearby offsets get interpolated values
        11 => 15.0,  // estimated: similar to 13 (secondary coprime pair)
        32 => 8.0,   // estimated: closes mid-range gap
        _  => 5.0,   // conservative estimate
    }
}

// ─── Coverage computation ─────────────────────────────────────────────────────

fn count_paths(offsets: &[usize], max_hops: usize, max_d: usize) -> Vec<u32> {
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

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ─── Candidate pool ───────────────────────────────────────────────────────────

fn candidate_pool() -> Vec<usize> {
    // All empirically motivated offsets, in priority order:
    // δ=1..28 (local density), δ=32 (mid-range), dyadic long-range
    let mut pool: Vec<usize> = (1..=28).collect();
    for &x in &[32usize, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] {
        pool.push(x);
    }
    // Additional coprime-generating candidates
    for &x in &[11usize, 29, 31, 33, 36, 40, 43, 47, 50] {
        if !pool.contains(&x) { pool.push(x); }
    }
    pool.sort(); pool.dedup();
    pool
}

// ─── Composite score ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct OffsetSetScore {
    offsets: Vec<usize>,
    j: usize,
    passkey_coverage: usize,   // 0..12
    passkey_redundancy: usize, // 0..12
    ppl_proxy: f64,
    passkey_proxy: f64,
    coprime_pairs: usize,
    frobenius_number: usize,
    composite: f64,
}

fn evaluate(offsets: &[usize]) -> OffsetSetScore {
    let paths = count_paths(offsets, MAX_HOPS, MAX_D);
    let j = offsets.len();

    let passkey_coverage = PASSKEY_DISTANCES.iter()
        .filter(|&&d| paths[d] >= 1)
        .count();

    let passkey_redundancy = PASSKEY_DISTANCES.iter()
        .filter(|&&d| paths[d] >= 2)
        .count();

    // PPL proxy: sum of PPL weights for each included offset
    let ppl_proxy: f64 = offsets.iter().map(|&o| ppl_proxy_weight(o)).sum();

    // Passkey proxy: sum of criticality × log(1 + paths) for passkey distances
    let passkey_proxy: f64 = PASSKEY_DISTANCES.iter()
        .map(|&d| passkey_criticality(offsets.iter().find(|&&o| o == d).copied().unwrap_or(0))
             * (1.0 + paths[d] as f64).ln())
        .sum::<f64>()
        // Also reward coverage of all passkey distances
        + passkey_coverage as f64 * 50.0
        + passkey_redundancy as f64 * 20.0;

    let mut coprime_pairs = 0usize;
    for i in 0..offsets.len() {
        for k in (i+1)..offsets.len() {
            if gcd(offsets[i], offsets[k]) == 1 { coprime_pairs += 1; }
        }
    }

    let frobenius_number = paths[1..=MAX_D].iter().enumerate()
        .rev()
        .find(|(_, &p)| p == 0)
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    // Composite: must hit full passkey coverage, then balance PPL + passkey + J
    let coverage_gate = if passkey_coverage < 12 {
        -10000.0 * (12 - passkey_coverage) as f64
    } else { 0.0 };

    let j_penalty = j as f64 * 5.0;

    // Weights: PPL and passkey roughly equal, J penalized moderately
    let composite = coverage_gate
        + ppl_proxy * 3.0
        + passkey_proxy * 1.0
        + coprime_pairs as f64 * 0.5
        - j_penalty;

    OffsetSetScore {
        offsets: offsets.to_vec(),
        j,
        passkey_coverage,
        passkey_redundancy,
        ppl_proxy,
        passkey_proxy,
        coprime_pairs,
        frobenius_number,
        composite,
    }
}

// ─── Greedy co-optimized build ────────────────────────────────────────────────

fn greedy_coopt(target_j: usize, pool: &[usize]) -> OffsetSetScore {
    // Always start with δ=1 (mandatory) and δ=4 (highest passkey criticality)
    let mut current: Vec<usize> = vec![1, 4];
    current.sort();

    while current.len() < target_j {
        // Find the offset that most improves composite, subject to:
        // - Prioritize passkey coverage first (until 12/12)
        // - Then maximize composite (PPL + passkey proxy combined)
        let cur_cov = evaluate(&current).passkey_coverage;
        let best_add = pool.iter()
            .filter(|&&o| !current.contains(&o))
            .map(|&o| {
                let mut trial = current.clone();
                trial.push(o);
                trial.sort();
                let s = evaluate(&trial);
                // Tiebreak: if coverage not full, prefer offsets that add coverage
                let sort_key = if cur_cov < 12 {
                    s.passkey_coverage as f64 * 10000.0 + s.composite
                } else {
                    s.composite
                };
                (o, sort_key, s)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match best_add {
            Some((o, _, _)) => {
                current.push(o);
                current.sort();
            }
            None => break,
        }
    }
    evaluate(&current)
}

/// Full swap refinement: try all pairwise swaps (remove one, add one from pool)
/// for multiple rounds until no improvement found.
fn full_refine(initial: &OffsetSetScore, pool: &[usize], rounds: usize) -> OffsetSetScore {
    let mut best = initial.clone();
    for _round in 0..rounds {
        let mut improved = false;
        // Single swap: replace one offset with a pool member
        for i in 0..best.offsets.len() {
            if best.offsets[i] == 1 { continue; } // never remove δ=1
            for &candidate in pool {
                if best.offsets.contains(&candidate) { continue; }
                let mut trial = best.offsets.clone();
                trial[i] = candidate;
                trial.sort();
                let s = evaluate(&trial);
                if s.composite > best.composite && s.passkey_coverage == 12 {
                    best = s;
                    improved = true;
                    break;
                }
            }
            if improved { break; }
        }
        // Addition: add one more offset if it improves composite (grow J)
        // — only if we haven't hit target J
        if !improved { break; }
    }
    best
}

/// Refinement: for each J, try all single-offset swaps to improve composite score.
fn refine(score: &OffsetSetScore, pool: &[usize]) -> OffsetSetScore {
    let mut best = score.clone();
    let j = best.j;

    // Try swapping each offset with each pool member
    for i in 0..j {
        if best.offsets[i] == 1 { continue; } // never remove δ=1
        for &candidate in pool {
            if best.offsets.contains(&candidate) { continue; }
            let mut trial = best.offsets.clone();
            trial[i] = candidate;
            trial.sort();
            let s = evaluate(&trial);
            if s.composite > best.composite && s.passkey_coverage == 12 {
                best = s;
            }
        }
    }
    best
}

// ─── Main ─────────────────────────────────────────────────────────────────────

pub fn run() {
    println!("================================================================");
    println!("  Co-Optimized Offset Search: Coverage + PPL + J Budget");
    println!("  Empirical weights from j24d_int2 ablation (March 15, 2026)");
    println!("================================================================\n");

    let pool = candidate_pool();

    // Baseline
    let j24_score = evaluate(J24D_OFFSETS);
    println!("── J=24 production baseline ─────────────────────────────────────");
    print_score(&j24_score);
    println!();

    // Search across J budgets 10..23
    println!("── Co-optimized greedy search (J=10..23) ────────────────────────");
    println!("  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>8}",
        "J", "pk_cov", "pk_red", "ppl_px", "pk_px", "composite", "frob");

    let mut best_overall: Option<OffsetSetScore> = None;
    let mut results: Vec<OffsetSetScore> = Vec::new();

    for target_j in 10..=23usize {
        let raw = greedy_coopt(target_j, &pool);
        let refined = full_refine(&raw, &pool, 10);
        let s = if refined.composite > raw.composite { refined } else { raw };

        println!("  {:>4}  {:>8}  {:>8}  {:>8.1}  {:>8.1}  {:>10.1}  {:>8}",
            s.j, s.passkey_coverage, s.passkey_redundancy,
            s.ppl_proxy, s.passkey_proxy, s.composite, s.frobenius_number);

        if s.passkey_coverage == 12 {
            match &best_overall {
                None => best_overall = Some(s.clone()),
                Some(b) => if s.composite > b.composite {
                    best_overall = Some(s.clone());
                }
            }
        }
        results.push(s);
    }
    println!();

    // Best result
    if let Some(best) = &best_overall {
        println!("── Best co-optimized result ──────────────────────────────────────");
        print_score(best);
        println!();

        // Compare to J=24
        println!("── Comparison: best co-opt vs J=24 baseline ─────────────────────");
        println!("  Metric          J=24 baseline    Best co-opt    Delta");
        println!("  J               {:>13}  {:>13}  {:>+8}",
            j24_score.j, best.j, best.j as i64 - j24_score.j as i64);
        println!("  Passkey cov     {:>13}  {:>13}  {:>+8}",
            j24_score.passkey_coverage, best.passkey_coverage,
            best.passkey_coverage as i64 - j24_score.passkey_coverage as i64);
        println!("  Passkey red     {:>13}  {:>13}  {:>+8}",
            j24_score.passkey_redundancy, best.passkey_redundancy,
            best.passkey_redundancy as i64 - j24_score.passkey_redundancy as i64);
        println!("  PPL proxy       {:>13.1}  {:>13.1}  {:>+8.1}",
            j24_score.ppl_proxy, best.ppl_proxy,
            best.ppl_proxy - j24_score.ppl_proxy);
        println!("  Coprime pairs   {:>13}  {:>13}  {:>+8}",
            j24_score.coprime_pairs, best.coprime_pairs,
            best.coprime_pairs as i64 - j24_score.coprime_pairs as i64);
        println!("  Composite       {:>13.1}  {:>13.1}  {:>+8.1}",
            j24_score.composite, best.composite,
            best.composite - j24_score.composite);
        println!();

        // Per-passkey-distance breakdown
        let j24_paths  = count_paths(J24D_OFFSETS, MAX_HOPS, MAX_D);
        let best_paths = count_paths(&best.offsets, MAX_HOPS, MAX_D);
        println!("── Per-passkey-distance path counts ─────────────────────────────");
        println!("  {:>6}  {:>12}  {:>12}",
            "d", "J=24 paths", "best paths");
        for &d in PASSKEY_DISTANCES {
            let better = if best_paths[d] > j24_paths[d] { " ▲" }
                        else if best_paths[d] < j24_paths[d] { " ▼" }
                        else { "  " };
            println!("  {:>6}  {:>12}  {:>12}{}",
                d, j24_paths[d], best_paths[d], better);
        }
        println!();

        // Sensitivity: what's most critical in the best set?
        println!("── Single-offset removal sensitivity (best co-opt set) ──────────");
        println!("  {:>6}  {:>8}  {:>8}  {:>10}  classification",
            "offset", "pk_cov", "pk_red", "composite_Δ");
        let mut sensitivities: Vec<(usize, f64, usize, usize)> = best.offsets.iter()
            .filter(|&&o| o != 1)
            .map(|&o| {
                let trial: Vec<usize> = best.offsets.iter()
                    .filter(|&&x| x != o).copied().collect();
                let s = evaluate(&trial);
                let delta = s.composite - best.composite;
                (o, delta, s.passkey_coverage, s.passkey_redundancy)
            })
            .collect();
        sensitivities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (off, delta, cov, red) in &sensitivities {
            let label = if *cov < 12 { "⚠ COVERAGE LOSS" }
                        else if *delta < -50.0 { "  critical" }
                        else if *delta < -20.0 { "  important" }
                        else { "  skippable" };
            println!("  {:>6}  {:>8}  {:>8}  {:>+10.1}  {}",
                off, cov, red, delta, label);
        }
    }

    // Pareto frontier: best PPL proxy vs passkey proxy for each J
    println!("\n── Pareto frontier: PPL proxy vs passkey coverage by J ──────────");
    println!("  {:>4}  {:>8}  {:>8}  {:>8}  offsets",
        "J", "pk_cov", "ppl_px", "pk_px");
    for s in results.iter().filter(|s| s.passkey_coverage == 12) {
        let offsets_str: Vec<String> = s.offsets.iter().map(|x| x.to_string()).collect();
        println!("  {:>4}  {:>8}  {:>8.1}  {:>8.1}  [{}]",
            s.j, s.passkey_coverage, s.ppl_proxy, s.passkey_proxy,
            offsets_str.join(", "));
    }

    println!("\n================================================================");
    println!("  Done. Recommend reviewing best co-opt set above.");
    println!("  If J < 24 with better/equal composite: candidate for V9 kernel.");
    println!("================================================================");
}

fn print_score(s: &OffsetSetScore) {
    let offsets_str: Vec<String> = s.offsets.iter().map(|x| x.to_string()).collect();
    println!("  Offsets [{}]: {:?}", s.j, s.offsets);
    println!("  Passkey coverage:   {}/{}", s.passkey_coverage, PASSKEY_DISTANCES.len());
    println!("  Passkey redundancy: {}/{}", s.passkey_redundancy, PASSKEY_DISTANCES.len());
    println!("  PPL proxy:          {:.2}", s.ppl_proxy);
    println!("  Passkey proxy:      {:.2}", s.passkey_proxy);
    println!("  Coprime pairs:      {}", s.coprime_pairs);
    println!("  Frobenius number:   {}", s.frobenius_number);
    println!("  Composite score:    {:.2}", s.composite);
}
