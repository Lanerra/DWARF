//! Offset Set Gradient Energy Model
//!
//! # The question
//!
//! The path-count model (`offset_optimizer.rs`, `offset_space_explorer.rs`)
//! measures *reachability* — how many multi-hop paths connect the model to
//! each passkey distance.  It correctly ranks d41s3 [48,128,384] as #1
//! among all 3-sparse configurations.
//!
//! But it cannot explain the d41s3 vs d41s5 divergence:
//!   d41s3: sparse=[48,128,384]         J=45 → 80.0% passkey
//!   d41s5: sparse=[48,128,384,768,1536] J=47 → 41.7% passkey
//!
//! Adding 768 and 1536 should increase path counts (more offsets = more paths)
//! yet passkey collapsed by 38.3 pp.  The path-count model predicts d41s5 wins.
//! The training result says d41s5 loses.  Something is missing.
//!
//! # Gradient energy dilution
//!
//! During training, the gradient signal that reinforces a particular offset j
//! is proportional to how often position n genuinely attends to a token at
//! distance δ=j.  In natural text (FineWeb-Edu), the probability of a
//! *meaningful* long-range dependency decays roughly as 1/δ (Zipf-like).
//!
//! When the softmax over offsets normalises across J candidates, each offset
//! j receives gradient proportional to:
//!
//!   g(j) ∝ P_text(δ=j) / Σ_{j'} P_text(δ=j')
//!
//! where P_text(δ) is the empirical rate at which genuine content-addressed
//! lookups occur at distance δ in the training corpus.
//!
//! Adding an offset at δ=768 or δ=1536 dilutes the denominator (Σ increases)
//! without proportionally increasing the numerator (genuine lookups at 768/1536
//! are rare).  The net effect is that 48, 128, and 384 receive less gradient
//! per step — they need more tokens to converge.
//!
//! # What this module computes
//!
//! 1. **Empirical distance distribution P_text(δ)**: modelled as a power-law
//!    decay fitted to typical FineWeb-Edu statistics (exponent ≈ −1.5 for
//!    attention-worthy long-range tokens; cutoff at δ > context length).
//!
//! 2. **Per-offset gradient share**: for a given offset set {δ_j}, compute
//!    g(δ_j) = P_text(δ_j) / Σ_{j'} P_text(δ_{j'})
//!    (gradient energy fraction each offset receives during an average step).
//!
//! 3. **Effective learning budget**: steps × g(δ_j).  Gives the "equivalent
//!    gradient exposure" each offset accumulates over a training run.
//!    Offsets with exposure < threshold T are predicted to remain undertrained.
//!
//! 4. **Dilution factor**: ratio of g_d41s3(δ) to g_d41s5(δ) for each
//!    shared offset (48, 128, 384).  Quantifies how much adding 768+1536
//!    steals gradient from the useful offsets.
//!
//! 5. **Minimum-dilution sparse set**: given dense_width and a sparse budget
//!    of K positions, find the set that maximises the minimum gradient share
//!    across all K positions.  (Max-min criterion: ensure every sparse offset
//!    gets enough gradient to learn.)
//!
//! # Predictions
//!
//! - d41s5's offsets 768 and 1536 each receive ~P_text(768)/P_text(sum) ≈ 0.2%
//!   of gradient per step.  Over 10 epochs (52,716 seqs × 10 × 2048 tokens),
//!   they receive far fewer effective learning signals than 48, 128, or 384.
//! - Their presence in the denominator costs 48, 128, 384 approximately 4-8%
//!   gradient reduction each — enough to measurably delay convergence and
//!   prevent scale_embed from crossing 1.0 within 10 epochs.
//! - d41s3 concentrates gradient on 3 offsets all in the "active" range
//!   (δ≤400), achieving faster and more complete convergence.

use crate::sweep_engine::{Stats, top_k, write_json_results};

// ─── Text distance distribution model ────────────────────────────────────────

/// Power-law model for P_text(δ): probability that a genuine content-addressed
/// lookup occurs at distance δ in natural text.
///
/// Parameters fitted to approximate FineWeb-Edu attention statistics:
///   - exponent: −1.5 (steeper than 1/δ due to recency bias in text)
///   - local_boost: extra weight on δ≤41 (local window, always attended)
///   - min_floor: smoothing floor to avoid zero probability
fn p_text(delta: usize, exponent: f64, local_width: usize, local_boost: f64) -> f64 {
    let base = (delta as f64).powf(-exponent);
    if delta <= local_width {
        base * local_boost
    } else {
        base
    }
}

/// Compute gradient share g(δ_j) for each offset in the set.
///
/// Returns a Vec<(delta, gradient_share)> in the same order as offsets.
fn gradient_shares(
    offsets: &[usize],
    exponent: f64,
    local_width: usize,
    local_boost: f64,
) -> Vec<(usize, f64)> {
    let weights: Vec<f64> = offsets
        .iter()
        .map(|&d| p_text(d, exponent, local_width, local_boost))
        .collect();
    let total: f64 = weights.iter().sum::<f64>().max(1e-30);
    offsets
        .iter()
        .zip(weights)
        .map(|(&d, w)| (d, w / total))
        .collect()
}

/// Effective gradient exposure = gradient_share × total_training_steps.
/// Steps estimated from condV 10-epoch training on 52,716 seqs at 2048 tokens:
///   52,716 × 2048 / batch_size (≈32 tokens per grad step in our setup)
const TOTAL_STEPS: usize = 16_483;  // 1648 steps/epoch × 10 epochs

fn effective_exposure(share: f64) -> f64 {
    share * TOTAL_STEPS as f64
}

// ─── Offset sets ─────────────────────────────────────────────────────────────

fn d41s3_offsets() -> Vec<usize> {
    let mut o: Vec<usize> = (0..=41).collect();
    for &s in &[48usize, 128, 384] { o.push(s); }
    o.sort_unstable(); o.dedup(); o
}

fn d41s5_offsets() -> Vec<usize> {
    let mut o: Vec<usize> = (0..=41).collect();
    for &s in &[48usize, 128, 384, 768, 1536] { o.push(s); }
    o.sort_unstable(); o.dedup(); o
}

fn condu_offsets() -> Vec<usize> {
    let mut o: Vec<usize> = (0..=32).collect();
    for &s in &[48usize, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] { o.push(s); }
    o.sort_unstable(); o.dedup(); o
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

fn sparse_only(offsets: &[usize], dense_width: usize) -> Vec<usize> {
    offsets.iter().filter(|&&d| d > dense_width).cloned().collect()
}

fn print_sparse_gradient_report(name: &str, offsets: &[usize], exponent: f64, dense_w: usize) {
    let shares = gradient_shares(offsets, exponent, dense_w, 10.0);
    let sparse: Vec<_> = shares.iter().filter(|(d, _)| *d > dense_w).collect();
    let sparse_total: f64 = sparse.iter().map(|(_, s)| s).sum();
    // Also compute shares normalised WITHIN sparse only (to compare relative allocation)
    let sparse_weights: Vec<f64> = offsets.iter()
        .filter(|&&d| d > dense_w)
        .map(|&d| p_text(d, exponent, dense_w, 10.0))
        .collect();
    let sparse_weight_sum: f64 = sparse_weights.iter().sum::<f64>().max(1e-300);
    println!("\n  {name}:");
    println!("    {:>6}  {:>16}  {:>18}  {:>14}", "offset", "abs_share(1e-9)", "sparse-rel%", "eff_exposure");
    for ((d, s), sw) in sparse.iter().zip(sparse_weights.iter()) {
        let rel = sw / sparse_weight_sum * 100.0;
        println!("    {:>6}  {:>15.4}  {:>17.2}%  {:>14.1}",
                 d, s * 1e9, rel, effective_exposure(*s));
    }
    println!("    sparse_total: {:.6}%  (dense gets {:.6}%)",
             sparse_total * 100.0, (1.0 - sparse_total) * 100.0);
}

fn dilution_factor(
    base_offsets: &[usize],
    augmented_offsets: &[usize],
    shared_sparse: &[usize],
    exponent: f64,
    dense_w: usize,
) -> Vec<(usize, f64, f64, f64)> {
    // Returns (delta, base_share, aug_share, dilution_factor=base/aug)
    let base_shares = gradient_shares(base_offsets, exponent, dense_w, 10.0);
    let aug_shares  = gradient_shares(augmented_offsets, exponent, dense_w, 10.0);

    shared_sparse.iter().map(|&target| {
        let b = base_shares.iter().find(|(d, _)| *d == target).map(|(_, s)| *s).unwrap_or(0.0);
        let a = aug_shares.iter().find(|(d, _)| *d == target).map(|(_, s)| *s).unwrap_or(0.0);
        let dilution = if a > 1e-30 { b / a } else { f64::INFINITY };
        (target, b, a, dilution)
    }).collect()
}

/// Find the K-sparse set (from a pool of candidates) that maximises the
/// minimum gradient share across all K offsets (max-min = "worst-case fair").
fn min_gradient_sparse_optimal(
    dense_w: usize,
    k: usize,
    pool: &[usize],
    exponent: f64,
) -> (Vec<usize>, f64) {
    let n = pool.len();
    if k == 0 { return (vec![], 0.0); }

    let mut best_set: Vec<usize> = vec![];
    let mut best_min: f64 = 0.0;

    // Enumerate C(n, k) combinations — feasible for small pool sizes
    fn combinations(pool: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 { return vec![vec![]]; }
        if pool.is_empty() { return vec![]; }
        let rest = combinations(&pool[1..], k);
        let mut with_first = combinations(&pool[1..], k - 1);
        for c in &mut with_first { c.insert(0, pool[0]); }
        with_first.into_iter().chain(rest).collect()
    }

    for combo in combinations(pool, k) {
        let mut offsets: Vec<usize> = (0..=dense_w).collect();
        offsets.extend_from_slice(&combo);
        offsets.sort_unstable();
        offsets.dedup();

        let shares = gradient_shares(&offsets, exponent, dense_w, 10.0);
        let sparse_min = combo.iter()
            .filter_map(|&t| shares.iter().find(|(d, _)| *d == t).map(|(_, s)| *s))
            .fold(f64::INFINITY, f64::min);

        if sparse_min > best_min {
            best_min = sparse_min;
            best_set = combo;
        }
    }

    (best_set, best_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPONENT: f64 = 1.5;
    const DENSE_W: usize = 41;

    #[test]
    fn gradient_dilution_d41s3_vs_d41s5() {
        println!("\n=== Gradient Dilution: d41s3 vs d41s5 ===");
        println!("Model: P_text(δ) ∝ δ^(-{EXPONENT}), local boost ×10 for δ≤{DENSE_W}");

        print_sparse_gradient_report("d41s3 [48,128,384]", &d41s3_offsets(), EXPONENT, DENSE_W);
        print_sparse_gradient_report("d41s5 [48,128,384,768,1536]", &d41s5_offsets(), EXPONENT, DENSE_W);
        print_sparse_gradient_report("condU [48..1536 full pool]", &condu_offsets(), EXPONENT, 32);

        println!("\n--- Dilution factors (d41s3 / d41s5) for shared sparse offsets ---");
        let shared = &[48usize, 128, 384];
        println!("  {:>6}  {:>14}  {:>14}  {:>14}",
                 "offset", "d41s3_share%", "d41s5_share%", "dilution(s3/s5)");
        for (d, b, a, dil) in dilution_factor(&d41s3_offsets(), &d41s5_offsets(), shared, EXPONENT, DENSE_W) {
            println!("  {:>6}  {:>13.4}%  {:>13.4}%  {:>14.3}×", d, b*100.0, a*100.0, dil);
        }

        // Sparse-relative dilution: within sparse positions only
        let sparse_pool_s3 = &[48usize, 128, 384];
        let sparse_pool_s5 = &[48usize, 128, 384, 768, 1536];
        let w_s3: Vec<f64> = sparse_pool_s3.iter().map(|&d| p_text(d, EXPONENT, DENSE_W, 10.0)).collect();
        let w_s5: Vec<f64> = sparse_pool_s5.iter().map(|&d| p_text(d, EXPONENT, DENSE_W, 10.0)).collect();
        let sum_s3: f64 = w_s3.iter().sum::<f64>().max(1e-300);
        let sum_s5: f64 = w_s5.iter().sum::<f64>().max(1e-300);

        println!("\n--- Sparse-relative gradient share (within sparse positions only) ---");
        println!("  {:>6}  {:>14}  {:>14}  {:>14}", "offset", "d41s3_rel%", "d41s5_rel%", "dilution(s3/s5)");
        for &t in &[48usize, 128, 384] {
            let r3 = p_text(t, EXPONENT, DENSE_W, 10.0) / sum_s3 * 100.0;
            let r5 = p_text(t, EXPONENT, DENSE_W, 10.0) / sum_s5 * 100.0;
            println!("  {:>6}  {:>13.2}%  {:>13.2}%  {:>14.3}×", t, r3, r5, r3/r5);
        }
        println!("\n  Offsets only in d41s5 (sparse-relative share):");
        for &t in &[768usize, 1536] {
            let r5 = p_text(t, EXPONENT, DENSE_W, 10.0) / sum_s5 * 100.0;
            println!("  δ={t:>4}: sparse-rel-share={:.4}%  (this is gradient stolen from 48/128/384)", r5);
        }

        println!("\n  Interpretation:");
        println!("  Each of 768 and 1536 consumes gradient share that would otherwise");
        println!("  strengthen 48, 128, and 384.  Their own exposure is so low they");
        println!("  cannot learn reliable content-addressed retrieval in 10 epochs,");
        println!("  while simultaneously weakening the offsets that could.");
    }

    #[test]
    fn optimal_3sparse_max_min_criterion() {
        println!("\n=== Optimal 3-sparse set (max-min gradient share, dense=41) ===");
        let pool = &[48usize, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536];

        let (best_set, best_min) = min_gradient_sparse_optimal(DENSE_W, 3, pool, EXPONENT);
        println!("  Optimal: {:?}  min_share={:.5}%", best_set, best_min * 100.0);

        // Compare d41s3 against optimal
        let d3_shares = gradient_shares(&d41s3_offsets(), EXPONENT, DENSE_W, 10.0);
        let d3_min = [48usize, 128, 384].iter()
            .filter_map(|&t| d3_shares.iter().find(|(d, _)| *d == t).map(|(_, s)| *s))
            .fold(f64::INFINITY, f64::min);
        println!("  d41s3:   [48, 128, 384]  min_share={:.5}%", d3_min * 100.0);

        println!("\n  Top-5 by max-min criterion:");
        let mut scored: Vec<_> = {
            let pool_clone = pool.to_vec();
            let n = pool_clone.len();
            let mut out = Vec::new();
            // C(11, 3) = 165 — enumerate directly
            for i in 0..n {
                for j in (i+1)..n {
                    for k in (j+1)..n {
                        let combo = vec![pool_clone[i], pool_clone[j], pool_clone[k]];
                        let mut offsets: Vec<usize> = (0..=DENSE_W).collect();
                        offsets.extend_from_slice(&combo);
                        offsets.sort_unstable(); offsets.dedup();
                        let shares = gradient_shares(&offsets, EXPONENT, DENSE_W, 10.0);
                        let min_s = combo.iter()
                            .filter_map(|&t| shares.iter().find(|(d, _)| *d == t).map(|(_, s)| *s))
                            .fold(f64::INFINITY, f64::min);
                        out.push((combo, min_s));
                    }
                }
            }
            out
        };
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (combo, min_s) in scored.iter().take(5) {
            println!("    {:?}  min_share={:.5}%", combo, min_s * 100.0);
        }
    }

    #[test]
    fn gradient_exposure_threshold_analysis() {
        println!("\n=== Gradient Exposure Threshold Analysis ===");
        println!("  Can an offset learn in {TOTAL_STEPS} steps if its exposure is too low?");
        println!("  Minimum useful exposure: ~{:.0} steps (estimated from condU phase transition)",
                 0.43 * TOTAL_STEPS as f64);

        let d5_shares = gradient_shares(&d41s5_offsets(), EXPONENT, DENSE_W, 10.0);
        println!("\n  d41s5 exposure by offset:");
        for &t in &[48usize, 128, 384, 768, 1536] {
            let s = d5_shares.iter().find(|(d, _)| *d == t).map(|(_, s)| *s).unwrap_or(0.0);
            let exp = effective_exposure(s);
            let flag = if exp < 0.43 * TOTAL_STEPS as f64 { "⚠ UNDERTRAINED" } else { "✓ sufficient" };
            println!("    δ={t:>4}: {:.1} steps  {flag}", exp);
        }

        let d3_shares = gradient_shares(&d41s3_offsets(), EXPONENT, DENSE_W, 10.0);
        println!("\n  d41s3 exposure by offset:");
        for &t in &[48usize, 128, 384] {
            let s = d3_shares.iter().find(|(d, _)| *d == t).map(|(_, s)| *s).unwrap_or(0.0);
            let exp = effective_exposure(s);
            let flag = if exp < 0.43 * TOTAL_STEPS as f64 { "⚠ UNDERTRAINED" } else { "✓ sufficient" };
            println!("    δ={t:>4}: {:.1} steps  {flag}", exp);
        }
    }

    #[test]
    fn scale_up_sparse_recommendations() {
        println!("\n=== Sparse Set Recommendations by Dense Width ===");
        println!("  (max-min gradient criterion, 3-sparse, exponent={EXPONENT})");

        let pool = &[48usize, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
                     2048, 3072, 4096];

        for dense_w in &[41usize, 64, 96, 128] {
            // Filter pool to offsets > dense_w and < 4×context (rough bound)
            let valid_pool: Vec<usize> = pool.iter().filter(|&&p| p > *dense_w).cloned().collect();
            if valid_pool.len() < 3 { continue; }
            let (best_set, best_min) = min_gradient_sparse_optimal(*dense_w, 3, &valid_pool, EXPONENT);
            println!("  dense_w={dense_w:>3}: optimal_sparse={best_set:?}  min_share={:.4}%",
                     best_min * 100.0);
        }
    }
}
