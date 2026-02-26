//! F_K normalization verification: linear attention denominator.
//!
//! ## What F_K normalization is
//!
//! Maintain a secondary D-vector field accumulating D4-weighted past keys:
//!   F_K,t = D4_Conv_vec(F_K,{t-1}) + K_t    (D-vector, not D×D matrix)
//!
//! Then normalize the gather output:
//!   output = (Q @ F_prop) / max(Q · F_K, ε)
//!
//! This is the standard linear attention denominator from Katharopoulos et al.
//! (2020), re-derived independently by Gemini as a DWARF improvement.
//!
//! ## Literature status (as of Feb 2026)
//!
//! Modern linear attention (DeltaNet, GLA, RWKV, Mamba) has **abandoned** the
//! F_K denominator due to numerical instability and marginal benefit. The
//! denominator is dropped in favour of output layer normalisation.
//!
//! This module provides a **debugging-only** reference test to quantify the
//! denominator collapse rate at our scale (d_h=32, 85M d_h=80). The test
//! informs whether to use F_K normalization as a stability fallback if condO
//! or condL exhibit unnormalized output divergence.
//!
//! ## What this module verifies (1 test)
//!
//! **F_K denominator collapse rate**: What fraction of (Q, F_K) pairs produce
//! |Q · F_K| < ε for ε ∈ {0.01, 0.001, 0.0001}? High collapse rate means the
//! denominator will cause NaN gradients during training. Measured at both
//! d_h=32 (13M) and d_h=80 (85M) to see if larger head dim mitigates collapse.
//!
//! If collapse rate > 1% at ε=0.001: F_K normalization requires careful ε-clamping
//! and is fragile. Better to use output LayerNorm instead (standard practice).

use crate::nonlinear_field::{causal_conv, DB4};

// ── Constants ─────────────────────────────────────────────────────────────────
const N_TRIALS: usize = 10_000;
const N_SCALES: usize = 11;   // j=0..10

// ── PRNG ──────────────────────────────────────────────────────────────────────
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005)
                  .wrapping_add(1_442_695_040_888_963_407);
    let bits = 0x3F80_0000u32 | ((*state >> 41) as u32 & 0x007F_FFFF);
    f32::from_bits(bits) - 1.0
}

fn randn(state: &mut u64) -> f32 {
    let u1 = lcg(state).max(1e-10);
    let u2 = lcg(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Build F_K by accumulating N_SCALES × 4 D4-weighted random K vectors.
/// This simulates F_K after ~44 steps of accumulation.
fn build_fk(d: usize, rng: &mut u64) -> Vec<f32> {
    let mut fk = vec![0.0f32; d];
    let mut history: Vec<Vec<f32>> = Vec::with_capacity(N_SCALES * 4);

    // Build K history (most recent first)
    for _ in 0..N_SCALES * 4 {
        let k: Vec<f32> = (0..d).map(|_| randn(rng) / (d as f32).sqrt()).collect();
        history.push(k);
    }

    // Accumulate with D4 weights (same profile as rank_bottleneck)
    for j in 0..N_SCALES {
        let scale_atten = 1.0_f32 / (2.0_f32.powi(j as i32)).sqrt();
        for t in 0..4 {
            let coeff = DB4[t].abs() * scale_atten;
            let k = &history[j * 4 + t];
            for i in 0..d {
                fk[i] += coeff * k[i];
            }
        }
    }

    // Apply causal_conv to F_K (simulate one D4_Conv_vec step)
    // This mirrors what would happen in the actual condO/F_K computation
    let fk_convolved = causal_conv(&fk, &DB4);
    fk_convolved
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — F_K denominator collapse rate
    // ─────────────────────────────────────────────────────────────────────────

    /// Measures what fraction of (Q, F_K) pairs produce near-zero denominator.
    ///
    /// During training, Q and K co-evolve, and the model may learn to push Q
    /// orthogonal to F_K if this helps minimize some other loss component.
    /// This test measures the RANDOM-INIT collapse rate — a lower bound on
    /// the training-time collapse risk.
    ///
    /// Setup:
    ///   - Q ~ N(0, 1/√d_h) (typical post-LayerNorm query scale)
    ///   - F_K built from D4-weighted accumulation of 44 past K vectors ~ N(0, 1/√d_h)
    ///   - Measure |Q · F_K| distribution and collapse fraction at multiple ε
    ///
    /// Also compares:
    ///   - d_h=32 (13M config): does the low dimension cause high collapse risk?
    ///   - d_h=80 (85M config): does larger head dim reduce collapse?
    ///
    /// Decision rule:
    ///   - Collapse rate at ε=0.001 > 1% → F_K normalization is fragile; use LayerNorm instead
    ///   - Collapse rate at ε=0.001 < 0.1% → F_K normalization is safe with simple ε clamping
    #[test]
    fn fk_denominator_collapse_rate() {
        println!("\n══ F_K Denominator Collapse Rate ════════════════════════════════════");
        println!("(Q ~ N(0,1/√d), F_K built from 44 D4-weighted past K vectors)");
        println!();

        for &d_h in &[32usize, 80] {
            let mut rng = 13579u64 + d_h as u64;

            let mut qdotfk: Vec<f32> = Vec::with_capacity(N_TRIALS);
            let mut neg_count = 0u32;

            for _ in 0..N_TRIALS {
                // Q ~ N(0, 1/√d_h) — post-LayerNorm query
                let q: Vec<f32> = (0..d_h).map(|_| randn(&mut rng) / (d_h as f32).sqrt()).collect();

                // F_K from D4-weighted accumulation
                let fk = build_fk(d_h, &mut rng);

                let val = dot(&q, &fk);
                if val < 0.0 { neg_count += 1; }
                qdotfk.push(val.abs());  // we can take |Q·F_K| (or use max(Q·F_K, ε))
            }

            // Sort for percentile analysis
            let mut sorted = qdotfk.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = N_TRIALS as f32;
            let mean  = qdotfk.iter().sum::<f32>() / n;
            let p1    = sorted[(N_TRIALS / 100)];
            let p5    = sorted[(N_TRIALS / 20)];
            let p50   = sorted[N_TRIALS / 2];
            let min   = sorted[0];
            let neg_pct = neg_count as f32 / n * 100.0;

            // Collapse rates at various ε thresholds
            let collapse_1pct  = qdotfk.iter().filter(|&&v| v < 0.01).count() as f32 / n * 100.0;
            let collapse_01pct = qdotfk.iter().filter(|&&v| v < 0.001).count() as f32 / n * 100.0;
            let collapse_001pct= qdotfk.iter().filter(|&&v| v < 0.0001).count() as f32 / n * 100.0;

            println!("── d_h = {d_h} ─────────────────────────────────────────────────────");
            println!("  |Q·F_K| distribution over {N_TRIALS} trials:");
            println!("    Min:    {min:.6}");
            println!("    P1:     {p1:.6}");
            println!("    P5:     {p5:.6}");
            println!("    Median: {p50:.6}");
            println!("    Mean:   {mean:.6}");
            println!("    Q·F_K < 0 (sign flip): {neg_pct:.1}%");
            println!();
            println!("  Collapse rate (|Q·F_K| < ε):");
            println!("    ε = 0.010 : {collapse_1pct:.2}%");
            println!("    ε = 0.001 : {collapse_01pct:.3}%  ← key threshold");
            println!("    ε = 0.0001: {collapse_001pct:.4}%");
            println!();

            if collapse_01pct > 1.0 {
                println!("  → FRAGILE: {collapse_01pct:.2}% collapse rate at ε=0.001");
                println!("    F_K normalization needs aggressive ε-clamping or safe-exp (ReGLA-style).");
                println!("    Recommendation: use output LayerNorm instead.");
            } else if collapse_01pct > 0.1 {
                println!("  → MARGINAL: {collapse_01pct:.3}% collapse rate at ε=0.001");
                println!("    Manageable with ε-clamping, but monitor during training.");
            } else {
                println!("  → SAFE: {collapse_01pct:.4}% collapse rate at ε=0.001");
                println!("    F_K normalization is numerically safe with ε = 0.001 clamping.");
            }
            println!();

            // ── Assertions ────────────────────────────────────────────────────
            // Absolute min must not be exactly zero (would indicate degenerate F_K build)
            assert!(
                min > 0.0,
                "d_h={d_h}: minimum |Q·F_K| = 0.0 — F_K build may be degenerate (all-zero field)"
            );

            // Median |Q·F_K| should be in a reasonable range (not astronomically large/small)
            assert!(
                p50 > 0.001 && p50 < 100.0,
                "d_h={d_h}: median |Q·F_K| = {p50:.6} is outside expected range [0.001, 100.0] — \
                 check Q and F_K scale"
            );
        }

        println!("── Summary ──────────────────────────────────────────────────────────");
        println!("Literature consensus (DeltaNet, GLA, RWKV, Mamba): F_K denominator");
        println!("is abandoned in modern linear attention due to instability.");
        println!("If collapse rate > 0.1% at ε=0.001: use output LayerNorm instead.");
        println!("F_K normalization should only be used as a debugging fallback,");
        println!("not as a primary architecture component.");
    }
}
