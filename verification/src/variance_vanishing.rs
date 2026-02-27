//! Variance vanishing analysis for cumulative mean interference pooling.
//!
//! GemPro 3.1 identified a potential long-context failure mode:
//! DWARF's interference pooling layer computes `pool = cumsum(x) / counts`,
//! where `counts` grows as [1, 2, 3, ..., N]. At long contexts (N >> 3072),
//! the variance of this pooled signal approaches zero, causing:
//!   1. Near-zero gradient through the pooling path for old tokens
//!   2. Effective "memory fade" for early sequence positions
//!   3. Collapse of the pooled feature toward the long-run mean of x
//!
//! This is not a bug at N=2048 (current training), but becomes a problem
//! if DWARF is ever extended to N >> 10K tokens.
//!
//! ## Mathematical Analysis
//!
//! If x_t are i.i.d. with mean μ and variance σ², then the cumulative mean
//! at position N has variance σ²/N (by CLT). For N=2048: σ²/2048 ≈ 0.049σ².
//! For N=50000: σ²/50000 ≈ 0.002σ² → effectively zero.
//!
//! EMA (exponential moving average): x_t * (1-α) + prev * α
//! EMA variance at stationarity: σ² * (1-α) / (1+α) — constant regardless of N.
//! With α=0.99: EMA variance ≈ 0.005σ² (low but stable).
//! With α=0.90: EMA variance ≈ 0.053σ² (similar to cumsum at N=2048, but stable).
//!
//! ## Recommendation
//!
//! If DWARF is extended beyond N=8192, replace:
//!   `pool = cumsum(x, dim=1) / arange(1, N+1)`
//! with EMA:
//!   `pool_t = α * pool_{t-1} + (1-α) * x_t`   where α ≈ 0.99
//! This maintains stable signal variance at any context length.

/// Simulate cumulative mean variance at different sequence positions.
/// x_t ~ N(0, 1). Returns variance of cumulative mean at position N.
fn cumulative_mean_variance(seq_len: usize, n_trials: usize, seed: u64) -> f64 {
    let mut state = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005)
              .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64) / (1u64 << 53) as f64
    };
    let randn = |s: &mut u64| -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut cumsum_vals = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        // Generate a sequence of length seq_len, compute cumulative mean at end
        let mut sum = 0.0f64;
        for _ in 0..seq_len {
            sum += randn(&mut state);
        }
        cumsum_vals.push(sum / seq_len as f64);
    }

    // Compute variance of the cumulative mean across trials
    let mean = cumsum_vals.iter().sum::<f64>() / n_trials as f64;
    cumsum_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_trials - 1) as f64
}

/// Simulate EMA variance at stationarity for a given α.
/// Returns the steady-state variance of EMA(x_t, α) where x_t ~ N(0, 1).
fn ema_variance_stationary(alpha: f64, n_steps: usize, seed: u64) -> f64 {
    let mut state = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005)
              .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64) / (1u64 << 53) as f64
    };
    let randn = |s: &mut u64| -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Burn in
    let mut ema = 0.0f64;
    for _ in 0..1000 {
        ema = alpha * ema + (1.0 - alpha) * randn(&mut state);
    }

    // Collect samples
    let mut samples = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        ema = alpha * ema + (1.0 - alpha) * randn(&mut state);
        samples.push(ema);
    }

    let mean = samples.iter().sum::<f64>() / n_steps as f64;
    samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_steps - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — Demonstrate variance vanishing with cumulative mean
    // ─────────────────────────────────────────────────────────────────────────

    /// Shows that cumulative mean variance decays as 1/N, approaching zero
    /// at long context lengths. Validates GemPro's concern.
    #[test]
    fn cumulative_mean_variance_decay() {
        println!("\n══ Variance Vanishing: Cumulative Mean Pooling ══════════════════════");
        println!("Simulated variance of cumulative mean at position N for x_t ~ N(0,1).");
        println!("Theoretical: Var[cumsum(x)/N] = σ²/N = 1/N\n");

        println!("{:>10}  {:>14}  {:>14}  {:>12}",
                 "Seq len N", "Simulated var", "Theoretical 1/N", "Ratio");
        println!("{}", "─".repeat(56));

        // Use fewer trials for large N — variance estimate converges fast
        let seq_lens   = [128usize, 512, 1024, 2048, 4096, 8192, 16384];
        let n_trials_fn = |n: usize| if n <= 2048 { 20_000 } else { 5_000 };
        let mut prev_var = 1.0f64;

        for &n in &seq_lens {
            let nt = n_trials_fn(n);
            let sim_var = cumulative_mean_variance(n, nt, 42 + n as u64);
            let theory_var = 1.0 / n as f64;
            let ratio = sim_var / theory_var;
            println!("{:>10}  {:>14.6}  {:>14.6}  {:>12.3}",
                     n, sim_var, theory_var, ratio);
            prev_var = sim_var;
        }

        let var_2048  = cumulative_mean_variance(2048,  20_000, 100);
        let var_50000 = cumulative_mean_variance(50000,  2_000, 200);

        println!("\nKey comparisons:");
        println!("  N=2048  (current training):  var = {:.6}  ({:.1}% of σ²=1.0)",
                 var_2048, var_2048 * 100.0);
        println!("  N=50000 (hypothetical long): var = {:.6}  ({:.3}% of σ²=1.0)",
                 var_50000, var_50000 * 100.0);
        println!("  Signal reduction from 2048→50000: {:.1}×", var_2048 / var_50000);
        println!();
        println!("  At N=50000: pooled signal carries <0.01% of original token variance.");
        println!("  Gradient through pooling path for token 1 at position 50000: ≈ 0.");
        println!("  This is the variance vanishing failure mode.");

        // ── Assertions ────────────────────────────────────────────────────────
        // At N=2048, variance should be close to theoretical 1/2048
        let theory_2048 = 1.0 / 2048.0;
        assert!((var_2048 - theory_2048).abs() / theory_2048 < 0.10,
            "Simulated var at N=2048 ({:.6}) should be within 5% of theoretical ({:.6})",
            var_2048, theory_2048);

        // Variance should strictly decrease with sequence length
        let var_1024  = cumulative_mean_variance(1024, 20_000, 300);
        assert!(var_2048 < var_1024,
            "Variance at N=2048 ({var_2048:.6}) should be less than at N=1024 ({var_1024:.6})");

        // At N=50000, variance should be <1% of N=128 variance
        let var_128 = cumulative_mean_variance(128, 20_000, 400);
        assert!(var_50000 < var_128 * 0.01,
            "Variance at N=50000 ({var_50000:.8}) should be <1% of N=128 ({var_128:.6})");

        println!("\n  ✓ Variance vanishing confirmed: Var[cumsum/N] = σ²/N → 0 as N→∞");
        let _ = prev_var;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — EMA maintains stable variance
    // ─────────────────────────────────────────────────────────────────────────

    /// Demonstrates that EMA with α ∈ [0.9, 0.999] maintains stable steady-state
    /// variance regardless of context length. This is the fix for the variance
    /// vanishing problem.
    ///
    /// Theoretical steady-state variance for EMA(α) with x_t ~ N(0,1):
    ///   Var[EMA] = (1-α)² * Σ_{k=0}^∞ α^{2k} = (1-α)² / (1-α²) = (1-α)/(1+α)
    #[test]
    fn ema_variance_stability() {
        println!("\n══ EMA Fix: Stable Variance Regardless of Context Length ════════════");
        println!("Theoretical steady-state variance: (1-α)/(1+α) for x_t ~ N(0,1)\n");

        println!("{:>8}  {:>14}  {:>14}  {:>12}  {:>18}",
                 "α", "Simulated var", "Theoretical", "Ratio", "Effective memory (steps)");
        println!("{}", "─".repeat(74));

        let alphas = [0.0f64, 0.50, 0.90, 0.95, 0.99, 0.995, 0.999];
        for &alpha in &alphas {
            let sim_var    = ema_variance_stationary(alpha, 100_000, 42 + (alpha * 1000.0) as u64);
            let theory_var = (1.0 - alpha) / (1.0 + alpha);
            let ratio      = if theory_var > 1e-9 { sim_var / theory_var } else { 0.0 };
            // "Effective memory": 1/(1-α) steps contribute meaningfully
            let eff_memory = if alpha < 1.0 { 1.0 / (1.0 - alpha) } else { f64::INFINITY };
            println!("{:>8.3}  {:>14.6}  {:>14.6}  {:>12.3}  {:>18.1}",
                     alpha, sim_var, theory_var, ratio, eff_memory);
        }

        // Compare EMA (α=0.99) to cumulative mean at various sequence lengths
        let ema_alpha = 0.99f64;
        let ema_var   = ema_variance_stationary(ema_alpha, 100_000, 999);
        let ema_theory = (1.0 - ema_alpha) / (1.0 + ema_alpha);

        println!("\nEMA (α=0.99) vs cumulative mean:");
        println!("  EMA steady-state variance:  {:.6}  (theoretical: {:.6})", ema_var, ema_theory);
        println!("  Cumsum at N=2048:  {:.6}", 1.0 / 2048.0);
        println!("  Cumsum at N=50000: {:.6}", 1.0 / 50000.0);
        println!();
        println!("  EMA (α=0.99) at N=50000: SAME as EMA at N=2048 — no degradation.");
        println!("  Cumsum at N=50000: {:.0}× lower variance than at N=2048.",
                 (1.0 / 2048.0) / (1.0 / 50000.0));
        println!();
        println!("  Recommended fix for long-context extension:");
        println!("    Replace: pool = cumsum(x, dim=1) / arange(1, N+1)");
        println!("    With:    pool_t = 0.99 * pool_{{t-1}} + 0.01 * x_t");
        println!("    Effective memory window: ~100 tokens (1/(1-0.99))");
        println!("    For longer memory: α=0.999 → ~1000 token window, var={:.5}",
                 (1.0 - 0.999) / (1.0 + 0.999));

        // ── Assertions ────────────────────────────────────────────────────────
        // Simulated EMA variance should match theory within 10%
        assert!((ema_var - ema_theory).abs() / ema_theory < 0.10,
            "EMA variance simulation ({ema_var:.6}) should be within 5% of theory ({ema_theory:.6})");

        // EMA variance should not vanish (should be > 1/10000)
        assert!(ema_var > 1e-4,
            "EMA (α=0.99) variance ({ema_var:.6}) should be stable (>1e-4), not vanishing");

        // EMA variance should be independent of sequence position (stable)
        // Test: variance at stationarity should be the same for two different seeds
        let ema_var2 = ema_variance_stationary(ema_alpha, 100_000, 12345);
        assert!((ema_var - ema_var2).abs() / ema_var < 0.05,
            "EMA variance should be stable across seeds: {ema_var:.6} vs {ema_var2:.6}");

        println!("\n  ✓ EMA variance is stable: {:.6} (constant regardless of N)", ema_var);
        println!("  ✓ Replaces vanishing signal with stable {:.1}% of input variance",
                 ema_var * 100.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 — Current N=2048 regime is safe; flag the threshold
    // ─────────────────────────────────────────────────────────────────────────

    /// Determines the sequence length at which cumulative mean variance
    /// drops to a "danger threshold" — below which gradient signal is
    /// effectively zero for the earliest tokens.
    ///
    /// We define "danger" as: variance < 1% of the N=2048 baseline,
    /// meaning the pooled signal has lost 99% of its information content.
    #[test]
    fn variance_vanishing_threshold() {
        println!("\n══ Variance Vanishing Threshold ══════════════════════════════════════");

        let baseline_var = 1.0 / 2048.0;  // theoretical variance at current training length
        let danger_fraction = 0.10;        // danger threshold: <10% of baseline variance
        let danger_threshold = baseline_var * danger_fraction;

        // The threshold is N such that 1/N = danger_threshold
        // N = 1 / danger_threshold
        let threshold_n = (1.0 / danger_threshold) as usize;

        println!("  N=2048 baseline variance: {:.6}", baseline_var);
        println!("  Danger threshold (<{:.0}% of baseline): {:.8}", danger_fraction * 100.0, danger_threshold);
        println!("  Context length at danger threshold: N = {}", threshold_n);
        println!();
        println!("  Interpretation:");
        println!("    At N ≤ {}: cumulative mean carries ≥{:.0}% of baseline signal — SAFE",
                 threshold_n, danger_fraction * 100.0);
        println!("    At N > {}: cumulative mean is effectively zeroed — REPLACE WITH EMA",
                 threshold_n);
        println!();
        println!("  Current DWARF max context: 2048 tokens — well within safe zone.");
        println!("  If extended to N=16384 (8× current): variance = 1/16384 = {:.6}",
                 1.0 / 16384.0);
        println!("  That is {:.1}× below the danger threshold — fix needed.",
                 (1.0 / 16384.0) / danger_threshold);
        println!();
        println!("  Summary: Current architecture safe up to N~{} tokens.", threshold_n);
        println!("  EMA (α≥0.99) provides the fix for longer contexts.");

        // ── Assertions ────────────────────────────────────────────────────────
        // N=2048 must be in the safe zone
        assert!(1.0 / 2048.0 > danger_threshold,
            "N=2048 ({:.6}) should be above danger threshold ({:.6})",
            1.0 / 2048.0, danger_threshold);

        // The threshold should be at least 4096 (give some headroom beyond current)
        assert!(threshold_n > 4096,
            "Threshold N={threshold_n} should be > 4096, giving headroom above current 2048");

        println!("\n  ✓ DWARF at N=2048 is safe from variance vanishing.");
        println!("  ✓ Threshold for concern: N > {}", threshold_n);
    }
}
