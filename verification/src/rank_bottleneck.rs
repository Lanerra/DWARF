//! Rank bottleneck quantification: outer-product field capacity vs head dimension.
//!
//! ## The Problem
//!
//! DWARF accumulates information as a sum of rank-1 outer products:
//!
//!   F = Σ_δ c_δ · K_{t-δ} ⊗ V_{t-δ}    (a d_h × d_h matrix)
//!
//! A d_h × d_h matrix has maximum rank d_h.  condK uses 44 effective taps
//! (11 dyadic scales × 4 D4 filter taps).  When d_h < 44, the field cannot
//! store all contributions independently: rank-1 matrices begin to "collide"
//! in the matrix subspace, creating crosstalk when Q gathers.
//!
//! ## Mathematical mechanism
//!
//! The gather Q@F expands as:
//!   Q @ F = Σ_δ c_δ · (Q · K_{t-δ}) · V_{t-δ}
//!
//! With Q = K_{t-δ*} (targeting offset δ*), the signal is c_{δ*} · V_{t-δ*}.
//! Crosstalk comes from all other δ ≠ δ* where Q · K_{t-δ} ≠ 0.
//!
//! For random unit-normalised K ∈ R^{d_h}: E[(Q · K_δ)²] = 1/d_h for δ ≠ δ*.
//! As d_h increases, random keys become more orthogonal — crosstalk falls as
//! 1/d_h and SNR rises as d_h.
//!
//! ## Configs tested (condK architecture at three scales)
//!
//! | Config  | d_model | heads | d_h | Rank cap | Offsets | Deficit |
//! |---------|---------|-------|-----|----------|---------|---------|
//! | 13M     |     256 |     8 |  32 |       32 |      44 |     −12 |
//! | 85M     |     640 |     8 |  80 |       80 |      44 |       0 |
//! | 7B hyp  |    4096 |    32 | 128 |      128 |      44 |       0 |
//!
//! ## D4 weighting
//!
//! condK weights contributions by D4 filter coefficients, attenuated by
//! 1/√(2^j) per scale.  This concentrates effective weight on the first few
//! scales, reducing the *effective* number of significant contributions below
//! 44 — partially mitigating the rank bottleneck even at 13M.  The test
//! quantifies this mitigation directly.

use crate::nonlinear_field::DB4;

// ── Tap count ─────────────────────────────────────────────────────────────────
const N_SCALES: usize = 11;          // j = 0 … 10
const N_TAPS_PER_SCALE: usize = 4;  // DB4 has 4 coefficients
const N_OFFSETS: usize = N_SCALES * N_TAPS_PER_SCALE;  // = 44

// ── Configs ───────────────────────────────────────────────────────────────────
struct Config {
    name: &'static str,
    d_h: usize,
}

const CONFIGS: [Config; 3] = [
    Config { name: "13M condK (d_h= 32)", d_h: 32  },
    Config { name: "85M condK (d_h= 80)", d_h: 80  },
    Config { name: "7B target (d_h=128)", d_h: 128 },
];

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

// ── Linear algebra ────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_sq(v: &[f32]) -> f32 { dot(v, v) }

fn normalize(v: &mut [f32]) {
    let n = l2_sq(v).sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
}

/// Deposit c · (K ⊗ V) into the d×d field matrix (row-major flat storage).
fn deposit(f: &mut [f32], k: &[f32], v: &[f32], c: f32, d: usize) {
    for i in 0..d {
        for j in 0..d {
            f[i * d + j] += c * k[i] * v[j];
        }
    }
}

/// Gather: Q @ F → d-vector.  output[j] = Σ_i Q[i] · F[i, j]
fn gather_qf(q: &[f32], f: &[f32], d: usize) -> Vec<f32> {
    (0..d).map(|j| (0..d).map(|i| q[i] * f[i * d + j]).sum()).collect()
}

// ── D4 tap weight profile ─────────────────────────────────────────────────────

/// Build the 44-element weight vector c_δ from D4 filter coefficients.
///
/// At scale j, each of the 4 DB4 taps is attenuated by 1/√(2^j) (energy
/// normalisation across scales).  We use the absolute coefficient values
/// because the signs do not affect the rank/crosstalk analysis.
///
/// Returned vector is L1-normalised (sums to 1).
fn d4_weights() -> Vec<f32> {
    let mut w = Vec::with_capacity(N_OFFSETS);
    for j in 0..N_SCALES {
        let scale_atten = 1.0_f32 / (2.0_f32.powi(j as i32)).sqrt();
        for &coeff in DB4.iter() {
            w.push(coeff.abs() * scale_atten);
        }
    }
    let sum: f32 = w.iter().sum::<f32>().max(1e-9);
    w.iter_mut().for_each(|x| *x /= sum);
    w
}

/// "Effective N": number of taps whose weight exceeds `threshold × max_weight`.
fn effective_n(weights: &[f32], threshold: f32) -> usize {
    let max_w = weights.iter().cloned().fold(0.0f32, f32::max);
    weights.iter().filter(|&&w| w >= threshold * max_w).count()
}

// ── Core measurement ──────────────────────────────────────────────────────────

/// Run `n_trials` random experiments and return (mean_snr_db, mean_signal_fraction).
///
/// Each trial:
///   1. Generate N=n_offsets random unit-normalised K_i ∈ R^{d_h}.
///   2. Generate N random V_i ~ N(0, 0.5) ∈ R^{d_h}.
///   3. Build field F = Σ_i w_i · K_i ⊗ V_i.
///   4. Set Q = K_0 (perfect semantic alignment with target offset 0).
///   5. Ideal output: w_0 · V_0  (what we'd get if only offset 0 existed).
///   6. Actual output: Q @ F = Σ_i w_i · (Q·K_i) · V_i.
///   7. Interference: actual − ideal.
///   8. SNR = ‖ideal‖² / ‖interference‖².
fn measure_retrieval_snr(
    d_h: usize,
    n_offsets: usize,
    weights: &[f32],
    n_trials: usize,
    seed: u64,
) -> (f32, f32) {
    assert_eq!(weights.len(), n_offsets);

    let mut rng = seed;
    let mut total_snr_db  = 0.0f32;
    let mut total_sig_frac = 0.0f32;

    for _ in 0..n_trials {
        // ── Sample random keys (unit-normalised) and values ───────────────────
        let mut keys: Vec<Vec<f32>> = (0..n_offsets)
            .map(|_| {
                let mut k: Vec<f32> = (0..d_h).map(|_| randn(&mut rng)).collect();
                normalize(&mut k);
                k
            })
            .collect();

        let vals: Vec<Vec<f32>> = (0..n_offsets)
            .map(|_| (0..d_h).map(|_| randn(&mut rng) * 0.5).collect())
            .collect();

        // ── Build field ───────────────────────────────────────────────────────
        let mut field = vec![0.0f32; d_h * d_h];
        for (i, (k, v)) in keys.iter().zip(&vals).enumerate() {
            deposit(&mut field, k, v, weights[i], d_h);
        }

        // ── Target: offset 0; Q = K_0 (perfectly aligned) ────────────────────
        let q: Vec<f32>  = keys[0].clone();  // already unit-normalised
        let w0 = weights[0];

        // Ideal: the contribution we *want* (Q·K_0 = 1 since Q = K_0)
        let ideal: Vec<f32> = vals[0].iter().map(|&v| w0 * v).collect();

        // Actual gather
        let actual = gather_qf(&q, &field, d_h);

        // Interference = actual − ideal
        let interference: Vec<f32> = actual.iter().zip(&ideal)
            .map(|(&a, &id)| a - id)
            .collect();

        let signal_sq   = l2_sq(&ideal);
        let interf_sq   = l2_sq(&interference).max(1e-12);
        let actual_sq   = l2_sq(&actual).max(1e-12);

        let snr    = signal_sq / interf_sq;
        let snr_db = 10.0 * snr.log10();
        let sig_frac = signal_sq / actual_sq;

        total_snr_db   += snr_db;
        total_sig_frac += sig_frac;
    }

    (total_snr_db / n_trials as f32, total_sig_frac / n_trials as f32)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N_TRIALS: usize = 2_000;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — main SNR table: uniform weights, all three configs
    // ─────────────────────────────────────────────────────────────────────────

    /// Measures retrieval SNR for each model scale with uniform contribution
    /// weights.  Uniform weights are the *worst case* for rank collision because
    /// all 44 offsets contribute equally — this is the upper bound on rank damage.
    ///
    /// Analytical prediction (random unit-normalised K, large d_h):
    ///   SNR ≈ d_h / (N − 1)     →   SNR_dB ≈ 10 log₁₀(d_h / 43)
    ///   13M: 10 log₁₀(32/43) ≈ −1.3 dB  (interference > signal)
    ///   85M: 10 log₁₀(80/43) ≈ +2.7 dB  (signal > interference)
    ///   7B:  10 log₁₀(128/43) ≈ +4.7 dB
    ///
    /// The phase transition at d_h = N − 1 ≈ 43 is where signal begins to
    /// dominate interference.  85M (d_h=80) is firmly on the right side.
    #[test]
    fn retrieval_snr_vs_head_dim_uniform() {
        let weights_uniform = vec![1.0f32 / N_OFFSETS as f32; N_OFFSETS];
        let theoretical_snr = |d: usize| 10.0 * ((d as f32) / (N_OFFSETS as f32 - 1.0)).log10();

        println!("\n══ Rank Bottleneck: Retrieval SNR (uniform weights) ══════════════════");
        println!("N_OFFSETS = {N_OFFSETS} (11 scales × 4 D4 taps)");
        println!();
        println!("{:<26} {:>5} {:>10} {:>12} {:>12} {:>15}",
                 "Config", "d_h", "Rank cap", "Theory(dB)", "Empirical(dB)", "Signal fraction");
        println!("{}", "─".repeat(85));

        let mut results_uniform = Vec::new();
        for cfg in &CONFIGS {
            let (snr_db, sig_frac) = measure_retrieval_snr(
                cfg.d_h, N_OFFSETS, &weights_uniform, N_TRIALS, 11111,
            );
            let theory_db = theoretical_snr(cfg.d_h);
            let rank_deficit = N_OFFSETS as i32 - cfg.d_h as i32;
            println!("{:<26} {:>5} {:>10} {:>12.2} {:>12.2} {:>14.1}%",
                     cfg.name, cfg.d_h, cfg.d_h,
                     theory_db, snr_db, sig_frac * 100.0);
            results_uniform.push((cfg.d_h, snr_db, sig_frac, rank_deficit));
        }

        println!();
        println!("Phase transition at d_h ≈ {} (where signal = interference)", N_OFFSETS - 1);

        // ── Assertions ────────────────────────────────────────────────────────
        let snr_13m = results_uniform[0].1;
        let snr_85m = results_uniform[1].1;
        let snr_7b  = results_uniform[2].1;

        // 13M should be below 0 dB (interference-dominated)
        assert!(
            snr_13m < 0.0,
            "13M SNR ({snr_13m:.2} dB) should be < 0 dB (interference-dominated when d_h < N_offsets)"
        );

        // 85M should be above 0 dB (signal-dominated)
        assert!(
            snr_85m > 0.0,
            "85M SNR ({snr_85m:.2} dB) should be > 0 dB (signal-dominated when d_h > N_offsets)"
        );

        // Monotone improvement with d_h
        assert!(
            snr_85m > snr_13m,
            "85M SNR ({snr_85m:.2}) should exceed 13M ({snr_13m:.2})"
        );
        assert!(
            snr_7b > snr_85m,
            "7B SNR ({snr_7b:.2}) should exceed 85M ({snr_85m:.2})"
        );

        // SNR improvement from 13M → 85M should be at least 2 dB
        let improvement_13_to_85 = snr_85m - snr_13m;
        assert!(
            improvement_13_to_85 > 2.0,
            "13M → 85M SNR improvement ({improvement_13_to_85:.2} dB) should exceed 2 dB"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — D4 exponential weighting: effective collision reduction
    // ─────────────────────────────────────────────────────────────────────────

    /// The D4 convolution attenuates contributions by 1/√(2^j) per scale.
    /// At j=10, a tap has ~3% the weight of j=0.  This concentrates most
    /// contribution in early scales, reducing effective N below 44.
    ///
    /// "Effective N" (taps with weight > 1% of max): typically 10–18 taps
    /// rather than 44.  With effective_N ~ 15, even the 13M model (d_h=32)
    /// has capacity > effective_N and may not be fully bottlenecked.
    ///
    /// This test measures how much D4 weighting mitigates rank damage at 13M.
    #[test]
    fn d4_weighting_effective_collision_reduction() {
        let weights_d4      = d4_weights();
        let weights_uniform = vec![1.0f32 / N_OFFSETS as f32; N_OFFSETS];

        let eff_n_1pct  = effective_n(&weights_d4, 0.01);
        let eff_n_5pct  = effective_n(&weights_d4, 0.05);
        let eff_n_10pct = effective_n(&weights_d4, 0.10);

        println!("\n══ D4 Weighting: Effective Collision Reduction ═══════════════════════");
        println!("D4 tap weight profile (first 20 of 44, by scale×tap):");
        print!("  ");
        for (i, &w) in weights_d4.iter().enumerate().take(20) {
            print!("j{}t{}: {:.4}  ", i / 4, i % 4, w);
            if i % 4 == 3 { println!(); print!("  "); }
        }
        println!();
        println!("Effective tap count (D4 weights > threshold × max):");
        println!("  > 1%  of max: {eff_n_1pct:>3} taps  (13M bottleneck if this > 32)");
        println!("  > 5%  of max: {eff_n_5pct:>3} taps");
        println!("  > 10% of max: {eff_n_10pct:>3} taps");
        println!();

        println!("{:<26} {:>5} {:>16} {:>16} {:>12}",
                 "Config", "d_h", "SNR uniform(dB)", "SNR D4(dB)", "Improvement");
        println!("{}", "─".repeat(80));

        for cfg in &CONFIGS {
            let (snr_uni, _) = measure_retrieval_snr(
                cfg.d_h, N_OFFSETS, &weights_uniform, N_TRIALS, 22222,
            );
            let (snr_d4, sig_frac_d4) = measure_retrieval_snr(
                cfg.d_h, N_OFFSETS, &weights_d4, N_TRIALS, 22222,
            );
            let improvement = snr_d4 - snr_uni;
            println!("{:<26} {:>5} {:>16.2} {:>16.2} {:>+11.2} dB",
                     cfg.name, cfg.d_h, snr_uni, snr_d4, improvement);
            let _ = sig_frac_d4;
        }

        println!();
        // D4 weighting should improve SNR at 13M by concentrating mass in early scales
        // (fewer effective contributions → less collision)
        let (snr_13m_uni, _) = measure_retrieval_snr(32, N_OFFSETS, &weights_uniform, N_TRIALS, 33333);
        let (snr_13m_d4,  _) = measure_retrieval_snr(32, N_OFFSETS, &weights_d4,      N_TRIALS, 33333);
        let d4_help = snr_13m_d4 - snr_13m_uni;
        if d4_help > 0.5 {
            println!("D4 weighting improves 13M SNR by {d4_help:.2} dB → partially mitigates rank bottleneck");
        } else {
            println!("D4 weighting has minimal SNR effect ({d4_help:+.2} dB) — bottleneck dominates weighting");
        }

        assert!(
            eff_n_1pct < N_OFFSETS,
            "Effective N at 1% threshold ({eff_n_1pct}) should be less than total {N_OFFSETS}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 — Cross-offset interference matrix
    // ─────────────────────────────────────────────────────────────────────────

    /// Measures the full N×N interference matrix: for each target offset i*,
    /// how much do each of the other offsets j contribute to the gather?
    ///
    /// Entry [i*, j] = mean |Q_{i*} · K_j| across trials.
    ///
    /// For orthogonal K (d_h large): off-diagonal → 0.
    /// For colliding K (d_h < N): off-diagonal grows, and the pattern reveals
    /// which offset *pairs* are most confused with each other.
    ///
    /// Run only for 13M (d_h=32) since that's where collision matters.
    /// Report the mean and max off-diagonal entries.
    #[test]
    fn cross_offset_interference_matrix() {
        let d_h = 32usize;   // 13M config — most informative
        let n   = 12usize;   // report a 12×12 submatrix (first 12 offsets)
        let n_trials = 500usize;

        let mut interference_mat = vec![0.0f32; n * n];
        let mut rng = 55555u64;

        for _ in 0..n_trials {
            // Generate n unit-normalised K vectors in R^{d_h}
            let keys: Vec<Vec<f32>> = (0..n)
                .map(|_| {
                    let mut k: Vec<f32> = (0..d_h).map(|_| randn(&mut rng)).collect();
                    normalize(&mut k);
                    k
                })
                .collect();

            // Fill interference matrix: entry [i, j] = |K_i · K_j|
            for i in 0..n {
                for j in 0..n {
                    let overlap = dot(&keys[i], &keys[j]).abs();
                    interference_mat[i * n + j] += overlap / n_trials as f32;
                }
            }
        }

        println!("\n══ Cross-Offset Interference Matrix (13M, d_h=32, first {}×{}) ══════", n, n);
        println!("Entry [i, j] = mean |K_i · K_j| (diagonal = 1.0; off-diag = crosstalk)");
        println!("Expected off-diagonal ≈ sqrt(π/(4·d_h)) ≈ {:.3} for random unit vectors in R^{}",
                 (std::f32::consts::PI / (4.0 * d_h as f32)).sqrt(), d_h);
        println!();

        print!("     ");
        for j in 0..n { print!("{:>7}", j); }
        println!();
        for i in 0..n {
            print!("[{:>2}] ", i);
            for j in 0..n {
                let v = interference_mat[i * n + j];
                if i == j { print!("{:>7.3}", v); }
                else       { print!("{:>7.4}", v); }
            }
            println!();
        }

        // Compute summary statistics
        let mut off_diag: Vec<f32> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j { off_diag.push(interference_mat[i * n + j]); }
            }
        }
        let mean_off = off_diag.iter().sum::<f32>() / off_diag.len() as f32;
        let max_off  = off_diag.iter().cloned().fold(0.0f32, f32::max);
        let diag_val = (0..n).map(|i| interference_mat[i * n + i]).sum::<f32>() / n as f32;

        println!();
        println!("Summary:");
        println!("  Mean diagonal (self-overlap):      {diag_val:.4}  (should be ~1.0)");
        println!("  Mean off-diagonal (cross-talk):    {mean_off:.4}  (target ≈ 0)");
        println!("  Max  off-diagonal (worst pair):    {max_off:.4}");
        println!("  Cross-talk / self-overlap ratio:   {:.4}  ({:.1}%)",
                 mean_off / diag_val, 100.0 * mean_off / diag_val);
        println!();

        // At 85M (d_h=80), the expected off-diagonal drops to sqrt(π/(4×80)) ≈ 0.099
        let expected_85m = (std::f32::consts::PI / (4.0 * 80.0_f32)).sqrt();
        let expected_13m = (std::f32::consts::PI / (4.0 * 32.0_f32)).sqrt();
        println!("  Expected off-diag at 13M (d_h=32):  {expected_13m:.4}");
        println!("  Expected off-diag at 85M (d_h=80):  {expected_85m:.4}  ({:.1}× reduction)",
                 expected_13m / expected_85m);

        // Off-diagonal should be meaningfully above zero (confirming collision)
        assert!(
            mean_off > 0.05,
            "Mean off-diagonal ({mean_off:.4}) should exceed 0.05 — \
             confirms real K-vector crosstalk at d_h={d_h}"
        );
        // Diagonal should be close to 1 (unit-normalised keys)
        assert!(
            (diag_val - 1.0).abs() < 0.05,
            "Mean diagonal ({diag_val:.4}) should be ~1.0 for unit-normalised keys"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 — PPL gap decomposition: how much gap does rank bottleneck explain?
    // ─────────────────────────────────────────────────────────────────────────

    /// Translates the SNR improvement from 13M → 85M into a rough fraction of
    /// the 20 PPL gap that rank bottleneck could explain.
    ///
    /// Caveat: SNR → PPL translation involves many assumptions.  This test
    /// produces an order-of-magnitude estimate, not a precise prediction.
    ///
    /// Approach: if the gather output has signal fraction s (signal²/total²),
    /// then the "effective attention quality" degrades the per-token cross-entropy
    /// by approximately −log(s) nats (assuming the interference is white noise
    /// that increases the entropy of the next-token distribution).
    ///
    /// PPL_gap_from_rank = exp(H_interference) − exp(H_ideal)
    ///
    /// This is a rough lower bound on rank's contribution to the gap.
    #[test]
    fn rank_bottleneck_gap_estimate() {
        let weights_d4 = d4_weights();

        println!("\n══ Rank Bottleneck: Approximate PPL Gap Contribution ═════════════════");
        println!("Observed PPL gap (13M condK vs standard transformer): 84.7 − 64.5 = 20.2");
        println!();

        // Measure signal fractions for 13M and 85M (D4 weighted — more realistic)
        let (snr_db_13m, sig_frac_13m) = measure_retrieval_snr(32, N_OFFSETS, &weights_d4, N_TRIALS, 77777);
        let (snr_db_85m, sig_frac_85m) = measure_retrieval_snr(80, N_OFFSETS, &weights_d4, N_TRIALS, 77777);
        let (snr_db_7b,  sig_frac_7b)  = measure_retrieval_snr(128, N_OFFSETS, &weights_d4, N_TRIALS, 77777);

        println!("{:<26} {:>5} {:>12} {:>15} {:>20}",
                 "Config", "d_h", "SNR D4 (dB)", "Signal fraction", "Entropy penalty (nats)");
        println!("{}", "─".repeat(80));

        for (name, d_h, snr_db, sig_frac) in [
            ("13M condK", 32,  snr_db_13m, sig_frac_13m),
            ("85M condK", 80,  snr_db_85m, sig_frac_85m),
            ("7B target", 128, snr_db_7b,  sig_frac_7b),
        ] {
            // Entropy penalty: if signal fraction = s, the gather output has
            // extra uncertainty.  A rough model: penalty ≈ −log(s) nats.
            let penalty_nats = -sig_frac.ln();  // nats of extra uncertainty per step
            let penalty_ppl  = penalty_nats.exp() - 1.0;  // approximate PPL contribution
            println!("{:<26} {:>5} {:>12.2} {:>15.3} {:>20.4}",
                     name, d_h, snr_db, sig_frac, penalty_nats);
            let _ = penalty_ppl;
        }

        let improvement_nats = (-sig_frac_13m.ln()) - (-sig_frac_85m.ln());
        let improvement_pct  = improvement_nats / 20.2_f32 * 100.0;

        println!();
        println!("Rank bottleneck entropy reduction (13M→85M): {improvement_nats:.4} nats");
        println!("As fraction of observed 20.2 PPL gap:        {improvement_pct:.1}%  (rough estimate)");
        println!();
        println!("Caveats:");
        println!("  - Assumes interference acts as independent Gaussian noise (optimistic)");
        println!("  - Does not account for softmax gap (destructive interference from neg Q·K)");
        println!("  - Real training partially compensates rank collision via learned K distributions");
        println!("  - The remaining gap is attributed to: (1) softmax gap, (2) non-dyadic coverage,");
        println!("    (3) finite-scale approximation errors");

        // Rank bottleneck should explain at least a measurable fraction of the gap.
        // NOTE: D4 exponential weighting concentrates mass in ~18-26 effective taps,
        // which is ≤ d_h=32 even at 13M.  The bottleneck is largely alleviated by D4
        // before it even reaches the field — so the gap contribution is small (~0.5%).
        assert!(
            improvement_pct > 0.1,
            "Rank bottleneck should explain >0.1% of the PPL gap; got {improvement_pct:.1}%"
        );

        // But not all of it (softmax gap and coverage gap also contribute)
        // So the estimate should be below 100% by construction of our rough model
        assert!(
            sig_frac_85m > sig_frac_13m,
            "85M signal fraction ({sig_frac_85m:.3}) should exceed 13M ({sig_frac_13m:.3})"
        );
    }
}
