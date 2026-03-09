//! EMA Window × Retrieval Distance SNR
//!
//! # New empirical data (d41s3 vs d41s5, March 9 2026)
//!
//!   d41s3: EMA factor b0 = 0.0023  →  window ≈ 1/0.0023 = 435 tokens
//!          passkey mean 80.0% (eval suite), d=1536: 40%
//!
//!   d41s5: EMA factor b0 = 0.0030  →  window ≈ 1/0.0030 = 333 tokens
//!          passkey mean 41.7%, d=1536: 0%
//!
//! Question: does the 30% wider EMA window in d41s3 causally contribute to its
//! better long-range retrieval, or is b0 just co-varying with the offset set?
//!
//! # EMA window and retrieval SNR
//!
//! The Kalman-EMA interference block maintains a running estimate of the
//! context state.  At distance δ, a token deposited δ steps ago has been
//! weighted down by (1 - b0)^δ in the running estimate.
//!
//! Signal retention at distance δ:
//!   R(δ, b0) = (1 - b0)^δ
//!
//! For retrieval at distance δ, the EMA must retain enough signal above noise.
//! Assuming noise floor ε:
//!   Effective SNR(δ, b0) = R(δ, b0) / ε = (1 - b0)^δ / ε
//!
//! Minimum signal retention for retrieval (empirical threshold from condU passkey data):
//!   passkey succeeds when SNR > SNR_threshold
//!   equivalently: (1 - b0)^δ > R_min
//!   → δ < log(R_min) / log(1 - b0)  =  maximum_reliable_distance(b0)
//!
//! # What this module verifies
//!
//! 1. Signal retention curves for d41s3 (b0=0.0023) and d41s5 (b0=0.0030)
//!    at each of the 12 passkey distances.
//!
//! 2. Maximum reliable retrieval distance for each b0 (where R drops below R_min).
//!
//! 3. Whether the 30% window difference produces a theoretically significant
//!    SNR advantage at d=512, d=1024, d=1536.
//!
//! 4. The sensitivity of retrieval SNR to b0 near the d=1536 threshold:
//!    is the d41s3/d41s5 window difference large enough to straddle the
//!    threshold at extreme distances?
//!
//! 5. Optimal b0 for a given target retrieval distance — given that we want
//!    reliable retrieval at d=1536 with N=2048 context, what is the maximum
//!    b0 that keeps SNR above threshold?

const PASSKEY_DISTANCES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536];

// ─── Known EMA factors from trained models ────────────────────────────────────

struct EmaConfig {
    name: &'static str,
    b0: f64,
    passkey_by_d: [f64; 12],  // empirical passkey accuracy at each PASSKEY_DISTANCES index
}

const D41S3_EMA: EmaConfig = EmaConfig {
    name: "d41s3 (b0=0.0023, window≈435)",
    b0: 0.0023,
    passkey_by_d: [1.0, 0.8, 0.8, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.6, 0.4, 0.4],
};

const D41S5_EMA: EmaConfig = EmaConfig {
    name: "d41s5 (b0=0.0030, window≈333)",
    b0: 0.0030,
    passkey_by_d: [0.4, 0.4, 0.6, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.0],
};

const CONDU_EMA: EmaConfig = EmaConfig {
    name: "condU (b0≈0.003, window≈333)",
    b0: 0.003,
    // condU 13M passkey from condU_results.json
    passkey_by_d: [0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.5, 0.4, 0.4, 0.2, 0.2, 0.2],
};

/// Signal retention at distance δ for EMA factor b0.
fn ema_retention(delta: usize, b0: f64) -> f64 {
    (1.0 - b0).powi(delta as i32)
}

/// Maximum retrieval distance where retention > R_min.
fn max_reliable_distance(b0: f64, r_min: f64) -> f64 {
    r_min.ln() / (1.0 - b0).ln()
}

/// SNR ratio: d41s3 retention / d41s5 retention at distance δ.
fn snr_advantage(delta: usize, b0_a: f64, b0_b: f64) -> f64 {
    let ra = ema_retention(delta, b0_a);
    let rb = ema_retention(delta, b0_b);
    if rb > 1e-30 { ra / rb } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_retention_curves() {
        println!("\n=== EMA Signal Retention at Passkey Distances ===");
        println!("  Retention = (1 - b0)^δ");
        println!();
        println!("  {:>6}  {:>16}  {:>16}  {:>12}  {:>12}  {:>12}",
                 "dist", "d41s3(b0=.0023)", "d41s5(b0=.0030)",
                 "s3_retain%", "s5_retain%", "s3/s5 ratio");

        for (i, &d) in PASSKEY_DISTANCES.iter().enumerate() {
            let r3 = ema_retention(d, D41S3_EMA.b0);
            let r5 = ema_retention(d, D41S5_EMA.b0);
            let ratio = snr_advantage(d, D41S3_EMA.b0, D41S5_EMA.b0);
            let pk3 = D41S3_EMA.passkey_by_d[i] * 100.0;
            let pk5 = D41S5_EMA.passkey_by_d[i] * 100.0;
            println!("  {:>6}  {:>16.6}  {:>16.6}  {:>11.2}%  {:>11.2}%  {:>12.4}×  (pk: {pk3:.0}% / {pk5:.0}%)",
                     d, r3, r5, r3*100.0, r5*100.0, ratio);
        }
    }

    #[test]
    fn max_reliable_distance_comparison() {
        println!("\n=== Maximum Reliable Retrieval Distance ===");
        println!("  (distance where EMA retention drops below threshold R_min)");
        println!();

        for r_min in &[0.50_f64, 0.30, 0.10, 0.05, 0.01] {
            let d3_max = max_reliable_distance(D41S3_EMA.b0, *r_min);
            let d5_max = max_reliable_distance(D41S5_EMA.b0, *r_min);
            let cu_max = max_reliable_distance(CONDU_EMA.b0, *r_min);
            println!("  R_min={r_min:.2}: d41s3≤{d3_max:.0}  d41s5≤{d5_max:.0}  condU≤{cu_max:.0}  (gap: +{:.0})",
                     d3_max - d5_max);
        }

        println!();
        println!("  Observation: d41s3 extends reliable range by ~{:.0} tokens at R_min=0.10,",
                 max_reliable_distance(D41S3_EMA.b0, 0.10) - max_reliable_distance(D41S5_EMA.b0, 0.10));
        println!("  which is {:>4.0} vs {:>4.0} tokens.",
                 max_reliable_distance(D41S3_EMA.b0, 0.10),
                 max_reliable_distance(D41S5_EMA.b0, 0.10));
        println!("  Context length is 2048; passkey distances go to 1536.");
    }

    #[test]
    fn snr_advantage_at_long_range() {
        println!("\n=== d41s3 SNR Advantage Over d41s5 at Each Distance ===");
        println!();

        let mut cumulative_advantage = 1.0_f64;
        for &d in PASSKEY_DISTANCES {
            let adv = snr_advantage(d, D41S3_EMA.b0, D41S5_EMA.b0);
            cumulative_advantage = adv;
            let significance = if adv < 1.05 { "negligible" }
                               else if adv < 1.20 { "marginal" }
                               else if adv < 1.50 { "moderate" }
                               else { "substantial" };
            println!("  δ={d:>4}: d41s3/d41s5 SNR = {adv:.4}×  ({significance})");
        }

        println!();
        println!("  At d=1536, d41s3 has {:.3}× better EMA signal retention.",
                 snr_advantage(1536, D41S3_EMA.b0, D41S5_EMA.b0));
        println!("  At d=1024, d41s3 has {:.3}× better EMA signal retention.",
                 snr_advantage(1024, D41S3_EMA.b0, D41S5_EMA.b0));
    }

    #[test]
    fn optimal_b0_for_target_distance() {
        println!("\n=== Optimal b0 for Target Retrieval Distance ===");
        println!("  (maximum b0 keeping R_min≥0.10 at each target distance)");
        println!();

        for &target_d in &[256usize, 512, 1024, 1536, 2048] {
            // R_min = (1-b0)^d ≥ 0.10
            // b0 ≤ 1 - 0.10^(1/d)
            let b0_max = 1.0 - (0.10_f64).powf(1.0 / target_d as f64);
            let window = 1.0 / b0_max;
            println!("  target d={target_d:>4}: max b0={b0_max:.6}  (window≥{window:.0} tokens)");
        }

        println!();
        println!("  d41s3 b0=0.0023: window=435, consistent with target d~{}",
                 max_reliable_distance(D41S3_EMA.b0, 0.10) as usize);
        println!("  d41s5 b0=0.0030: window=333, consistent with target d~{}",
                 max_reliable_distance(D41S5_EMA.b0, 0.10) as usize);
        println!();
        println!("  Recommendation: to ensure reliable retrieval at d=1536 with N=2048,");
        println!("  target b0 ≤ {:.5}  (window ≥ {} tokens)",
                 1.0 - 0.1_f64.powf(1.0/1536.0),
                 (1.0 / (1.0 - 0.1_f64.powf(1.0/1536.0))) as usize);
    }

    #[test]
    fn correlation_ema_window_passkey() {
        println!("\n=== Correlation: EMA window vs passkey accuracy ===");
        println!("  (empirical passkey vs theoretical EMA retention)");
        println!();

        // At each distance, compare EMA retention ratio to passkey accuracy ratio
        println!("  {:>6}  {:>14}  {:>14}  {:>14}  {:>14}",
                 "dist", "s3_retention", "s5_retention", "s3_passkey%", "s5_passkey%");
        for (i, &d) in PASSKEY_DISTANCES.iter().enumerate() {
            let r3 = ema_retention(d, D41S3_EMA.b0);
            let r5 = ema_retention(d, D41S5_EMA.b0);
            let pk3 = D41S3_EMA.passkey_by_d[i] * 100.0;
            let pk5 = D41S5_EMA.passkey_by_d[i] * 100.0;
            println!("  {:>6}  {:>14.6}  {:>14.6}  {:>13.0}%  {:>13.0}%", d, r3, r5, pk3, pk5);
        }

        println!();
        println!("  EMA retention alone predicts a {:.1}× SNR advantage for d41s3 at d=1536",
                 snr_advantage(1536, D41S3_EMA.b0, D41S5_EMA.b0));
        println!("  Empirical passkey gap at d=1536: 40% vs 0% — EMA window contributes");
        println!("  but the effect is amplified by the offset set (d41s3 has multi-hop paths");
        println!("  to d=1536; d41s5's direct d=1536 offset learned no content-addressing).");
        println!();
        println!("  Conclusion: EMA window is a CONTRIBUTING factor, not the sole cause.");
        println!("  The offset set determines whether long-range paths exist at all;");
        println!("  the EMA window determines how well signal is preserved over those paths.");
    }
}
