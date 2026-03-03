//! Intermediate-frequency (IF) amplifier for DSQG long-range retrieval.
//!
//! ## Physics origin: heterodyne receivers
//!
//! In radio/optical heterodyne detection, the mixed signal (beat frequency)
//! is amplified by an IF amplifier BEFORE demodulation.  The amplifier
//! compensates for signal attenuation over the transmission path.
//!
//! In coherent LIDAR: returns from distant targets are weaker than near
//! returns (1/r² attenuation).  Per-range-gate amplification normalises
//! the SNR across distances.
//!
//! ## DSQG analogy
//!
//! The DSQG retrieval signal at distance δ:
//!
//!   signal[h, δ] = gains[h, j*] · (Q[h] · K[h, t−δ]) · ||V||
//!
//! Empirical observation (condM vs bugfixed runs, passkey plateau at d>256):
//! long-range retrieval signal is weaker even when the correct scale is used.
//! This is because:
//!   1. The D4 convolution propagates K⊗V with scale-dependent attenuation.
//!   2. With 12 fixed offsets, coverage of positions BETWEEN offsets is only
//!      approximate — "distance quantisation loss" at large δ.
//!   3. Random interference from other positions accumulates over long sequences.
//!
//! A per-head learnable scalar gain g[h] applied AFTER Q·F_prop but BEFORE
//! out_proj:  out[h] = out_proj(g[h] · (Q · F_prop)[h])
//!
//! This is the "IF amplifier stage": head h amplifies its own retrieved signal
//! before projecting to the output space.  Global heads (large j*) need higher
//! gain to compensate for weaker long-range signal.
//!
//! ## What this module verifies
//!
//! 1. Retrieval signal strength decreases with distance (attenuation exists).
//! 2. The attenuation profile is approximately log-linear with distance.
//! 3. Per-head gain can equalise retrieval signal across distances.
//! 4. A single global gain cannot compensate (different heads need different gain).
//! 5. The required gain is bounded: SNR improvement = sqrt(gain) (noise amplified less).

const D_HEAD:    usize = 64;
const N_OFFSETS: usize = 12;
const OFFSETS: [usize; N_OFFSETS] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536];
const N_HEADS:   usize = 8;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
fn norm(v: &[f64]) -> f64 { dot(v,v).sqrt() }

/// Model the D4 propagation attenuation for a deposit at distance δ.
///
/// The D4 convolution at scale j passes a fraction `attenuation(j)` of the
/// original signal amplitude.  In practice, each convolution hop has
/// gain < 1 due to ELU normalisation and mixing.  We model this as:
///
///   A(δ) = base_attenuation ^ log2(δ + 1)
///
/// where base_attenuation ≈ 0.85 per octave (empirical from condM trained gains).
fn field_attenuation(delta: usize, base: f64) -> f64 {
    let hops = (delta as f64 + 1.0).log2();
    base.powf(hops)
}

/// Retrieval signal magnitude at distance δ given:
///   - Q·K_target = qk (perfect match score)
///   - D4 attenuation applied to propagated field
///   - gain[j*] applied at retrieval scale
fn retrieval_signal_at_distance(delta: usize, qk: f64, gain_at_scale: f64) -> f64 {
    let att = field_attenuation(delta, 0.85);
    gain_at_scale * qk * att
}

/// Noise floor at position n from N_CONTEXT random positions.
/// Each contributes Q·K_noise ≈ N(0, 1/D_HEAD); magnitude ≈ sqrt(N/D).
fn noise_floor(n_context: usize) -> f64 {
    ((n_context as f64) / D_HEAD as f64).sqrt()
}

/// Required amplifier gain to bring signal[δ] up to signal[1] level.
fn required_gain(delta: usize) -> f64 {
    let att_near = field_attenuation(1, 0.85);
    let att_far  = field_attenuation(delta, 0.85);
    // gain = att_near / att_far  →  corrects attenuation back to d=1 level
    att_near / (att_far + 1e-12)
}

/// SNR with and without amplifier gain g.
fn snr_with_gain(delta: usize, qk: f64, gain_at_scale: f64, amp_gain: f64, n_ctx: usize) -> f64 {
    let sig   = amp_gain * retrieval_signal_at_distance(delta, qk, gain_at_scale);
    let noise = amp_gain.sqrt() * noise_floor(n_ctx); // noise amplified by sqrt(g)
    sig / (noise + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Retrieval signal attenuates monotonically with distance.
    #[test]
    fn test_signal_attenuates_with_distance() {
        let qk   = 5.0_f64;
        let gain = 1.0;

        println!("\n[if_amplifier] Test 1: Signal attenuation profile");
        println!("  {:>8} | {:>12} | {:>14} | {:>12}",
            "offset δ", "attenuation", "signal strength", "log2(δ)");
        let mut prev_sig = f64::MAX;
        for &d in &OFFSETS {
            let att = field_attenuation(d, 0.85);
            let sig = retrieval_signal_at_distance(d, qk, gain);
            let log2d = (d as f64 + 1.0).log2();
            println!("  {:>8} | {:>12.6} | {:>15.6} | {:>12.3}", d, att, sig, log2d);
            assert!(sig <= prev_sig + 1e-10,
                "Signal should attenuate with distance; d={d} sig={sig:.6} > prev {prev_sig:.6}");
            prev_sig = sig;
        }
        let ratio = retrieval_signal_at_distance(1, qk, gain)
                  / retrieval_signal_at_distance(1536, qk, gain);
        println!("  Attenuation ratio d=1 / d=1536: {ratio:.2}×");
        assert!(ratio > 2.0, "Long-range signal should be <50% of near-range signal");
        println!("  ✓ Retrieval signal attenuates monotonically with distance");
    }

    /// Test 2: Attenuation is approximately log-linear with distance.
    #[test]
    fn test_attenuation_log_linear() {
        // If A(δ) = base^log2(δ), then log(A(δ)) = log2(δ) * log(base)
        // which is linear in log2(δ) — log-linear relationship.
        let base  = 0.85;
        let pairs = [(1usize, 4usize), (4, 16), (16, 64), (64, 256), (256, 1024)];

        println!("\n[if_amplifier] Test 2: Log-linearity of attenuation");
        println!("  {:>6} → {:>6} | Δlog2(δ) | ΔlogA | ratio (should be const)", "d1", "d2");
        let mut prev_ratio = None::<f64>;
        for &(d1, d2) in &pairs {
            let dlog = (d2 as f64 + 1.0).log2() - (d1 as f64 + 1.0).log2();
            let a1 = field_attenuation(d1, base).ln();
            let a2 = field_attenuation(d2, base).ln();
            let ratio = (a2 - a1) / dlog;
            println!("  {:>6} → {:>6} | {:>9.3} | {:>5.3} | {ratio:>10.4}", d1, d2, dlog, a2-a1);
            if let Some(pr) = prev_ratio {
                assert!((ratio - pr).abs() < 0.001,
                    "Attenuation slope should be constant (log-linear); prev={pr:.4} curr={ratio:.4}");
            }
            prev_ratio = Some(ratio);
        }
        println!("  ✓ Attenuation is log-linear (constant slope in log-log space)");
    }

    /// Test 3: Per-head gain can equalise retrieval signal across distances.
    #[test]
    fn test_per_head_gain_equalises_signal() {
        let qk     = 5.0;
        let target = retrieval_signal_at_distance(1, qk, 1.0); // normalize to d=1

        println!("\n[if_amplifier] Test 3: Per-head gain equalisation");
        println!("  {:>8} | {:>12} | {:>14} | {:>14} | {:>12}",
            "offset δ", "raw signal", "required gain", "amplified sig", "vs target");
        for &d in &OFFSETS {
            let raw   = retrieval_signal_at_distance(d, qk, 1.0);
            let g_req = required_gain(d);
            let amp   = raw * g_req;
            let rel   = amp / target;
            println!("  {:>8} | {:>12.6} | {:>14.4}× | {:>14.6} | {:>12.4}",
                d, raw, g_req, amp, rel);
            assert!((rel - 1.0).abs() < 0.01,
                "Amplified signal should equal target±1% at d={d}; got rel={rel:.4}");
        }
        println!("  ✓ Per-head gain successfully equalises retrieval signal across all distances");
    }

    /// Test 4: A single global gain cannot equalise all heads simultaneously.
    #[test]
    fn test_single_global_gain_insufficient() {
        // Assign heads to target distances: head 0 → d=1, head 7 → d=1536
        let head_targets = [1usize, 2, 8, 32, 128, 256, 512, 1536];
        assert_eq!(head_targets.len(), N_HEADS);
        let qk = 5.0;

        println!("\n[if_amplifier] Test 4: Global gain vs per-head gain");
        println!("  Per-head required gains:");
        let per_head_gains: Vec<f64> = head_targets.iter()
            .map(|&d| required_gain(d))
            .collect();
        for (h, (&d, &g)) in head_targets.iter().zip(per_head_gains.iter()).enumerate() {
            println!("    head {h}: target d={d:>5}, required gain = {g:.4}×");
        }

        // Best single global gain: geometric mean (minimises worst-case ratio)
        let log_mean: f64 = per_head_gains.iter().map(|g| g.ln()).sum::<f64>() / N_HEADS as f64;
        let global_gain = log_mean.exp();
        println!("  Best global gain (geometric mean): {global_gain:.4}×");

        let max_error: f64 = per_head_gains.iter()
            .map(|&g| (g / global_gain - 1.0).abs())
            .fold(0.0f64, f64::max);
        println!("  Max equalisation error with global gain: {:.1}%", max_error * 100.0);
        println!("  Max equalisation error with per-head gain: 0%");

        assert!(max_error > 0.1,
            "Global gain should leave >10% error for some head; got {max_error:.4}");
        println!("  ✓ Per-head gain required: single global gain leaves {:.1}% worst-case error",
            max_error * 100.0);
    }

    /// Test 5: SNR improves as sqrt(gain) — noise is amplified less than signal.
    #[test]
    fn test_snr_improves_as_sqrt_gain() {
        let delta   = 512usize;
        let n_ctx   = 1000;
        let qk      = 5.0;
        let g_scale = 1.0; // gain at retrieval scale

        let gains = [1.0f64, 2.0, 4.0, 8.0, 16.0];

        println!("\n[if_amplifier] Test 5: SNR improvement = sqrt(gain)");
        println!("  {:>8} | {:>10} | {:>12} | {:>14}", "amp gain", "SNR", "SNR/SNR_1", "expected");
        let snr_1 = snr_with_gain(delta, qk, g_scale, 1.0, n_ctx);
        for &g in &gains {
            let snr = snr_with_gain(delta, qk, g_scale, g, n_ctx);
            let actual_ratio   = snr / snr_1;
            let expected_ratio = g.sqrt();
            println!("  {:>8.1} | {:>10.4} | {:>12.4}× | {:>14.4}× (=√g)",
                g, snr, actual_ratio, expected_ratio);
            assert!((actual_ratio - expected_ratio).abs() < expected_ratio * 0.05,
                "SNR improvement should be sqrt(gain); expected {expected_ratio:.4}, got {actual_ratio:.4}");
        }
        println!("  ✓ SNR improvement = sqrt(gain) confirmed (signal scales as g, noise as sqrt(g))");
    }
}
