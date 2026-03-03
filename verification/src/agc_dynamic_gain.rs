//! Automatic Gain Control (AGC) for dynamic IF amplifier in DSQG.
//!
//! ## Physics origin
//!
//! In radio receivers, AGC is a closed-loop feedback circuit that adjusts
//! amplifier gain in response to measured output signal level:
//!
//!   gain_t = target_level / ||signal_t||
//!   output_t = gain_t * signal_t
//!   → ||output_t|| = target_level  (constant, regardless of input strength)
//!
//! AGC is essential in coherent receivers because the signal amplitude at the
//! IF stage varies enormously with target distance, channel conditions, and
//! frequency fading.  Without AGC, strong signals saturate the detector
//! and weak signals fall below the noise floor.
//!
//! ## DSQG analogy
//!
//! condU implemented a static IF amplifier: per-head learnable scalars
//!   if_gain[h] ∈ {0.93 … 1.03}  (learned, fixed at inference time)
//!
//! The static gain learned the *mean* attenuation profile from training data.
//! It cannot adapt to:
//!   - Varying sequence lengths at inference time
//!   - Domain shift (different text has different SNR profile)
//!   - Within-sequence variation (a paragraph may shift register)
//!
//! AGC makes the gain dynamic:
//!   effective_gain[h, n] = target_level / ||attn_out[h, n]||
//!
//! No new trainable parameters required.  The target_level can be a fixed
//! scalar (e.g., 1.0) or a learned per-head target (one scalar per head).
//!
//! ## What this module verifies
//!
//! 1. Static gain: output amplitude varies proportionally with input strength.
//! 2. AGC: output amplitude is constant (= target_level) regardless of input.
//! 3. AGC preserves direction: cosine similarity of output with input is 1.0.
//! 4. AGC compresses strong signals and amplifies weak ones (monotone inverse).
//! 5. Dynamic range: AGC handles signals across the range observed in condU
//!    (IF gains 0.93–1.03 imply ~10% amplitude variation; AGC handles 10×+ range).
//! 6. Static gain achieves equal output only when all inputs have equal amplitude
//!    (necessary condition for static gain being optimal).

const D_HEAD: usize = 32;
const N_HEADS: usize = 8;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
fn norm(v: &[f64]) -> f64 { dot(v,v).sqrt().max(1e-12) }

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    dot(a,b) / (norm(a) * norm(b))
}

/// Static gain: out = gain_scalar * signal (fixed scalar, same for all inputs)
fn static_gain(signal: &[f64], gain: f64) -> Vec<f64> {
    signal.iter().map(|x| x * gain).collect()
}

/// AGC: out = (target_level / ||signal||) * signal
/// → ||out|| = target_level exactly
pub fn agc(signal: &[f64], target_level: f64) -> Vec<f64> {
    let n = norm(signal);
    let gain = target_level / n;
    signal.iter().map(|x| x * gain).collect()
}

/// Generate signal with specified L2 amplitude in a fixed direction.
fn signal_at_amplitude(amplitude: f64, direction: &[f64]) -> Vec<f64> {
    let n = norm(direction);
    direction.iter().map(|x| x / n * amplitude).collect()
}

fn pseudo_unit_vec(d: usize, seed: u64) -> Vec<f64> {
    let mut lcg = seed;
    let raw: Vec<f64> = (0..d).map(|_| {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
    }).collect();
    let n = norm(&raw);
    raw.iter().map(|x| x/n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Property 1: Static gain does NOT normalise ────────────────────────────

    /// Static gain simply scales amplitude: if input has amplitude A,
    /// output has amplitude g*A.  Not constant unless all inputs equal A.
    #[test]
    fn static_gain_does_not_normalise() {
        let direction = pseudo_unit_vec(D_HEAD, 1);
        let amplitudes = [0.5_f64, 1.0, 2.0, 5.0, 10.0];
        let g = 1.0; // unit static gain

        let output_norms: Vec<f64> = amplitudes.iter().map(|&a| {
            let sig = signal_at_amplitude(a, &direction);
            let out = static_gain(&sig, g);
            norm(&out)
        }).collect();

        // Output norms must vary — they equal g*amplitude
        let first = output_norms[0];
        assert!(
            output_norms.iter().any(|&n| (n - first).abs() > 0.1),
            "Static gain must produce varying output amplitudes for varying inputs"
        );
    }

    // ── Property 2: AGC normalises to target_level ────────────────────────────

    /// Regardless of input amplitude, AGC output must have exactly target_level.
    #[test]
    fn agc_normalises_to_target_level() {
        let target = 1.0;
        let direction = pseudo_unit_vec(D_HEAD, 2);
        let amplitudes = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];

        for &amp in &amplitudes {
            let sig = signal_at_amplitude(amp, &direction);
            let out = agc(&sig, target);
            let out_norm = norm(&out);
            assert!(
                (out_norm - target).abs() < 1e-9,
                "AGC output norm = {out_norm:.8} must equal target {target} (input amp={amp})"
            );
        }
    }

    // ── Property 3: AGC preserves direction ───────────────────────────────────

    /// AGC must NOT change the direction of the signal vector.
    /// cos_sim(AGC(signal), signal) = 1.0 exactly.
    #[test]
    fn agc_preserves_direction() {
        for seed in [10u64, 20, 30, 40, 50] {
            let direction = pseudo_unit_vec(D_HEAD, seed);
            for &amp in &[0.1_f64, 1.0, 5.0] {
                let sig = signal_at_amplitude(amp, &direction);
                let out = agc(&sig, 1.0);
                let sim = cosine_similarity(&sig, &out);
                assert!(
                    (sim - 1.0).abs() < 1e-9,
                    "AGC must preserve direction: cos_sim={sim:.8} (amp={amp}, seed={seed})"
                );
            }
        }
    }

    // ── Property 4: AGC is monotone inverse ───────────────────────────────────

    /// AGC gain = target / ||signal||.
    /// As input amplitude increases, effective gain decreases (compression).
    /// As input amplitude decreases, effective gain increases (amplification).
    #[test]
    fn agc_gain_is_monotone_inverse_of_amplitude() {
        let target    = 1.0;
        let direction = pseudo_unit_vec(D_HEAD, 7);
        let amplitudes = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];

        let gains: Vec<f64> = amplitudes.iter().map(|&amp| {
            let sig = signal_at_amplitude(amp, &direction);
            let out = agc(&sig, target);
            norm(&out) / amp // effective gain = output_amp / input_amp = target / amp
        }).collect();

        // Gains must be strictly decreasing as amplitude increases
        for w in gains.windows(2) {
            assert!(
                w[0] > w[1],
                "AGC gain must decrease with increasing amplitude: {:.4} -> {:.4}",
                w[0], w[1]
            );
        }
    }

    // ── Property 5: AGC dynamic range covers condU IF variation ──────────────

    /// condU IF gains range from 0.93 (local heads) to 1.03 (global heads).
    /// This 10% variation reflects signal amplitude ratio of 1.03/0.93 ≈ 1.108.
    /// AGC should handle a much wider range — at least 10:1 amplitude variation —
    /// while maintaining output norm within 1% of target.
    #[test]
    fn agc_handles_ten_to_one_dynamic_range() {
        let target    = 1.0;
        let direction = pseudo_unit_vec(D_HEAD, 99);
        let amplitudes = [0.1_f64, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]; // 100:1 range

        for &amp in &amplitudes {
            let sig = signal_at_amplitude(amp, &direction);
            let out = agc(&sig, target);
            let err = (norm(&out) - target).abs() / target;
            assert!(
                err < 0.001,
                "AGC must maintain output within 0.1% of target for amp={amp}; err={err:.6}"
            );
        }
    }

    // ── Property 6: Static gain optimal only when all inputs equal amplitude ──

    /// For a static scalar gain g* = target_level / mean(||signal||),
    /// the output amplitude is approximately target only when input amplitudes
    /// are all equal.  With varying amplitudes, the mean-squared error of
    /// static gain EXCEEDS that of AGC.
    #[test]
    fn static_gain_suboptimal_for_variable_amplitude_inputs() {
        let target    = 1.0_f64;
        let direction = pseudo_unit_vec(D_HEAD, 55);

        // Simulate N_HEADS heads with the IF gain profile seen in condU:
        // global heads (0-2) have gain ~1.03; local heads (5-7) have gain ~0.93
        // We model this as varying signal amplitudes: global=1.03, local=0.93
        let signal_amps = [1.03_f64, 1.02, 1.03, 0.99, 1.01, 0.95, 0.95, 0.93];
        assert_eq!(signal_amps.len(), N_HEADS);

        let signals: Vec<Vec<f64>> = signal_amps.iter().map(|&a| {
            signal_at_amplitude(a, &direction)
        }).collect();

        // Optimal static gain: target / mean(amplitude)
        let mean_amp: f64 = signal_amps.iter().sum::<f64>() / N_HEADS as f64;
        let g_static = target / mean_amp;

        // MSE of static gain outputs from target
        let mse_static: f64 = signals.iter().map(|sig| {
            let out_norm = norm(&static_gain(sig, g_static));
            (out_norm - target).powi(2)
        }).sum::<f64>() / N_HEADS as f64;

        // MSE of AGC outputs from target (must be 0)
        let mse_agc: f64 = signals.iter().map(|sig| {
            let out_norm = norm(&agc(sig, target));
            (out_norm - target).powi(2)
        }).sum::<f64>() / N_HEADS as f64;

        assert!(
            mse_agc < 1e-18,
            "AGC MSE must be ~0; got {mse_agc:.2e}"
        );
        assert!(
            mse_static > mse_agc,
            "Static gain MSE {mse_static:.6} must exceed AGC MSE {mse_agc:.2e} \
             for variable-amplitude inputs"
        );

        println!("\nAGC vs static gain (condU IF amplitude profile: {signal_amps:?})");
        println!("  Mean amplitude: {mean_amp:.4}  → static gain g*={g_static:.4}");
        println!("  Static gain MSE from target: {mse_static:.8}");
        println!("  AGC MSE from target:         {mse_agc:.2e}");
        println!("  AGC improvement: {:.1}x", mse_static / mse_agc.max(1e-20));
    }

    #[test]
    fn agc_print_summary() {
        let direction = pseudo_unit_vec(D_HEAD, 11);
        let target    = 1.0;
        println!("\nAGC normalisation summary (target=1.0, D={D_HEAD}):");
        println!("{:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
                 "input_amp", "static_out", "agc_out", "static_err%", "agc_err%");
        for &amp in &[0.1_f64, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0] {
            let sig = signal_at_amplitude(amp, &direction);
            let static_out = norm(&static_gain(&sig, 1.0));
            let agc_out    = norm(&agc(&sig, target));
            println!("{:>8.2}  {:>12.4}  {:>12.4}  {:>11.2}%  {:>11.4}%",
                     amp, static_out, agc_out,
                     (static_out - target).abs() / target * 100.0,
                     (agc_out - target).abs() / target * 100.0);
        }
    }
}
