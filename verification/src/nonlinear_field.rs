//! Verification of the minimal KdV nonlinear field extension.
//!
//! ## Architectural proposal
//!
//! **Current (C_causal):**
//! ```text
//! h_out = causal_wavelet_conv(h, kernel)
//! ```
//!
//! **Proposed (V5-KdV):**
//! ```text
//! h_conv = causal_wavelet_conv(h, kernel)
//! delta  = h_conv[g] − h_conv[g−1]          (causal first-order difference)
//! h_out  = h_conv + α · h_conv · delta       (α = learned per-head scalar, init 0)
//! ```
//!
//! `α = 0` recovers C_causal exactly at initialisation — no regression risk.
//!
//! ## What this module verifies
//!
//! 1. **Causality**: `h_out[g]` depends only on `h[i]` for `i ≤ g`.
//!    Tested via Jacobian sparsity: impulse at position g₀ must produce
//!    zero output at all positions g < g₀.
//!
//! 2. **Minimum-phase energy concentration**: Daubechies-4 filter has
//!    93% of its energy in the first 2 of 4 taps.  A Morlet wavelet at
//!    the same scale spreads its energy over a much wider window.
//!    Tested via cumulative energy curve.
//!
//! 3. **KdV nonlinearity slows dispersive amplitude decay**:
//!    a Gaussian pulse convolved with a spreading kernel, then stabilised
//!    with the KdV term, retains higher peak amplitude than without the
//!    nonlinear term.  The effect is monotone in small positive α.
//!
//! ## Connection to soliton.rs
//!
//! `soliton.rs` proves the continuous-KdV principle.  This module verifies
//! the *discrete* field-level operation that would actually appear in the
//! Python training code.

// ─── Discrete field operations ────────────────────────────────────────────────

/// Causal first-order difference: `delta[g] = h[g] − h[g−1]`, `delta[0] = 0`.
///
/// This is the discrete analogue of `∂_x u` in the KdV nonlinear term.
/// Causality is structural: position g is computed only from positions ≤ g.
pub fn causal_diff(h: &[f32]) -> Vec<f32> {
    let mut d = vec![0.0f32; h.len()];
    for g in 1..h.len() {
        d[g] = h[g] - h[g - 1];
    }
    d
}

/// Apply one KdV nonlinear field update step:
///
///   `h_out[g] = h[g] + α · h[g] · delta[g]`
///
/// where `delta = causal_diff(h)`.
///
/// At α = 0 this is the identity.  At small positive α it stabilises forward-
/// propagating structures.  At large α it may amplify and destabilise.
pub fn kdv_field_step(h: &[f32], alpha: f32) -> Vec<f32> {
    let delta = causal_diff(h);
    h.iter().zip(&delta)
        .map(|(&hi, &di)| hi + alpha * hi * di)
        .collect()
}

/// Causal linear convolution of `h` with `kernel` (FIR, length K).
///
/// `kernel[0]` is the most-recent tap; `kernel[K-1]` is the oldest.
/// Position g uses only `h[g], h[g−1], …, h[g−K+1]` (zero-padded for g < K).
pub fn causal_conv(h: &[f32], kernel: &[f32]) -> Vec<f32> {
    let n = h.len();
    (0..n).map(|g| {
        kernel.iter().enumerate().map(|(j, &kj)| {
            if g >= j { h[g - j] * kj } else { 0.0 }
        }).sum()
    }).collect()
}

/// Peak (max) value and its index.
pub fn peak(h: &[f32]) -> (f32, usize) {
    h.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (v, i))
        .unwrap()
}

/// Mass-weighted centroid position.
///
/// More robust than peak position for spread or multi-modal distributions.
/// Uses absolute values so negative excursions (DB4 has a negative tap) count.
pub fn centroid(h: &[f32]) -> f32 {
    let total = h.iter().map(|x| x.abs()).sum::<f32>().max(1e-9);
    h.iter().enumerate()
        .map(|(i, x)| i as f32 * x.abs())
        .sum::<f32>() / total
}

// ─── Kernel definitions ───────────────────────────────────────────────────────

/// Daubechies-4 (db4) minimum-phase low-pass filter coefficients.
///
/// 4-tap filter; all z-plane zeros are strictly inside the unit disk.
/// Consequence: energy is maximally front-loaded (recent context dominant).
///
/// Normalised so Σ k[i]² = 1.
pub const DB4: [f32; 4] = [0.4829629, 0.8365163, 0.2241439, -0.1294095];

/// Causal Morlet wavelet evaluated at integer taps t = 0, 1, …, K−1.
///
/// k(t) = exp(−t²/(2σ²)) · cos(2π·t/σ)
/// Truncated to `k_len` taps then L2-normalised.
pub fn morlet_kernel(sigma: f32, k_len: usize) -> Vec<f32> {
    let mut k: Vec<f32> = (0..k_len)
        .map(|t| {
            let tf = t as f32;
            (-tf * tf / (2.0 * sigma * sigma)).exp() * (2.0 * std::f32::consts::PI * tf / sigma).cos()
        })
        .collect();
    let norm = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    k.iter_mut().for_each(|x| *x /= norm);
    k
}

/// Causal Gaussian spreading kernel: k(t) = exp(−t²/(2σ²)), t = 0…K−1.
///
/// Normalised to **sum to 1** (probability kernel / weighted average).
/// This makes it amplitude-reducing for any peaked input: the kernel
/// blurs the peak over its neighbours, lowering the maximum.
/// Use this (not L2-normalised) for "spreading" tests.
pub fn gaussian_kernel(sigma: f32, k_len: usize) -> Vec<f32> {
    let mut k: Vec<f32> = (0..k_len)
        .map(|t| {
            let tf = t as f32;
            (-tf * tf / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum = k.iter().sum::<f32>().max(1e-9);
    k.iter_mut().for_each(|x| *x /= sum);
    k
}

// ─── Verification tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const G: usize = 64; // field length

    fn gaussian_pulse(center: usize, width: f32) -> Vec<f32> {
        (0..G).map(|i| {
            let d = (i as f32 - center as f32) / width;
            (-0.5 * d * d).exp()
        }).collect()
    }

    // ── Property 1: causal_diff is causal ────────────────────────────────────

    /// causal_diff[0] must always be 0 (no prior position exists).
    #[test]
    fn causal_diff_first_element_is_zero() {
        let h: Vec<f32> = (0..G).map(|i| i as f32 * 0.1 + 1.0).collect();
        let d = causal_diff(&h);
        assert_eq!(d[0], 0.0, "causal_diff[0] must be 0 (no prior position)");
    }

    /// For an impulse at position g₀, causal_diff must be 0 for all g < g₀.
    #[test]
    fn causal_diff_does_not_look_ahead() {
        let g0 = 20usize;
        let mut h = vec![0.0f32; G];
        h[g0] = 1.0;
        let d = causal_diff(&h);
        for g in 0..g0 {
            assert_eq!(d[g], 0.0, "causal_diff[{g}] must be 0 for impulse at {g0}");
        }
        // At g0 itself: delta = h[g0] - h[g0-1] = 1.0 - 0.0 = 1.0
        assert!((d[g0] - 1.0).abs() < 1e-6, "causal_diff[g0] = {}", d[g0]);
        // At g0+1: delta = h[g0+1] - h[g0] = 0.0 - 1.0 = -1.0
        assert!((d[g0 + 1] + 1.0).abs() < 1e-6, "causal_diff[g0+1] = {}", d[g0 + 1]);
    }

    // ── Property 2: kdv_field_step is causal ─────────────────────────────────

    /// Structural causality proof via Jacobian sparsity:
    /// if h[j] = 0 for all j ≤ g−1, then h_out[g'] = 0 for all g' ≤ g−1.
    /// We test this with an impulse at g₀: h_out[g] must be 0 for g < g₀.
    #[test]
    fn kdv_field_step_is_causal() {
        let g0  = 30usize;
        let alpha = 0.1;
        let mut h = vec![0.0f32; G];
        h[g0] = 1.0;
        let h_out = kdv_field_step(&h, alpha);
        for g in 0..g0 {
            assert!(
                h_out[g].abs() < 1e-9,
                "kdv_field_step output at g={g} = {:.2e} must be 0 \
                 (impulse at g₀={g0}, α={alpha})",
                h_out[g],
            );
        }
        // Output at g₀ must be nonzero (the impulse is there)
        assert!(h_out[g0].abs() > 1e-6, "Output at g₀={g0} should be nonzero");
    }

    /// α=0 must give identity: h_out[g] = h[g] for all g.
    #[test]
    fn kdv_field_step_alpha_zero_is_identity() {
        let h = gaussian_pulse(G / 3, 3.0);
        let h_out = kdv_field_step(&h, 0.0);
        for (a, b) in h.iter().zip(&h_out) {
            assert!((a - b).abs() < 1e-9, "α=0 step must be identity");
        }
    }

    // ── Property 3: minimum-phase energy concentration ───────────────────────

    /// Daubechies-4 reaches 80% of its cumulative energy in the first 2 of 4 taps.
    /// A Morlet kernel at σ=4 (same approximate bandwidth) needs more taps to
    /// reach the same fraction.  This quantifies the "recent-context bias" of
    /// minimum-phase filters.
    #[test]
    fn db4_energy_more_front_loaded_than_morlet() {
        let db4 = DB4;
        let morlet = morlet_kernel(4.0, 24); // 24 taps to capture Morlet envelope

        // Cumulative energy fraction: taps 0..k / total
        let total_db4: f32 = db4.iter().map(|x| x * x).sum();
        let total_morlet: f32 = morlet.iter().map(|x| x * x).sum();

        // Taps needed to reach 80% of energy
        let taps_for_80pct = |kernel: &[f32], total: f32| -> usize {
            let mut cumsum = 0.0f32;
            for (i, &k) in kernel.iter().enumerate() {
                cumsum += k * k;
                if cumsum / total >= 0.80 {
                    return i + 1; // number of taps (1-indexed)
                }
            }
            kernel.len()
        };

        let db4_taps   = taps_for_80pct(&db4, total_db4);
        let morlet_taps = taps_for_80pct(&morlet, total_morlet);

        assert!(
            db4_taps < morlet_taps,
            "DB4 reaches 80% energy in {db4_taps} taps; Morlet(σ=4) needs {morlet_taps} taps. \
             DB4 should require fewer (minimum-phase = front-loaded energy)"
        );
    }

    /// DB4 specifically: 93%+ of energy in the first 2 of 4 taps.
    /// This quantifies how strongly minimum-phase the db4 filter is.
    #[test]
    fn db4_energy_concentrated_in_early_taps() {
        let db4 = DB4;
        let total: f32 = db4.iter().map(|x| x * x).sum();
        let first_two: f32 = db4[0] * db4[0] + db4[1] * db4[1];
        let fraction = first_two / total;
        assert!(
            fraction > 0.90,
            "DB4 first-2-tap energy fraction = {:.3} should be > 0.90 (minimum-phase property)",
            fraction
        );
    }

    // ── Property 4: KdV nonlinearity slows dispersive spreading ──────────────

    /// A Gaussian pulse convolved with a spreading (Gaussian) kernel loses
    /// amplitude — the pulse broadens and its peak decreases.
    /// Adding the KdV nonlinear update after each convolution step partially
    /// counteracts this: peak amplitude is higher with α > 0 than with α = 0.
    ///
    /// This is the discrete analogue of the soliton.rs finding:
    /// nonlinear > linear for amplitude preservation.
    #[test]
    fn kdv_nonlinearity_slows_dispersive_amplitude_decay() {
        const STEPS: usize = 8;
        let kernel = gaussian_kernel(2.0, 8); // spreading kernel, σ=2, 8 taps
        let h0 = gaussian_pulse(G / 3, 3.0);
        let (amp0, _) = peak(&h0);

        // Evolve under spreading kernel only (α=0)
        let mut h_linear = h0.clone();
        for _ in 0..STEPS {
            h_linear = causal_conv(&h_linear, &kernel);
        }
        let (amp_linear, _) = peak(&h_linear);

        // Evolve under spreading kernel + KdV stabilisation (α=0.3)
        let alpha = 0.3;
        let mut h_kdv = h0.clone();
        for _ in 0..STEPS {
            h_kdv = causal_conv(&h_kdv, &kernel);
            h_kdv = kdv_field_step(&h_kdv, alpha);
        }
        let (amp_kdv, _) = peak(&h_kdv);

        // Both should decay relative to initial
        assert!(amp_linear < amp0, "Linear convolution should reduce amplitude");
        assert!(amp_kdv    < amp0, "KdV + convolution also reduces amplitude (just less)");

        // KdV should preserve amplitude better than linear
        assert!(
            amp_kdv > amp_linear,
            "KdV amplitude {amp_kdv:.4} should exceed linear amplitude {amp_linear:.4} \
             after {STEPS} spreading + stabilisation steps"
        );
    }

    /// Monotonicity check: increasing α (in the small-positive range) should
    /// monotonically increase amplitude preservation, up to an instability point.
    /// This establishes that α is a meaningful control parameter.
    #[test]
    fn kdv_alpha_monotone_in_small_range() {
        const STEPS: usize = 5;
        let kernel = gaussian_kernel(2.0, 6);
        let h0 = gaussian_pulse(G / 3, 3.0);

        let alphas = [0.0f32, 0.05, 0.10, 0.20, 0.40];
        let mut prev_amp = 0.0f32;

        for &alpha in &alphas {
            let mut h = h0.clone();
            for _ in 0..STEPS {
                h = causal_conv(&h, &kernel);
                h = kdv_field_step(&h, alpha);
            }
            let (amp, _) = peak(&h);
            assert!(
                amp >= prev_amp - 1e-4,
                "α={alpha:.2} amplitude {amp:.4} should be ≥ previous α amplitude {prev_amp:.4} \
                 (amplitude should increase with α in this small range)"
            );
            prev_amp = amp;
        }
    }

    // ── Property 5: causal convolution is causal ─────────────────────────────

    /// Baseline: verify causal_conv doesn't look ahead.
    /// An impulse at g₀ should produce no output at positions g < g₀.
    #[test]
    fn causal_conv_is_causal() {
        let g0 = 15usize;
        let mut h = vec![0.0f32; G];
        h[g0] = 1.0;
        let out = causal_conv(&h, &DB4);
        for g in 0..g0 {
            assert!(
                out[g].abs() < 1e-9,
                "causal_conv output at g={g} = {:.2e} must be 0 (impulse at {g0})",
                out[g],
            );
        }
    }

    // ── Diagnostic print ─────────────────────────────────────────────────────

    #[test]
    fn nonlinear_field_print_summary() {
        const STEPS: usize = 8;
        let kernel = gaussian_kernel(2.0, 8);
        let h0 = gaussian_pulse(G / 3, 3.0);
        let (amp0, _) = peak(&h0);

        println!("\nNonlinear field update — G={G}, spreading σ=2, {STEPS} steps");
        println!("Initial amplitude: {amp0:.4}");

        for &alpha in &[0.0f32, 0.10, 0.20, 0.30, 0.50] {
            let mut h = h0.clone();
            for _ in 0..STEPS {
                h = causal_conv(&h, &kernel);
                h = kdv_field_step(&h, alpha);
            }
            let (amp, pos) = peak(&h);
            let dx = 1.0f32;
            println!("  α={alpha:.2}:  peak {amp:.4}  pos {:.1}  retained {:.1}%",
                     pos as f32 * dx,
                     amp / amp0 * 100.0);
        }

        // DB4 vs Morlet energy concentration
        let morlet = morlet_kernel(4.0, 24);
        let total_db4: f32 = DB4.iter().map(|x| x * x).sum();
        let total_morlet: f32 = morlet.iter().map(|x| x * x).sum();

        let first_two_db4 = (DB4[0] * DB4[0] + DB4[1] * DB4[1]) / total_db4;
        let first_two_m = morlet.iter().take(2).map(|x| x * x).sum::<f32>() / total_morlet;
        println!("\nEnergy concentration (first 2 taps / total):");
        println!("  DB4 (minimum-phase): {:.1}%", first_two_db4 * 100.0);
        println!("  Morlet (σ=4):        {:.1}%", first_two_m * 100.0);
    }

    /// Morlet + KdV instability diagnostic — fine sweep + per-step breakdown.
    ///
    /// The broad sweep showed a non-monotonic minimum at α≈0.20 for the Morlet
    /// kernel.  This test:
    ///   (a) Does a fine sweep [0.00, 0.50] in steps of 0.02 to locate the minimum precisely
    ///   (b) Shows step-by-step amplitude for α=0.40 to reveal the instability mechanism
    ///   (c) Compares drift AND amplitude decay for each α — are they correlated?
    ///
    /// Hypothesis: at α > 0.20, the broad Morlet support (wide energy spread)
    /// causes the KdV nonlinear term to amplify side-lobes rather than the peak,
    /// increasing effective centroid drift while also degrading amplitude.
    #[test]
    fn morlet_kdv_instability_diagnostic() {
        const STEPS: usize = 8;
        const CENTER: usize = G / 3;

        let h0     = gaussian_pulse(CENTER, 3.0);
        let c0     = centroid(&h0);
        let (a0, _) = peak(&h0);
        let morlet = morlet_kernel(4.0, 16);

        println!("\nMorlet + KdV instability — fine sweep α ∈ [0.00, 0.50]");
        println!("Pulse center={}, initial centroid={:.2}, initial peak={:.4}", CENTER, c0, a0);
        println!("\n{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
                 "α", "centroid", "drift", "peak amp", "amp retained%");

        let alphas: Vec<f32> = (0..=25).map(|i| i as f32 * 0.02).collect();
        let mut min_drift = f32::MAX;
        let mut min_alpha = 0.0f32;

        for &alpha in &alphas {
            let mut h = h0.clone();
            for _ in 0..STEPS {
                h = causal_conv(&h, &morlet);
                h = kdv_field_step(&h, alpha);
            }
            let c    = centroid(&h);
            let drift = c - c0;
            let (amp, _) = peak(&h);
            let retained = amp / a0 * 100.0;
            println!("{:>6.2}  {:>12.4}  {:>12.4}  {:>12.4}  {:>12.1}",
                     alpha, c, drift, amp, retained);
            if drift.abs() < min_drift {
                min_drift = drift.abs();
                min_alpha = alpha;
            }
        }
        println!("\nMinimum |drift| = {:.4} at α = {:.2}", min_drift, min_alpha);

        // ── Per-step breakdown at α=0.40 (past instability onset) ────────────
        println!("\nPer-step breakdown: α=0.40 (Morlet — past instability onset)");
        println!("{:>6}  {:>12}  {:>12}  {:>12}", "step", "centroid", "drift", "peak amp");

        let mut h_inst = h0.clone();
        for step in 1..=STEPS {
            h_inst = causal_conv(&h_inst, &morlet);
            h_inst = kdv_field_step(&h_inst, 0.40);
            let c = centroid(&h_inst);
            let (amp, _) = peak(&h_inst);
            println!("{:>6}  {:>12.4}  {:>12.4}  {:>12.4}", step, c, c - c0, amp);
        }

        // ── Same breakdown at α=0.20 (near minimum) ──────────────────────────
        println!("\nPer-step breakdown: α=0.20 (Morlet — near minimum drift)");
        println!("{:>6}  {:>12}  {:>12}  {:>12}", "step", "centroid", "drift", "peak amp");

        let mut h_opt = h0.clone();
        for step in 1..=STEPS {
            h_opt = causal_conv(&h_opt, &morlet);
            h_opt = kdv_field_step(&h_opt, 0.20);
            let c = centroid(&h_opt);
            let (amp, _) = peak(&h_opt);
            println!("{:>6}  {:>12.4}  {:>12.4}  {:>12.4}", step, c, c - c0, amp);
        }
    }

    /// α sweep diagnostic: find zero-drift point for each kernel.
    ///
    /// Causal convolution shifts the pulse centroid rightward (toward later positions).
    /// KdV nonlinearity shifts it leftward.  There exists an α* where the two effects
    /// cancel and the centroid stays near its initial position.
    ///
    /// We sweep α ∈ [0.0, 1.0] and report centroid drift under three kernels:
    ///   - Gaussian spreading (baseline, σ=2, 8 taps)
    ///   - DB4 minimum-phase (proposed V5-KdV basis)
    ///   - Morlet causal (current C_causal baseline)
    ///
    /// The kernel that requires the *smallest* α to reach zero-drift interacts
    /// most favourably with the KdV nonlinear term.
    #[test]
    fn alpha_sweep_zero_drift() {
        const STEPS: usize = 8;
        const CENTER: usize = G / 3;

        let h0 = gaussian_pulse(CENTER, 3.0);
        let c0 = centroid(&h0);

        let gaussian_k = gaussian_kernel(2.0, 8);
        let morlet_k   = morlet_kernel(4.0, 16);
        // DB4 as a sum-to-1 kernel (amplitude-normalised for fair drift comparison)
        let db4_sum: f32 = DB4.iter().sum::<f32>().abs().max(1e-9);
        let db4_norm: Vec<f32> = DB4.iter().map(|x| x / db4_sum).collect();

        let kernels: [(&str, &[f32]); 3] = [
            ("Gaussian (σ=2)", &gaussian_k),
            ("DB4 (norm)",     &db4_norm),
            ("Morlet (σ=4)",   &morlet_k),
        ];

        let alphas: Vec<f32> = (0..=20).map(|i| i as f32 * 0.05).collect(); // 0.00 to 1.00

        println!("\nα-sweep: centroid drift after {} convolution+KdV steps, pulse center={}",
                 STEPS, CENTER);
        println!("α=0 baseline drifts (centroid − {:.1}):", c0);

        for (name, kernel) in &kernels {
            // Drift at α=0
            let mut h = h0.clone();
            for _ in 0..STEPS { h = causal_conv(&h, kernel); }
            let drift0 = centroid(&h) - c0;
            println!("  {:18}  {:.3}", name, drift0);
        }

        println!("\n{:>6}  {:>20}  {:>20}  {:>20}",
                 "α", "Gaussian drift", "DB4 drift", "Morlet drift");

        // Track zero-crossings per kernel
        let mut prev_drifts = [f32::NAN; 3];
        let mut zero_alpha  = [f32::NAN; 3];

        for &alpha in &alphas {
            let mut row = [0.0f32; 3];
            for (ki, (_, kernel)) in kernels.iter().enumerate() {
                let mut h = h0.clone();
                for _ in 0..STEPS {
                    h = causal_conv(&h, kernel);
                    h = kdv_field_step(&h, alpha);
                }
                row[ki] = centroid(&h) - c0;
            }
            println!("{:>6.2}  {:>20.4}  {:>20.4}  {:>20.4}",
                     alpha, row[0], row[1], row[2]);

            // Detect zero-crossing (sign change)
            for ki in 0..3 {
                if zero_alpha[ki].is_nan() && !prev_drifts[ki].is_nan() {
                    if prev_drifts[ki] * row[ki] <= 0.0 {
                        zero_alpha[ki] = alpha;
                    }
                }
                prev_drifts[ki] = row[ki];
            }
        }

        println!("\nZero-drift α* (first crossing):");
        let names = ["Gaussian", "DB4", "Morlet"];
        for (ki, name) in names.iter().enumerate() {
            if zero_alpha[ki].is_nan() {
                println!("  {:8}  not found in [0.0, 1.0]", name);
            } else {
                println!("  {:8}  α* ≈ {:.2}", name, zero_alpha[ki]);
            }
        }

        // Sanity: at α=0 all kernels should drift positively (rightward)
        let mut h_g = h0.clone();
        for _ in 0..STEPS { h_g = causal_conv(&h_g, &gaussian_k); }
        let drift_g0 = centroid(&h_g) - c0;
        assert!(
            drift_g0 > 0.0,
            "Gaussian kernel at α=0 should drift rightward; got drift={:.4}", drift_g0
        );
    }
}
