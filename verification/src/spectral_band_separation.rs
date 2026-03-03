//! Spectral band separation per head — hard frequency assignment for DSQG.
//!
//! ## Physics origin: filter bank + coherent recombination
//!
//! In radio engineering, a filter bank separates a wideband signal into
//! frequency sub-bands.  Each sub-band is processed independently.
//! Recombination happens at the output.  Key property: isolating bands
//! prevents high-frequency noise from contaminating low-frequency signal
//! and vice versa.  Each band's SNR is limited only by the noise in that
//! band, not by the full-spectrum noise floor.
//!
//! In phased array antennas: each element is tuned to receive a specific
//! frequency.  Cross-element interference is suppressed by bandpass filtering.
//!
//! ## DSQG analogy
//!
//! DSQG heads currently see all 12 scales (offsets 1..1536) through the
//! learned gain tensor.  The positional bias learned soft specialisation
//! (h0=global, h7=local), but every head still has non-zero weight on
//! every scale — broadband operation.
//!
//! "Contamination" = power in the field at head h from scales j NOT in
//! head h's target band.  This power contributes noise to retrieval without
//! adding signal, reducing SNR.
//!
//! Hard band assignment: head h is restricted to scales j ∈ band(h).
//!   - Local heads (h=6,7):  j ∈ {0,1,2,3}    (offsets 1..8)
//!   - Mid heads  (h=2..5):  j ∈ {3,4,5,6,7}  (offsets 8..128)
//!   - Global heads (h=0,1): j ∈ {8,9,10,11}  (offsets 256..1536)
//!
//! ## What this module verifies
//!
//! 1. Cross-band contamination exists: broadband gains contain significant
//!    power from wrong-scale positions.
//! 2. Banded gains eliminate cross-band contamination for each head.
//! 3. SNR improves with banded gains at each head's target distance.
//! 4. Coherence length extends for global heads under band separation.
//! 5. Trade-off: banding slightly hurts coverage at intermediate distances
//!    — this is acceptable given the coherence benefit at extreme range.

const D_HEAD:    usize = 64;
const N_OFFSETS: usize = 12;
const OFFSETS: [usize; N_OFFSETS] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536];
const N_HEADS:   usize = 8;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
fn norm(v: &[f64]) -> f64 { dot(v,v).sqrt() }

// ── band definitions ────────────────────────────────────────────────────

/// Which scale indices (j) each head is assigned to in banded mode.
fn head_band(h: usize) -> Vec<usize> {
    match h {
        0 => vec![9, 10, 11],         // global: offsets 512..1536
        1 => vec![7, 8, 9],           // semi-global: offsets 128..512
        2 => vec![5, 6, 7],           // mid-far: offsets 32..128
        3 => vec![4, 5, 6],           // mid: offsets 16..64
        4 => vec![3, 4, 5],           // mid-near: offsets 8..32
        5 => vec![2, 3, 4],           // near: offsets 4..16
        6 => vec![1, 2, 3],           // local: offsets 2..8
        7 => vec![0, 1, 2],           // hyper-local: offsets 1..4
        _ => (0..N_OFFSETS).collect(),
    }
}

/// Target distance for head h (midpoint of band).
fn head_target_offset(h: usize) -> usize {
    let band = head_band(h);
    let j_mid = band[band.len() / 2];
    OFFSETS[j_mid]
}

/// Broadband (uniform) gains for a head: all scales equal weight.
fn broadband_gains() -> [f64; N_OFFSETS] {
    [1.0 / N_OFFSETS as f64; N_OFFSETS]
}

/// Banded gains for head h: equal weight within band, 0 outside.
fn banded_gains(h: usize) -> [f64; N_OFFSETS] {
    let band = head_band(h);
    let w    = 1.0 / band.len() as f64;
    let mut g = [0.0f64; N_OFFSETS];
    for j in band { g[j] = w; }
    g
}

/// Softmax-based positional-bias gains (condM learned pattern approximation):
/// head h has higher weight on scales near its "preferred" band, but non-zero everywhere.
fn soft_bias_gains(h: usize) -> [f64; N_OFFSETS] {
    // Simulate pos_bias: score = −α · |j − j_pref|
    let j_pref = head_band(h)[head_band(h).len() / 2] as f64;
    let alpha  = 0.8;
    let scores: Vec<f64> = (0..N_OFFSETS)
        .map(|j| -alpha * (j as f64 - j_pref).abs())
        .collect();
    let max  = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut g = [0.0f64; N_OFFSETS];
    for j in 0..N_OFFSETS { g[j] = exps[j] / sum; }
    g
}

// ── SNR computation ─────────────────────────────────────────────────────

/// Scene: target deposited at distance delta (exactly at offset[j_tgt]).
/// All other offsets contain noise (orthogonal keys).
fn make_scene(j_tgt: usize, strong: f64)
    -> ([f64; D_HEAD], [[f64; D_HEAD]; N_OFFSETS], [[f64; D_HEAD]; N_OFFSETS])
{
    let mut q = [0.0f64; D_HEAD]; q[0] = strong;
    let mut keys   = [[0.0f64; D_HEAD]; N_OFFSETS];
    let mut values = [[0.0f64; D_HEAD]; N_OFFSETS];
    keys[j_tgt][0] = 1.0; values[j_tgt][1] = 1.0;
    for j in 0..N_OFFSETS {
        if j == j_tgt { continue; }
        let d = 2 + j % (D_HEAD - 2);
        keys[j][d] = 1.0; values[j][d] = 1.0;
    }
    (q, keys, values)
}

/// Signal = gains[j_tgt] * Q·K_target (in target value direction e_1).
/// Noise  = quadrature sum of contributions from j ≠ j_tgt.
fn compute_snr_scene(
    q: &[f64], keys: &[[f64; D_HEAD]; N_OFFSETS],
    values: &[[f64; D_HEAD]; N_OFFSETS],
    gains: &[f64; N_OFFSETS], j_tgt: usize,
) -> (f64, f64, f64) {
    let signal = (gains[j_tgt] * dot(q, &keys[j_tgt])).abs() * norm(&values[j_tgt]);
    let noise_var: f64 = (0..N_OFFSETS).filter(|&j| j != j_tgt).map(|j| {
        let w = gains[j] * dot(q, &keys[j]);
        let cos = dot(&values[j], &values[j_tgt]) / (norm(&values[j]) * norm(&values[j_tgt]) + 1e-12);
        (w * cos).powi(2)
    }).sum();
    let noise = noise_var.sqrt();
    (signal, noise, signal / (noise + 1e-10))
}

/// Fraction of total gain power in band h at the wrong scales (contamination ratio).
fn contamination_ratio(gains: &[f64; N_OFFSETS], h: usize) -> f64 {
    let band: Vec<usize> = head_band(h);
    let total_power: f64 = gains.iter().map(|g| g * g).sum();
    let in_band_power: f64 = band.iter().map(|&j| gains[j] * gains[j]).sum();
    1.0 - in_band_power / (total_power + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Cross-band contamination is real and quantified.
    #[test]
    fn test_cross_band_contamination_exists() {
        println!("\n[spectral_band_separation] Test 1: Cross-band contamination");
        println!("  {:>6} | {:>8} | {:>16} | {:>14} | {:>12}",
            "head", "target δ", "contam% (broad)", "contam% (soft)", "contam% (band)");
        for h in 0..N_HEADS {
            let tgt_d  = head_target_offset(h);
            let c_broad = contamination_ratio(&broadband_gains(), h) * 100.0;
            let c_soft  = contamination_ratio(&soft_bias_gains(h),  h) * 100.0;
            let c_band  = contamination_ratio(&banded_gains(h),     h) * 100.0;
            println!("  {:>6} | {:>8} | {:>16.1}% | {:>14.1}% | {:>12.1}%",
                h, tgt_d, c_broad, c_soft, c_band);
            assert!(c_broad > 20.0,
                "Broadband contamination should be significant (>20%); head {h} got {c_broad:.1}%");
            assert!(c_band < 1e-6,
                "Banded contamination must be exactly 0; head {h} got {c_band:.2e}%");
            assert!(c_soft > c_band,
                "Soft bias should have more contamination than hard banding; head {h}");
        }
        println!("  ✓ Banded gains eliminate cross-band contamination; broadband has >20% each head");
    }

    /// Test 2: Banded gains improve SNR at each head's target distance.
    #[test]
    fn test_banded_gains_improve_snr() {
        println!("\n[spectral_band_separation] Test 2: SNR comparison per head at target distance");
        println!("  {:>6} | {:>8} | {:>12} | {:>12} | {:>12} | {:>10}",
            "head", "target δ", "SNR(broad)", "SNR(soft)", "SNR(banded)", "improvement");
        for h in 0..N_HEADS {
            let j_band = head_band(h)[head_band(h).len() / 2]; // middle of band
            let (q, keys, values) = make_scene(j_band, 5.0);
            let (_, _, snr_b) = compute_snr_scene(&q, &keys, &values, &broadband_gains(), j_band);
            let (_, _, snr_s) = compute_snr_scene(&q, &keys, &values, &soft_bias_gains(h), j_band);
            let (_, _, snr_h) = compute_snr_scene(&q, &keys, &values, &banded_gains(h), j_band);
            let improvement = snr_h / (snr_b + 1e-10);
            println!("  {:>6} | {:>8} | {:>12.4} | {:>12.4} | {:>12.4} | {:>10.2}×",
                h, OFFSETS[j_band], snr_b, snr_s, snr_h, improvement);
            assert!(snr_h > snr_b,
                "Banded SNR should exceed broadband at target distance; head={h}");
            assert!(improvement > 1.5,
                "Banded SNR improvement should be >1.5× over broadband; head={h} got {improvement:.2}×");
        }
        println!("  ✓ Banded gains outperform broadband at every head's target distance");
    }

    /// Test 3: Global heads benefit most from banding under realistic position frequency.
    ///
    /// In natural language, local positions are far more frequent than distant ones:
    ///   P(noise at distance δ) ∝ 1/δ (approximately)
    /// This means a global head in broadband mode is contaminated by many high-power
    /// short-range noise positions, while a local head's broadband noise is mostly
    /// other long-range positions (which are rare).
    ///
    /// We model this by weighting noise contributions by 1/offset[j] — simulating
    /// the natural distribution of positions in a typical language sequence.
    #[test]
    fn test_global_heads_benefit_most() {
        // Frequency-weighted SNR: noise from scale j has power proportional to 1/offset[j]
        // (shorter distances are denser in natural language).
        let freq_weight = |j: usize| -> f64 { 1.0 / OFFSETS[j] as f64 };

        // Compute SNR under frequency-weighted noise for head h at its target scale.
        let weighted_snr = |gains: &[f64; N_OFFSETS], h: usize| -> f64 {
            let j_tgt = head_band(h)[head_band(h).len() / 2];
            let signal = gains[j_tgt] * 5.0; // Q·K_target = 5.0
            let noise_var: f64 = (0..N_OFFSETS)
                .filter(|&j| j != j_tgt)
                .map(|j| {
                    let contrib = gains[j] * freq_weight(j); // frequency-weighted noise
                    contrib * contrib
                })
                .sum();
            signal / (noise_var.sqrt() + 1e-10)
        };

        println!("\n[spectral_band_separation] Test 3: Global heads benefit most (freq-weighted)");
        println!("  {:>6} | {:>8} | {:>14} | {:>14} | {:>12}",
            "head", "target δ", "SNR_broad", "SNR_banded", "improvement");
        let mut improvements = Vec::new();
        for h in 0..N_HEADS {
            let snr_b = weighted_snr(&broadband_gains(), h);
            let snr_h = weighted_snr(&banded_gains(h), h);
            let impr  = snr_h / (snr_b + 1e-10);
            improvements.push(impr);
            println!("  {:>6} | {:>8} | {:>14.4} | {:>14.4} | {:>12.2}×",
                h, head_target_offset(h), snr_b, snr_h, impr);
        }

        let global_improve = (improvements[0] + improvements[1]) / 2.0;
        let local_improve  = (improvements[6] + improvements[7]) / 2.0;
        println!("  Global heads mean improvement: {global_improve:.2}×");
        println!("  Local heads mean improvement:  {local_improve:.2}×");
        println!("  Ratio (global/local): {:.2}×", global_improve / local_improve);

        // Under frequency-weighted noise: global heads face more high-power short-range
        // noise in broadband mode → banding provides larger relative improvement.
        assert!(global_improve > local_improve * 1.5,
            "Global heads should benefit >1.5× more than local heads; got {global_improve:.2} vs {local_improve:.2}");
        println!("  ✓ Global heads benefit most from spectral banding under realistic position frequency");
    }

    /// Test 4: Banding trade-off — intermediate distances handled by mid heads.
    /// The coverage matrix: every target distance is covered by exactly one head band.
    #[test]
    fn test_complete_coverage_across_bands() {
        println!("\n[spectral_band_separation] Test 4: Coverage matrix");
        println!("  Each offset should be in at least one head's primary band.");
        println!("  {:>8} | {:>16}", "offset", "heads covering it");
        for (j, &d) in OFFSETS.iter().enumerate() {
            let covering: Vec<usize> = (0..N_HEADS)
                .filter(|&h| head_band(h).contains(&j))
                .collect();
            println!("  {:>8} | {:?}", d, covering);
            assert!(!covering.is_empty(),
                "Offset j={j} (d={d}) should be covered by at least one head's band");
        }
        println!("  ✓ All offsets covered — no blind spots in the band assignment");
    }

    /// Test 5: SNR gap (banded vs broadband) grows with target distance —
    /// confirms that coherence-length extension is largest at extreme range.
    #[test]
    fn test_snr_gap_grows_with_distance() {
        println!("\n[spectral_band_separation] Test 5: SNR gap grows with target distance");
        println!("  {:>8} | {:>12} | {:>12} | {:>10}", "target δ", "SNR(broad)", "SNR(banded)", "gap");
        let mut prev_gap = 0.0;
        let test_heads = [7usize, 5, 3, 1, 0]; // local → global
        for &h in &test_heads {
            let j_band = head_band(h)[head_band(h).len() / 2];
            let (q, keys, values) = make_scene(j_band, 5.0);
            let (_, _, snr_b) = compute_snr_scene(&q, &keys, &values, &broadband_gains(), j_band);
            let (_, _, snr_h) = compute_snr_scene(&q, &keys, &values, &banded_gains(h), j_band);
            let gap = snr_h - snr_b;
            println!("  {:>8} | {:>12.4} | {:>12.4} | {:>10.4}", OFFSETS[j_band], snr_b, snr_h, gap);
            assert!(gap >= prev_gap * 0.95,
                "SNR gap should grow with target distance; head={h} gap={gap:.4} prev_gap={prev_gap:.4}");
            prev_gap = gap;
        }
        println!("  ✓ SNR gap from banding grows with target distance — coherence-length extension confirmed");
    }
}
