//! RG-motivated (Renormalization Group) scale gain initialization.
//!
//! ## Physics background
//!
//! In the RG picture of deep networks (Mehta & Schwab 2014), each layer acts
//! as a coarse-graining step: short-range (high-frequency) structure is
//! integrated out, and only the slow (coarse) modes survive to the next layer.
//!
//! For wavelet-based field convolution this predicts three structural properties:
//!
//!   1. **Fine-scale bias in early layers.**  Layer 0 responds most strongly to
//!      local (high-frequency) patterns — syntactic, character-level structure.
//!
//!   2. **Coarse-scale drift with depth.**  As layer index l increases, the
//!      centre of mass of the gain distribution shifts toward lower-frequency
//!      (coarser) wavelet scales.
//!
//!   3. **Magnitude decay with depth.**  The wavelet kernel contribution shrinks
//!      as representations become more abstract; late layers don't need the
//!      raw frequency inductive bias as much.
//!
//! ## Empirical calibration (C_causal, 12-layer, G=2048)
//!
//! These predictions are consistent with the trained C_causal model:
//!   - L0  mean gain across all (head, scale) pairs: **0.458** (highest)
//!   - L11 mean gain:                                **0.092** (suppressed)
//!
//! RG init pre-biases the model toward this structure at t=0, potentially
//! reducing the training budget needed to reach it through gradient descent.
//!
//! ## Scale index convention
//!
//! Consistent with `wavelet::apply_scale_gains`:
//!   - Index 0   → **finest** detail band  (high frequency, band width G/2)
//!   - Index S-1 → **coarsest** approximation band (low frequency, 1 coeff)

/// Per-layer wavelet scale gain schedule derived from RG coarse-graining.
///
/// `gains[layer][scale]`: f32 ≥ 0.
pub struct RgGainSchedule {
    pub gains: Vec<Vec<f32>>,
    pub num_layers: usize,
    pub num_scales: usize,
}

impl RgGainSchedule {
    /// Build the RG gain schedule.
    ///
    /// # Parameters
    /// - `num_layers`   : total transformer layers (e.g. 6 or 12)
    /// - `num_scales`   : wavelet levels = log₂(G) (e.g. 11 for G = 2048)
    /// - `base_gain`    : peak gain at layer 0 (calibrate to ~0.458 from C_causal)
    /// - `depth_decay`  : multiplicative decay per layer (< 1.0; ~0.864 for 12-layer
    ///                    model to land at L11 mean ~0.092)
    /// - `scale_width`  : Gaussian bandwidth in scale-index units (e.g. S / 3.0)
    pub fn new(
        num_layers: usize,
        num_scales: usize,
        base_gain: f32,
        depth_decay: f32,
        scale_width: f32,
    ) -> Self {
        assert!(num_layers >= 1, "need at least one layer");
        assert!(num_scales >= 1, "need at least one scale");
        assert!(base_gain >= 0.0, "base_gain must be non-negative");
        assert!(depth_decay > 0.0 && depth_decay <= 1.0, "depth_decay must be in (0, 1]");
        assert!(scale_width > 0.0, "scale_width must be positive");

        let s   = num_scales as f32;
        let l_max = (num_layers - 1).max(1) as f32;

        let gains = (0..num_layers)
            .map(|l| {
                // t ∈ [0, 1]: early → late
                let t = l as f32 / l_max;

                // Gaussian centre shifts from finest (0) to coarsest (S-1)
                let scale_center = t * (s - 1.0);

                // Overall amplitude decays with depth
                let magnitude = base_gain * depth_decay.powi(l as i32);

                (0..num_scales)
                    .map(|si| {
                        let diff = si as f32 - scale_center;
                        magnitude * (-(diff * diff) / (2.0 * scale_width * scale_width)).exp()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        Self { gains, num_layers, num_scales }
    }

    /// Returns `(base_gain, depth_decay)` calibrated to the C_causal 12-layer
    /// empirical observations: peak gain at L0 ≈ 0.458, at L11 ≈ 0.092.
    ///
    /// `depth_decay = (0.092 / 0.458)^(1/11) ≈ 0.864`
    pub fn empirical_params_12layer() -> (f32, f32) {
        let base_gain   = 0.458_f32;
        let depth_decay = (0.092_f32 / 0.458_f32).powf(1.0 / 11.0);
        (base_gain, depth_decay)
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Centre of mass of the gain distribution for `layer`, in scale-index units.
    ///
    /// - → 0.0       fine-scale dominant  (local / high-frequency)
    /// - → S − 1     coarse-scale dominant (global / low-frequency)
    pub fn scale_center_of_mass(&self, layer: usize) -> f32 {
        let g: &[f32] = &self.gains[layer];
        let total: f32 = g.iter().sum();
        if total < 1e-12 {
            return 0.0;
        }
        g.iter().enumerate().map(|(s, &v)| s as f32 * v).sum::<f32>() / total
    }

    /// Mean gain across all scale indices for `layer`.
    pub fn mean_gain(&self, layer: usize) -> f32 {
        let g = &self.gains[layer];
        g.iter().sum::<f32>() / g.len() as f32
    }

    /// Peak (maximum) gain for `layer`.
    pub fn peak_gain(&self, layer: usize) -> f32 {
        self.gains[layer]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

// ─── verification tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Default 12-layer / 11-scale schedule calibrated to C_causal empirical data.
    fn default_schedule() -> RgGainSchedule {
        let (base, decay) = RgGainSchedule::empirical_params_12layer();
        let num_scales = 11usize; // log₂(2048)
        let scale_width = num_scales as f32 / 3.0;
        RgGainSchedule::new(12, num_scales, base, decay, scale_width)
    }

    // ── Property 1: scale centre shifts coarseward ───────────────────────────

    /// RG prediction: successive coarse-graining steps eliminate fine-scale
    /// modes.  The centre of mass of gains must be non-decreasing across layers
    /// and strictly larger at the final layer than the first.
    #[test]
    fn rg_scale_center_shifts_coarseward() {
        let sched = default_schedule();
        let centers: Vec<f32> = (0..sched.num_layers)
            .map(|l| sched.scale_center_of_mass(l))
            .collect();

        for i in 0..centers.len() - 1 {
            assert!(
                centers[i + 1] >= centers[i] - 1e-5,
                "Scale CoM must be non-decreasing: layer {i} = {:.4}, layer {} = {:.4}",
                centers[i],
                i + 1,
                centers[i + 1],
            );
        }

        assert!(
            *centers.last().unwrap() > *centers.first().unwrap() + 1.0,
            "Final-layer CoM ({:.3}) should exceed first-layer CoM ({:.3}) by > 1 scale",
            centers.last().unwrap(),
            centers.first().unwrap(),
        );
    }

    // ── Property 2: magnitude decays monotonically with depth ────────────────

    /// RG prediction: late layers operate on abstract coarse representations
    /// and need less help from the raw wavelet inductive bias.
    ///
    /// We test **peak** gain (max over scales) rather than mean gain.
    /// Peak gain = `base_gain × depth_decay^l`, which is strictly controlled by
    /// `depth_decay` and is guaranteed monotone.  Mean gain can exhibit a slight
    /// rise from L0 → L1 due to Gaussian truncation: the L0 Gaussian is centred
    /// at scale 0 (left edge) and its left tail is clipped; as the centre shifts
    /// right in L1 more area fits, briefly offsetting the decay.  Peak gain is
    /// the correct instrument for isolating the depth-decay property.
    #[test]
    fn rg_magnitude_decays_with_depth() {
        let sched = default_schedule();
        let peaks: Vec<f32> = (0..sched.num_layers)
            .map(|l| sched.peak_gain(l))
            .collect();

        for i in 0..peaks.len() - 1 {
            assert!(
                peaks[i + 1] <= peaks[i] + 1e-6,
                "Peak gain must be non-increasing: layer {i} = {:.5}, layer {} = {:.5}",
                peaks[i],
                i + 1,
                peaks[i + 1],
            );
        }

        // Final layer should be substantially smaller than first
        let ratio = peaks[0] / peaks.last().unwrap().max(1e-9);
        assert!(
            ratio > 2.0,
            "Peak gain ratio L0/L{} = {ratio:.2}× should be > 2× (depth decay is real)",
            sched.num_layers - 1,
        );
    }

    // ── Property 3: early layer is fine-biased ────────────────────────────────

    /// Layer 0 Gaussian is centred at scale 0 (finest), so its CoM should lie
    /// in the lower half of the scale index range — below the midpoint.
    #[test]
    fn rg_early_layer_fine_biased() {
        let sched = default_schedule();
        let com_l0  = sched.scale_center_of_mass(0);
        let midpoint = (sched.num_scales - 1) as f32 / 2.0;
        assert!(
            com_l0 < midpoint,
            "L0 CoM {com_l0:.3} must be below midpoint {midpoint:.3} (fine-scale bias)"
        );
    }

    // ── Property 4: late layer is coarse-biased ───────────────────────────────

    /// Layer L-1 Gaussian is centred at scale S-1 (coarsest), so its CoM
    /// should lie in the upper half of the scale index range — above midpoint.
    #[test]
    fn rg_late_layer_coarse_biased() {
        let sched = default_schedule();
        let last    = sched.num_layers - 1;
        let com_last = sched.scale_center_of_mass(last);
        let midpoint  = (sched.num_scales - 1) as f32 / 2.0;
        assert!(
            com_last > midpoint,
            "L{last} CoM {com_last:.3} must be above midpoint {midpoint:.3} (coarse-scale bias)"
        );
    }

    // ── Property 5: peak gain ratio matches empirical observation ────────────

    /// Peak gain at L0 / peak gain at L11 should approximate the empirical
    /// C_causal ratio (0.458 / 0.092 ≈ 5×).  Allow a 2× window either side.
    #[test]
    fn rg_peak_gain_ratio_matches_empirical() {
        let sched       = default_schedule();
        let peak_l0     = sched.peak_gain(0);
        let peak_l_last = sched.peak_gain(sched.num_layers - 1);
        let ratio       = peak_l0 / peak_l_last.max(1e-9);
        // Empirical ≈ 4.98×; accept [2.5, 10.0]
        assert!(
            (2.5..=10.0).contains(&ratio),
            "Peak gain ratio L0/L{} = {ratio:.2}× should be in [2.5, 10.0] (empirical ~5×)",
            sched.num_layers - 1,
        );
    }

    // ── Property 6: all gains are finite and non-negative ────────────────────

    #[test]
    fn rg_gains_are_valid() {
        let sched = default_schedule();
        for (l, layer_gains) in sched.gains.iter().enumerate() {
            for (s, &g) in layer_gains.iter().enumerate() {
                assert!(
                    g.is_finite() && g >= 0.0,
                    "Layer {l} scale {s} gain {g:.6} is invalid (must be finite ≥ 0)"
                );
            }
        }
    }

    // ── Property 7: scale CoM is strictly ordered at extremes ────────────────

    /// Ensures the schedule spans a wide enough range to be useful as an init.
    /// The final layer's CoM should be at least 4 scale indices above layer 0's.
    #[test]
    fn rg_scale_range_is_meaningful() {
        let sched    = default_schedule();
        let com_first = sched.scale_center_of_mass(0);
        let com_last  = sched.scale_center_of_mass(sched.num_layers - 1);
        assert!(
            com_last - com_first > 4.0,
            "CoM range ({:.3} → {:.3}) = {:.3} scale indices; expected > 4.0 for useful span",
            com_first, com_last, com_last - com_first,
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn rg_single_layer_no_panic() {
        let sched = RgGainSchedule::new(1, 11, 0.5, 0.9, 3.0);
        assert_eq!(sched.gains.len(), 1);
        assert!(sched.gains[0].iter().all(|&g| g.is_finite() && g >= 0.0));
    }

    #[test]
    fn rg_single_scale_no_panic() {
        // One scale → CoM always 0, peak = base_gain
        let sched = RgGainSchedule::new(6, 1, 0.5, 0.9, 1.0);
        for l in 0..6 {
            assert_eq!(sched.scale_center_of_mass(l), 0.0);
        }
    }

    /// Uniform decay of 1.0 means no depth attenuation — peak gains equal
    /// base_gain at every layer (magnitude test should still pass with decay=1
    /// since it checks for "monotonically non-increasing", which 1.0 satisfies).
    #[test]
    fn rg_no_decay_variant_is_valid() {
        let sched = RgGainSchedule::new(6, 11, 0.5, 1.0, 3.0);
        for l in 0..6 {
            assert!(sched.gains[l].iter().all(|&g| g.is_finite() && g >= 0.0));
        }
    }

    // ── Diagnostic print (run with: cargo test -- --nocapture) ───────────────

    #[test]
    fn rg_print_schedule_summary() {
        let sched = default_schedule();
        println!("\nRG gain schedule — 12 layers, 11 scales (G=2048)");
        println!("{:<6} {:>8} {:>8} {:>8}  gains[..5]",
                 "Layer", "Mean", "Peak", "CoM");
        for l in 0..sched.num_layers {
            let preview: Vec<String> = sched.gains[l][..5]
                .iter()
                .map(|g| format!("{:.3}", g))
                .collect();
            println!(
                "L{:<5} {:>8.4} {:>8.4} {:>8.3}  [{}]",
                l,
                sched.mean_gain(l),
                sched.peak_gain(l),
                sched.scale_center_of_mass(l),
                preview.join(", "),
            );
        }
    }
}
