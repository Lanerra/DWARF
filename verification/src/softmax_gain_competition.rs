//! Softmax Gain-Weight Competition
//!
//! # Why `offset_gradient_energy.rs` is wrong
//!
//! The gradient-energy model attributes d41s5's failure to text statistics
//! P_text(δ) — how often natural text has meaningful lookups at distance δ.
//! This is **not** the correct mechanism.
//!
//! The actual bottleneck is **softmax gain-weight competition**: during
//! training, the gradient signal for pos_bias[δ,h] is proportional to the
//! SOFTMAX WEIGHT that offset δ currently receives.  If the initial softmax
//! weight is ≈0 (due to very negative pos_bias init), the offset cannot
//! learn — zero gradient regardless of text statistics.
//!
//! # pos_bias initialization
//!
//! ```text
//! pb(δ, α) = -log(1 + δ) × α
//! ```
//!
//! where α ∈ [0.2, 2.0] (linspace over num_heads=8), δ = offset distance.
//!
//! # Initial softmax weight
//!
//! ```text
//! w(δ_i, h) = exp(pb(δ_i, α_h)) / Σ_j exp(pb(δ_j, α_h))
//!           ≈ (1+δ_i)^{-α_h} / Σ_j (1+δ_j)^{-α_h}
//! ```
//!
//! (deterministic: ignores Q·K random component which averages to ≈0
//! vs the dominant pos_bias)
//!
//! # Gradient coupling
//!
//! ```text
//! ∂L/∂pb(δ,h) ≈ w(δ,h) × signal(δ)
//! ```
//!
//! If w(δ,h) ≈ 0, gradient is ≈0 regardless of signal.  The offset is
//! trapped at its initial value and never learns.

/// pos_bias value for offset δ with decay rate α.
///
/// `pb(δ, α) = -ln(1 + δ) × α`
fn pos_bias(delta: usize, alpha: f64) -> f64 {
    -(1.0 + delta as f64).ln() * alpha
}

/// Initial softmax weight for a single offset δ among all offsets,
/// for a head with decay rate α.
///
/// `w(δ_i, h) = exp(pb(δ_i, α_h)) / Σ_j exp(pb(δ_j, α_h))`
///            = `(1+δ_i)^{-α} / Σ_j (1+δ_j)^{-α}`
pub fn initial_softmax_weight(delta: usize, alpha: f64, all_offsets: &[usize]) -> f64 {
    let numerator = pos_bias(delta, alpha).exp();
    let denominator: f64 = all_offsets
        .iter()
        .map(|&offset| pos_bias(offset, alpha).exp())
        .sum();
    numerator / denominator
}

/// Minimum softmax weight across all (offset, head) pairs.
///
/// The worst-case pair determines learnability: if any head's weight
/// for any offset is below threshold, that offset is stuck on that head.
pub fn min_softmax_weight_in_set(offsets: &[usize], alphas: &[f64]) -> f64 {
    let mut minimum = f64::INFINITY;
    for &offset in offsets {
        for &alpha in alphas {
            let weight = initial_softmax_weight(offset, alpha, offsets);
            if weight < minimum {
                minimum = weight;
            }
        }
    }
    minimum
}

/// Softmax weight profile: average weight per offset across all heads.
///
/// Returns one value per offset (same order as `all_offsets`), each being
/// the mean of `initial_softmax_weight(δ, α_h, all_offsets)` over all heads.
pub fn softmax_weight_profile(all_offsets: &[usize], alphas: &[f64]) -> Vec<f64> {
    all_offsets
        .iter()
        .map(|&delta| {
            let sum: f64 = alphas
                .iter()
                .map(|&alpha| initial_softmax_weight(delta, alpha, all_offsets))
                .sum();
            sum / alphas.len() as f64
        })
        .collect()
}

/// Maximum learnable offset δ within a training budget.
///
/// An offset accumulates gradient proportional to its softmax weight
/// each step.  Over `training_steps` steps, accumulated gradient signal
/// is approximately `w(δ, h) × training_steps`.  The offset is learnable
/// if this exceeds some minimum threshold for all heads.
///
/// Returns the largest δ (from 0..dense_width+some_margin) whose
/// minimum softmax weight × training_steps ≥ min_weight_threshold × training_steps,
/// i.e. whose minimum softmax weight ≥ min_weight_threshold.
///
/// Uses a dense offset set of 0..dense_width plus candidate δ values
/// up to a generous upper bound.
pub fn learnable_cutoff_delta(
    training_steps: usize,
    min_weight_threshold: f64,
    alphas: &[f64],
    dense_width: usize,
) -> usize {
    let effective_threshold = min_weight_threshold / training_steps as f64;

    let max_candidate = dense_width * 100;
    let mut best_delta = 0usize;

    for candidate in 0..=max_candidate {
        let mut all_offsets: Vec<usize> = (0..=dense_width).collect();
        if candidate > dense_width {
            all_offsets.push(candidate);
        }
        all_offsets.sort_unstable();
        all_offsets.dedup();

        let mut learnable = true;
        for &alpha in alphas {
            let weight = initial_softmax_weight(candidate, alpha, &all_offsets);
            if weight < effective_threshold {
                learnable = false;
                break;
            }
        }

        if learnable {
            best_delta = candidate;
        }
    }

    best_delta
}

/// Standard α values: 8 heads, linspace from 0.2 to 2.0.
fn standard_alphas() -> Vec<f64> {
    (0..8).map(|i| 0.2 + (2.0 - 0.2) * i as f64 / 7.0).collect()
}

/// Build offset set: dense range 0..=dense_width plus sparse offsets.
fn build_offset_set(dense_width: usize, sparse: &[usize]) -> Vec<usize> {
    let mut offsets: Vec<usize> = (0..=dense_width).collect();
    offsets.extend_from_slice(sparse);
    offsets.sort_unstable();
    offsets.dedup();
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alphas_8head() -> Vec<f64> {
        standard_alphas()
    }

    #[test]
    fn d41s3_all_offsets_learnable() {
        // d41s3: sparse=[48,128,384], dense=range(42), J=45
        let sparse = &[48usize, 128, 384];
        let dense_width = 41;
        let offsets = build_offset_set(dense_width, sparse);
        assert_eq!(offsets.len(), 45, "d41s3 should have J=45 offsets");

        let alphas = alphas_8head();
        let threshold = 2e-6;

        println!("\n=== d41s3 softmax gain competition ===");
        println!("  offsets: {:?}", sparse);
        println!("  dense_width: {dense_width}, J={}", offsets.len());

        for &offset in sparse {
            let min_weight = alphas
                .iter()
                .map(|&alpha| initial_softmax_weight(offset, alpha, &offsets))
                .fold(f64::INFINITY, f64::min);
            println!("  δ={offset:>4}: min_softmax_weight = {min_weight:.2e}");
            assert!(
                min_weight > threshold,
                "δ={offset} has min softmax weight {min_weight:.2e} < {threshold:.0e} — unlearnable!"
            );
        }
    }

    #[test]
    fn d41s5_long_offsets_unlearnable() {
        // d41s5: sparse=[48,128,384,768,1536], dense=range(42), J=47
        let sparse = &[48usize, 128, 384, 768, 1536];
        let dense_width = 41;
        let offsets = build_offset_set(dense_width, sparse);
        assert_eq!(offsets.len(), 47, "d41s5 should have J=47 offsets");

        let alphas = alphas_8head();
        // Learnability threshold: offsets with min softmax weight below this
        // cannot accumulate enough gradient to escape their initial bias.
        // Empirically: d41s3's δ=384 sits at ~4e-6 (learnable), while
        // d41s5's δ=768 sits at ~1e-6 and δ=1536 at ~3e-7 (unlearnable).
        let threshold = 2e-6;

        println!("\n=== d41s5 softmax gain competition ===");
        println!("  offsets: {:?}", sparse);
        println!("  dense_width: {dense_width}, J={}", offsets.len());

        for &offset in sparse {
            let min_weight = alphas
                .iter()
                .map(|&alpha| initial_softmax_weight(offset, alpha, &offsets))
                .fold(f64::INFINITY, f64::min);
            println!("  δ={offset:>4}: min_softmax_weight = {min_weight:.2e}");
        }

        for &offset in &[768usize, 1536] {
            let min_weight = alphas
                .iter()
                .map(|&alpha| initial_softmax_weight(offset, alpha, &offsets))
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_weight < threshold,
                "δ={offset} has min softmax weight {min_weight:.2e} ≥ {threshold:.0e} — expected unlearnable!"
            );
        }
    }

    #[test]
    fn d41_35m_sparse_offsets_learnable() {
        // d41_35m: sparse=[96,128,384], dense=range(49), J=52
        let sparse = &[96usize, 128, 384];
        let dense_width = 48;
        let offsets = build_offset_set(dense_width, sparse);
        assert_eq!(offsets.len(), 52, "d41_35m should have J=52 offsets");

        let alphas = alphas_8head();
        let threshold = 2e-6;

        println!("\n=== d41_35m softmax gain competition ===");
        println!("  offsets: {:?}", sparse);
        println!("  dense_width: {dense_width}, J={}", offsets.len());

        for &offset in sparse {
            let min_weight = alphas
                .iter()
                .map(|&alpha| initial_softmax_weight(offset, alpha, &offsets))
                .fold(f64::INFINITY, f64::min);
            println!("  δ={offset:>4}: min_softmax_weight = {min_weight:.2e}");
            assert!(
                min_weight > threshold,
                "δ={offset} has min softmax weight {min_weight:.2e} < {threshold:.0e} — unlearnable!"
            );
        }
    }

    #[test]
    fn dense_window_dominates_softmax() {
        let alphas = alphas_8head();

        println!("\n=== Dense window softmax dominance ===");
        for dense_width in &[20usize, 41, 64, 96] {
            let sparse = &[128usize, 384, 768];
            let offsets = build_offset_set(*dense_width, sparse);

            let profile = softmax_weight_profile(&offsets, &alphas);
            let dense_total: f64 = profile.iter().take(*dense_width + 1).sum();

            println!("  dense_width={dense_width:>3}: dense_weight_share = {:.4}%", dense_total * 100.0);
            assert!(
                dense_total > 0.95,
                "dense_width={dense_width}: dense offsets hold only {:.2}% of softmax weight, expected >95%",
                dense_total * 100.0
            );
        }
    }

    #[test]
    fn power_law_weight_profile() {
        let alphas = alphas_8head();
        let offsets = vec![1usize, 2];

        println!("\n=== Power-law weight profile (per-head verification) ===");

        // For a 2-offset set {1, 2}, each head's weight ratio is exact:
        //   w(1,h) / w(2,h) = (1+1)^{-α} / (1+2)^{-α} = (3/2)^α
        for &alpha in &alphas {
            let w1 = initial_softmax_weight(1, alpha, &offsets);
            let w2 = initial_softmax_weight(2, alpha, &offsets);
            let actual_ratio = w1 / w2;
            let expected_ratio = (3.0_f64 / 2.0).powf(alpha);

            println!("  α={alpha:.3}: w(1)/w(2) = {actual_ratio:.6}, expected (3/2)^α = {expected_ratio:.6}");

            let relative_error = ((actual_ratio - expected_ratio) / expected_ratio).abs();
            assert!(
                relative_error < 1e-10,
                "α={alpha}: ratio {actual_ratio:.10} ≠ expected {expected_ratio:.10}, error={relative_error:.2e}"
            );
        }
    }

    #[test]
    fn learnable_cutoff_increases_with_steps() {
        let alphas = alphas_8head();
        let dense_width = 41;
        let threshold = 1.0;

        let cutoff_short = learnable_cutoff_delta(16_483, threshold, &alphas, dense_width);
        let cutoff_long = learnable_cutoff_delta(100_000, threshold, &alphas, dense_width);

        println!("\n=== Learnable cutoff vs training steps ===");
        println!("  dense_width={dense_width}, threshold={threshold}");
        println!("  steps=16,483 → max learnable δ = {cutoff_short}");
        println!("  steps=100,000 → max learnable δ = {cutoff_long}");

        assert!(
            cutoff_long > cutoff_short,
            "cutoff at 100k steps ({cutoff_long}) should exceed cutoff at 16k steps ({cutoff_short})"
        );
    }
}
