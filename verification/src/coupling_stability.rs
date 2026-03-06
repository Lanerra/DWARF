//! Coupling stability pre-flight check for DWARF hybrid architectures.
//!
//! ## The co-training necessity problem
//!
//! DWARF hybrid models (condM, condU) consist of two coupled sub-systems:
//!
//! - **DSQG component (A):** Sparse dyadic attention with physics-inspired
//!   preprocessing (Q-weighted scale gains, IF amplifiers, learned offsets).
//! - **Full attention component (B):** Dense causal self-attention at a single
//!   fixed layer (the "full_layer").
//!
//! Empirically, these components cannot be trained independently and then
//!  combined (the "Engram-retrofit failure," Feb 2026): both components must be
//! co-trained from random initialization. The coupling stability framework
//! explains *why* some architectures co-train successfully and others do not.
//!
//! ## Coupling coefficient
//!
//! The structural coupling between A and B depends on *how* the interference
//! signal is injected into the full-attention pathway:
//!
//! ### K/V injection (condU family)
//!
//! ```text
//! k += W_ki · inter(x)
//! v += W_vi · inter(x)
//! ```
//!
//! The injected signal passes through the nonlinear softmax pathway.  Each
//! head h independently attends to the modified K, so the gradient of the
//! loss w.r.t. the injection weights couples all heads.  Across L layers,
//! each layer's DSQG output can influence every subsequent full-attention
//! computation through the residual stream + injection chain.
//!
//! **K/V coupling coefficient = H × D × L**
//!
//! This grows superlinearly with model size.  At 13M (H=8, D=256, L=6):
//! coeff = 12,288.  At 85M (H=12, D=768, L=8): coeff = 73,728.
//!
//! ### Residual injection (condM family)
//!
//! ```text
//! x += gate(x) * pool(x)
//! ```
//!
//! The injection is a direct addition to the residual stream.  The gradient
//! coupling is bounded by the stream dimension D alone; it does not multiply
//! with the number of heads (scalar addition, not per-head K/V modification)
//! and does not propagate through the softmax nonlinearity.
//!
//! **Residual coupling coefficient = D**
//!
//! This grows linearly with D only.  At 85M (D=640): coeff = 640, which
//! stays far below the stability threshold even at large scale.
//!
//! ## Stability threshold calibration
//!
//! Empirical training outcomes (March 2026):
//!
//! | Model      | Type     | Coeff   | Measured coupling | Stable |
//! |------------|----------|---------|-------------------|--------|
//! | condU 13M  | K/V      | 12,288  | 0.611             | ✓      |
//! | condU 39M  | K/V      | 24,576  | 0.678             | ✓      |
//! | condU 85M  | K/V      | 73,728  | —                 | ✗ memo |
//! | condM 13M  | residual | 256     | 0.663             | ✓      |
//! | condM 85M  | residual | 640     | 0.496             | ✓      |
//!
//! "Measured coupling" = empirical gradient-dependency metric (see
//! tools/compute_coupling.py).  Stable models cluster in 0.50–0.68; condU
//! 85M failed with train/val PPL memorization before convergence.
//!
//! Threshold: **K/V = 24,577** (conservative: just above max stable coeff).
//! The log-midpoint estimate is 42,566; arithmetic midpoint is 49,152.
//! A condU 27M datapoint (~19,200 coeff) would tighten the bracket further.
//!
//! Residual injection has no stability failures in the experimental record;
//! no upper threshold is currently calibrated.
//!
//! ## Usage as pre-flight check
//!
//! Call `check_stability` before committing GPU hours to a new architecture.
//! A `Err` result means the training run will likely memorize before converging.
//!
//! ```rust
//! use coupling_stability::{InjectionType, check_stability};
//!
//! // Planning a condU run at 85M scale — will this work?
//! let result = check_stability(768, 12, 8, InjectionType::Kv);
//! assert!(result.is_err(), "condU 85M K/V coupling exceeds stability threshold");
//! ```

/// Injection mechanism used by the interference layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InjectionType {
    /// K/V injection: `k += W_ki·inter, v += W_vi·inter`.
    /// Coupling scales as H × D × L (nonlinear softmax pathway).
    Kv,
    /// Residual injection: `x += gate * pool(x)`.
    /// Coupling scales as D (linear residual stream).
    Residual,
}

/// Stability threshold for K/V coupling coefficient.
///
/// Calibrated March 5, 2026:
/// - max stable: 24,576 (condU 39M, D=512, H=8, L=6)
/// - min unstable: 73,728 (condU 85M, D=768, H=12, L=8 — memorization failure)
///
/// Conservative estimate (just above max stable). Update when a condU model
/// between 39M and 85M provides a tighter bracket.
const STABILITY_THRESHOLD_KV: f64 = 24_577.0;

/// Compute the structural coupling coefficient for a given architecture.
///
/// # Arguments
/// - `d_model`: embedding dimension
/// - `n_heads`: number of attention heads
/// - `n_layers`: total number of transformer layers (DSQG + full attention)
/// - `injection`: injection mechanism
///
/// # Returns
/// Coupling coefficient (dimensionless, higher = more structurally coupled).
pub fn coupling_coefficient(
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    injection: InjectionType,
) -> f64 {
    match injection {
        InjectionType::Kv       => (n_heads * d_model * n_layers) as f64,
        InjectionType::Residual => d_model as f64,
    }
}

/// Pre-flight stability check.
///
/// Returns `Ok(coeff)` if the architecture is expected to co-train stably,
/// or `Err(msg)` with a human-readable explanation if it exceeds the
/// stability threshold.
///
/// # Example
/// ```
/// use coupling_stability::{InjectionType, check_stability};
/// assert!(check_stability(256, 8, 6, InjectionType::Kv).is_ok());   // condU 13M
/// assert!(check_stability(768, 12, 8, InjectionType::Kv).is_err()); // condU 85M
/// ```
pub fn check_stability(
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    injection: InjectionType,
) -> Result<f64, String> {
    let coeff = coupling_coefficient(d_model, n_heads, n_layers, injection);
    let threshold = match injection {
        InjectionType::Kv => STABILITY_THRESHOLD_KV,
        InjectionType::Residual => f64::INFINITY, // no failures observed; no upper bound
    };

    if coeff > threshold {
        Err(format!(
            "Coupling stability check FAILED:\n  \
             Architecture: D={d_model}, H={n_heads}, L={n_layers}, injection={injection:?}\n  \
             K/V coupling coefficient: {coeff:.0} > threshold {threshold:.0}\n  \
             Prediction: train/val PPL memorization before convergence (condU 85M precedent).\n  \
             Options:\n  \
             (a) Switch to residual injection (condM-style) — coupling = {d_model} (safe)\n  \
             (b) Reduce scale: D≤512, H≤8, L≤6 keeps K/V coeff ≤ 24,576\n  \
             (c) Multi-task training (PPL + passkey) may shift stability boundary\n  \
             Threshold calibrated March 2026 from 5 training outcomes. \
             A condU ~300-400M datapoint would tighten the bracket."
        ))
    } else {
        Ok(coeff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Coupling coefficient formula
    // -----------------------------------------------------------------------

    #[test]
    fn kv_coefficient_condU_13M() {
        // D=256, H=8, L=6 → 256 * 8 * 6 = 12,288
        let coeff = coupling_coefficient(256, 8, 6, InjectionType::Kv);
        assert_eq!(coeff, 12_288.0, "condU 13M K/V coeff should be 12,288");
    }

    #[test]
    fn kv_coefficient_condU_39M() {
        // D=512, H=8, L=6 → 512 * 8 * 6 = 24,576
        let coeff = coupling_coefficient(512, 8, 6, InjectionType::Kv);
        assert_eq!(coeff, 24_576.0, "condU 39M K/V coeff should be 24,576");
    }

    #[test]
    fn kv_coefficient_condU_85M() {
        // D=768, H=12, L=8 → 768 * 12 * 8 = 73,728
        let coeff = coupling_coefficient(768, 12, 8, InjectionType::Kv);
        assert_eq!(coeff, 73_728.0, "condU 85M K/V coeff should be 73,728");
    }

    #[test]
    fn residual_coefficient_condM_13M() {
        // Residual injection: coeff = D = 256
        let coeff = coupling_coefficient(256, 8, 6, InjectionType::Residual);
        assert_eq!(coeff, 256.0, "condM 13M residual coeff should be 256");
    }

    #[test]
    fn residual_coefficient_condM_85M() {
        // D=640, H=8, L=12 → residual coeff = D = 640
        let coeff = coupling_coefficient(640, 8, 12, InjectionType::Residual);
        assert_eq!(coeff, 640.0, "condM 85M residual coeff should be 640");
    }

    #[test]
    fn residual_coefficient_independent_of_h_and_l() {
        // Residual coupling depends only on D, not H or L
        let c1 = coupling_coefficient(512, 4,  6, InjectionType::Residual);
        let c2 = coupling_coefficient(512, 16, 12, InjectionType::Residual);
        assert_eq!(c1, c2, "Residual coupling must be independent of H and L");
        assert_eq!(c1, 512.0);
    }

    // -----------------------------------------------------------------------
    // Stability check — all 5 known training outcomes
    // -----------------------------------------------------------------------

    #[test]
    fn condU_13M_stable() {
        let r = check_stability(256, 8, 6, InjectionType::Kv);
        assert!(r.is_ok(), "condU 13M should be predicted stable; got {:?}", r);
        assert_eq!(r.unwrap(), 12_288.0);
    }

    #[test]
    fn condU_39M_stable() {
        let r = check_stability(512, 8, 6, InjectionType::Kv);
        assert!(r.is_ok(), "condU 39M should be predicted stable; got {:?}", r);
    }

    #[test]
    fn condU_85M_unstable() {
        let r = check_stability(768, 12, 8, InjectionType::Kv);
        assert!(r.is_err(), "condU 85M should be predicted UNSTABLE");
        let msg = r.unwrap_err();
        assert!(msg.contains("FAILED"), "Error message should say FAILED");
        assert!(msg.contains("73728") || msg.contains("73,728"),
                "Error message should include the coupling coefficient");
        println!("\ncondU 85M stability error message:\n{}", msg);
    }

    #[test]
    fn condM_13M_stable() {
        let r = check_stability(256, 8, 6, InjectionType::Residual);
        assert!(r.is_ok(), "condM 13M (residual) should be predicted stable");
    }

    #[test]
    fn condM_85M_stable() {
        let r = check_stability(640, 8, 12, InjectionType::Residual);
        assert!(r.is_ok(), "condM 85M (residual) should be predicted stable");
    }

    // -----------------------------------------------------------------------
    // Threshold boundary behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn exactly_at_max_stable_coeff_passes() {
        // condU 39M coeff = 24,576; threshold = 24,577 → should pass
        let coeff = coupling_coefficient(512, 8, 6, InjectionType::Kv);
        assert!(coeff < STABILITY_THRESHOLD_KV,
                "condU 39M coeff ({}) must be < threshold ({})", coeff, STABILITY_THRESHOLD_KV);
    }

    #[test]
    fn one_above_threshold_fails() {
        // Threshold is 24,577 — construct an architecture with coeff = 24,578
        // 24,578 / 8 = 3072.25 → not integer, try coeff = 24,576 + 8 = 24,584
        // D=512, H=8, L=6 gives 24,576 (stable). Bump L to 7 → 512*8*7 = 28,672 (unstable)
        let r = check_stability(512, 8, 7, InjectionType::Kv);
        assert!(r.is_err(),
                "D=512 H=8 L=7 coeff=28,672 should exceed threshold and fail");
    }

    #[test]
    fn residual_never_fails_at_large_scale() {
        // Even a 7B residual model (D=4096) should pass
        let r = check_stability(4096, 32, 32, InjectionType::Residual);
        assert!(r.is_ok(), "Residual injection should never fail stability check");
    }

    // -----------------------------------------------------------------------
    // Measured coupling clustering (empirical, not enforced by formula)
    // -----------------------------------------------------------------------

    /// All stable models measured coupling_mean between 0.40 and 0.70.
    /// This test documents the empirical range; it does not use the formula.
    #[test]
    fn stable_models_cluster_in_empirical_range() {
        let measured = [
            ("condU_13M_ep10", 0.6107_f64),
            ("condU_35M_ep10", 0.6784_f64),
            ("condM_I2G0_ep10", 0.6629_f64),
            ("condM_85M_best",  0.4961_f64),
        ];
        for (label, coupling) in measured {
            assert!(
                coupling >= 0.40 && coupling <= 0.70,
                "{label}: measured coupling {coupling:.4} outside expected [0.40, 0.70] range"
            );
        }
    }

    /// K/V injection coupling starts higher at epoch 1 for larger models,
    /// consistent with the structural coupling coefficient scaling with D*H*L.
    #[test]
    fn kv_coupling_epoch1_increases_with_scale() {
        let condU_13M_ep1: f64 = 0.4073;
        let condU_35M_ep1: f64 = 0.5357;
        assert!(
            condU_35M_ep1 > condU_13M_ep1,
            "Larger K/V model should have higher epoch-1 coupling (structural entanglement from init)"
        );
    }
}
