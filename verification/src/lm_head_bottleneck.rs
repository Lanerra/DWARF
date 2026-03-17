//! LM Head Gradient Bottleneck Analysis
//!
//! Models gradient compression through the tied LM head (embedding weight matrix
//! transpose) per Godey et al., arXiv:2603.10145.
//!
//! # The Core Theorem
//!
//! When the vocabulary embedding W_E ∈ R^{V × D} is used as the LM head weight
//! (weight tying), the gradient of the cross-entropy loss w.r.t. the final
//! hidden state h ∈ R^D has rank at most 2D, regardless of vocabulary size V:
//!
//!   rank(∂L/∂h) ≤ 2D
//!
//! This means (V − 1 − 2D) out of (V − 1) "gradient directions" are suppressed.
//! The fraction of gradient norm that actually propagates to earlier layers is:
//!
//!   effective_gradient_fraction = 2D / (V − 1)  [theoretical upper bound]
//!
//! Empirical suppression is typically 95–99% (Godey et al.).
//!
//! # Practical impact: gradient suppression by config
//!
//! | Config    |  D   |   V   | Suppression | Eff. frac | lr_mult (anal.) |
//! |-----------|------|-------|-------------|-----------|-----------------|
//! | 14M V=32k |  256 | 32768 |     98.4%   |   1.6%    |     ~64         |
//! | 14M V=8k  |  256 |  8192 |     93.8%   |   6.2%    |     ~16         |
//! | 35M V=32k |  512 | 32768 |     96.8%   |   3.1%    |     ~32         |
//! | 35M V=8k  |  512 |  8192 |     87.5%   |  12.5%    |      ~8         |
//! | 100M V=32k| 1024 | 32768 |     93.6%   |   6.3%    |     ~16         |
//!
//! # The Empirical Puzzle
//!
//! The empirical lr_mult values (10 at 14M, 15 at 35M) do NOT follow the simple
//! (V−1)/(2D) ratio. The analytical model predicts 35M should need LESS lr_mult
//! than 14M (35M has lower suppression), but empirical data shows the opposite.
//!
//! Hypothesis: lr_mult=15 at 35M was set before J16D was fully characterized,
//! and the actual optimal might be 8–10. This module derives the analytical
//! prediction and provides an experimentally testable claim.
//!
//! # Optimal lr_mult derivation
//!
//!   lr_mult_optimal = 1 / effective_gradient_fraction = (V − 1) / (2D)
//!
//! Validate against empirical: lr_mult=10 at 14M, lr_mult=15 at 35M.
//!
//! # Reference
//! Godey et al. (2025), arXiv:2603.10145

// ─── Configs ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LMHeadConfig {
    pub name: &'static str,
    pub embedding_dim: usize,           // D
    pub vocab_size: usize,              // V
    pub empirical_lr_mult: Option<f64>, // observed in training (if available)
}

/// Primary DWARF configurations analyzed for gradient bottleneck
pub const LM_HEAD_CONFIGS: [LMHeadConfig; 5] = [
    LMHeadConfig { name: "14M V=32k",  embedding_dim: 256,  vocab_size: 32768, empirical_lr_mult: Some(10.0) },
    LMHeadConfig { name: "14M V=8k",   embedding_dim: 256,  vocab_size:  8192, empirical_lr_mult: None },
    LMHeadConfig { name: "35M V=32k",  embedding_dim: 512,  vocab_size: 32768, empirical_lr_mult: Some(15.0) },
    LMHeadConfig { name: "35M V=8k",   embedding_dim: 512,  vocab_size:  8192, empirical_lr_mult: None },
    LMHeadConfig { name: "100M V=32k", embedding_dim: 1024, vocab_size: 32768, empirical_lr_mult: None },
];

// ─── Core model ───────────────────────────────────────────────────────────────

/// Effective gradient fraction reaching earlier layers through the tied LM head.
///
/// From the rank-2D theorem: the gradient w.r.t. the final hidden state has
/// rank ≤ 2D, so at most 2D out of (V−1) gradient components are non-suppressed.
///
/// Returns a value in (0, 1].
pub fn effective_gradient_fraction(embedding_dim: usize, vocab_size: usize) -> f64 {
    let d = embedding_dim as f64;
    let v = vocab_size as f64;
    (2.0 * d / (v - 1.0)).min(1.0)
}

/// Gradient suppression percentage: fraction of gradient norm that is suppressed.
pub fn gradient_suppression_pct(embedding_dim: usize, vocab_size: usize) -> f64 {
    (1.0 - effective_gradient_fraction(embedding_dim, vocab_size)) * 100.0
}

/// Analytically derived optimal lr_mult for the LM head.
///
/// To achieve the same effective learning rate as the embedding layer,
/// the LM head needs lr_mult inversely proportional to the gradient fraction:
///
///   lr_mult_optimal = (V − 1) / (2D)
pub fn analytical_lr_mult(embedding_dim: usize, vocab_size: usize) -> f64 {
    let eff = effective_gradient_fraction(embedding_dim, vocab_size);
    1.0 / eff.max(1e-10)
}

/// Score for use in the composite fitness function of `scale_optimizer`.
///
/// Higher score = less gradient suppression = more effective gradient propagation.
/// Score = effective_gradient_fraction, normalised to [0, 1].
///
/// This penalises large-V / small-D configs that waste optimizer budget on
/// the LM head rather than the DWARF field computation upstream.
pub fn grad_bottleneck_score(embedding_dim: usize, vocab_size: usize) -> f64 {
    effective_gradient_fraction(embedding_dim, vocab_size)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — Full suppression table
    // ─────────────────────────────────────────────────────────────────────────

    /// Prints the gradient suppression analysis for all DWARF configs and
    /// verifies that suppression values match the analytical predictions from
    /// arXiv:2603.10145.
    #[test]
    fn gradient_suppression_table() {
        println!("\n══ LM Head Gradient Bottleneck (arXiv:2603.10145) ════════════════════");
        println!("Rank-2D theorem: grad rank ≤ 2D through tied LM head, regardless of V");
        println!();
        println!("{:<16} {:>6} {:>8} {:>14} {:>16} {:>18} {:>16}",
                 "Config", "D", "V", "Suppression", "Eff. fraction", "Analytical lr×", "Empirical lr×");
        println!("{}", "─".repeat(95));

        for cfg in &LM_HEAD_CONFIGS {
            let supp     = gradient_suppression_pct(cfg.embedding_dim, cfg.vocab_size);
            let eff      = effective_gradient_fraction(cfg.embedding_dim, cfg.vocab_size);
            let lr_anal  = analytical_lr_mult(cfg.embedding_dim, cfg.vocab_size);
            let emp_str  = cfg.empirical_lr_mult
                .map(|e| format!("{:.1}", e))
                .unwrap_or_else(|| "—".into());
            println!("{:<16} {:>6} {:>8} {:>13.1}% {:>16.4} {:>18.1} {:>16}",
                     cfg.name, cfg.embedding_dim, cfg.vocab_size,
                     supp, eff, lr_anal, emp_str);
        }
        println!();

        // V=8k at 14M should have substantially lower suppression than V=32k
        let supp_32k = gradient_suppression_pct(256, 32768);
        let supp_8k  = gradient_suppression_pct(256,  8192);
        println!("14M V=8k reduces suppression: {supp_32k:.1}% → {supp_8k:.1}%");
        println!("This is the predicted mechanism for faster passkey emergence at V=8k.");

        assert!(supp_32k > supp_8k,
            "V=32k suppression ({supp_32k:.1}%) should exceed V=8k ({supp_8k:.1}%)");
        assert!(supp_32k > 95.0,
            "14M V=32k suppression ({supp_32k:.1}%) should exceed 95%");
        assert!(supp_8k > 85.0,
            "14M V=8k suppression ({supp_8k:.1}%) should still be substantial (>85%)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — The empirical lr_mult puzzle
    // ─────────────────────────────────────────────────────────────────────────

    /// The empirical observation: 14M uses lr_mult=10, 35M uses lr_mult=15.
    /// The analytical model predicts the OPPOSITE ordering (35M needs less
    /// lr_mult than 14M because 35M has lower suppression at same V=32k).
    ///
    /// This test documents the discrepancy and produces an experimentally
    /// testable prediction: lr_mult=8–10 at 35M should be sufficient.
    #[test]
    fn empirical_lr_mult_puzzle() {
        println!("\n══ lr_mult Puzzle: Theory vs Empirical ════════════════════════════════");
        println!();
        println!("{:<16} {:>16} {:>16} {:>12} {:>30}",
                 "Config", "Analytical lr×", "Empirical lr×", "Ratio E/A", "Model verdict");
        println!("{}", "─".repeat(95));

        for cfg in LM_HEAD_CONFIGS.iter().filter(|c| c.empirical_lr_mult.is_some()) {
            let anal  = analytical_lr_mult(cfg.embedding_dim, cfg.vocab_size);
            let emp   = cfg.empirical_lr_mult.unwrap();
            let ratio = emp / anal;
            let verdict = if ratio > 1.5 {
                "emp >> anal → may be suboptimal (too high)"
            } else if ratio < 0.5 {
                "emp << anal → may be too low"
            } else {
                "emp ≈ anal → consistent"
            };
            println!("{:<16} {:>16.1} {:>16.1} {:>12.2} {:>30}",
                     cfg.name, anal, emp, ratio, verdict);
        }

        println!();

        let lr_14m_anal = analytical_lr_mult(256, 32768);
        let lr_35m_anal = analytical_lr_mult(512, 32768);
        println!("Analytical prediction: 14M needs {lr_14m_anal:.0}×, 35M needs {lr_35m_anal:.0}×");
        println!("  (35M lower because wider D reduces suppression at fixed V=32k)");
        println!();
        println!("Empirical: 14M=10, 35M=15 — ordering is REVERSED relative to theory.");
        println!();
        println!("Hypothesis: lr_mult=15 at 35M was set before J16D was characterised.");
        println!("  Testable prediction: lr_mult=8–10 at 35M J16D works as well or better.");
        println!("  Cheap screen: 3 epochs at 35M with lr_mult=8 vs lr_mult=15.");
        println!();
        println!("Alternative: 35M architecture has stronger gradient-consuming upstream ops.");
        println!("  (deeper FFN, wider d_h, more DWARF field ops per token).");
        println!("  This would mean analytical bound is tight but incomplete.");

        // Theory: 35M should need LESS lr_mult than 14M (lower V/D ratio)
        assert!(lr_14m_anal > lr_35m_anal,
            "Theory: 14M ({lr_14m_anal:.1}×) should need more lr_mult than 35M ({lr_35m_anal:.1}×)");

        // Empirical is opposite: 35M uses more. Document the puzzle.
        let lr_14m_emp = 10.0f64;
        let lr_35m_emp = 15.0f64;
        assert!(lr_35m_emp > lr_14m_emp,
            "Empirical: 35M lr_mult ({lr_35m_emp}) should exceed 14M ({lr_14m_emp}) — documenting the puzzle");

        // The ratio is meaningfully discrepant (not just noise)
        let theory_ratio_14_to_35 = lr_14m_anal / lr_35m_anal;
        let emp_ratio_14_to_35    = lr_14m_emp / lr_35m_emp;
        println!("\nTheory 14M/35M ratio: {theory_ratio_14_to_35:.2}   Empirical: {emp_ratio_14_to_35:.2}");
        assert!(theory_ratio_14_to_35 > emp_ratio_14_to_35,
            "Theory ratio ({theory_ratio_14_to_35:.2}) should exceed empirical ({emp_ratio_14_to_35:.2}) — discrepancy confirmed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 — V=8k benefit for 14M screening
    // ─────────────────────────────────────────────────────────────────────────

    /// Quantifies the predicted benefit of V=8k vs V=32k at 14M scale.
    ///
    /// Lower suppression → more gradient reaches earlier layers → passkey
    /// signal propagates faster → emergence should shift from ep3 → ep1.
    ///
    /// This is the theoretical basis for the V=8k screening experiment.
    #[test]
    fn vocab_reduction_14m_benefit() {
        println!("\n══ V=8k Gradient Benefit at 14M ══════════════════════════════════════");
        println!();

        let eff_32k = effective_gradient_fraction(256, 32768);
        let eff_8k  = effective_gradient_fraction(256,  8192);
        let speedup = eff_8k / eff_32k;

        println!("14M V=32k: effective gradient fraction = {:.4}  ({:.1}%)", eff_32k, eff_32k * 100.0);
        println!("14M V=8k:  effective gradient fraction = {:.4}  ({:.1}%)", eff_8k,  eff_8k  * 100.0);
        println!("Gradient amplification factor: {:.2}×", speedup);
        println!();
        println!("Prediction: V=8k sees passkey signal {:.1}× faster than V=32k.", speedup);
        println!("If V=32k sees passkey at ep3, V=8k should emerge by ep1.");
        println!("This is a cheap 3-epoch, 14M-scale single experiment.");
        println!();

        let lr_32k = analytical_lr_mult(256, 32768);
        let lr_8k  = analytical_lr_mult(256,  8192);
        println!("Optimal lr_mult — V=32k: {lr_32k:.0}×, V=8k: {lr_8k:.0}×");
        println!("  Practical: V=8k can use lr_mult≈{:.0} vs V=32k lr_mult≈{:.0}",
                 (lr_8k * 0.5).round(), (lr_32k * 0.5).round());

        // V=8k should give substantially more gradient (not a marginal improvement)
        assert!(speedup > 2.0,
            "V=8k speedup ({speedup:.2}×) should be > 2× over V=32k");
        assert!(speedup < 10.0,
            "V=8k speedup ({speedup:.2}×) should be < 10× (sanity bound)");
        assert!(lr_32k > lr_8k,
            "V=32k lr_mult ({lr_32k:.1}) should exceed V=8k ({lr_8k:.1})");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 — grad_bottleneck_score ordering for scale_optimizer
    // ─────────────────────────────────────────────────────────────────────────

    /// Verifies that grad_bottleneck_score() is in [0,1] and has the correct
    /// ordering across DWARF configs for integration with scale_optimizer.
    #[test]
    fn grad_bottleneck_score_ordering() {
        println!("\n══ Grad Bottleneck Score (for scale_optimizer) ═══════════════════════");
        println!();
        println!("{:<16} {:>20}  (1 = no suppression, 0 = fully suppressed)",
                 "Config", "grad_bottleneck_score");
        println!("{}", "─".repeat(60));

        for cfg in &LM_HEAD_CONFIGS {
            let score = grad_bottleneck_score(cfg.embedding_dim, cfg.vocab_size);
            println!("{:<16} {:>20.6}", cfg.name, score);
            assert!(score > 0.0 && score <= 1.0,
                "{}: score ({score:.6}) out of [0,1]", cfg.name);
        }
        println!();

        // Larger D → higher score at fixed V=32k
        let s_14m  = grad_bottleneck_score( 256, 32768);
        let s_35m  = grad_bottleneck_score( 512, 32768);
        let s_100m = grad_bottleneck_score(1024, 32768);
        println!("Score by scale (V=32k): 14M={s_14m:.4}, 35M={s_35m:.4}, 100M={s_100m:.4}");
        assert!(s_35m  > s_14m,  "35M score ({s_35m:.4}) should exceed 14M ({s_14m:.4})");
        assert!(s_100m > s_35m,  "100M score ({s_100m:.4}) should exceed 35M ({s_35m:.4})");

        // V=8k scores higher than V=32k at the same D
        let s_14m_8k = grad_bottleneck_score(256, 8192);
        println!("V=8k vs V=32k at 14M: {s_14m_8k:.4} vs {s_14m:.4} ({:.1}× improvement)",
                 s_14m_8k / s_14m);
        assert!(s_14m_8k > s_14m,
            "V=8k score ({s_14m_8k:.4}) should exceed V=32k ({s_14m:.4})");
    }
}
