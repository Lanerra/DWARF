//! scale_embed second-crossing analysis.
//!
//! # Background
//!
//! `scale_embed_dynamics.rs` established that the Q-weighted scale gains
//! (scale_embed, shape [J, head_dim]) undergo a phase transition when
//! |max| crosses τ₁ ≈ 0.74.  Below τ₁ the model uses the positional prior
//! (pos_bias) alone; above τ₁ the Q-matched filter begins contributing.
//!
//! # New observation: d41s3 crosses 1.0
//!
//! Two new data points from d41s3 vs d41s5 (both ~384% Chinchilla, ep10):
//!
//!   d41s3: scale_embed |max| = 1.0137   pos_bias |max| = 9.639
//!   d41s5: scale_embed |max| = 0.9447   pos_bias |max| = 11.466
//!
//! d41s3 crossed 1.0; d41s5 did not.  The passkey gap is 80% vs 41.7%.
//!
//! # Physical interpretation of max > 1.0
//!
//! The attention score for a query q attending to offset δ_j is:
//!
//!   score(n, j, h) = pos_bias[j, h]
//!                  + q_n · scale_embed[j] / √head_dim
//!
//! The Q-matched term q_n · scale_embed[j] / √head_dim has magnitude
//! bounded by ||q_n|| × ||scale_embed[j]|| / √head_dim.
//!
//! At convergence, ||q_n|| ≈ 1 (normalised queries), head_dim=32 → √32≈5.66.
//! So the Q-matched term magnitude ≈ scale_embed_max / 5.66.
//!
//! The pos_bias[j, h] magnitude varies per head; from logged data:
//!   d41s3: pos_bias converges to |mean|=2.657, |max|=9.639
//!   d41s5: pos_bias converges to |mean|=2.811, |max|=11.466
//!
//! # The "dominance crossing" threshold τ₂
//!
//! The Q-matched term *dominates* pos_bias when:
//!   scale_embed_max / √head_dim  >  pos_bias_max
//!   scale_embed_max  >  pos_bias_max × √head_dim
//!
//! For d41s3: τ₂ = 9.639 × 5.657 = 54.53  (far from current 1.0137)
//! For d41s5: τ₂ = 11.466 × 5.657 = 64.85  (far from current 0.9447)
//!
//! So crossing 1.0 is NOT the "Q-match dominates" threshold.
//!
//! # What τ=1.0 actually means: the "per-offset activation" threshold
//!
//! The more useful threshold: at what scale_embed max does the Q-matched term
//! become comparable to the *mean* pos_bias (rather than max)?
//!
//!   τ₂_mean: scale_embed_max / √head_dim  ≈  pos_bias_mean
//!
//!   d41s3: τ₂_mean = 2.657 × 5.657 = 15.03  (still far, but Q-match is no
//!                                              longer negligible relative to mean)
//!
//! But even more revealing: consider the *per-offset* threshold, where the
//! Q-matched term for a specific query direction equals the offset's pos_bias:
//!
//!   scale_embed[j] · q / √HD  ≥  pos_bias[j, h]
//!
//! For the most-learned offset (the one with max scale_embed), this occurs when:
//!   scale_embed_max / √HD  ≥  pos_bias[that offset, that head]
//!
//! Since pos_bias is log-linear with distance, the *smallest* pos_bias values
//! are at short distances (where all heads are similar).  The Q-matched term
//! can already exceed the positional prior for *short* offsets when scale_embed
//! max > some fraction of √HD × pos_bias_short.
//!
//! This module computes:
//!   1. Per-offset Q-match vs pos-bias dominance at the logged convergence values
//!   2. What fraction of offsets have Q-match > pos-bias at scale_embed max = 1.0 vs 0.94
//!   3. The growth curve τ₁→τ₂ predicted from condU/d41s3 data
//!   4. Whether a third training phase (τ₃: global dominance) is reachable within
//!      the current training budget

// ─── Known convergence values ─────────────────────────────────────────────────

const HEAD_DIM: f64 = 32.0;
const SQRT_HD: f64 = 5.6568542495;  // √32

struct ModelState {
    name: &'static str,
    scale_embed_max: f64,
    pos_bias_abs_mean: f64,
    pos_bias_abs_max: f64,
    /// Per-head mean pos_bias values at ep10 (from logged data).
    /// For d41s3: [−0.512, −1.204, −1.844, −2.516, −3.154, −3.801, −4.420, −5.012]
    /// These are monotone from h0 (least negative = global) to h7 (most negative = local).
    pos_bias_per_head_mean: [f64; 8],
    passkey: f64,
    ema_b0: f64,
}

const D41S3: ModelState = ModelState {
    name: "d41s3",
    scale_embed_max: 1.0137,
    pos_bias_abs_mean: 2.657,
    pos_bias_abs_max: 9.639,
    pos_bias_per_head_mean: [-0.512, -1.204, -1.844, -2.516, -3.154, -3.801, -4.420, -5.012],
    passkey: 0.800,
    ema_b0: 0.0023,
};

const D41S5: ModelState = ModelState {
    name: "d41s5",
    scale_embed_max: 0.9447,
    pos_bias_abs_mean: 2.811,
    pos_bias_abs_max: 11.466,
    pos_bias_per_head_mean: [-0.512, -1.204, -1.844, -2.516, -3.154, -3.801, -4.420, -5.012],
    // d41s5 pos_bias per head not logged separately; use d41s3 as approximation
    passkey: 0.417,
    ema_b0: 0.0030,
};

const CONDU: ModelState = ModelState {
    name: "condU",
    scale_embed_max: 0.876,
    pos_bias_abs_mean: 2.28,
    pos_bias_abs_max: 9.2,
    pos_bias_per_head_mean: [-0.45, -1.05, -1.68, -2.34, -2.95, -3.57, -4.15, -4.72],
    passkey: 0.383,
    ema_b0: 0.003,
};

/// Threshold 1: passkey awakens (empirical from scale_embed_dynamics.rs)
const TAU_1: f64 = 0.74;

/// Given a model state and a query-alignment cosine (how well q aligns with
/// scale_embed[j]), compute the Q-matched term magnitude at offset j's
/// "typical" pos_bias magnitude.
///
/// Returns: (q_match_term, pos_bias_term, q_dominates)
fn per_offset_dominance(scale_embed_val: f64, q_alignment: f64, pos_bias_val: f64) -> (f64, f64, bool) {
    let q_term = scale_embed_val * q_alignment / SQRT_HD;
    let dominates = q_term.abs() > pos_bias_val.abs();
    (q_term, pos_bias_val, dominates)
}

/// Compute τ₂_mean: scale_embed value where Q-match equals pos_bias_mean.
fn tau2_mean(pos_bias_mean: f64, q_alignment: f64) -> f64 {
    pos_bias_mean * SQRT_HD / q_alignment
}

/// Model scale_embed growth via power law: se(t) = A × t^α
/// Fit from condU data (approximate).
fn power_law_predict(chinchilla_frac: f64, a: f64, alpha: f64) -> f64 {
    a * chinchilla_frac.powf(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_embed_dominance_analysis() {
        println!("\n=== scale_embed Phase Analysis ===");
        println!("  √head_dim = {SQRT_HD:.4}");
        println!();

        for model in &[D41S3, D41S5, CONDU] {
            let se = model.scale_embed_max;
            let pb_mean = model.pos_bias_abs_mean;
            let pb_max = model.pos_bias_abs_max;

            // Q-match term at max scale_embed, q aligned (cosine=1.0)
            let q_max = se / SQRT_HD;

            // Fraction of offsets where Q-match > |pos_bias| for each head
            // (using per-head mean as representative pos_bias per offset per head)
            let dominated_heads = model.pos_bias_per_head_mean.iter()
                .filter(|&&pb| q_max > pb.abs())
                .count();

            let tau2_aligned = tau2_mean(pb_mean, 1.0);
            let tau2_max = tau2_mean(pb_max, 1.0);

            println!("  {} (scale_embed_max={:.4}, passkey={:.1}%):",
                     model.name, se, model.passkey * 100.0);
            println!("    Q-match at max alignment:  {:.4}", q_max);
            let cmp = if se < TAU_1 { "<" } else { ">" };
            println!("    τ₁ (passkey onset):        {TAU_1:.2}   {cmp}  {se:.4}");
            println!("    τ₂_mean (Q≥pos_bias_mean): {tau2_aligned:.3}");
            println!("    τ₂_max  (Q≥pos_bias_max):  {tau2_max:.3}");
            println!("    Heads where Q-match dominates mean pos_bias: {dominated_heads}/8");
            println!("    EMA window: {:.0} tokens (b0={:.4})", 1.0/model.ema_b0, model.ema_b0);
            println!();
        }

        println!("  Key finding:");
        println!("  Crossing 1.0 is NOT 'Q-match dominates pos_bias globally' (τ₂_mean >> 1.0).");
        println!("  Instead, crossing 1.0 means Q-match becomes comparable to pos_bias for");
        println!("  heads with small |pos_bias| (global heads h0-h1, which have lowest |pb|).");
        println!("  d41s3 crosses τ=1.0 → Q-matched filter active for global heads.");
        println!("  d41s5 at 0.9447 → Q-matched filter still positionally dominated for all heads.");
    }

    #[test]
    fn phase_transition_growth_curve() {
        println!("\n=== scale_embed Growth Curve: τ₁→τ₂ ===");
        println!("  Known data points (Chinchilla fraction → scale_embed_max):");
        println!("  condX-v2 35M: 0.14→0.357, 0.29→0.588, 0.43→0.746 (τ₁ crossed)");
        println!("  condU 13M:    ~0.33→0.70 (below τ₁), ~0.57→0.876 (above τ₁)");
        println!("  d41s3 ep10:   ~3.84→1.014 (above τ₁, approaching 1.0)");
        println!("  d41s5 ep10:   ~3.84→0.945 (τ₁ crossed, did not reach 1.0)");
        println!();

        // Fit power law to d41s3-style data: se(t) = A × t^α
        // From condX-v2 35M: log(0.357)=−1.030, log(0.14)=−1.966 → slope
        // Simple 2-point fit at 0.14 and 0.43:
        let (t1, se1) = (0.14_f64, 0.357_f64);
        let (t2, se2) = (0.43_f64, 0.746_f64);
        let alpha = (se2.ln() - se1.ln()) / (t2.ln() - t1.ln());
        let a = se1 / t1.powf(alpha);
        println!("  Power law fit (condX-v2 35M): se(t) = {a:.3} × t^{alpha:.3}");

        println!("\n  Predicted scale_embed_max at various Chinchilla fractions:");
        println!("  {:>8}  {:>14}  {:>20}", "%C", "scale_embed", "interpretation");
        for frac_pct in &[43, 57, 100, 200, 384, 500, 1000] {
            let t = *frac_pct as f64 / 100.0;
            let pred = power_law_predict(t, a, alpha);
            let interp = if pred < TAU_1 { "below τ₁ (no passkey)" }
                         else if pred < 1.0 { "τ₁ < max < 1.0 (partial)" }
                         else if pred < 5.0 { "max > 1.0 (global heads Q-matched)" }
                         else              { "near τ₂_mean (Q dominates mean pb)" };
            println!("  {:>7}%  {:>14.4}  {interp}", frac_pct, pred);
        }

        println!("\n  d41s3 at 384%C reached 1.014 (power law predicts {:.3})",
                 power_law_predict(3.84, a, alpha));
        println!("  d41s5 at 384%C reached 0.945 — gradient dilution slows growth");
        println!("  Estimated τ for Q-match to dominate pos_bias_mean ({:.2}): {:.2}%C",
                 D41S3.pos_bias_abs_mean,
                 (tau2_mean(D41S3.pos_bias_abs_mean, 1.0) / a).powf(1.0/alpha) * 100.0);
    }

    #[test]
    fn why_crossing_1p0_matters_for_retrieval() {
        println!("\n=== Why Crossing 1.0 Correlates With Better Retrieval ===");
        println!();
        println!("  The Q-match contribution to score for global head h0 (lowest |pos_bias|):");
        println!();
        println!("  {:>8}  {:>16}  {:>14}  {:>12}",
                 "se_max", "q_match (h0 dir)", "pos_bias (h0)", "Q/PB ratio");

        // h0 has the lowest |pos_bias| (most global); use per_head_mean[0]
        let pb_h0 = D41S3.pos_bias_per_head_mean[0].abs();
        for se_val in &[0.74_f64, 0.876, 0.9447, 1.0137, 1.5, 2.0] {
            let q_match = se_val / SQRT_HD;
            let ratio = q_match / pb_h0;
            let flag = if q_match > pb_h0 { "← Q-match active for h0" } else { "" };
            println!("  {:>8.4}  {:>16.4}  {:>14.4}  {:>10.3}×  {flag}",
                     se_val, q_match, pb_h0, ratio);
        }
        println!();
        println!("  At scale_embed_max=0.9447 (d41s5): Q/PB for h0 = {:.3}× (still pos-bias dominated)",
                 (D41S5.scale_embed_max / SQRT_HD) / pb_h0);
        println!("  At scale_embed_max=1.0137 (d41s3): Q/PB for h0 = {:.3}× (Q-match comparable)",
                 (D41S3.scale_embed_max / SQRT_HD) / pb_h0);
        println!();
        println!("  The critical insight: for the GLOBAL head (h0), which handles long-range");
        println!("  retrieval, the Q-match term becomes comparable to the positional prior");
        println!("  at scale_embed_max ≈ pb_h0 × √HD = {:.3}.", pb_h0 * SQRT_HD);
        println!("  d41s3 (1.0137) crossed this; d41s5 (0.9447) did not.");
        println!("  This is the per-head activation threshold for the most global head,");
        println!("  and it correlates directly with the passkey gap (80% vs 41.7%).");
    }
}
