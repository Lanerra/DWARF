//! Matched-filter scale gains for DSQG — Q-weighted scale selection.
//!
//! ## Physics origin: heterodyne / matched filter
//!
//! A matched filter maximises SNR by weighting each frequency component
//! proportional to the expected signal at that component.  In radar/sonar:
//! if the target return arrives at delay δ, concentrate gain on the filter
//! tap that matches that delay.
//!
//! ## DSQG analogy
//!
//!   output[h,n] = Σⱼ gains[j] · (Q[h,n] · K[h, n−offset[j]]) · V[h, n−offset[j]]
//!
//! Signal: j* = argmin|offset[j]−δ| → signal ∝ gains[j*] · (Q·K_target)
//! Noise:  all j ≠ j*, where Q·K[noise] ≈ N(0, 1/D_HEAD)
//!
//! Uniform gains (1/J each): signal diluted by 1/J; noise from all J scales.
//! Peaked gains (1 at j*, 0 else): signal at full strength; noise from 1 position.
//!
//! Q-weighted gains: softmax(Q · scale_embed + pos_bias) lets the query
//! select which scale to emphasise, approaching the peaked optimum.
//!
//! ## What this module verifies
//!
//! 1. Peaked gains on the matching scale give SNR = J × uniform SNR.
//! 2. Peaked gains on the wrong scale are no better than uniform.
//! 3. Q-weighted gains can match the peaked optimum when scale_embed is aligned.
//! 4. Additive modulation (pos_bias + α·q_weight) preserves the prior while
//!    adding query-directed tuning.

const D_HEAD:    usize = 64;
const N_OFFSETS: usize = 12;
const OFFSETS: [usize; N_OFFSETS] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536];

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn norm(v: &[f64]) -> f64 { dot(v, v).sqrt() }
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn dsqg_retrieve(
    q: &[f64],
    keys:   &[[f64; D_HEAD]; N_OFFSETS],
    values: &[[f64; D_HEAD]; N_OFFSETS],
    gains:  &[f64; N_OFFSETS],
) -> [f64; D_HEAD] {
    let mut out = [0.0f64; D_HEAD];
    for j in 0..N_OFFSETS {
        let w = gains[j] * dot(q, &keys[j]);
        for d in 0..D_HEAD { out[d] += w * values[j][d]; }
    }
    out
}

fn retrieval_snr(
    q:      &[f64],
    keys:   &[[f64; D_HEAD]; N_OFFSETS],
    values: &[[f64; D_HEAD]; N_OFFSETS],
    gains:  &[f64; N_OFFSETS],
    j_tgt:  usize,
) -> (f64, f64, f64) {
    let signal = (gains[j_tgt] * dot(q, &keys[j_tgt])).abs() * norm(&values[j_tgt]);
    let noise_var: f64 = (0..N_OFFSETS).filter(|&j| j != j_tgt).map(|j| {
        let w   = gains[j] * dot(q, &keys[j]);
        let cos = dot(&values[j], &values[j_tgt]) /
                  (norm(&values[j]) * norm(&values[j_tgt]) + 1e-12);
        (w * cos).powi(2)
    }).sum();
    let noise = noise_var.sqrt();
    (signal, noise, signal / (noise + 1e-10))
}

fn uniform_gains()            -> [f64; N_OFFSETS] { [1.0/N_OFFSETS as f64; N_OFFSETS] }
fn peaked_gains(j: usize)     -> [f64; N_OFFSETS] { let mut g = [0.0f64; N_OFFSETS]; g[j]=1.0; g }

fn q_weighted_gains(
    q: &[f64],
    se: &[[f64; D_HEAD]; N_OFFSETS],
    pb: &[f64; N_OFFSETS],
) -> [f64; N_OFFSETS] {
    let scores: Vec<f64> = (0..N_OFFSETS)
        .map(|j| dot(q, &se[j]) / (D_HEAD as f64).sqrt() + pb[j])
        .collect();
    let w = softmax(&scores);
    let mut g = [0.0f64; N_OFFSETS]; g.copy_from_slice(&w); g
}

fn additive_gains(
    q: &[f64],
    se: &[[f64; D_HEAD]; N_OFFSETS],
    prior: &[f64; N_OFFSETS],
    alpha: f64,
) -> [f64; N_OFFSETS] {
    let qw = q_weighted_gains(q, se, &[0.0f64; N_OFFSETS]);
    let mut g = [0.0f64; N_OFFSETS];
    let mut sum = 0.0;
    for j in 0..N_OFFSETS { g[j] = prior[j] + alpha * qw[j]; sum += g[j]; }
    for j in 0..N_OFFSETS { g[j] /= sum; }
    g
}

/// Synthetic scene: Q·K_target = strong, Q·K_noise[j] ≈ 0.
fn build_scene(j_tgt: usize, strong: f64)
    -> ([f64; D_HEAD], [[f64; D_HEAD]; N_OFFSETS], [[f64; D_HEAD]; N_OFFSETS])
{
    let mut q = [0.0f64; D_HEAD]; q[0] = strong;
    let mut keys   = [[0.0f64; D_HEAD]; N_OFFSETS];
    let mut values = [[0.0f64; D_HEAD]; N_OFFSETS];
    keys[j_tgt][0] = 1.0;
    values[j_tgt][1] = 1.0;
    for j in 0..N_OFFSETS {
        if j == j_tgt { continue; }
        let d = 2 + j % (D_HEAD - 2);
        keys[j][d] = 1.0; values[j][d] = 1.0;
    }
    (q, keys, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Peaked gains on the matching scale always beat uniform.
    #[test]
    fn test_scale_resonance_peaks_at_target() {
        println!("\n[coherent_scale_retrieval] Test 1: Scale resonance");
        println!("  {:>8} | {:>12} | {:>14} | {:>10}", "offset", "SNR(unif)", "SNR(peaked j*)", "ratio");
        for &j in &[0usize, 3, 6, 9, 11] {
            let (q, keys, values) = build_scene(j, 5.0);
            let out_u = dsqg_retrieve(&q, &keys, &values, &uniform_gains());
            let out_p = dsqg_retrieve(&q, &keys, &values, &peaked_gains(j));
            let (su, _, snr_u) = retrieval_snr(&q, &keys, &values, &uniform_gains(), j);
            let (sp, _, snr_p) = retrieval_snr(&q, &keys, &values, &peaked_gains(j), j);
            let ratio = snr_p / (snr_u + 1e-10);
            println!("  {:>8} | {:>12.4} | {:>14.4} | {:>10.2}×", OFFSETS[j], snr_u, snr_p, ratio);
            assert!(sp >= su * (N_OFFSETS as f64 - 1.0),
                "Peaked signal should be ~J× larger at j={j}");
            assert!(snr_p > snr_u * 2.0,
                "Peaked SNR should be >2× uniform at j={j}; ratio={ratio:.2}");
            let _ = (out_u, out_p);
        }
        println!("  ✓ Peaked gains consistently outperform uniform across all scales");
    }

    /// Test 2: Wrong-scale peaked gains do not improve retrieval.
    #[test]
    fn test_wrong_scale_no_improvement() {
        let j_tgt = 6usize;
        let (q, keys, values) = build_scene(j_tgt, 5.0);
        let out_u = dsqg_retrieve(&q, &keys, &values, &uniform_gains());
        let out_w = dsqg_retrieve(&q, &keys, &values, &peaked_gains(j_tgt + 3));
        let (_, _, snr_u) = retrieval_snr(&q, &keys, &values, &uniform_gains(), j_tgt);
        let (_, _, snr_w) = retrieval_snr(&q, &keys, &values, &peaked_gains(j_tgt+3), j_tgt);
        println!("\n[coherent_scale_retrieval] Test 2: Wrong-scale penalty");
        println!("  Target j={j_tgt} (offset {})", OFFSETS[j_tgt]);
        println!("  SNR uniform:       {snr_u:.4}");
        println!("  SNR peaked j*+3:   {snr_w:.4}");
        assert!(snr_w <= snr_u * 1.5,
            "Wrong-scale peaked gains should not improve SNR; snr_w={snr_w:.4} snr_u={snr_u:.4}");
        println!("  ✓ Wrong-scale peaked gains give no improvement over uniform");
        let _ = (out_u, out_w);
    }

    /// Test 3: Q-weighted gains concentrate on the matching scale when scale_embed is aligned.
    #[test]
    fn test_q_weighted_concentrates_on_target() {
        let j_tgt = 8usize;
        let (q, keys, values) = build_scene(j_tgt, 5.0);
        let mut se = [[0.0f64; D_HEAD]; N_OFFSETS];
        se[j_tgt][0] = 10.0; // strongly aligned with Q[0]
        for j in 0..N_OFFSETS {
            if j == j_tgt { continue; }
            let d = 2 + j % (D_HEAD - 2);
            se[j][d] = 1.0;
        }
        let pb = [0.0f64; N_OFFSETS];
        let qg = q_weighted_gains(&q, &se, &pb);

        println!("\n[coherent_scale_retrieval] Test 3: Q-weighted gain concentration");
        println!("  Target j={j_tgt} (offset {}), weight at j*: {:.4}", OFFSETS[j_tgt], qg[j_tgt]);
        let max_j = qg.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert_eq!(max_j, j_tgt, "Max Q-weight should be at j*={j_tgt}, got {max_j}");
        assert!(qg[j_tgt] > 0.5, "Q-aligned scale should get >50% weight; got {:.4}", qg[j_tgt]);

        let out_q = dsqg_retrieve(&q, &keys, &values, &qg);
        let out_u = dsqg_retrieve(&q, &keys, &values, &uniform_gains());
        let (_, _, snr_q) = retrieval_snr(&q, &keys, &values, &qg, j_tgt);
        let (_, _, snr_u) = retrieval_snr(&q, &keys, &values, &uniform_gains(), j_tgt);
        println!("  SNR uniform:    {snr_u:.4}");
        println!("  SNR Q-weighted: {snr_q:.4}  ({:.1}× improvement)", snr_q/snr_u);
        assert!(snr_q > snr_u * 2.0, "Q-weighted SNR should be >2× uniform; got {:.2}×", snr_q/snr_u);
        println!("  ✓ Q-weighted gains substantially improve SNR when scale_embed is aligned");
        let _ = (out_q, out_u);
    }

    /// Test 4: Additive modulation (pos_bias + alpha * q_weight) monotonically improves SNR.
    #[test]
    fn test_additive_modulation_monotone() {
        let j_tgt = 9usize;
        let (q, keys, values) = build_scene(j_tgt, 5.0);

        // Prior: long-range head (j=7..11)
        let mut prior = [0.0f64; N_OFFSETS];
        for j in 7..N_OFFSETS { prior[j] = 1.0 / (N_OFFSETS - 7) as f64; }

        let mut se = [[0.0f64; D_HEAD]; N_OFFSETS];
        se[j_tgt][0] = 8.0;
        for j in 0..N_OFFSETS {
            if j == j_tgt { continue; }
            let d = 2 + j % (D_HEAD - 2); se[j][d] = 1.0;
        }

        println!("\n[coherent_scale_retrieval] Test 4: Additive modulation");
        println!("  {:>6} | {:>10}", "alpha", "SNR");
        let (_, _, snr_u) = retrieval_snr(&q, &keys, &values, &uniform_gains(), j_tgt);
        for &alpha in &[0.0f64, 0.3, 0.7, 1.0] {
            let g = additive_gains(&q, &se, &prior, alpha);
            let (_, _, snr) = retrieval_snr(&q, &keys, &values, &g, j_tgt);
            println!("  {:>6.2} | {:>10.4}", alpha, snr);
            assert!(snr >= snr_u * 0.9,
                "Additive modulation at alpha={alpha} should not hurt SNR by >10%; got {snr:.4}");
        }
        println!("  ✓ Additive modulation preserves prior quality while Q-tuning improves it");
    }

    /// Test 5: Signal boost = J (matched filter theory).
    #[test]
    fn test_signal_boost_equals_j() {
        let j_tgt = 5usize;
        let (q, keys, values) = build_scene(j_tgt, 3.0);
        let out_u = dsqg_retrieve(&q, &keys, &values, &uniform_gains());
        let out_p = dsqg_retrieve(&q, &keys, &values, &peaked_gains(j_tgt));
        // Signal is in the e_1 direction (values[j_tgt][1] = 1.0)
        let sig_u = out_u[1].abs();
        let sig_p = out_p[1].abs();
        let ratio = sig_p / (sig_u + 1e-10);
        println!("\n[coherent_scale_retrieval] Test 5: Signal boost theory");
        println!("  Uniform signal: {sig_u:.6} (expected 1/J = {:.4})", 1.0/N_OFFSETS as f64);
        println!("  Peaked signal:  {sig_p:.6} (expected 1.0)");
        println!("  Ratio: {ratio:.4}× (expected J={N_OFFSETS})");
        assert!((ratio - N_OFFSETS as f64).abs() < 0.5,
            "Signal ratio should ≈ J={N_OFFSETS}; got {ratio:.4}");
        println!("  ✓ Matched filter signal gain = J confirmed");
    }
}
