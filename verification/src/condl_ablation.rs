//! condL design ablation tests.
//!
//! Before scripting condL (DSQG training run), two questions need empirical answers:
//!
//! ## Test 1 — ELU normalizer: load-bearing for DSQG or a no-op?
//!
//! condK added a linear-attention ELU normalizer (φ(x) = elu(x)+1, output/Z)
//! to prevent magnitude blow-up in the unnormalised OPWF gather Q@F_prop.
//! DSQG uses softmax(Q·K/√d) over offsets, producing a convex combination of V
//! vectors — already bounded by max‖V‖. Does ELU still materially change output
//! variance for DSQG outputs?
//!
//! Decision rule (printed):
//!   - DSQG variance ratio (post/pre ELU) within ±10% of 1.0 → omit from condL
//!   - Otherwise → keep ELU in condL for fair comparison
//!
//! ## Test 2 — RG init: does scale separation survive early gradient updates?
//!
//! condK's RG-inspired init pre-biases early layers toward fine scales (j0–j3)
//! and late layers toward coarse scales (j8–j10). This test measures:
//!   (a) Scale separation (CoM_last − CoM_first) for RG init vs random init at t=0
//!   (b) Gradient-descent steps for random init to reach 90% of RG separation
//!   (c) Whether the CoM ordering survives Gaussian noise at 5%–100% of gain magnitude
//!
//! Decision rule (printed):
//!   - <100 GD steps to match RG separation → RG init doesn't matter much; skip ablation
//!   - >200 GD steps → RG init provides meaningful head start; isolate in GPU ablation
//!   - Ordering survives 50%+ noise → robust inductive bias; ordering breaks at <20% → fragile

use crate::rg_init::RgGainSchedule;

// ── condK 13M configuration constants ────────────────────────────────────────
const D: usize = 32;          // head dim = 256 hidden / 8 heads
const N_OFFSETS: usize = 11;  // j = 0..10, offsets 1, 2, 4, 8, …, 1024
const N_LAYERS: usize = 6;
const N_SCALES: usize = 11;   // log₂(2048 field length)

// ── Deterministic PRNG (LCG + Box-Muller) ────────────────────────────────────
// No external crates needed; fully reproducible across runs.

fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005)
                  .wrapping_add(1_442_695_040_888_963_407);
    let bits = 0x3F80_0000u32 | ((*state >> 41) as u32 & 0x007F_FFFF);
    f32::from_bits(bits) - 1.0   // uniform in [0, 1)
}

fn randn(state: &mut u64) -> f32 {
    // Box-Muller transform: two uniform → one standard normal
    let u1 = lcg_next(state).max(1e-10);
    let u2 = lcg_next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// φ(x) = elu(x) + 1  (kernel function for linear attention normaliser).
///
///   x ≥ 0  →  φ(x) = x + 1       (linear, always > 0)
///   x < 0  →  φ(x) = exp(x)      (= elu(x)+1; guaranteed ∈ (0,1))
///
/// Returns (φ_vec, Z) where Z = Σ φ(xᵢ) ≥ 0.  Used to normalise gather output.
fn elu_phi(x: &[f32]) -> (Vec<f32>, f32) {
    let phi: Vec<f32> = x.iter()
        .map(|&v| if v >= 0.0 { v + 1.0 } else { v.exp() })
        .collect();
    let z = phi.iter().sum::<f32>().max(1e-9);
    (phi, z)
}

/// Stable softmax.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let s = exp.iter().sum::<f32>().max(1e-9);
    exp.iter().map(|&e| e / s).collect()
}

/// Sample variance of a slice.
fn variance(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mu = x.iter().sum::<f32>() / n;
    x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / n
}

/// Gain-weighted centre-of-mass over scale indices.
fn scale_com(gains: &[f32]) -> f32 {
    let total: f32 = gains.iter().sum::<f32>().max(1e-9);
    gains.iter().enumerate()
         .map(|(s, &g)| s as f32 * g)
         .sum::<f32>() / total
}

/// Scale separation of a gain schedule: CoM(last_layer) − CoM(first_layer).
fn separation(gains_all: &[Vec<f32>]) -> f32 {
    scale_com(gains_all.last().unwrap()) - scale_com(gains_all.first().unwrap())
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — ELU normaliser variance effect
    // ─────────────────────────────────────────────────────────────────────────

    /// Compares output-variance ratio (after ELU / before ELU) for:
    ///   - OPWF-style outputs: Q@F_prop, D-dim, std ~ 0.5 (unbounded)
    ///   - DSQG-style outputs: softmax-weighted convex combination of V (bounded)
    ///
    /// OPWF: ELU normaliser divides by Z = Σφ(output[i]).  For outputs with
    /// typical DWARF init scale (~0.5 per element, D=32 elements), Z ~ 32×1.5 ~ 48,
    /// so post-ELU scale ~ 0.5/48 — significant compression.
    ///
    /// DSQG: output is Σ_j softmax(scores)[j] · V[j], a convex combination.
    /// Each V element is O(0.5), softmax weights sum to 1, so output elements
    /// are O(0.5) regardless of N_OFFSETS.  Z ~ 32×1.5 ~ 48 again, so ELU
    /// still compresses — but the key question is whether the ratio is *different*
    /// enough from OPWF's to matter for design.
    ///
    /// What we actually care about: do DSQG outputs need the ELU normaliser
    /// for scale stability, or does softmax already handle that?
    #[test]
    fn elu_normalizer_variance_effect() {
        let mut rng = 12_345u64;
        let n_trials = 2_000usize;

        let (mut opwf_pre_sum, mut opwf_post_sum) = (0.0f32, 0.0f32);
        let (mut dsqg_pre_sum, mut dsqg_post_sum) = (0.0f32, 0.0f32);
        // Also track mean |output| to understand scale, not just variance
        let (mut opwf_mag_pre, mut opwf_mag_post) = (0.0f32, 0.0f32);
        let (mut dsqg_mag_pre, mut dsqg_mag_post) = (0.0f32, 0.0f32);

        for _ in 0..n_trials {
            // ── OPWF-style output ─────────────────────────────────────────────
            // Q@F_prop: D-dimensional vector.  With random Q (std 1/√D ≈ 0.18)
            // and F_prop accumulating K⊗V outer products, a reasonable init
            // scale is std ~ 0.5 per element.
            let opwf_out: Vec<f32> = (0..D).map(|_| randn(&mut rng) * 0.5).collect();
            let (_, z_opwf) = elu_phi(&opwf_out);
            let opwf_normed: Vec<f32> = opwf_out.iter().map(|&v| v / z_opwf).collect();

            opwf_pre_sum  += variance(&opwf_out);
            opwf_post_sum += variance(&opwf_normed);
            opwf_mag_pre  += opwf_out.iter().map(|v| v.abs()).sum::<f32>() / D as f32;
            opwf_mag_post += opwf_normed.iter().map(|v| v.abs()).sum::<f32>() / D as f32;

            // ── DSQG-style output ─────────────────────────────────────────────
            // Softmax(Q·K/√d) weighted sum of V vectors.  The output is a
            // convex combination of V vectors (weights sum to 1), so its
            // L∞ norm is bounded by max‖V‖.  For V ~ N(0, 0.5), each element
            // is O(0.5) regardless of N_OFFSETS.
            let scores: Vec<f32> = (0..N_OFFSETS).map(|_| randn(&mut rng)).collect();
            let w = softmax(&scores);
            let vs: Vec<Vec<f32>> = (0..N_OFFSETS)
                .map(|_| (0..D).map(|_| randn(&mut rng) * 0.5).collect::<Vec<_>>())
                .collect();
            let dsqg_out: Vec<f32> = (0..D)
                .map(|d| w.iter().zip(&vs).map(|(&wi, v)| wi * v[d]).sum())
                .collect();
            let (_, z_dsqg) = elu_phi(&dsqg_out);
            let dsqg_normed: Vec<f32> = dsqg_out.iter().map(|&v| v / z_dsqg).collect();

            dsqg_pre_sum  += variance(&dsqg_out);
            dsqg_post_sum += variance(&dsqg_normed);
            dsqg_mag_pre  += dsqg_out.iter().map(|v| v.abs()).sum::<f32>() / D as f32;
            dsqg_mag_post += dsqg_normed.iter().map(|v| v.abs()).sum::<f32>() / D as f32;
        }

        let n = n_trials as f32;
        let (opwf_pre, opwf_post) = (opwf_pre_sum / n, opwf_post_sum / n);
        let (dsqg_pre, dsqg_post) = (dsqg_pre_sum / n, dsqg_post_sum / n);
        let (om_pre, om_post)     = (opwf_mag_pre / n, opwf_mag_post / n);
        let (dm_pre, dm_post)     = (dsqg_mag_pre / n, dsqg_mag_post / n);

        let opwf_var_ratio = opwf_post / opwf_pre.max(1e-9);
        let dsqg_var_ratio = dsqg_post / dsqg_pre.max(1e-9);
        let opwf_mag_ratio = om_post   / om_pre.max(1e-9);
        let dsqg_mag_ratio = dm_post   / dm_pre.max(1e-9);

        println!("\n══ ELU Normaliser Variance Effect ════════════════════════════════════");
        println!("                  │ Var pre  │ Var post │ Var ratio │ |x| pre │ |x| post │ Mag ratio");
        println!("──────────────────┼──────────┼──────────┼───────────┼─────────┼──────────┼──────────");
        println!("OPWF (Q@F_prop)   │ {opwf_pre:8.5} │ {opwf_post:8.5} │ {opwf_var_ratio:9.4} │ {om_pre:7.5} │ {om_post:8.5} │ {opwf_mag_ratio:8.4}");
        println!("DSQG (softmax·V)  │ {dsqg_pre:8.5} │ {dsqg_post:8.5} │ {dsqg_var_ratio:9.4} │ {dm_pre:7.5} │ {dm_post:8.5} │ {dsqg_mag_ratio:8.4}");
        println!();

        // The variance ratio tells us how much ELU *changes* the output distribution.
        // But both architectures will see the same Z denominator structure, so the
        // real question is whether they need it for different reasons:
        //   - OPWF: without ELU, Z can be near-zero if output elements are all negative
        //           (linear attention instability: denominator collapse)
        //   - DSQG: softmax-weighted V cannot easily have all-negative outputs unless V
        //           itself is all-negative. The risk of denominator collapse is lower.

        // Check for denominator collapse risk: how often is Z < 1.0?
        let mut opwf_collapse_count = 0u32;
        let mut dsqg_collapse_count = 0u32;
        let mut rng2 = 99_999u64;
        for _ in 0..n_trials {
            let opwf: Vec<f32> = (0..D).map(|_| randn(&mut rng2) * 0.5).collect();
            let (_, z) = elu_phi(&opwf);
            if z < D as f32 * 0.5 { opwf_collapse_count += 1; }  // Z below half expected

            let scores: Vec<f32> = (0..N_OFFSETS).map(|_| randn(&mut rng2)).collect();
            let w = softmax(&scores);
            let vs: Vec<Vec<f32>> = (0..N_OFFSETS)
                .map(|_| (0..D).map(|_| randn(&mut rng2) * 0.5).collect::<Vec<_>>())
                .collect();
            let dsqg: Vec<f32> = (0..D)
                .map(|d| w.iter().zip(&vs).map(|(&wi, v)| wi * v[d]).sum())
                .collect();
            let (_, zd) = elu_phi(&dsqg);
            if zd < D as f32 * 0.5 { dsqg_collapse_count += 1; }
        }

        let opwf_collapse_pct = opwf_collapse_count as f32 / n_trials as f32 * 100.0;
        let dsqg_collapse_pct = dsqg_collapse_count as f32 / n_trials as f32 * 100.0;
        println!("Denominator near-collapse risk (Z < D/2 = {}):", D / 2);
        println!("  OPWF: {opwf_collapse_pct:.1}% of trials");
        println!("  DSQG: {dsqg_collapse_pct:.1}% of trials");
        println!();

        // ── Conclusions ───────────────────────────────────────────────────────
        let dsqg_elu_no_op    = (dsqg_var_ratio - 1.0).abs() < 0.10;
        let dsqg_lower_risk   = dsqg_collapse_pct < opwf_collapse_pct * 0.5;

        println!("── Conclusions ──────────────────────────────────────────────────────");
        println!("  DSQG variance ratio: {dsqg_var_ratio:.4}  (1.0 = no effect; ±10% = near-noop)");
        println!("  ELU is near-noop for DSQG outputs: {dsqg_elu_no_op}");
        println!("  DSQG has substantially lower collapse risk than OPWF: {dsqg_lower_risk}");
        if dsqg_elu_no_op && dsqg_lower_risk {
            println!("  → OMIT ELU normaliser from condL (softmax already handles normalisation)");
        } else if dsqg_elu_no_op {
            println!("  → ELU changes variance little, but collapse risk is similar — consider keeping for consistency");
        } else {
            println!("  → KEEP ELU in condL (still affecting output scale materially)");
        }

        // Hard assertion: at least one architecture must show material ELU effect,
        // otherwise the normaliser is useless in both and should be removed from condK too.
        let either_has_effect = (opwf_var_ratio - 1.0).abs() > 0.05
                              || (dsqg_var_ratio - 1.0).abs() > 0.05;
        assert!(
            either_has_effect,
            "ELU normaliser has <5% variance effect on BOTH architectures — may be dead code"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — RG init scale-separation persistence
    // ─────────────────────────────────────────────────────────────────────────

    /// Measures whether RG-inspired init provides a meaningful inductive-bias
    /// head start over random init, and whether that head start survives early
    /// gradient updates.
    ///
    /// Three sub-measurements:
    ///
    /// (A) Separation gap at t=0: CoM(L_last) − CoM(L_first) for RG vs random.
    ///     RG init should have substantially larger separation.
    ///
    /// (B) Gradient steps (L2 loss toward RG target) for random init to reach
    ///     90% of RG separation.  Proxy for "how much training budget does RG
    ///     init save?"
    ///     - <100 steps at lr=0.05 → init barely matters (easy to learn)
    ///     - >500 steps → meaningful head start
    ///
    /// (C) Noise survival: apply N(0, σ) to RG gains (clamped ≥ 0) at increasing
    ///     σ as a fraction of mean gain magnitude.  Does the CoM ordering
    ///     (L0 < L1 < … < L_last) survive?  This proxies for how quickly
    ///     gradient noise in early training would erase the inductive bias.
    ///     - Survives 50%+ noise → robust, likely to guide training meaningfully
    ///     - Breaks at <20% noise → fragile, probably washes out in epoch 1
    #[test]
    fn rg_init_scale_separation_persistence() {
        let (base, decay) = RgGainSchedule::empirical_params_12layer();
        let scale_width   = N_SCALES as f32 / 3.0;
        let rg = RgGainSchedule::new(N_LAYERS, N_SCALES, base, decay, scale_width);

        let rg_sep = separation(&rg.gains);
        let rg_mean_mag: f32 = rg.gains.iter()
            .flat_map(|g| g.iter().cloned())
            .sum::<f32>() / (N_LAYERS * N_SCALES) as f32;

        println!("\n══ RG Init Scale Separation Persistence ══════════════════════════════");
        println!("Config: {N_LAYERS} layers, {N_SCALES} scales (condK 13M)");
        println!("RG init:  CoM L0 = {:.3},  CoM L{} = {:.3},  separation = {rg_sep:.3} scale indices",
                 rg.scale_center_of_mass(0),
                 N_LAYERS - 1,
                 rg.scale_center_of_mass(N_LAYERS - 1));
        println!("RG mean gain magnitude = {rg_mean_mag:.4}");

        // ── Part A: random init starting separation ───────────────────────────
        let mut rng = 54_321u64;
        let gains_rand_init: Vec<Vec<f32>> = (0..N_LAYERS)
            .map(|_| (0..N_SCALES)
                .map(|_| (lcg_next(&mut rng) * rg_mean_mag * 2.0).max(0.0))
                .collect())
            .collect();

        let rand_sep = separation(&gains_rand_init);
        let sep_advantage = rg_sep / rand_sep.abs().max(0.01);
        println!("\n── (A) Separation at t=0 ────────────────────────────────────────────");
        println!("  Random init separation = {rand_sep:.3}  (RG has {rg_sep:.3}, {sep_advantage:.1}× advantage)");

        // ── Part B: GD steps for random init to reach 90% of RG separation ───
        let mut gains_gd = gains_rand_init.clone();
        let lr = 0.05f32;
        let mut steps_to_match: Option<usize> = None;
        let target = rg_sep * 0.90;

        for step in 0..10_000usize {
            for l in 0..N_LAYERS {
                for s in 0..N_SCALES {
                    let grad = 2.0 * (gains_gd[l][s] - rg.gains[l][s]);
                    gains_gd[l][s] = (gains_gd[l][s] - lr * grad).max(0.0);
                }
            }
            if steps_to_match.is_none() {
                let sep = separation(&gains_gd);
                if sep >= target {
                    steps_to_match = Some(step + 1);
                }
            }
        }

        println!("\n── (B) GD steps to reach 90% of RG separation (lr={lr}) ────────────");
        match steps_to_match {
            Some(s) => {
                println!("  Steps needed: {s}");
                if s < 100 {
                    println!("  → RG separation is EASY to learn; init advantage is small");
                } else if s < 500 {
                    println!("  → Moderate head start (~{s} steps saved); worth keeping");
                } else {
                    println!("  → Large head start ({s}+ steps saved); RG init is meaningful");
                }
            }
            None => {
                println!("  Never reached 90% of RG separation within 10,000 steps (lr={lr})");
                println!("  → RG init provides structure that random init cannot easily recover");
            }
        }

        // ── Part C: noise survival ────────────────────────────────────────────
        println!("\n── (C) CoM ordering survival under Gaussian noise ───────────────────");
        println!("  (Noise applied once to all gains; gains clamped ≥ 0)");
        println!("  {:<14} {:<12} {:<20} Conclusion", "Noise σ/μ", "Ordering OK?", "Separation after");

        let noise_fractions: &[f32] = &[0.05, 0.10, 0.20, 0.50, 1.00, 2.00];
        for &frac in noise_fractions {
            // Use a different seed for each noise level, but deterministic
            let seed = 11_111u64 + (frac * 1000.0) as u64;
            let mut rng2 = seed;

            let noisy: Vec<Vec<f32>> = rg.gains.iter()
                .map(|layer| layer.iter()
                    .map(|&g| (g + randn(&mut rng2) * frac * rg_mean_mag).max(0.0))
                    .collect())
                .collect();

            let coms: Vec<f32> = (0..N_LAYERS).map(|l| scale_com(&noisy[l])).collect();
            let sep_after = coms[N_LAYERS - 1] - coms[0];

            // Ordering "survives" if every layer's CoM is no more than 0.5 scale
            // indices below the previous layer's CoM (allow small reversals from noise).
            let ordering_ok = coms.windows(2).all(|w| w[1] >= w[0] - 0.5);

            let conclusion = match (ordering_ok, frac) {
                (true, f) if f >= 0.50 => "robust inductive bias",
                (true, _)              => "ordering preserved",
                (false, f) if f < 0.20 => "FRAGILE — washes out fast",
                (false, _)             => "ordering broken at high noise",
            };

            println!("  {:<14.0}%  {:<12} {:<20.3} {}",
                     frac * 100.0,
                     if ordering_ok { "YES" } else { "NO" },
                     sep_after,
                     conclusion);
        }

        // ── Assertions ────────────────────────────────────────────────────────
        // These encode the minimum bar for RG init to be meaningful at all.

        assert!(
            rg_sep > 2.0,
            "RG init separation ({rg_sep:.3}) should exceed 2 scale indices; \
             if not, RG init is barely different from uniform init"
        );

        // Random init should start with less separation than RG (by design).
        // If rand_sep > rg_sep, random init accidentally has the right structure
        // and RG init is not providing any inductive bias.
        assert!(
            rand_sep < rg_sep,
            "Random init separation ({rand_sep:.3}) should be less than RG ({rg_sep:.3}); \
             if not, RG init provides no advantage over random"
        );

        // Ordering should survive 10% noise (very small perturbation).
        // If it doesn't, the schedule is too narrow to be useful as an init.
        {
            let mut rng3 = 77_777u64;
            let noisy_10: Vec<Vec<f32>> = rg.gains.iter()
                .map(|layer| layer.iter()
                    .map(|&g| (g + randn(&mut rng3) * 0.10 * rg_mean_mag).max(0.0))
                    .collect())
                .collect();
            let coms_10: Vec<f32> = (0..N_LAYERS).map(|l| scale_com(&noisy_10[l])).collect();
            let ordering_at_10pct = coms_10.windows(2).all(|w| w[1] >= w[0] - 0.5);
            assert!(
                ordering_at_10pct,
                "RG CoM ordering breaks under 10% noise — schedule is too narrow; \
                 consider increasing scale_width"
            );
        }

        // Print decision summary
        println!("\n── Decision Summary ─────────────────────────────────────────────────");
        let ablation_needed = match steps_to_match {
            Some(s) if s < 100  => false,
            Some(s) if s < 500  => true,   // moderate advantage
            Some(_)             => true,   // large advantage
            None                => true,   // never converges
        };
        println!("  Run GPU ablation to isolate RG init contribution: {ablation_needed}");
        if ablation_needed {
            println!("  Suggested: condK-no-RG-init (flat uniform gains) vs condK");
            println!("  Expected: RG init should show faster early convergence (epoch 1-3 PPL)");
        } else {
            println!("  RG init advantage is small enough that ablation can be skipped");
        }
    }
}
