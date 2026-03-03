//! Phase-Locked Loop (PLL) for adaptive Q tracking in DSQG attention.
//!
//! ## Physics origin
//!
//! A phase-locked loop (PLL) synchronises a local oscillator (LO) to an incoming
//! signal by measuring the phase error and feeding it back into the VCO.
//!
//!   Q_new = (1−α)·Q̂ + α·K̂_retrieved   (normalised first-order PLL)
//!
//! where Q̂ = Q/||Q|| and K̂ = K/||K|| are unit vectors.
//! Operating on unit vectors makes the PLL direction-only (magnitude-independent).
//!
//! ## DSQG analogy
//!
//! In the heterodyne receiver framing:
//!   - Q = local oscillator (LO): currently generated fresh from x at each position
//!   - K = incoming signal
//!   - Q·K/√D = phase comparison
//!
//! Current condU is open-loop: Q has no memory of prior retrievals.
//! PLL closes the loop: Q_n uses the direction of K retrieved at n−1.
//!
//! ## What this module verifies
//!
//! 1. Phase error (1 − cos_sim) decreases monotonically with PLL updates.
//! 2. Open-loop Q (α=0): no alignment improvement regardless of repetitions.
//! 3. Lock-time formula: T_lock ≈ ceil(log(ε)/log(1−α)) steps to reach 1−ε alignment.
//! 4. Higher α = fewer steps to reach a given alignment threshold.
//! 5. Induction scenario: after k repetitions of (A→B pattern), PLL alignment
//!    grows monotonically; open-loop stays flat.

const D_HEAD: usize = 32;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
fn norm(v: &[f64]) -> f64 { dot(v,v).sqrt().max(1e-12) }

fn normalise(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    v.iter().map(|x| x/n).collect()
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    dot(a,b) / (norm(a) * norm(b))
}

/// First-order PLL update operating on unit vectors.
/// Q_new = normalise((1−α)·Q̂ + α·K̂)
/// This is magnitude-independent: only the angle between Q and K_retrieved matters.
pub fn pll_step_normalised(q: &[f64], k_retrieved: &[f64], alpha: f64) -> Vec<f64> {
    let q_hat = normalise(q);
    let k_hat = normalise(k_retrieved);
    let mixed: Vec<f64> = q_hat.iter().zip(&k_hat)
        .map(|(&qi, &ki)| (1.0-alpha)*qi + alpha*ki)
        .collect();
    normalise(&mixed)
}

/// Simulate normalised PLL for `steps` steps against a fixed K_target.
fn pll_trace(q_init: &[f64], k_target: &[f64], alpha: f64, steps: usize) -> Vec<f64> {
    let mut q = normalise(q_init);
    let k_hat = normalise(k_target);
    let mut trace = vec![cosine_similarity(&q, &k_hat)];
    for _ in 0..steps {
        q = pll_step_normalised(&q, &k_hat, alpha);
        trace.push(cosine_similarity(&q, &k_hat));
    }
    trace
}

fn basis_vec(d: usize, idx: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; d]; v[idx] = 1.0; v
}

fn pseudo_unit_vec(d: usize, seed: u64) -> Vec<f64> {
    let mut lcg = seed;
    let raw: Vec<f64> = (0..d).map(|_| {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
    }).collect();
    normalise(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Property 1: PLL reduces phase error monotonically ────────────────────

    #[test]
    fn pll_reduces_phase_error_monotonically() {
        let alpha    = 0.15;
        let k_target = basis_vec(D_HEAD, 0);
        let q_init   = basis_vec(D_HEAD, 1); // orthogonal start: cos_sim = 0

        let trace = pll_trace(&q_init, &k_target, alpha, 80);
        for w in trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-12,
                "cos_sim must be non-decreasing: {:.6} -> {:.6}", w[0], w[1]);
        }
        assert!(*trace.last().unwrap() > 0.99,
            "PLL must converge to near-full alignment; got {:.6}", trace.last().unwrap());
    }

    // ── Property 2: Open loop shows no improvement ────────────────────────────

    #[test]
    fn open_loop_shows_no_alignment_improvement() {
        let k_target = basis_vec(D_HEAD, 0);
        let q_init   = pseudo_unit_vec(D_HEAD, 42);
        let trace    = pll_trace(&q_init, &k_target, 0.0, 50);
        let initial  = trace[0];
        for (t, &sim) in trace.iter().enumerate() {
            assert!((sim - initial).abs() < 1e-12,
                "Open-loop must stay constant at step {t}: {sim:.8} != {initial:.8}");
        }
    }

    // ── Property 3: Lock time ─────────────────────────────────────────────────

    /// Starting from orthogonal (cos_sim=0), T_lock = ceil(log(ε)/log(1−α)) steps
    /// to reach cos_sim > 1−ε.
    #[test]
    fn lock_time_matches_formula() {
        let q_init   = basis_vec(D_HEAD, 1);
        let k_target = basis_vec(D_HEAD, 0);
        let eps: f64 = 0.05;

        for &alpha in &[0.05_f64, 0.10, 0.20, 0.30] {
            let t_pred = (eps.ln() / (1.0_f64 - alpha).ln()).ceil() as usize;
            let trace  = pll_trace(&q_init, &k_target, alpha, t_pred + 5);
            assert!(trace[t_pred] > 1.0 - eps,
                "alpha={alpha:.2}: must reach >{:.2} by step {t_pred}; got {:.6}",
                1.0-eps, trace[t_pred]);
        }
    }

    // ── Property 4: Higher alpha = fewer steps to threshold ───────────────────

    /// Lock steps to reach 0.90 alignment must decrease as alpha increases.
    /// We measure this by checking lock steps at each alpha — higher alpha
    /// must lock in fewer steps than lower alpha.
    #[test]
    fn higher_alpha_locks_in_fewer_steps() {
        let q_init    = basis_vec(D_HEAD, 1);
        let k_target  = basis_vec(D_HEAD, 0);
        let threshold = 0.90;
        let alphas    = [0.05_f64, 0.10, 0.20, 0.40];

        let lock_steps: Vec<usize> = alphas.iter().map(|&alpha| {
            let trace = pll_trace(&q_init, &k_target, alpha, 200);
            trace.iter().position(|&s| s >= threshold).unwrap_or(200)
        }).collect();

        // Lock steps must be strictly decreasing as alpha increases
        for (i, w) in lock_steps.windows(2).enumerate() {
            assert!(w[1] < w[0],
                "alpha={:.2} must lock faster than alpha={:.2}: steps {} vs {}",
                alphas[i+1], alphas[i], w[1], w[0]);
        }
    }

    // ── Property 5: Induction scenario ────────────────────────────────────────

    /// Q starts nearly orthogonal to K_target.
    /// After each repetition of (attend to K_target), PLL alignment grows.
    /// Open-loop stays flat.
    #[test]
    fn induction_scenario_alignment_grows_with_repetitions() {
        let alpha    = 0.15;
        let k_target = pseudo_unit_vec(D_HEAD, 123);
        // Force near-orthogonal start: use the component of random vec orthogonal to k_target
        let raw       = pseudo_unit_vec(D_HEAD, 456);
        let proj      = dot(&raw, &k_target);
        let q_init: Vec<f64> = raw.iter().zip(&k_target).map(|(&r,&k)| r - proj*k).collect();
        let q_init    = normalise(&q_init); // now orthogonal to k_target

        let open_sim  = cosine_similarity(&q_init, &k_target);
        assert!(open_sim.abs() < 0.05, "Q must start near-orthogonal; cos_sim={open_sim:.4}");

        let mut q_pll = q_init.clone();
        let mut pll_sims = vec![cosine_similarity(&q_pll, &k_target)];
        for _ in 0..8 {
            q_pll = pll_step_normalised(&q_pll, &k_target, alpha);
            pll_sims.push(cosine_similarity(&q_pll, &k_target));
        }

        // PLL alignment must grow monotonically from near-zero
        for w in pll_sims.windows(2) {
            assert!(w[1] >= w[0] - 1e-12,
                "Induction alignment must grow: {:.6} -> {:.6}", w[0], w[1]);
        }
        // After 8 reps, must be substantially better than initial
        let final_sim = *pll_sims.last().unwrap();
        assert!(final_sim > 0.70,
            "After 8 reps from orthogonal start, alignment must exceed 0.70; got {final_sim:.4}");
    }

    // ── Property 6: Magnitude independence (normalised PLL) ───────────────────

    /// Normalised PLL (operating on unit vectors) must give identical
    /// cosine similarity traces regardless of input magnitudes.
    #[test]
    fn normalised_pll_is_magnitude_independent() {
        let k_target = basis_vec(D_HEAD, 0);
        let q_dir    = basis_vec(D_HEAD, 1);
        let q_small: Vec<f64> = q_dir.iter().map(|x| x*0.1).collect();
        let q_large: Vec<f64> = q_dir.iter().map(|x| x*10.0).collect();

        let trace_s = pll_trace(&q_small, &k_target, 0.20, 30);
        let trace_l = pll_trace(&q_large, &k_target, 0.20, 30);
        for (t, (&s, &l)) in trace_s.iter().zip(&trace_l).enumerate() {
            assert!((s - l).abs() < 1e-10,
                "Normalised PLL must be magnitude-independent at step {t}: {s:.8} vs {l:.8}");
        }
    }

    #[test]
    fn pll_print_summary() {
        let q_init   = basis_vec(D_HEAD, 1);
        let k_target = basis_vec(D_HEAD, 0);
        println!("\nNormalised PLL alignment (Q⊥K_target, D={})", D_HEAD);
        println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "step","open(0)","a=0.05","a=0.10","a=0.20","a=0.40");
        for step in [0,5,10,20,40,60,80] {
            let sims: Vec<f64> = [0.0_f64,0.05,0.10,0.20,0.40].iter()
                .map(|&a| *pll_trace(&q_init,&k_target,a,step).last().unwrap())
                .collect();
            println!("{:>6}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                     step,sims[0],sims[1],sims[2],sims[3],sims[4]);
        }
    }
}
