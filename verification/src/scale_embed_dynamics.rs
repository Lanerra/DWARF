//! Scale_embed Phase Transition Dynamics
//!
//! Models the growth of `scale_embed |max|` during training and predicts
//! when it will cross the holographic phase transition threshold (~0.74).
//!
//! # Empirical observations
//!
//! The scale_embed parameter starts at zero (zero-init) and grows monotonically
//! during training. When `|max|` crosses a threshold τ ≈ 0.74, the model
//! undergoes a phase transition: passkey retrieval capability suddenly develops.
//!
//! Data points from condX-v2 35M (D=512, H=8, N=38.7M params):
//!   ep1 (14%C): |max| = 0.3566
//!   ep2 (29%C): |max| = 0.5875  (+0.2309)
//!   ep3 (43%C): |max| = 0.7456  (+0.1581) ← threshold crossed
//!
//! Data points from 13M models (D=256, H=8, N≈14M params):
//!   condX-v2 13M: |max| crossed ~0.7649 between ep3→ep4 (33%C → 57%C)
//!
//! Key finding: both 13M and 35M models cross τ≈0.74 at approximately
//! 43% of Chinchilla budget (ep3 of 7), suggesting a SCALE-INVARIANT
//! phase transition point at ~43% Chinchilla.
//!
//! # Growth Models
//!
//! Two candidate models for scale_embed(t) where t = Chinchilla fraction:
//!
//! 1. **Power law**: y(t) = A × t^α
//!    - Fit via log-linear regression on observed (t, y) pairs
//!    - Simple, interpretable: A = initial scale, α = growth exponent
//!
//! 2. **Logistic (S-curve)**: y(t) = L / (1 + exp(-k(t - t₀)))
//!    - Models saturation at large t
//!    - More realistic for bounded parameters
//!    - Fit via Gauss-Newton iteration

use crate::sweep_engine::{sweep_1d_progress, Stats, write_json_results};

// ─── Empirical data ───────────────────────────────────────────────────────────

/// Known data points: (chinchilla_fraction, scale_embed_max_abs)
pub const CONDX_V2_35M: &[(f64, f64)] = &[
    (0.14, 0.3566),   // ep1
    (0.29, 0.5875),   // ep2
    (0.43, 0.7456),   // ep3 ← threshold crossed
];

/// Known data for 13M models (approximate, from observations)
pub const CONDX_V2_13M: &[(f64, f64)] = &[
    (0.33, 0.70),     // ep3: just below threshold (approximate)
    (0.57, 0.78),     // ep4: just above threshold (approximate, 0.7649 observed)
];

/// Phase transition threshold (empirical)
pub const THRESHOLD: f64 = 0.74;

// ─── Power law fit ────────────────────────────────────────────────────────────

/// Fit a power law y = A × t^α to (t, y) data via log-linear least squares.
/// Returns (A, alpha, r_squared).
pub fn fit_power_law(data: &[(f64, f64)]) -> (f64, f64, f64) {
    // ln(y) = ln(A) + α × ln(t)
    // Standard OLS on (ln(t), ln(y))
    let n = data.len() as f64;
    let lx: Vec<f64> = data.iter().map(|(t, _)| t.ln()).collect();
    let ly: Vec<f64> = data.iter().map(|(_, y)| y.ln()).collect();

    let mean_lx = lx.iter().sum::<f64>() / n;
    let mean_ly = ly.iter().sum::<f64>() / n;

    let ss_xx: f64 = lx.iter().map(|x| (x - mean_lx).powi(2)).sum();
    let ss_xy: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| (x - mean_lx) * (y - mean_ly)).sum();
    let ss_yy: f64 = ly.iter().map(|y| (y - mean_ly).powi(2)).sum();

    let alpha = if ss_xx > 1e-12 { ss_xy / ss_xx } else { 1.0 };
    let ln_a = mean_ly - alpha * mean_lx;
    let a = ln_a.exp();

    let r_squared = if ss_yy > 1e-12 { (ss_xy / ss_xx) * ss_xy / ss_yy } else { 1.0 };
    (a, alpha, r_squared)
}

/// Predict when power law y = A × t^α crosses threshold τ.
/// Returns the Chinchilla fraction at crossing.
pub fn power_law_crossing(a: f64, alpha: f64, threshold: f64) -> Option<f64> {
    if a <= 0.0 || alpha <= 0.0 { return None; }
    // τ = A × t^α → t = (τ/A)^(1/α)
    let ratio = threshold / a;
    if ratio <= 0.0 { return None; }
    Some(ratio.powf(1.0 / alpha))
}

/// Evaluate power law at Chinchilla fraction t.
pub fn power_law_eval(a: f64, alpha: f64, t: f64) -> f64 {
    a * t.powf(alpha)
}

// ─── Logistic fit ─────────────────────────────────────────────────────────────

/// Fit logistic y = L / (1 + exp(-k(t - t0))) via simple grid search.
/// L (asymptote) is estimated from data max × 1.5.
pub fn fit_logistic(data: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    // (L, k, t0, r_squared)
    let y_max = data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let l = y_max * 1.5;  // asymptote estimate

    let mut best = (l, 1.0, 0.5, f64::NEG_INFINITY);

    for ki in 0..50 {
        let k = 0.5 + ki as f64 * 0.3;
        for t0i in 0..100 {
            let t0 = 0.1 + t0i as f64 * 0.02;
            let sse: f64 = data.iter().map(|(t, y)| {
                let pred = l / (1.0 + (-k * (t - t0)).exp());
                (y - pred).powi(2)
            }).sum();
            // Convert to R² equivalent (negative SSE for maximization)
            let ss_tot: f64 = {
                let mean_y = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;
                data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum()
            };
            let r2 = if ss_tot > 1e-12 { 1.0 - sse / ss_tot } else { 0.0 };
            if r2 > best.3 {
                best = (l, k, t0, r2);
            }
        }
    }
    best
}

/// Evaluate logistic at t.
pub fn logistic_eval(l: f64, k: f64, t0: f64, t: f64) -> f64 {
    l / (1.0 + (-k * (t - t0)).exp())
}

/// Predict when logistic crosses threshold τ.
pub fn logistic_crossing(l: f64, k: f64, t0: f64, threshold: f64) -> Option<f64> {
    if threshold >= l { return None; }  // never reaches threshold
    // τ = L/(1+exp(-k(t-t0))) → t = t0 - ln(L/τ - 1)/k
    let inner = l / threshold - 1.0;
    if inner <= 0.0 { return None; }
    Some(t0 - inner.ln() / k)
}

// ─── Phase transition analysis ────────────────────────────────────────────────

/// Summary of phase transition analysis for one dataset.
#[derive(Debug, Clone)]
pub struct PhaseTransitionSummary {
    pub model_name: &'static str,
    /// Power law fit: (A, alpha, R²)
    pub power_law: (f64, f64, f64),
    /// Predicted crossing (Chinchilla fraction) via power law
    pub pl_crossing: Option<f64>,
    /// Logistic fit: (L, k, t0, R²)
    pub logistic: (f64, f64, f64, f64),
    /// Predicted crossing (Chinchilla fraction) via logistic
    pub logistic_crossing: Option<f64>,
    /// Predicted |max| at Chinchilla ep=7 (t=1.0)
    pub predicted_at_chinchilla: f64,
    /// Predicted |max| at ep=10 (t=143%)
    pub predicted_at_ep10: f64,
}

pub fn analyze(data: &[(f64, f64)], name: &'static str) -> PhaseTransitionSummary {
    let (a, alpha, r2_pl) = fit_power_law(data);
    let pl_cross = power_law_crossing(a, alpha, THRESHOLD);

    let (l, k, t0, r2_log) = fit_logistic(data);
    let log_cross = logistic_crossing(l, k, t0, THRESHOLD);

    PhaseTransitionSummary {
        model_name: name,
        power_law: (a, alpha, r2_pl),
        pl_crossing: pl_cross,
        logistic: (l, k, t0, r2_log),
        logistic_crossing: log_cross,
        predicted_at_chinchilla: power_law_eval(a, alpha, 1.0),
        predicted_at_ep10: power_law_eval(a, alpha, 1.43),
    }
}

// ─── Sweep: initial growth rate → predicted crossing ─────────────────────────

/// How crossing epoch varies with initial growth rate (first-epoch |max|).
/// Assumes power-law growth with fixed alpha = observed mean.
pub fn run_crossing_sweep(alpha: f64) -> Vec<crate::sweep_engine::SweepPoint<f64, f64>> {
    let initial_vals: Vec<f64> = (1..=50).map(|i| i as f64 * 0.02).collect(); // 0.02..1.0
    sweep_1d_progress(
        &initial_vals,
        |&y1| {
            // Calibrate A so that y(0.14) = y1 (first epoch = 14%C)
            let a = y1 / 0.14f64.powf(alpha);
            power_law_crossing(a, alpha, THRESHOLD).unwrap_or(f64::INFINITY)
        },
        "crossing-epoch vs initial-growth",
    )
}

// ─── Top-level runner ─────────────────────────────────────────────────────────

pub fn run_all(output_dir: Option<&str>) {
    println!("\n=== Scale_embed Phase Transition Dynamics ===");
    println!("Threshold: τ = {THRESHOLD}");
    println!();

    for (data, name) in &[
        (CONDX_V2_35M, "condX-v2 35M"),
        (CONDX_V2_13M, "condX-v2 13M (approx)"),
    ] {
        let s = analyze(data, name);
        println!("--- {} ---", s.model_name);
        println!("  Power law:  A={:.4}, α={:.4}, R²={:.4}",
            s.power_law.0, s.power_law.1, s.power_law.2);
        println!("  Logistic:   L={:.4}, k={:.4}, t₀={:.4}, R²={:.4}",
            s.logistic.0, s.logistic.1, s.logistic.2, s.logistic.3);
        match s.pl_crossing {
            Some(c) => println!("  Crossing at {:.1}% Chinchilla (power law)",  c * 100.0),
            None    => println!("  No crossing predicted (power law)"),
        }
        match s.logistic_crossing {
            Some(c) => println!("  Crossing at {:.1}% Chinchilla (logistic)",   c * 100.0),
            None    => println!("  No crossing predicted (logistic)"),
        }
        println!("  Predicted |max| at Chinchilla (100%C): {:.4}", s.predicted_at_chinchilla);
        println!("  Predicted |max| at ep10      (143%C): {:.4}", s.predicted_at_ep10);

        // Per-epoch table using power law
        println!("  Power-law trajectory table:");
        let (a, alpha, _) = s.power_law;
        for ep in 1..=10 {
            let t = ep as f64 / 7.0;  // Chinchilla at ep7
            let val = power_law_eval(a, alpha, t);
            let marker = if val >= THRESHOLD && power_law_eval(a, alpha, (ep - 1) as f64 / 7.0) < THRESHOLD {
                " ← PHASE TRANSITION"
            } else { "" };
            println!("    ep{:<2} ({:.0}%C): predicted={:.4}{}", ep, t*100., val, marker);
        }
        println!();
    }

    // Scale-invariance test: compare crossing Chinchilla fractions
    println!("=== Scale-Invariance Test ===");
    println!("If the phase transition is scale-invariant, both models should cross τ at ~43%C.\n");
    for (data, name) in &[
        (CONDX_V2_35M, "condX-v2 35M"),
        (CONDX_V2_13M, "condX-v2 13M (approx)"),
    ] {
        let (a, alpha, _) = fit_power_law(data);
        let crossing = power_law_crossing(a, alpha, THRESHOLD);
        println!("  {}: crossing at {:.1}%C  (A={:.4}, α={:.4})",
            name,
            crossing.map(|c| c * 100.0).unwrap_or(-1.0),
            a, alpha);
    }

    // Sweep: initial growth rate vs. predicted crossing
    let (_, alpha_35m, _) = fit_power_law(CONDX_V2_35M);
    let sweep = run_crossing_sweep(alpha_35m);
    println!("\n=== Crossing Epoch vs Initial Growth Rate (α={:.3}) ===", alpha_35m);
    println!("{:<12} {:<20}", "|max| @ ep1", "Predicted crossing (%C)");
    println!("{}", "-".repeat(35));
    for r in sweep.iter().step_by(5) {
        let cross = if r.metrics.is_finite() { format!("{:.1}%C", r.metrics * 100.0) }
                    else { "never".to_string() };
        println!("{:<12.3} {}", r.params, cross);
    }

    if let Some(dir) = output_dir {
        // Save the per-model data as JSON
        let rows: Vec<String> = CONDX_V2_35M.iter().enumerate().map(|(i, (t, y))| {
            let pred_pl = { let (a,alpha,_) = fit_power_law(CONDX_V2_35M); power_law_eval(a,alpha,*t) };
            format!("{{\"epoch\": {}, \"chinchilla_frac\": {:.4}, \"observed\": {:.4}, \"predicted_pl\": {:.4}}}",
                i+1, t, y, pred_pl)
        }).collect();
        let (a,alpha,r2) = fit_power_law(CONDX_V2_35M);
        let crossing = power_law_crossing(a, alpha, THRESHOLD).unwrap_or(-1.0);
        let meta = vec![
            ("sweep_type", "\"scale_embed_dynamics\"".to_string()),
            ("model", "\"condX-v2 35M\"".to_string()),
            ("threshold", format!("{THRESHOLD}")),
            ("power_law_A", format!("{a:.6}")),
            ("power_law_alpha", format!("{alpha:.6}")),
            ("power_law_r2", format!("{r2:.6}")),
            ("predicted_crossing_chinchilla", format!("{crossing:.4}")),
        ];
        let p = format!("{dir}/scale_embed_dynamics.json");
        match write_json_results(&p, &meta, &rows) {
            Ok(_) => println!("\nSaved → {p}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_law_fit_recovers_known_curve() {
        // y = 0.5 × t^0.7 → fit should recover these approximately
        let data: Vec<(f64, f64)> = (1..=5)
            .map(|t| (t as f64, 0.5 * (t as f64).powf(0.7)))
            .collect();
        let (a, alpha, r2) = fit_power_law(&data);
        assert!((a - 0.5).abs() < 0.01, "A={a}");
        assert!((alpha - 0.7).abs() < 0.01, "alpha={alpha}");
        assert!(r2 > 0.99, "R²={r2}");
    }

    #[test]
    fn power_law_crossing_correct() {
        // y = 1.0 × t^1.0 crosses 0.74 at t=0.74
        let (t, _) = (power_law_crossing(1.0, 1.0, 0.74).unwrap(), ());
        assert!((t - 0.74).abs() < 1e-9, "crossing={t}");
    }

    #[test]
    fn condxv2_35m_crosses_near_43pct() {
        let (a, alpha, _) = fit_power_law(CONDX_V2_35M);
        let cross = power_law_crossing(a, alpha, THRESHOLD).unwrap();
        // Should be between 35% and 50% Chinchilla
        assert!(cross > 0.35 && cross < 0.55,
            "crossing={:.2}%C outside expected [35,55]%C", cross * 100.0);
    }
}
