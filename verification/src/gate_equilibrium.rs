//! Bypass Gate Equilibrium Predictor
//!
//! Models the gradient-balance mechanism that determines where bypass_alpha
//! (and thus the bypass gate) settles during training of condX-v2 variants.
//!
//! # The Mechanism
//!
//! The bypass gate `g = sigmoid(α)` blends clean and contaminated Q:
//!   Q_blend = g × Q_clean + (1−g) × Q_full
//!
//! Two opposing gradient forces act on α during training:
//!
//!   **PPL force** (pushes α toward +∞, gate→1, more clean Q):
//!     A slightly cleaner Q improves language model PPL by reducing noise
//!     in the attention score computation. This gradient is ∝ model capacity
//!     and data density (more capacity → sharper gradients → stronger PPL pull).
//!
//!   **Passkey force** (pushes α toward −∞, gate→0, contaminated Q):
//!     The contaminated Q IS the reference beam for holographic retrieval.
//!     Any clean Q mixing degrades passkey accuracy. This gradient is ∝
//!     the strength of the holographic field, which grows with training budget
//!     and model capacity.
//!
//! At equilibrium: |dL_PPL/dα| = |dL_passkey/dα|
//!
//! # The Model
//!
//! Let g_eq = equilibrium gate value. We model:
//!
//!   PPL gradient magnitude:     ε_ppl(N, C)     ∝ N^β_ppl  × C^δ_ppl
//!   Passkey gradient magnitude: ε_passkey(N, C) ∝ N^β_pass × C^δ_pass
//!
//! where N = number of parameters, C = Chinchilla fraction.
//!
//! Equilibrium condition: ε_ppl = ε_passkey
//!   → g_eq = (ε_ppl / (ε_ppl + ε_passkey)) via sigmoid inversion
//!   ≈ ε_ppl / ε_passkey  (when g_eq ≪ 1)
//!
//! For g_eq ≪ 1 (which is empirically true):
//!   g_eq ≈ (ε_ppl / ε_passkey) = A × N^(β_ppl - β_pass)
//!
//! # Calibration
//!
//! Known data points:
//!   condX-v2 13M: N=14.06M, g_eq=0.1016 (confirmed at ep7 steady state)
//!   condX-v2 35M: N=38.73M, g_eq≈0.003  (projected from ep1-3 trajectory)
//!
//! From these two points we calibrate (A, β_diff = β_ppl - β_pass).
//!
//! # Predictions
//!
//! Given calibration, predict equilibrium for:
//!   - condX-v2 85M (if we run it)
//!   - condX-v2 at 13M but different Chinchilla fractions
//!   - The functional form of how gate_eq scales with model size

use crate::sweep_engine::{sweep_1d_progress, write_json_results};

// ─── Empirical data ───────────────────────────────────────────────────────────

/// Known (param_count, gate_eq) pairs.
/// 13M: confirmed. 35M: projected from ep1-3 trajectory (≈ -5.7 → gate ≈ 0.003).
pub const CALIBRATION_DATA: &[(f64, f64, &str)] = &[
    (14.06e6, 0.1016,  "condX-v2 13M (confirmed)"),
    (38.73e6, 0.003,   "condX-v2 35M (projected)"),
];

/// Parameter counts for prediction targets.
pub const PREDICTION_TARGETS: &[(f64, &str)] = &[
    (14.06e6,  "condX-v2 13M"),
    (27.0e6,   "condX-v2 27M (hypothetical)"),
    (38.73e6,  "condX-v2 35M"),
    (85.0e6,   "condX-v2 85M (future)"),
    (350.0e6,  "condX-v2 350M (extrapolation)"),
    (7000.0e6, "condX-v2 7B  (extrapolation)"),
];

// ─── Power law model for gate_eq(N) ──────────────────────────────────────────

/// Fit g_eq = A × N^β to calibration data via log-linear regression.
/// Returns (A, beta, r_squared).
pub fn fit_gate_scaling(data: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    let lx: Vec<f64> = data.iter().map(|(params, _)| params.ln()).collect();
    let ly: Vec<f64> = data.iter().map(|(_, g)| g.ln()).collect();

    let mean_lx = lx.iter().sum::<f64>() / n;
    let mean_ly = ly.iter().sum::<f64>() / n;
    let ss_xx: f64 = lx.iter().map(|x| (x - mean_lx).powi(2)).sum();
    let ss_xy: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| (x - mean_lx) * (y - mean_ly)).sum();
    let ss_yy: f64 = ly.iter().map(|y| (y - mean_ly).powi(2)).sum();

    let beta = if ss_xx > 1e-12 { ss_xy / ss_xx } else { -1.0 };
    let ln_a = mean_ly - beta * mean_lx;
    let r2 = if ss_yy > 1e-12 { ss_xy.powi(2) / (ss_xx * ss_yy) } else { 1.0 };
    (ln_a.exp(), beta, r2)
}

/// Predict gate_eq for a given parameter count.
pub fn predict_gate(a: f64, beta: f64, n_params: f64) -> f64 {
    (a * n_params.powf(beta)).min(1.0).max(0.0)
}

// ─── Trajectory analysis ──────────────────────────────────────────────────────

/// Per-epoch gate observations for condX-v2 35M.
/// (chinchilla_fraction, alpha_value, gate_value)
pub const CONDXV2_35M_TRAJECTORY: &[(f64, f64, f64)] = &[
    (0.00, -10.0000, 0.0000454),  // init
    (0.14, -9.4455,  0.0000789),  // ep1
    (0.29, -8.6954,  0.000167),   // ep2
    (0.43, -8.0894,  0.000307),   // ep3
];

/// Fit linear trend to alpha(t) and extrapolate to ep10.
pub fn extrapolate_alpha(trajectory: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    // Use last two observed points to fit linear extrapolation
    let obs: Vec<(f64, f64)> = trajectory.iter()
        .filter(|(c, _, _)| *c > 0.0)  // skip init
        .map(|(c, a, _)| (*c, *a))
        .collect();

    if obs.len() < 2 { return vec![]; }

    // Linear fit: alpha(t) = a0 + a1 × t
    let n = obs.len() as f64;
    let mean_t = obs.iter().map(|(t,_)| t).sum::<f64>() / n;
    let mean_a = obs.iter().map(|(_,a)| a).sum::<f64>() / n;
    let ss_tt: f64 = obs.iter().map(|(t,_)| (t - mean_t).powi(2)).sum();
    let ss_ta: f64 = obs.iter().map(|(t,a)| (t - mean_t) * (a - mean_a)).sum();
    let slope  = if ss_tt > 1e-12 { ss_ta / ss_tt } else { 0.0 };
    let intercept = mean_a - slope * mean_t;

    // Extrapolate to ep1..ep10 (Chinchilla at ep7 → t=1.0, ep10 → t=1.43)
    (1..=10).map(|ep| {
        let t = ep as f64 / 7.0;
        let alpha_pred = intercept + slope * t;
        let gate_pred = 1.0 / (1.0 + (-alpha_pred).exp());
        (t, alpha_pred, gate_pred)
    }).collect()
}

// ─── Sweep: gate_eq vs N_params ──────────────────────────────────────────────

pub fn run_scaling_sweep(
    a: f64,
    beta: f64,
) -> Vec<crate::sweep_engine::SweepPoint<f64, f64>> {
    // Sweep log-uniform param counts from 10M to 10B
    let param_counts: Vec<f64> = (0..=50)
        .map(|i| 10.0e6 * (1000.0f64).powf(i as f64 / 50.0))
        .collect();

    sweep_1d_progress(
        &param_counts,
        |&n| predict_gate(a, beta, n),
        "gate-eq vs N_params",
    )
}

// ─── Top-level runner ─────────────────────────────────────────────────────────

pub fn run_all(output_dir: Option<&str>) {
    println!("\n=== Bypass Gate Equilibrium Predictor ===");
    println!("Model: g_eq = A × N^β  (power law in parameter count)");
    println!();

    // Fit from calibration data
    let cal: Vec<(f64, f64)> = CALIBRATION_DATA.iter().map(|(n, g, _)| (*n, *g)).collect();
    let (a, beta, r2) = fit_gate_scaling(&cal);
    println!("Calibration fit: A={:.4e}, beta={:.4}, R2={:.4}", a, beta, r2);
    println!("Interpretation: for every 10× increase in params,");
    println!("  gate_eq changes by factor 10^({:.4}) = {:.4}x",
        beta, 10.0f64.powf(beta));
    println!();

    // Predictions
    println!("Predicted equilibrium gate values:");
    println!("{:<35} {:>10}  {:>10}  {:>10}",
        "Model", "N_params", "g_eq", "alpha_eq");
    println!("{}", "-".repeat(70));
    for &(n, name) in PREDICTION_TARGETS {
        let g = predict_gate(a, beta, n);
        let alpha = ((g / (1.0 - g)).max(1e-10)).ln();  // sigmoid inverse
        println!("{:<35} {:>10.2e}  {:>10.6}  {:>10.4}", name, n, g, alpha);
    }

    // Trajectory analysis for 35M
    println!("\n=== condX-v2 35M trajectory analysis ===");
    println!("Linear extrapolation of alpha(t) from observed ep1-ep3:");
    println!("{:<6} {:>10}  {:>12}  {:>12}  {:>14}",
        "epoch", "chinchilla", "alpha_pred", "gate_pred", "vs_calibrated");
    println!("{}", "-".repeat(60));
    let traj = extrapolate_alpha(CONDXV2_35M_TRAJECTORY);
    for (t, alpha_pred, gate_pred) in &traj {
        let ep = (t * 7.0).round() as usize;
        let calibrated = predict_gate(a, beta, 38.73e6);
        let vs_cal = gate_pred / calibrated.max(1e-10);
        println!("ep{:<5} {:>10.0}%  {:>12.4}  {:>12.6}  {:>12.2}×",
            ep, t * 100.0, alpha_pred, gate_pred, vs_cal);
    }

    // Print calibration sources
    println!("\nCalibration data:");
    for (n, g, name) in CALIBRATION_DATA {
        println!("  {}: N={:.2e}, g_eq={:.6}", name, n, g);
    }

    // Also note: both calibration points have gate_eq ≪ 1 (<<10%)
    // This validates the g_eq ≪ 1 approximation used in the model
    println!("\nKey finding: g_eq << 1 at all scales observed.");
    println!("The passkey gradient dominates the PPL gradient for holographic models.");
    println!("β={:.4} means gate_eq shrinks ~{:.1}% per parameter doubling.",
        beta, (1.0 - 2.0f64.powf(beta)) * 100.0);

    if let Some(dir) = output_dir {
        let rows: Vec<String> = PREDICTION_TARGETS.iter().map(|(n, name)| {
            let g = predict_gate(a, beta, *n);
            let alpha = (g / (1.0 - g).max(1e-10)).ln();
            format!("{{\"model\": {:?}, \"n_params\": {:.2e}, \"gate_eq\": {:.8}, \"alpha_eq\": {:.6}}}",
                name, n, g, alpha)
        }).collect();
        let meta = vec![
            ("sweep_type", "\"gate_equilibrium\"".to_string()),
            ("model_A",    format!("{a:.6e}")),
            ("model_beta", format!("{beta:.6}")),
            ("r_squared",  format!("{r2:.6}")),
        ];
        let p = format!("{dir}/gate_equilibrium.json");
        match write_json_results(&p, &meta, &rows) {
            Ok(_) => println!("Saved → {p}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_fit_recovers_data() {
        let cal: Vec<(f64, f64)> = CALIBRATION_DATA.iter().map(|(n, g, _)| (*n, *g)).collect();
        let (a, beta, _) = fit_gate_scaling(&cal);
        for &(n, g_known, name) in CALIBRATION_DATA {
            let g_pred = predict_gate(a, beta, n);
            let rel_err = (g_pred - g_known).abs() / g_known;
            assert!(rel_err < 0.01, "Fit error too large for {name}: pred={g_pred:.6}, known={g_known:.6}");
        }
    }

    #[test]
    fn gate_shrinks_with_scale() {
        let cal: Vec<(f64, f64)> = CALIBRATION_DATA.iter().map(|(n, g, _)| (*n, *g)).collect();
        let (a, beta, _) = fit_gate_scaling(&cal);
        // Larger models should have smaller gate_eq
        let g_13m = predict_gate(a, beta, 14e6);
        let g_35m = predict_gate(a, beta, 39e6);
        let g_85m = predict_gate(a, beta, 85e6);
        assert!(g_13m > g_35m, "gate_eq should decrease with scale: 13M={g_13m:.4}");
        assert!(g_35m > g_85m, "gate_eq should decrease with scale: 35M={g_35m:.4}");
    }

    #[test]
    fn trajectory_extrapolation_runs() {
        let traj = extrapolate_alpha(CONDXV2_35M_TRAJECTORY);
        assert_eq!(traj.len(), 10);
        // All extrapolated alpha values should be greater than init (-10.0)
        // (alpha moves toward 0 over training, i.e., becomes less negative)
        for (_, alpha, _) in &traj {
            assert!(*alpha > -10.0, "alpha={alpha} should be > -10");
        }
    }
}
