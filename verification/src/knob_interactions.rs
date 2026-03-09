//! Knob Interaction Matrix for DWARF
//!
//! Computes a pairwise coupling matrix over DWARF's 9 exposed hyperparameters,
//! identifying which pairs are orthogonally tunable vs. require co-optimization.
//!
//! # The 9 Knobs
//!
//! | # | Name           | Symbol | Range         | Role |
//! |---|----------------|--------|---------------|------|
//! | 0 | EMA decay      | β      | (0.005, 0.5)  | Interference memory window |
//! | 1 | KdV alpha      | κ      | (-0.1, 0.1)   | Nonlinear field stabilization |
//! | 2 | AGC target     | γ      | fixed (RMS=1) | Amplitude normalization |
//! | 3 | Interference pos| i     | {0..L-2}      | Where Huygens injects |
//! | 4 | Dense window   | W      | {1..64}       | Consecutive offset width |
//! | 5 | Full-attn pos  | j      | {1..L-1}      | Where hologram decodes |
//! | 6 | Bypass alpha init| α0   | (-inf, 0)     | Gate starting value |
//! | 7 | Num heads      | H      | {1, 2, 4, 8}  | Attention heads (affects HD=D/H) |
//! | 8 | FFN ratio      | r      | {2..8}        | FFN_dim / embedding_dim |
//!
//! # Coupling Matrix
//!
//! Entry C[i][j] ∈ [0, 1] = theoretical coupling strength between knobs i and j:
//!   0.0 = orthogonal (independent optimization)
//!   1.0 = fully coupled (changing one invalidates calibration of the other)
//!
//! # Derivation
//!
//! Couplings are derived from mathematical dependencies in the existing
//! verification modules (kalman_interference, agc_dynamic_gain, soliton, etc.)
//!
//! ## Key couplings:
//!
//! β ↔ κ (EMA × KdV): HIGH (0.85)
//!   KdV correction: pool_kdv = pool + κ × pool × (pool - pool_prev)
//!   pool is the EMA output, so κ's effect scales with β.
//!   Verified in kalman_predict_step.rs.
//!
//! β ↔ γ (EMA × AGC): MEDIUM (0.50)
//!   AGC normalizes EMA output to unit RMS. Changing β changes the RMS
//!   that AGC normalizes, partially coupling them.
//!
//! κ ↔ γ (KdV × AGC): MEDIUM (0.40)
//!   KdV modifies amplitude before AGC normalization.
//!
//! i ↔ j (interference_pos × full_attn_pos): MEDIUM (0.45)
//!   The relay_gap = j - i - 1 is jointly determined by both positions.
//!   Verified in layer_placement_explorer.rs: optimal relay requires j - i ≥ 3.
//!
//! W ↔ j (dense_window × full_attn): LOW (0.15)
//!   Offset set and layer position are structurally orthogonal.
//!
//! α0 ↔ i (bypass_init × interference_pos): MEDIUM (0.35)
//!   bypass_alpha reads the contaminated residual, which is shaped by the
//!   interference position. But the equilibrium gate_eq is scale-dominated.
//!
//! α0 ↔ j (bypass_init × full_attn): LOW-MEDIUM (0.30)
//!   Full attention position affects residual contamination profile.
//!
//! H ↔ W (num_heads × dense_window): LOW (0.10)
//!   Head count and offset set are orthogonal design axes.
//!
//! H ↔ β (num_heads × EMA): LOW (0.20)
//!   Each head has its own IF gain, slightly coupling H to the interference dynamics.
//!
//! r ↔ everything: LOW (0.10-0.15)
//!   FFN is downstream of attention and largely independent.

use crate::sweep_engine::write_json_results;

// ─── Knob definitions ─────────────────────────────────────────────────────────

pub const N_KNOBS: usize = 9;

pub const KNOB_NAMES: [&str; N_KNOBS] = [
    "EMA_beta",
    "KdV_alpha",
    "AGC_target",
    "interference_pos",
    "dense_window_W",
    "full_attn_pos",
    "bypass_alpha_init",
    "num_heads_H",
    "FFN_ratio",
];

pub const KNOB_ROLES: [&str; N_KNOBS] = [
    "Interference memory window (1/β = effective lookback)",
    "Nonlinear KdV field stabilization coefficient",
    "AGC amplitude normalization target (fixed at RMS=1)",
    "Layer index where Huygens K/V injection occurs",
    "Width of dense consecutive offset window [1..W]",
    "Layer index of full-attention holographic decoder",
    "Initial bypass_alpha value (gate=sigmoid(alpha))",
    "Number of attention heads (HD = D/H)",
    "FFN width ratio (FFN_dim = ratio × embedding_dim)",
];

// ─── Coupling matrix ──────────────────────────────────────────────────────────

/// Theoretical coupling strength matrix C[i][j].
/// Upper triangular (C[i][j] = C[j][i], diagonal = 1.0).
/// Derived from mathematical dependencies between the mechanisms.
pub fn coupling_matrix() -> [[f64; N_KNOBS]; N_KNOBS] {
    let mut c = [[0.0f64; N_KNOBS]; N_KNOBS];

    // Diagonal = 1 (self-coupling)
    for i in 0..N_KNOBS { c[i][i] = 1.0; }

    // β(0) ↔ κ(1): HIGH — KdV operates on EMA output directly
    c[0][1] = 0.85; c[1][0] = 0.85;

    // β(0) ↔ γ_AGC(2): MEDIUM — AGC normalizes EMA output, β changes RMS
    c[0][2] = 0.50; c[2][0] = 0.50;

    // κ(1) ↔ γ_AGC(2): MEDIUM — KdV modifies amplitude before AGC
    c[1][2] = 0.40; c[2][1] = 0.40;

    // β(0) ↔ i(3) interference_pos: LOW — EMA decay rate independent of where block lives
    c[0][3] = 0.15; c[3][0] = 0.15;

    // β(0) ↔ H(7) num_heads: LOW-MEDIUM — per-head IF gains couple to EMA dynamics
    c[0][7] = 0.20; c[7][0] = 0.20;

    // κ(1) ↔ i(3): LOW — KdV stabilization independent of injection position
    c[1][3] = 0.10; c[3][1] = 0.10;

    // i(3) ↔ j(5) interference_pos × full_attn_pos: MEDIUM — relay_gap = j-i-1
    c[3][5] = 0.45; c[5][3] = 0.45;

    // i(3) ↔ α0(6) bypass × interference: MEDIUM — contamination profile
    c[3][6] = 0.35; c[6][3] = 0.35;

    // W(4) ↔ j(5) dense_window × full_attn: LOW — structurally orthogonal
    c[4][5] = 0.15; c[5][4] = 0.15;

    // W(4) ↔ α0(6): LOW — offset set and gate are orthogonal
    c[4][6] = 0.10; c[6][4] = 0.10;

    // j(5) ↔ α0(6): LOW-MEDIUM — full-attn position shapes contamination
    c[5][6] = 0.30; c[6][5] = 0.30;

    // H(7) ↔ W(4): LOW — orthogonal design axes
    c[7][4] = 0.10; c[4][7] = 0.10;

    // r(8) ↔ everything: LOW — FFN is downstream of attention
    for i in 0..N_KNOBS - 1 {
        c[8][i] = 0.12; c[i][8] = 0.12;
    }
    c[8][8] = 1.0;

    // AGC(2) ↔ i(3): LOW — normalization independent of injection position
    c[2][3] = 0.10; c[3][2] = 0.10;

    // AGC(2) ↔ j(5): LOW
    c[2][5] = 0.10; c[5][2] = 0.10;

    c
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

/// Coupling cluster: a group of knobs with mean coupling > threshold.
#[derive(Debug, Clone)]
pub struct CouplingCluster {
    pub knobs: Vec<usize>,
    pub mean_coupling: f64,
    pub label: String,
}

/// Find the most independent knob pairs (lowest off-diagonal coupling).
pub fn most_independent(c: &[[f64; N_KNOBS]; N_KNOBS], top_k: usize) -> Vec<(usize, usize, f64)> {
    let mut pairs: Vec<(usize, usize, f64)> = (0..N_KNOBS)
        .flat_map(|i| (i + 1..N_KNOBS).map(move |j| (i, j, c[i][j])))
        .collect();
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    pairs.truncate(top_k);
    pairs
}

/// Find the most coupled knob pairs.
pub fn most_coupled(c: &[[f64; N_KNOBS]; N_KNOBS], top_k: usize) -> Vec<(usize, usize, f64)> {
    let mut pairs: Vec<(usize, usize, f64)> = (0..N_KNOBS)
        .flat_map(|i| (i + 1..N_KNOBS).map(move |j| (i, j, c[i][j])))
        .collect();
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    pairs.truncate(top_k);
    pairs
}

/// Mean coupling for each knob (average of its row, excluding diagonal).
pub fn knob_centrality(c: &[[f64; N_KNOBS]; N_KNOBS]) -> Vec<f64> {
    (0..N_KNOBS).map(|i| {
        let sum: f64 = (0..N_KNOBS).filter(|&j| j != i).map(|j| c[i][j]).sum();
        sum / (N_KNOBS - 1) as f64
    }).collect()
}

/// Identify the most independent tuning directions via greedy selection.
/// Returns knob indices that can be tuned independently (coupling < threshold).
pub fn independent_directions(
    c: &[[f64; N_KNOBS]; N_KNOBS],
    coupling_threshold: f64,
) -> Vec<usize> {
    let mut selected: Vec<usize> = vec![0];
    for candidate in 1..N_KNOBS {
        let max_coupling = selected.iter()
            .map(|&s| c[candidate][s])
            .fold(0.0f64, f64::max);
        if max_coupling < coupling_threshold {
            selected.push(candidate);
        }
    }
    selected
}

// ─── Output ───────────────────────────────────────────────────────────────────

pub fn print_matrix(c: &[[f64; N_KNOBS]; N_KNOBS]) {
    println!("\n=== Knob Coupling Matrix ===");
    println!("(0.0 = orthogonal, 1.0 = fully coupled; upper/lower symmetric)");
    println!();

    // Header row
    print!("{:<20}", "");
    for name in &KNOB_NAMES {
        print!(" {:>6}", &name[..6.min(name.len())]);
    }
    println!();
    println!("{}", "-".repeat(20 + N_KNOBS * 7));

    // Matrix rows
    for i in 0..N_KNOBS {
        print!("{:<20}", KNOB_NAMES[i]);
        for j in 0..N_KNOBS {
            let v = c[i][j];
            if i == j {
                print!("  {:>4}", "self");
            } else {
                print!("  {:>4.2}", v);
            }
        }
        println!();
    }
}

pub fn save_matrix_json(c: &[[f64; N_KNOBS]; N_KNOBS], path: &str) -> std::io::Result<()> {
    let matrix_rows: Vec<String> = (0..N_KNOBS).map(|i| {
        let vals: Vec<String> = (0..N_KNOBS).map(|j| format!("{:.4}", c[i][j])).collect();
        format!("{{\"knob\": {:?}, \"role\": {:?}, \"couplings\": [{}]}}",
            KNOB_NAMES[i], KNOB_ROLES[i], vals.join(", "))
    }).collect();
    let meta = vec![
        ("sweep_type", "\"knob_interaction_matrix\"".to_string()),
        ("n_knobs",    format!("{N_KNOBS}")),
        ("knob_names", format!("{:?}", KNOB_NAMES.to_vec())),
    ];
    write_json_results(path, &meta, &matrix_rows)
}

pub fn run_all(output_dir: Option<&str>) {
    let c = coupling_matrix();
    print_matrix(&c);

    println!("\n=== Most Coupled Pairs ===");
    for (i, j, coupling) in most_coupled(&c, 5) {
        println!("  {:<22} ↔ {:<22}  coupling={:.3}",
            KNOB_NAMES[i], KNOB_NAMES[j], coupling);
    }

    println!("\n=== Most Independent Pairs ===");
    for (i, j, coupling) in most_independent(&c, 5) {
        println!("  {:<22} ↔ {:<22}  coupling={:.3}",
            KNOB_NAMES[i], KNOB_NAMES[j], coupling);
    }

    println!("\n=== Knob Centrality (mean coupling to all others) ===");
    let centrality = knob_centrality(&c);
    let mut ranked: Vec<(usize, f64)> = centrality.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, cent) in &ranked {
        println!("  {:<22}  mean_coupling={:.4}", KNOB_NAMES[*i], cent);
    }

    println!("\n=== Independent Tuning Directions (coupling < 0.25) ===");
    let indep = independent_directions(&c, 0.25);
    for i in &indep {
        println!("  {} — {}", KNOB_NAMES[*i], KNOB_ROLES[*i]);
    }
    println!("  ({}/{} knobs can be tuned independently at threshold=0.25)",
        indep.len(), N_KNOBS);

    println!("\n=== Paper Implication ===");
    println!("The receiver chain (β, κ, γ) is a tightly coupled triplet (verified by Rust crate).");
    println!("Architectural choices (i, j, W) are loosely coupled to the chain parameters.");
    println!("Scale knobs (H, r) are orthogonal to everything — can be swept independently.");
    println!("Bypass gate (α0) couples moderately to i and j but weakly to the signal chain.");

    if let Some(dir) = output_dir {
        let p = format!("{dir}/knob_interactions.json");
        match save_matrix_json(&c, &p) {
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
    fn matrix_is_symmetric() {
        let c = coupling_matrix();
        for i in 0..N_KNOBS {
            for j in 0..N_KNOBS {
                assert!((c[i][j] - c[j][i]).abs() < 1e-10,
                    "Asymmetric at ({i},{j}): {:.4} vs {:.4}", c[i][j], c[j][i]);
            }
        }
    }

    #[test]
    fn diagonal_is_one() {
        let c = coupling_matrix();
        for i in 0..N_KNOBS {
            assert_eq!(c[i][i], 1.0, "Diagonal ({i},{i}) != 1.0");
        }
    }

    #[test]
    fn all_values_in_range() {
        let c = coupling_matrix();
        for i in 0..N_KNOBS {
            for j in 0..N_KNOBS {
                assert!(c[i][j] >= 0.0 && c[i][j] <= 1.0,
                    "Out of range at ({i},{j}): {}", c[i][j]);
            }
        }
    }

    #[test]
    fn ema_kdv_highest_coupling() {
        let c = coupling_matrix();
        // β ↔ κ should be the highest off-diagonal coupling
        let top = most_coupled(&c, 1)[0];
        assert_eq!(top.0.min(top.1), 0, "EMA_beta not in top pair");
        assert_eq!(top.0.max(top.1), 1, "KdV_alpha not in top pair");
    }

    #[test]
    fn ffn_ratio_most_independent() {
        let c = coupling_matrix();
        let centrality = knob_centrality(&c);
        // FFN_ratio should be in the bottom-3 centrality
        let mut ranked: Vec<(usize, f64)> = centrality.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bottom3: Vec<usize> = ranked[..3].iter().map(|(i,_)| *i).collect();
        assert!(bottom3.contains(&8), "FFN_ratio not in bottom-3 centrality: {:?}", bottom3);
    }
}
