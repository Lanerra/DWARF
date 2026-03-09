//! Layer Placement Explorer for DWARF
//!
//! Analytically evaluates all (interference_layer, full_attn_layer) configurations
//! for a given NUM_LAYERS, predicting holographic relay effectiveness.
//!
//! # Model
//!
//! DWARF's retrieval mechanism is a **two-stage relay**:
//!
//!   1. Interference block (at layer `i`) injects Huygens K/V into the residual stream,
//!      writing the passkey token into the distributed holographic field.
//!   2. DSQG staging layers (between `i` and `j`) propagate and consolidate the field
//!      across the residual stream — essential preprocessing for full attention decode.
//!   3. Full attention block (at layer `j`) reads the holographic field via contaminated Q.
//!
//! ## Relay Score Model
//!
//! The composite relay score for a given (interference=i, full_attn=j) pair:
//!
//!   staging_quality(n)   = 1 − exp(−n × λ_relay)
//!     where n = j − i − 1 staging layers between them.
//!     λ_relay ≈ 1.0 (calibrated so n=2 → q≈0.86, matching DWARF's actual config)
//!
//!   residual_coverage(j) = (NUM_LAYERS − 1 − j) / (NUM_LAYERS − 1)
//!     Full attention contribution to the final output; higher = more residual layers
//!     downstream. Note: last layer (j=NUM_LAYERS-1) has zero downstream residual —
//!     BUT its output goes directly into the output projection, so it's still effective.
//!     We model this as: residual_coverage(j) = exp(−(NUM_LAYERS−1−j)² × γ),
//!     which has a peak at the last layer (j = NUM_LAYERS−1).
//!     γ = 0.5 gives a soft peak.
//!
//!   early_penalty(i) = exp(−i × κ)
//!     Interference too early means less DSQG field pre-processing before injection.
//!     κ = 0.3.
//!
//!   relay_score = staging_quality × residual_coverage × early_penalty
//!
//! ## Calibration
//!
//! DWARF's actual config: INTERFERENCE at L2, FULL_ATTN at L5, NUM_LAYERS=6.
//! Empirical: L4 ablation → −48.3pp passkey (critical staging layer).
//! The model should rank (i=2, j=5) near the top for N=6.
//!
//! ## Multi-injection configurations
//!
//! Also sweeps two-interference-layer configs: (i1, i2, j) where i1 < i2 < j.
//! Combined staging quality = max(staging_quality(i2→j), staging_quality(i1→j)).

use crate::sweep_engine::{sweep_1d_progress, sweep_2d_progress, top_k, Stats, write_json_results};

/// DWARF default architecture constants
pub const NUM_LAYERS: usize = 6;

/// Relay model hyperparameters (calibrated to condU empirical data)
const LAMBDA_RELAY: f64 = 1.0;  // staging quality decay constant
const GAMMA_COVER:  f64 = 0.5;  // residual coverage peak sharpness
const KAPPA_EARLY:  f64 = 0.3;  // early injection penalty

// ─── Core scoring functions ───────────────────────────────────────────────────

/// Staging quality: how well `n` DSQG layers between interference and full attention
/// consolidate the holographic field. n=0 → 0.0, n≥3 → saturates near 1.0.
pub fn staging_quality(n_staging: usize) -> f64 {
    1.0 - (-LAMBDA_RELAY * n_staging as f64).exp()
}

/// Residual coverage: full attention's contribution to the final output.
/// Peaks at the last layer (distance from end = 0) with soft Gaussian falloff.
pub fn residual_coverage(full_attn_layer: usize, num_layers: usize) -> f64 {
    let dist_from_end = (num_layers - 1 - full_attn_layer) as f64;
    (-GAMMA_COVER * dist_from_end * dist_from_end).exp()
}

/// Early injection penalty: interference too early means the DSQG field
/// has less accumulated signal before injection.
pub fn early_penalty(interference_layer: usize) -> f64 {
    (-(KAPPA_EARLY * interference_layer as f64)).exp()
    // Note: exp(0)=1 at layer 0, smaller penalty at higher layers
    // We actually want LESS penalty for later interference → invert:
}

/// Correct early penalty: small penalty for early interference (layer 0 = worst).
/// A later interference position is better (more DSQG preprocessing before injection).
pub fn injection_readiness(interference_layer: usize) -> f64 {
    1.0 - (-KAPPA_EARLY * interference_layer as f64).exp()
}

// ─── Single-injection config ──────────────────────────────────────────────────

/// Configuration: one interference layer + one full-attention layer.
#[derive(Debug, Clone)]
pub struct SingleConfig {
    pub interference: usize,
    pub full_attn: usize,
    pub num_layers: usize,
}

/// Metrics for a single-injection configuration.
#[derive(Debug, Clone)]
pub struct SingleMetrics {
    pub staging_layers: usize,
    pub staging_quality: f64,
    pub residual_coverage: f64,
    pub injection_readiness: f64,
    pub relay_score: f64,
    pub is_dwarf_default: bool,
}

impl SingleConfig {
    pub fn is_valid(&self) -> bool {
        self.interference < self.full_attn && self.full_attn < self.num_layers
    }
}

pub fn evaluate_single(cfg: &SingleConfig) -> SingleMetrics {
    let n_staging = cfg.full_attn - cfg.interference - 1;
    let sq = staging_quality(n_staging);
    let rc = residual_coverage(cfg.full_attn, cfg.num_layers);
    let ir = injection_readiness(cfg.interference);
    SingleMetrics {
        staging_layers: n_staging,
        staging_quality: sq,
        residual_coverage: rc,
        injection_readiness: ir,
        relay_score: sq * rc * ir,
        is_dwarf_default: cfg.interference == 2 && cfg.full_attn == 5 && cfg.num_layers == 6,
    }
}

// ─── Double-injection config ──────────────────────────────────────────────────

/// Configuration: two interference layers + one full-attention layer.
#[derive(Debug, Clone)]
pub struct DoubleConfig {
    pub interference1: usize,
    pub interference2: usize,
    pub full_attn: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone)]
pub struct DoubleMetrics {
    pub staging_from_i1: usize,
    pub staging_from_i2: usize,
    pub relay_score: f64,
}

impl DoubleConfig {
    pub fn is_valid(&self) -> bool {
        self.interference1 < self.interference2
            && self.interference2 < self.full_attn
            && self.full_attn < self.num_layers
    }
}

pub fn evaluate_double(cfg: &DoubleConfig) -> DoubleMetrics {
    let s1 = cfg.full_attn - cfg.interference1 - 1;
    let s2 = cfg.full_attn - cfg.interference2 - 1;
    // Two injection points: field gets written at both; second write augments the first
    let sq = (staging_quality(s1) + 0.5 * staging_quality(s2)).min(1.0);
    let rc = residual_coverage(cfg.full_attn, cfg.num_layers);
    let ir = injection_readiness(cfg.interference2); // later injection more relevant
    DoubleMetrics {
        staging_from_i1: s1,
        staging_from_i2: s2,
        relay_score: sq * rc * ir,
    }
}

// ─── Sweeps ───────────────────────────────────────────────────────────────────

/// Run single-injection sweep: all valid (interference, full_attn) pairs.
pub fn run_single_sweep(
    num_layers: usize,
) -> Vec<crate::sweep_engine::SweepPoint<SingleConfig, SingleMetrics>> {
    let configs: Vec<SingleConfig> = (0..num_layers)
        .flat_map(|i| {
            (i + 1..num_layers).map(move |j| SingleConfig {
                interference: i,
                full_attn: j,
                num_layers,
            })
        })
        .filter(|c| c.is_valid())
        .collect();

    sweep_1d_progress(&configs, |c| evaluate_single(c), "single-injection sweep")
}

/// Run double-injection sweep: all valid (i1, i2, full_attn) triples.
pub fn run_double_sweep(
    num_layers: usize,
) -> Vec<crate::sweep_engine::SweepPoint<DoubleConfig, DoubleMetrics>> {
    let configs: Vec<DoubleConfig> = (0..num_layers)
        .flat_map(|i1| {
            (i1 + 1..num_layers).flat_map(move |i2| {
                (i2 + 1..num_layers).map(move |j| DoubleConfig {
                    interference1: i1,
                    interference2: i2,
                    full_attn: j,
                    num_layers,
                })
            })
        })
        .filter(|c| c.is_valid())
        .collect();

    sweep_1d_progress(&configs, |c| evaluate_double(c), "double-injection sweep")
}

// ─── Output ───────────────────────────────────────────────────────────────────

pub fn print_single_summary(
    results: &[crate::sweep_engine::SweepPoint<SingleConfig, SingleMetrics>],
) {
    let top = top_k(results, results.len(), |m| m.relay_score);
    println!("\n=== Single-injection layer placement (N={}) ===", NUM_LAYERS);
    println!("{:<6} {:<10} {:<8} {:<9} {:<9} {:<9} {:<12} {}",
        "int", "full_attn", "staging", "stag_q", "res_cov", "inj_rdy",
        "relay_score", "note");
    println!("{}", "-".repeat(82));
    for r in &top {
        let note = if r.metrics.is_dwarf_default { " ← DWARF" } else { "" };
        println!("{:<6} {:<10} {:<8} {:<9.4} {:<9.4} {:<9.4} {:<12.4}{}",
            r.params.interference, r.params.full_attn,
            r.metrics.staging_layers, r.metrics.staging_quality,
            r.metrics.residual_coverage, r.metrics.injection_readiness,
            r.metrics.relay_score, note);
    }

    let scores: Vec<f64> = results.iter().map(|r| r.metrics.relay_score).collect();
    let s = Stats::of(&scores);
    println!("Score stats: {}", s.summary());
}

pub fn save_single_json(
    results: &[crate::sweep_engine::SweepPoint<SingleConfig, SingleMetrics>],
    path: &str,
) -> std::io::Result<()> {
    let rows: Vec<String> = results.iter().map(|r| {
        format!(
            "{{\"interference\": {}, \"full_attn\": {}, \"num_layers\": {}, \
             \"staging_layers\": {}, \"staging_quality\": {:.6}, \
             \"residual_coverage\": {:.6}, \"injection_readiness\": {:.6}, \
             \"relay_score\": {:.6}, \"is_dwarf_default\": {}}}",
            r.params.interference, r.params.full_attn, r.params.num_layers,
            r.metrics.staging_layers, r.metrics.staging_quality,
            r.metrics.residual_coverage, r.metrics.injection_readiness,
            r.metrics.relay_score, r.metrics.is_dwarf_default,
        )
    }).collect();
    let meta = vec![
        ("sweep_type", "\"layer_placement_single\"".to_string()),
        ("num_layers", format!("{NUM_LAYERS}")),
        ("lambda_relay", format!("{LAMBDA_RELAY}")),
        ("gamma_cover", format!("{GAMMA_COVER}")),
        ("kappa_early", format!("{KAPPA_EARLY}")),
    ];
    write_json_results(path, &meta, &rows)
}

/// Run all layer placement sweeps.
pub fn run_all(output_dir: Option<&str>) {
    println!("\n=== Layer Placement Explorer ===");
    println!("Model: relay_score = staging_quality × residual_coverage × injection_readiness");
    println!("Calibration: DWARF default (int=2, full=5, N=6) should rank near top\n");

    // Print model function values for reference
    println!("staging_quality by gap:     0→{:.4} 1→{:.4} 2→{:.4} 3→{:.4} 4→{:.4}",
        staging_quality(0), staging_quality(1), staging_quality(2),
        staging_quality(3), staging_quality(4));
    println!("residual_coverage by layer: L0→{:.4} L2→{:.4} L3→{:.4} L4→{:.4} L5→{:.4}",
        residual_coverage(0,6), residual_coverage(2,6), residual_coverage(3,6),
        residual_coverage(4,6), residual_coverage(5,6));
    println!("injection_readiness by pos: L0→{:.4} L1→{:.4} L2→{:.4} L3→{:.4}",
        injection_readiness(0), injection_readiness(1),
        injection_readiness(2), injection_readiness(3));
    println!();

    let single = run_single_sweep(NUM_LAYERS);
    print_single_summary(&single);

    if let Some(dir) = output_dir {
        let p = format!("{dir}/layer_placement_single.json");
        match save_single_json(&single, &p) {
            Ok(_) => println!("Saved → {p}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }

    // Double-injection summary
    let double = run_double_sweep(NUM_LAYERS);
    let top_double = top_k(&double, 5, |m| m.relay_score);
    println!("\n=== Top-5 double-injection configs ===");
    println!("{:<6} {:<6} {:<10} {:<12}",
        "int1", "int2", "full_attn", "relay_score");
    println!("{}", "-".repeat(40));
    for r in &top_double {
        println!("{:<6} {:<6} {:<10} {:.4}",
            r.params.interference1, r.params.interference2,
            r.params.full_attn, r.metrics.relay_score);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dwarf_default_near_top() {
        let results = run_single_sweep(NUM_LAYERS);
        let top3 = top_k(&results, 3, |m| m.relay_score);
        let dwarf_in_top3 = top3.iter().any(|r| r.metrics.is_dwarf_default);
        // The DWARF default config (int=2, full=5) should be in the top 3
        assert!(dwarf_in_top3,
            "DWARF default not in top-3: {:?}",
            top3.iter().map(|r| (r.params.interference, r.params.full_attn)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn staging_quality_monotone() {
        for n in 0..5 {
            assert!(staging_quality(n) < staging_quality(n + 1),
                "staging_quality not monotone at n={}", n);
        }
    }

    #[test]
    fn zero_staging_is_zero() {
        assert_eq!(staging_quality(0), 0.0);
    }

    #[test]
    fn last_layer_full_attn_max_coverage() {
        // Full attn at last layer should have highest residual_coverage
        let last = residual_coverage(NUM_LAYERS - 1, NUM_LAYERS);
        for l in 0..NUM_LAYERS - 1 {
            assert!(last >= residual_coverage(l, NUM_LAYERS),
                "last layer coverage not maximal: L{} > L{}", l, NUM_LAYERS-1);
        }
    }
}
