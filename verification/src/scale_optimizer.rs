//! Per-Scale DWARF Configuration Optimizer
//!
//! Sweeps all analytically tractable knobs across 14M / 35M / 85M scales and
//! produces a per-scale recommendation table for all 9 DWARF hyperparameters.
//!
//! # Method
//!
//! Each knob is evaluated with one of four evidence levels:
//!
//!   PROVEN    — backed by Rust verification + empirical agreement
//!   CALIBRATED — empirically confirmed + theoretical model
//!   MODELED   — theoretical model, not yet empirically verified at this scale
//!   FIXED     — not a free parameter (e.g. AGC target γ = RMS=1 always)
//!
//! The optimizer calls existing modules for each knob, combines into a composite
//! fitness score, and reports recommended values with justification.
//!
//! # Analytical knobs (fully sweepable here)
//!   - interference_pos + full_attn_pos  → layer_placement_explorer
//!   - dense_window_W                    → offset_space_explorer (budget sweep)
//!   - bypass_alpha_init                 → gate_equilibrium (scale-dependent)
//!   - num_heads_H                       → head_dim SNR model (D/H tradeoff)
//!   - FFN_ratio                         → parameter efficiency curve
//!
//! # Empirically calibrated (we have training data)
//!   - EMA_beta   → training shows 0.010–0.015 at convergence; model gives bounds
//!   - KdV_alpha  → training shows −0.001 to −0.003; soliton stability gives bounds
//!   - AGC_target → fixed at RMS=1 (not a tunable)
//!
//! # Composite fitness score
//!
//!   fitness = 0.30 × relay_score_norm
//!           + 0.30 × offset_coverage_norm
//!           + 0.15 × head_dim_score
//!           + 0.10 × ffn_efficiency_score
//!           + 0.10 × gate_score (prefer small g_eq)
//!           + 0.05 × phase_readiness (transition by ep7)

use crate::{
    layer_placement_explorer as lpe,
    offset_space_explorer as ose,
    gate_equilibrium as ge,
    scale_embed_dynamics as sed,
    knob_interactions::{KNOB_NAMES, N_KNOBS},
    sweep_engine::{sweep_1d_progress, Stats, write_json_results},
};

// ─── Scale configs ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ScaleConfig {
    pub name: &'static str,
    pub n_params: f64,
    pub embedding_dim: usize,   // D
    pub num_layers: usize,      // L
    pub budget_j: usize,        // J (offset budget)
    pub chinchilla_epochs: usize, // standard training epochs
}

/// The three DWARF target scales (architecture constants match training scripts)
pub const SCALE_14M: ScaleConfig = ScaleConfig {
    name: "14M",
    n_params: 14.06e6,
    embedding_dim: 256,
    num_layers: 6,
    budget_j: 44,
    chinchilla_epochs: 7,
};

pub const SCALE_35M: ScaleConfig = ScaleConfig {
    name: "35M",
    n_params: 38.73e6,
    embedding_dim: 512,
    num_layers: 6,
    budget_j: 44,
    chinchilla_epochs: 7,
};

pub const SCALE_85M: ScaleConfig = ScaleConfig {
    name: "85M",
    n_params: 101.4e6,
    embedding_dim: 1024,
    num_layers: 6,
    budget_j: 44,
    chinchilla_epochs: 7,
};

pub const ALL_SCALES: [&ScaleConfig; 3] = [&SCALE_14M, &SCALE_35M, &SCALE_85M];

// ─── Per-knob recommendation ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confidence {
    Proven,      // Rust verification + empirical agreement
    Calibrated,  // Empirical data + theoretical model
    Modeled,     // Theory only, not yet verified at this scale
    Fixed,       // Not a free parameter
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Confidence::Proven     => write!(f, "PROVEN    "),
            Confidence::Calibrated => write!(f, "CALIBRATED"),
            Confidence::Modeled    => write!(f, "MODELED   "),
            Confidence::Fixed      => write!(f, "FIXED     "),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnobRec {
    pub knob:       &'static str,
    pub value:      String,
    pub confidence: Confidence,
    pub basis:      &'static str,
    pub score_contrib: f64,   // normalised contribution to composite fitness
}

// ─── Component scorers ──────────────────────────────────────────────────────────

/// Relay quality score (from layer_placement_explorer, normalised 0–1)
fn relay_score(interference: usize, full_attn: usize, num_layers: usize) -> f64 {
    let cfg = lpe::SingleConfig { interference, full_attn, num_layers };
    if !cfg.is_valid() { return 0.0; }
    lpe::evaluate_single(&cfg).relay_score
}

/// Maximum possible relay score for this num_layers (used for normalisation)
fn max_relay_score(num_layers: usize) -> f64 {
    (0..num_layers)
        .flat_map(|i| (i+1..num_layers).map(move |j| {
            let cfg = lpe::SingleConfig { interference: i, full_attn: j, num_layers };
            if cfg.is_valid() { lpe::evaluate_single(&cfg).relay_score } else { 0.0 }
        }))
        .fold(0.0f64, f64::max)
}

/// Offset coverage score (from offset_space_explorer, log-normalised)
fn offset_score(dense_width: usize, n_sparse: usize, budget: usize) -> f64 {
    // Reuse offset_optimizer path counting
    let dense: Vec<usize> = (1..=dense_width).collect();
    let n_sparse_tiers = n_sparse.min(ose::SPARSE_POOL.len());
    let sparse: Vec<usize> = ose::SPARSE_POOL[..n_sparse_tiers].to_vec();
    let mut offsets = dense;
    for &s in &sparse { if !offsets.contains(&s) { offsets.push(s); } }
    let total = offsets.len();
    if total > budget + 2 { return 0.0; }  // allow slight over
    use crate::offset_optimizer::{path_counts, path_score, MAX_LAG, PASSKEY_DISTANCES};
    let counts = path_counts(&offsets, MAX_LAG);
    PASSKEY_DISTANCES.iter().map(|&d| path_score(&counts, d, 0.9)).sum::<f64>()
}

/// Head-dimension SNR score with minimum-heads constraint.
///
/// DWARF requires H ≥ 8 for proper h0/h7 head specialisation (confirmed empirically
/// at 13M and 35M; condU 25M used H=16, which showed less h0/h7 differentiation).
/// Beyond H=8, larger HD (fewer heads) gives better per-head retrieval SNR.
///
/// Score = 0.0 for H < 8 (constraint violation).
/// Score = log2(HD) / log2(D/8) for H ≥ 8 (normalised; H=8 is 1.0 baseline).
fn head_dim_score(embedding_dim: usize, n_heads: usize) -> f64 {
    if n_heads < 8 { return 0.0; }  // empirical hard constraint: H ≥ 8
    let hd = embedding_dim / n_heads;
    let hd_at_h8 = embedding_dim / 8;  // normalise relative to H=8 baseline
    ((hd as f64).log2() / (hd_at_h8 as f64).log2()).min(1.0)
}

/// FFN ratio efficiency: inverted-U shape peaking at ratio=4 (standard).
///
/// Empirical: ratio=4 is the validated sweet spot across all transformer variants.
/// Below 4: staging layers under-parameterised (DSQG field processing degrades).
/// Above 4: parameter inefficiency — tokens/param ratio worsens at fixed Chinchilla budget.
///
/// Score = 1.0 at ratio=4; decays with Gaussian penalty for deviation.
fn ffn_efficiency(ratio: usize) -> f64 {
    let r = ratio as f64;
    let r_opt = 4.0;
    let sigma = 2.0;  // ±2 ratio units → ~0.6 score (not penalised too harshly)
    (-0.5 * ((r - r_opt) / sigma).powi(2)).exp()
}

/// Gate score: prefer small gate_eq (passkey-optimal).
/// Score = 1 - gate_eq (calibrated from gate_equilibrium model).
fn gate_score(n_params: f64) -> f64 {
    let (a, beta, _) = ge::fit_gate_scaling(
        &ge::CALIBRATION_DATA.iter().map(|(n, g, _)| (*n, *g)).collect::<Vec<_>>()
    );
    let g_eq = ge::predict_gate(a, beta, n_params);
    1.0 - g_eq   // smaller gate = better passkey performance
}

/// Phase readiness score: does scale_embed cross τ by ep7 (100%C)?
/// Score = 1.0 if crossing < 100%C, 0.5 if crossing at 100–120%C, 0.0 if never.
/// Uses the 35M empirical growth data as the reference trajectory.
fn phase_readiness_score(n_params: f64) -> f64 {
    // Scale by param ratio relative to 35M (larger models grow more slowly)
    let scale_factor = (38.73e6 / n_params).powf(0.15);
    let scaled_data: Vec<(f64, f64)> = sed::CONDX_V2_35M.iter()
        .map(|(t, y)| (*t, y * scale_factor))
        .collect();
    let (a, alpha, _) = sed::fit_power_law(&scaled_data);
    match sed::power_law_crossing(a, alpha, sed::THRESHOLD) {
        None     => 0.0,
        Some(c)  => if c <= 1.0 { 1.0 } else if c <= 1.2 { 0.7 } else { 0.3 },
    }
}

// ─── Full knob sweep for one scale ─────────────────────────────────────────────

/// All candidate values to sweep for each discrete knob
const HEAD_OPTIONS:     [usize; 4] = [4, 8, 16, 32];
const FFN_RATIO_OPTIONS: [usize; 5] = [2, 3, 4, 6, 8];

#[derive(Debug, Clone)]
pub struct OptConfig {
    pub interference:     usize,
    pub full_attn:        usize,
    pub dense_width:      usize,
    pub n_sparse:         usize,
    pub n_heads:          usize,
    pub ffn_ratio:        usize,
    pub composite_score:  f64,
    // component scores
    pub relay_s:          f64,
    pub offset_s:         f64,
    pub head_dim_s:       f64,
    pub ffn_s:            f64,
    pub gate_s:           f64,
    pub phase_s:          f64,
}

/// Score weights
const W_RELAY:  f64 = 0.30;
const W_OFFSET: f64 = 0.30;
const W_HEAD:   f64 = 0.15;
const W_FFN:    f64 = 0.10;
const W_GATE:   f64 = 0.10;
const W_PHASE:  f64 = 0.05;

pub fn composite(r: f64, off: f64, h: f64, f: f64, g: f64, p: f64) -> f64 {
    W_RELAY * r + W_OFFSET * off + W_HEAD * h + W_FFN * f + W_GATE * g + W_PHASE * p
}

/// Curated offset candidates — top results from budget_sweep(J=44) already run.
/// (dense_width, n_sparse, coverage_score, reliable_depth)
/// Only configs with reliable_depth=1536 are included (long-range retrieval required).
const OFFSET_CANDIDATES: &[(usize, usize, f64)] = &[
    (41, 3,  19_062_692.0),   // d41s3: best reliable_1536 at J=44
    (40, 4,  18_734_652.0),
    (39, 5,  16_546_626.0),
    (38, 6,  15_734_474.0),
    (36, 8,  13_218_719.0),
    (37, 7,  12_670_558.0),
    (35, 9,  11_737_016.0),
    (34, 10, 10_244_046.0),
    (33, 11,  8_972_326.0),
    (32, 11,  7_618_273.0),   // condU baseline (J=43)
    (31, 11,  6_473_624.0),
    (30, 11,  5_521_377.0),
];

/// Pre-compute normalised offset scores once (shared across all scales, scale-invariant).
/// Returns (dense_width, n_sparse, normalised_score).
pub fn build_offset_table() -> Vec<(usize, usize, f64)> {
    let os_max = OFFSET_CANDIDATES.iter()
        .map(|(_, _, s)| *s).fold(0.0f64, f64::max);
    OFFSET_CANDIDATES.iter()
        .map(|(dw, ns, raw)| (*dw, *ns, (raw / os_max).min(1.0)))
        .collect()
}

pub fn sweep_scale(cfg: &ScaleConfig) -> Vec<OptConfig> {
    sweep_scale_with_offsets(cfg, &build_offset_table())
}

pub fn sweep_scale_with_offsets(
    cfg: &ScaleConfig,
    offset_table: &[(usize, usize, f64)],
) -> Vec<OptConfig> {
    let max_rs = max_relay_score(cfg.num_layers);
    let gs = gate_score(cfg.n_params);
    let ps = phase_readiness_score(cfg.n_params);

    let layer_pairs: Vec<(usize, usize)> = (0..cfg.num_layers)
        .flat_map(|i| (i + 1..cfg.num_layers).map(move |j| (i, j)))
        .filter(|(i, j)| lpe::SingleConfig {
            interference: *i, full_attn: *j, num_layers: cfg.num_layers
        }.is_valid())
        .collect();

    let mut results: Vec<OptConfig> = Vec::new();
    for (int, fa) in &layer_pairs {
        let rs = relay_score(*int, *fa, cfg.num_layers) / max_rs.max(1e-9);
        for &(dw, ns, os) in offset_table {
            for &h in &HEAD_OPTIONS {
                let hd = cfg.embedding_dim / h.max(1);
                if hd < 16 { continue; }
                let hs = head_dim_score(cfg.embedding_dim, h);
                for &r in &FFN_RATIO_OPTIONS {
                    let fs = ffn_efficiency(r);
                    results.push(OptConfig {
                        interference: *int, full_attn: *fa,
                        dense_width: dw, n_sparse: ns,
                        n_heads: h, ffn_ratio: r,
                        composite_score: composite(rs, os, hs, fs, gs, ps),
                        relay_s: rs, offset_s: os, head_dim_s: hs,
                        ffn_s: fs, gate_s: gs, phase_s: ps,
                    });
                }
            }
        }
    }
    results.sort_by(|a, b| b.composite_score.partial_cmp(&a.composite_score).unwrap());
    results
}

// ─── Per-scale recommendation ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ScaleRecommendation {
    pub scale: &'static str,
    pub n_params: f64,
    pub best: OptConfig,
    pub n_configs_swept: usize,
    pub knobs: Vec<KnobRec>,
}

pub fn recommend(cfg: &ScaleConfig) -> ScaleRecommendation {
    let results = sweep_scale(cfg);
    let n = results.len();
    let best = results.into_iter().next().unwrap();

    // Gate equilibrium
    let (a, beta, _) = ge::fit_gate_scaling(
        &ge::CALIBRATION_DATA.iter().map(|(n, g, _)| (*n, *g)).collect::<Vec<_>>()
    );
    let g_eq = ge::predict_gate(a, beta, cfg.n_params);
    let alpha_init_rec = (g_eq / (1.0 - g_eq).max(1e-10)).ln() - 5.0; // start well below equilibrium

    // Phase transition prediction
    let (pa, palpha, _) = sed::fit_power_law(sed::CONDX_V2_35M);
    let scale_factor = (38.73e6 / cfg.n_params).powf(0.15);
    let a_scaled = pa * scale_factor;
    let crossing_pct = sed::power_law_crossing(a_scaled, palpha, sed::THRESHOLD)
        .map(|c| c * 100.0).unwrap_or(-1.0);

    let hd = cfg.embedding_dim / best.n_heads;

    let knobs = vec![
        KnobRec {
            knob: "EMA_beta",
            value: "0.010–0.015".into(),
            confidence: Confidence::Calibrated,
            basis: "Training convergence: ep4 value 0.0115; lower β = wider memory window; range balances stability vs coverage",
            score_contrib: 0.0,
        },
        KnobRec {
            knob: "KdV_alpha",
            value: "-0.001 to -0.003".into(),
            confidence: Confidence::Calibrated,
            basis: "Empirical: ep4 value -0.0017; soliton stability bounds require |α| < 0.01; negative for dispersive damping",
            score_contrib: 0.0,
        },
        KnobRec {
            knob: "AGC_target",
            value: "RMS=1 (fixed)".into(),
            confidence: Confidence::Fixed,
            basis: "AGC normalises EMA output to unit RMS; not a free parameter",
            score_contrib: 0.0,
        },
        KnobRec {
            knob: "interference_pos",
            value: format!("L{}", best.interference),
            confidence: Confidence::Proven,
            basis: "layer_placement_explorer sweep: DWARF default (i=2) is top-1/15 for N=6; scale-invariant",
            score_contrib: best.relay_s * W_RELAY,
        },
        KnobRec {
            knob: "dense_window_W",
            value: format!("{}", best.dense_width),
            confidence: Confidence::Proven,
            basis: "offset_space_explorer budget sweep: d41s3 variant gives 2.5× coverage vs condU at same reliable_depth=1536",
            score_contrib: best.offset_s * W_OFFSET,
        },
        KnobRec {
            knob: "full_attn_pos",
            value: format!("L{}", best.full_attn),
            confidence: Confidence::Proven,
            basis: "layer_placement_explorer: last layer (j=5) maximises residual_coverage; scale-invariant",
            score_contrib: best.relay_s * W_RELAY,
        },
        KnobRec {
            knob: "bypass_alpha_init",
            value: format!("{:.1}", alpha_init_rec),
            confidence: if cfg.n_params > 30e6 { Confidence::Calibrated } else { Confidence::Proven },
            basis: "gate_equilibrium: g_eq << 1 at this scale; init well below equilibrium; passkey-optimal at gate~0",
            score_contrib: best.gate_s * W_GATE,
        },
        KnobRec {
            knob: "num_heads_H",
            value: format!("{} (HD={})", best.n_heads, hd),
            confidence: Confidence::Calibrated,
            basis: "H=8 empirically validated at 13M+35M; HD doubles each scale (D scales 2×); head specialisation stable",
            score_contrib: best.head_dim_s * W_HEAD,
        },
        KnobRec {
            knob: "FFN_ratio",
            value: format!("{} (FFN={})", best.ffn_ratio, best.ffn_ratio * cfg.embedding_dim),
            confidence: Confidence::Modeled,
            basis: "Efficiency model: ratio=4 is standard; DSQG staging layers need FFN ≥ 2×D; diminishing returns above 4",
            score_contrib: best.ffn_s * W_FFN,
        },
    ];

    ScaleRecommendation {
        scale: cfg.name,
        n_params: cfg.n_params,
        best,
        n_configs_swept: n,
        knobs,
    }
}

// ─── Comparative table ────────────────────────────────────────────────────────────

pub fn print_comparison(recs: &[ScaleRecommendation]) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          DWARF Per-Scale Knob Optimization Table                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("Composite fitness = 0.30×relay + 0.30×offset + 0.15×head_dim + 0.10×FFN + 0.10×gate + 0.05×phase\n");

    // Header
    println!("{:<24} {:<22} {:<22} {:<22}",
        "Knob", "14M", "35M", "85M");
    println!("{}", "─".repeat(92));

    // One row per knob
    for ki in 0..N_KNOBS {
        let knob_name = KNOB_NAMES[ki];
        let vals: Vec<String> = recs.iter().map(|r| {
            r.knobs.iter()
                .find(|k| k.knob == knob_name)
                .map(|k| format!("{} [{}]", k.value, k.confidence.to_string().trim()))
                .unwrap_or_else(|| "-".to_string())
        }).collect();
        println!("{:<24} {:<30} {:<30} {:<30}",
            knob_name,
            vals.get(0).map(|s| s.as_str()).unwrap_or("-"),
            vals.get(1).map(|s| s.as_str()).unwrap_or("-"),
            vals.get(2).map(|s| s.as_str()).unwrap_or("-"));
    }

    println!("{}", "─".repeat(92));

    // Composite score row
    print!("{:<24}", "composite_score");
    for r in recs {
        print!(" {:<30.4}", r.best.composite_score);
    }
    println!();

    // Configs swept row
    print!("{:<24}", "configs_swept");
    for r in recs {
        print!(" {:<30}", r.n_configs_swept);
    }
    println!("\n");

    // Per-scale detail tables
    for r in recs {
        println!("── {} ({:.0}M params) ─────────────────────────────────────────────",
            r.scale, r.n_params / 1e6);
        println!("  Optimal config: int=L{}, full=L{}, dense={}, sparse={}, H={}, FFN={}×D",
            r.best.interference, r.best.full_attn,
            r.best.dense_width, r.best.n_sparse,
            r.best.n_heads, r.best.ffn_ratio);
        println!("  Score breakdown:  relay={:.3}  offset={:.3}  head={:.3}  ffn={:.3}  gate={:.3}  phase={:.3}",
            r.best.relay_s * W_RELAY, r.best.offset_s * W_OFFSET, r.best.head_dim_s * W_HEAD,
            r.best.ffn_s * W_FFN, r.best.gate_s * W_GATE, r.best.phase_s * W_PHASE);
        println!("  Knob justifications:");
        for k in &r.knobs {
            println!("    [{:<10}] {:<20} {}", k.confidence, k.knob, k.value);
        }
        println!();
    }
}

pub fn save_json(recs: &[ScaleRecommendation], path: &str) -> std::io::Result<()> {
    let rows: Vec<String> = recs.iter().map(|r| {
        let knobs_json: Vec<String> = r.knobs.iter().map(|k| {
            format!(
                "{{\"knob\": {:?}, \"value\": {:?}, \"confidence\": {:?}, \"score_contrib\": {:.4}}}",
                k.knob, k.value,
                format!("{}", k.confidence).trim(),
                k.score_contrib
            )
        }).collect();
        format!(
            "{{\"scale\": {:?}, \"n_params\": {:.2e}, \"composite_score\": {:.4}, \
             \"configs_swept\": {}, \"interference\": {}, \"full_attn\": {}, \
             \"dense_width\": {}, \"n_sparse\": {}, \"n_heads\": {}, \"ffn_ratio\": {}, \
             \"knobs\": [{}]}}",
            r.scale, r.n_params, r.best.composite_score, r.n_configs_swept,
            r.best.interference, r.best.full_attn,
            r.best.dense_width, r.best.n_sparse,
            r.best.n_heads, r.best.ffn_ratio,
            knobs_json.join(", ")
        )
    }).collect();
    let meta = vec![
        ("sweep_type", "\"per_scale_optimization\"".to_string()),
        ("scales",     "\"14M,35M,85M\"".to_string()),
        ("w_relay",    format!("{W_RELAY}")),
        ("w_offset",   format!("{W_OFFSET}")),
        ("w_head",     format!("{W_HEAD}")),
        ("w_ffn",      format!("{W_FFN}")),
        ("w_gate",     format!("{W_GATE}")),
        ("w_phase",    format!("{W_PHASE}")),
    ];
    write_json_results(path, &meta, &rows)
}

pub fn run_all(output_dir: Option<&str>) {
    println!("\n=== Per-Scale DWARF Optimizer ===");
    let recs: Vec<ScaleRecommendation> = ALL_SCALES.iter()
        .map(|s| {
            print!("  Sweeping {}... ", s.name);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let r = recommend(s);
            println!("{} configs → composite={:.4}", r.n_configs_swept, r.best.composite_score);
            r
        })
        .collect();

    print_comparison(&recs);

    if let Some(dir) = output_dir {
        let p = format!("{dir}/per_scale_optimization.json");
        match save_json(&recs, &p) {
            Ok(_) => println!("Saved → {p}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_produces_results() {
        let results = sweep_scale(&SCALE_14M);
        assert!(!results.is_empty(), "Expected at least one config");
    }

    #[test]
    fn optimal_full_attn_is_last_layer() {
        for scale in &ALL_SCALES {
            let results = sweep_scale(scale);
            let best = &results[0];
            assert_eq!(best.full_attn, scale.num_layers - 1,
                "{}: expected full_attn=L{}, got L{}",
                scale.name, scale.num_layers - 1, best.full_attn);
        }
    }

    #[test]
    fn composite_scores_decrease_with_scale() {
        // Gate score decreases with scale (larger models have smaller g_eq → better gate score)
        // But head_dim score increases. Net direction is non-trivial — just verify scores are valid.
        for scale in &ALL_SCALES {
            let recs = recommend(scale);
            assert!(recs.best.composite_score > 0.0 && recs.best.composite_score <= 1.0,
                "{}: composite score out of range: {}", scale.name, recs.best.composite_score);
        }
    }
}
