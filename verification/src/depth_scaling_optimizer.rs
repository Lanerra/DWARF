//! Depth Scaling Hyperparameter Optimizer
//!
//! Analytically computes optimal (or principled starting-point) values for
//! hyperparameters that scale with model depth (L) and parameter count.
//!
//! ## Rust-analyzable hyperparameters
//!
//! 1. **LR scaling formula** — `LR ∝ 1/sqrt(L/L_base)` (depth-adjusted)
//! 2. **scale_embed_lr_mult scaling** — `mult ≈ 15 * sqrt(L_base / L)`
//! 3. **Optimal effective batch size** — `BS_eff ∝ sqrt(params / params_base)`
//! 4. **Parameter counts** at each depth variant
//! 5. **Relay coverage** — gap-free frontier at each depth
//! 6. **IF block count and positions** at each depth
//! 7. **Gradient path length** — max hops from loss to L0 params
//! 8. **EMA block functional budget** — how many timescale registers available
//!
//! ## Not Rust-analyzable (requires training)
//!   - Warmup length optimal value
//!   - Dropout optimal value
//!   - Weight decay at large scale
//!   - Whether LR formula actually holds (empirical validation needed)

/// Base configuration (L=6, our calibrated starting point)
#[derive(Debug, Clone)]
pub struct DepthConfig {
    pub name: &'static str,
    pub num_layers: usize,          // L
    pub embedding_dim: usize,       // D
    pub ffn_dim: usize,             // FFN
    pub num_heads: usize,           // H
    pub num_offsets: usize,         // J
    pub interference_interval: usize, // every N layers has IF block
    pub vocab_size: usize,
    pub seq_len: usize,
}

const BASE_LR: f64 = 3e-4;
const BASE_LR_MULT: f64 = 15.0;
const BASE_BATCH: usize = 32;       // effective batch size at L=6
const BASE_L: usize = 6;
const BASE_PARAMS: f64 = 39_510_090.0; // J20D V10 L=6 actual

/// All depth variants we're testing
pub const DEPTH_CONFIGS: &[DepthConfig] = &[
    DepthConfig {
        name: "J20D-V10-L6 (baseline)",
        num_layers: 6, embedding_dim: 512, ffn_dim: 2048,
        num_heads: 8, num_offsets: 20, interference_interval: 2,
        vocab_size: 32000, seq_len: 2048,
    },
    DepthConfig {
        name: "J20D-V10-L8",
        num_layers: 8, embedding_dim: 512, ffn_dim: 2048,
        num_heads: 8, num_offsets: 20, interference_interval: 2,
        vocab_size: 32000, seq_len: 2048,
    },
    DepthConfig {
        name: "J20D-V10-L10",
        num_layers: 10, embedding_dim: 512, ffn_dim: 2048,
        num_heads: 8, num_offsets: 20, interference_interval: 2,
        vocab_size: 32000, seq_len: 2048,
    },
    DepthConfig {
        name: "J20D-V10-L12",
        num_layers: 12, embedding_dim: 512, ffn_dim: 2048,
        num_heads: 8, num_offsets: 20, interference_interval: 2,
        vocab_size: 32000, seq_len: 2048,
    },
    DepthConfig {
        name: "J20D-V10-L32 (extreme)",
        num_layers: 32, embedding_dim: 512, ffn_dim: 2048,
        num_heads: 8, num_offsets: 20, interference_interval: 2,
        vocab_size: 32000, seq_len: 2048,
    },
];

/// Compute exact parameter count for a DWARF V10 model
/// V10 kernel: no NPCI/MOVT/phase params (stripped from V8)
/// Per DSQG layer: norm1(D*2) + norm2(D*2) + QKV(D*3D+3D) + out(D*D+D) + gate(D*D+D)
///                + FFN(D*FFN+FFN + FFN*D+D) + pos_bias(J*H) + scale_embed(J*HD)
/// Per IF block extra: inter_norm(D*2) + inter_gate(D*D+D) + inter_k(D*D+D) + inter_v(D*D+D)
///                    + ema_factor(1)  [KdV removed in V10]
/// Full attn layer: norm1+norm2(D*4) + QKV(D*3D+3D) + out(D*D+D) + gate(D*D+D) + FFN
/// Embedding: V*D, pos_embed: (SEQ+2)*D, final norm: D*2, output: tied (0 extra)
pub fn param_count(cfg: &DepthConfig) -> usize {
    let d = cfg.embedding_dim;
    let h = cfg.num_heads;
    let hd = d / h;
    let ffn = cfg.ffn_dim;
    let j = cfg.num_offsets;
    let v = cfg.vocab_size;
    let seq = cfg.seq_len;
    let l = cfg.num_layers;
    let fa = l - 1; // full attn always last layer
    let iv = cfg.interference_interval;

    let mut total = 0usize;

    // Embeddings
    total += v * d;           // token embedding (tied with output)
    total += (seq + 2) * d;   // positional embedding

    for i in 0..l {
        if i == fa {
            // Full attention block
            total += d * 2 + d * 2;           // norm1 + norm2 (weight + bias each)
            total += d * 3 * d + 3 * d;       // QKV proj (weight + bias)
            total += d * d + d;               // out proj
            total += d * d + d;               // gate proj
            total += d * ffn + ffn;           // FFN fc1
            total += ffn * d + d;             // FFN fc2
        } else {
            let has_if = i % iv == iv - 1;
            // DSQG block (V10: no NPCI/MOVT/phase_base/phase_gain/query_probes/key_probes)
            total += d * 2 + d * 2;           // norm1 + norm2
            total += d * 3 * d + 3 * d;       // QKV proj
            total += d * d + d;               // out proj
            total += d * d + d;               // gate proj
            total += d * ffn + ffn;           // FFN fc1
            total += ffn * d + d;             // FFN fc2
            total += j * h;                   // pos_bias [J, H]
            total += j * hd;                  // scale_embed [J, HD]
            if has_if {
                total += d * 2 + d * 2;       // inter_norm (weight + bias)
                total += d * d + d;           // inter_gate
                total += d * d + d;           // inter_k_proj
                total += d * d + d;           // inter_v_proj
                total += 1;                   // ema_factor
                // kdv_alpha removed in V10
            }
        }
    }

    // Final norm
    total += d * 2;   // weight + bias

    total
}

/// Adjusted LR based on depth ratio (μP-inspired: LR ∝ 1/sqrt(L/L_base))
pub fn adjusted_lr(l: usize) -> f64 {
    BASE_LR / (l as f64 / BASE_L as f64).sqrt()
}

/// Adjusted scale_embed LR multiplier (gradient magnitude grows with depth)
/// At deeper L, more DSQG layers contribute gradient to scale_embed → effective
/// update is larger → reduce multiplier proportionally
pub fn adjusted_se_mult(l: usize) -> f64 {
    // scale_embed gradient ∝ (L-1) DSQG layers (exclude full attn layer)
    // At L=6: 5 DSQG layers. At L=N: N-1 DSQG layers.
    // Normalize: mult * sqrt(base_dsqg / current_dsqg)
    let base_dsqg = (BASE_L - 1) as f64;
    let curr_dsqg = (l - 1) as f64;
    BASE_LR_MULT * (base_dsqg / curr_dsqg).sqrt()
}

/// Optimal effective batch size (scales with sqrt of parameter ratio)
pub fn adjusted_batch(params: usize) -> usize {
    let ratio = params as f64 / BASE_PARAMS;
    let batch_f = BASE_BATCH as f64 * ratio.sqrt();
    // Round up to nearest multiple of 8 (for efficient GPU utilization)
    let batch = batch_f.ceil() as usize;
    ((batch + 7) / 8) * 8
}

/// Count IF (interference/EMA) blocks and their layer positions
pub fn if_block_info(cfg: &DepthConfig) -> (usize, Vec<usize>) {
    let fa = cfg.num_layers - 1;
    let iv = cfg.interference_interval;
    let positions: Vec<usize> = (0..cfg.num_layers)
        .filter(|&i| i != fa && i % iv == iv - 1)
        .collect();
    (positions.len(), positions)
}

/// Relay coverage: gap-free frontier formula
/// From March 13 analysis: gap-free frontier(L) ≈ 1723 + (L-5) * 1024 for L ≥ 5
/// Based on J24D offset set gap analysis. With J=20, coverage is similar.
pub fn relay_coverage(l: usize) -> usize {
    if l < 5 {
        // Approximate for shallow nets
        return l * 300;
    }
    1723 + (l.saturating_sub(5)) * 1024
}

/// Maximum gradient path length from loss to earliest layer parameter
/// = number of residual blocks the gradient must traverse
/// With residual connections, effective path ≈ L (not exponential)
/// but multiplicative factor across L LayerNorm + attention ops still matters
pub fn gradient_path_length(l: usize) -> usize {
    l // each block = 1 path step with residual shortcut
}

/// EMA functional timescale budget
/// Each IF block provides one timescale register.
/// Observed pattern: registers spontaneously specialize into
/// fast(~10t), medium(~40t), relay(~170t), ultra-long(dead-zone ~500t+)
/// More IF blocks → richer temporal hierarchy possible
pub fn ema_timescale_budget(cfg: &DepthConfig) -> (usize, String) {
    let (n_if, positions) = if_block_info(cfg);
    let description = match n_if {
        0 => "No EMA — pure DSQG".to_string(),
        1 => "1 register: single timescale (limited)".to_string(),
        2 => "2 registers: local + relay (current L=6 baseline)".to_string(),
        3 => "3 registers: local + relay + ultra-long (L=8, observed good)".to_string(),
        4 => "4 registers: local + intermediate + relay + ultra-long (L=10, observed good)".to_string(),
        5 => "5 registers: full hierarchy + redundancy (L=12)".to_string(),
        n if n <= 8  => format!("{n} registers: over-specified, likely overlap/collapse"),
        n            => format!("{n} registers: almost certainly collapsing to 3-4 effective"),
    };
    let _ = positions; // used for count
    (n_if, description)
}

/// Full analysis for one depth config
pub fn analyze(cfg: &DepthConfig) -> DepthAnalysis {
    let params = param_count(cfg);
    let lr = adjusted_lr(cfg.num_layers);
    let se_mult = adjusted_se_mult(cfg.num_layers);
    let batch = adjusted_batch(params);
    let coverage = relay_coverage(cfg.num_layers);
    let grad_path = gradient_path_length(cfg.num_layers);
    let (n_if, ema_desc) = ema_timescale_budget(cfg);
    let (_, if_positions) = if_block_info(cfg);

    DepthAnalysis {
        name: cfg.name,
        num_layers: cfg.num_layers,
        params,
        params_m: params as f64 / 1e6,
        lr_current: BASE_LR,
        lr_adjusted: lr,
        lr_reduction_pct: (1.0 - lr / BASE_LR) * 100.0,
        se_mult_current: BASE_LR_MULT,
        se_mult_adjusted: se_mult,
        se_mult_reduction_pct: (1.0 - se_mult / BASE_LR_MULT) * 100.0,
        batch_current: BASE_BATCH,
        batch_adjusted: batch,
        relay_coverage_tokens: coverage,
        relay_coverage_k: coverage as f64 / 1024.0,
        gradient_path: grad_path,
        n_if_blocks: n_if,
        if_positions,
        ema_description: ema_desc,
    }
}

#[derive(Debug)]
pub struct DepthAnalysis {
    pub name: &'static str,
    pub num_layers: usize,
    pub params: usize,
    pub params_m: f64,
    pub lr_current: f64,
    pub lr_adjusted: f64,
    pub lr_reduction_pct: f64,
    pub se_mult_current: f64,
    pub se_mult_adjusted: f64,
    pub se_mult_reduction_pct: f64,
    pub batch_current: usize,
    pub batch_adjusted: usize,
    pub relay_coverage_tokens: usize,
    pub relay_coverage_k: f64,
    pub gradient_path: usize,
    pub n_if_blocks: usize,
    pub if_positions: Vec<usize>,
    pub ema_description: String,
}

/// Print a formatted report for all depth configs
pub fn print_report() {
    println!("\n{}", "═".repeat(90));
    println!("  DWARF DEPTH SCALING HYPERPARAMETER ANALYSIS");
    println!("  Analytically-derived optimal settings for L=6..32");
    println!("{}", "═".repeat(90));

    let analyses: Vec<DepthAnalysis> = DEPTH_CONFIGS.iter().map(analyze).collect();

    // ── Parameter counts ──────────────────────────────────────────────────────
    println!("\n── 1. PARAMETER COUNTS ──────────────────────────────────────────────────");
    println!("  {:<28} {:>6}  {:>10}  {:>8}", "Config", "L", "Params", "ΔParams");
    println!("  {}", "─".repeat(58));
    let base_params = analyses[0].params;
    for a in &analyses {
        let delta = if a.params > base_params {
            format!("+{:.1}M", (a.params - base_params) as f64 / 1e6)
        } else {
            "baseline".to_string()
        };
        println!("  {:<28} {:>6}  {:>8.2}M  {:>8}",
            a.name, a.num_layers, a.params_m, delta);
    }

    // ── LR adjustments ────────────────────────────────────────────────────────
    println!("\n── 2. LEARNING RATE (LR ∝ 1/√(L/6)) ────────────────────────────────────");
    println!("  {:<28} {:>6}  {:>10}  {:>10}  {:>8}",
        "Config", "L", "LR current", "LR adjusted", "Reduction");
    println!("  {}", "─".repeat(68));
    for a in &analyses {
        println!("  {:<28} {:>6}  {:>10.2e}  {:>10.2e}  {:>7.1}%",
            a.name, a.num_layers,
            a.lr_current, a.lr_adjusted, a.lr_reduction_pct);
    }
    println!("  Note: Current runs use LR_current throughout. LR_adjusted is recommended.");

    // ── scale_embed LR mult ───────────────────────────────────────────────────
    println!("\n── 3. SCALE_EMBED LR MULTIPLIER (mult ∝ √(5/(L-1))) ─────────────────────");
    println!("  {:<28} {:>6}  {:>12}  {:>12}  {:>8}",
        "Config", "L", "Mult current", "Mult adjusted", "Reduction");
    println!("  {}", "─".repeat(72));
    for a in &analyses {
        println!("  {:<28} {:>6}  {:>12.2}  {:>12.2}  {:>7.1}%",
            a.name, a.num_layers,
            a.se_mult_current, a.se_mult_adjusted, a.se_mult_reduction_pct);
    }

    // ── Effective batch size ──────────────────────────────────────────────────
    println!("\n── 4. EFFECTIVE BATCH SIZE (BS ∝ √(params/params_base)) ─────────────────");
    println!("  {:<28} {:>6}  {:>10}  {:>12}  {:>14}",
        "Config", "L", "BS current", "BS adjusted", "Config (BS/GA)");
    println!("  {}", "─".repeat(72));
    for a in &analyses {
        // Suggest BS=8 (VRAM constraint) with GRAD_ACCUM adjusted
        let ga = (a.batch_adjusted + 7) / 8;
        println!("  {:<28} {:>6}  {:>10}  {:>12}  {:>14}",
            a.name, a.num_layers,
            a.batch_current, a.batch_adjusted,
            format!("BS=8, GA={ga}"));
    }

    // ── Relay coverage ────────────────────────────────────────────────────────
    println!("\n── 5. RELAY COVERAGE (gap-free frontier ≈ 1723 + (L-5)×1024) ────────────");
    println!("  {:<28} {:>6}  {:>10}  {:>10}",
        "Config", "L", "Coverage", "vs N=2048");
    println!("  {}", "─".repeat(58));
    for a in &analyses {
        let vs_n = if a.relay_coverage_tokens >= 2048 {
            format!("{:.1}× seq_len", a.relay_coverage_tokens as f64 / 2048.0)
        } else {
            format!("{:.0}% seq_len", a.relay_coverage_tokens as f64 / 2048.0 * 100.0)
        };
        println!("  {:<28} {:>6}  {:>7}t ({:>4.1}K)  {:>10}",
            a.name, a.num_layers,
            a.relay_coverage_tokens, a.relay_coverage_k, vs_n);
    }

    // ── EMA blocks and timescale budget ──────────────────────────────────────
    println!("\n── 6. EMA BLOCKS AND TIMESCALE HIERARCHY ────────────────────────────────");
    println!("  {:<28} {:>6}  {:>3}  {:>12}  {}",
        "Config", "L", "IF#", "Positions", "Timescale budget");
    println!("  {}", "─".repeat(90));
    for a in &analyses {
        let pos_str = format!("{:?}", a.if_positions)
            .replace('[', "").replace(']', "")
            .replace(' ', "");
        println!("  {:<28} {:>6}  {:>3}  {:>12}  {}",
            a.name, a.num_layers, a.n_if_blocks, pos_str, a.ema_description);
    }

    // ── Gradient path ─────────────────────────────────────────────────────────
    println!("\n── 7. GRADIENT PATH AND DEPTH RISK ──────────────────────────────────────");
    println!("  {:<28} {:>6}  {:>8}  {}",
        "Config", "L", "Path len", "Risk assessment");
    println!("  {}", "─".repeat(80));
    for a in &analyses {
        let risk = match a.num_layers {
            l if l <= 8  => "LOW    — standard residual nets handle this easily",
            l if l <= 16 => "MEDIUM — monitor L0 pos_bias development at ep1",
            l if l <= 24 => "HIGH   — warmup strongly recommended; watch L0 closely",
            _            => "VERY HIGH — consider auxiliary losses if L0 not learning",
        };
        println!("  {:<28} {:>6}  {:>8}  {}",
            a.name, a.num_layers, a.gradient_path, risk);
    }

    // ── Summary recommendations ───────────────────────────────────────────────
    println!("\n{}", "═".repeat(90));
    println!("  RECOMMENDED SETTINGS FOR NEXT RUNS");
    println!("{}", "═".repeat(90));
    println!("  {:<28}  {:<10}  {:<7}  {:<6}  {}", "Config", "LR", "SE_mult", "Eff_BS", "Warmup");
    println!("  {}", "─".repeat(70));
    for a in &analyses {
        let warmup = match a.num_layers {
            l if l <= 8  => "none",
            l if l <= 12 => "500 steps",
            l if l <= 24 => "1000 steps",
            _            => "2000 steps",
        };
        let ga = (a.batch_adjusted + 7) / 8;
        println!("  {:<28}  {:.2e}  {:>7.2}  {:>6}  {}",
            a.name, a.lr_adjusted, a.se_mult_adjusted,
            format!("BS=8,GA={ga}"), warmup);
    }
    println!();
    println!("  Note: LR formula is μP-inspired heuristic, not empirically validated.");
    println!("  Validate with 3-point LR sweep at L=12 before committing to L=32.");
    println!("{}", "═".repeat(90));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_counts_match_known() {
        // L=6 J20D V10 actual: 39,510,090 (from training log)
        let cfg = &DEPTH_CONFIGS[0];
        let computed = param_count(cfg);
        let known = 39_510_090usize;
        let err_pct = (computed as f64 - known as f64).abs() / known as f64 * 100.0;
        println!("L=6 computed: {computed}, known: {known}, err: {err_pct:.2}%");
        assert!(err_pct < 2.0, "L=6 param count off by more than 2%: {computed} vs {known}");

        // L=8 actual: 47,132,059 (from training log)
        let cfg8 = &DEPTH_CONFIGS[1];
        let computed8 = param_count(cfg8);
        let known8 = 47_132_059usize;
        let err8 = (computed8 as f64 - known8 as f64).abs() / known8 as f64 * 100.0;
        println!("L=8 computed: {computed8}, known: {known8}, err: {err8:.2}%");
        assert!(err8 < 2.0, "L=8 param count off by more than 2%: {computed8} vs {known8}");

        // L=10 actual: 54,754,028 (from training log)
        let cfg10 = &DEPTH_CONFIGS[2];
        let computed10 = param_count(cfg10);
        let known10 = 54_754_028usize;
        let err10 = (computed10 as f64 - known10 as f64).abs() / known10 as f64 * 100.0;
        println!("L=10 computed: {computed10}, known: {known10}, err: {err10:.2}%");
        assert!(err10 < 2.0, "L=10 param count off by more than 2%");
    }

    #[test]
    fn test_lr_scaling_monotone() {
        // LR should strictly decrease as L increases
        let lrs: Vec<f64> = DEPTH_CONFIGS.iter()
            .map(|c| adjusted_lr(c.num_layers))
            .collect();
        for w in lrs.windows(2) {
            assert!(w[0] > w[1], "LR not monotonically decreasing with L");
        }
    }

    #[test]
    fn test_se_mult_scaling_monotone() {
        // SE mult should strictly decrease as L increases
        let mults: Vec<f64> = DEPTH_CONFIGS.iter()
            .map(|c| adjusted_se_mult(c.num_layers))
            .collect();
        for w in mults.windows(2) {
            assert!(w[0] > w[1], "SE mult not monotonically decreasing with L");
        }
    }

    #[test]
    fn test_relay_coverage_grows() {
        // Coverage should grow with L
        let covs: Vec<usize> = DEPTH_CONFIGS.iter()
            .map(|c| relay_coverage(c.num_layers))
            .collect();
        for w in covs.windows(2) {
            assert!(w[0] < w[1], "Relay coverage not growing with L");
        }
    }

    #[test]
    fn test_if_block_counts() {
        // L=6: 5 DSQG layers, IF at every 2nd → L1, L3 = 2 blocks
        let (n6, pos6) = if_block_info(&DEPTH_CONFIGS[0]);
        assert_eq!(n6, 2, "L=6 should have 2 IF blocks");
        assert_eq!(pos6, vec![1, 3]);

        // L=8: 7 DSQG layers, IF at L1,L3,L5 = 3 blocks
        let (n8, pos8) = if_block_info(&DEPTH_CONFIGS[1]);
        assert_eq!(n8, 3, "L=8 should have 3 IF blocks");
        assert_eq!(pos8, vec![1, 3, 5]);

        // L=10: 9 DSQG layers, IF at L1,L3,L5,L7 = 4 blocks
        let (n10, pos10) = if_block_info(&DEPTH_CONFIGS[2]);
        assert_eq!(n10, 4, "L=10 should have 4 IF blocks");
        assert_eq!(pos10, vec![1, 3, 5, 7]);

        // L=32: 31 DSQG layers, IF at every odd layer except last = 15 blocks
        let (n32, _) = if_block_info(&DEPTH_CONFIGS[4]);
        assert_eq!(n32, 15, "L=32 should have 15 IF blocks");
    }

    #[test]
    fn print_full_report() {
        print_report();
    }
}
