//! Offset Set Space Explorer for DWARF
//!
//! Exhaustively searches offset set design configurations and predicts
//! retrieval capability before committing to any training run.
//!
//! # Core question
//!
//! *Given a budget of J offsets, what is the best allocation between
//! dense-local (consecutive from δ=1) and sparse-long-range (dyadic tiers)?*
//!
//! DWARF currently uses J=44 offsets: dense [1..32] (32 offsets) + sparse
//! [48,64,96,128,192,256,384,512,768,1024,1536] (11 offsets) = 43 offsets.
//!
//! # Sweeps provided
//!
//! 1. **2D free sweep** — dense_width ∈ [1,64] × n_sparse ∈ [0,11].
//!    All 780 configurations, Rayon parallel. Shows the full capability surface.
//!
//! 2. **Budget-constrained sweep** — fix J=44 total offsets, vary dense_width
//!    from 1..43. Sparse tiers fill the remaining budget. Shows which allocation
//!    of the *same budget* maximises retrieval depth and coverage.
//!
//! 3. **Dense-only vs sparse-only extremes** — ablation confirming that both
//!    components are necessary. Pure dense runs out of gradient paths above
//!    the window; pure sparse cannot reach short distances directly.
//!
//! # Metrics
//!
//! - `path_count[d]` — number of ≤6-hop paths reaching passkey distance d.
//!   More paths = stronger gradient signal = faster learning.
//! - `coverage_score` — weighted sum of path scores across all 12 passkey
//!   distances (hop_discount=0.75 models multi-hop learning cost).
//! - `max_retrieval_depth` — farthest passkey distance with path_count > 0.
//! - `budget_used` — actual number of distinct offsets (≤ dense_width + n_sparse).
//!
//! # Running
//!
//! ```bash
//! cd verification
//! PATH="$HOME/.cargo/bin:$PATH" cargo test offset_space -- --nocapture 2>&1 | head -80
//! ```
//!
//! To save JSON output, call `run_and_save()` from an example binary.

use crate::offset_optimizer::{path_counts, path_score, MAX_LAG, NUM_LAYERS, PASSKEY_DISTANCES};
use crate::sweep_engine::{sweep_1d_progress, sweep_2d_progress, top_k, Stats, write_json_results};

// ─── Offset set construction ──────────────────────────────────────────────────

/// The 11 standard dyadic sparse tiers used by condU/condV/condX.
pub const SPARSE_POOL: &[usize] = &[48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536];

/// Standard condU/condV/condX offset set (J=43).
pub fn offsets_condu() -> Vec<usize> {
    let mut o: Vec<usize> = (1..=32).collect();
    o.extend_from_slice(SPARSE_POOL);
    o.sort_unstable();
    o.dedup();
    o
}

/// Build an offset set from a dense window [1..dense_width] plus a selection
/// of sparse tiers. `sparse_tiers` should be a subset of SPARSE_POOL or
/// any custom sparse positions.
pub fn build_offset_set(dense_width: usize, sparse_tiers: &[usize]) -> Vec<usize> {
    let mut o: Vec<usize> = (1..=dense_width).collect();
    for &t in sparse_tiers {
        if t >= 1 && t < MAX_LAG {
            o.push(t);
        }
    }
    o.sort_unstable();
    o.dedup();
    o
}

/// Select `n` evenly log-spaced tiers from SPARSE_POOL (1..=11).
/// n=0 → empty; n=11 → full pool.
pub fn canonical_sparse(n: usize) -> Vec<usize> {
    let pool = SPARSE_POOL;
    if n == 0 { return vec![]; }
    let n = n.min(pool.len());
    if n == pool.len() { return pool.to_vec(); }
    // Pick indices 0, step, 2*step, ... where step = (pool.len()-1)/(n-1)
    // so first and last are always included.
    if n == 1 { return vec![pool[pool.len() / 2]]; }
    let step = (pool.len() - 1) as f64 / (n - 1) as f64;
    (0..n).map(|i| pool[(i as f64 * step).round() as usize]).collect()
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Metrics computed for one offset set configuration.
#[derive(Debug, Clone)]
pub struct OffsetMetrics {
    /// Path score for each passkey distance in PASSKEY_DISTANCES (same order).
    pub scores_by_distance: Vec<f64>,
    /// Raw path count (summed over all hops) per passkey distance.
    pub paths_by_distance: Vec<u64>,
    /// Weighted sum of scores across all passkey distances (hop_discount=0.75).
    pub coverage_score: f64,
    /// Farthest passkey distance with at least one path (0 if none).
    pub max_retrieval_depth: usize,
    /// Farthest passkey distance with path_count >= MIN_PATHS_THRESHOLD.
    pub reliable_retrieval_depth: usize,
    /// Total number of distinct offsets in the configuration.
    pub budget_used: usize,
}

/// Minimum path count considered "reliable" for gradient-based learning.
const MIN_PATHS_THRESHOLD: u64 = 2;

/// Hop discount: weight for k-hop paths relative to 1-hop. 0.75 means each
/// additional hop is 25% less valuable (multi-hop paths are harder to learn).
const HOP_DISCOUNT: f64 = 0.75;

/// Compute all metrics for a given offset set.
pub fn compute_metrics(offsets: &[usize]) -> OffsetMetrics {
    let counts = path_counts(offsets, NUM_LAYERS);

    let mut scores_by_distance = Vec::with_capacity(PASSKEY_DISTANCES.len());
    let mut paths_by_distance = Vec::with_capacity(PASSKEY_DISTANCES.len());
    let mut coverage_score = 0.0f64;
    let mut max_retrieval_depth = 0usize;
    let mut reliable_retrieval_depth = 0usize;

    for &d in PASSKEY_DISTANCES {
        let score = path_score(&counts, d, HOP_DISCOUNT);
        let total_paths: u64 = (1..=NUM_LAYERS).map(|k| counts[k][d]).sum();

        scores_by_distance.push(score);
        paths_by_distance.push(total_paths);
        coverage_score += score;

        if total_paths > 0 {
            max_retrieval_depth = max_retrieval_depth.max(d);
        }
        if total_paths >= MIN_PATHS_THRESHOLD {
            reliable_retrieval_depth = reliable_retrieval_depth.max(d);
        }
    }

    OffsetMetrics {
        scores_by_distance,
        paths_by_distance,
        coverage_score,
        max_retrieval_depth,
        reliable_retrieval_depth,
        budget_used: offsets.len(),
    }
}

// ─── Configuration types ──────────────────────────────────────────────────────

/// Parameters for one sweep configuration.
#[derive(Debug, Clone)]
pub struct OffsetConfig {
    pub dense_width: usize,
    pub n_sparse: usize,
}

/// Budget-constrained configuration: dense_width + sparse fill = J.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    pub dense_width: usize,
    pub total_budget: usize,
}

// ─── Sweeps ───────────────────────────────────────────────────────────────────

/// Run the 2D free sweep: dense_width ∈ [1,64] × n_sparse ∈ [0,11].
/// Returns all 780 sweep points.
pub fn run_2d_sweep() -> Vec<crate::sweep_engine::SweepPoint<OffsetConfig, OffsetMetrics>> {
    let dense_widths: Vec<usize> = (1..=64).collect();
    let n_sparse_vals: Vec<usize> = (0..=SPARSE_POOL.len()).collect();

    sweep_2d_progress(
        &dense_widths,
        &n_sparse_vals,
        |&dw, &ns| {
            let sparse = canonical_sparse(ns);
            let offsets = build_offset_set(dw, &sparse);
            let metrics = compute_metrics(&offsets);
            // Pack into struct; sweep_2d gives us (dw, ns) as params separately
            // so we wrap in OffsetConfig here for a consistent result type.
            let _ = (dw, ns); // params captured via closure, not returned inline
            metrics
        },
        "2D offset sweep (dense_width × n_sparse)",
    )
    .into_iter()
    .map(|sp| crate::sweep_engine::SweepPoint {
        params: OffsetConfig { dense_width: sp.params.0, n_sparse: sp.params.1 },
        metrics: sp.metrics,
    })
    .collect()
}

/// Run the budget-constrained sweep: fix J=44, vary dense_width from 1..43.
/// Remaining budget J - dense_width is allocated to canonical sparse tiers
/// (evenly log-spaced from SPARSE_POOL).
pub fn run_budget_sweep(total_budget: usize) -> Vec<crate::sweep_engine::SweepPoint<BudgetConfig, OffsetMetrics>> {
    let configs: Vec<BudgetConfig> = (1..total_budget)
        .map(|dw| BudgetConfig { dense_width: dw, total_budget })
        .collect();

    sweep_1d_progress(
        &configs,
        |cfg| {
            let n_sparse = total_budget.saturating_sub(cfg.dense_width).min(SPARSE_POOL.len());
            let sparse = canonical_sparse(n_sparse);
            let offsets = build_offset_set(cfg.dense_width, &sparse);
            compute_metrics(&offsets)
        },
        &format!("budget-constrained sweep (J={total_budget})"),
    )
}

/// Run the dense-only sweep: sparse tiers = empty, dense_width ∈ [1,100].
/// Shows where dense-only configurations cap out.
pub fn run_dense_only_sweep() -> Vec<crate::sweep_engine::SweepPoint<usize, OffsetMetrics>> {
    let widths: Vec<usize> = (1..=100).collect();
    sweep_1d_progress(
        &widths,
        |&dw| {
            let offsets = build_offset_set(dw, &[]);
            compute_metrics(&offsets)
        },
        "dense-only sweep",
    )
}

// ─── Output / reporting ───────────────────────────────────────────────────────

/// Format one OffsetMetrics as a JSON object string (no trailing comma).
fn metrics_to_json(cfg_json: &str, m: &OffsetMetrics) -> String {
    let scores: Vec<String> =
        m.scores_by_distance.iter().map(|s| format!("{s:.4}")).collect();
    let paths: Vec<String> = m.paths_by_distance.iter().map(|p| p.to_string()).collect();

    let dist_labels: Vec<String> = PASSKEY_DISTANCES.iter().map(|d| d.to_string()).collect();
    let scores_obj: String = dist_labels
        .iter()
        .zip(scores.iter())
        .map(|(d, s)| format!("\"{d}\": {s}"))
        .collect::<Vec<_>>()
        .join(", ");
    let paths_obj: String = dist_labels
        .iter()
        .zip(paths.iter())
        .map(|(d, p)| format!("\"{d}\": {p}"))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "{{{cfg_json}, \"budget_used\": {}, \"coverage_score\": {:.4}, \
         \"max_retrieval_depth\": {}, \"reliable_retrieval_depth\": {}, \
         \"scores_by_distance\": {{{scores_obj}}}, \
         \"paths_by_distance\": {{{paths_obj}}}}}",
        m.budget_used, m.coverage_score, m.max_retrieval_depth, m.reliable_retrieval_depth,
    )
}

/// Print a compact summary table for the top-K results of the 2D sweep.
pub fn print_top_k_summary(
    results: &[crate::sweep_engine::SweepPoint<OffsetConfig, OffsetMetrics>],
    k: usize,
) {
    let top = top_k(results, k, |m| m.coverage_score);
    println!("\n=== Top-{k} by coverage score ===");
    println!("{:<12} {:<10} {:<12} {:<20} {:<26}",
        "dense_width", "n_sparse", "budget", "max_retrieval", "coverage_score");
    println!("{}", "-".repeat(80));
    for r in &top {
        println!(
            "{:<12} {:<10} {:<12} {:<20} {:.4}",
            r.params.dense_width,
            r.params.n_sparse,
            r.metrics.budget_used,
            r.metrics.max_retrieval_depth,
            r.metrics.coverage_score,
        );
    }

    // Also show where condU sits in this ranking
    let condu_offsets = offsets_condu();
    let condu_metrics = compute_metrics(&condu_offsets);
    let rank = results
        .iter()
        .filter(|r| r.metrics.coverage_score > condu_metrics.coverage_score)
        .count()
        + 1;
    println!("\ncondU (dense=32, sparse=11): coverage={:.4}, max_depth={}, rank={}/{}",
        condu_metrics.coverage_score,
        condu_metrics.max_retrieval_depth,
        rank,
        results.len(),
    );
}

/// Print a summary table for the budget-constrained sweep.
pub fn print_budget_summary(
    results: &[crate::sweep_engine::SweepPoint<BudgetConfig, OffsetMetrics>],
    budget: usize,
) {
    println!("\n=== Budget-constrained sweep (J={budget}) — ranked by coverage ===");
    println!("{:<12} {:<10} {:<12} {:<22} {:<26}",
        "dense_width", "n_sparse", "budget", "reliable_depth", "coverage_score");
    println!("{}", "-".repeat(82));

    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| {
        b.metrics.coverage_score.partial_cmp(&a.metrics.coverage_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for r in sorted.iter().take(15) {
        let n_sparse = budget.saturating_sub(r.params.dense_width).min(SPARSE_POOL.len());
        println!(
            "{:<12} {:<10} {:<12} {:<22} {:.4}",
            r.params.dense_width,
            n_sparse,
            r.metrics.budget_used,
            r.metrics.reliable_retrieval_depth,
            r.metrics.coverage_score,
        );
    }

    // Highlight condU allocation (dense=32, sparse=11 → budget used=43)
    if let Some(condu) = results.iter().find(|r| r.params.dense_width == 32) {
        println!("\n→ condU allocation (dense=32): coverage={:.4}, reliable_depth={}",
            condu.metrics.coverage_score, condu.metrics.reliable_retrieval_depth);
    }

    // Coverage stats
    let all_scores: Vec<f64> = results.iter().map(|r| r.metrics.coverage_score).collect();
    let s = Stats::of(&all_scores);
    println!("Coverage score stats: {}", s.summary());
}

/// Save 2D sweep results to a JSON file.
pub fn save_2d_sweep_json(
    results: &[crate::sweep_engine::SweepPoint<OffsetConfig, OffsetMetrics>],
    path: &str,
) -> std::io::Result<()> {
    let rows: Vec<String> = results
        .iter()
        .map(|r| {
            let cfg_json =
                format!("\"dense_width\": {}, \"n_sparse\": {}", r.params.dense_width, r.params.n_sparse);
            metrics_to_json(&cfg_json, &r.metrics)
        })
        .collect();

    let meta = vec![
        ("sweep_type", "\"offset_space_2d\"".to_string()),
        ("dense_width_range", format!("[1, 64]")),
        ("n_sparse_range", format!("[0, {}]", SPARSE_POOL.len())),
        ("sparse_pool", format!("{:?}", SPARSE_POOL).replace('[', "[").replace(']', "]")),
        ("hop_discount", format!("{HOP_DISCOUNT}")),
        ("min_paths_threshold", format!("{MIN_PATHS_THRESHOLD}")),
        ("num_layers_max_hops", format!("{NUM_LAYERS}")),
        ("total_configs", format!("{}", results.len())),
    ];
    write_json_results(path, &meta, &rows)
}

/// Save budget-constrained sweep results to a JSON file.
pub fn save_budget_sweep_json(
    results: &[crate::sweep_engine::SweepPoint<BudgetConfig, OffsetMetrics>],
    path: &str,
    budget: usize,
) -> std::io::Result<()> {
    let rows: Vec<String> = results
        .iter()
        .map(|r| {
            let n_sparse =
                budget.saturating_sub(r.params.dense_width).min(SPARSE_POOL.len());
            let cfg_json = format!(
                "\"dense_width\": {}, \"n_sparse_allocated\": {}, \"total_budget\": {}",
                r.params.dense_width, n_sparse, r.params.total_budget
            );
            metrics_to_json(&cfg_json, &r.metrics)
        })
        .collect();

    let meta = vec![
        ("sweep_type", "\"offset_space_budget_constrained\"".to_string()),
        ("total_budget", format!("{budget}")),
        ("hop_discount", format!("{HOP_DISCOUNT}")),
        ("num_layers_max_hops", format!("{NUM_LAYERS}")),
    ];
    write_json_results(path, &meta, &rows)
}

// ─── Top-level runner ─────────────────────────────────────────────────────────

/// Run all three sweeps and optionally save JSON results.
///
/// Call this from an example binary or from tests with `--nocapture`.
pub fn run_all(output_dir: Option<&str>) {
    // 1. condU reference point
    let condu = offsets_condu();
    let condu_m = compute_metrics(&condu);
    println!("=== condU reference (J={}) ===", condu.len());
    println!("  coverage_score:         {:.4}", condu_m.coverage_score);
    println!("  max_retrieval_depth:    {}", condu_m.max_retrieval_depth);
    println!("  reliable_depth:         {}", condu_m.reliable_retrieval_depth);
    println!("  scores per distance:");
    for (&d, s) in PASSKEY_DISTANCES.iter().zip(condu_m.scores_by_distance.iter()) {
        println!("    d={:<6} score={:.4}  paths={}", d, s,
            condu_m.paths_by_distance[PASSKEY_DISTANCES.iter().position(|&x| x == d).unwrap()]);
    }
    println!();

    // 2. 2D free sweep
    let results_2d = run_2d_sweep();
    print_top_k_summary(&results_2d, 10);

    if let Some(dir) = output_dir {
        let path = format!("{dir}/offset_space_2d.json");
        match save_2d_sweep_json(&results_2d, &path) {
            Ok(_) => println!("Saved 2D sweep → {path}"),
            Err(e) => eprintln!("Failed to save 2D sweep: {e}"),
        }
    }

    // 3. Budget-constrained sweep (J=44)
    let budget = condu.len() + 1; // condU uses 43; round to 44 for clean budget
    let results_budget = run_budget_sweep(budget);
    print_budget_summary(&results_budget, budget);

    if let Some(dir) = output_dir {
        let path = format!("{dir}/offset_space_budget_{budget}.json");
        match save_budget_sweep_json(&results_budget, &path, budget) {
            Ok(_) => println!("Saved budget sweep → {path}"),
            Err(e) => eprintln!("Failed to save budget sweep: {e}"),
        }
    }

    // 4. Dense-only ablation
    let results_dense = run_dense_only_sweep();
    println!("\n=== Dense-only ablation ===");
    for r in results_dense.iter().filter(|r| [8,16,32,44,64,80,100].contains(&r.params)) {
        println!(
            "  dense={:<4} max_depth={:<6} reliable={:<6} coverage={:.4}",
            r.params, r.metrics.max_retrieval_depth,
            r.metrics.reliable_retrieval_depth, r.metrics.coverage_score,
        );
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condu_offsets_correct_count() {
        let o = offsets_condu();
        // Dense 1..32 = 32 offsets, sparse pool = 11, no overlap
        assert_eq!(o.len(), 43, "condU should have 43 distinct offsets");
        assert!(o.contains(&1));
        assert!(o.contains(&32));
        assert!(o.contains(&1536));
    }

    #[test]
    fn canonical_sparse_boundary_cases() {
        assert!(canonical_sparse(0).is_empty());
        assert_eq!(canonical_sparse(11).len(), 11); // full pool
        assert_eq!(canonical_sparse(1).len(), 1);
    }

    #[test]
    fn condu_can_reach_all_passkey_distances() {
        let o = offsets_condu();
        let m = compute_metrics(&o);
        // condU should have at least one path to every passkey distance
        for (&d, &paths) in PASSKEY_DISTANCES.iter().zip(m.paths_by_distance.iter()) {
            assert!(paths > 0, "condU has no path to d={d}");
        }
        assert_eq!(m.max_retrieval_depth, 1536);
    }

    #[test]
    fn dense_only_cannot_reach_long_distances() {
        // Dense 1..32 alone: max reachable = 32 * 6 hops = 192 in theory,
        // but combination-tone paths allow some longer reach.
        // Definitely cannot reach d=1536 without long-range offsets.
        let o = build_offset_set(32, &[]);
        let m = compute_metrics(&o);
        assert!(m.max_retrieval_depth < 1536,
            "dense-only should not reach d=1536, got {}", m.max_retrieval_depth);
    }

    #[test]
    fn build_offset_set_no_duplicates() {
        let o = build_offset_set(32, SPARSE_POOL);
        let mut sorted = o.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(o.len(), sorted.len(), "build_offset_set should return deduplicated set");
    }

    #[test]
    fn budget_sweep_smoke() {
        // Quick sanity: run budget sweep with J=10, should produce 9 results
        let results = run_budget_sweep(10);
        assert_eq!(results.len(), 9);
        for r in &results {
            assert!(r.metrics.budget_used <= 10);
        }
    }

    #[test]
    fn offset_space_run_all_smoke() {
        // Runs the full analysis; prints to stdout. Should not panic.
        // Use: cargo test offset_space_run_all_smoke -- --nocapture
        run_all(None);
    }
}
