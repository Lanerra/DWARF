//! Generic parallel sweep engine for DWARF exploration.
//!
//! Provides 1D and 2D grid sweeps with Rayon parallelism, progress tracking,
//! descriptive statistics, and JSON output compatible with existing result files.
//!
//! # Design goals
//!
//! - Any exploration reducible to "vary params, compute metric" fits here.
//! - Rayon outer loop: CPU-seconds for sweeps that would otherwise take hours.
//! - JSON output: format compatible with `benchmarks/logs/*.json` result files.
//! - No new crate dependencies (uses only the existing rayon dep + std).
//!
//! # Quick start
//!
//! ```rust
//! use wave_field_verification::sweep_engine::{sweep_1d_progress, top_k};
//!
//! let widths: Vec<usize> = (1..=64).collect();
//! let results = sweep_1d_progress(&widths, |w| my_score(*w), "dense-width");
//! for r in top_k(&results, 5, |s| *s) {
//!     println!("width={} → score={:.3}", r.params, r.metrics);
//! }
//! ```

use rayon::prelude::*;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ─── Core result type ─────────────────────────────────────────────────────────

/// One point in a sweep: the parameter configuration and its measured metric.
#[derive(Debug, Clone)]
pub struct SweepPoint<P, M> {
    pub params: P,
    pub metrics: M,
}

// ─── 1D sweeps ────────────────────────────────────────────────────────────────

/// Parallel 1D sweep. Results are in the **same order** as `params`.
///
/// `f` is called once per parameter value, in parallel via Rayon.
pub fn sweep_1d<P, M, F>(params: &[P], f: F) -> Vec<SweepPoint<P, M>>
where
    P: Send + Sync + Clone,
    M: Send,
    F: Fn(&P) -> M + Sync,
{
    params
        .par_iter()
        .map(|p| SweepPoint { params: p.clone(), metrics: f(p) })
        .collect()
}

/// Parallel 1D sweep with a stderr progress counter.
///
/// Prints "label — N configs, T threads" at the start, then "n/N (pct%)" as
/// work completes (at ~5% intervals). Final "done" on completion.
pub fn sweep_1d_progress<P, M, F>(params: &[P], f: F, label: &str) -> Vec<SweepPoint<P, M>>
where
    P: Send + Sync + Clone,
    M: Send,
    F: Fn(&P) -> M + Sync,
{
    let total = params.len();
    let done = Arc::new(AtomicUsize::new(0));
    let step = (total / 20).max(1);
    eprintln!("[{label}] {total} configs, {} threads", rayon::current_num_threads());

    let results: Vec<_> = {
        let done = done.clone();
        params
            .par_iter()
            .map(|p| {
                let m = f(p);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % step == 0 || n == total {
                    eprint!("\r  {n}/{total} ({:.0}%)   ", 100.0 * n as f64 / total as f64);
                    let _ = io::stderr().flush();
                }
                SweepPoint { params: p.clone(), metrics: m }
            })
            .collect()
    };

    eprintln!("\n[{label}] done");
    results
}

// ─── 2D sweeps ────────────────────────────────────────────────────────────────

/// Parallel 2D sweep over all (p, q) ∈ ps × qs.
///
/// Pairs are generated sequentially (fast), then the metric is computed
/// in parallel over all pairs. Results are in row-major order (outer × inner).
pub fn sweep_2d<P, Q, M, F>(ps: &[P], qs: &[Q], f: F) -> Vec<SweepPoint<(P, Q), M>>
where
    P: Send + Sync + Clone,
    Q: Send + Sync + Clone,
    M: Send,
    F: Fn(&P, &Q) -> M + Sync,
{
    // Pre-generate all pairs in order, then parallelise the expensive metric step.
    let pairs: Vec<(P, Q)> = ps
        .iter()
        .flat_map(|p| qs.iter().map(move |q| (p.clone(), q.clone())))
        .collect();

    pairs
        .par_iter()
        .map(|(p, q)| SweepPoint { params: (p.clone(), q.clone()), metrics: f(p, q) })
        .collect()
}

/// Parallel 2D sweep with a stderr progress counter.
pub fn sweep_2d_progress<P, Q, M, F>(
    ps: &[P],
    qs: &[Q],
    f: F,
    label: &str,
) -> Vec<SweepPoint<(P, Q), M>>
where
    P: Send + Sync + Clone,
    Q: Send + Sync + Clone,
    M: Send,
    F: Fn(&P, &Q) -> M + Sync,
{
    let pairs: Vec<(P, Q)> = ps
        .iter()
        .flat_map(|p| qs.iter().map(move |q| (p.clone(), q.clone())))
        .collect();

    let total = pairs.len();
    let done = Arc::new(AtomicUsize::new(0));
    let step = (total / 20).max(1);
    eprintln!(
        "[{label}] {}×{}={total} configs, {} threads",
        ps.len(),
        qs.len(),
        rayon::current_num_threads()
    );

    let results: Vec<_> = {
        let done = done.clone();
        pairs
            .par_iter()
            .map(|(p, q)| {
                let m = f(p, q);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % step == 0 || n == total {
                    eprint!("\r  {n}/{total} ({:.0}%)   ", 100.0 * n as f64 / total as f64);
                    let _ = io::stderr().flush();
                }
                SweepPoint { params: (p.clone(), q.clone()), metrics: m }
            })
            .collect()
    };

    eprintln!("\n[{label}] done");
    results
}

// ─── Ranking helpers ──────────────────────────────────────────────────────────

/// Return references to the top-K results, ranked by `score` descending.
/// K is clamped to results.len() if larger.
pub fn top_k<'a, P, M, F>(
    results: &'a [SweepPoint<P, M>],
    k: usize,
    score: F,
) -> Vec<&'a SweepPoint<P, M>>
where
    F: Fn(&M) -> f64,
{
    let mut indexed: Vec<(usize, f64)> =
        results.iter().enumerate().map(|(i, r)| (i, score(&r.metrics))).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k.min(results.len()));
    indexed.iter().map(|(i, _)| &results[*i]).collect()
}

/// Return references to the bottom-K results, ranked by `score` ascending.
pub fn bottom_k<'a, P, M, F>(
    results: &'a [SweepPoint<P, M>],
    k: usize,
    score: F,
) -> Vec<&'a SweepPoint<P, M>>
where
    F: Fn(&M) -> f64,
{
    let mut indexed: Vec<(usize, f64)> =
        results.iter().enumerate().map(|(i, r)| (i, score(&r.metrics))).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k.min(results.len()));
    indexed.iter().map(|(i, _)| &results[*i]).collect()
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Basic descriptive statistics for a slice of f64 values.
#[derive(Debug, Clone)]
pub struct Stats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

impl Stats {
    /// Compute statistics for `values`. Returns all-zero Stats for empty slices.
    pub fn of(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self { min: 0.0, max: 0.0, mean: 0.0, std: 0.0, count: 0 };
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / n as f64;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        Self { min, max, mean, std: var.sqrt(), count: n }
    }

    /// Compact one-line summary: "min=X mean=X max=X σ=X"
    pub fn summary(&self) -> String {
        format!(
            "min={:.4} mean={:.4} max={:.4} σ={:.4} (n={})",
            self.min, self.mean, self.max, self.std, self.count
        )
    }
}

/// Collect a scalar from each result and compute stats.
pub fn metric_stats<P, M, F>(results: &[SweepPoint<P, M>], extract: F) -> Stats
where
    F: Fn(&M) -> f64,
{
    let values: Vec<f64> = results.iter().map(|r| extract(&r.metrics)).collect();
    Stats::of(&values)
}

// ─── JSON output ──────────────────────────────────────────────────────────────

/// Write sweep results to a JSON file.
///
/// The caller formats each result as a JSON object string (without trailing comma).
/// This function wraps them in the standard result file structure:
///
/// ```json
/// {
///   "meta_key": meta_value,
///   ...
///   "results": [
///     { ... },
///     { ... }
///   ]
/// }
/// ```
///
/// `meta_kv` is a list of (key, already-JSON-formatted value) pairs.
/// `rows` is a list of pre-formatted JSON object strings for the results array.
pub fn write_json_results(
    path: &str,
    meta_kv: &[(&str, String)],
    rows: &[String],
) -> io::Result<()> {
    let mut out = String::with_capacity(256 + rows.len() * 128);
    out.push_str("{\n");
    for (k, v) in meta_kv {
        out.push_str(&format!("  \"{k}\": {v},\n"));
    }
    out.push_str("  \"results\": [\n");
    for (i, row) in rows.iter().enumerate() {
        out.push_str("    ");
        out.push_str(row);
        if i + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    std::fs::write(path, out)
}

/// Escape a string for JSON output (handles quotes and backslashes).
pub fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_1d_correct_order() {
        let params: Vec<usize> = (0..10).collect();
        let results = sweep_1d(&params, |p| *p * 2);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.params, i);
            assert_eq!(r.metrics, i * 2);
        }
    }

    #[test]
    fn sweep_2d_row_major_order() {
        let ps = vec![1usize, 2];
        let qs = vec![10usize, 20, 30];
        let results = sweep_2d(&ps, &qs, |p, q| p + q);
        assert_eq!(results.len(), 6);
        assert_eq!(results[0].params, (1, 10));
        assert_eq!(results[0].metrics, 11);
        assert_eq!(results[3].params, (2, 10));
        assert_eq!(results[5].params, (2, 30));
    }

    #[test]
    fn top_k_returns_highest() {
        let params: Vec<usize> = (0..5).collect();
        let results = sweep_1d(&params, |p| *p as f64);
        let top = top_k(&results, 2, |m| *m);
        assert_eq!(top[0].metrics, 4.0);
        assert_eq!(top[1].metrics, 3.0);
    }

    #[test]
    fn stats_basic() {
        let s = Stats::of(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 5.0).abs() < 1e-9);
        assert_eq!(s.count, 5);
    }

    #[test]
    fn stats_empty() {
        let s = Stats::of(&[]);
        assert_eq!(s.count, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn write_json_produces_valid_structure() {
        let rows = vec![
            "{\"x\": 1, \"y\": 2.0}".to_string(),
            "{\"x\": 2, \"y\": 4.0}".to_string(),
        ];
        let mut out_path = std::env::temp_dir();
        out_path.push("sweep_engine_test.json");
        let path_str = out_path.to_str().unwrap();
        write_json_results(path_str, &[("n", "2".to_string())], &rows).unwrap();
        let content = std::fs::read_to_string(path_str).unwrap();
        assert!(content.contains("\"results\""));
        assert!(content.contains("\"x\": 1"));
        assert!(content.contains("\"x\": 2"));
        // Last entry should NOT have trailing comma
        assert!(!content.contains("4.0},"));
    }
}
