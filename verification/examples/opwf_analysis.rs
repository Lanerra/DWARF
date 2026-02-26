//! OPWF (Outer-Product Wave Field) — Pre-training Rust Verification
//!
//! Answers four questions before writing condJ:
//!
//! 1. Math equivalence: does `Q @ (K⊗V)` exactly equal `(Q·K) * V`?
//! 2. Rank structure: how fast does the singular spectrum of `Σ K_t⊗V_t` decay?
//! 3. m/r factoring: is m=8,r=4 or m=4,r=8 more accurate for retrieval?
//! 4. Memory footprint: is the full 1024-channel OPWF feasible at batch=32?
//!
//! Run: `cd verification && cargo run --example opwf_analysis --release`

use rand::prelude::*;
use rand_distr::StandardNormal;

// ─── tiny linear algebra on f64 slices ────────────────────────────────────────

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Outer product: a[i]*b[j] → flat row-major [n x m].
fn outer(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; a.len() * b.len()];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i * b.len() + j] = ai * bj;
        }
    }
    out
}

/// Matrix–vector: M (n×m, row-major) times v (m) → w (n).  i.e. M @ v
fn matvec(m_mat: &[f64], n_rows: usize, n_cols: usize, v: &[f64]) -> Vec<f64> {
    assert_eq!(m_mat.len(), n_rows * n_cols);
    assert_eq!(v.len(), n_cols);
    (0..n_rows)
        .map(|i| dot(&m_mat[i * n_cols..(i + 1) * n_cols], v))
        .collect()
}

/// Left matrix–vector: q (n) times M (n×m, row-major) → w (m).  i.e. q^T @ M
/// This is the OPWF gather: output[j] = Σ_i Q[i] * F[i,j]
fn left_matvec(q: &[f64], m_mat: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    assert_eq!(q.len(), n_rows);
    assert_eq!(m_mat.len(), n_rows * n_cols);
    let mut result = vec![0.0f64; n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            result[j] += q[i] * m_mat[i * n_cols + j];
        }
    }
    result
}

/// Element-wise scale: returns v * scalar.
fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Element-wise add in place: a += b.
fn add_inplace(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Frobenius norm of a flat matrix.
fn frobenius(m: &[f64]) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L2 norm of a vector.
fn l2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalize a vector in place; returns the old norm.
fn normalize(v: &mut [f64]) -> f64 {
    let n = l2(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

/// Matrix multiply: A (n×k) × B (k×m) → C (n×m), all row-major.
fn matmul(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * m];
    for i in 0..n {
        for l in 0..k {
            for j in 0..m {
                c[i * m + j] += a[i * k + l] * b[l * m + j];
            }
        }
    }
    c
}

/// Transpose a (n×m) row-major matrix → (m×n).
fn transpose(a: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; m * n];
    for i in 0..n {
        for j in 0..m {
            t[j * n + i] = a[i * m + j];
        }
    }
    t
}

// ─── Power-iteration SVD (top-k singular values) ─────────────────────────────
//
// For an (n×n) matrix F, computes top-k singular triplets (σ_i, u_i, v_i)
// by deflation. Each step:
//   1. Power iteration on F^T F to find dominant right singular vector v_1.
//   2. σ_1 = ||F v_1||, u_1 = F v_1 / σ_1.
//   3. Deflate: F ← F - σ_1 u_1 v_1^T.

fn top_k_svd(f: &[f64], d: usize, k: usize, n_iter: usize, rng: &mut StdRng)
    -> Vec<(f64, Vec<f64>, Vec<f64>)>  // (sigma, u, v)
{
    let mut f_work = f.to_vec();
    let mut result = Vec::with_capacity(k);

    for _ in 0..k {
        // Random starting vector
        let mut v: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
        normalize(&mut v);

        // Power iterations on F^T F
        for _ in 0..n_iter {
            // w = F v
            let w = matvec(&f_work, d, d, &v);
            // v_new = F^T w
            let f_t = transpose(&f_work, d, d);
            let mut v_new = matvec(&f_t, d, d, &w);
            normalize(&mut v_new);
            v = v_new;
        }

        // σ = ||F v||, u = F v / σ
        let fv = matvec(&f_work, d, d, &v);
        let sigma = l2(&fv);
        if sigma < 1e-12 {
            break;
        }
        let u: Vec<f64> = fv.iter().map(|x| x / sigma).collect();

        // Deflate: F ← F - σ u v^T
        let uv = outer(&u, &v);
        for (fi, ui) in f_work.iter_mut().zip(uv.iter()) {
            *fi -= sigma * ui;
        }

        result.push((sigma, u, v));
    }
    result
}

// ─── Random normal vector helper ─────────────────────────────────────────────

fn randn_vec(d: usize, std: f64, rng: &mut StdRng) -> Vec<f64> {
    (0..d).map(|_| rng.sample::<f64, _>(StandardNormal) * std).collect()
}

// ─── Test 1: OPWF math equivalence ───────────────────────────────────────────

fn test_opwf_equivalence() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("TEST 1 — OPWF Math Equivalence");
    println!("Claim: Q @ (K ⊗ V) = (Q·K) * V  for any Q, K, V ∈ ℝ^d");
    println!("══════════════════════════════════════════════════════════");

    let d = 32usize;
    let n_trials = 10_000;
    let mut rng = StdRng::seed_from_u64(42);
    let mut max_err = 0.0f64;

    for _ in 0..n_trials {
        let q = randn_vec(d, 1.0, &mut rng);
        let k = randn_vec(d, 1.0, &mut rng);
        let v = randn_vec(d, 1.0, &mut rng);

        // Method A: full outer product, then left-multiply by Q
        // Q @ (K⊗V): result[j] = Σ_i Q[i] * K[i] * V[j] = (Q·K) * V[j]
        let kv = outer(&k, &v);      // [d×d], row-major: kv[i,j] = K[i]*V[j]
        let opwf_out = left_matvec(&q, &kv, d, d);  // [d]

        // Method B: scalar dot product, then scale
        let qk: f64 = dot(&q, &k);
        let direct_out = scale(&v, qk);

        // Error
        let err: f64 = opwf_out.iter().zip(&direct_out).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        if err > max_err { max_err = err; }
    }

    println!("  Trials:      {}", n_trials);
    println!("  d_head:      {}", d);
    println!("  Max abs err: {:.2e}   (expected < 1e-10)", max_err);
    let pass = max_err < 1e-10;
    println!("  PASS: {}", pass);
    assert!(pass, "OPWF equivalence test failed!");
}

// ─── Test 2: Accumulated field rank spectrum ──────────────────────────────────

fn test_field_rank_spectrum() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("TEST 2 — Singular Spectrum of Accumulated Field");
    println!("Claim: F = Σ_t K_t⊗V_t decays fast; top-4 captures most variance");
    println!("══════════════════════════════════════════════════════════");

    let d = 32usize;
    let n_contexts = 128usize;  // tokens per accumulation
    let n_trials = 100usize;
    let n_svd_iter = 50usize;
    let mut rng = StdRng::seed_from_u64(123);

    // Statistics we'll average over trials
    let mut frac_top4_total = 0.0f64;
    let mut frac_top8_total = 0.0f64;
    let mut frac_top1_total = 0.0f64;
    let mut sigma1_frac_total = 0.0f64;

    for _ in 0..n_trials {
        // Build accumulated field F = sum of T random outer products K_t ⊗ V_t
        // Use std=0.5 to simulate post-layer-norm attention vectors
        let mut f_accum = vec![0.0f64; d * d];
        for _ in 0..n_contexts {
            let k = randn_vec(d, 0.5, &mut rng);
            let v = randn_vec(d, 0.5, &mut rng);
            let kv = outer(&k, &v);
            add_inplace(&mut f_accum, &kv);
        }

        // Compute top-8 singular values
        let triplets = top_k_svd(&f_accum, d, 8, n_svd_iter, &mut rng);
        let sigmas: Vec<f64> = triplets.iter().map(|(s, _, _)| *s).collect();

        // Total squared Frobenius norm
        let _total_sq = frobenius(&f_accum).powi(2) +
            sigmas.iter().map(|s| s * s).sum::<f64>();
        // (Note: top_k_svd deflates, so the remaining F has lower energy;
        //  total = sum of all sigma^2, estimated as top-8 + remainder)
        // For display purposes, compute fraction of power in top-k vs total from SVD
        let sum_sq: f64 = sigmas.iter().map(|s| s * s).sum();
        let f_remaining_norm = frobenius(&f_accum);  // energy not in top 8
        let total_approx = sum_sq + f_remaining_norm * f_remaining_norm;

        let frac_top1 = sigmas[0] * sigmas[0] / total_approx;
        let frac_top4 = sigmas.iter().take(4).map(|s| s * s).sum::<f64>() / total_approx;
        let frac_top8 = sum_sq / total_approx;

        frac_top1_total += frac_top1;
        frac_top4_total += frac_top4;
        frac_top8_total += frac_top8;
        sigma1_frac_total += sigmas[0] / sigmas.iter().sum::<f64>();
    }

    let frac_top1 = frac_top1_total / n_trials as f64;
    let frac_top4 = frac_top4_total / n_trials as f64;
    let frac_top8 = frac_top8_total / n_trials as f64;
    let sigma1_frac = sigma1_frac_total / n_trials as f64;

    println!("  d_head:          {}", d);
    println!("  Context length:  {} tokens per field", n_contexts);
    println!("  Trials:          {}", n_trials);
    println!("  Avg variance captured:");
    println!("    Top-1 singular vector: {:.1}%", frac_top1 * 100.0);
    println!("    Top-4 singular vectors: {:.1}%", frac_top4 * 100.0);
    println!("    Top-8 singular vectors: {:.1}%", frac_top8 * 100.0);
    println!("  σ_1 / Σσ_i (spectral dominance): {:.3}", sigma1_frac);

    // Key question: is the field low-rank enough that factored approximation makes sense?
    println!("\n  Interpretation:");
    if frac_top4 > 0.80 {
        println!("    ✓ Top-4 captures >{:.0}% — rank-4 factoring preserves most info", frac_top4 * 100.0);
    } else {
        println!("    ✗ Top-4 only {:.0}% — field is higher-rank; full OPWF preferred", frac_top4 * 100.0);
    }
    if sigma1_frac > 0.50 {
        println!("    ✓ Single dominant singular direction (σ₁ = {:.0}% of sum)", sigma1_frac * 100.0);
    } else {
        println!("    ✓ Spread spectrum — multiple directions matter (σ₁ = {:.0}% of sum)", sigma1_frac * 100.0);
    }
}

// ─── Gram-Schmidt orthonormalisation ─────────────────────────────────────────

/// Generate a random (k × d) matrix with orthonormal rows via Gram-Schmidt.
fn random_ortho(k: usize, d: usize, rng: &mut StdRng) -> Vec<f64> {
    assert!(k <= d);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut v: Vec<f64> = (0..d).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();
        // subtract projections onto prior rows
        for r in &rows {
            let proj = dot(&v, r);
            for (vi, ri) in v.iter_mut().zip(r.iter()) {
                *vi -= proj * ri;
            }
        }
        normalize(&mut v);
        rows.push(v);
    }
    // flatten to [k × d] row-major
    rows.into_iter().flatten().collect()
}

/// Project vector `v` (length d) through a (k × d) row-major projection matrix.
/// Returns a k-dimensional vector.
fn project(v: &[f64], proj: &[f64], k: usize, d: usize) -> Vec<f64> {
    matvec(proj, k, d, v)
}

// ─── Test 3: m=8,r=4 vs m=4,r=8 retrieval accuracy ──────────────────────────

fn test_factored_retrieval() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("TEST 3 — Factored Retrieval Accuracy: m=8,r=4 vs m=4,r=8");
    println!("Metric A: Q-K routing rank correlation vs true Q·K");
    println!("Metric B: Value reconstruction error (relative L2)");
    println!("══════════════════════════════════════════════════════════");
    println!();
    println!("  Projection strategy: random orthonormal bases (optimal for");
    println!("  isotropic K,V vectors, which approximate post-layernorm attn).");
    println!("  W_k = random ortho [m×d]; W_v = random ortho [r×d].");
    println!("  W_o = W_v^T (optimal reconstruction from rank-r projection).");

    let d = 32usize;
    let n_ctx = 64usize;
    let n_trials = 1000usize;
    let mut rng = StdRng::seed_from_u64(999);

    // routing accuracy: Spearman corr between true Q·K and approx (W_qQ)·(W_kK)
    let mut routing_m8 = 0.0f64;
    let mut routing_m4 = 0.0f64;

    // value reconstruction: ||W_v^T (W_v V) - V||_2 / ||V||_2
    let mut val_err_r4 = 0.0f64;
    let mut val_err_r8 = 0.0f64;

    // combined output error: full OPWF vs factored (relative L2)
    let mut out_err_m8r4 = 0.0f64;
    let mut out_err_m4r8 = 0.0f64;

    fn rank_vec(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut ranks = vec![0.0f64; v.len()];
        for (r, &i) in idx.iter().enumerate() { ranks[i] = r as f64; }
        ranks
    }
    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let sa = a.iter().map(|x| (x-ma).powi(2)).sum::<f64>().sqrt();
        let sb = b.iter().map(|x| (x-mb).powi(2)).sum::<f64>().sqrt();
        if sa < 1e-12 || sb < 1e-12 { 0.0 } else { cov / (sa * sb) }
    }

    for _ in 0..n_trials {
        // Random projection matrices, fresh each trial
        let wk_m8 = random_ortho(8, d, &mut rng);  // [8 × d]
        let wk_m4 = random_ortho(4, d, &mut rng);  // [4 × d]
        let wv_r4 = random_ortho(4, d, &mut rng);  // [4 × d]
        let wv_r8 = random_ortho(8, d, &mut rng);  // [8 × d]

        let q = randn_vec(d, 0.5, &mut rng);
        let keys: Vec<Vec<f64>> = (0..n_ctx).map(|_| randn_vec(d, 0.5, &mut rng)).collect();
        let vals: Vec<Vec<f64>> = (0..n_ctx).map(|_| randn_vec(d, 0.5, &mut rng)).collect();

        // ── True full-rank output: Q @ Σ(K_t⊗V_t) = Σ_t (Q·K_t)*V_t ──────────
        let mut full_out = vec![0.0f64; d];
        for (k, v) in keys.iter().zip(vals.iter()) {
            add_inplace(&mut full_out, &scale(v, dot(&q, k)));
        }

        // ── Routing accuracy (Metric A) ───────────────────────────────────────
        let true_scores: Vec<f64> = keys.iter().map(|k| dot(&q, k)).collect();

        // m=8: Q and K projected to 8 dims → routing score = (W_m8 Q)·(W_m8 K)
        let q_m8 = project(&q, &wk_m8, 8, d);
        let approx_m8: Vec<f64> = keys.iter().map(|k| dot(&q_m8, &project(k, &wk_m8, 8, d))).collect();

        // m=4: same with 4-dim projection
        let q_m4 = project(&q, &wk_m4, 4, d);
        let approx_m4: Vec<f64> = keys.iter().map(|k| dot(&q_m4, &project(k, &wk_m4, 4, d))).collect();

        routing_m8 += pearson(&rank_vec(&true_scores), &rank_vec(&approx_m8));
        routing_m4 += pearson(&rank_vec(&true_scores), &rank_vec(&approx_m4));

        // ── Value reconstruction accuracy (Metric B) ──────────────────────────
        // W_v^T (W_v V) = projection of V onto r-dim subspace
        // Error = ||V - W_v^T W_v V||_2 / ||V||_2
        // For orthonormal W_v: W_v^T W_v is a projection matrix
        for v in &vals {
            // r=4
            let vp4 = project(v, &wv_r4, 4, d);
            let vr4 = matvec(&transpose(&wv_r4, 4, d), d, 4, &vp4); // W_v^T @ W_v @ v
            let err4: f64 = v.iter().zip(&vr4).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt() / l2(v).max(1e-12);
            val_err_r4 += err4;

            // r=8
            let vp8 = project(v, &wv_r8, 8, d);
            let vr8 = matvec(&transpose(&wv_r8, 8, d), d, 8, &vp8);
            let err8: f64 = v.iter().zip(&vr8).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt() / l2(v).max(1e-12);
            val_err_r8 += err8;
        }

        // ── Combined output error (Metric C): factored OPWF vs full ───────────
        // m=8,r=4: deposit φ_k(K)⊗(W_v V) into field [8×4], gather with W_k Q
        // Reconstruct with W_v^T (= W_v^T @ gathered [4]) → [d]
        let mut field_m8r4 = vec![0.0f64; 8 * 4];
        for (k, v) in keys.iter().zip(vals.iter()) {
            let kp = project(k, &wk_m8, 8, d);
            let vp = project(v, &wv_r4, 4, d);
            add_inplace(&mut field_m8r4, &outer(&kp, &vp));
        }
        let gathered_m8r4 = left_matvec(&q_m8, &field_m8r4, 8, 4);       // [4]
        let out_m8r4 = matvec(&transpose(&wv_r4, 4, d), d, 4, &gathered_m8r4); // [d]

        // m=4,r=8: K→4 dims, V→8 dims
        let mut field_m4r8 = vec![0.0f64; 4 * 8];
        for (k, v) in keys.iter().zip(vals.iter()) {
            let kp = project(k, &wk_m4, 4, d);
            let vp = project(v, &wv_r8, 8, d);
            add_inplace(&mut field_m4r8, &outer(&kp, &vp));
        }
        let gathered_m4r8 = left_matvec(&q_m4, &field_m4r8, 4, 8);       // [8]
        let out_m4r8 = matvec(&transpose(&wv_r8, 8, d), d, 8, &gathered_m4r8); // [d]

        let fn_full = l2(&full_out).max(1e-12);
        out_err_m8r4 += l2(&full_out.iter().zip(&out_m8r4).map(|(a,b)| a-b).collect::<Vec<_>>()) / fn_full;
        out_err_m4r8 += l2(&full_out.iter().zip(&out_m4r8).map(|(a,b)| a-b).collect::<Vec<_>>()) / fn_full;
    }

    let nt = n_trials as f64;
    let nv = (n_trials * n_ctx) as f64;

    println!();
    println!("  d_head={d}  context={n_ctx}  trials={n_trials}");
    println!();
    println!("  Metric A — Routing rank correlation (Spearman vs true Q·K):");
    println!("    m=8:  {:.4}  (higher m → better Q·K approximation)", routing_m8 / nt);
    println!("    m=4:  {:.4}", routing_m4 / nt);
    println!("    Δ routing corr: {:.4}  (m=8 advantage)", (routing_m8 - routing_m4) / nt);
    println!();
    println!("  Metric B — Value reconstruction error (relative L2):");
    println!("    r=4:  {:.4}  (lower r → worse value preservation)", val_err_r4 / nv);
    println!("    r=8:  {:.4}", val_err_r8 / nv);
    println!("    Δ value err: {:.4}  (r=4 disadvantage)", (val_err_r4 - val_err_r8) / nv);
    println!();
    println!("  Metric C — Combined output error (relative L2 vs full OPWF):");
    println!("    m=8,r=4:  {:.4}", out_err_m8r4 / nt);
    println!("    m=4,r=8:  {:.4}", out_err_m4r8 / nt);
    println!();
    println!("  Verdict:");
    let routing_advantage = (routing_m8 - routing_m4) / nt;
    let value_disadvantage = (val_err_r4 - val_err_r8) / nv;
    let out_m8 = out_err_m8r4 / nt;
    let out_m4 = out_err_m4r8 / nt;
    if out_m8 < out_m4 {
        println!("    m=8,r=4 produces lower combined output error ({:.4} vs {:.4}).", out_m8, out_m4);
        println!("    Routing improvement (+{:.4}) outweighs value degradation (+{:.4}).", routing_advantage, value_disadvantage);
        println!("    RECOMMENDATION: use m=8, r=4.");
    } else {
        println!("    m=4,r=8 produces lower combined output error ({:.4} vs {:.4}).", out_m4, out_m8);
        println!("    Value preservation outweighs routing discrimination at d_head={}.", d);
        println!("    RECOMMENDATION: use m=4, r=8.");
    }
    println!();
    println!("  Note: both options use {}-channel fields (32 total).", 8*4);
    println!("  Full OPWF (no factoring) uses {}-channel fields — 32× larger.", d*d);
}

// ─── Test 4: Memory footprint estimate ────────────────────────────────────────

fn test_memory_footprint() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("TEST 4 — Memory Footprint Estimate at Training Scale");
    println!("Config: B=32, H=8, N=2048, d_head=32, bf16");
    println!("══════════════════════════════════════════════════════════");

    let b: u64 = 32;
    let h: u64 = 8;
    let n: u64 = 2048;
    let d: u64 = 32;
    let bf16_bytes: u64 = 2;

    // Full OPWF: field [B, H, N, d, d]
    let full_field_elems = b * h * n * d * d;
    let full_field_mb = full_field_elems * bf16_bytes / 1_000_000;

    // Factored (m=8,r=4): field [B, H, N, 8, 4]
    let fact_field_elems = b * h * n * 8 * 4;
    let fact_field_mb = fact_field_elems * bf16_bytes / 1_000_000;

    // With gradients (×2) and activation checkpointing estimate (×0.5 for checkpointed)
    // Plus: 11 scales of convolved field (gains applied)
    let n_scales: u64 = 11;

    println!("\n  --- Full OPWF (d_head × d_head = 1024 channels) ---");
    println!("  Field tensor:        {:>6} MB  ({} elements)", full_field_mb, full_field_elems);
    println!("  + gradients:         {:>6} MB", full_field_mb);
    println!("  Conv outputs (×11):  {:>6} MB", full_field_mb * n_scales);
    println!("  ────────────────────────────");
    let full_total = full_field_mb * 2 + full_field_mb * n_scales;
    println!("  Peak estimate:       {:>6} MB  ({:.1} GB)", full_total, full_total as f64 / 1024.0);
    println!("  RTX 4090 (24 GB):   {}", if full_total < 20_000 { "FEASIBLE" } else { "LIKELY OOM" });

    println!("\n  --- Factored OPWF (m=8, r=4 = 32 channels) ---");
    println!("  Field tensor:        {:>6} MB  ({} elements)", fact_field_mb, fact_field_elems);
    println!("  + gradients:         {:>6} MB", fact_field_mb);
    println!("  Conv outputs (×11):  {:>6} MB", fact_field_mb * n_scales);
    let fact_total = fact_field_mb * 2 + fact_field_mb * n_scales;
    println!("  ────────────────────────────");
    println!("  Peak estimate:       {:>6} MB  ({:.1} GB)", fact_total, fact_total as f64 / 1024.0);
    println!("  RTX 4090 (24 GB):   {}", if fact_total < 20_000 { "FEASIBLE" } else { "LIKELY OOM" });

    println!("\n  Note: these are field-tensor estimates only, not full model memory.");
    println!("  Model activations, optimizer states (Adam ×3), and embeddings add ~6-8 GB.");
    println!("  At batch=32, the field tensor cost dominates for full OPWF.");
    println!("  Recommendation: If full OPWF OOMs, reduce batch to 16 or use factored variant.");
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Outer-Product Wave Field (OPWF) — Pre-Training Verification ║");
    println!("║  condJ design analysis — wave-field-llm                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    test_opwf_equivalence();
    test_field_rank_spectrum();
    test_factored_retrieval();
    test_memory_footprint();

    println!("\n══════════════════════════════════════════════════════════");
    println!("All tests complete. Fill §8.6 of write-up with results above.");
    println!("══════════════════════════════════════════════════════════\n");
}
