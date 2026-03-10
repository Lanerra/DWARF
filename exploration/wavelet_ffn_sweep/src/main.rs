use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::SQRT_2;
use std::io::Write;

const D: usize = 256;
const H: usize = 8;
const HEAD_DIM: usize = D / H;
const STANDARD_FFN_PARAMS: usize = 2 * D * 4 * D;
const FULL_LEVELS: usize = 8;
#[allow(dead_code)]
const HEAD_LEVELS: usize = 5;
const FILTER_LENGTH: usize = 4;
const GELU_FLOPS_PER_ELEMENT: usize = 14;

// ─── Error function (Abramowitz & Stegun 7.1.26, max error < 1.5e-7) ───

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * abs_x);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-abs_x * abs_x).exp())
}

fn gelu(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf_approx(x / SQRT_2))
}

// ─── D4 wavelet filter coefficients ───

fn d4_filters() -> ([f64; 4], [f64; 4]) {
    let sqrt3 = 3.0_f64.sqrt();
    let denom = 4.0 * SQRT_2;
    let h0 = (1.0 + sqrt3) / denom;
    let h1 = (3.0 + sqrt3) / denom;
    let h2 = (3.0 - sqrt3) / denom;
    let h3 = (1.0 - sqrt3) / denom;
    let low = [h0, h1, h2, h3];
    // QMF: g[k] = (-1)^k * h[N-1-k]
    let high = [h3, -h2, h1, -h0];
    (low, high)
}

// ─── DWT core operations ───

fn circular_convolve_downsample(signal: &[f64], filter: &[f64; 4]) -> Vec<f64> {
    let length = signal.len();
    let output_length = length / 2;
    (0..output_length)
        .map(|i| {
            let pos = 2 * i;
            (0..4)
                .map(|j| filter[j] * signal[(pos + length - j) % length])
                .sum()
        })
        .collect()
}

fn upsample_circular_convolve(
    coefficients: &[f64],
    filter: &[f64; 4],
    output_length: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; output_length];
    for (i, &coeff) in coefficients.iter().enumerate() {
        let up_pos = 2 * i;
        for j in 0..4 {
            output[(up_pos + output_length - j) % output_length] += coeff * filter[j];
        }
    }
    output
}

struct DwtCoefficients {
    details: Vec<Vec<f64>>,
    approximation: Vec<f64>,
}

fn dwt_forward(signal: &[f64], levels: usize, low: &[f64; 4], high: &[f64; 4]) -> DwtCoefficients {
    let mut approx = signal.to_vec();
    let mut details = Vec::with_capacity(levels);
    for _ in 0..levels {
        let detail = circular_convolve_downsample(&approx, high);
        approx = circular_convolve_downsample(&approx, low);
        details.push(detail);
    }
    DwtCoefficients {
        details,
        approximation: approx,
    }
}

fn dwt_inverse(coeffs: &DwtCoefficients, low: &[f64; 4], high: &[f64; 4]) -> Vec<f64> {
    let mut approx = coeffs.approximation.clone();
    for level in (0..coeffs.details.len()).rev() {
        let target = coeffs.details[level].len() * 2;
        let low_part = upsample_circular_convolve(&approx, low, target);
        let high_part = upsample_circular_convolve(&coeffs.details[level], high, target);
        approx = low_part
            .iter()
            .zip(high_part.iter())
            .map(|(a, b)| a + b)
            .collect();
    }
    approx
}

// ─── Level selection ───

#[derive(Clone, Copy)]
enum LevelSelection {
    All,
    DenseOnly,
    SparseOnly,
    DsqgZones,
    LowFreq,
    HighFreq,
    Alternating,
    NoGelu,
}

const ALL_SELECTIONS: [LevelSelection; 8] = [
    LevelSelection::All,
    LevelSelection::DenseOnly,
    LevelSelection::SparseOnly,
    LevelSelection::DsqgZones,
    LevelSelection::LowFreq,
    LevelSelection::HighFreq,
    LevelSelection::Alternating,
    LevelSelection::NoGelu,
];

fn selection_name(sel: LevelSelection) -> &'static str {
    match sel {
        LevelSelection::All => "ALL",
        LevelSelection::DenseOnly => "DENSE_ONLY",
        LevelSelection::SparseOnly => "SPARSE_ONLY",
        LevelSelection::DsqgZones => "DSQG_ZONES",
        LevelSelection::LowFreq => "LOW_FREQ",
        LevelSelection::HighFreq => "HIGH_FREQ",
        LevelSelection::Alternating => "ALTERNATING",
        LevelSelection::NoGelu => "NONE",
    }
}

fn gelu_mask(sel: LevelSelection, max_levels: usize) -> (Vec<bool>, bool) {
    let mut detail = vec![false; max_levels];
    let approx;
    match sel {
        LevelSelection::All => {
            detail.fill(true);
            approx = true;
        }
        LevelSelection::DenseOnly => {
            for i in 0..max_levels.min(6) {
                detail[i] = true;
            }
            approx = false;
        }
        LevelSelection::SparseOnly => {
            for i in 6..max_levels {
                detail[i] = true;
            }
            approx = true;
        }
        LevelSelection::DsqgZones => {
            for &i in &[5_usize, 6, 7] {
                if i < max_levels {
                    detail[i] = true;
                }
            }
            approx = false;
        }
        LevelSelection::LowFreq => {
            for i in 4..max_levels {
                detail[i] = true;
            }
            approx = true;
        }
        LevelSelection::HighFreq => {
            for i in 0..max_levels.min(4) {
                detail[i] = true;
            }
            approx = false;
        }
        LevelSelection::Alternating => {
            for i in (0..max_levels).step_by(2) {
                detail[i] = true;
            }
            approx = true;
        }
        LevelSelection::NoGelu => {
            approx = false;
        }
    }
    (detail, approx)
}

// ─── Approximation coefficient handling ───

enum ApproxHandling {
    Identity,
    ApplyGelu,
    LinearProject(Vec<Vec<f64>>),
}

// ─── Wavelet FFN forward pass ───

fn wavelet_ffn_forward(
    input: &[f64],
    levels: usize,
    detail_gelu: &[bool],
    approx_handling: &ApproxHandling,
    low: &[f64; 4],
    high: &[f64; 4],
) -> Vec<f64> {
    let coeffs = dwt_forward(input, levels, low, high);

    let new_details: Vec<Vec<f64>> = coeffs
        .details
        .iter()
        .enumerate()
        .map(|(level, detail)| {
            if detail_gelu[level] {
                detail.iter().map(|&x| gelu(x)).collect()
            } else {
                detail.clone()
            }
        })
        .collect();

    let new_approx = match approx_handling {
        ApproxHandling::Identity => coeffs.approximation.clone(),
        ApproxHandling::ApplyGelu => coeffs.approximation.iter().map(|&x| gelu(x)).collect(),
        ApproxHandling::LinearProject(weights) => {
            let n = coeffs.approximation.len();
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| weights[i][j] * coeffs.approximation[j])
                        .sum::<f64>()
                })
                .collect()
        }
    };

    dwt_inverse(
        &DwtCoefficients {
            details: new_details,
            approximation: new_approx,
        },
        low,
        high,
    )
}

// ─── FLOPs computation ───

fn dwt_base_flops(signal_dim: usize, levels: usize) -> usize {
    let mut total = 0;
    let mut current = signal_dim;
    for _ in 0..levels {
        let output = current / 2;
        // 2 filters × output_dim outputs × filter_length multiply-adds × 2 FLOPs each
        total += 2 * output * FILTER_LENGTH * 2;
        current = output;
    }
    total
}

fn count_gelu_elements(
    detail_gelu: &[bool],
    approx_gelu: bool,
    signal_dim: usize,
    levels: usize,
) -> usize {
    let mut count = 0;
    let mut current = signal_dim;
    for level in 0..levels {
        current /= 2;
        if detail_gelu[level] {
            count += current;
        }
    }
    if approx_gelu {
        count += current;
    }
    count
}

fn total_flops(signal_dim: usize, levels: usize, gelu_element_count: usize) -> usize {
    let dwt = dwt_base_flops(signal_dim, levels);
    let idwt = dwt;
    let scale_bias = signal_dim * 2;
    let gelu_cost = gelu_element_count * GELU_FLOPS_PER_ELEMENT;
    dwt + idwt + scale_bias + gelu_cost
}

fn wavelet_ffn_params(
    signal_dim: usize,
    num_heads: usize,
    approx_is_proj: bool,
    approx_dim: usize,
) -> usize {
    let scale_bias = signal_dim * 2;
    let proj_params = if approx_is_proj {
        num_heads * approx_dim * approx_dim
    } else {
        0
    };
    scale_bias + proj_params
}

// ─── Metric helpers ───

fn norm_squared(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn random_vector(dim: usize, normal: &Normal<f64>, rng: &mut StdRng) -> Vec<f64> {
    (0..dim).map(|_| normal.sample(rng)).collect()
}

// ─── Energy preservation: E[||output||² / ||input||²] ───

fn compute_energy_preserved(
    dim: usize,
    levels: usize,
    detail_gelu: &[bool],
    approx_handling: &ApproxHandling,
    low: &[f64; 4],
    high: &[f64; 4],
    rng: &mut StdRng,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let samples = 1000;
    let mut ratio_sum = 0.0;
    for _ in 0..samples {
        let input = random_vector(dim, &normal, rng);
        let output = wavelet_ffn_forward(&input, levels, detail_gelu, approx_handling, low, high);
        let input_energy = norm_squared(&input);
        if input_energy > 1e-30 {
            ratio_sum += norm_squared(&output) / input_energy;
        }
    }
    ratio_sum / samples as f64
}

// ─── Gradient flow: E[||J||_F] via random projections ───
// For unit v uniform on sphere: E[||Jv||²] = ||J||²_F / dim
// So ||J||²_F ≈ dim × mean(||Jv||²)

fn compute_gradient_flow(
    dim: usize,
    levels: usize,
    detail_gelu: &[bool],
    approx_handling: &ApproxHandling,
    low: &[f64; 4],
    high: &[f64; 4],
    rng: &mut StdRng,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let input_samples = 100;
    let projections = 50;
    let epsilon = 1e-5;
    let mut frobenius_sum = 0.0;

    for _ in 0..input_samples {
        let input = random_vector(dim, &normal, rng);
        let output = wavelet_ffn_forward(&input, levels, detail_gelu, approx_handling, low, high);

        let mut jacobian_v_norm_sq_sum = 0.0;
        for _ in 0..projections {
            let direction = random_vector(dim, &normal, rng);
            let dir_norm = norm_squared(&direction).sqrt();
            let perturbed: Vec<f64> = input
                .iter()
                .zip(direction.iter())
                .map(|(&x, &d)| x + epsilon * d / dir_norm)
                .collect();
            let perturbed_output =
                wavelet_ffn_forward(&perturbed, levels, detail_gelu, approx_handling, low, high);
            let jacobian_times_v: Vec<f64> = perturbed_output
                .iter()
                .zip(output.iter())
                .map(|(&po, &o)| (po - o) / epsilon)
                .collect();
            jacobian_v_norm_sq_sum += norm_squared(&jacobian_times_v);
        }

        let estimated_frob_sq = (dim as f64) * jacobian_v_norm_sq_sum / (projections as f64);
        frobenius_sum += estimated_frob_sq.sqrt();
    }
    frobenius_sum / input_samples as f64
}

// ─── Effective rank: trace(C)² / trace(C²) ───
// Where C is the output covariance matrix

fn compute_effective_rank(
    dim: usize,
    levels: usize,
    detail_gelu: &[bool],
    approx_handling: &ApproxHandling,
    low: &[f64; 4],
    high: &[f64; 4],
    rng: &mut StdRng,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let samples = 1000;

    let mut outputs = Vec::with_capacity(samples);
    for _ in 0..samples {
        let input = random_vector(dim, &normal, rng);
        let output = wavelet_ffn_forward(&input, levels, detail_gelu, approx_handling, low, high);
        outputs.push(output);
    }

    let mut mean = vec![0.0; dim];
    for output in &outputs {
        for j in 0..dim {
            mean[j] += output[j];
        }
    }
    for value in &mut mean {
        *value /= samples as f64;
    }

    // trace(C) = (1/N) Σ_i ||y_i - ȳ||²
    let mut trace_covariance = 0.0;
    for output in &outputs {
        for j in 0..dim {
            let centered = output[j] - mean[j];
            trace_covariance += centered * centered;
        }
    }
    trace_covariance /= samples as f64;

    // Compute upper triangle of C, then trace(C²) = ||C||²_F
    let mut covariance = vec![vec![0.0; dim]; dim];
    for output in &outputs {
        for j in 0..dim {
            let cj = output[j] - mean[j];
            for k in j..dim {
                covariance[j][k] += cj * (output[k] - mean[k]);
            }
        }
    }

    let mut trace_covariance_squared = 0.0;
    for j in 0..dim {
        covariance[j][j] /= samples as f64;
        trace_covariance_squared += covariance[j][j] * covariance[j][j];
        for k in (j + 1)..dim {
            covariance[j][k] /= samples as f64;
            // Off-diagonal appears twice (j,k) and (k,j)
            trace_covariance_squared += 2.0 * covariance[j][k] * covariance[j][k];
        }
    }

    if trace_covariance_squared < 1e-30 {
        return 0.0;
    }
    trace_covariance * trace_covariance / trace_covariance_squared
}

// ─── Cross-head information flow (Full variant only) ───
// Measures ||C_off_diagonal||²_F / ||C||²_F as percentage

fn compute_cross_head_flow(
    levels: usize,
    detail_gelu: &[bool],
    approx_handling: &ApproxHandling,
    low: &[f64; 4],
    high: &[f64; 4],
    rng: &mut StdRng,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let samples = 500;

    let mut outputs = Vec::with_capacity(samples);
    for _ in 0..samples {
        let input = random_vector(D, &normal, rng);
        let output = wavelet_ffn_forward(&input, levels, detail_gelu, approx_handling, low, high);
        outputs.push(output);
    }

    let mut mean = vec![0.0; D];
    for output in &outputs {
        for j in 0..D {
            mean[j] += output[j];
        }
    }
    for value in &mut mean {
        *value /= samples as f64;
    }

    // Compute full covariance, then partition into within-head and cross-head
    let mut covariance = vec![vec![0.0; D]; D];
    for output in &outputs {
        for j in 0..D {
            let cj = output[j] - mean[j];
            for k in j..D {
                covariance[j][k] += cj * (output[k] - mean[k]);
            }
        }
    }
    for j in 0..D {
        covariance[j][j] /= samples as f64;
        for k in (j + 1)..D {
            covariance[j][k] /= samples as f64;
            covariance[k][j] = covariance[j][k];
        }
    }

    let mut total_frob_sq = 0.0;
    let mut within_frob_sq = 0.0;
    for j in 0..D {
        for k in 0..D {
            let val_sq = covariance[j][k] * covariance[j][k];
            total_frob_sq += val_sq;
            if j / HEAD_DIM == k / HEAD_DIM {
                within_frob_sq += val_sq;
            }
        }
    }

    if total_frob_sq < 1e-30 {
        return 0.0;
    }
    (total_frob_sq - within_frob_sq) / total_frob_sq * 100.0
}

// ─── Result structs ───

struct DimensionOneResult {
    selection: LevelSelection,
    energy_percent: f64,
    gradient_flow: f64,
    effective_rank: f64,
    total_flops_k: f64,
    combined_score: f64,
}

struct BlockCompareResult {
    selection_name: String,
    structure_name: &'static str,
    cross_head_percent: f64,
    params: usize,
    versus_ffn_percent: f64,
}

struct ApproxSweepResult {
    selection_name: String,
    approx_mode_name: &'static str,
    energy_percent: f64,
    gradient_flow: f64,
}

fn main() {
    println!(
        "🌊 Wavelet FFN Design Sweep (D={}, H={}, D4 wavelet)",
        D, H
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (low, high) = d4_filters();
    let mut rng = StdRng::seed_from_u64(42);

    // ── Verify DWT round-trip ──

    println!("▸ Verifying DWT round-trip...");
    let normal = Normal::new(0.0, 1.0).unwrap();
    let test_signal: Vec<f64> = (0..D).map(|_| normal.sample(&mut rng)).collect();
    let coeffs = dwt_forward(&test_signal, FULL_LEVELS, &low, &high);
    let reconstructed = dwt_inverse(&coeffs, &low, &high);
    let roundtrip_error: f64 = test_signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / norm_squared(&test_signal);
    assert!(
        roundtrip_error < 1e-10,
        "DWT round-trip error too large: {:.2e}",
        roundtrip_error
    );
    println!("  Round-trip MSE = {:.2e} ✓\n", roundtrip_error);

    // ═══ Dimension 1: Level Selection Sweep (Full variant) ═══

    println!("▸ Dimension 1: Level Selection Sweep (Full variant)");

    let mut dimension_one_results: Vec<DimensionOneResult> = Vec::new();

    for &sel in &ALL_SELECTIONS {
        let (detail_gelu, approx_active) = gelu_mask(sel, FULL_LEVELS);
        let approx_handling = if approx_active {
            ApproxHandling::ApplyGelu
        } else {
            ApproxHandling::Identity
        };

        println!("  {}...", selection_name(sel));

        let energy =
            compute_energy_preserved(D, FULL_LEVELS, &detail_gelu, &approx_handling, &low, &high, &mut rng);
        let gradient =
            compute_gradient_flow(D, FULL_LEVELS, &detail_gelu, &approx_handling, &low, &high, &mut rng);
        let rank =
            compute_effective_rank(D, FULL_LEVELS, &detail_gelu, &approx_handling, &low, &high, &mut rng);

        let gelu_count = count_gelu_elements(&detail_gelu, approx_active, D, FULL_LEVELS);
        let flops = total_flops(D, FULL_LEVELS, gelu_count);

        // Scoring: balance gradient stability, energy preservation, and nonlinearity coverage
        // An FFN replacement MUST have nonlinearity, so NONE gets score 0
        let ideal_gradient = (D as f64).sqrt();
        let gradient_proximity = (1.0 - (gradient / ideal_gradient - 1.0).abs()).max(0.0);
        let energy_stability = 1.0 - (1.0 - energy).powi(2);
        let coverage = gelu_count as f64 / D as f64;
        let nonlinearity_factor = if coverage > 0.0 { coverage.sqrt() } else { 0.0 };
        let combined = gradient_proximity * energy_stability * nonlinearity_factor;

        dimension_one_results.push(DimensionOneResult {
            selection: sel,
            energy_percent: energy * 100.0,
            gradient_flow: gradient,
            effective_rank: rank,
            total_flops_k: flops as f64 / 1000.0,
            combined_score: combined,
        });
    }

    // Rank by combined score (descending)
    let mut ranked_indices: Vec<usize> = (0..dimension_one_results.len()).collect();
    ranked_indices.sort_by(|&a, &b| {
        dimension_one_results[b]
            .combined_score
            .partial_cmp(&dimension_one_results[a].combined_score)
            .unwrap()
    });

    // ═══ Dimension 2: Block Structure Comparison (top-3) ═══

    println!("\n▸ Dimension 2: Block Structure Comparison (top-3)");

    let mut dimension_two_results: Vec<BlockCompareResult> = Vec::new();

    for &idx in ranked_indices.iter().take(3) {
        let sel = dimension_one_results[idx].selection;
        let name = selection_name(sel).to_string();
        println!("  {}...", name);

        // Full variant
        {
            let (detail_gelu, approx_active) = gelu_mask(sel, FULL_LEVELS);
            let approx_handling = if approx_active {
                ApproxHandling::ApplyGelu
            } else {
                ApproxHandling::Identity
            };
            let cross_head = compute_cross_head_flow(
                FULL_LEVELS,
                &detail_gelu,
                &approx_handling,
                &low,
                &high,
                &mut rng,
            );
            let params = wavelet_ffn_params(D, 1, false, 1);
            dimension_two_results.push(BlockCompareResult {
                selection_name: name.clone(),
                structure_name: "Full",
                cross_head_percent: cross_head,
                params,
                versus_ffn_percent: params as f64 / STANDARD_FFN_PARAMS as f64 * 100.0,
            });
        }

        // Per-Head variant
        {
            let params = wavelet_ffn_params(D, H, false, 1);
            dimension_two_results.push(BlockCompareResult {
                selection_name: name,
                structure_name: "Per-Head",
                cross_head_percent: 0.0,
                params,
                versus_ffn_percent: params as f64 / STANDARD_FFN_PARAMS as f64 * 100.0,
            });
        }
    }

    // ═══ Dimension 3: Approximation Coefficient Handling (top-2) ═══

    println!("\n▸ Dimension 3: Approximation Coefficient Handling (top-2)");

    let mut dimension_three_results: Vec<ApproxSweepResult> = Vec::new();

    for &idx in ranked_indices.iter().take(2) {
        let sel = dimension_one_results[idx].selection;
        let name = selection_name(sel).to_string();
        let (detail_gelu, _) = gelu_mask(sel, FULL_LEVELS);
        println!("  {}...", name);

        // PASS
        {
            let energy = compute_energy_preserved(
                D, FULL_LEVELS, &detail_gelu, &ApproxHandling::Identity, &low, &high, &mut rng,
            );
            let gradient = compute_gradient_flow(
                D, FULL_LEVELS, &detail_gelu, &ApproxHandling::Identity, &low, &high, &mut rng,
            );
            dimension_three_results.push(ApproxSweepResult {
                selection_name: name.clone(),
                approx_mode_name: "PASS",
                energy_percent: energy * 100.0,
                gradient_flow: gradient,
            });
        }

        // GELU
        {
            let energy = compute_energy_preserved(
                D, FULL_LEVELS, &detail_gelu, &ApproxHandling::ApplyGelu, &low, &high, &mut rng,
            );
            let gradient = compute_gradient_flow(
                D, FULL_LEVELS, &detail_gelu, &ApproxHandling::ApplyGelu, &low, &high, &mut rng,
            );
            dimension_three_results.push(ApproxSweepResult {
                selection_name: name.clone(),
                approx_mode_name: "GELU",
                energy_percent: energy * 100.0,
                gradient_flow: gradient,
            });
        }

        // LINEAR_PROJ
        {
            let approx_dim = D / (1 << FULL_LEVELS);
            let proj_normal = Normal::new(0.0, (1.0 / approx_dim as f64).sqrt()).unwrap();
            let proj_weights: Vec<Vec<f64>> = (0..approx_dim)
                .map(|_| (0..approx_dim).map(|_| proj_normal.sample(&mut rng)).collect())
                .collect();
            let handling = ApproxHandling::LinearProject(proj_weights);
            let energy = compute_energy_preserved(
                D, FULL_LEVELS, &detail_gelu, &handling, &low, &high, &mut rng,
            );
            let gradient = compute_gradient_flow(
                D, FULL_LEVELS, &detail_gelu, &handling, &low, &high, &mut rng,
            );
            dimension_three_results.push(ApproxSweepResult {
                selection_name: name,
                approx_mode_name: "LINEAR_PROJ",
                energy_percent: energy * 100.0,
                gradient_flow: gradient,
            });
        }
    }

    // ═══ Format output ═══

    let mut output = String::new();
    output.push_str(&format!(
        "=== Wavelet FFN Design Sweep (D={}, H={}, D4 wavelet) ===\n\n",
        D, H
    ));

    // Dimension 1 table
    output.push_str("LEVEL SELECTION SWEEP (Full variant):\n");
    output.push_str(&format!(
        "{:<15} {:>8} {:>10} {:>10} {:>9}\n",
        "Selection", "Energy%", "Grad_flow", "Eff_rank", "FLOPs(K)"
    ));
    for result in &dimension_one_results {
        output.push_str(&format!(
            "{:<15} {:>7.1}% {:>10.3} {:>7.0}/{:<3} {:>8.1}\n",
            selection_name(result.selection),
            result.energy_percent,
            result.gradient_flow,
            result.effective_rank,
            D,
            result.total_flops_k,
        ));
    }
    output.push('\n');

    // Dimension 2 table
    output.push_str("BLOCK STRUCTURE COMPARISON (top-3 level selections):\n");
    output.push_str(&format!(
        "{:<15} {:<10} {:>12} {:>10} {:>8}\n",
        "Selection", "Structure", "Cross-head%", "Params", "vs_FFN%"
    ));
    for result in &dimension_two_results {
        output.push_str(&format!(
            "{:<15} {:<10} {:>11.1}% {:>10} {:>7.2}%\n",
            result.selection_name,
            result.structure_name,
            result.cross_head_percent,
            result.params,
            result.versus_ffn_percent,
        ));
    }
    output.push('\n');

    // Dimension 3 table
    output.push_str("APPROXIMATION COEFFICIENT HANDLING (top-2 level selections):\n");
    output.push_str(&format!(
        "{:<15} {:<12} {:>8} {:>10}\n",
        "Selection", "Approx_mode", "Energy%", "Grad_flow"
    ));
    for result in &dimension_three_results {
        output.push_str(&format!(
            "{:<15} {:<12} {:>7.1}% {:>10.3}\n",
            result.selection_name,
            result.approx_mode_name,
            result.energy_percent,
            result.gradient_flow,
        ));
    }
    output.push('\n');

    // ═══ Recommendation ═══

    let best_index = ranked_indices[0];
    let best = &dimension_one_results[best_index];
    let best_name = selection_name(best.selection);
    let (best_detail, _) = gelu_mask(best.selection, FULL_LEVELS);

    let active_level_list: Vec<String> = best_detail
        .iter()
        .enumerate()
        .filter(|(_, &active)| active)
        .map(|(i, _)| i.to_string())
        .collect();

    // Determine block structure from cross-head flow
    let full_cross = dimension_two_results
        .iter()
        .find(|r| r.selection_name == best_name && r.structure_name == "Full")
        .map(|r| r.cross_head_percent)
        .unwrap_or(0.0);

    let optimal_structure = if full_cross > 5.0 {
        "Full"
    } else {
        "Per-Head"
    };
    let structure_dim = if optimal_structure == "Full" {
        format!("Full D={}", D)
    } else {
        format!("Per-Head D/H={}", HEAD_DIM)
    };
    let structure_reason = if full_cross > 5.0 {
        format!(
            "cross-head mixing is {:.1}%, indicating meaningful inter-head information flow through the wavelet basis",
            full_cross
        )
    } else {
        "minimal cross-head information flow suggests per-head independence is sufficient".to_string()
    };

    // Determine approx handling from dimension 3
    let best_approx_results: Vec<&ApproxSweepResult> = dimension_three_results
        .iter()
        .filter(|r| r.selection_name == best_name)
        .collect();

    let ideal_gradient = (D as f64).sqrt();
    let optimal_approx = if let Some(best_ar) = best_approx_results.iter().min_by(|a, b| {
        let distance_a = (a.gradient_flow - ideal_gradient).abs();
        let distance_b = (b.gradient_flow - ideal_gradient).abs();
        distance_a.partial_cmp(&distance_b).unwrap()
    }) {
        best_ar.approx_mode_name
    } else {
        "PASS"
    };

    let best_params = dimension_two_results
        .iter()
        .find(|r| r.selection_name == best_name && r.structure_name == optimal_structure)
        .map(|r| r.params)
        .unwrap_or(512);

    output.push_str("=== RECOMMENDATION ===\n");
    output.push_str(&format!("Optimal level selection: {}\n", best_name));

    let level_reason = match best.selection {
        LevelSelection::All => "full nonlinearity across all scales maximizes expressivity while maintaining stable gradient flow",
        LevelSelection::DenseOnly => "dense-zone levels (0-5) capture the primary DSQG offset structure δ=1..48 where most attention energy concentrates",
        LevelSelection::SparseOnly => "sparse-zone levels focus nonlinearity where long-range dependencies δ=96,128,384 need the most transformation",
        LevelSelection::DsqgZones => "boundary levels (5,6,7) span the transition between dense and sparse DSQG zones, maximizing scale-selective expressivity",
        LevelSelection::LowFreq => "low-frequency levels capture smooth, long-range feature interactions that benefit most from nonlinear mixing",
        LevelSelection::HighFreq => "high-frequency levels capture local embedding structure that needs nonlinear feature extraction",
        LevelSelection::Alternating => "alternating levels provide multi-scale nonlinearity while preserving linear pathways for gradient flow",
        LevelSelection::NoGelu => "pure linear wavelet transform provides baseline reference",
    };
    output.push_str(&format!("Reason: {}\n\n", level_reason));

    output.push_str(&format!(
        "Optimal block structure: {}\n",
        optimal_structure
    ));
    output.push_str(&format!("Reason: {}\n\n", structure_reason));

    output.push_str(&format!("Optimal approx handling: {}\n\n", optimal_approx));

    output.push_str("Final wavelet FFN design:\n");
    output.push_str(&format!("  DWT variant: {}\n", structure_dim));
    let approx_in_gelu = optimal_approx == "GELU";
    output.push_str(&format!(
        "  Levels with GELU: [{}]{}\n",
        active_level_list.join(", "),
        if approx_in_gelu { " + approx" } else { "" }
    ));
    output.push_str(&format!("  Approx handling: {}\n", optimal_approx));
    output.push_str(&format!(
        "  Parameter count: {} vs standard FFN {} ({:.1}%)\n",
        best_params,
        STANDARD_FFN_PARAMS,
        best_params as f64 / STANDARD_FFN_PARAMS as f64 * 100.0,
    ));

    let benefit = if best.effective_rank > D as f64 * 0.8 {
        "medium"
    } else if best.effective_rank > D as f64 * 0.5 {
        "small"
    } else {
        "small"
    };
    output.push_str(&format!(
        "  Expected benefit vs standard FFN: {} (wavelet basis provides natural multi-scale structure with {:.0}× parameter reduction)\n",
        benefit,
        STANDARD_FFN_PARAMS as f64 / best_params as f64,
    ));

    let confidence = if best.gradient_flow > ideal_gradient * 0.5
        && best.gradient_flow < ideal_gradient * 2.0
        && best.effective_rank > D as f64 * 0.3
    {
        "HIGH"
    } else if best.effective_rank > D as f64 * 0.1 {
        "MEDIUM"
    } else {
        "LOW"
    };
    output.push_str(&format!("  Confidence: {}\n", confidence));

    print!("{}", output);

    let output_path = "results.txt";
    let mut file = std::fs::File::create(output_path).expect("Failed to create results.txt");
    file.write_all(output.as_bytes())
        .expect("Failed to write results.txt");
    println!("\n✅ Results saved to {}", output_path);
}
