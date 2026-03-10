use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;

const EMBEDDING_DIM: usize = 512;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = EMBEDDING_DIM / NUM_HEADS;
const FFN_DIM: usize = 2048;
const FFN_DIM_PER_HEAD: usize = FFN_DIM / NUM_HEADS;

const MIXING_SAMPLES: usize = 10_000;
const JACOBIAN_SAMPLES: usize = 1_000;
const SNR_SAMPLES: usize = 1_000;
const GRADIENT_SAMPLES: usize = 1_000;

const SIGMA_GLOBAL: f64 = 2.0;
const SIGMA_LOCAL: f64 = 0.3;

// ─── Math helpers ───

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

fn gelu(x: f64) -> f64 {
    x * standard_normal_cdf(x)
}

fn gelu_derivative(x: f64) -> f64 {
    standard_normal_cdf(x) + x * standard_normal_pdf(x)
}

fn kaiming_init(rows: usize, columns: usize, fan_in: usize, rng: &mut StdRng) -> Array2<f64> {
    let sigma = (2.0 / fan_in as f64).sqrt();
    let normal = Normal::new(0.0, sigma).unwrap();
    Array2::from_shape_fn((rows, columns), |_| normal.sample(rng))
}

fn random_unit_norm(dimension: usize, rng: &mut StdRng) -> Array1<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut vector = Array1::from_shape_fn(dimension, |_| normal.sample(rng));
    let dot_product: f64 = vector.dot(&vector);
    vector /= dot_product.sqrt();
    vector
}

fn matmul(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    matrix.dot(vector)
}

fn matrix_matmul(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    left.dot(right)
}

// ─── FFN forward passes ───

fn full_ffn_forward(
    input: &Array1<f64>,
    weight1: &Array2<f64>,
    weight2: &Array2<f64>,
) -> Array1<f64> {
    let hidden = matmul(weight1, input).mapv(gelu);
    matmul(weight2, &hidden)
}

fn block_diagonal_ffn_forward(
    input: &Array1<f64>,
    block_weights1: &[Array2<f64>],
    block_weights2: &[Array2<f64>],
) -> Array1<f64> {
    let mut output = Array1::zeros(EMBEDDING_DIM);
    for head in 0..NUM_HEADS {
        let start = head * HEAD_DIM;
        let end = start + HEAD_DIM;
        let block_input = input.slice(ndarray::s![start..end]).to_owned();
        let hidden = matmul(&block_weights1[head], &block_input).mapv(gelu);
        let block_output = matmul(&block_weights2[head], &hidden);
        output
            .slice_mut(ndarray::s![start..end])
            .assign(&block_output);
    }
    output
}

// ─── Efficient Jacobian computation ───
// J = W2 · diag(gelu'(W1 · x)) · W1
// Avoid forming the FFN_DIM × FFN_DIM diagonal matrix:
//   scaled_w1[i, j] = gelu'((W1 · x)[i]) * W1[i, j]
//   J = W2 · scaled_w1

fn full_ffn_jacobian(
    input: &Array1<f64>,
    weight1: &Array2<f64>,
    weight2: &Array2<f64>,
) -> Array2<f64> {
    let pre_activation = matmul(weight1, input);
    let gelu_deriv = pre_activation.mapv(gelu_derivative);
    let ffn_dim = weight1.nrows();
    let input_dim = weight1.ncols();
    let mut scaled_w1 = Array2::zeros((ffn_dim, input_dim));
    for row in 0..ffn_dim {
        let scale = gelu_deriv[row];
        for column in 0..input_dim {
            scaled_w1[[row, column]] = scale * weight1[[row, column]];
        }
    }
    matrix_matmul(weight2, &scaled_w1)
}

// For gradient flow: compute J^T g without forming J explicitly
// J^T g = W1^T · diag(gelu'(W1 · x)) · W2^T · g
fn full_ffn_jacobian_transpose_vector(
    input: &Array1<f64>,
    weight1: &Array2<f64>,
    weight2: &Array2<f64>,
    gradient: &Array1<f64>,
) -> Array1<f64> {
    let pre_activation = matmul(weight1, input);
    let gelu_deriv = pre_activation.mapv(gelu_derivative);
    let w2t_g = matmul(&weight2.t().to_owned(), gradient);
    let scaled = &w2t_g * &gelu_deriv;
    matmul(&weight1.t().to_owned(), &scaled)
}

fn block_ffn_jacobian_transpose_vector(
    input: &Array1<f64>,
    block_weights1: &[Array2<f64>],
    block_weights2: &[Array2<f64>],
    gradient: &Array1<f64>,
) -> Array1<f64> {
    let mut result = Array1::zeros(EMBEDDING_DIM);
    for head in 0..NUM_HEADS {
        let start = head * HEAD_DIM;
        let end = start + HEAD_DIM;
        let block_input = input.slice(ndarray::s![start..end]).to_owned();
        let block_gradient = gradient.slice(ndarray::s![start..end]).to_owned();
        let block_result = full_ffn_jacobian_transpose_vector(
            &block_input,
            &block_weights1[head],
            &block_weights2[head],
            &block_gradient,
        );
        result.slice_mut(ndarray::s![start..end]).assign(&block_result);
    }
    result
}

// ─── Spectral statistics via power iteration ───

struct SpectralStats {
    spectral_radius: f64,
    condition_number: f64,
    effective_rank: f64,
}

fn power_iteration_largest_eigenvalue(matrix: &Array2<f64>, iterations: usize, rng: &mut StdRng) -> f64 {
    let size = matrix.nrows();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut vector = Array1::from_shape_fn(size, |_| normal.sample(rng));
    let dot_product: f64 = vector.dot(&vector);
    vector /= dot_product.sqrt();

    let mut eigenvalue = 0.0;
    for _ in 0..iterations {
        let next = matmul(matrix, &vector);
        eigenvalue = next.dot(&vector);
        let norm: f64 = next.dot(&next);
        vector = next / norm.sqrt();
    }
    eigenvalue
}

struct LuFactorization {
    lu: Array2<f64>,
    pivot_indices: Vec<usize>,
    singular: bool,
}

fn lu_factorize(matrix: &Array2<f64>) -> LuFactorization {
    let size = matrix.nrows();
    let mut lu = matrix.clone();
    let mut pivot_indices: Vec<usize> = (0..size).collect();

    for pivot in 0..size {
        let mut max_row = pivot;
        let mut max_value = lu[[pivot, pivot]].abs();
        for row in (pivot + 1)..size {
            if lu[[row, pivot]].abs() > max_value {
                max_value = lu[[row, pivot]].abs();
                max_row = row;
            }
        }
        if max_value < 1e-14 {
            return LuFactorization { lu, pivot_indices, singular: true };
        }
        if max_row != pivot {
            pivot_indices.swap(pivot, max_row);
            for column in 0..size {
                let temporary = lu[[pivot, column]];
                lu[[pivot, column]] = lu[[max_row, column]];
                lu[[max_row, column]] = temporary;
            }
        }
        let pivot_value = lu[[pivot, pivot]];
        for row in (pivot + 1)..size {
            let factor = lu[[row, pivot]] / pivot_value;
            lu[[row, pivot]] = factor;
            for column in (pivot + 1)..size {
                lu[[row, column]] -= factor * lu[[pivot, column]];
            }
        }
    }
    LuFactorization { lu, pivot_indices, singular: false }
}

fn lu_solve(factorization: &LuFactorization, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    if factorization.singular {
        return None;
    }
    let size = factorization.lu.nrows();
    let mut permuted_rhs = Array1::zeros(size);
    for row in 0..size {
        permuted_rhs[row] = rhs[factorization.pivot_indices[row]];
    }

    // Forward substitution (L · y = permuted_rhs)
    let mut y = permuted_rhs;
    for row in 1..size {
        for column in 0..row {
            y[row] -= factorization.lu[[row, column]] * y[column];
        }
    }

    // Back substitution (U · x = y)
    let mut solution = y;
    for row in (0..size).rev() {
        for column in (row + 1)..size {
            solution[row] -= factorization.lu[[row, column]] * solution[column];
        }
        solution[row] /= factorization.lu[[row, row]];
    }
    Some(solution)
}

fn inverse_power_iteration_smallest_eigenvalue(
    matrix: &Array2<f64>,
    iterations: usize,
    rng: &mut StdRng,
) -> f64 {
    let size = matrix.nrows();
    let factorization = lu_factorize(matrix);
    if factorization.singular {
        return 0.0;
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut vector = Array1::from_shape_fn(size, |_| normal.sample(rng));
    let dot_product: f64 = vector.dot(&vector);
    vector /= dot_product.sqrt();

    let mut eigenvalue = 1.0;
    for _ in 0..iterations {
        let next = match lu_solve(&factorization, &vector) {
            Some(solution) => solution,
            None => return 0.0,
        };
        let norm: f64 = next.dot(&next);
        eigenvalue = vector.dot(&next) / norm;
        vector = &next / norm.sqrt();
    }
    eigenvalue.abs()
}

fn compute_spectral_stats_for_jacobian(
    jacobian: &Array2<f64>,
    rng: &mut StdRng,
) -> SpectralStats {
    let jtj = matrix_matmul(&jacobian.t().to_owned(), jacobian);

    let frobenius_squared: f64 = jacobian.iter().map(|x| x * x).sum();

    let largest_eigenvalue = power_iteration_largest_eigenvalue(&jtj, 60, rng);
    let sigma_max = largest_eigenvalue.max(0.0).sqrt();

    let smallest_eigenvalue = inverse_power_iteration_smallest_eigenvalue(&jtj, 60, rng);
    let sigma_min = smallest_eigenvalue.max(0.0).sqrt().max(1e-15);

    let condition_number = sigma_max / sigma_min;
    let effective_rank = frobenius_squared / largest_eigenvalue.max(1e-30);

    SpectralStats {
        spectral_radius: sigma_max,
        condition_number,
        effective_rank,
    }
}

fn compute_spectral_stats_block_diagonal(
    input: &Array1<f64>,
    block_weights1: &[Array2<f64>],
    block_weights2: &[Array2<f64>],
    rng: &mut StdRng,
) -> SpectralStats {
    let mut max_sigma = 0.0f64;
    let mut min_sigma = f64::MAX;
    let mut total_effective_rank = 0.0;

    for head in 0..NUM_HEADS {
        let start = head * HEAD_DIM;
        let end = start + HEAD_DIM;
        let block_input = input.slice(ndarray::s![start..end]).to_owned();
        let block_jacobian =
            full_ffn_jacobian(&block_input, &block_weights1[head], &block_weights2[head]);
        let stats = compute_spectral_stats_for_jacobian(&block_jacobian, rng);

        max_sigma = max_sigma.max(stats.spectral_radius);
        if stats.spectral_radius > 1e-10 {
            let block_sigma_min = stats.spectral_radius / stats.condition_number;
            min_sigma = min_sigma.min(block_sigma_min);
        }
        total_effective_rank += stats.effective_rank;
    }

    if min_sigma == f64::MAX {
        min_sigma = 1e-15;
    }

    SpectralStats {
        spectral_radius: max_sigma,
        condition_number: max_sigma / min_sigma.max(1e-15),
        effective_rank: total_effective_rank,
    }
}

// ─── Analysis 1: Parameter count ───

fn analysis_parameter_count() -> String {
    let full_parameters = 2 * EMBEDDING_DIM * FFN_DIM;
    let block_parameters = NUM_HEADS * 2 * HEAD_DIM * FFN_DIM_PER_HEAD;
    let ratio = full_parameters / block_parameters;

    format!(
        "1. PARAMETER COUNT\n\
         \x20  Algebraic: H × 2 × (D/H) × (FFN/H) = 2·D·FFN/H (NOT 2·D·FFN)\n\
         \x20  Full FFN:        {:>12} (2 × D × FFN_DIM)\n\
         \x20  Block-diagonal:  {:>12} (H × 2 × D/H × FFN/H = 2·D·FFN/H)\n\
         \x20  Block-diagonal uses {}× fewer params — zero cross-block connectivity\n",
        format_number(full_parameters),
        format_number(block_parameters),
        ratio,
    )
}

fn format_number(number: usize) -> String {
    let string = number.to_string();
    let mut result = String::new();
    for (index, character) in string.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            result.push(',');
        }
        result.push(character);
    }
    result.chars().rev().collect()
}

// ─── Analysis 2: Mixing capacity ───

struct MixingResult {
    text: String,
    cross_block_fraction: f64,
}

fn analysis_mixing_capacity(rng: &mut StdRng) -> MixingResult {
    let weight1_full = kaiming_init(FFN_DIM, EMBEDDING_DIM, EMBEDDING_DIM, rng);
    let weight2_full = kaiming_init(EMBEDDING_DIM, FFN_DIM, FFN_DIM, rng);

    let mut block_weights1 = Vec::with_capacity(NUM_HEADS);
    let mut block_weights2 = Vec::with_capacity(NUM_HEADS);
    for _ in 0..NUM_HEADS {
        block_weights1.push(kaiming_init(FFN_DIM_PER_HEAD, HEAD_DIM, HEAD_DIM, rng));
        block_weights2.push(kaiming_init(HEAD_DIM, FFN_DIM_PER_HEAD, FFN_DIM_PER_HEAD, rng));
    }

    let mut outputs_full = Array2::zeros((MIXING_SAMPLES, EMBEDDING_DIM));
    let mut outputs_block = Array2::zeros((MIXING_SAMPLES, EMBEDDING_DIM));

    for sample in 0..MIXING_SAMPLES {
        let input = random_unit_norm(EMBEDDING_DIM, rng);
        let output_full = full_ffn_forward(&input, &weight1_full, &weight2_full);
        let output_block =
            block_diagonal_ffn_forward(&input, &block_weights1, &block_weights2);
        outputs_full.row_mut(sample).assign(&output_full);
        outputs_block.row_mut(sample).assign(&output_block);
    }

    let mean_full = outputs_full.mean_axis(Axis(0)).unwrap();
    let mean_block = outputs_block.mean_axis(Axis(0)).unwrap();

    let mut covariance_full = Array2::zeros((EMBEDDING_DIM, EMBEDDING_DIM));
    let mut covariance_block = Array2::zeros((EMBEDDING_DIM, EMBEDDING_DIM));

    for sample in 0..MIXING_SAMPLES {
        let centered_full = &outputs_full.row(sample) - &mean_full;
        let centered_block = &outputs_block.row(sample) - &mean_block;
        for row in 0..EMBEDDING_DIM {
            for column in 0..EMBEDDING_DIM {
                covariance_full[[row, column]] += centered_full[row] * centered_full[column];
                covariance_block[[row, column]] +=
                    centered_block[row] * centered_block[column];
            }
        }
    }
    covariance_full /= MIXING_SAMPLES as f64;
    covariance_block /= MIXING_SAMPLES as f64;

    let mut cross_block_variance_full = 0.0;
    let mut cross_block_variance_block = 0.0;
    for row in 0..EMBEDDING_DIM {
        for column in 0..EMBEDDING_DIM {
            let head_row = row / HEAD_DIM;
            let head_column = column / HEAD_DIM;
            if head_row != head_column {
                cross_block_variance_full += covariance_full[[row, column]].powi(2);
                cross_block_variance_block += covariance_block[[row, column]].powi(2);
            }
        }
    }

    let total_frobenius_full: f64 = covariance_full.iter().map(|x| x.powi(2)).sum();
    let total_frobenius_block: f64 = covariance_block.iter().map(|x| x.powi(2)).sum();

    let cross_block_fraction_full = cross_block_variance_full / total_frobenius_full * 100.0;
    let cross_block_fraction_block = cross_block_variance_block / total_frobenius_block * 100.0;

    MixingResult {
        text: format!(
            "2. MIXING CAPACITY (N={} random inputs)\n\
             \x20  Cross-block output variance in full FFN:   {:.1}%\n\
             \x20  Cross-block output variance in block-diag: {:.1}% (exact)\n\
             \x20  Information lost from blocking:             {:.1}% cross-block signal eliminated\n",
            MIXING_SAMPLES,
            cross_block_fraction_full,
            cross_block_fraction_block,
            cross_block_fraction_full,
        ),
        cross_block_fraction: cross_block_fraction_full,
    }
}

// ─── Analysis 3: Jacobian spectral analysis ───

struct AggregatedSpectralStats {
    mean_spectral_radius: f64,
    mean_condition_number: f64,
    mean_effective_rank: f64,
}

fn analysis_jacobian_spectral(
    rng: &mut StdRng,
) -> (String, AggregatedSpectralStats, AggregatedSpectralStats) {
    let weight1_full = kaiming_init(FFN_DIM, EMBEDDING_DIM, EMBEDDING_DIM, rng);
    let weight2_full = kaiming_init(EMBEDDING_DIM, FFN_DIM, FFN_DIM, rng);

    let mut block_weights1 = Vec::with_capacity(NUM_HEADS);
    let mut block_weights2 = Vec::with_capacity(NUM_HEADS);
    for _ in 0..NUM_HEADS {
        block_weights1.push(kaiming_init(FFN_DIM_PER_HEAD, HEAD_DIM, HEAD_DIM, rng));
        block_weights2.push(kaiming_init(HEAD_DIM, FFN_DIM_PER_HEAD, FFN_DIM_PER_HEAD, rng));
    }

    let mut spectral_radius_full = 0.0;
    let mut condition_number_full = 0.0;
    let mut effective_rank_full = 0.0;
    let mut spectral_radius_block = 0.0;
    let mut condition_number_block = 0.0;
    let mut effective_rank_block = 0.0;

    for sample in 0..JACOBIAN_SAMPLES {
        if sample % 100 == 0 {
            eprintln!("  sample {}/{}", sample, JACOBIAN_SAMPLES);
        }
        let input = random_unit_norm(EMBEDDING_DIM, rng);

        let jacobian_full = full_ffn_jacobian(&input, &weight1_full, &weight2_full);
        let stats_full = compute_spectral_stats_for_jacobian(&jacobian_full, rng);
        spectral_radius_full += stats_full.spectral_radius;
        condition_number_full += stats_full.condition_number;
        effective_rank_full += stats_full.effective_rank;

        let stats_block = compute_spectral_stats_block_diagonal(
            &input,
            &block_weights1,
            &block_weights2,
            rng,
        );
        spectral_radius_block += stats_block.spectral_radius;
        condition_number_block += stats_block.condition_number;
        effective_rank_block += stats_block.effective_rank;
    }

    let count = JACOBIAN_SAMPLES as f64;
    let aggregated_full = AggregatedSpectralStats {
        mean_spectral_radius: spectral_radius_full / count,
        mean_condition_number: condition_number_full / count,
        mean_effective_rank: effective_rank_full / count,
    };
    let aggregated_block = AggregatedSpectralStats {
        mean_spectral_radius: spectral_radius_block / count,
        mean_condition_number: condition_number_block / count,
        mean_effective_rank: effective_rank_block / count,
    };

    let condition_delta =
        aggregated_block.mean_condition_number / aggregated_full.mean_condition_number;
    let condition_label = if condition_delta < 1.0 {
        "better"
    } else {
        "worse"
    };

    let text = format!(
        "3. JACOBIAN SPECTRAL ANALYSIS (N={} inputs)\n\
         \x20  Architecture      Spectral Radius   Condition Number   Effective Rank\n\
         \x20  Full FFN          {:<17.3} {:<18.3} {:.0} / {}\n\
         \x20  Block-diagonal    {:<17.3} {:<18.3} {:.0} / {}\n\
         \x20  Δ condition number: {} by {:.2}x\n",
        JACOBIAN_SAMPLES,
        aggregated_full.mean_spectral_radius,
        aggregated_full.mean_condition_number,
        aggregated_full.mean_effective_rank,
        EMBEDDING_DIM,
        aggregated_block.mean_spectral_radius,
        aggregated_block.mean_condition_number,
        aggregated_block.mean_effective_rank,
        EMBEDDING_DIM,
        condition_label,
        if condition_delta < 1.0 {
            1.0 / condition_delta
        } else {
            condition_delta
        },
    );

    (text, aggregated_full, aggregated_block)
}

// ─── Analysis 4: Head specialization SNR ───

fn generate_specialized_input(rng: &mut StdRng) -> Array1<f64> {
    let mut input = Array1::zeros(EMBEDDING_DIM);
    for head in 0..NUM_HEADS {
        let start = head * HEAD_DIM;
        let sigma = match head {
            0 | 1 => SIGMA_GLOBAL,
            6 | 7 => SIGMA_LOCAL,
            _ => 1.0,
        };
        let normal = Normal::new(0.0, sigma).unwrap();
        for index in start..(start + HEAD_DIM) {
            input[index] = normal.sample(rng);
        }
    }
    input
}

fn multivariate_r_squared(
    predictor: &Array2<f64>,
    response: &Array2<f64>,
) -> f64 {
    let number_of_samples = predictor.nrows();
    let predictor_dim = predictor.ncols();
    let response_dim = response.ncols();

    let mean_predictor = predictor.mean_axis(Axis(0)).unwrap();
    let mean_response = response.mean_axis(Axis(0)).unwrap();

    let mut centered_predictor = predictor.clone();
    let mut centered_response = response.clone();
    for sample in 0..number_of_samples {
        for dimension in 0..predictor_dim {
            centered_predictor[[sample, dimension]] -= mean_predictor[dimension];
        }
        for dimension in 0..response_dim {
            centered_response[[sample, dimension]] -= mean_response[dimension];
        }
    }

    // OLS: β = (X^T X)^{-1} X^T Y
    let xtx = matrix_matmul(&centered_predictor.t().to_owned(), &centered_predictor);
    let xty = matrix_matmul(&centered_predictor.t().to_owned(), &centered_response);
    let factorization = lu_factorize(&xtx);
    if factorization.singular {
        return 0.0;
    }

    let mut beta = Array2::zeros((predictor_dim, response_dim));
    for column in 0..response_dim {
        let rhs = xty.column(column).to_owned();
        if let Some(solution) = lu_solve(&factorization, &rhs) {
            beta.column_mut(column).assign(&solution);
        }
    }

    let predicted = matrix_matmul(&centered_predictor, &beta);
    let residual = &centered_response - &predicted;

    let total_sum_of_squares: f64 = centered_response.iter().map(|x| x * x).sum();
    let residual_sum_of_squares: f64 = residual.iter().map(|x| x * x).sum();

    if total_sum_of_squares < 1e-30 {
        return 0.0;
    }
    1.0 - residual_sum_of_squares / total_sum_of_squares
}

fn analysis_head_specialization_snr(rng: &mut StdRng) -> String {
    let weight1_full = kaiming_init(FFN_DIM, EMBEDDING_DIM, EMBEDDING_DIM, rng);
    let weight2_full = kaiming_init(EMBEDDING_DIM, FFN_DIM, FFN_DIM, rng);

    let global_dim = 2 * HEAD_DIM;
    let local_dim = 2 * HEAD_DIM;
    let local_start = 6 * HEAD_DIM;
    let cross_block_dim = EMBEDDING_DIM - global_dim;

    let mut input_global = Array2::zeros((SNR_SAMPLES, global_dim));
    let mut input_cross = Array2::zeros((SNR_SAMPLES, cross_block_dim));
    let mut output_local = Array2::zeros((SNR_SAMPLES, local_dim));
    let mut output_global = Array2::zeros((SNR_SAMPLES, global_dim));

    for sample in 0..SNR_SAMPLES {
        let input = generate_specialized_input(rng);
        let output = full_ffn_forward(&input, &weight1_full, &weight2_full);
        for dimension in 0..global_dim {
            input_global[[sample, dimension]] = input[dimension];
            output_global[[sample, dimension]] = output[dimension];
        }
        for dimension in 0..cross_block_dim {
            input_cross[[sample, dimension]] = input[global_dim + dimension];
        }
        for dimension in 0..local_dim {
            output_local[[sample, dimension]] = output[local_start + dimension];
        }
    }

    // Leakage: R² of predicting local-head outputs from global-head inputs
    let leakage_r_squared = multivariate_r_squared(&input_global, &output_local);
    let leakage_percent = leakage_r_squared * 100.0;

    let epsilon = f64::EPSILON;
    let snr_improvement_decibels =
        10.0 * (leakage_r_squared.max(epsilon) / epsilon).log10();

    // Dilution: R² of predicting global-head outputs from cross-block inputs
    let dilution_r_squared = multivariate_r_squared(&input_cross, &output_global);
    let dilution_percent = dilution_r_squared * 100.0;

    format!(
        "4. HEAD SPECIALIZATION SNR (N={} specialized inputs)\n\
         \x20  Global→local leakage (full FFN): {:.2}%\n\
         \x20  Global→local leakage (block-diag): 0.00%\n\
         \x20  SNR improvement from blocking: {:.1} dB\n\
         \x20  Global head output dilution (full FFN): {:.2}%\n",
        SNR_SAMPLES, leakage_percent, snr_improvement_decibels, dilution_percent,
    )
}

// ─── Analysis 5: Gradient flow ───

struct GradientFlowResult {
    text: String,
    ratio: f64,
}

fn analysis_gradient_flow(rng: &mut StdRng) -> GradientFlowResult {
    let weight1_full = kaiming_init(FFN_DIM, EMBEDDING_DIM, EMBEDDING_DIM, rng);
    let weight2_full = kaiming_init(EMBEDDING_DIM, FFN_DIM, FFN_DIM, rng);

    let mut block_weights1 = Vec::with_capacity(NUM_HEADS);
    let mut block_weights2 = Vec::with_capacity(NUM_HEADS);
    for _ in 0..NUM_HEADS {
        block_weights1.push(kaiming_init(FFN_DIM_PER_HEAD, HEAD_DIM, HEAD_DIM, rng));
        block_weights2.push(kaiming_init(HEAD_DIM, FFN_DIM_PER_HEAD, FFN_DIM_PER_HEAD, rng));
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut gradient_norm_squared_full = 0.0;
    let mut gradient_norm_squared_block = 0.0;

    for _ in 0..GRADIENT_SAMPLES {
        let input = random_unit_norm(EMBEDDING_DIM, rng);
        let upstream_gradient =
            Array1::from_shape_fn(EMBEDDING_DIM, |_| normal.sample(rng));

        let input_gradient_full = full_ffn_jacobian_transpose_vector(
            &input,
            &weight1_full,
            &weight2_full,
            &upstream_gradient,
        );
        let dot_full: f64 = input_gradient_full.dot(&input_gradient_full);
        gradient_norm_squared_full += dot_full;

        let input_gradient_block = block_ffn_jacobian_transpose_vector(
            &input,
            &block_weights1,
            &block_weights2,
            &upstream_gradient,
        );
        let dot_block: f64 = input_gradient_block.dot(&input_gradient_block);
        gradient_norm_squared_block += dot_block;
    }

    let mean_gradient_full = gradient_norm_squared_full / GRADIENT_SAMPLES as f64;
    let mean_gradient_block = gradient_norm_squared_block / GRADIENT_SAMPLES as f64;
    let ratio = mean_gradient_block / mean_gradient_full;

    let assessment = if ratio > 1.05 {
        "IMPROVES"
    } else if ratio < 0.95 {
        "HURTS"
    } else {
        "NEUTRAL on"
    };

    GradientFlowResult {
        text: format!(
            "5. GRADIENT FLOW (N={} random pairs)\n\
             \x20  E[||grad||²] full FFN:        {:.3}\n\
             \x20  E[||grad||²] block-diagonal:  {:.3}\n\
             \x20  Ratio (block/full):           {:.3}\n\
             \x20  Assessment: blocking {} gradient flow\n",
            GRADIENT_SAMPLES, mean_gradient_full, mean_gradient_block, ratio, assessment,
        ),
        ratio,
    }
}

// ─── Prediction synthesis ───

fn generate_prediction(
    cross_block_fraction: f64,
    statistics_full: &AggregatedSpectralStats,
    statistics_block: &AggregatedSpectralStats,
    gradient_ratio: f64,
) -> String {
    let condition_improvement =
        statistics_full.mean_condition_number / statistics_block.mean_condition_number;
    let rank_loss = statistics_full.mean_effective_rank - statistics_block.mean_effective_rank;

    let mut score = 0.0f64;

    if cross_block_fraction > 50.0 {
        score -= 2.0;
    } else if cross_block_fraction > 30.0 {
        score -= 1.0;
    } else {
        score -= 0.5;
    }

    if condition_improvement > 1.5 {
        score += 1.5;
    } else if condition_improvement > 1.1 {
        score += 0.5;
    }

    if gradient_ratio > 1.05 {
        score += 1.0;
    } else if gradient_ratio < 0.95 {
        score -= 0.5;
    }

    if rank_loss > 50.0 {
        score -= 1.0;
    }

    score += 1.0;

    let primary_hurt = format!(
        "Cross-block mixing ({:.0}% of covariance) carries substantial information that blocking discards.",
        cross_block_fraction,
    );

    let (prediction, confidence, primary, secondary, recommended) = if score > 1.0 {
        (
            "HELP",
            "MEDIUM",
            "Head specialization preservation outweighs mixing capacity loss in DWARF's specialized-head architecture.".to_string(),
            "Reduced condition number improves trainability despite lower effective rank.",
            "YES run experiment — blocking aligns with DWARF's head-specialized design.",
        )
    } else if score < -0.5 {
        (
            "HURT",
            "MEDIUM",
            primary_hurt,
            "Head specialization benefit is insufficient to offset mixing capacity and rank loss.",
            "NO skip — full FFN's cross-block mixing is too valuable to discard.",
        )
    } else {
        (
            "NEUTRAL",
            "LOW",
            "Competing effects roughly cancel: specialization gains vs mixing capacity losses."
                .to_string(),
            "Result is sensitive to learned weight structure; random-init analysis may not predict trained behavior.",
            "CONDITIONAL on a short ablation (1-2 epochs) to measure empirical PPL delta.",
        )
    };

    format!(
        "=== PREDICTION ===\n\
         Block-diagonal FFN is predicted to {} PPL.\n\
         Confidence: {}\n\
         Primary reason: {}\n\
         Secondary effect: {}\n\
         Recommended: {}\n",
        prediction, confidence, primary, secondary, recommended,
    )
}

// ─── Main ───

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let mut output = String::new();
    output.push_str("=== Block-Diagonal FFN Analysis (D=512, H=8, FFN_DIM=2048) ===\n\n");

    eprintln!("⏳ Running analysis 1/5: parameter count...");
    let section1 = analysis_parameter_count();
    output.push_str(&section1);
    output.push('\n');

    eprintln!(
        "⏳ Running analysis 2/5: mixing capacity (N={})...",
        MIXING_SAMPLES
    );
    let mixing_result = analysis_mixing_capacity(&mut rng);
    output.push_str(&mixing_result.text);
    output.push('\n');

    eprintln!(
        "⏳ Running analysis 3/5: Jacobian spectral analysis (N={})...",
        JACOBIAN_SAMPLES
    );
    let (section3, statistics_full, statistics_block) = analysis_jacobian_spectral(&mut rng);
    output.push_str(&section3);
    output.push('\n');

    eprintln!(
        "⏳ Running analysis 4/5: head specialization SNR (N={})...",
        SNR_SAMPLES
    );
    let section4 = analysis_head_specialization_snr(&mut rng);
    output.push_str(&section4);
    output.push('\n');

    eprintln!(
        "⏳ Running analysis 5/5: gradient flow (N={})...",
        GRADIENT_SAMPLES
    );
    let gradient_result = analysis_gradient_flow(&mut rng);
    output.push_str(&gradient_result.text);
    output.push('\n');

    let prediction = generate_prediction(
        mixing_result.cross_block_fraction,
        &statistics_full,
        &statistics_block,
        gradient_result.ratio,
    );
    output.push_str(&prediction);

    print!("{}", output);

    let results_path = "results.txt";
    let mut file = fs::File::create(results_path).expect("Failed to create results.txt");
    file.write_all(output.as_bytes())
        .expect("Failed to write results.txt");
    eprintln!("\n✅ Results written to {}", results_path);
}
