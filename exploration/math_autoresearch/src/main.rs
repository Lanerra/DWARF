use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};
use std::time::Instant;

const SIGNAL_LENGTH: usize = 2048;
const DWT_LEVELS: usize = 11;
const GRADIENT_SIGNAL_LENGTH: usize = 256;
const GRADIENT_DWT_LEVELS: usize = 8;
const SPARSE_K: usize = 20;
const N_GRADIENT_SIGNALS: usize = 100;
const LOCALIZATION_THRESHOLD: f64 = 0.5;
const RNG_SEED: u64 = 42;

// ── DSQG offset definitions ──

fn dsqg_dense_offsets() -> Vec<usize> {
    (1..=48).collect()
}

fn dsqg_mid_offsets() -> Vec<usize> {
    vec![96, 128]
}

fn dsqg_long_offsets() -> Vec<usize> {
    vec![384]
}

fn all_dsqg_offsets() -> Vec<usize> {
    let mut offsets = dsqg_dense_offsets();
    offsets.extend(dsqg_mid_offsets());
    offsets.extend(dsqg_long_offsets());
    offsets
}

// ── Config / Results structs ──

#[derive(Deserialize, Clone)]
struct Config {
    #[allow(dead_code)]
    search_mode: String,
    description: String,
    wavelet: WaveletConfig,
}

#[derive(Deserialize, Clone)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
    levels_with_gelu: Vec<usize>,
    #[allow(dead_code)]
    block_structure: String,
}

#[derive(Serialize)]
struct Results {
    config_description: String,
    filter_taps: usize,
    lattice_angles_deg: Vec<f64>,
    filter_coefficients: Vec<f64>,
    metrics: Metrics,
    composite_score: f64,
    vs_d4_delta: f64,
    runtime_ms: u64,
}

#[derive(Serialize)]
struct Metrics {
    sparse_approx_mse: f64,
    scale_separation: f64,
    leakage: f64,
    reconstruction_quality: f64,
    gradient_flow: f64,
    dsqg_dense_coverage: f64,
    dsqg_mid_coverage: f64,
    dsqg_long_coverage: f64,
}

// ── Wavelet filter construction ──

fn construct_filter_4tap(theta: f64) -> Vec<f64> {
    let c = theta.cos();
    let s = theta.sin();
    let denominator = 2.0 * SQRT_2;
    vec![
        (1.0 - c + s) / denominator,
        (1.0 + c + s) / denominator,
        (1.0 + c - s) / denominator,
        (1.0 - c - s) / denominator,
    ]
}

fn apply_rotation(polyphase: &mut [[[f64; 2]; 2]], theta: f64, degree: usize) {
    let c = theta.cos();
    let s = theta.sin();
    for k in 0..degree {
        let old = polyphase[k];
        polyphase[k] = [
            [
                c * old[0][0] + s * old[1][0],
                c * old[0][1] + s * old[1][1],
            ],
            [
                -s * old[0][0] + c * old[1][0],
                -s * old[0][1] + c * old[1][1],
            ],
        ];
    }
}

fn construct_filter_lattice(angles: &[f64], half_taps: usize) -> Vec<f64> {
    let m = half_taps;
    let mut polyphase = vec![[[0.0f64; 2]; 2]; m];
    let inverse_sqrt2 = 1.0 / SQRT_2;
    polyphase[0] = [
        [inverse_sqrt2, inverse_sqrt2],
        [inverse_sqrt2, -inverse_sqrt2],
    ];

    let theta_zero: f64 = -angles.iter().sum::<f64>();
    apply_rotation(&mut polyphase, theta_zero, m);

    for &angle in angles {
        for k in (1..m).rev() {
            polyphase[k][1] = polyphase[k - 1][1];
        }
        polyphase[0][1] = [0.0, 0.0];
        apply_rotation(&mut polyphase, angle, m);
    }

    let mut filter = vec![0.0; 2 * m];
    for k in 0..m {
        filter[2 * k] = polyphase[k][0][0];
        filter[2 * k + 1] = polyphase[k][1][0];
    }
    filter
}

fn construct_filter(config: &WaveletConfig) -> Vec<f64> {
    let expected_angles = config.filter_taps / 2 - 1;
    assert_eq!(
        config.lattice_angles.len(),
        expected_angles,
        "Expected {} angles for {}-tap filter, got {}",
        expected_angles,
        config.filter_taps,
        config.lattice_angles.len()
    );
    assert!(
        config.filter_taps >= 4 && config.filter_taps % 2 == 0,
        "filter_taps must be even and >= 4"
    );

    if config.filter_taps == 4 {
        construct_filter_4tap(config.lattice_angles[0])
    } else {
        construct_filter_lattice(&config.lattice_angles, config.filter_taps / 2)
    }
}

// ── QMF highpass ──

fn qmf_highpass(lowpass: &[f64]) -> Vec<f64> {
    let length = lowpass.len();
    (0..length)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * lowpass[length - 1 - k]
        })
        .collect()
}

// ── DWT forward / inverse ──

fn circular_convolve_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let length = signal.len();
    let filter_length = filter.len();
    let output_length = length / 2;
    let mut output = vec![0.0; output_length];
    for output_index in 0..output_length {
        let signal_index = 2 * output_index;
        let mut accumulator = 0.0;
        for filter_index in 0..filter_length {
            let sample_position = (signal_index + length - filter_index) % length;
            accumulator += filter[filter_index] * signal[sample_position];
        }
        output[output_index] = accumulator;
    }
    output
}

fn upsample_circular_convolve(
    coefficients: &[f64],
    filter: &[f64],
    output_length: usize,
) -> Vec<f64> {
    let filter_length = filter.len();
    let mut output = vec![0.0; output_length];
    for (coefficient_index, &coefficient) in coefficients.iter().enumerate() {
        let upsampled_position = 2 * coefficient_index;
        for filter_index in 0..filter_length {
            let output_index =
                (upsampled_position + output_length - filter_index) % output_length;
            output[output_index] += coefficient * filter[filter_index];
        }
    }
    output
}

struct DwtResult {
    approximation: Vec<f64>,
    details: Vec<Vec<f64>>,
    levels: usize,
}

fn dwt_forward(
    signal: &[f64],
    lowpass: &[f64],
    highpass: &[f64],
    levels: usize,
) -> DwtResult {
    let mut approximation = signal.to_vec();
    let mut details = Vec::with_capacity(levels);
    for _ in 0..levels {
        let detail = circular_convolve_downsample(&approximation, highpass);
        approximation = circular_convolve_downsample(&approximation, lowpass);
        details.push(detail);
    }
    DwtResult {
        approximation,
        details,
        levels,
    }
}

fn dwt_inverse(result: &DwtResult, lowpass: &[f64], highpass: &[f64]) -> Vec<f64> {
    let mut approximation = result.approximation.clone();
    for level in (0..result.levels).rev() {
        let target_length = result.details[level].len() * 2;
        let reconstructed_low =
            upsample_circular_convolve(&approximation, lowpass, target_length);
        let reconstructed_high =
            upsample_circular_convolve(&result.details[level], highpass, target_length);
        approximation = reconstructed_low
            .iter()
            .zip(reconstructed_high.iter())
            .map(|(low, high)| low + high)
            .collect();
    }
    approximation
}

fn all_coefficients_flat(result: &DwtResult) -> Vec<f64> {
    let mut coefficients = Vec::new();
    for detail in &result.details {
        coefficients.extend(detail);
    }
    coefficients.extend(&result.approximation);
    coefficients
}

fn signal_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

// ── GELU derivative (tanh approximation) ──

fn gelu_derivative(x: f64) -> f64 {
    let a = (2.0 / PI).sqrt();
    let b = 0.044715;
    let inner = a * (x + b * x * x * x);
    let tanh_value = inner.tanh();
    let sech_squared = 1.0 - tanh_value * tanh_value;
    0.5 * (1.0 + tanh_value) + 0.5 * x * a * (1.0 + 3.0 * b * x * x) * sech_squared
}

// ── Build DSQG offset signal ──

fn build_dsqg_signal(signal_length: usize) -> Vec<f64> {
    let mut signal = vec![0.0; signal_length];
    for &offset in &all_dsqg_offsets() {
        if offset < signal_length {
            signal[offset] = 1.0;
        }
    }
    signal
}

// ── Metric 1: Sparse Approximation MSE ──

fn compute_sparse_approx_mse(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let signal = build_dsqg_signal(SIGNAL_LENGTH);
    let original_energy = signal_energy(&signal);

    let dwt_result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);
    let all_coefficients = all_coefficients_flat(&dwt_result);
    let total_count = all_coefficients.len();

    let mut indexed: Vec<(usize, f64)> = all_coefficients
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut sparse_coefficients = vec![0.0; total_count];
    for &(index, _) in indexed.iter().take(SPARSE_K) {
        sparse_coefficients[index] = all_coefficients[index];
    }

    let mut offset = 0;
    let mut sparse_details = Vec::new();
    for detail in &dwt_result.details {
        let length = detail.len();
        sparse_details.push(sparse_coefficients[offset..offset + length].to_vec());
        offset += length;
    }
    let sparse_approximation = sparse_coefficients[offset..].to_vec();

    let sparse_dwt = DwtResult {
        approximation: sparse_approximation,
        details: sparse_details,
        levels: dwt_result.levels,
    };
    let reconstructed = dwt_inverse(&sparse_dwt, lowpass, highpass);

    let error_energy: f64 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    error_energy / original_energy
}

// ── Metric 2: Scale Separation ──

fn primary_level_for_offset(offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }
    let level = (offset as f64).log2().floor() as usize;
    level.min(DWT_LEVELS - 1)
}

fn compute_scale_separation(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let zones: Vec<Vec<usize>> = vec![
        dsqg_dense_offsets(),
        dsqg_mid_offsets(),
        dsqg_long_offsets(),
    ];

    let mut zone_scores = Vec::new();
    for offsets in &zones {
        let mut zone_alignment = 0.0;
        for &offset in offsets {
            let mut signal = vec![0.0; SIGNAL_LENGTH];
            signal[offset] = 1.0;
            let result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);

            let level_energies: Vec<f64> =
                result.details.iter().map(|d| signal_energy(d)).collect();
            let approx_energy = signal_energy(&result.approximation);
            let total_energy: f64 = level_energies.iter().sum::<f64>() + approx_energy;

            if total_energy < 1e-15 {
                continue;
            }

            let expected_level = primary_level_for_offset(offset);

            let mut captured_energy = 0.0;
            let low = if expected_level > 0 {
                expected_level - 1
            } else {
                0
            };
            let high = (expected_level + 1).min(DWT_LEVELS - 1);
            for level in low..=high {
                if level < level_energies.len() {
                    captured_energy += level_energies[level];
                }
            }
            if expected_level >= level_energies.len() {
                captured_energy += approx_energy;
            }

            zone_alignment += captured_energy / total_energy;
        }
        zone_scores.push(zone_alignment / offsets.len() as f64);
    }
    zone_scores.iter().sum::<f64>() / zone_scores.len() as f64
}

// ── Metric 3: Cross-Scale Leakage ──

fn compute_leakage(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let sparse_offsets = [96_usize, 128, 384];
    let mut leakages = Vec::new();

    for &offset in &sparse_offsets {
        let mut signal = vec![0.0; SIGNAL_LENGTH];
        signal[offset] = 1.0;
        let result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);

        let level_energies: Vec<f64> =
            result.details.iter().map(|d| signal_energy(d)).collect();
        let primary_level = primary_level_for_offset(offset);

        let primary_energy = level_energies
            .get(primary_level)
            .copied()
            .unwrap_or(0.0);

        let mut adjacent_energy = 0.0;
        if primary_level > 0 {
            adjacent_energy += level_energies
                .get(primary_level - 1)
                .copied()
                .unwrap_or(0.0);
        }
        adjacent_energy += level_energies
            .get(primary_level + 1)
            .copied()
            .unwrap_or(0.0);

        let leakage = if primary_energy > 1e-15 {
            adjacent_energy / primary_energy
        } else {
            f64::MAX
        };
        leakages.push(leakage);
    }
    leakages.iter().sum::<f64>() / leakages.len() as f64
}

// ── Metric 4: Reconstruction Quality ──

fn compute_reconstruction_quality(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let signal = build_dsqg_signal(SIGNAL_LENGTH);
    let original_energy = signal_energy(&signal);
    if original_energy < 1e-15 {
        return 0.0;
    }
    let result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);
    let reconstructed = dwt_inverse(&result, lowpass, highpass);
    let error_energy: f64 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    error_energy / original_energy
}

// ── Metric 5: Gradient Flow ──

fn compute_gradient_flow(
    lowpass: &[f64],
    highpass: &[f64],
    levels_with_gelu: &[usize],
    rng: &mut StdRng,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut total_frobenius_squared = 0.0;

    for _ in 0..N_GRADIENT_SIGNALS {
        let signal: Vec<f64> = (0..GRADIENT_SIGNAL_LENGTH)
            .map(|_| normal.sample(rng))
            .collect();
        let result = dwt_forward(&signal, lowpass, highpass, GRADIENT_DWT_LEVELS);

        let mut frobenius_squared = 0.0;

        for (level_index, detail) in result.details.iter().enumerate() {
            let has_gelu = levels_with_gelu.contains(&level_index);
            for &coefficient in detail {
                let derivative = if has_gelu {
                    gelu_derivative(coefficient)
                } else {
                    1.0
                };
                frobenius_squared += derivative * derivative;
            }
        }

        for _ in &result.approximation {
            frobenius_squared += 1.0;
        }

        total_frobenius_squared += frobenius_squared;
    }
    (total_frobenius_squared / N_GRADIENT_SIGNALS as f64).sqrt()
}

// ── Metric 6: DSQG Coverage ──

struct CoverageResult {
    dense: f64,
    mid: f64,
    long: f64,
}

fn compute_dsqg_coverage(lowpass: &[f64], highpass: &[f64]) -> CoverageResult {
    let compute_zone_coverage = |offsets: &[usize]| -> f64 {
        let mut well_localized = 0;
        for &offset in offsets {
            let mut signal = vec![0.0; SIGNAL_LENGTH];
            signal[offset] = 1.0;
            let result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);

            let level_energies: Vec<f64> =
                result.details.iter().map(|d| signal_energy(d)).collect();
            let approx_energy = signal_energy(&result.approximation);
            let total_energy: f64 = level_energies.iter().sum::<f64>() + approx_energy;

            if total_energy < 1e-15 {
                continue;
            }

            let max_energy = level_energies
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(approx_energy);

            if max_energy / total_energy > LOCALIZATION_THRESHOLD {
                well_localized += 1;
            }
        }
        well_localized as f64 / offsets.len() as f64
    };

    let dense_offsets = dsqg_dense_offsets();
    let mid_offsets = dsqg_mid_offsets();
    let long_offsets = dsqg_long_offsets();

    CoverageResult {
        dense: compute_zone_coverage(&dense_offsets),
        mid: compute_zone_coverage(&mid_offsets),
        long: compute_zone_coverage(&long_offsets),
    }
}

// ── Composite Score ──
// Normalization anchored so D4 baseline scores ~0.834.
// Each metric normalized such that D4's value maps to ~0.80.

fn compute_composite_score(metrics: &Metrics, d4_reference: &Metrics) -> f64 {
    let mse_reference = d4_reference.sparse_approx_mse / 0.2;
    let scale_reference = d4_reference.scale_separation / 0.85;
    let gradient_reference = d4_reference.gradient_flow / 0.80;
    let leakage_reference = d4_reference.leakage / 0.2;

    let score_mse = (1.0 - (metrics.sparse_approx_mse / mse_reference).min(1.0)).max(0.0);
    let score_scale = (metrics.scale_separation / scale_reference).clamp(0.0, 1.0);
    let score_coverage =
        (metrics.dsqg_dense_coverage + metrics.dsqg_mid_coverage + metrics.dsqg_long_coverage)
            / 3.0;
    let score_gradient = (metrics.gradient_flow / gradient_reference).min(1.0).max(0.0);
    let score_leakage = (1.0 - (metrics.leakage / leakage_reference).min(1.0)).max(0.0);

    0.30 * score_mse
        + 0.25 * score_scale
        + 0.20 * score_coverage
        + 0.15 * score_gradient
        + 0.10 * score_leakage
}

// ── Full evaluation pipeline ──

fn evaluate_wavelet(config: &WaveletConfig) -> (Vec<f64>, Metrics) {
    let lowpass = construct_filter(config);
    let highpass = qmf_highpass(&lowpass);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let sparse_approx_mse = compute_sparse_approx_mse(&lowpass, &highpass);
    let scale_separation = compute_scale_separation(&lowpass, &highpass);
    let leakage = compute_leakage(&lowpass, &highpass);
    let reconstruction_quality = compute_reconstruction_quality(&lowpass, &highpass);
    let gradient_flow =
        compute_gradient_flow(&lowpass, &highpass, &config.levels_with_gelu, &mut rng);
    let coverage = compute_dsqg_coverage(&lowpass, &highpass);

    let metrics = Metrics {
        sparse_approx_mse,
        scale_separation,
        leakage,
        reconstruction_quality,
        gradient_flow,
        dsqg_dense_coverage: coverage.dense,
        dsqg_mid_coverage: coverage.mid,
        dsqg_long_coverage: coverage.long,
    };

    (lowpass, metrics)
}

// ── Main ──

fn main() {
    let config_path = "exploration/math_autoresearch/current_config.json";
    let results_path = "exploration/math_autoresearch/current_results.json";

    let config_string =
        std::fs::read_to_string(config_path).expect("Failed to read current_config.json");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse current_config.json");

    assert_eq!(
        config.search_mode, "wavelet_filter",
        "Only wavelet_filter search_mode is supported"
    );

    let start = Instant::now();

    let d4_config = WaveletConfig {
        filter_taps: 4,
        lattice_angles: vec![PI / 3.0],
        levels_with_gelu: config.wavelet.levels_with_gelu.clone(),
        block_structure: config.wavelet.block_structure.clone(),
    };
    let (_, d4_metrics) = evaluate_wavelet(&d4_config);
    let d4_composite = compute_composite_score(&d4_metrics, &d4_metrics);

    let (filter_coefficients, metrics) = evaluate_wavelet(&config.wavelet);
    let composite_score = compute_composite_score(&metrics, &d4_metrics);

    let elapsed = start.elapsed().as_millis() as u64;

    let results = Results {
        config_description: config.description.clone(),
        filter_taps: config.wavelet.filter_taps,
        lattice_angles_deg: config
            .wavelet
            .lattice_angles
            .iter()
            .map(|a| a.to_degrees())
            .collect(),
        filter_coefficients,
        metrics,
        composite_score,
        vs_d4_delta: composite_score - d4_composite,
        runtime_ms: elapsed,
    };

    let results_json =
        serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    std::fs::write(results_path, &results_json).expect("Failed to write current_results.json");

    eprintln!(
        "score={:.4} (vs D4: {:+.4}) in {}ms",
        composite_score,
        composite_score - d4_composite,
        elapsed
    );
}
