use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

const SIGNAL_LENGTH: usize = 512;
const DWT_LEVELS: usize = 9;
const SPARSE_TOP_K: usize = 12;
const DFT_RESOLUTION: usize = 256;

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

// ── Config structs ──

#[derive(Deserialize)]
struct Config {
    wavelet: WaveletConfig,
}

#[derive(Deserialize)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
}

// ── Output struct ──

#[derive(Serialize)]
struct Output {
    per_offset_reconstruction_error: f64,
    dense_coverage_uniformity: f64,
    coverage_gap_pattern: f64,
    harmonic_alignment_score: f64,
    per_offset_errors: Vec<f64>,
}

// ── Wavelet filter construction (matches math_autoresearch) ──

fn construct_filter_4tap(theta: f64) -> Vec<f64> {
    let cosine = theta.cos();
    let sine = theta.sin();
    let denominator = 2.0 * SQRT_2;
    vec![
        (1.0 - cosine + sine) / denominator,
        (1.0 + cosine + sine) / denominator,
        (1.0 + cosine - sine) / denominator,
        (1.0 - cosine - sine) / denominator,
    ]
}

fn apply_rotation(polyphase: &mut [[[f64; 2]; 2]], theta: f64, degree: usize) {
    let cosine = theta.cos();
    let sine = theta.sin();
    for k in 0..degree {
        let old = polyphase[k];
        polyphase[k] = [
            [
                cosine * old[0][0] + sine * old[1][0],
                cosine * old[0][1] + sine * old[1][1],
            ],
            [
                -sine * old[0][0] + cosine * old[1][0],
                -sine * old[0][1] + cosine * old[1][1],
            ],
        ];
    }
}

fn construct_filter_lattice(angles: &[f64], half_taps: usize) -> Vec<f64> {
    let stage_count = half_taps;
    let mut polyphase = vec![[[0.0f64; 2]; 2]; stage_count];
    let inverse_sqrt2 = 1.0 / SQRT_2;
    polyphase[0] = [
        [inverse_sqrt2, inverse_sqrt2],
        [inverse_sqrt2, -inverse_sqrt2],
    ];

    let theta_zero: f64 = -angles.iter().sum::<f64>();
    apply_rotation(&mut polyphase, theta_zero, stage_count);

    for &angle in angles {
        for k in (1..stage_count).rev() {
            polyphase[k][1] = polyphase[k - 1][1];
        }
        polyphase[0][1] = [0.0, 0.0];
        apply_rotation(&mut polyphase, angle, stage_count);
    }

    let mut filter = vec![0.0; 2 * stage_count];
    for k in 0..stage_count {
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

// ── DWT forward / inverse (matches math_autoresearch) ──

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

// ── Sparse approximation per offset ──

fn sparse_reconstruct_error(
    signal: &[f64],
    lowpass: &[f64],
    highpass: &[f64],
) -> f64 {
    let dwt_result = dwt_forward(signal, lowpass, highpass, DWT_LEVELS);
    let all_coefficients = all_coefficients_flat(&dwt_result);
    let total_count = all_coefficients.len();

    let mut indexed: Vec<(usize, f64)> = all_coefficients
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut sparse_coefficients = vec![0.0; total_count];
    for &(index, _) in indexed.iter().take(SPARSE_TOP_K) {
        sparse_coefficients[index] = all_coefficients[index];
    }

    let mut coefficient_offset = 0;
    let mut sparse_details = Vec::new();
    for detail in &dwt_result.details {
        let length = detail.len();
        sparse_details.push(
            sparse_coefficients[coefficient_offset..coefficient_offset + length].to_vec(),
        );
        coefficient_offset += length;
    }
    let sparse_approximation = sparse_coefficients[coefficient_offset..].to_vec();

    let sparse_dwt = DwtResult {
        approximation: sparse_approximation,
        details: sparse_details,
        levels: dwt_result.levels,
    };
    let reconstructed = dwt_inverse(&sparse_dwt, lowpass, highpass);

    signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(&original, &recovered)| (original - recovered).powi(2))
        .sum::<f64>()
}

// ── Metric 1: Per-offset reconstruction errors ──

fn compute_per_offset_errors(
    offsets: &[usize],
    lowpass: &[f64],
    highpass: &[f64],
) -> Vec<f64> {
    offsets
        .iter()
        .map(|&delta| {
            let mut signal = vec![0.0; SIGNAL_LENGTH];
            assert!(
                delta < SIGNAL_LENGTH,
                "Offset {} exceeds signal length {}",
                delta,
                SIGNAL_LENGTH
            );
            signal[delta] = 1.0;
            sparse_reconstruct_error(&signal, lowpass, highpass)
        })
        .collect()
}

// ── Metric 2: Dense coverage uniformity (coefficient of variation) ──

fn compute_dense_coverage_uniformity(errors: &[f64]) -> f64 {
    let dense_count = 48;
    assert!(errors.len() >= dense_count);
    let dense_errors = &errors[..dense_count];

    let mean = dense_errors.iter().sum::<f64>() / dense_count as f64;
    if mean < 1e-15 {
        return 0.0;
    }

    let variance = dense_errors
        .iter()
        .map(|&error| (error - mean).powi(2))
        .sum::<f64>()
        / dense_count as f64;
    variance.sqrt() / mean
}

// ── Metric 3: Coverage gap pattern (max gap run / 48) ──

fn compute_coverage_gap_pattern(errors: &[f64]) -> f64 {
    let dense_count = 48;
    assert!(errors.len() >= dense_count);
    let dense_errors = &errors[..dense_count];

    let mut sorted = dense_errors.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = (sorted[dense_count / 2 - 1] + sorted[dense_count / 2]) / 2.0;

    let mut max_gap_run: usize = 0;
    let mut current_run: usize = 0;

    for &error in dense_errors {
        if error >= median {
            current_run += 1;
            max_gap_run = max_gap_run.max(current_run);
        } else {
            current_run = 0;
        }
    }

    max_gap_run as f64 / dense_count as f64
}

// ── Metric 4: Harmonic alignment score ──

fn compute_harmonic_alignment_score(lowpass: &[f64], offsets: &[usize]) -> f64 {
    let highpass = qmf_highpass(lowpass);

    let mut max_magnitude = 0.0_f64;
    let mut dominant_bin = 0_usize;

    for bin in 1..DFT_RESOLUTION / 2 {
        let omega = 2.0 * PI * bin as f64 / DFT_RESOLUTION as f64;
        let mut real_part = 0.0;
        let mut imaginary_part = 0.0;
        for (tap_index, &coefficient) in highpass.iter().enumerate() {
            real_part += coefficient * (omega * tap_index as f64).cos();
            imaginary_part -= coefficient * (omega * tap_index as f64).sin();
        }
        let magnitude = (real_part * real_part + imaginary_part * imaginary_part).sqrt();
        if magnitude > max_magnitude {
            max_magnitude = magnitude;
            dominant_bin = bin;
        }
    }

    let dominant_frequency = dominant_bin as f64 / DFT_RESOLUTION as f64;

    let mut total_alignment = 0.0;
    for &delta in offsets {
        let product = dominant_frequency * delta as f64;
        let fractional = product - product.floor();
        let distance_to_integer = fractional.min(1.0 - fractional);
        let alignment = 1.0 - 2.0 * distance_to_integer;
        total_alignment += alignment;
    }

    total_alignment / offsets.len() as f64
}

// ── Main ──

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        3,
        "Usage: {} <input.json> <output.json>",
        arguments[0]
    );

    let input_path = &arguments[1];
    let output_path = &arguments[2];

    let input_text = std::fs::read_to_string(input_path)
        .unwrap_or_else(|error| panic!("Failed to read {}: {}", input_path, error));
    let config: Config = serde_json::from_str(&input_text)
        .unwrap_or_else(|error| panic!("Failed to parse config: {}", error));

    let lowpass = construct_filter(&config.wavelet);
    let highpass = qmf_highpass(&lowpass);

    let offsets = all_dsqg_offsets();
    let per_offset_errors = compute_per_offset_errors(&offsets, &lowpass, &highpass);

    let per_offset_reconstruction_error =
        per_offset_errors.iter().sum::<f64>() / per_offset_errors.len() as f64;
    let dense_coverage_uniformity = compute_dense_coverage_uniformity(&per_offset_errors);
    let coverage_gap_pattern = compute_coverage_gap_pattern(&per_offset_errors);
    let harmonic_alignment_score = compute_harmonic_alignment_score(&lowpass, &offsets);

    let output = Output {
        per_offset_reconstruction_error,
        dense_coverage_uniformity,
        coverage_gap_pattern,
        harmonic_alignment_score,
        per_offset_errors,
    };

    let output_json = serde_json::to_string_pretty(&output)
        .unwrap_or_else(|error| panic!("Failed to serialize output: {}", error));
    std::fs::write(output_path, &output_json)
        .unwrap_or_else(|error| panic!("Failed to write {}: {}", output_path, error));

    eprintln!(
        "✅ {} offsets | mean_error={:.6} | uniformity={:.4} | gap_pattern={:.4} | alignment={:.4}",
        offsets.len(),
        per_offset_reconstruction_error,
        dense_coverage_uniformity,
        coverage_gap_pattern,
        harmonic_alignment_score,
    );
}
