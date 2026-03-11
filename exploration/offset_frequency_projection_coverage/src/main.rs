use serde::{Deserialize, Serialize};
use std::f64::consts::SQRT_2;

// ── DSQG constants ──

const DENSE_ZONE_START: usize = 1;
const DENSE_ZONE_END: usize = 48;
const DILATION_COUNT: usize = 48;
const LOG_FREQUENCY_BINS: usize = 256;

fn dense_offsets() -> Vec<usize> {
    (DENSE_ZONE_START..=DENSE_ZONE_END).collect()
}

fn mid_offsets() -> Vec<usize> {
    vec![96, 128]
}

fn long_offsets() -> Vec<usize> {
    vec![384]
}

fn all_offsets() -> Vec<usize> {
    let mut offsets = dense_offsets();
    offsets.extend(mid_offsets());
    offsets.extend(long_offsets());
    offsets
}

fn dilations() -> Vec<usize> {
    (1..=DILATION_COUNT).collect()
}

// ── Config structs (matching math_autoresearch format) ──

#[derive(Deserialize)]
struct Config {
    search_mode: String,
    wavelet: WaveletConfig,
}

#[derive(Deserialize)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
    #[allow(dead_code)]
    levels_with_gelu: Vec<usize>,
    #[allow(dead_code)]
    block_structure: String,
}

// ── Output struct ──

#[derive(Serialize)]
struct OutputMetrics {
    log_frequency_coverage_uniformity: f64,
    max_log_frequency_gap_octaves: f64,
    offset_step_to_dilation_ratio: f64,
    dense_zone_projected_frequency_utilization: f64,
}

// ── Wavelet filter construction (from math_autoresearch) ──

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

// ── Frequency projection ──
//
// For each (offset, dilation) pair, the projected frequency is:
//   f = theta^dilation * offset
// where theta is the dominant lattice angle.

fn compute_projected_frequencies(theta: f64, offsets: &[usize], dilations: &[usize]) -> Vec<f64> {
    let mut frequencies = Vec::with_capacity(offsets.len() * dilations.len());
    for &dilation in dilations {
        let scale = theta.abs().powi(dilation as i32);
        for &offset in offsets {
            let frequency = scale * (offset as f64);
            if frequency > 0.0 && frequency.is_finite() {
                frequencies.push(frequency);
            }
        }
    }
    frequencies
}

// ── Metric computation ──

fn compute_log_frequency_histogram(frequencies: &[f64], bin_count: usize) -> (Vec<f64>, f64, f64) {
    assert!(!frequencies.is_empty(), "No valid projected frequencies");

    let log_frequencies: Vec<f64> = frequencies.iter().map(|f| f.ln()).collect();
    let min_log = log_frequencies.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_log = log_frequencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_log - min_log;

    assert!(range > 0.0, "All frequencies are identical");

    let mut histogram = vec![0.0; bin_count];
    for &log_frequency in &log_frequencies {
        let bin = ((log_frequency - min_log) / range * (bin_count as f64 - 1.0)).round() as usize;
        let clamped_bin = bin.min(bin_count - 1);
        histogram[clamped_bin] += 1.0;
    }

    let total: f64 = histogram.iter().sum();
    for count in histogram.iter_mut() {
        *count /= total;
    }

    (histogram, min_log, max_log)
}

fn compute_coverage_uniformity(histogram: &[f64]) -> f64 {
    // D_KL(uniform || actual): how far the actual distribution
    // deviates from uniform coverage.
    // Returns normalized score where 1.0 = perfectly uniform.
    let bin_count = histogram.len();
    let uniform_probability = 1.0 / bin_count as f64;
    let epsilon = 1e-12;

    let mut kl_divergence = 0.0;
    for &probability in histogram {
        let clamped = probability.max(epsilon);
        kl_divergence += uniform_probability * (uniform_probability / clamped).ln();
    }

    let max_kl = (bin_count as f64).ln();
    let normalized = 1.0 - (kl_divergence / max_kl).min(1.0);
    normalized.max(0.0)
}

fn compute_max_log_frequency_gap_octaves(frequencies: &[f64]) -> f64 {
    let mut log_frequencies: Vec<f64> = frequencies.iter().map(|f| f.log2()).collect();
    log_frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    log_frequencies.dedup();

    if log_frequencies.len() < 2 {
        return 0.0;
    }

    let mut max_gap = 0.0_f64;
    for window in log_frequencies.windows(2) {
        let gap = window[1] - window[0];
        max_gap = max_gap.max(gap);
    }

    max_gap
}

fn compute_offset_step_to_dilation_ratio(theta: f64) -> f64 {
    // For log-uniform coverage, offset stepping should be multiplicative.
    // With uniform additive stepping (step=1), we measure how close
    // the resulting log-frequency increments are to the ideal.
    let dilations_list = dilations();
    let offsets_list = dense_offsets();

    let ideal_log_step =
        (DENSE_ZONE_END as f64 / DENSE_ZONE_START as f64).ln() / (offsets_list.len() as f64 - 1.0);

    let mut total_score = 0.0;
    let mut count = 0.0;

    for &dilation in &dilations_list {
        let scale = theta.abs().powi(dilation as i32);
        if scale <= 0.0 || !scale.is_finite() {
            continue;
        }

        for &offset in &offsets_list {
            let frequency = scale * (offset as f64);
            let next_frequency = scale * (offset as f64 + 1.0);

            if frequency <= 0.0 || next_frequency <= 0.0 {
                continue;
            }

            let actual_log_step = (next_frequency / frequency).ln();
            let ratio = actual_log_step / ideal_log_step;
            let step_score = 1.0 / (1.0 + (ratio - 1.0).abs());

            total_score += step_score;
            count += 1.0;
        }
    }

    if count == 0.0 {
        return 0.0;
    }

    total_score / count
}

fn compute_dense_zone_frequency_utilization(theta: f64) -> f64 {
    // Fraction of the dense zone's reachable log-frequency span
    // that the 48 taps actually sample, measured via bin occupancy.
    let offsets_list = dense_offsets();
    let dilations_list = dilations();

    let mut all_log_frequencies: Vec<f64> = Vec::new();
    for &dilation in &dilations_list {
        let scale = theta.abs().powi(dilation as i32);
        for &offset in &offsets_list {
            let frequency = scale * (offset as f64);
            if frequency > 0.0 && frequency.is_finite() {
                all_log_frequencies.push(frequency.ln());
            }
        }
    }

    if all_log_frequencies.is_empty() {
        return 0.0;
    }

    let total_min = all_log_frequencies
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let total_max = all_log_frequencies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let total_span = total_max - total_min;

    if total_span <= 0.0 {
        return 0.0;
    }

    let bin_count = LOG_FREQUENCY_BINS;
    let mut occupied = vec![false; bin_count];

    for &log_frequency in &all_log_frequencies {
        let bin =
            ((log_frequency - total_min) / total_span * (bin_count as f64 - 1.0)).round() as usize;
        let clamped_bin = bin.min(bin_count - 1);
        occupied[clamped_bin] = true;
    }

    let occupied_count = occupied.iter().filter(|&&b| b).count();
    occupied_count as f64 / bin_count as f64
}

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

    let config_string =
        std::fs::read_to_string(input_path).expect("Failed to read input config JSON");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse input config JSON");
    assert_eq!(
        config.search_mode, "wavelet_filter",
        "Only wavelet_filter search_mode is supported"
    );

    let filter = construct_filter(&config.wavelet);
    let theta = config.wavelet.lattice_angles[0];

    eprintln!("⚙ Filter taps: {}", config.wavelet.filter_taps);
    eprintln!("⚙ Lattice angle (rad): {}", theta);
    eprintln!("⚙ Filter coefficients: {:?}", filter);

    let offsets_list = all_offsets();
    let dilations_list = dilations();

    let frequencies = compute_projected_frequencies(theta, &offsets_list, &dilations_list);
    eprintln!(
        "⚙ Projected {} frequency points from {} offsets × {} dilations",
        frequencies.len(),
        offsets_list.len(),
        dilations_list.len()
    );

    let (histogram, _min_log, _max_log) =
        compute_log_frequency_histogram(&frequencies, LOG_FREQUENCY_BINS);

    let coverage_uniformity = compute_coverage_uniformity(&histogram);
    let max_gap_octaves = compute_max_log_frequency_gap_octaves(&frequencies);
    let step_ratio = compute_offset_step_to_dilation_ratio(theta);
    let utilization = compute_dense_zone_frequency_utilization(theta);

    let output = OutputMetrics {
        log_frequency_coverage_uniformity: coverage_uniformity,
        max_log_frequency_gap_octaves: max_gap_octaves,
        offset_step_to_dilation_ratio: step_ratio,
        dense_zone_projected_frequency_utilization: utilization,
    };

    let output_json =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output metrics");
    std::fs::write(output_path, &output_json).expect("Failed to write output JSON");

    eprintln!("✓ Results written to {}", output_path);
    eprintln!("{}", output_json);
}
