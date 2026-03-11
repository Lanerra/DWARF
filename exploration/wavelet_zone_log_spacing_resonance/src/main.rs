use serde::Deserialize;
use std::f64::consts::{PI, SQRT_2};

const FREQUENCY_RESOLUTION: usize = 8192;
const DWT_LEVELS: usize = 10;
const ZONE_DELTAS: [f64; 5] = [1.0, 48.0, 96.0, 128.0, 384.0];
const PEAK_THRESHOLD_FRACTION: f64 = 0.1;

// ── Config structs (compatible with math_autoresearch format) ──

#[derive(Deserialize)]
struct Config {
    #[allow(dead_code)]
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

// ── Wavelet filter construction (from math_autoresearch) ──

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
    let order = half_taps;
    let mut polyphase = vec![[[0.0f64; 2]; 2]; order];
    let inverse_sqrt2 = 1.0 / SQRT_2;
    polyphase[0] = [
        [inverse_sqrt2, inverse_sqrt2],
        [inverse_sqrt2, -inverse_sqrt2],
    ];
    let theta_zero: f64 = -angles.iter().sum::<f64>();
    apply_rotation(&mut polyphase, theta_zero, order);
    for &angle in angles {
        for k in (1..order).rev() {
            polyphase[k][1] = polyphase[k - 1][1];
        }
        polyphase[0][1] = [0.0, 0.0];
        apply_rotation(&mut polyphase, angle, order);
    }
    let mut filter = vec![0.0; 2 * order];
    for k in 0..order {
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

fn qmf_highpass(lowpass: &[f64]) -> Vec<f64> {
    let length = lowpass.len();
    (0..length)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * lowpass[length - 1 - k]
        })
        .collect()
}

// ── Frequency response computation ──

fn compute_filter_response(filter: &[f64], frequency_count: usize) -> Vec<(f64, f64)> {
    (0..frequency_count)
        .map(|bin| {
            let omega = 2.0 * PI * bin as f64 / frequency_count as f64;
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (tap_index, &coefficient) in filter.iter().enumerate() {
                real += coefficient * (omega * tap_index as f64).cos();
                imaginary -= coefficient * (omega * tap_index as f64).sin();
            }
            (real, imaginary)
        })
        .collect()
}

// ── Multi-level DWT composite frequency response ──
// Computes Σ_l |H_detail_l(ω)| across all DWT levels, yielding a
// frequency curve with peaks at each subband's center frequency.

fn compute_composite_magnitude_response(
    lowpass_response: &[(f64, f64)],
    highpass_response: &[(f64, f64)],
    levels: usize,
) -> Vec<f64> {
    let frequency_count = lowpass_response.len();
    let positive_bins = frequency_count / 2 + 1;
    let mut composite = vec![0.0; positive_bins];

    for level in 0..levels {
        for bin in 0..positive_bins {
            let highpass_index = (bin << level) % frequency_count;
            let (mut real, mut imaginary) = highpass_response[highpass_index];
            for stage in 0..level {
                let lowpass_index = (bin << stage) % frequency_count;
                let (lowpass_real, lowpass_imaginary) = lowpass_response[lowpass_index];
                let new_real = real * lowpass_real - imaginary * lowpass_imaginary;
                let new_imaginary = real * lowpass_imaginary + imaginary * lowpass_real;
                real = new_real;
                imaginary = new_imaginary;
            }
            composite[bin] += (real * real + imaginary * imaginary).sqrt();
        }
    }

    composite
}

// ── Peak detection in composite response ──

fn find_peaks(response: &[f64], threshold_fraction: f64) -> Vec<usize> {
    let maximum = response
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let threshold = threshold_fraction * maximum;
    let length = response.len();
    let mut peaks = Vec::new();

    for bin in 1..length - 1 {
        if response[bin] > threshold
            && response[bin] > response[bin - 1]
            && response[bin] > response[bin + 1]
        {
            peaks.push(bin);
        }
    }

    if length > 1 && response[length - 1] > threshold && response[length - 1] > response[length - 2]
    {
        peaks.push(length - 1);
    }

    peaks
}

// ── Statistical functions ──

fn pearson_correlation(x_values: &[f64], y_values: &[f64]) -> f64 {
    assert_eq!(x_values.len(), y_values.len());
    let count = x_values.len() as f64;
    if count < 2.0 {
        return 0.0;
    }
    let mean_x = x_values.iter().sum::<f64>() / count;
    let mean_y = y_values.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for index in 0..x_values.len() {
        let deviation_x = x_values[index] - mean_x;
        let deviation_y = y_values[index] - mean_y;
        covariance += deviation_x * deviation_y;
        variance_x += deviation_x * deviation_x;
        variance_y += deviation_y * deviation_y;
    }
    if variance_x < 1e-15 || variance_y < 1e-15 {
        return 0.0;
    }
    covariance / (variance_x.sqrt() * variance_y.sqrt())
}

fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; values.len()];
    let mut position = 0;
    while position < indexed.len() {
        let mut end = position + 1;
        while end < indexed.len() && (indexed[end].1 - indexed[position].1).abs() < 1e-12 {
            end += 1;
        }
        let average_rank = (position + end) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(end).skip(position) {
            ranks[item.0] = average_rank;
        }
        position = end;
    }
    ranks
}

fn spearman_correlation(x_values: &[f64], y_values: &[f64]) -> f64 {
    let rank_x = compute_ranks(x_values);
    let rank_y = compute_ranks(y_values);
    pearson_correlation(&rank_x, &rank_y)
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

    let config_string = std::fs::read_to_string(input_path)
        .unwrap_or_else(|error| panic!("Failed to read {}: {}", input_path, error));
    let config: Config = serde_json::from_str(&config_string)
        .unwrap_or_else(|error| panic!("Failed to parse {}: {}", input_path, error));

    let lowpass = construct_filter(&config.wavelet);
    let highpass = qmf_highpass(&lowpass);

    let lowpass_response = compute_filter_response(&lowpass, FREQUENCY_RESOLUTION);
    let highpass_response = compute_filter_response(&highpass, FREQUENCY_RESOLUTION);

    let composite =
        compute_composite_magnitude_response(&lowpass_response, &highpass_response, DWT_LEVELS);

    let peak_bins = find_peaks(&composite, PEAK_THRESHOLD_FRACTION);

    // Convert peak bins to log-periods (bin k → period N/k → log(N/k))
    let mut log_peaks: Vec<f64> = peak_bins
        .iter()
        .filter(|&&bin| bin > 0)
        .map(|&bin| (FREQUENCY_RESOLUTION as f64 / bin as f64).ln())
        .collect();
    log_peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let log_bounds: Vec<f64> = ZONE_DELTAS.iter().map(|delta| delta.ln()).collect();

    // ── Metric 1: zone_log_boundary_peak_alignment ──
    let zone_log_boundary_peak_alignment = if log_peaks.is_empty() {
        0.0
    } else {
        let average_spacing = (log_bounds.last().unwrap() - log_bounds.first().unwrap())
            / (log_bounds.len() - 1) as f64;

        let mean_distance: f64 = log_bounds
            .iter()
            .map(|&bound| {
                log_peaks
                    .iter()
                    .map(|&peak| (bound - peak).abs())
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / log_bounds.len() as f64;

        (1.0 - mean_distance / average_spacing).clamp(0.0, 1.0)
    };

    // ── Metric 2: inter_zone_log_gap_resonance ──
    let zone_gaps: Vec<f64> = log_bounds.windows(2).map(|window| window[1] - window[0]).collect();
    let peak_gaps: Vec<f64> = log_peaks
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .collect();

    let inter_zone_log_gap_resonance = if peak_gaps.len() >= 2 && zone_gaps.len() >= 2 {
        let minimum_length = zone_gaps.len().min(peak_gaps.len());
        let rho = spearman_correlation(&zone_gaps[..minimum_length], &peak_gaps[..minimum_length]);
        ((rho + 1.0) / 2.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // ── Metric 3: frequency_peak_to_log_spacing_correlation ──
    let frequency_peak_to_log_spacing_correlation =
        if peak_gaps.len() >= 2 && zone_gaps.len() >= 2 {
            let minimum_length = zone_gaps.len().min(peak_gaps.len());
            let correlation =
                pearson_correlation(&zone_gaps[..minimum_length], &peak_gaps[..minimum_length]);
            correlation.max(0.0)
        } else {
            0.0
        };

    let output = serde_json::json!({
        "zone_log_boundary_peak_alignment": zone_log_boundary_peak_alignment,
        "inter_zone_log_gap_resonance": inter_zone_log_gap_resonance,
        "frequency_peak_to_log_spacing_correlation": frequency_peak_to_log_spacing_correlation,
    });

    let output_string =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    std::fs::write(output_path, &output_string)
        .unwrap_or_else(|error| panic!("Failed to write {}: {}", output_path, error));

    eprintln!(
        "alignment={:.4} resonance={:.4} correlation={:.4}",
        zone_log_boundary_peak_alignment,
        inter_zone_log_gap_resonance,
        frequency_peak_to_log_spacing_correlation,
    );
}
