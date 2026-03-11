use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

const FREQUENCY_RESOLUTION: usize = 4096;
const EPSILON: f64 = 1e-10;

// ── Target log-ratios from consecutive DSQG zone boundaries ──
// 96/48 = 2.0, 128/96 ≈ 1.333, 384/128 = 3.0

const TARGET_LOG_RATIOS: [f64; 3] = [
    0.6931471805599453,
    0.28768207245178085,
    1.0986122886681098,
];

// ── DSQG zone boundaries ──

const DENSE_ZONE_MAX_OFFSET: f64 = 48.0;
const LONG_ZONE_MAX_OFFSET: f64 = 384.0;

// ── Config structs (matches math_autoresearch) ──

#[derive(Deserialize)]
struct Config {
    wavelet: WaveletConfig,
}

#[derive(Deserialize)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
}

// ── Output ──

#[derive(Serialize)]
struct Output {
    frequency_response_log_alignment: f64,
    peak_concentration_sharpness: f64,
    dense_zone_frequency_focus_efficiency: f64,
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

// ── Frequency response via DTFT ──
// H(ω) = Σ h[n] * e^{-jωn}, evaluate |H(ω)| at N points over [0, π]

fn compute_frequency_response(filter: &[f64], resolution: usize) -> Vec<f64> {
    let mut magnitude = vec![0.0; resolution];
    for k in 0..resolution {
        let omega = PI * k as f64 / resolution as f64;
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (n, &h) in filter.iter().enumerate() {
            real += h * (omega * n as f64).cos();
            imaginary -= h * (omega * n as f64).sin();
        }
        magnitude[k] = (real * real + imaginary * imaginary).sqrt();
    }
    magnitude
}

// ── Peak detection ──

struct Peak {
    index: usize,
    frequency: f64,
    amplitude: f64,
}

fn detect_peaks(magnitude: &[f64], resolution: usize) -> Vec<Peak> {
    let mut peaks = Vec::new();

    if magnitude[0] > magnitude[1] {
        peaks.push(Peak {
            index: 0,
            frequency: 0.0,
            amplitude: magnitude[0],
        });
    }

    for k in 1..(magnitude.len() - 1) {
        if magnitude[k] > magnitude[k - 1] && magnitude[k] > magnitude[k + 1] {
            peaks.push(Peak {
                index: k,
                frequency: PI * k as f64 / resolution as f64,
                amplitude: magnitude[k],
            });
        }
    }

    let last = magnitude.len() - 1;
    if magnitude[last] > magnitude[last - 1] {
        peaks.push(Peak {
            index: last,
            frequency: PI * last as f64 / resolution as f64,
            amplitude: magnitude[last],
        });
    }

    peaks.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap());
    peaks
}

// ── Metric 1: frequency_response_log_alignment ──
// For each pair of peaks, compute log-ratio of frequencies and compare
// against target ratios [ln(2.0), ln(1.333), ln(3.0)].

fn compute_frequency_response_log_alignment(peaks: &[Peak]) -> f64 {
    if peaks.len() < 2 {
        return 0.0;
    }

    let mut alignment_scores = Vec::new();

    for i in 0..peaks.len() {
        for j in (i + 1)..peaks.len() {
            let frequency_high = peaks[i].frequency.max(peaks[j].frequency);
            let frequency_low = peaks[i].frequency.min(peaks[j].frequency);

            if frequency_low < EPSILON {
                continue;
            }

            let log_ratio = (frequency_high / frequency_low).ln();

            let minimum_distance = TARGET_LOG_RATIOS
                .iter()
                .map(|&target| (log_ratio - target).abs())
                .fold(f64::INFINITY, f64::min);

            alignment_scores.push((-minimum_distance * 3.0).exp());
        }
    }

    if alignment_scores.is_empty() {
        return 0.0;
    }

    alignment_scores.iter().sum::<f64>() / alignment_scores.len() as f64
}

// ── Metric 2: peak_concentration_sharpness ──
// FWHM-based: sharper peaks (lower FWHM) → higher score.

fn compute_peak_concentration_sharpness(
    magnitude: &[f64],
    peaks: &[Peak],
    resolution: usize,
) -> f64 {
    if peaks.is_empty() {
        return 0.0;
    }

    let mut sharpness_sum = 0.0;
    let mut peak_count = 0;

    for peak in peaks {
        let half_max = peak.amplitude / 2.0;

        let mut left_index = peak.index;
        while left_index > 0 && magnitude[left_index] > half_max {
            left_index -= 1;
        }

        let mut right_index = peak.index;
        while right_index < magnitude.len() - 1 && magnitude[right_index] > half_max {
            right_index += 1;
        }

        let fwhm_bins = (right_index - left_index) as f64;
        let fwhm_omega = fwhm_bins * PI / resolution as f64;

        if fwhm_omega > EPSILON {
            sharpness_sum += PI / fwhm_omega;
            peak_count += 1;
        }
    }

    if peak_count == 0 {
        return 0.0;
    }

    let average_sharpness = sharpness_sum / peak_count as f64;

    // Normalize to [0, 1]: higher sharpness → closer to 1
    1.0 - (-average_sharpness / 10.0).exp()
}

// ── Metric 3: dense_zone_frequency_focus_efficiency ──
// Ratio of energy in dense-zone frequencies vs total DSQG-relevant energy,
// using log-uniform weighting (1/ω) to avoid bias toward high frequencies.

fn compute_dense_zone_frequency_focus_efficiency(
    magnitude: &[f64],
    resolution: usize,
) -> f64 {
    let omega_dense_lower = PI / DENSE_ZONE_MAX_OFFSET;
    let omega_total_lower = PI / LONG_ZONE_MAX_OFFSET;

    let mut dense_zone_energy = 0.0;
    let mut total_energy = 0.0;

    for k in 1..resolution {
        let omega = PI * k as f64 / resolution as f64;
        let energy = magnitude[k] * magnitude[k];
        let weight = 1.0 / omega;
        let weighted_energy = energy * weight;

        if omega >= omega_total_lower {
            total_energy += weighted_energy;
        }
        if omega >= omega_dense_lower {
            dense_zone_energy += weighted_energy;
        }
    }

    if total_energy < EPSILON {
        return 0.0;
    }

    dense_zone_energy / total_energy
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

    let filter = construct_filter(&config.wavelet);
    let magnitude = compute_frequency_response(&filter, FREQUENCY_RESOLUTION);
    let peaks = detect_peaks(&magnitude, FREQUENCY_RESOLUTION);

    let frequency_response_log_alignment = compute_frequency_response_log_alignment(&peaks);
    let peak_concentration_sharpness =
        compute_peak_concentration_sharpness(&magnitude, &peaks, FREQUENCY_RESOLUTION);
    let dense_zone_frequency_focus_efficiency =
        compute_dense_zone_frequency_focus_efficiency(&magnitude, FREQUENCY_RESOLUTION);

    let output = Output {
        frequency_response_log_alignment,
        peak_concentration_sharpness,
        dense_zone_frequency_focus_efficiency,
    };

    let output_json = serde_json::to_string_pretty(&output)
        .unwrap_or_else(|error| panic!("Failed to serialize output: {}", error));
    std::fs::write(output_path, &output_json)
        .unwrap_or_else(|error| panic!("Failed to write {}: {}", output_path, error));

    eprintln!(
        "✅ taps={} peaks={} | log_alignment={:.4} | sharpness={:.4} | focus_efficiency={:.4}",
        config.wavelet.filter_taps,
        peaks.len(),
        frequency_response_log_alignment,
        peak_concentration_sharpness,
        dense_zone_frequency_focus_efficiency,
    );
}
