use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

const FFT_SIZE: usize = 2048;

const DENSE_ZONE_START: usize = 1;
const DENSE_ZONE_END: usize = 48;
const MID_ZONE_OFFSETS: [usize; 2] = [96, 128];
const LONG_ZONE_OFFSET: usize = 384;

const ZONE_BOUNDARIES: [usize; 4] = [48, 96, 128, 384];

#[derive(Deserialize)]
struct Config {
    wavelet: WaveletConfig,
}

#[derive(Deserialize)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
}

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
    let inverse_sqrt2 = 1.0 / SQRT_2;
    let mut polyphase = vec![
        [[0.0f64; 2]; 2];
        half_taps
    ];
    polyphase[0] = [
        [inverse_sqrt2, inverse_sqrt2],
        [inverse_sqrt2, -inverse_sqrt2],
    ];

    let theta_zero: f64 = -angles.iter().sum::<f64>();
    apply_rotation(&mut polyphase, theta_zero, half_taps);

    for &angle in angles {
        for k in (1..half_taps).rev() {
            polyphase[k][1] = polyphase[k - 1][1];
        }
        polyphase[0][1] = [0.0, 0.0];
        apply_rotation(&mut polyphase, angle, half_taps);
    }

    let mut filter = vec![0.0; 2 * half_taps];
    for k in 0..half_taps {
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

fn offset_to_frequency(offset: usize) -> f64 {
    assert!(offset >= 1, "Offset must be >= 1");
    (offset as f64).ln() / (LONG_ZONE_OFFSET as f64).ln() * PI
}

fn compute_magnitude_spectrum(filter: &[f64]) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    for (index, &coefficient) in filter.iter().enumerate() {
        buffer[index] = Complex::new(coefficient, 0.0);
    }

    fft.process(&mut buffer);

    buffer.iter().map(|c| c.norm()).collect()
}

fn spectrum_at_frequency(magnitude_spectrum: &[f64], omega: f64) -> f64 {
    let bin = omega / (2.0 * PI) * FFT_SIZE as f64;
    let bin_low = (bin.floor() as usize).min(FFT_SIZE - 1);
    let bin_high = (bin_low + 1).min(FFT_SIZE - 1);
    let fraction = bin - bin_low as f64;
    magnitude_spectrum[bin_low] * (1.0 - fraction) + magnitude_spectrum[bin_high] * fraction
}

fn compute_zone_boundary_coherence(magnitude_spectrum: &[f64]) -> f64 {
    let delta_omega = 2.0 * PI / FFT_SIZE as f64;
    let mut total_gradient = 0.0;

    for &boundary in &ZONE_BOUNDARIES {
        let omega = offset_to_frequency(boundary);
        let omega_low = (omega - delta_omega).max(0.0);
        let omega_high = (omega + delta_omega).min(PI);

        let magnitude_low = spectrum_at_frequency(magnitude_spectrum, omega_low);
        let magnitude_high = spectrum_at_frequency(magnitude_spectrum, omega_high);

        let gradient = (magnitude_high - magnitude_low).abs() / (omega_high - omega_low);
        total_gradient += gradient;
    }

    total_gradient / ZONE_BOUNDARIES.len() as f64
}

fn compute_intra_zone_frequency_uniformity(magnitude_spectrum: &[f64]) -> f64 {
    let mut samples = Vec::with_capacity(DENSE_ZONE_END - DENSE_ZONE_START + 1);

    for offset in DENSE_ZONE_START..=DENSE_ZONE_END {
        let omega = offset_to_frequency(offset);
        let value = spectrum_at_frequency(magnitude_spectrum, omega);
        samples.push(value);
    }

    let count = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / count;

    if mean.abs() < 1e-15 {
        return 0.0;
    }

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
    let standard_deviation = variance.sqrt();

    standard_deviation / mean
}

fn compute_zone_frequency_separation_ratio(magnitude_spectrum: &[f64]) -> f64 {
    let frequency_48 = offset_to_frequency(DENSE_ZONE_END);
    let frequency_96 = offset_to_frequency(MID_ZONE_OFFSETS[0]);
    let frequency_128 = offset_to_frequency(MID_ZONE_OFFSETS[1]);
    let frequency_384 = offset_to_frequency(LONG_ZONE_OFFSET);

    let band_boundaries = [
        (frequency_48, frequency_96),
        (frequency_96, frequency_128),
        (frequency_128, frequency_384),
    ];

    let sample_count_per_band = 64;
    let mut band_means = Vec::new();

    for &(start, end) in &band_boundaries {
        let step = (end - start) / sample_count_per_band as f64;
        let mut sum = 0.0;
        for index in 0..sample_count_per_band {
            let omega = start + step * (index as f64 + 0.5);
            sum += spectrum_at_frequency(magnitude_spectrum, omega);
        }
        band_means.push(sum / sample_count_per_band as f64);
    }

    let overall_mean = band_means.iter().sum::<f64>() / band_means.len() as f64;
    let between_zone_variance = band_means
        .iter()
        .map(|m| (m - overall_mean).powi(2))
        .sum::<f64>()
        / band_means.len() as f64;

    let mut within_zone_variances = Vec::new();
    for (band_index, &(start, end)) in band_boundaries.iter().enumerate() {
        let step = (end - start) / sample_count_per_band as f64;
        let mut samples = Vec::new();
        for index in 0..sample_count_per_band {
            let omega = start + step * (index as f64 + 0.5);
            samples.push(spectrum_at_frequency(magnitude_spectrum, omega));
        }
        let band_mean = band_means[band_index];
        let variance = samples.iter().map(|x| (x - band_mean).powi(2)).sum::<f64>()
            / samples.len() as f64;
        within_zone_variances.push(variance);
    }

    let mean_within_variance =
        within_zone_variances.iter().sum::<f64>() / within_zone_variances.len() as f64;

    if mean_within_variance < 1e-15 {
        return between_zone_variance / 1e-15;
    }

    between_zone_variance / mean_within_variance
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        3,
        "Usage: dsqg_zone_frequency_alignment <input.json> <output.json>"
    );

    let input_path = &arguments[1];
    let output_path = &arguments[2];

    let config_string =
        std::fs::read_to_string(input_path).expect("Failed to read input JSON");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse input JSON");

    let filter = construct_filter(&config.wavelet);
    let magnitude_spectrum = compute_magnitude_spectrum(&filter);

    let zone_boundary_coherence = compute_zone_boundary_coherence(&magnitude_spectrum);
    let intra_zone_frequency_uniformity =
        compute_intra_zone_frequency_uniformity(&magnitude_spectrum);
    let zone_frequency_separation_ratio =
        compute_zone_frequency_separation_ratio(&magnitude_spectrum);

    let mut output = HashMap::new();
    output.insert(
        "zone_boundary_coherence".to_string(),
        zone_boundary_coherence,
    );
    output.insert(
        "intra_zone_frequency_uniformity".to_string(),
        intra_zone_frequency_uniformity,
    );
    output.insert(
        "zone_frequency_separation_ratio".to_string(),
        zone_frequency_separation_ratio,
    );

    let output_json =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    std::fs::write(output_path, &output_json).expect("Failed to write output JSON");

    println!("✓ Wrote metrics to {}", output_path);
    println!("  zone_boundary_coherence:          {:.6}", zone_boundary_coherence);
    println!(
        "  intra_zone_frequency_uniformity:  {:.6}",
        intra_zone_frequency_uniformity
    );
    println!(
        "  zone_frequency_separation_ratio:  {:.6}",
        zone_frequency_separation_ratio
    );
}
