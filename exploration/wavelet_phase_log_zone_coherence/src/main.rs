use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

const FFT_SIZE: usize = 4096;

// ── DSQG zone definitions ──

const DENSE_ZONE_MAX: usize = 48;
const MID_ZONE_OFFSETS: [usize; 2] = [96, 128];
const LONG_ZONE_OFFSET: usize = 384;
const ZONE_BOUNDARIES: [usize; 4] = [48, 96, 128, 384];

// ── Config ──

#[derive(Deserialize)]
struct Config {
    wavelet: WaveletConfig,
}

#[derive(Deserialize)]
struct WaveletConfig {
    filter_taps: usize,
    lattice_angles: Vec<f64>,
}

// ── Wavelet filter construction (lattice parameterization) ──

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
    let mut polyphase = vec![[[0.0f64; 2]; 2]; half_taps];
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

// ── Frequency response ──

fn offset_to_frequency(offset: usize) -> f64 {
    assert!(offset >= 1, "Offset must be >= 1");
    (offset as f64).ln() / (LONG_ZONE_OFFSET as f64).ln() * PI
}

fn compute_complex_spectrum(filter: &[f64]) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    for (index, &coefficient) in filter.iter().enumerate() {
        buffer[index] = Complex::new(coefficient, 0.0);
    }

    fft.process(&mut buffer);
    buffer
}

fn phase_at_frequency(spectrum: &[Complex<f64>], omega: f64) -> f64 {
    let bin = omega / (2.0 * PI) * FFT_SIZE as f64;
    let bin_low = (bin.floor() as usize).min(FFT_SIZE - 1);
    let bin_high = (bin_low + 1).min(FFT_SIZE - 1);
    let fraction = bin - bin_low as f64;

    let interpolated = spectrum[bin_low] * (1.0 - fraction) + spectrum[bin_high] * fraction;
    interpolated.arg()
}

// ── Phase unwrapping ──

fn unwrap_phase(phases: &[f64]) -> Vec<f64> {
    let mut unwrapped = vec![phases[0]];
    for i in 1..phases.len() {
        let mut delta = phases[i] - phases[i - 1];
        while delta > PI {
            delta -= 2.0 * PI;
        }
        while delta < -PI {
            delta += 2.0 * PI;
        }
        unwrapped.push(unwrapped[i - 1] + delta);
    }
    unwrapped
}

// ── Metrics ──

fn compute_zone_boundary_phase_gradient(spectrum: &[Complex<f64>]) -> (f64, Vec<f64>) {
    let mut gradients = Vec::with_capacity(ZONE_BOUNDARIES.len());

    for &boundary in &ZONE_BOUNDARIES {
        let offset_below = (boundary as f64 * 0.95).max(1.0) as usize;
        let offset_above = ((boundary as f64 * 1.05) as usize).min(512);

        let omega_below = offset_to_frequency(offset_below.max(1));
        let omega_above = offset_to_frequency(offset_above.max(1));

        let phase_below = phase_at_frequency(spectrum, omega_below);
        let phase_above = phase_at_frequency(spectrum, omega_above);

        let mut delta_phase = phase_above - phase_below;
        while delta_phase > PI {
            delta_phase -= 2.0 * PI;
        }
        while delta_phase < -PI {
            delta_phase += 2.0 * PI;
        }

        let log_omega_span = (omega_above - omega_below).max(1e-10);
        let gradient = (delta_phase / log_omega_span).abs();
        gradients.push(gradient);

        eprintln!(
            "  boundary δ={}: ω=[{:.4},{:.4}], |∂φ/∂ω|={:.6}",
            boundary, omega_below, omega_above, gradient
        );
    }

    let mean_gradient = gradients.iter().sum::<f64>() / gradients.len() as f64;
    (mean_gradient, gradients)
}

fn compute_inter_zone_phase_coherence(spectrum: &[Complex<f64>]) -> f64 {
    let dense_zone_samples = sample_zone_phases(spectrum, 1, DENSE_ZONE_MAX, 64);
    let mid_zone_samples = sample_zone_phases(
        spectrum,
        MID_ZONE_OFFSETS[0],
        MID_ZONE_OFFSETS[1],
        64,
    );
    let long_zone_samples = sample_zone_phases(
        spectrum,
        MID_ZONE_OFFSETS[1] + 1,
        LONG_ZONE_OFFSET,
        64,
    );

    let coherence_dense_mid =
        circular_phase_correlation(&dense_zone_samples, &mid_zone_samples);
    let coherence_mid_long =
        circular_phase_correlation(&mid_zone_samples, &long_zone_samples);

    eprintln!(
        "  coherence dense→mid: {:.6}, mid→long: {:.6}",
        coherence_dense_mid, coherence_mid_long
    );

    (coherence_dense_mid + coherence_mid_long) / 2.0
}

fn sample_zone_phases(
    spectrum: &[Complex<f64>],
    offset_start: usize,
    offset_end: usize,
    sample_count: usize,
) -> Vec<f64> {
    let omega_start = offset_to_frequency(offset_start);
    let omega_end = offset_to_frequency(offset_end);
    let step = (omega_end - omega_start) / sample_count as f64;

    (0..sample_count)
        .map(|i| {
            let omega = omega_start + step * (i as f64 + 0.5);
            phase_at_frequency(spectrum, omega)
        })
        .collect()
}

fn circular_phase_correlation(phases_a: &[f64], phases_b: &[f64]) -> f64 {
    assert_eq!(phases_a.len(), phases_b.len());

    let unwrapped_a = unwrap_phase(phases_a);
    let unwrapped_b = unwrap_phase(phases_b);

    let mean_a = unwrapped_a.iter().sum::<f64>() / unwrapped_a.len() as f64;
    let mean_b = unwrapped_b.iter().sum::<f64>() / unwrapped_b.len() as f64;

    let centered_a: Vec<f64> = unwrapped_a.iter().map(|p| p - mean_a).collect();
    let centered_b: Vec<f64> = unwrapped_b.iter().map(|p| p - mean_b).collect();

    let dot: f64 = centered_a
        .iter()
        .zip(centered_b.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f64 = centered_a.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_b: f64 = centered_b.iter().map(|b| b * b).sum::<f64>().sqrt();

    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 1.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn compute_per_boundary_phase_values(spectrum: &[Complex<f64>]) -> Vec<f64> {
    ZONE_BOUNDARIES
        .iter()
        .map(|&boundary| {
            let omega = offset_to_frequency(boundary);
            phase_at_frequency(spectrum, omega)
        })
        .collect()
}

// ── Main ──

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        3,
        "Usage: wavelet_phase_log_zone_coherence <input.json> <output.json>"
    );

    let input_path = &arguments[1];
    let output_path = &arguments[2];

    let config_string =
        std::fs::read_to_string(input_path).expect("Failed to read input JSON");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse input JSON");

    let filter = construct_filter(&config.wavelet);
    eprintln!("🔬 Filter coefficients: {:?}", filter);

    let spectrum = compute_complex_spectrum(&filter);

    eprintln!("📐 Computing zone boundary phase gradient...");
    let (zone_boundary_phase_gradient, _per_boundary_gradients) =
        compute_zone_boundary_phase_gradient(&spectrum);

    eprintln!("🔗 Computing inter-zone phase coherence...");
    let inter_zone_phase_coherence = compute_inter_zone_phase_coherence(&spectrum);

    let normalized_gradient =
        zone_boundary_phase_gradient / (1.0 + zone_boundary_phase_gradient);
    let phase_alignment_score =
        (1.0 - normalized_gradient) * inter_zone_phase_coherence.max(0.0);

    let per_boundary_phase_values = compute_per_boundary_phase_values(&spectrum);

    let mut output: HashMap<String, serde_json::Value> = HashMap::new();
    output.insert(
        "zone_boundary_phase_gradient".to_string(),
        serde_json::json!(zone_boundary_phase_gradient),
    );
    output.insert(
        "inter_zone_phase_coherence".to_string(),
        serde_json::json!(inter_zone_phase_coherence),
    );
    output.insert(
        "phase_alignment_score".to_string(),
        serde_json::json!(phase_alignment_score),
    );
    output.insert(
        "per_boundary_phase_values".to_string(),
        serde_json::json!(per_boundary_phase_values),
    );

    let output_json =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    std::fs::write(output_path, &output_json).expect("Failed to write output JSON");

    println!("✓ Wrote metrics to {}", output_path);
    println!(
        "  zone_boundary_phase_gradient:  {:.6}",
        zone_boundary_phase_gradient
    );
    println!(
        "  inter_zone_phase_coherence:    {:.6}",
        inter_zone_phase_coherence
    );
    println!(
        "  phase_alignment_score:         {:.6}",
        phase_alignment_score
    );
}
