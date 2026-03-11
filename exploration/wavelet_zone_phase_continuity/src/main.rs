use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

const FFT_SIZE: usize = 4096;
const SIGNAL_LENGTH: usize = 512;
const DWT_LEVELS: usize = 9;
const SPARSE_TOP_K: usize = 12;
const PHASE_SAMPLE_COUNT: usize = 128;

// ── DSQG zone definitions ──

const DENSE_ZONE_MAX: usize = 48;
const MID_ZONE_OFFSETS: [usize; 2] = [96, 128];
const LONG_ZONE_OFFSET: usize = 384;

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

// ── Frequency domain ──

fn offset_to_frequency(offset: f64) -> f64 {
    assert!(offset >= 1.0, "Offset must be >= 1");
    offset.ln() / (LONG_ZONE_OFFSET as f64).ln() * PI
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

fn magnitude_squared_at_frequency(spectrum: &[Complex<f64>], omega: f64) -> f64 {
    let bin = omega / (2.0 * PI) * FFT_SIZE as f64;
    let bin_low = (bin.floor() as usize).min(FFT_SIZE - 1);
    let bin_high = (bin_low + 1).min(FFT_SIZE - 1);
    let fraction = bin - bin_low as f64;

    let interpolated = spectrum[bin_low] * (1.0 - fraction) + spectrum[bin_high] * fraction;
    interpolated.norm_sqr()
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

// ── Sparse reconstruction error ──

fn sparse_reconstruct_error(signal: &[f64], lowpass: &[f64], highpass: &[f64]) -> f64 {
    let dwt_result = dwt_forward(signal, lowpass, highpass, DWT_LEVELS);

    let mut all_coefficients: Vec<f64> = Vec::new();
    for detail in &dwt_result.details {
        all_coefficients.extend(detail);
    }
    all_coefficients.extend(&dwt_result.approximation);
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
        sparse_details
            .push(sparse_coefficients[coefficient_offset..coefficient_offset + length].to_vec());
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

// ── Pearson correlation ──

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let count = x.len() as f64;

    let mean_x = x.iter().sum::<f64>() / count;
    let mean_y = y.iter().sum::<f64>() / count;

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;

    for i in 0..x.len() {
        let deviation_x = x[i] - mean_x;
        let deviation_y = y[i] - mean_y;
        covariance += deviation_x * deviation_y;
        variance_x += deviation_x * deviation_x;
        variance_y += deviation_y * deviation_y;
    }

    let denominator = (variance_x * variance_y).sqrt();
    if denominator < 1e-15 {
        return 1.0;
    }

    (covariance / denominator).clamp(-1.0, 1.0)
}

// ── Metric 1: zone_boundary_phase_discontinuity ──
// Max |dφ⁻/dω - dφ⁺/dω| across dense→mid and mid→long boundaries

fn compute_zone_boundary_phase_discontinuity(
    spectrum: &[Complex<f64>],
) -> (f64, Vec<f64>) {
    let boundary_pairs: [(f64, f64); 2] = [
        (DENSE_ZONE_MAX as f64, MID_ZONE_OFFSETS[0] as f64),
        (MID_ZONE_OFFSETS[1] as f64, LONG_ZONE_OFFSET as f64),
    ];

    let epsilon = 0.5;
    let mut per_edge_discontinuity = Vec::new();

    for &(offset_below, offset_above) in &boundary_pairs {
        let omega_below = offset_to_frequency(offset_below);
        let omega_below_minus = offset_to_frequency((offset_below - epsilon).max(1.0));
        let phase_at_below = phase_at_frequency(spectrum, omega_below);
        let phase_at_below_minus = phase_at_frequency(spectrum, omega_below_minus);

        let mut delta_phase_below = phase_at_below - phase_at_below_minus;
        while delta_phase_below > PI {
            delta_phase_below -= 2.0 * PI;
        }
        while delta_phase_below < -PI {
            delta_phase_below += 2.0 * PI;
        }
        let gradient_below = delta_phase_below / (omega_below - omega_below_minus).max(1e-10);

        let omega_above = offset_to_frequency(offset_above);
        let omega_above_plus = offset_to_frequency(offset_above + epsilon);
        let phase_at_above = phase_at_frequency(spectrum, omega_above);
        let phase_at_above_plus = phase_at_frequency(spectrum, omega_above_plus);

        let mut delta_phase_above = phase_at_above_plus - phase_at_above;
        while delta_phase_above > PI {
            delta_phase_above -= 2.0 * PI;
        }
        while delta_phase_above < -PI {
            delta_phase_above += 2.0 * PI;
        }
        let gradient_above = delta_phase_above / (omega_above_plus - omega_above).max(1e-10);

        let mismatch = (gradient_below - gradient_above).abs();
        per_edge_discontinuity.push(mismatch);

        eprintln!(
            "  boundary δ=[{},{}]: grad⁻={:.6}, grad⁺={:.6}, mismatch={:.6}",
            offset_below, offset_above, gradient_below, gradient_above, mismatch
        );
    }

    let max_mismatch = per_edge_discontinuity
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    (max_mismatch, per_edge_discontinuity)
}

// ── Metric 2: phase_frequency_smoothness_score ──
// ∫(dφ/dω)² dω across all zones

fn compute_phase_frequency_smoothness_score(spectrum: &[Complex<f64>]) -> f64 {
    let sample_count = PHASE_SAMPLE_COUNT * 4;
    let omega_min = offset_to_frequency(1.0);
    let omega_max = offset_to_frequency(LONG_ZONE_OFFSET as f64);
    let step = (omega_max - omega_min) / sample_count as f64;

    let phases: Vec<f64> = (0..=sample_count)
        .map(|i| {
            let omega = omega_min + step * i as f64;
            phase_at_frequency(spectrum, omega)
        })
        .collect();

    let unwrapped = unwrap_phase(&phases);

    let mut integral = 0.0;
    for i in 1..unwrapped.len() {
        let derivative = (unwrapped[i] - unwrapped[i - 1]) / step;
        integral += derivative * derivative * step;
    }

    integral
}

// ── Metric 3: cross_zone_phase_coherence ──
// Pearson correlation of unwrapped phase between adjacent zones at boundaries

fn sample_zone_phases(
    spectrum: &[Complex<f64>],
    offset_start: f64,
    offset_end: f64,
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

fn compute_cross_zone_phase_coherence(
    spectrum: &[Complex<f64>],
) -> (f64, f64, f64) {
    let dense_boundary_phases =
        sample_zone_phases(spectrum, 24.0, 48.0, PHASE_SAMPLE_COUNT);
    let mid_start_phases =
        sample_zone_phases(spectrum, 96.0, 128.0, PHASE_SAMPLE_COUNT);

    let mid_end_phases =
        sample_zone_phases(spectrum, 96.0, 128.0, PHASE_SAMPLE_COUNT);
    let long_zone_phases =
        sample_zone_phases(spectrum, 256.0, 384.0, PHASE_SAMPLE_COUNT);

    let dense_unwrapped = unwrap_phase(&dense_boundary_phases);
    let mid_start_unwrapped = unwrap_phase(&mid_start_phases);
    let mid_end_unwrapped = unwrap_phase(&mid_end_phases);
    let long_unwrapped = unwrap_phase(&long_zone_phases);

    let coherence_dense_mid = pearson_correlation(&dense_unwrapped, &mid_start_unwrapped);
    let coherence_mid_long = pearson_correlation(&mid_end_unwrapped, &long_unwrapped);

    let average = (coherence_dense_mid + coherence_mid_long) / 2.0;

    eprintln!(
        "  coherence dense→mid: {:.6}, mid→long: {:.6}",
        coherence_dense_mid, coherence_mid_long
    );

    (average, coherence_dense_mid, coherence_mid_long)
}

// ── Metric 4: signal_reconstruction_by_zone ──
// Per-zone reconstruction MSE from wavelet taps

fn compute_signal_reconstruction_by_zone(
    lowpass: &[f64],
    highpass: &[f64],
) -> (f64, f64, f64, f64) {
    let mut dense_error_sum = 0.0;
    for delta in 1..=DENSE_ZONE_MAX {
        let mut signal = vec![0.0; SIGNAL_LENGTH];
        signal[delta] = 1.0;
        dense_error_sum += sparse_reconstruct_error(&signal, lowpass, highpass);
    }
    let dense_mse = dense_error_sum / DENSE_ZONE_MAX as f64;

    let mut mid_error_sum = 0.0;
    for &delta in &MID_ZONE_OFFSETS {
        let mut signal = vec![0.0; SIGNAL_LENGTH];
        signal[delta] = 1.0;
        mid_error_sum += sparse_reconstruct_error(&signal, lowpass, highpass);
    }
    let mid_mse = mid_error_sum / MID_ZONE_OFFSETS.len() as f64;

    let mut signal = vec![0.0; SIGNAL_LENGTH];
    signal[LONG_ZONE_OFFSET] = 1.0;
    let long_mse = sparse_reconstruct_error(&signal, lowpass, highpass);

    let overall_mse = (dense_mse + mid_mse + long_mse) / 3.0;

    eprintln!(
        "  reconstruction MSE: dense={:.6}, mid={:.6}, long={:.6}, overall={:.6}",
        dense_mse, mid_mse, long_mse, overall_mse
    );

    (overall_mse, dense_mse, mid_mse, long_mse)
}

// ── Metric 5: phase_smoothness_vs_harmonic_ratio_trade ──
// Pareto frontier of (phase_discontinuity, harmonic_ratio) over zone offset perturbations

fn compute_harmonic_ratio_at_boundaries(
    spectrum: &[Complex<f64>],
    boundary_offsets: &[(f64, f64)],
) -> f64 {
    let bandwidth = 0.03 * PI;
    let sample_count = 512;

    let mut boundary_energy = 0.0;
    let mut total_energy = 0.0;

    for i in 0..sample_count {
        let omega = PI * i as f64 / (sample_count - 1) as f64;
        let energy = magnitude_squared_at_frequency(spectrum, omega);

        total_energy += energy;

        for &(offset_low, offset_high) in boundary_offsets {
            let boundary_omega =
                (offset_to_frequency(offset_low) + offset_to_frequency(offset_high)) / 2.0;
            if (omega - boundary_omega).abs() <= bandwidth {
                boundary_energy += energy;
                break;
            }
        }
    }

    if total_energy < 1e-15 {
        return 0.0;
    }
    boundary_energy / total_energy
}

fn compute_phase_smoothness_vs_harmonic_ratio_trade(
    spectrum: &[Complex<f64>],
) -> f64 {
    let perturbation_steps = 21;
    let mut phase_discontinuities = Vec::with_capacity(perturbation_steps);
    let mut harmonic_ratios = Vec::with_capacity(perturbation_steps);

    for step in 0..perturbation_steps {
        let perturbation = (step as f64 - 10.0) * 0.05;

        let dense_mid_lower = (DENSE_ZONE_MAX as f64 * (1.0 + perturbation)).max(2.0);
        let dense_mid_upper =
            (MID_ZONE_OFFSETS[0] as f64 * (1.0 + perturbation)).min(383.0);
        let mid_long_lower =
            (MID_ZONE_OFFSETS[1] as f64 * (1.0 + perturbation)).max(dense_mid_upper + 1.0);
        let mid_long_upper = LONG_ZONE_OFFSET as f64;

        let boundary_pairs = [
            (dense_mid_lower, dense_mid_upper),
            (mid_long_lower, mid_long_upper),
        ];

        let epsilon = 0.5;
        let mut max_mismatch = 0.0_f64;

        for &(offset_below, offset_above) in &boundary_pairs {
            let omega_below = offset_to_frequency(offset_below);
            let omega_below_minus = offset_to_frequency((offset_below - epsilon).max(1.0));
            let phase_below = phase_at_frequency(spectrum, omega_below);
            let phase_below_minus = phase_at_frequency(spectrum, omega_below_minus);

            let mut delta_below = phase_below - phase_below_minus;
            while delta_below > PI {
                delta_below -= 2.0 * PI;
            }
            while delta_below < -PI {
                delta_below += 2.0 * PI;
            }
            let gradient_below = delta_below / (omega_below - omega_below_minus).max(1e-10);

            let omega_above = offset_to_frequency(offset_above);
            let omega_above_plus = offset_to_frequency(offset_above + epsilon);
            let phase_above = phase_at_frequency(spectrum, omega_above);
            let phase_above_plus = phase_at_frequency(spectrum, omega_above_plus);

            let mut delta_above = phase_above_plus - phase_above;
            while delta_above > PI {
                delta_above -= 2.0 * PI;
            }
            while delta_above < -PI {
                delta_above += 2.0 * PI;
            }
            let gradient_above = delta_above / (omega_above_plus - omega_above).max(1e-10);

            max_mismatch = max_mismatch.max((gradient_below - gradient_above).abs());
        }

        phase_discontinuities.push(max_mismatch);
        harmonic_ratios.push(compute_harmonic_ratio_at_boundaries(
            spectrum,
            &boundary_pairs,
        ));
    }

    let correlation = pearson_correlation(&phase_discontinuities, &harmonic_ratios);

    (correlation + 1.0) / 2.0
}

// ── Main ──

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        3,
        "Usage: wavelet_zone_phase_continuity <input.json> <output.json>"
    );

    let input_path = &arguments[1];
    let output_path = &arguments[2];

    let config_string =
        std::fs::read_to_string(input_path).expect("Failed to read input JSON");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse input JSON");

    let filter = construct_filter(&config.wavelet);
    let highpass = qmf_highpass(&filter);
    eprintln!("🔬 Filter coefficients: {:?}", filter);

    let spectrum = compute_complex_spectrum(&filter);

    eprintln!("📐 Computing zone boundary phase discontinuity...");
    let (zone_boundary_phase_discontinuity, per_edge_discontinuity) =
        compute_zone_boundary_phase_discontinuity(&spectrum);

    eprintln!("📊 Computing phase frequency smoothness score...");
    let phase_frequency_smoothness_score =
        compute_phase_frequency_smoothness_score(&spectrum);

    eprintln!("🔗 Computing cross-zone phase coherence...");
    let (cross_zone_phase_coherence, coherence_dense_mid, coherence_mid_long) =
        compute_cross_zone_phase_coherence(&spectrum);

    eprintln!("🔄 Computing signal reconstruction by zone...");
    let (signal_reconstruction_by_zone, dense_mse, mid_mse, long_mse) =
        compute_signal_reconstruction_by_zone(&filter, &highpass);

    eprintln!("⚖️ Computing phase smoothness vs harmonic ratio trade...");
    let phase_smoothness_vs_harmonic_ratio_trade =
        compute_phase_smoothness_vs_harmonic_ratio_trade(&spectrum);

    let mut output: HashMap<String, serde_json::Value> = HashMap::new();

    output.insert(
        "zone_boundary_phase_discontinuity".to_string(),
        serde_json::json!(zone_boundary_phase_discontinuity),
    );
    output.insert(
        "phase_frequency_smoothness_score".to_string(),
        serde_json::json!(phase_frequency_smoothness_score),
    );
    output.insert(
        "cross_zone_phase_coherence".to_string(),
        serde_json::json!(cross_zone_phase_coherence),
    );
    output.insert(
        "signal_reconstruction_by_zone".to_string(),
        serde_json::json!(signal_reconstruction_by_zone),
    );
    output.insert(
        "phase_smoothness_vs_harmonic_ratio_trade".to_string(),
        serde_json::json!(phase_smoothness_vs_harmonic_ratio_trade),
    );

    output.insert(
        "zone_boundary_phase_discontinuity_per_edge".to_string(),
        serde_json::json!({
            "dense_to_mid": per_edge_discontinuity[0],
            "mid_to_long": per_edge_discontinuity[1],
        }),
    );
    output.insert(
        "cross_zone_correlations".to_string(),
        serde_json::json!({
            "dense_mid": coherence_dense_mid,
            "mid_long": coherence_mid_long,
        }),
    );
    output.insert(
        "per_zone_reconstruction_mse".to_string(),
        serde_json::json!({
            "dense": dense_mse,
            "mid": mid_mse,
            "long": long_mse,
        }),
    );

    let output_json =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    std::fs::write(output_path, &output_json).expect("Failed to write output JSON");

    println!("✓ Wrote metrics to {}", output_path);
    println!(
        "  zone_boundary_phase_discontinuity: {:.6}",
        zone_boundary_phase_discontinuity
    );
    println!(
        "  phase_frequency_smoothness_score:   {:.6}",
        phase_frequency_smoothness_score
    );
    println!(
        "  cross_zone_phase_coherence:         {:.6}",
        cross_zone_phase_coherence
    );
    println!(
        "  signal_reconstruction_by_zone:      {:.6}",
        signal_reconstruction_by_zone
    );
    println!(
        "  phase_smoothness_vs_harmonic_ratio: {:.6}",
        phase_smoothness_vs_harmonic_ratio_trade
    );
}
