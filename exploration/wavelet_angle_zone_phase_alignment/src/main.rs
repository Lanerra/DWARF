use num_complex::Complex64;
use serde::Deserialize;
use std::f64::consts::{PI, SQRT_2};

const EMBEDDING_DIMENSION: usize = 256;
const FREQUENCY_POINTS: usize = 4096;
const BOUNDARY_BANDWIDTH_FRACTION: f64 = 0.03;

// ── DSQG zone definitions ──

fn dsqg_zone_boundary_offsets() -> Vec<f64> {
    vec![48.0, 96.0, 128.0, 384.0]
}

fn offset_to_frequency(delta: f64) -> f64 {
    let raw = PI * delta / EMBEDDING_DIMENSION as f64;
    let wrapped = raw % (2.0 * PI);
    if wrapped > PI {
        2.0 * PI - wrapped
    } else {
        wrapped
    }
}

fn unique_boundary_frequencies() -> Vec<f64> {
    let mut frequencies: Vec<f64> = dsqg_zone_boundary_offsets()
        .iter()
        .map(|&delta| offset_to_frequency(delta))
        .collect();
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    frequencies.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    frequencies
}

fn dense_zone_frequency_range() -> (f64, f64) {
    (offset_to_frequency(1.0), offset_to_frequency(48.0))
}

fn mid_zone_frequency_range() -> (f64, f64) {
    (offset_to_frequency(96.0), offset_to_frequency(128.0))
}

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
    let mut polyphase = vec![[[0.0f64; 2]; 2]; half_taps];
    let inverse_sqrt2 = 1.0 / SQRT_2;
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

fn frequency_response(filter: &[f64], omega: f64) -> Complex64 {
    filter
        .iter()
        .enumerate()
        .map(|(k, &coefficient)| {
            let phase = -(k as f64) * omega;
            Complex64::new(coefficient, 0.0) * Complex64::new(phase.cos(), phase.sin())
        })
        .sum()
}

fn evaluate_frequency_grid(filter: &[f64]) -> (Vec<f64>, Vec<Complex64>) {
    let frequencies: Vec<f64> = (0..FREQUENCY_POINTS)
        .map(|i| PI * i as f64 / (FREQUENCY_POINTS - 1) as f64)
        .collect();
    let responses: Vec<Complex64> = frequencies
        .iter()
        .map(|&omega| frequency_response(filter, omega))
        .collect();
    (frequencies, responses)
}

fn frequency_to_bin(omega: f64) -> usize {
    ((omega / PI) * (FREQUENCY_POINTS - 1) as f64)
        .round()
        .clamp(0.0, (FREQUENCY_POINTS - 1) as f64) as usize
}

// ── Metric 1: Phase coherence at zone boundaries ──

fn compute_phase_coherence(filter: &[f64]) -> f64 {
    let boundaries = unique_boundary_frequencies();

    // Phase of H(ω) at each zone boundary frequency
    let boundary_phases: Vec<f64> = boundaries
        .iter()
        .map(|&omega| frequency_response(filter, omega).arg())
        .collect();

    // Phase coherence = |mean(e^{jφ})| across boundary points
    // Equals 1 when all phases are identical, 0 when uniformly distributed
    let phasor_sum: Complex64 = boundary_phases
        .iter()
        .map(|&phase| Complex64::new(phase.cos(), phase.sin()))
        .sum();
    let mutual_coherence = (phasor_sum / boundary_phases.len() as f64).norm();

    // Alignment of peak frequency phase with boundary phases
    let (frequencies, responses) = evaluate_frequency_grid(filter);
    let skip = FREQUENCY_POINTS / 20;
    let peak_index = (skip..FREQUENCY_POINTS)
        .max_by(|&a, &b| {
            responses[a]
                .norm()
                .partial_cmp(&responses[b].norm())
                .unwrap()
        })
        .unwrap_or(skip);
    let peak_phase = responses[peak_index].arg();
    let _peak_frequency = frequencies[peak_index];

    let minimum_angular_distance: f64 = boundary_phases
        .iter()
        .map(|&boundary_phase| {
            let difference = (peak_phase - boundary_phase).abs();
            difference.min(2.0 * PI - difference)
        })
        .fold(f64::MAX, f64::min);

    let peak_alignment = 1.0 - minimum_angular_distance / PI;

    0.6 * mutual_coherence + 0.4 * peak_alignment
}

// ── Metric 2: Energy envelope zone fit ──

fn compute_envelope_zone_fit(filter: &[f64]) -> f64 {
    let (_frequencies, responses) = evaluate_frequency_grid(filter);
    let energies: Vec<f64> = responses.iter().map(|response| response.norm_sqr()).collect();
    let total_energy: f64 = energies.iter().sum();

    if total_energy < 1e-15 {
        return 0.0;
    }

    let (dense_start, dense_end) = dense_zone_frequency_range();
    let (mid_start, mid_end) = mid_zone_frequency_range();

    let dense_start_bin = frequency_to_bin(dense_start);
    let dense_end_bin = frequency_to_bin(dense_end);
    let mid_start_bin = frequency_to_bin(mid_start);
    let mid_end_bin = frequency_to_bin(mid_end);

    let dense_energy: f64 = energies[dense_start_bin..=dense_end_bin].iter().sum();
    let mid_energy: f64 = energies[mid_start_bin..=mid_end_bin].iter().sum();
    let zone_coverage = (dense_energy + mid_energy) / total_energy;

    // Transition sharpness at zone boundaries
    let boundary_frequencies = unique_boundary_frequencies();
    let delta_omega = PI / (FREQUENCY_POINTS - 1) as f64;
    let mut transition_sharpness = 0.0;
    let mut boundary_count = 0;

    for &boundary_omega in &boundary_frequencies {
        if boundary_omega <= delta_omega || boundary_omega >= PI - delta_omega {
            continue;
        }
        let energy_before = frequency_response(filter, boundary_omega - delta_omega).norm_sqr();
        let energy_at = frequency_response(filter, boundary_omega).norm_sqr();
        let energy_after = frequency_response(filter, boundary_omega + delta_omega).norm_sqr();

        // Curvature of energy envelope at boundary
        let curvature =
            ((energy_after - 2.0 * energy_at + energy_before) / (delta_omega * delta_omega)).abs();
        let mean_energy_per_bin = total_energy / FREQUENCY_POINTS as f64;
        transition_sharpness += curvature / mean_energy_per_bin.max(1e-15);
        boundary_count += 1;
    }

    if boundary_count > 0 {
        transition_sharpness /= boundary_count as f64;
    }

    let sharpness_score = transition_sharpness / (1.0 + transition_sharpness);
    0.5 * zone_coverage + 0.5 * sharpness_score
}

// ── Metric 3: Boundary vs intra-zone harmonic ratio ──

fn compute_harmonic_ratio(filter: &[f64]) -> f64 {
    let boundary_frequencies = unique_boundary_frequencies();
    let bandwidth = BOUNDARY_BANDWIDTH_FRACTION * PI;

    let (_frequencies, responses) = evaluate_frequency_grid(filter);
    let energies: Vec<f64> = responses.iter().map(|response| response.norm_sqr()).collect();

    let mut boundary_energy = 0.0;
    let mut intra_zone_energy = 0.0;

    for i in 0..FREQUENCY_POINTS {
        let omega = PI * i as f64 / (FREQUENCY_POINTS - 1) as f64;
        let at_boundary = boundary_frequencies
            .iter()
            .any(|&boundary_omega| (omega - boundary_omega).abs() <= bandwidth);

        if at_boundary {
            boundary_energy += energies[i];
        } else {
            intra_zone_energy += energies[i];
        }
    }

    if intra_zone_energy < 1e-15 {
        return 0.0;
    }

    let log_ratio = (boundary_energy / intra_zone_energy).ln();

    // Normalize to non-negative score (higher = better separation)
    if log_ratio > 0.0 {
        log_ratio / (1.0 + log_ratio)
    } else {
        (log_ratio + 2.0).max(0.0) / (3.0 + (log_ratio + 2.0).max(0.0))
    }
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

    let input_content =
        std::fs::read_to_string(&arguments[1]).expect("Failed to read input JSON");
    let config: Config =
        serde_json::from_str(&input_content).expect("Failed to parse input config");

    assert!(
        config.wavelet.filter_taps >= 4 && config.wavelet.filter_taps % 2 == 0,
        "filter_taps must be even and >= 4"
    );

    let filter = construct_filter(&config.wavelet);

    let phase_coherence = compute_phase_coherence(&filter);
    let envelope_zone_fit = compute_envelope_zone_fit(&filter);
    let harmonic_ratio = compute_harmonic_ratio(&filter);

    let output = serde_json::json!({
        "angle_zone_phase_coherence": phase_coherence,
        "angle_frequency_envelope_zone_fit": envelope_zone_fit,
        "angle_zone_boundary_harmonic_ratio": harmonic_ratio,
    });

    std::fs::write(
        &arguments[2],
        serde_json::to_string_pretty(&output).expect("Failed to serialize output"),
    )
    .expect("Failed to write output JSON");
}
