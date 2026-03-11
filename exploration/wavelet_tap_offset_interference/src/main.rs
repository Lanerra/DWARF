use serde::Deserialize;
use std::collections::BTreeMap;
use std::f64::consts::{PI, SQRT_2};

const DEAD_ZONE_THRESHOLD_FRACTION: f64 = 0.3;

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

// ── Frequency response at a single frequency ──
// H(ω) = Σ_{k=0}^{N-1} h[k] · e^{-jωk}

fn frequency_response_magnitude(filter: &[f64], omega: f64) -> f64 {
    let mut real = 0.0;
    let mut imaginary = 0.0;
    for (k, &h_k) in filter.iter().enumerate() {
        let angle = -omega * k as f64;
        real += h_k * angle.cos();
        imaginary += h_k * angle.sin();
    }
    (real * real + imaginary * imaginary).sqrt()
}

// ── Per-tap phase coherence ──
// Each tap k contributes with phase arg(h[k] · e^{-jωk}).
// Coherence = resultant vector length of unit-phase contributions / active tap count.

fn phase_coherence_at_frequency(filter: &[f64], omega: f64) -> f64 {
    let mut resultant_real = 0.0;
    let mut resultant_imaginary = 0.0;
    let mut active_tap_count = 0;

    for (k, &h_k) in filter.iter().enumerate() {
        if h_k.abs() < 1e-15 {
            continue;
        }
        let tap_phase = if h_k > 0.0 {
            -omega * k as f64
        } else {
            PI - omega * k as f64
        };
        resultant_real += tap_phase.cos();
        resultant_imaginary += tap_phase.sin();
        active_tap_count += 1;
    }

    if active_tap_count == 0 {
        return 0.0;
    }

    let resultant_length =
        (resultant_real * resultant_real + resultant_imaginary * resultant_imaginary).sqrt();
    resultant_length / active_tap_count as f64
}

// ── Per-offset metric bundle ──

struct OffsetMetrics {
    reinforcement_factor: f64,
    phase_coherence: f64,
}

fn compute_offset_metrics(filter: &[f64], offset: usize) -> OffsetMetrics {
    let omega = 2.0 * PI / offset as f64;
    OffsetMetrics {
        reinforcement_factor: frequency_response_magnitude(filter, omega),
        phase_coherence: phase_coherence_at_frequency(filter, omega),
    }
}

// ── Zone averaging ──

fn zone_averages(
    all_metrics: &[(usize, OffsetMetrics)],
    zone_offsets: &[usize],
) -> (f64, f64) {
    let zone: Vec<&OffsetMetrics> = all_metrics
        .iter()
        .filter(|(offset, _)| zone_offsets.contains(offset))
        .map(|(_, metrics)| metrics)
        .collect();

    if zone.is_empty() {
        return (0.0, 0.0);
    }

    let count = zone.len() as f64;
    let average_reinforcement =
        zone.iter().map(|m| m.reinforcement_factor).sum::<f64>() / count;
    let average_coherence =
        zone.iter().map(|m| m.phase_coherence).sum::<f64>() / count;

    (average_reinforcement, average_coherence)
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

    assert!(
        !config.wavelet.lattice_angles.is_empty(),
        "lattice_angles must contain at least one angle"
    );

    let filter = construct_filter(&config.wavelet);

    let offsets = all_dsqg_offsets();
    let all_metrics: Vec<(usize, OffsetMetrics)> = offsets
        .iter()
        .map(|&offset| (offset, compute_offset_metrics(&filter, offset)))
        .collect();

    let total_count = all_metrics.len() as f64;
    let global_reinforcement = all_metrics
        .iter()
        .map(|(_, m)| m.reinforcement_factor)
        .sum::<f64>()
        / total_count;
    let global_coherence = all_metrics
        .iter()
        .map(|(_, m)| m.phase_coherence)
        .sum::<f64>()
        / total_count;

    let max_reinforcement = all_metrics
        .iter()
        .map(|(_, m)| m.reinforcement_factor)
        .fold(f64::NEG_INFINITY, f64::max);
    let dead_zone_threshold = DEAD_ZONE_THRESHOLD_FRACTION * max_reinforcement;
    let dead_zone_count = all_metrics
        .iter()
        .filter(|(_, m)| m.reinforcement_factor < dead_zone_threshold)
        .count();

    let dense = dsqg_dense_offsets();
    let mid = dsqg_mid_offsets();
    let long = dsqg_long_offsets();

    let (dense_reinforcement, dense_coherence) = zone_averages(&all_metrics, &dense);
    let (mid_reinforcement, mid_coherence) = zone_averages(&all_metrics, &mid);
    let (long_reinforcement, long_coherence) = zone_averages(&all_metrics, &long);

    let mut output: BTreeMap<String, f64> = BTreeMap::new();
    output.insert(
        "offset_tap_reinforcement_factor".into(),
        global_reinforcement,
    );
    output.insert("offset_phase_coherence".into(), global_coherence);
    output.insert("dead_zone_penalty".into(), dead_zone_count as f64);
    output.insert("dense_zone_reinforcement".into(), dense_reinforcement);
    output.insert("dense_zone_phase_coherence".into(), dense_coherence);
    output.insert("mid_zone_reinforcement".into(), mid_reinforcement);
    output.insert("mid_zone_phase_coherence".into(), mid_coherence);
    output.insert("long_zone_reinforcement".into(), long_reinforcement);
    output.insert("long_zone_phase_coherence".into(), long_coherence);
    output.insert(
        "dead_zone_fraction".into(),
        dead_zone_count as f64 / total_count,
    );
    output.insert("max_reinforcement".into(), max_reinforcement);

    let output_json = serde_json::to_string_pretty(&output)
        .unwrap_or_else(|error| panic!("Failed to serialize output: {}", error));
    std::fs::write(output_path, &output_json)
        .unwrap_or_else(|error| panic!("Failed to write {}: {}", output_path, error));

    eprintln!(
        "✅ taps={} | reinforcement={:.4} | coherence={:.4} | dead_zones={}",
        config.wavelet.filter_taps, global_reinforcement, global_coherence, dead_zone_count,
    );
}
