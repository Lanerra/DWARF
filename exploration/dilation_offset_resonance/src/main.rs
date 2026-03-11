use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10;

// ── DSQG zone boundaries ──

const ZONE_BOUNDARIES: [f64; 4] = [48.0, 96.0, 128.0, 384.0];
const DENSE_ZONE_UPPER: f64 = 48.0;

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

// ── Output struct ──

#[derive(Serialize)]
struct Output {
    zone_boundary_scale_proximity: f64,
    offset_scale_harmonic_ratio: f64,
    dense_zone_scale_coverage_uniformity: f64,
}

// ── Scale generation ──
// scale[k] = θ^k for k = 0..N-1

fn generate_scales(theta: f64, count: usize) -> Vec<f64> {
    (0..count).map(|k| theta.powi(k as i32)).collect()
}

// ── Metric 1: zone_boundary_scale_proximity ──
// For each zone boundary, find closest scale in log-space and sum inverse distances.

fn compute_zone_boundary_scale_proximity(scales: &[f64]) -> f64 {
    let log_scales: Vec<f64> = scales.iter().map(|s| s.ln()).collect();

    ZONE_BOUNDARIES
        .iter()
        .map(|&boundary| {
            let log_boundary = boundary.ln();
            let minimum_distance = log_scales
                .iter()
                .map(|&log_scale| (log_scale - log_boundary).abs())
                .fold(f64::INFINITY, f64::min);
            1.0 / (minimum_distance + EPSILON)
        })
        .sum()
}

// ── Metric 2: offset_scale_harmonic_ratio ──
// Measure whether scale ratio θ forms simple rational relationships with offset ratios.

fn compute_offset_scale_harmonic_ratio(theta: f64) -> f64 {
    let log_theta = theta.ln();
    if log_theta.abs() < EPSILON {
        return 0.0;
    }

    // Zone boundary ratios relative to dense zone upper bound
    let zone_ratios: [f64; 3] = [
        96.0 / 48.0,  // = 2.0
        128.0 / 48.0, // ≈ 2.667
        384.0 / 48.0, // = 8.0
    ];

    let mut total_alignment = 0.0;
    for &ratio in &zone_ratios {
        let log_ratio = ratio.ln();

        // How many scale steps θ^k fit into this ratio?
        let continuous_steps = log_ratio / log_theta;

        let best_alignment = best_rational_alignment(continuous_steps);
        total_alignment += best_alignment;
    }

    total_alignment / zone_ratios.len() as f64
}

// Find how close a value is to a simple rational number p/q (q ≤ 8).
// Returns alignment score in [0, 1] where 1 = exact rational match.
fn best_rational_alignment(value: f64) -> f64 {
    let mut minimum_deviation = f64::INFINITY;

    for denominator in 1..=8 {
        let nearest_numerator = (value * denominator as f64).round();
        let rational_value = nearest_numerator / denominator as f64;
        let deviation = (value - rational_value).abs();
        minimum_deviation = minimum_deviation.min(deviation);
    }

    // Convert deviation to alignment: 0 deviation → 1.0, large deviation → 0.0
    (-minimum_deviation * 4.0).exp()
}

// ── Metric 3: dense_zone_scale_coverage_uniformity ──
// Map scales into [0, log(48)] and measure variance of positions.

fn compute_dense_zone_scale_coverage_uniformity(scales: &[f64]) -> f64 {
    let log_upper = DENSE_ZONE_UPPER.ln();

    let positions: Vec<f64> = scales
        .iter()
        .map(|&scale| scale.ln().clamp(0.0, log_upper))
        .collect();

    let count = positions.len() as f64;
    if count < 2.0 {
        return 0.0;
    }

    let mean = positions.iter().sum::<f64>() / count;
    let variance = positions
        .iter()
        .map(|&position| (position - mean).powi(2))
        .sum::<f64>()
        / count;

    variance
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

    let theta = config.wavelet.lattice_angles[0];
    let tap_count = config.wavelet.filter_taps;

    let scales = generate_scales(theta, tap_count);

    let zone_boundary_scale_proximity = compute_zone_boundary_scale_proximity(&scales);
    let offset_scale_harmonic_ratio = compute_offset_scale_harmonic_ratio(theta);
    let dense_zone_scale_coverage_uniformity =
        compute_dense_zone_scale_coverage_uniformity(&scales);

    let output = Output {
        zone_boundary_scale_proximity,
        offset_scale_harmonic_ratio,
        dense_zone_scale_coverage_uniformity,
    };

    let output_json = serde_json::to_string_pretty(&output)
        .unwrap_or_else(|error| panic!("Failed to serialize output: {}", error));
    std::fs::write(output_path, &output_json)
        .unwrap_or_else(|error| panic!("Failed to write {}: {}", output_path, error));

    eprintln!(
        "✅ θ={:.4} taps={} | proximity={:.4} | harmonic={:.4} | uniformity={:.6}",
        theta,
        tap_count,
        zone_boundary_scale_proximity,
        offset_scale_harmonic_ratio,
        dense_zone_scale_coverage_uniformity,
    );
}
