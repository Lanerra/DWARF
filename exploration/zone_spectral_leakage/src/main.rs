use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

const SIGNAL_LENGTH: usize = 2048;
const DWT_LEVELS: usize = 11;
const BOUNDARY_SAMPLE_COUNT: usize = 5;
const UTILIZATION_THRESHOLD: f64 = 0.10;
const PHASE_SAMPLE_COUNT: usize = 200;

// ── Zone DWT level assignments ──
// Level l captures offsets with δ ≈ 2^l (via primary_level = floor(log2(δ)))
// Dense zone (δ=1..48): levels 0-5
// Mid zone (δ=96,128): levels 6-7
// Long zone (δ=384): levels 8-10 + approximation

fn zone_level_set(zone: usize) -> &'static [usize] {
    match zone {
        0 => &[0, 1, 2, 3, 4, 5],
        1 => &[6, 7],
        2 => &[8, 9, 10],
        _ => unreachable!(),
    }
}

fn zone_includes_approximation(zone: usize) -> bool {
    zone == 2
}

fn dsqg_dense_offsets() -> Vec<usize> {
    (1..=48).collect()
}

fn dsqg_mid_offsets() -> Vec<usize> {
    vec![96, 128]
}

fn dsqg_long_offsets() -> Vec<usize> {
    vec![384]
}

fn all_zone_offsets() -> [Vec<usize>; 3] {
    [dsqg_dense_offsets(), dsqg_mid_offsets(), dsqg_long_offsets()]
}

// ── Config / Output structs ──

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

#[derive(Serialize)]
struct Output {
    zone_spectral_retention_rate: f64,
    boundary_leakage_asymmetry: f64,
    cross_zone_phase_continuity: f64,
    leakage_to_coverage_ratio: f64,
    zone_frequency_utilization_efficiency: f64,
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

fn qmf_highpass(lowpass: &[f64]) -> Vec<f64> {
    let length = lowpass.len();
    (0..length)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * lowpass[length - 1 - k]
        })
        .collect()
}

// ── DWT forward ──

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

struct DwtResult {
    approximation: Vec<f64>,
    details: Vec<Vec<f64>>,
}

fn dwt_forward(signal: &[f64], lowpass: &[f64], highpass: &[f64], levels: usize) -> DwtResult {
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
    }
}

fn signal_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

// ── DWT energy analysis for a single offset ──

struct OffsetEnergyProfile {
    level_energies: Vec<f64>,
    approximation_energy: f64,
    total_energy: f64,
}

fn compute_offset_energy(
    offset: usize,
    lowpass: &[f64],
    highpass: &[f64],
) -> OffsetEnergyProfile {
    assert!(offset < SIGNAL_LENGTH);
    let mut signal = vec![0.0; SIGNAL_LENGTH];
    signal[offset] = 1.0;
    let result = dwt_forward(&signal, lowpass, highpass, DWT_LEVELS);
    let level_energies: Vec<f64> = result.details.iter().map(|d| signal_energy(d)).collect();
    let approximation_energy = signal_energy(&result.approximation);
    let total_energy: f64 = level_energies.iter().sum::<f64>() + approximation_energy;
    OffsetEnergyProfile {
        level_energies,
        approximation_energy,
        total_energy,
    }
}

fn zone_retained_energy(profile: &OffsetEnergyProfile, zone: usize) -> f64 {
    let levels = zone_level_set(zone);
    let mut energy = 0.0;
    for &level in levels {
        if level < profile.level_energies.len() {
            energy += profile.level_energies[level];
        }
    }
    if zone_includes_approximation(zone) {
        energy += profile.approximation_energy;
    }
    energy
}

// ── Metric 1: zone_spectral_retention_rate ──
// For each zone, fraction of its offsets' energy that stays in the zone's DWT levels.
// Higher = better.

fn compute_zone_spectral_retention_rate(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let zones = all_zone_offsets();
    let mut zone_scores = Vec::new();

    for (zone_index, offsets) in zones.iter().enumerate() {
        let mut retention_sum = 0.0;
        let mut count = 0;
        for &delta in offsets {
            if delta >= SIGNAL_LENGTH {
                continue;
            }
            let profile = compute_offset_energy(delta, lowpass, highpass);
            if profile.total_energy < 1e-15 {
                continue;
            }
            let retained = zone_retained_energy(&profile, zone_index);
            retention_sum += retained / profile.total_energy;
            count += 1;
        }
        if count > 0 {
            zone_scores.push(retention_sum / count as f64);
        }
    }

    if zone_scores.is_empty() {
        return 0.0;
    }
    zone_scores.iter().sum::<f64>() / zone_scores.len() as f64
}

// ── Metric 2: boundary_leakage_asymmetry ──
// For each zone, compare leakage at lower-δ boundary (high freq) vs upper-δ boundary (low freq).
// Lower = better (symmetric leakage preferred).

fn compute_boundary_leakage_asymmetry(lowpass: &[f64], highpass: &[f64]) -> f64 {
    struct BoundarySpec {
        zone_index: usize,
        upper_deltas: Vec<usize>,
        lower_deltas: Vec<usize>,
    }

    let boundaries = [
        BoundarySpec {
            zone_index: 0,
            upper_deltas: (1..=BOUNDARY_SAMPLE_COUNT).collect(),
            lower_deltas: (44..=48).collect(),
        },
        BoundarySpec {
            zone_index: 1,
            upper_deltas: (96..=100).collect(),
            lower_deltas: (124..=128).collect(),
        },
        BoundarySpec {
            zone_index: 2,
            upper_deltas: (384..=388).collect(),
            lower_deltas: (508..=512).collect(),
        },
    ];

    let compute_mean_leakage =
        |deltas: &[usize], zone: usize, lowpass: &[f64], highpass: &[f64]| -> f64 {
            let mut leakage_sum = 0.0;
            let mut count = 0;
            for &delta in deltas {
                if delta >= SIGNAL_LENGTH {
                    continue;
                }
                let profile = compute_offset_energy(delta, lowpass, highpass);
                if profile.total_energy < 1e-15 {
                    continue;
                }
                let retained = zone_retained_energy(&profile, zone);
                leakage_sum += 1.0 - retained / profile.total_energy;
                count += 1;
            }
            if count > 0 {
                leakage_sum / count as f64
            } else {
                0.0
            }
        };

    let mut asymmetry_sum = 0.0;
    let mut zone_count = 0;

    for spec in &boundaries {
        let upper_leakage = compute_mean_leakage(
            &spec.upper_deltas,
            spec.zone_index,
            lowpass,
            highpass,
        );
        let lower_leakage = compute_mean_leakage(
            &spec.lower_deltas,
            spec.zone_index,
            lowpass,
            highpass,
        );
        let denominator = upper_leakage + lower_leakage;
        if denominator > 1e-15 {
            asymmetry_sum += (upper_leakage - lower_leakage).abs() / denominator;
            zone_count += 1;
        }
    }

    if zone_count > 0 {
        asymmetry_sum / zone_count as f64
    } else {
        0.0
    }
}

// ── Metric 3: cross_zone_phase_continuity ──
// Phase smoothness at zone boundary frequencies via iterated filter bank.
// Higher = better (smoother phase transitions).

fn filter_response_complex(filter: &[f64], omega: f64) -> (f64, f64) {
    let mut real = 0.0;
    let mut imag = 0.0;
    for (n, &h) in filter.iter().enumerate() {
        let phase = omega * n as f64;
        real += h * phase.cos();
        imag -= h * phase.sin();
    }
    (real, imag)
}

fn complex_multiply(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn level_response_complex(
    lowpass: &[f64],
    highpass: &[f64],
    level: usize,
    omega: f64,
) -> (f64, f64) {
    let mut product = (1.0, 0.0);
    for k in 0..level {
        let scale = (1u64 << k) as f64;
        let response = filter_response_complex(lowpass, omega * scale);
        product = complex_multiply(product, response);
    }
    let scale = (1u64 << level) as f64;
    let highpass_response = filter_response_complex(highpass, omega * scale);
    complex_multiply(product, highpass_response)
}

fn dominant_level_phase(lowpass: &[f64], highpass: &[f64], omega: f64) -> f64 {
    let mut max_power = 0.0;
    let mut best_phase = 0.0;
    for level in 0..DWT_LEVELS {
        let (real, imag) = level_response_complex(lowpass, highpass, level, omega);
        let power = real * real + imag * imag;
        if power > max_power {
            max_power = power;
            best_phase = imag.atan2(real);
        }
    }
    best_phase
}

fn unwrap_phase(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return Vec::new();
    }
    let mut unwrapped = vec![phases[0]];
    for i in 1..phases.len() {
        let mut diff = phases[i] - phases[i - 1];
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        unwrapped.push(unwrapped[i - 1] + diff);
    }
    unwrapped
}

fn compute_cross_zone_phase_continuity(lowpass: &[f64], highpass: &[f64]) -> f64 {
    // Zone boundary gaps in δ-space: dense↔mid (δ 48..96), mid↔long (δ 128..384)
    let boundary_gaps: [(f64, f64); 2] = [(48.0, 96.0), (128.0, 384.0)];

    let mut continuity_scores = Vec::new();

    for &(delta_inside, delta_outside) in &boundary_gaps {
        let omega_low = 2.0 * PI / delta_outside;
        let omega_high = 2.0 * PI / delta_inside;

        let mut phases = Vec::with_capacity(PHASE_SAMPLE_COUNT);
        for i in 0..PHASE_SAMPLE_COUNT {
            let t = i as f64 / (PHASE_SAMPLE_COUNT - 1) as f64;
            let omega = omega_low + t * (omega_high - omega_low);
            phases.push(dominant_level_phase(lowpass, highpass, omega));
        }

        let unwrapped = unwrap_phase(&phases);
        if unwrapped.len() < 2 {
            continue;
        }

        let gradients: Vec<f64> = unwrapped.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let mean_gradient = gradients.iter().sum::<f64>() / gradients.len() as f64;
        let variance = gradients
            .iter()
            .map(|&g| (g - mean_gradient) * (g - mean_gradient))
            .sum::<f64>()
            / gradients.len() as f64;

        let smoothness = 1.0 / (1.0 + variance.sqrt());
        continuity_scores.push(smoothness);
    }

    if continuity_scores.is_empty() {
        return 0.0;
    }
    continuity_scores.iter().sum::<f64>() / continuity_scores.len() as f64
}

// ── Metric 4: leakage_to_coverage_ratio ──
// Total spectral power leaking out of dense zone / total power assigned to dense zone.
// Lower = better.

fn compute_leakage_to_coverage_ratio(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let dense_offsets = dsqg_dense_offsets();
    let mut total_leaked = 0.0;
    let mut total_retained = 0.0;

    for &delta in &dense_offsets {
        if delta >= SIGNAL_LENGTH {
            continue;
        }
        let profile = compute_offset_energy(delta, lowpass, highpass);
        if profile.total_energy < 1e-15 {
            continue;
        }
        let retained = zone_retained_energy(&profile, 0);
        total_leaked += profile.total_energy - retained;
        total_retained += retained;
    }

    if total_retained > 1e-15 {
        total_leaked / total_retained
    } else {
        f64::MAX
    }
}

// ── Metric 5: zone_frequency_utilization_efficiency ──
// For each zone's full δ-range, fraction of positions where the dominant DWT level
// captures >10% of total energy (vs. being diffusely spread = "dead").
// Higher = better.

fn utilization_range(zone: usize) -> std::ops::RangeInclusive<usize> {
    match zone {
        0 => 1..=48,
        1 => 96..=128,
        2 => 384..=512,
        _ => unreachable!(),
    }
}

fn compute_zone_frequency_utilization_efficiency(lowpass: &[f64], highpass: &[f64]) -> f64 {
    let mut zone_scores = Vec::new();

    for zone in 0..3_usize {
        let range = utilization_range(zone);
        let mut active = 0_usize;
        let mut total = 0_usize;

        for delta in range {
            if delta >= SIGNAL_LENGTH {
                continue;
            }
            let profile = compute_offset_energy(delta, lowpass, highpass);
            if profile.total_energy < 1e-15 {
                continue;
            }

            let max_fraction = profile
                .level_energies
                .iter()
                .chain(std::iter::once(&profile.approximation_energy))
                .map(|&energy| energy / profile.total_energy)
                .fold(0.0_f64, f64::max);

            if max_fraction > UTILIZATION_THRESHOLD {
                active += 1;
            }
            total += 1;
        }

        if total > 0 {
            zone_scores.push(active as f64 / total as f64);
        }
    }

    if zone_scores.is_empty() {
        return 0.0;
    }
    zone_scores.iter().sum::<f64>() / zone_scores.len() as f64
}

// ── Main ──

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "Usage: zone_spectral_leakage <input.json> <output.json>"
    );

    let input_path = &args[1];
    let output_path = &args[2];

    let config_string =
        std::fs::read_to_string(input_path).expect("Failed to read input JSON");
    let config: Config =
        serde_json::from_str(&config_string).expect("Failed to parse input JSON");

    let lowpass = construct_filter(&config.wavelet);
    let highpass = qmf_highpass(&lowpass);

    let zone_spectral_retention_rate =
        compute_zone_spectral_retention_rate(&lowpass, &highpass);
    let boundary_leakage_asymmetry =
        compute_boundary_leakage_asymmetry(&lowpass, &highpass);
    let cross_zone_phase_continuity =
        compute_cross_zone_phase_continuity(&lowpass, &highpass);
    let leakage_to_coverage_ratio =
        compute_leakage_to_coverage_ratio(&lowpass, &highpass);
    let zone_frequency_utilization_efficiency =
        compute_zone_frequency_utilization_efficiency(&lowpass, &highpass);

    let output = Output {
        zone_spectral_retention_rate,
        boundary_leakage_asymmetry,
        cross_zone_phase_continuity,
        leakage_to_coverage_ratio,
        zone_frequency_utilization_efficiency,
    };

    let output_json =
        serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    std::fs::write(output_path, &output_json).expect("Failed to write output JSON");

    eprintln!(
        "retention={:.4} asymmetry={:.4} phase_continuity={:.4} leak_ratio={:.4} utilization={:.4}",
        zone_spectral_retention_rate,
        boundary_leakage_asymmetry,
        cross_zone_phase_continuity,
        leakage_to_coverage_ratio,
        zone_frequency_utilization_efficiency,
    );
}
