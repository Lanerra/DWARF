use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::fs;
use std::io::Write;

const D: usize = 256;
const H: usize = 8;
const HEAD_DIM: usize = D / H;
const J: usize = 52;
const TOP_K: usize = 5;
const ENTROPY_SAMPLES: usize = 10_000;
const HEAD_DIFF_PAIRS: usize = 1000;
const SENSITIVITY_SAMPLES: usize = 1000;
const ENTROPY_BINS: usize = 100;
const SOFTMAX_TEMPERATURE: f64 = 0.1;

// Sparse offset indices within the J=52 set (offsets 49,50,51 = delta 96,128,384)
const SPARSE_OFFSET_START: usize = 48;

// ── Scale Embed Sampling ──
// Calibrated from trained d41_35m statistics:
//   base σ=0.15, sparse offsets 2× magnitude, h0 1.5×, h7 0.6×

fn sample_scale_embed(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let base_distribution = Normal::new(0.0, 0.15).unwrap();
    let mut scale_embed = vec![vec![0.0; HEAD_DIM]; J];
    for j in 0..J {
        for d in 0..HEAD_DIM {
            scale_embed[j][d] = base_distribution.sample(rng);
        }
        if j >= SPARSE_OFFSET_START {
            for d in 0..HEAD_DIM {
                scale_embed[j][d] *= 2.0;
            }
        }
    }
    scale_embed
}

fn sample_scale_embed_calibrated(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut embed = sample_scale_embed(rng);
    // h0 (global head) has higher magnitude from IF gain 1.12
    // h7 (local head) has lower magnitude from IF gain 0.93
    // Since scale_embed is J×head_dim (not per-head partitioned),
    // we model this by generating separate per-head magnitude scalars
    // that the signal extraction functions use.
    let _ = &mut embed;
    embed
}

// Per-head magnitude scalars (from IF gains)
fn head_magnitude_scalars() -> [f64; H] {
    // h0=1.5 (global, high sparse), h7=0.6 (local, high dense)
    // intermediate heads interpolate
    let mut scalars = [1.0; H];
    scalars[0] = 1.5;
    scalars[1] = 1.3;
    scalars[2] = 1.15;
    scalars[3] = 1.05;
    scalars[4] = 1.0;
    scalars[5] = 0.9;
    scalars[6] = 0.75;
    scalars[7] = 0.6;
    scalars
}

// ── Helper: absolute mean of a flat slice ──
fn abs_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64
}

fn abs_max(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
}

fn flatten_embed(embed: &[Vec<f64>]) -> Vec<f64> {
    embed.iter().flat_map(|row| row.iter().copied()).collect()
}

// ── Signal Extraction Functions ──

// Signal A: Scalar Global Mean
fn signal_a(embed: &[Vec<f64>]) -> Vec<f64> {
    let flat = flatten_embed(embed);
    vec![abs_mean(&flat)]
}

// Signal B: Scalar Global Max
fn signal_b(embed: &[Vec<f64>]) -> Vec<f64> {
    let flat = flatten_embed(embed);
    vec![abs_max(&flat)]
}

// Signal C: Per-Head Vector (H=8 values)
// For each head h, compute mean abs of scale_embed weighted by head magnitude
fn signal_c(embed: &[Vec<f64>]) -> Vec<f64> {
    let scalars = head_magnitude_scalars();
    let mut result = vec![0.0; H];
    for h in 0..H {
        let mut sum = 0.0;
        let mut count = 0;
        for j in 0..J {
            for d in 0..HEAD_DIM {
                sum += (embed[j][d] * scalars[h]).abs();
                count += 1;
            }
        }
        result[h] = sum / count as f64;
    }
    result
}

// Signal D: Per-Offset Sparse (top-K=5 highest magnitude offsets)
fn signal_d(embed: &[Vec<f64>]) -> Vec<f64> {
    let mut offset_magnitudes: Vec<(usize, f64)> = (0..J)
        .map(|j| (j, abs_mean(&embed[j])))
        .collect();
    offset_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    offset_magnitudes
        .iter()
        .take(TOP_K)
        .map(|(_, magnitude)| *magnitude)
        .collect()
}

// Signal E: Softmax-Sharpened Mean
fn signal_e(embed: &[Vec<f64>]) -> Vec<f64> {
    let offset_means: Vec<f64> = (0..J).map(|j| abs_mean(&embed[j])).collect();
    let scaled: Vec<f64> = offset_means.iter().map(|v| v / SOFTMAX_TEMPERATURE).collect();
    let max_scaled = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = scaled.iter().map(|v| (v - max_scaled).exp()).collect();
    let sum_exp: f64 = exp_values.iter().sum();
    let weights: Vec<f64> = exp_values.iter().map(|v| v / sum_exp).collect();
    let weighted_sum: f64 = weights
        .iter()
        .zip(offset_means.iter())
        .map(|(w, m)| w * m)
        .sum();
    vec![weighted_sum]
}

type SignalFunction = fn(&[Vec<f64>]) -> Vec<f64>;

struct SignalVariant {
    name: &'static str,
    label: &'static str,
    extra_parameters: usize,
    signal_function: SignalFunction,
    dimensionality: usize,
}

fn signal_variants() -> Vec<SignalVariant> {
    vec![
        SignalVariant {
            name: "Scalar Global Mean",
            label: "A(mean)",
            extra_parameters: 2 * D,
            signal_function: signal_a,
            dimensionality: 1,
        },
        SignalVariant {
            name: "Scalar Global Max",
            label: "B(max)",
            extra_parameters: 2 * D,
            signal_function: signal_b,
            dimensionality: 1,
        },
        SignalVariant {
            name: "Per-Head Vector",
            label: "C(head)",
            extra_parameters: 2 * D,
            signal_function: signal_c,
            dimensionality: H,
        },
        SignalVariant {
            name: "Per-Offset Sparse",
            label: "D(top-K)",
            extra_parameters: 2 * D,
            signal_function: signal_d,
            dimensionality: TOP_K,
        },
        SignalVariant {
            name: "Softmax-Sharpened",
            label: "E(sharp)",
            extra_parameters: 2 * D,
            signal_function: signal_e,
            dimensionality: 1,
        },
    ]
}

// ── Metric 1: Information Content (Entropy) ──
// For scalar signals: 1D histogram with ENTROPY_BINS bins
// For vector signals: sum of marginal entropies

fn empirical_entropy_1d(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_value - min_value;
    if range < 1e-12 {
        return 0.0;
    }
    let mut bins = vec![0usize; ENTROPY_BINS];
    for &value in values {
        let bin = ((value - min_value) / range * (ENTROPY_BINS as f64 - 1.0)) as usize;
        bins[bin.min(ENTROPY_BINS - 1)] += 1;
    }
    let total = values.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let probability = count as f64 / total;
            entropy -= probability * probability.log2();
        }
    }
    entropy
}

fn compute_entropy(
    variant: &SignalVariant,
    samples: &[Vec<Vec<f64>>],
) -> f64 {
    let signals: Vec<Vec<f64>> = samples
        .iter()
        .map(|embed| (variant.signal_function)(embed))
        .collect();

    if variant.dimensionality == 1 {
        let values: Vec<f64> = signals.iter().map(|s| s[0]).collect();
        empirical_entropy_1d(&values)
    } else {
        let mut total_entropy = 0.0;
        for dimension in 0..variant.dimensionality {
            let values: Vec<f64> = signals.iter().map(|s| s[dimension]).collect();
            total_entropy += empirical_entropy_1d(&values);
        }
        total_entropy
    }
}

// ── Metric 2: Head Differentiation Capture ──

fn sample_h0_dominant(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut embed = sample_scale_embed(rng);
    // Amplify all values by h0 scalar (global head dominant)
    for j in 0..J {
        for d in 0..HEAD_DIM {
            embed[j][d] *= 1.5;
        }
    }
    // Suppress dense offsets (h7 local pattern)
    for j in 0..SPARSE_OFFSET_START.min(J) {
        for d in 0..HEAD_DIM {
            embed[j][d] *= 0.3;
        }
    }
    embed
}

fn sample_h7_dominant(rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut embed = sample_scale_embed(rng);
    // Amplify dense offsets (h7 local head dominant)
    for j in 0..SPARSE_OFFSET_START.min(J) {
        for d in 0..HEAD_DIM {
            embed[j][d] *= 1.8;
        }
    }
    // Suppress sparse offsets
    for j in SPARSE_OFFSET_START..J {
        for d in 0..HEAD_DIM {
            embed[j][d] *= 0.2;
        }
    }
    embed
}

fn compute_head_differentiation(
    variant: &SignalVariant,
    rng: &mut StdRng,
) -> f64 {
    let mut total_distance = 0.0;
    for _ in 0..HEAD_DIFF_PAIRS {
        let h0_embed = sample_h0_dominant(rng);
        let h7_embed = sample_h7_dominant(rng);
        let signal_h0 = (variant.signal_function)(&h0_embed);
        let signal_h7 = (variant.signal_function)(&h7_embed);
        let squared_distance: f64 = signal_h0
            .iter()
            .zip(signal_h7.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        total_distance += squared_distance;
    }
    total_distance / HEAD_DIFF_PAIRS as f64
}

// ── Metric 3: Sensitivity in Operating Range ──
// Compute ∂(FiLM_output)/∂s numerically for each signal value
// FiLM: gamma = W_gamma * s + 1.0, output = gamma * ffn_out + W_beta * s
// ∂output/∂s = W_gamma * ffn_out + W_beta
// For vector signals: Jacobian Frobenius norm

fn compute_sensitivity(
    variant: &SignalVariant,
    rng: &mut StdRng,
) -> (f64, f64) {
    let epsilon = 1e-5;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut sensitivities = Vec::with_capacity(SENSITIVITY_SAMPLES);

    // Generate random W_gamma, W_beta, and ffn_output
    let weight_gamma: Vec<f64> = (0..D).map(|_| normal.sample(rng) * 0.01).collect();
    let weight_beta: Vec<f64> = (0..D).map(|_| normal.sample(rng) * 0.01).collect();

    for _ in 0..SENSITIVITY_SAMPLES {
        let embed = sample_scale_embed_calibrated(rng);
        let signal = (variant.signal_function)(&embed);

        // ffn_output ~ N(0,1) for each of D dims
        let ffn_output: Vec<f64> = (0..D).map(|_| normal.sample(rng)).collect();

        // Compute FiLM output at signal s
        let film_output = compute_film_output(&signal, &weight_gamma, &weight_beta, &ffn_output, variant.dimensionality);

        // Perturb each signal dimension and compute Jacobian
        let mut jacobian_frobenius_squared = 0.0;
        for dim in 0..variant.dimensionality {
            let mut signal_plus = signal.clone();
            signal_plus[dim] += epsilon;

            let film_output_plus = compute_film_output(
                &signal_plus,
                &weight_gamma,
                &weight_beta,
                &ffn_output,
                variant.dimensionality,
            );

            for output_dim in 0..D {
                let derivative = (film_output_plus[output_dim] - film_output[output_dim]) / epsilon;
                jacobian_frobenius_squared += derivative * derivative;
            }
        }
        sensitivities.push(jacobian_frobenius_squared.sqrt());
    }

    let mean_sensitivity = sensitivities.iter().sum::<f64>() / sensitivities.len() as f64;
    let variance = sensitivities
        .iter()
        .map(|s| (s - mean_sensitivity) * (s - mean_sensitivity))
        .sum::<f64>()
        / sensitivities.len() as f64;
    (mean_sensitivity, variance.sqrt())
}

fn compute_film_output(
    signal: &[f64],
    weight_gamma: &[f64],
    weight_beta: &[f64],
    ffn_output: &[f64],
    signal_dimensionality: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; D];

    if signal_dimensionality == 1 {
        let s = signal[0];
        for i in 0..D {
            let gamma = weight_gamma[i] * s + 1.0;
            let beta = weight_beta[i] * s;
            output[i] = gamma * ffn_output[i] + beta;
        }
    } else if signal_dimensionality == H {
        // Per-head: each head h conditions dims [h*HEAD_DIM..(h+1)*HEAD_DIM]
        for h in 0..H {
            let s_h = signal[h];
            let start = h * HEAD_DIM;
            let end = start + HEAD_DIM;
            for i in start..end {
                let gamma = weight_gamma[i] * s_h + 1.0;
                let beta = weight_beta[i] * s_h;
                output[i] = gamma * ffn_output[i] + beta;
            }
        }
    } else {
        // TOP_K or other: broadcast mean of signal components
        let s_mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        for i in 0..D {
            let gamma = weight_gamma[i] * s_mean + 1.0;
            let beta = weight_beta[i] * s_mean;
            output[i] = gamma * ffn_output[i] + beta;
        }
    }
    output
}

// ── Metric 4: Gradient Compatibility ──
// Condition number of ∂s/∂scale_embed
// For scalar s: gradient is a vector of length J*HEAD_DIM
// Compute via finite differences, report spectral radius (max singular value / min singular value)

fn compute_gradient_condition(
    variant: &SignalVariant,
    rng: &mut StdRng,
) -> f64 {
    let epsilon = 1e-5;
    let number_of_probes = 50;
    let embed_size = J * HEAD_DIM;
    let mut condition_numbers = Vec::new();

    for _ in 0..number_of_probes {
        let embed = sample_scale_embed_calibrated(rng);
        let base_signal = (variant.signal_function)(&embed);

        // Compute gradient ∂s/∂(each element of embed) via finite differences
        // For vector signals, compute Jacobian and use its condition
        let mut gradient_magnitudes = Vec::new();

        // Sample a subset of embed dimensions for efficiency
        let probe_dimensions = 20.min(embed_size);
        let stride = embed_size / probe_dimensions;

        for probe in 0..probe_dimensions {
            let flat_index = probe * stride;
            let j_index = flat_index / HEAD_DIM;
            let d_index = flat_index % HEAD_DIM;

            let mut embed_perturbed = embed.clone();
            embed_perturbed[j_index][d_index] += epsilon;
            let perturbed_signal = (variant.signal_function)(&embed_perturbed);

            let gradient_magnitude: f64 = base_signal
                .iter()
                .zip(perturbed_signal.iter())
                .map(|(base, perturbed)| {
                    let derivative = (perturbed - base) / epsilon;
                    derivative * derivative
                })
                .sum::<f64>();
            gradient_magnitudes.push(gradient_magnitude.sqrt());
        }

        let max_gradient = gradient_magnitudes
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let min_gradient = gradient_magnitudes
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        if min_gradient > 1e-12 {
            condition_numbers.push(max_gradient / min_gradient);
        } else {
            condition_numbers.push(max_gradient / 1e-12);
        }
    }

    condition_numbers.iter().sum::<f64>() / condition_numbers.len() as f64
}

// ── Initialization Strategy Sweep ──

struct InitializationResult {
    name: &'static str,
    steps_to_activate: usize,
    gradient_magnitude_step_1: f64,
    max_gradient_norm: f64,
}

fn sweep_initialization_strategies(
    winning_variant: &SignalVariant,
    rng: &mut StdRng,
) -> Vec<InitializationResult> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let gradient_steps = 100;
    let learning_rate = 0.001;
    let activation_threshold = 0.01;

    let mut results = Vec::new();

    // Compute expected signal magnitude for NORMALIZED init
    let mut signal_magnitudes = Vec::new();
    for _ in 0..1000 {
        let embed = sample_scale_embed_calibrated(rng);
        let signal = (winning_variant.signal_function)(&embed);
        let magnitude: f64 = signal.iter().map(|v| v.abs()).sum::<f64>() / signal.len() as f64;
        signal_magnitudes.push(magnitude);
    }
    let expected_signal = signal_magnitudes.iter().sum::<f64>() / signal_magnitudes.len() as f64;

    for (strategy_name, initial_scale) in [
        ("ZERO", 0.0),
        ("SMALL_RANDOM", 0.01),
        ("NORMALIZED", 0.1 / expected_signal.max(1e-8)),
    ] {
        let mut weight_gamma: Vec<f64> = if initial_scale == 0.0 {
            vec![0.0; D]
        } else if strategy_name == "SMALL_RANDOM" {
            let distribution = Normal::new(0.0, initial_scale).unwrap();
            (0..D).map(|_| distribution.sample(rng)).collect()
        } else {
            vec![initial_scale; D]
        };

        let mut steps_to_activate = gradient_steps;
        let mut gradient_magnitude_step_1 = 0.0;
        let mut max_gradient_norm = 0.0_f64;

        for step in 0..gradient_steps {
            let embed = sample_scale_embed_calibrated(rng);
            let signal = (winning_variant.signal_function)(&embed);
            let ffn_output: Vec<f64> = (0..D).map(|_| normal.sample(rng)).collect();
            let upstream_gradient: Vec<f64> = (0..D).map(|_| normal.sample(rng)).collect();

            let s = if signal.len() == 1 {
                signal[0]
            } else {
                signal.iter().sum::<f64>() / signal.len() as f64
            };

            // ∂loss/∂W_gamma_i = upstream_i * s * ffn_output_i
            let mut gradient_norm_squared = 0.0;
            for i in 0..D {
                let gradient = upstream_gradient[i] * s * ffn_output[i];
                gradient_norm_squared += gradient * gradient;
                weight_gamma[i] -= learning_rate * gradient;
            }
            let gradient_norm = gradient_norm_squared.sqrt();

            if step == 0 {
                gradient_magnitude_step_1 = gradient_norm;
            }
            max_gradient_norm = max_gradient_norm.max(gradient_norm);

            // Check activation: ||gamma - 1||² > threshold
            let film_effect: f64 = weight_gamma
                .iter()
                .map(|w| {
                    let gamma_minus_1 = w * s;
                    gamma_minus_1 * gamma_minus_1
                })
                .sum();
            if film_effect > activation_threshold && steps_to_activate == gradient_steps {
                steps_to_activate = step + 1;
            }
        }

        results.push(InitializationResult {
            name: strategy_name,
            steps_to_activate,
            gradient_magnitude_step_1,
            max_gradient_norm,
        });
    }

    results
}

// ── Output Formatting ──

struct MetricResults {
    entropy: f64,
    head_differentiation: f64,
    mean_sensitivity: f64,
    sensitivity_std: f64,
    gradient_condition: f64,
}

fn format_output(
    variants: &[SignalVariant],
    metrics: &[MetricResults],
    initialization_results: &[InitializationResult],
    winning_index: usize,
) -> String {
    let mut output = String::new();

    output.push_str("=== FiLM Hyperparameter Sweep (D=256, H=8, J=52) ===\n\n");

    output.push_str("CONDITIONING SIGNAL COMPARISON:\n");
    output.push_str(&format!(
        "{:<10} {:>12} {:>14} {:>10} {:>10} {:>10}\n",
        "Signal", "Extra_params", "Entropy(bits)", "Head_diff", "Mean_sens", "Grad_cond"
    ));

    for (index, variant) in variants.iter().enumerate() {
        let metric = &metrics[index];
        output.push_str(&format!(
            "{:<10} {:>12} {:>14.2} {:>10.3} {:>10.3} {:>10.2}\n",
            variant.label,
            variant.extra_parameters,
            metric.entropy,
            metric.head_differentiation,
            metric.mean_sensitivity,
            metric.gradient_condition,
        ));
    }

    output.push('\n');

    // Rankings
    output.push_str("RANKING:\n");

    let best_entropy_index = metrics
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.entropy.partial_cmp(&b.1.entropy).unwrap())
        .unwrap()
        .0;
    output.push_str(&format!(
        "By information content:      {} ({:.2} bits)\n",
        variants[best_entropy_index].label,
        metrics[best_entropy_index].entropy,
    ));

    let best_head_diff_index = metrics
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.head_differentiation
                .partial_cmp(&b.1.head_differentiation)
                .unwrap()
        })
        .unwrap()
        .0;
    output.push_str(&format!(
        "By head differentiation:     {} ({:.3} distance)\n",
        variants[best_head_diff_index].label,
        metrics[best_head_diff_index].head_differentiation,
    ));

    let best_sensitivity_index = metrics
        .iter()
        .enumerate()
        .min_by(|a, b| {
            a.1.sensitivity_std
                .partial_cmp(&b.1.sensitivity_std)
                .unwrap()
        })
        .unwrap()
        .0;
    output.push_str(&format!(
        "By sensitivity stability:    {} (std={:.3})\n",
        variants[best_sensitivity_index].label,
        metrics[best_sensitivity_index].sensitivity_std,
    ));

    let best_gradient_index = metrics
        .iter()
        .enumerate()
        .min_by(|a, b| {
            a.1.gradient_condition
                .partial_cmp(&b.1.gradient_condition)
                .unwrap()
        })
        .unwrap()
        .0;
    output.push_str(&format!(
        "By gradient compatibility:   {} (cond={:.2})\n",
        variants[best_gradient_index].label,
        metrics[best_gradient_index].gradient_condition,
    ));

    output.push_str(&format!(
        "Overall winner:               {}\n",
        variants[winning_index].label,
    ));

    output.push('\n');

    // Initialization Strategy
    output.push_str(&format!(
        "INITIALIZATION STRATEGY (for {}):\n",
        variants[winning_index].label,
    ));
    output.push_str(&format!(
        "{:<14} {:>17} {:>15} {:>14}\n",
        "Strategy", "Steps_to_activate", "Grad_mag_step1", "Max_grad_norm"
    ));

    let mut best_init_index = 0;
    let mut best_init_score = f64::MAX;

    for (index, result) in initialization_results.iter().enumerate() {
        output.push_str(&format!(
            "{:<14} {:>17} {:>15.2e} {:>14.2}\n",
            result.name,
            result.steps_to_activate,
            result.gradient_magnitude_step_1,
            result.max_gradient_norm,
        ));
        // Score: lower steps + reasonable gradient = better
        let score = result.steps_to_activate as f64
            + (result.max_gradient_norm - result.gradient_magnitude_step_1).abs() * 10.0;
        if score < best_init_score {
            best_init_score = score;
            best_init_index = index;
        }
    }

    output.push_str(&format!(
        "Best init:     {}\n",
        initialization_results[best_init_index].name,
    ));

    output.push('\n');

    // Recommendation
    output.push_str("=== RECOMMENDATION ===\n");

    let winner = &variants[winning_index];
    let winner_metrics = &metrics[winning_index];
    let baseline_metrics = &metrics[0];

    let entropy_reason = if winner_metrics.entropy > baseline_metrics.entropy {
        "higher information content"
    } else {
        "comparable information content"
    };
    let head_reason = if winner_metrics.head_differentiation > baseline_metrics.head_differentiation
    {
        "superior head differentiation"
    } else {
        "comparable head differentiation"
    };

    output.push_str(&format!(
        "Optimal conditioning signal: {} ({})\n",
        winner.label, winner.name,
    ));
    output.push_str(&format!(
        "Reason: {} with {}, providing richer conditioning with same parameter budget.\n",
        entropy_reason, head_reason,
    ));

    output.push('\n');

    output.push_str(&format!(
        "Optimal initialization: {}\n",
        initialization_results[best_init_index].name,
    ));

    let init_reason = if initialization_results[best_init_index].steps_to_activate < 50 {
        "fast activation with stable gradient landscape"
    } else {
        "safe convergence with moderate activation speed"
    };
    output.push_str(&format!("Reason: {}.\n", init_reason));

    output.push('\n');

    // Upgrade impact
    let entropy_gain = winner_metrics.entropy - baseline_metrics.entropy;
    let entropy_ratio = if baseline_metrics.entropy > 0.0 {
        winner_metrics.entropy / baseline_metrics.entropy
    } else {
        1.0
    };
    let head_diff_gain =
        winner_metrics.head_differentiation - baseline_metrics.head_differentiation;
    let head_diff_ratio = if baseline_metrics.head_differentiation > 0.0 {
        winner_metrics.head_differentiation / baseline_metrics.head_differentiation
    } else {
        1.0
    };

    let baseline_init = &initialization_results[0];
    let best_init = &initialization_results[best_init_index];
    let speed_ratio = if best_init.steps_to_activate > 0 {
        baseline_init.steps_to_activate as f64 / best_init.steps_to_activate as f64
    } else {
        1.0
    };

    output.push_str("Upgrade impact vs current (Signal A, ZERO init):\n");
    output.push_str(&format!(
        "  Information gain: {:+.2} bits ({:.1}\u{00d7} more informative)\n",
        entropy_gain, entropy_ratio,
    ));
    output.push_str(&format!(
        "  Head differentiation gain: {:+.3} ({:.1}\u{00d7} better)\n",
        head_diff_gain, head_diff_ratio,
    ));
    output.push_str(&format!(
        "  Training speed gain: activates {:.0}\u{00d7} faster\n",
        speed_ratio,
    ));

    output.push('\n');

    let should_upgrade = if entropy_gain > 1.0 || head_diff_ratio > 2.0 {
        "YES"
    } else if entropy_gain > 0.3 || head_diff_ratio > 1.3 {
        "AFTER_RESULTS"
    } else {
        "NO"
    };

    let confidence = if winner_metrics.gradient_condition < 50.0
        && winner_metrics.sensitivity_std < winner_metrics.mean_sensitivity
    {
        "HIGH"
    } else if winner_metrics.gradient_condition < 200.0 {
        "MEDIUM"
    } else {
        "LOW"
    };

    output.push_str(&format!(
        "Should we upgrade the running FiLM experiment? {}\n",
        should_upgrade,
    ));
    output.push_str(&format!("Confidence: {}\n", confidence));

    output
}

// ── Main ──

fn compute_overall_winner(metrics: &[MetricResults]) -> usize {
    let max_entropy = metrics
        .iter()
        .map(|m| m.entropy)
        .fold(0.0_f64, f64::max);
    let max_head_diff = metrics
        .iter()
        .map(|m| m.head_differentiation)
        .fold(0.0_f64, f64::max);
    let max_sensitivity = metrics
        .iter()
        .map(|m| m.mean_sensitivity)
        .fold(0.0_f64, f64::max);
    let max_gradient_cond = metrics
        .iter()
        .map(|m| m.gradient_condition)
        .fold(0.0_f64, f64::max);

    metrics
        .iter()
        .enumerate()
        .map(|(index, metric)| {
            let normalized_entropy = if max_entropy > 0.0 {
                metric.entropy / max_entropy
            } else {
                0.0
            };
            let normalized_head_diff = if max_head_diff > 0.0 {
                metric.head_differentiation / max_head_diff
            } else {
                0.0
            };
            let normalized_sensitivity = if max_sensitivity > 0.0 {
                metric.mean_sensitivity / max_sensitivity
            } else {
                0.0
            };
            // Lower condition number is better, so invert
            let normalized_gradient = if max_gradient_cond > 0.0 {
                1.0 - (metric.gradient_condition / max_gradient_cond)
            } else {
                0.0
            };

            let composite = normalized_entropy * 0.25
                + normalized_head_diff * 0.35
                + normalized_sensitivity * 0.15
                + normalized_gradient * 0.25;

            (index, composite)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

fn main() {
    println!("\u{1f52c} FiLM Hyperparameter Sweep (D={}, H={}, J={})", D, H, J);
    println!();

    let mut rng = StdRng::seed_from_u64(42);
    let variants = signal_variants();

    // Pre-generate samples for entropy computation
    println!("  Generating {} calibrated scale_embed samples...", ENTROPY_SAMPLES);
    let entropy_samples: Vec<Vec<Vec<f64>>> = (0..ENTROPY_SAMPLES)
        .map(|_| sample_scale_embed_calibrated(&mut rng))
        .collect();

    let mut all_metrics = Vec::new();

    for variant in &variants {
        println!("  Analyzing signal {}...", variant.label);

        let entropy = compute_entropy(variant, &entropy_samples);
        let head_differentiation = compute_head_differentiation(variant, &mut rng);
        let (mean_sensitivity, sensitivity_std) = compute_sensitivity(variant, &mut rng);
        let gradient_condition = compute_gradient_condition(variant, &mut rng);

        all_metrics.push(MetricResults {
            entropy,
            head_differentiation,
            mean_sensitivity,
            sensitivity_std,
            gradient_condition,
        });
    }

    let winning_index = compute_overall_winner(&all_metrics);
    println!(
        "  Overall winner: {} — running initialization sweep...",
        variants[winning_index].label,
    );

    let initialization_results =
        sweep_initialization_strategies(&variants[winning_index], &mut rng);

    let formatted = format_output(&variants, &all_metrics, &initialization_results, winning_index);

    print!("{}", formatted);

    let output_path = "exploration/film_hyper_sweep/results.txt";
    let full_path = std::path::Path::new("/home/dlewis3/Desktop/AI/DWARF").join(output_path);
    let mut file = fs::File::create(&full_path).expect("Failed to create results.txt");
    file.write_all(formatted.as_bytes())
        .expect("Failed to write results.txt");
    println!(
        "\n\u{1f4c4} Results saved to {}",
        full_path.display(),
    );
}
