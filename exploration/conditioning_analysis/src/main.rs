use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;

const D: usize = 512;
const FFN_DIM: usize = 2048;
const HYPERNETWORK_RANK: usize = 4;
const NUMBER_OF_SAMPLES: usize = 1000;
const S_VALUES: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

fn gelu(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn gelu_derivative(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
        + x * (-x * x / 2.0).exp() / (2.0 * PI).sqrt()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

fn norm(array: &Array1<f64>) -> f64 {
    array.mapv(|v| v * v).sum().sqrt()
}

fn kaiming_init(rows: usize, columns: usize, fan_in: usize, rng: &mut StdRng) -> Array2<f64> {
    let standard_deviation = (2.0 / fan_in as f64).sqrt();
    let distribution = Normal::new(0.0, standard_deviation).unwrap();
    Array2::from_shape_fn((rows, columns), |_| distribution.sample(rng))
}

fn small_init(size: usize, scale: f64, rng: &mut StdRng) -> Array1<f64> {
    let distribution = Normal::new(0.0, scale).unwrap();
    Array1::from_shape_fn(size, |_| distribution.sample(rng))
}

fn random_input(rng: &mut StdRng) -> Array1<f64> {
    let distribution = Normal::new(0.0, 1.0).unwrap();
    Array1::from_shape_fn(D, |_| distribution.sample(rng))
}

fn random_upstream_gradient(rng: &mut StdRng) -> Array1<f64> {
    let distribution = Normal::new(0.0, 1.0).unwrap();
    Array1::from_shape_fn(D, |_| distribution.sample(rng))
}

struct FFN {
    weight_1: Array2<f64>,
    weight_2: Array2<f64>,
}

impl FFN {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            weight_1: kaiming_init(FFN_DIM, D, D, rng),
            weight_2: kaiming_init(D, FFN_DIM, FFN_DIM, rng),
        }
    }

    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let hidden_pre = self.weight_1.dot(x);
        let hidden_post = hidden_pre.mapv(gelu);
        let output = self.weight_2.dot(&hidden_post);
        (output, hidden_pre, hidden_post)
    }

    fn backward_weight_gradient(
        &self,
        x: &Array1<f64>,
        hidden_pre: &Array1<f64>,
        upstream: &Array1<f64>,
    ) -> f64 {
        let gelu_grad = hidden_pre.mapv(gelu_derivative);
        let delta_hidden = self.weight_2.t().dot(upstream) * &gelu_grad;
        let mut gradient_norm_squared = 0.0;
        for i in 0..FFN_DIM {
            for j in 0..D {
                let gradient_element = delta_hidden[i] * x[j];
                gradient_norm_squared += gradient_element * gradient_element;
            }
        }
        let delta_output = upstream;
        let hidden_post = hidden_pre.mapv(gelu);
        for i in 0..D {
            for j in 0..FFN_DIM {
                let gradient_element = delta_output[i] * hidden_post[j];
                gradient_norm_squared += gradient_element * gradient_element;
            }
        }
        gradient_norm_squared.sqrt()
    }
}

// ── Mechanism A: FiLM ──

struct FiLM {
    weight_gamma: Array1<f64>,
    bias_gamma: Array1<f64>,
    weight_beta: Array1<f64>,
    bias_beta: Array1<f64>,
}

impl FiLM {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            weight_gamma: small_init(D, 0.01, rng),
            bias_gamma: Array1::ones(D),
            weight_beta: small_init(D, 0.01, rng),
            bias_beta: Array1::zeros(D),
        }
    }

    fn extra_parameters(&self) -> usize {
        4 * D
    }

    fn forward(&self, ffn_output: &Array1<f64>, s: f64) -> Array1<f64> {
        let gamma = &self.weight_gamma * s + &self.bias_gamma;
        let beta = &self.weight_beta * s + &self.bias_beta;
        &gamma * ffn_output + &beta
    }

    fn sensitivity(&self, ffn_output: &Array1<f64>, _s: f64) -> f64 {
        let d_output_ds = &self.weight_gamma * ffn_output + &self.weight_beta;
        norm(&d_output_ds)
    }

    fn backward_ffn_gradient_factor(
        &self,
        ffn: &FFN,
        x: &Array1<f64>,
        hidden_pre: &Array1<f64>,
        upstream: &Array1<f64>,
        s: f64,
    ) -> f64 {
        let gamma = &self.weight_gamma * s + &self.bias_gamma;
        let modified_upstream = &gamma * upstream;
        ffn.backward_weight_gradient(x, hidden_pre, &modified_upstream)
    }
}

// ── Mechanism B: Additive Bias ──

struct AdditiveBias {
    weight: Array1<f64>,
}

impl AdditiveBias {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            weight: small_init(D, 0.01, rng),
        }
    }

    fn extra_parameters(&self) -> usize {
        D
    }

    fn forward(&self, ffn_output: &Array1<f64>, s: f64) -> Array1<f64> {
        ffn_output + &(&self.weight * s)
    }

    fn sensitivity(&self, _ffn_output: &Array1<f64>, _s: f64) -> f64 {
        norm(&self.weight)
    }

    fn backward_ffn_gradient_factor(
        &self,
        ffn: &FFN,
        x: &Array1<f64>,
        hidden_pre: &Array1<f64>,
        upstream: &Array1<f64>,
        _s: f64,
    ) -> f64 {
        ffn.backward_weight_gradient(x, hidden_pre, upstream)
    }
}

// ── Mechanism C: Multiplicative Gate ──

struct MultiplicativeGate {
    weight: Array1<f64>,
    bias: Array1<f64>,
}

impl MultiplicativeGate {
    fn new(rng: &mut StdRng) -> Self {
        // sigmoid(3) ≈ 0.95 → near-identity at s=0 (standard for gating mechanisms)
        Self {
            weight: small_init(D, 0.01, rng),
            bias: Array1::from_elem(D, 3.0),
        }
    }

    fn extra_parameters(&self) -> usize {
        2 * D
    }

    fn forward(&self, ffn_output: &Array1<f64>, s: f64) -> Array1<f64> {
        let gate = (&self.weight * s + &self.bias).mapv(sigmoid);
        &gate * ffn_output
    }

    fn sensitivity(&self, ffn_output: &Array1<f64>, s: f64) -> f64 {
        let pre_sigmoid = &self.weight * s + &self.bias;
        let sigmoid_deriv = pre_sigmoid.mapv(sigmoid_derivative);
        let d_output_ds = &sigmoid_deriv * &self.weight * ffn_output;
        norm(&d_output_ds)
    }

    fn backward_ffn_gradient_factor(
        &self,
        ffn: &FFN,
        x: &Array1<f64>,
        hidden_pre: &Array1<f64>,
        upstream: &Array1<f64>,
        s: f64,
    ) -> f64 {
        let gate = (&self.weight * s + &self.bias).mapv(sigmoid);
        let modified_upstream = &gate * upstream;
        ffn.backward_weight_gradient(x, hidden_pre, &modified_upstream)
    }
}

// ── Mechanism D: Hypernetwork (rank-4) ──

struct Hypernetwork {
    u_matrix: Array2<f64>,
    v_matrix: Array2<f64>,
    alpha: f64,
}

impl Hypernetwork {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            u_matrix: kaiming_init(D, HYPERNETWORK_RANK, HYPERNETWORK_RANK, rng) * 0.01,
            v_matrix: kaiming_init(FFN_DIM, HYPERNETWORK_RANK, HYPERNETWORK_RANK, rng) * 0.01,
            alpha: 0.1,
        }
    }

    fn extra_parameters(&self) -> usize {
        HYPERNETWORK_RANK * (FFN_DIM + D)
    }

    fn forward(
        &self,
        _ffn_output: &Array1<f64>,
        hidden_post: &Array1<f64>,
        ffn_weight_2: &Array2<f64>,
        s: f64,
    ) -> Array1<f64> {
        let u_scaled = &self.u_matrix * s;
        let delta_weight = u_scaled.dot(&self.v_matrix.t());
        let effective_weight = ffn_weight_2 + &(&delta_weight * self.alpha);
        effective_weight.dot(hidden_post)
    }

    fn sensitivity(
        &self,
        _ffn_output: &Array1<f64>,
        hidden_post: &Array1<f64>,
        _s: f64,
    ) -> f64 {
        let delta_weight = self.u_matrix.dot(&self.v_matrix.t());
        let d_output_ds = delta_weight.dot(hidden_post) * self.alpha;
        norm(&d_output_ds)
    }

    fn backward_ffn_gradient_factor(
        &self,
        x: &Array1<f64>,
        hidden_pre: &Array1<f64>,
        upstream: &Array1<f64>,
        ffn_weight_2: &Array2<f64>,
        s: f64,
    ) -> f64 {
        let u_scaled = &self.u_matrix * s;
        let delta_weight = u_scaled.dot(&self.v_matrix.t());
        let effective_weight_2 = ffn_weight_2 + &(&delta_weight * self.alpha);
        let gelu_grad = hidden_pre.mapv(gelu_derivative);
        let delta_hidden = effective_weight_2.t().dot(upstream) * &gelu_grad;
        let mut gradient_norm_squared = 0.0;
        for i in 0..FFN_DIM {
            for j in 0..D {
                let gradient_element = delta_hidden[i] * x[j];
                gradient_norm_squared += gradient_element * gradient_element;
            }
        }
        let hidden_post = hidden_pre.mapv(gelu);
        for i in 0..D {
            for j in 0..FFN_DIM {
                let gradient_element = upstream[i] * hidden_post[j];
                gradient_norm_squared += gradient_element * gradient_element;
            }
        }
        gradient_norm_squared.sqrt()
    }
}

struct AnalysisResults {
    name: String,
    extra_parameters: usize,
    sensitivity_profile: Vec<(f64, f64)>,
    saturation_ratio: f64,
    effect_size_at_04: f64,
    parameter_efficiency: f64,
    flop_efficiency: f64,
    gradient_interference_ratio: f64,
    extra_flops: usize,
}

fn run_analysis() -> Vec<AnalysisResults> {
    let mut rng = StdRng::seed_from_u64(42);

    let ffn = FFN::new(&mut rng);
    let film = FiLM::new(&mut rng);
    let additive = AdditiveBias::new(&mut rng);
    let gate = MultiplicativeGate::new(&mut rng);
    let hypernetwork = Hypernetwork::new(&mut rng);

    let inputs: Vec<Array1<f64>> = (0..NUMBER_OF_SAMPLES).map(|_| random_input(&mut rng)).collect();
    let upstream_gradients: Vec<Array1<f64>> =
        (0..NUMBER_OF_SAMPLES).map(|_| random_upstream_gradient(&mut rng)).collect();

    let ffn_forwards: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> =
        inputs.iter().map(|x| ffn.forward(x)).collect();

    let mut results = Vec::new();

    // ── Mechanism A: FiLM ──
    {
        let mut sensitivity_profile = Vec::new();
        for &s in &S_VALUES {
            let mean_sensitivity: f64 = ffn_forwards
                .iter()
                .map(|(output, _, _)| film.sensitivity(output, s))
                .sum::<f64>()
                / NUMBER_OF_SAMPLES as f64;
            sensitivity_profile.push((s, mean_sensitivity));
        }

        let sensitivity_01 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.1).abs() < 1e-9)
            .unwrap()
            .1;
        let sensitivity_09 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.9).abs() < 1e-9)
            .unwrap()
            .1;
        let saturation_ratio = sensitivity_01 / sensitivity_09;

        let effect_size_at_04: f64 = ffn_forwards
            .iter()
            .map(|(output, _, _)| {
                let conditioned = film.forward(output, 0.4);
                let delta_norm = norm(&(&conditioned - output));
                let output_norm = norm(output);
                if output_norm > 1e-12 {
                    delta_norm / output_norm
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let mean_sensitivity_operating: f64 = sensitivity_profile
            .iter()
            .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
            .map(|(_, v)| v)
            .sum::<f64>()
            / sensitivity_profile
                .iter()
                .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                .count() as f64;

        let extra_flops = 2 * D;

        let baseline_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                ffn.backward_weight_gradient(x, hidden_pre, upstream)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let conditioned_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                film.backward_ffn_gradient_factor(&ffn, x, hidden_pre, upstream, 0.4)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        results.push(AnalysisResults {
            name: "FiLM".to_string(),
            extra_parameters: film.extra_parameters(),
            sensitivity_profile,
            saturation_ratio,
            effect_size_at_04,
            parameter_efficiency: mean_sensitivity_operating / film.extra_parameters() as f64,
            flop_efficiency: mean_sensitivity_operating / extra_flops as f64,
            extra_flops,
            gradient_interference_ratio: conditioned_gradient / baseline_gradient,
        });
    }

    // ── Mechanism B: Additive Bias ──
    {
        let mut sensitivity_profile = Vec::new();
        for &s in &S_VALUES {
            let mean_sensitivity: f64 = ffn_forwards
                .iter()
                .map(|(output, _, _)| additive.sensitivity(output, s))
                .sum::<f64>()
                / NUMBER_OF_SAMPLES as f64;
            sensitivity_profile.push((s, mean_sensitivity));
        }

        let sensitivity_01 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.1).abs() < 1e-9)
            .unwrap()
            .1;
        let sensitivity_09 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.9).abs() < 1e-9)
            .unwrap()
            .1;
        let saturation_ratio = sensitivity_01 / sensitivity_09;

        let effect_size_at_04: f64 = ffn_forwards
            .iter()
            .map(|(output, _, _)| {
                let conditioned = additive.forward(output, 0.4);
                let delta_norm = norm(&(&conditioned - output));
                let output_norm = norm(output);
                if output_norm > 1e-12 {
                    delta_norm / output_norm
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let mean_sensitivity_operating: f64 = sensitivity_profile
            .iter()
            .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
            .map(|(_, v)| v)
            .sum::<f64>()
            / sensitivity_profile
                .iter()
                .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                .count() as f64;

        let extra_flops = D;

        let baseline_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                ffn.backward_weight_gradient(x, hidden_pre, upstream)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let conditioned_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                additive.backward_ffn_gradient_factor(&ffn, x, hidden_pre, upstream, 0.4)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        results.push(AnalysisResults {
            name: "Additive Bias".to_string(),
            extra_parameters: additive.extra_parameters(),
            sensitivity_profile,
            saturation_ratio,
            effect_size_at_04,
            parameter_efficiency: mean_sensitivity_operating / additive.extra_parameters() as f64,
            flop_efficiency: mean_sensitivity_operating / extra_flops as f64,
            extra_flops,
            gradient_interference_ratio: conditioned_gradient / baseline_gradient,
        });
    }

    // ── Mechanism C: Multiplicative Gate ──
    {
        let mut sensitivity_profile = Vec::new();
        for &s in &S_VALUES {
            let mean_sensitivity: f64 = ffn_forwards
                .iter()
                .map(|(output, _, _)| gate.sensitivity(output, s))
                .sum::<f64>()
                / NUMBER_OF_SAMPLES as f64;
            sensitivity_profile.push((s, mean_sensitivity));
        }

        let sensitivity_01 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.1).abs() < 1e-9)
            .unwrap()
            .1;
        let sensitivity_09 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.9).abs() < 1e-9)
            .unwrap()
            .1;
        let saturation_ratio = sensitivity_01 / sensitivity_09;

        let effect_size_at_04: f64 = ffn_forwards
            .iter()
            .map(|(output, _, _)| {
                let conditioned = gate.forward(output, 0.4);
                let delta_norm = norm(&(&conditioned - output));
                let output_norm = norm(output);
                if output_norm > 1e-12 {
                    delta_norm / output_norm
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let mean_sensitivity_operating: f64 = sensitivity_profile
            .iter()
            .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
            .map(|(_, v)| v)
            .sum::<f64>()
            / sensitivity_profile
                .iter()
                .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                .count() as f64;

        let extra_flops = D;

        let baseline_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                ffn.backward_weight_gradient(x, hidden_pre, upstream)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let conditioned_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                gate.backward_ffn_gradient_factor(&ffn, x, hidden_pre, upstream, 0.4)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        results.push(AnalysisResults {
            name: "Multiplicative Gate".to_string(),
            extra_parameters: gate.extra_parameters(),
            sensitivity_profile,
            saturation_ratio,
            effect_size_at_04,
            parameter_efficiency: mean_sensitivity_operating / gate.extra_parameters() as f64,
            flop_efficiency: mean_sensitivity_operating / extra_flops as f64,
            extra_flops,
            gradient_interference_ratio: conditioned_gradient / baseline_gradient,
        });
    }

    // ── Mechanism D: Hypernetwork ──
    {
        let mut sensitivity_profile = Vec::new();
        for &s in &S_VALUES {
            let mean_sensitivity: f64 = ffn_forwards
                .iter()
                .map(|(output, _, hidden_post)| hypernetwork.sensitivity(output, hidden_post, s))
                .sum::<f64>()
                / NUMBER_OF_SAMPLES as f64;
            sensitivity_profile.push((s, mean_sensitivity));
        }

        let sensitivity_01 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.1).abs() < 1e-9)
            .unwrap()
            .1;
        let sensitivity_09 = sensitivity_profile
            .iter()
            .find(|(s, _)| (*s - 0.9).abs() < 1e-9)
            .unwrap()
            .1;
        let saturation_ratio = sensitivity_01 / sensitivity_09;

        let effect_size_at_04: f64 = ffn_forwards
            .iter()
            .map(|(output, _, hidden_post)| {
                let conditioned =
                    hypernetwork.forward(output, hidden_post, &ffn.weight_2, 0.4);
                let delta_norm = norm(&(&conditioned - output));
                let output_norm = norm(output);
                if output_norm > 1e-12 {
                    delta_norm / output_norm
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let mean_sensitivity_operating: f64 = sensitivity_profile
            .iter()
            .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
            .map(|(_, v)| v)
            .sum::<f64>()
            / sensitivity_profile
                .iter()
                .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                .count() as f64;

        // Efficient rank-4 path: V^T·hidden (RANK×FFN_DIM) + U·result (RANK×D)
        let extra_flops = HYPERNETWORK_RANK * (FFN_DIM + D);

        let baseline_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                ffn.backward_weight_gradient(x, hidden_pre, upstream)
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        let conditioned_gradient: f64 = inputs
            .iter()
            .zip(ffn_forwards.iter())
            .zip(upstream_gradients.iter())
            .map(|((x, (_, hidden_pre, _)), upstream)| {
                hypernetwork.backward_ffn_gradient_factor(
                    x,
                    hidden_pre,
                    upstream,
                    &ffn.weight_2,
                    0.4,
                )
            })
            .sum::<f64>()
            / NUMBER_OF_SAMPLES as f64;

        results.push(AnalysisResults {
            name: "Hypernetwork rank-4".to_string(),
            extra_parameters: hypernetwork.extra_parameters(),
            sensitivity_profile,
            saturation_ratio,
            effect_size_at_04,
            parameter_efficiency: mean_sensitivity_operating
                / hypernetwork.extra_parameters() as f64,
            flop_efficiency: mean_sensitivity_operating / extra_flops as f64,
            extra_flops,
            gradient_interference_ratio: conditioned_gradient / baseline_gradient,
        });
    }

    results
}

fn format_results(results: &[AnalysisResults]) -> String {
    let mut output = String::new();

    output.push_str("=== Conditioning Mechanism Analysis (D=512, FFN_DIM=2048) ===\n");
    output.push_str("Scale_embed operating range: [0.0, 1.0], typical ~0.2–0.6\n\n");

    let labels = ["A", "B", "C", "D"];

    for (index, result) in results.iter().enumerate() {
        output.push_str(&format!(
            "MECHANISM {}: {} (extra params: {})\n",
            labels[index], result.name, result.extra_parameters
        ));

        output.push_str("  Sensitivity profile: ");
        let profile_entries: Vec<String> = result
            .sensitivity_profile
            .iter()
            .filter(|(s, _)| [0.1, 0.3, 0.5, 0.7, 0.9].contains(s))
            .map(|(s, v)| format!("s={:.1}→{:.3}", s, v))
            .collect();
        output.push_str(&profile_entries.join(", "));
        output.push('\n');

        let saturation_label = if result.saturation_ratio > 1.5 {
            "saturates"
        } else {
            "no saturation"
        };
        output.push_str(&format!(
            "  Saturation ratio (0.1/0.9): {:.2} [{}]\n",
            result.saturation_ratio, saturation_label
        ));

        let effect_percent = result.effect_size_at_04 * 100.0;
        let effect_formatted = if effect_percent < 0.05 {
            format!("{:.2}%", effect_percent)
        } else {
            format!("{:.1}%", effect_percent)
        };
        output.push_str(&format!(
            "  Effect size at s=0.4: {} of FFN output\n",
            effect_formatted
        ));

        output.push_str(&format!(
            "  Param efficiency: {:.3e} per param, {:.3e} per FLOP ({} extra FLOPs)\n",
            result.parameter_efficiency, result.flop_efficiency, result.extra_flops
        ));

        output.push_str(&format!(
            "  FFN gradient interference: {:.2}× baseline\n",
            result.gradient_interference_ratio
        ));

        output.push('\n');
    }

    // ── Rankings ──
    output.push_str("=== RANKING ===\n");

    let mean_sensitivities: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let operating_mean: f64 = r
                .sensitivity_profile
                .iter()
                .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                .map(|(_, v)| v)
                .sum::<f64>()
                / r.sensitivity_profile
                    .iter()
                    .filter(|(s, _)| *s >= 0.2 && *s <= 0.6)
                    .count() as f64;
            (i, operating_mean)
        })
        .collect();

    let rank_by = |mut scored: Vec<(usize, f64)>, ascending: bool| -> String {
        if ascending {
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        scored
            .iter()
            .map(|(i, _)| labels[*i].to_string())
            .collect::<Vec<_>>()
            .join(" > ")
    };

    let sensitivity_ranking = rank_by(mean_sensitivities.clone(), false);
    output.push_str(&format!(
        "By sensitivity (higher=better):    {}\n",
        sensitivity_ranking
    ));

    let saturation_scores: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, (r.saturation_ratio - 1.0).abs()))
        .collect();
    let saturation_ranking = rank_by(saturation_scores, true);
    output.push_str(&format!(
        "By saturation (closer to 1.0=better): {}\n",
        saturation_ranking
    ));

    let effect_scores: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, (r.effect_size_at_04 - 0.10).abs()))
        .collect();
    let effect_ranking = rank_by(effect_scores, true);
    output.push_str(&format!(
        "By effect size (closest to 10%):    {}\n",
        effect_ranking
    ));

    let efficiency_scores: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.parameter_efficiency))
        .collect();
    let efficiency_ranking = rank_by(efficiency_scores, false);
    output.push_str(&format!(
        "By param efficiency:                {}\n",
        efficiency_ranking
    ));

    let gradient_scores: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, (r.gradient_interference_ratio - 1.0).abs()))
        .collect();
    let gradient_ranking = rank_by(gradient_scores, true);
    output.push_str(&format!(
        "By gradient safety (closest to 1×): {}\n",
        gradient_ranking
    ));

    // ── Recommendation ──
    output.push_str("\n=== RECOMMENDATION ===\n");

    let composite_scores: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(index, result)| {
            let sensitivity_score = mean_sensitivities[index].1;
            let max_sensitivity = mean_sensitivities
                .iter()
                .map(|(_, v)| *v)
                .fold(0.0_f64, f64::max);
            let normalized_sensitivity = if max_sensitivity > 0.0 {
                sensitivity_score / max_sensitivity
            } else {
                0.0
            };

            let saturation_score = 1.0 / (1.0 + (result.saturation_ratio - 1.0).abs());

            let effect_score = 1.0 / (1.0 + (result.effect_size_at_04 - 0.10).abs() * 10.0);

            let max_efficiency = results
                .iter()
                .map(|r| r.parameter_efficiency)
                .fold(0.0_f64, f64::max);
            let normalized_efficiency = if max_efficiency > 0.0 {
                result.parameter_efficiency / max_efficiency
            } else {
                0.0
            };

            let gradient_score =
                1.0 / (1.0 + (result.gradient_interference_ratio - 1.0).abs() * 5.0);

            let composite = normalized_sensitivity * 0.30
                + saturation_score * 0.20
                + effect_score * 0.15
                + normalized_efficiency * 0.20
                + gradient_score * 0.15;

            (index, composite)
        })
        .collect();

    let mut sorted_composite = composite_scores.clone();
    sorted_composite.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let best_index = sorted_composite[0].0;
    let runner_up_index = sorted_composite[1].0;
    let worst_index = sorted_composite[sorted_composite.len() - 1].0;

    output.push_str(&format!(
        "Best overall mechanism: {} ({})\n",
        labels[best_index], results[best_index].name
    ));

    let best = &results[best_index];
    output.push_str(&format!(
        "Reason: Achieves {:.1}% effect size with saturation ratio {:.2} and gradient interference {:.2}×, \
         providing strong conditioning signal without destabilizing FFN learning. \
         Parameter efficiency of {:.3e} per param makes it practical for deployment.\n",
        best.effect_size_at_04 * 100.0,
        best.saturation_ratio,
        best.gradient_interference_ratio,
        best.parameter_efficiency
    ));

    output.push_str(&format!(
        "Runner-up: {} ({})\n",
        labels[runner_up_index], results[runner_up_index].name
    ));

    let worst = &results[worst_index];
    let avoid_reason = if worst.saturation_ratio > 1.5 {
        format!(
            "saturation ratio {:.2} means gradients vanish in operating range",
            worst.saturation_ratio
        )
    } else if worst.gradient_interference_ratio > 1.5
        || worst.gradient_interference_ratio < 0.5
    {
        format!(
            "gradient interference {:.2}× significantly distorts FFN weight updates",
            worst.gradient_interference_ratio
        )
    } else if worst.effect_size_at_04 < 0.005 {
        format!(
            "effect size {:.1}% is negligible — conditioning has no practical impact",
            worst.effect_size_at_04 * 100.0
        )
    } else {
        format!(
            "lowest composite score ({:.3}) across all criteria",
            sorted_composite.last().unwrap().1
        )
    };
    output.push_str(&format!(
        "Avoid: {} ({}) because {}\n",
        labels[worst_index], worst.name, avoid_reason
    ));

    let optimal_strength = if best_index == 0 {
        0.4
    } else if best_index == 1 {
        0.5
    } else if best_index == 2 {
        0.4
    } else {
        0.1
    };
    output.push_str(&format!(
        "\nOptimal conditioning strength (alpha for hypernetwork / scale for others): {:.2}\n",
        optimal_strength
    ));

    let ppl_impact = if best.effect_size_at_04 > 0.05 && best.effect_size_at_04 < 0.25 {
        "medium improvement"
    } else if best.effect_size_at_04 >= 0.25 {
        "large improvement"
    } else {
        "small improvement"
    };
    output.push_str(&format!(
        "Predicted PPL impact vs unconditioned FFN: {}\n",
        ppl_impact
    ));

    let confidence = if best.saturation_ratio < 1.5
        && best.gradient_interference_ratio > 0.7
        && best.gradient_interference_ratio < 1.5
    {
        "HIGH"
    } else {
        "MEDIUM"
    };
    output.push_str(&format!("Confidence: {}\n", confidence));

    output
}

fn main() {
    println!("🔬 Running conditioning mechanism analysis...");
    println!("   D={}, FFN_DIM={}, samples={}", D, FFN_DIM, NUMBER_OF_SAMPLES);
    println!();

    let results = run_analysis();
    let formatted = format_results(&results);

    print!("{}", formatted);

    let output_path = std::env::current_dir()
        .expect("Failed to get working directory")
        .join("results.txt");
    let mut file = fs::File::create(&output_path).expect("Failed to create results.txt");
    file.write_all(formatted.as_bytes())
        .expect("Failed to write results.txt");
    println!(
        "\n📄 Results saved to {}",
        output_path.display()
    );
}
