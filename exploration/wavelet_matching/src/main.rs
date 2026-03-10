use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::SQRT_2;
use std::fs;
use std::io::Write;

const SIGNAL_LENGTH: usize = 2048;
const DWT_LEVELS: usize = 11;

// ── DSQG d41_35m offset definitions ──

fn dsqg_dense_offsets() -> Vec<usize> {
    (1..=48).collect()
}

fn dsqg_sparse_offsets() -> Vec<usize> {
    vec![96, 128, 384]
}

fn all_dsqg_offsets() -> Vec<usize> {
    let mut offsets = dsqg_dense_offsets();
    offsets.extend(dsqg_sparse_offsets());
    offsets
}

// ── Signal construction ──

fn build_unweighted_signal() -> Vec<f64> {
    let mut signal = vec![0.0; SIGNAL_LENGTH];
    for &offset in &all_dsqg_offsets() {
        signal[offset] = 1.0;
    }
    signal
}

fn build_weighted_signal() -> Vec<f64> {
    let mut signal = vec![0.0; SIGNAL_LENGTH];
    for delta in 1..=48_usize {
        signal[delta] = 1.0 / delta as f64;
    }
    signal[96] = 0.8;
    signal[128] = 0.7;
    signal[384] = 0.4;
    signal
}

// ── Wavelet filter definitions ──

#[derive(Clone)]
struct WaveletFilters {
    name: &'static str,
    decomposition_low: Vec<f64>,
    decomposition_high: Vec<f64>,
    reconstruction_low: Vec<f64>,
    reconstruction_high: Vec<f64>,
    vanishing_moments: usize,
    use_lifting: bool,
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

fn build_haar() -> WaveletFilters {
    let low = vec![1.0 / SQRT_2, 1.0 / SQRT_2];
    let high = qmf_highpass(&low);
    WaveletFilters {
        name: "Haar",
        decomposition_low: low.clone(),
        decomposition_high: high.clone(),
        reconstruction_low: low,
        reconstruction_high: high,
        vanishing_moments: 1,
        use_lifting: false,
    }
}

fn build_d4() -> WaveletFilters {
    let sqrt3 = 3.0_f64.sqrt();
    let denom = 4.0 * SQRT_2;
    let low = vec![
        (1.0 + sqrt3) / denom,
        (3.0 + sqrt3) / denom,
        (3.0 - sqrt3) / denom,
        (1.0 - sqrt3) / denom,
    ];
    let high = qmf_highpass(&low);
    WaveletFilters {
        name: "D4",
        decomposition_low: low.clone(),
        decomposition_high: high.clone(),
        reconstruction_low: low,
        reconstruction_high: high,
        vanishing_moments: 2,
        use_lifting: false,
    }
}

fn build_d8() -> WaveletFilters {
    let low = vec![
        0.23037781330886,
        0.71484657055292,
        0.63088076792959,
        -0.02798376941685,
        -0.18703481171909,
        0.03084138183556,
        0.03288301166688,
        -0.01059740178507,
    ];
    let high = qmf_highpass(&low);
    WaveletFilters {
        name: "D8",
        decomposition_low: low.clone(),
        decomposition_high: high.clone(),
        reconstruction_low: low,
        reconstruction_high: high,
        vanishing_moments: 4,
        use_lifting: false,
    }
}

fn build_coiflet6() -> WaveletFilters {
    let low = vec![
        -0.0156557281354600,
        -0.0727326195128539,
        0.3848648468648579,
        0.8525720202122554,
        0.3378976624578092,
        -0.0727326195128539,
    ];
    let high = qmf_highpass(&low);
    WaveletFilters {
        name: "C6",
        decomposition_low: low.clone(),
        decomposition_high: high.clone(),
        reconstruction_low: low,
        reconstruction_high: high,
        vanishing_moments: 2,
        use_lifting: false,
    }
}

fn build_symlet8() -> WaveletFilters {
    let low = vec![
        -0.07576571478927,
        -0.02963552764600,
        0.49761866763202,
        0.80373875180592,
        0.29785779560528,
        -0.09921954357685,
        -0.01260396726203,
        0.03222310060404,
    ];
    let high = qmf_highpass(&low);
    WaveletFilters {
        name: "S8",
        decomposition_low: low.clone(),
        decomposition_high: high.clone(),
        reconstruction_low: low,
        reconstruction_high: high,
        vanishing_moments: 4,
        use_lifting: false,
    }
}

fn build_bior22() -> WaveletFilters {
    let decomposition_low = vec![
        -0.1767766952966369,
        0.3535533905932738,
        1.0606601717798214,
        0.3535533905932738,
        -0.1767766952966369,
    ];
    let decomposition_high = vec![
        0.3535533905932738,
        -0.7071067811865476,
        0.3535533905932738,
    ];
    let reconstruction_low = vec![
        0.3535533905932738,
        0.7071067811865476,
        0.3535533905932738,
    ];
    let reconstruction_high = vec![
        0.1767766952966369,
        0.3535533905932738,
        -1.0606601717798214,
        0.3535533905932738,
        0.1767766952966369,
    ];

    WaveletFilters {
        name: "Bior2.2",
        decomposition_low,
        decomposition_high,
        reconstruction_low,
        reconstruction_high,
        vanishing_moments: 2,
        use_lifting: true,
    }
}

fn all_wavelets() -> Vec<WaveletFilters> {
    vec![
        build_haar(),
        build_d4(),
        build_d8(),
        build_coiflet6(),
        build_symlet8(),
        build_bior22(),
    ]
}

// ── DWT core ──

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

fn upsample_circular_convolve(coefficients: &[f64], filter: &[f64], output_length: usize) -> Vec<f64> {
    let filter_length = filter.len();
    let mut output = vec![0.0; output_length];

    for coefficient_index in 0..coefficients.len() {
        let upsampled_position = 2 * coefficient_index;
        for filter_index in 0..filter_length {
            let output_index =
                (upsampled_position + output_length - filter_index) % output_length;
            output[output_index] += coefficients[coefficient_index] * filter[filter_index];
        }
    }
    output
}

struct DwtResult {
    approximation: Vec<f64>,
    details: Vec<Vec<f64>>,
    levels: usize,
}

fn lifting_forward_one_level(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let length = signal.len();
    let half = length / 2;
    let even: Vec<f64> = (0..half).map(|n| signal[2 * n]).collect();
    let odd: Vec<f64> = (0..half).map(|n| signal[2 * n + 1]).collect();

    let detail: Vec<f64> = (0..half)
        .map(|n| odd[n] - (even[n] + even[(n + 1) % half]) / 2.0)
        .collect();

    let approx: Vec<f64> = (0..half)
        .map(|n| even[n] + (detail[(n + half - 1) % half] + detail[n]) / 4.0)
        .collect();

    let scaled_approx: Vec<f64> = approx.iter().map(|v| v * SQRT_2).collect();
    let scaled_detail: Vec<f64> = detail.iter().map(|v| v / SQRT_2).collect();

    (scaled_approx, scaled_detail)
}

fn lifting_inverse_one_level(approximation: &[f64], detail: &[f64]) -> Vec<f64> {
    let half = approximation.len();
    let unscaled_approx: Vec<f64> = approximation.iter().map(|v| v / SQRT_2).collect();
    let unscaled_detail: Vec<f64> = detail.iter().map(|v| v * SQRT_2).collect();

    let even: Vec<f64> = (0..half)
        .map(|n| {
            unscaled_approx[n]
                - (unscaled_detail[(n + half - 1) % half] + unscaled_detail[n]) / 4.0
        })
        .collect();

    let odd: Vec<f64> = (0..half)
        .map(|n| unscaled_detail[n] + (even[n] + even[(n + 1) % half]) / 2.0)
        .collect();

    let mut output = vec![0.0; half * 2];
    for n in 0..half {
        output[2 * n] = even[n];
        output[2 * n + 1] = odd[n];
    }
    output
}

fn dwt_forward(signal: &[f64], wavelet: &WaveletFilters, levels: usize) -> DwtResult {
    let mut approximation = signal.to_vec();
    let mut details = Vec::with_capacity(levels);

    for _ in 0..levels {
        if wavelet.use_lifting {
            let (approx, detail) = lifting_forward_one_level(&approximation);
            details.push(detail);
            approximation = approx;
        } else {
            let detail =
                circular_convolve_downsample(&approximation, &wavelet.decomposition_high);
            approximation =
                circular_convolve_downsample(&approximation, &wavelet.decomposition_low);
            details.push(detail);
        }
    }

    DwtResult {
        approximation,
        details,
        levels,
    }
}

fn dwt_inverse(result: &DwtResult, wavelet: &WaveletFilters) -> Vec<f64> {
    let mut approximation = result.approximation.clone();

    for level in (0..result.levels).rev() {
        if wavelet.use_lifting {
            approximation = lifting_inverse_one_level(&approximation, &result.details[level]);
        } else {
            let target_length = result.details[level].len() * 2;
            let reconstructed_low = upsample_circular_convolve(
                &approximation,
                &wavelet.reconstruction_low,
                target_length,
            );
            let reconstructed_high = upsample_circular_convolve(
                &result.details[level],
                &wavelet.reconstruction_high,
                target_length,
            );
            approximation = reconstructed_low
                .iter()
                .zip(reconstructed_high.iter())
                .map(|(low, high)| low + high)
                .collect();
        }
    }
    approximation
}

fn all_coefficients_flat(result: &DwtResult) -> Vec<f64> {
    let mut coefficients = Vec::new();
    for detail in &result.details {
        coefficients.extend(detail);
    }
    coefficients.extend(&result.approximation);
    coefficients
}

fn reconstruct_from_sparse(
    result: &DwtResult,
    top_k: usize,
    wavelet: &WaveletFilters,
) -> Vec<f64> {
    let all_coefficients = all_coefficients_flat(result);
    let total_count = all_coefficients.len();

    let mut indexed: Vec<(usize, f64)> = all_coefficients
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut mask = vec![false; total_count];
    for &(index, _) in indexed.iter().take(top_k) {
        mask[index] = true;
    }

    let mut offset = 0;
    let mut sparse_details = Vec::new();
    for detail in &result.details {
        let length = detail.len();
        let sparse_detail: Vec<f64> = detail
            .iter()
            .enumerate()
            .map(|(index, &value)| if mask[offset + index] { value } else { 0.0 })
            .collect();
        sparse_details.push(sparse_detail);
        offset += length;
    }

    let approx_length = result.approximation.len();
    let sparse_approximation: Vec<f64> = result
        .approximation
        .iter()
        .enumerate()
        .map(|(index, &value)| if mask[offset + index] { value } else { 0.0 })
        .collect();
    assert_eq!(offset + approx_length, total_count);

    let sparse_result = DwtResult {
        approximation: sparse_approximation,
        details: sparse_details,
        levels: result.levels,
    };
    dwt_inverse(&sparse_result, wavelet)
}

fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

fn mean_squared_error_normalized(original: &[f64], reconstructed: &[f64]) -> f64 {
    let original_energy = energy(original);
    if original_energy == 0.0 {
        return 0.0;
    }
    let error_energy: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    error_energy / original_energy
}

// ── Analysis 1: Sparse Approximation Quality ──

struct SparseApproximationResult {
    wavelet_name: String,
    mse_per_k: Vec<(usize, f64)>,
    energy_fraction_per_k: Vec<(usize, f64)>,
}

fn analysis_sparse_approximation(
    signal: &[f64],
    wavelet: &WaveletFilters,
) -> SparseApproximationResult {
    let result = dwt_forward(signal, wavelet, DWT_LEVELS);
    let k_values = vec![5, 10, 20, 30, 50, 100];
    let total_energy = energy(signal);
    let all_coefficients = all_coefficients_flat(&result);

    let mut indexed: Vec<(usize, f64)> = all_coefficients
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut mse_per_k = Vec::new();
    let mut energy_fraction_per_k = Vec::new();

    for &k in &k_values {
        let reconstructed = reconstruct_from_sparse(&result, k, wavelet);
        let mse = mean_squared_error_normalized(signal, &reconstructed);
        mse_per_k.push((k, mse));

        let top_k_energy: f64 = indexed.iter().take(k).map(|(_, v)| v * v).sum();
        let fraction = if total_energy > 0.0 {
            top_k_energy / energy(&all_coefficients)
        } else {
            0.0
        };
        energy_fraction_per_k.push((k, fraction));
    }

    SparseApproximationResult {
        wavelet_name: wavelet.name.to_string(),
        mse_per_k,
        energy_fraction_per_k,
    }
}

// ── Analysis 2: Scale-Space Decomposition ──

struct ScaleSpaceResult {
    wavelet_name: String,
    level_energies: Vec<f64>,
    dense_energy_fraction: f64,
    transition_energy_fraction: f64,
    sparse_mid_energy_fraction: f64,
    sparse_long_energy_fraction: f64,
}

fn analysis_scale_space(signal: &[f64], wavelet: &WaveletFilters) -> ScaleSpaceResult {
    let result = dwt_forward(signal, wavelet, DWT_LEVELS);

    let level_energies: Vec<f64> = result.details.iter().map(|d| energy(d)).collect();
    let approx_energy = energy(&result.approximation);
    let total_detail_energy: f64 = level_energies.iter().sum::<f64>() + approx_energy;

    let dense_energy: f64 = level_energies.iter().take(4).sum();
    let transition_energy: f64 = level_energies.iter().skip(4).take(2).sum();
    let sparse_mid_energy: f64 = level_energies.iter().skip(6).take(2).sum();
    let sparse_long_energy: f64 = level_energies.iter().skip(8).take(3).sum::<f64>() + approx_energy;

    let safe_divide = |numerator: f64, denominator: f64| {
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    };

    ScaleSpaceResult {
        wavelet_name: wavelet.name.to_string(),
        level_energies,
        dense_energy_fraction: safe_divide(dense_energy, total_detail_energy),
        transition_energy_fraction: safe_divide(transition_energy, total_detail_energy),
        sparse_mid_energy_fraction: safe_divide(sparse_mid_energy, total_detail_energy),
        sparse_long_energy_fraction: safe_divide(sparse_long_energy, total_detail_energy),
    }
}

// ── Analysis 3: Cross-Scale Leakage ──

struct LeakageResult {
    wavelet_name: String,
    leakage_at_96: f64,
    leakage_at_128: f64,
    leakage_at_384: f64,
    mean_leakage: f64,
}

fn primary_level_for_offset(offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }
    let level = (offset as f64).log2().floor() as usize;
    level.min(DWT_LEVELS - 1)
}

fn analysis_cross_scale_leakage(wavelet: &WaveletFilters) -> LeakageResult {
    let offsets = [96_usize, 128, 384];
    let mut leakages = Vec::new();

    for &offset in &offsets {
        let mut signal = vec![0.0; SIGNAL_LENGTH];
        signal[offset] = 1.0;
        let result = dwt_forward(&signal, wavelet, DWT_LEVELS);

        let level_energies: Vec<f64> = result.details.iter().map(|d| energy(d)).collect();
        let primary_level = primary_level_for_offset(offset);

        let primary_energy = level_energies
            .get(primary_level)
            .copied()
            .unwrap_or(0.0);

        let mut adjacent_energy = 0.0;
        if primary_level > 0 {
            adjacent_energy += level_energies
                .get(primary_level - 1)
                .copied()
                .unwrap_or(0.0);
        }
        adjacent_energy += level_energies
            .get(primary_level + 1)
            .copied()
            .unwrap_or(0.0);

        let leakage = if primary_energy > 0.0 {
            adjacent_energy / primary_energy
        } else {
            f64::INFINITY
        };
        leakages.push(leakage);
    }

    let mean_leakage = leakages.iter().sum::<f64>() / leakages.len() as f64;

    LeakageResult {
        wavelet_name: wavelet.name.to_string(),
        leakage_at_96: leakages[0],
        leakage_at_128: leakages[1],
        leakage_at_384: leakages[2],
        mean_leakage,
    }
}

// ── Analysis 4: Offset Localization ──

struct LocalizationResult {
    wavelet_name: String,
    score_at_96: f64,
    score_at_128: f64,
    score_at_384: f64,
    mean_score: f64,
}

fn analysis_offset_localization(wavelet: &WaveletFilters) -> LocalizationResult {
    let offsets = [96_usize, 128, 384];
    let mut scores = Vec::new();

    for &offset in &offsets {
        let mut signal = vec![0.0; SIGNAL_LENGTH];
        signal[offset] = 1.0;
        let result = dwt_forward(&signal, wavelet, DWT_LEVELS);

        let level_energies: Vec<f64> = result.details.iter().map(|d| energy(d)).collect();
        let approx_energy = energy(&result.approximation);
        let total_energy: f64 = level_energies.iter().sum::<f64>() + approx_energy;

        let max_level_energy = level_energies
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(approx_energy);

        let score = if total_energy > 0.0 {
            max_level_energy / total_energy
        } else {
            0.0
        };
        scores.push(score);
    }

    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

    LocalizationResult {
        wavelet_name: wavelet.name.to_string(),
        score_at_96: scores[0],
        score_at_128: scores[1],
        score_at_384: scores[2],
        mean_score,
    }
}

// ── Analysis 6: FFN Basis Alignment Score ──

struct FfnAlignmentResult {
    wavelet_name: String,
    compaction_unweighted: f64,
    compaction_weighted: f64,
}

fn kaiming_init_matrix(rows: usize, columns: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let standard_deviation = (2.0 / rows as f64).sqrt();
    let normal = Normal::new(0.0, standard_deviation).unwrap();
    (0..rows)
        .map(|_| (0..columns).map(|_| normal.sample(rng)).collect())
        .collect()
}

fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn relu(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| x.max(0.0)).collect()
}

fn analysis_ffn_alignment(
    unweighted_signal: &[f64],
    weighted_signal: &[f64],
    wavelet: &WaveletFilters,
) -> FfnAlignmentResult {
    let dimension = SIGNAL_LENGTH;
    let ffn_dimension = dimension * 4;
    let sample_count = 100;
    let mut rng = StdRng::seed_from_u64(42);

    let weight_up = kaiming_init_matrix(ffn_dimension, dimension, &mut rng);
    let weight_down = kaiming_init_matrix(dimension, ffn_dimension, &mut rng);

    let mut compute_compaction = |base_signal: &[f64]| -> f64 {
        let mut total_compaction = 0.0;
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..sample_count {
            let input: Vec<f64> = (0..dimension).map(|index| {
                base_signal[index] + normal.sample(&mut rng) * 0.1
            }).collect();

            let hidden = relu(&matrix_vector_multiply(&weight_up, &input));
            let output = matrix_vector_multiply(&weight_down, &hidden);

            let result = dwt_forward(&output, wavelet, DWT_LEVELS);
            let all_coefficients = all_coefficients_flat(&result);
            let total_energy = energy(&all_coefficients);

            if total_energy == 0.0 {
                continue;
            }

            let mut sorted_energies: Vec<f64> =
                all_coefficients.iter().map(|x| x * x).collect();
            sorted_energies.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let top_count = (sorted_energies.len() as f64 * 0.2).ceil() as usize;
            let top_energy: f64 = sorted_energies.iter().take(top_count).sum();

            total_compaction += top_energy / total_energy;
        }

        total_compaction / sample_count as f64
    };

    FfnAlignmentResult {
        wavelet_name: wavelet.name.to_string(),
        compaction_unweighted: compute_compaction(unweighted_signal),
        compaction_weighted: compute_compaction(weighted_signal),
    }
}

// ── Ranking and recommendation ──

struct RankingScores {
    wavelet_name: String,
    sparse_approx_score: f64,
    scale_space_score: f64,
    localization_score: f64,
    leakage_score: f64,
    ffn_alignment_score: f64,
    overall_score: f64,
}

fn compute_rankings(
    sparse_results: &[SparseApproximationResult],
    scale_results: &[ScaleSpaceResult],
    localization_results: &[LocalizationResult],
    leakage_results: &[LeakageResult],
    ffn_results: &[FfnAlignmentResult],
) -> Vec<RankingScores> {
    let wavelet_count = sparse_results.len();

    let sparse_k20: Vec<f64> = sparse_results
        .iter()
        .map(|r| r.mse_per_k.iter().find(|(k, _)| *k == 20).unwrap().1)
        .collect();

    let scale_concentration: Vec<f64> = scale_results
        .iter()
        .map(|r| r.dense_energy_fraction + r.sparse_mid_energy_fraction + r.sparse_long_energy_fraction)
        .collect();

    let localization_scores: Vec<f64> = localization_results
        .iter()
        .map(|r| r.mean_score)
        .collect();

    let leakage_scores: Vec<f64> = leakage_results.iter().map(|r| r.mean_leakage).collect();

    let ffn_scores: Vec<f64> = ffn_results.iter().map(|r| r.compaction_unweighted).collect();

    let normalize_lower_better = |values: &[f64]| -> Vec<f64> {
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        if (max - min).abs() < 1e-12 {
            return vec![1.0; values.len()];
        }
        values.iter().map(|v| 1.0 - (v - min) / (max - min)).collect()
    };

    let normalize_higher_better = |values: &[f64]| -> Vec<f64> {
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        if (max - min).abs() < 1e-12 {
            return vec![1.0; values.len()];
        }
        values.iter().map(|v| (v - min) / (max - min)).collect()
    };

    let normalized_sparse = normalize_lower_better(&sparse_k20);
    let normalized_scale = normalize_higher_better(&scale_concentration);
    let normalized_localization = normalize_higher_better(&localization_scores);
    let normalized_leakage = normalize_lower_better(&leakage_scores);
    let normalized_ffn = normalize_higher_better(&ffn_scores);

    (0..wavelet_count)
        .map(|index| {
            let overall = 0.25 * normalized_sparse[index]
                + 0.25 * normalized_scale[index]
                + 0.20 * normalized_localization[index]
                + 0.15 * normalized_leakage[index]
                + 0.15 * normalized_ffn[index];
            RankingScores {
                wavelet_name: sparse_results[index].wavelet_name.clone(),
                sparse_approx_score: normalized_sparse[index],
                scale_space_score: normalized_scale[index],
                localization_score: normalized_localization[index],
                leakage_score: normalized_leakage[index],
                ffn_alignment_score: normalized_ffn[index],
                overall_score: overall,
            }
        })
        .collect()
}

// ── Output formatting ──

fn format_results(
    sparse_results: &[SparseApproximationResult],
    scale_results: &[ScaleSpaceResult],
    localization_results: &[LocalizationResult],
    leakage_results: &[LeakageResult],
    ffn_results: &[FfnAlignmentResult],
    rankings: &[RankingScores],
) -> String {
    let mut output = String::new();

    output.push_str("=== Wavelet Basis Matching for DSQG d41_35m Offsets ===\n");
    output.push_str("Signal: Dirac comb at {1..48, 96, 128, 384}, length=2048\n\n");

    // ── Sparse Approximation ──
    output.push_str("SPARSE APPROXIMATION (reconstruction MSE, lower=better):\n");
    output.push_str(&format!(
        "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
        "Wavelet", "K=5", "K=10", "K=20", "K=30", "K=50", "K=100"
    ));
    for result in sparse_results {
        let values: Vec<String> = result
            .mse_per_k
            .iter()
            .map(|(_, mse)| format!("{:.2e}", mse))
            .collect();
        output.push_str(&format!(
            "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
            result.wavelet_name,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5]
        ));
    }
    output.push('\n');

    output.push_str("ENERGY CAPTURED BY TOP-K COEFFICIENTS:\n");
    output.push_str(&format!(
        "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
        "Wavelet", "K=5", "K=10", "K=20", "K=30", "K=50", "K=100"
    ));
    for result in sparse_results {
        let values: Vec<String> = result
            .energy_fraction_per_k
            .iter()
            .map(|(_, fraction)| format!("{:.1}%", fraction * 100.0))
            .collect();
        output.push_str(&format!(
            "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n",
            result.wavelet_name,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5]
        ));
    }
    output.push('\n');

    // ── Scale-Space Separation ──
    output.push_str("SCALE-SPACE SEPARATION (energy fraction per zone):\n");
    output.push_str(&format!(
        "{:<10} {:<16} {:<18} {:<18} {:<18}\n",
        "Wavelet", "L0-3(dense)", "L4-5(transition)", "L6-7(sparse_mid)", "L8+(sparse_long)"
    ));
    for result in scale_results {
        output.push_str(&format!(
            "{:<10} {:<16} {:<18} {:<18} {:<18}\n",
            result.wavelet_name,
            format!("{:.1}%", result.dense_energy_fraction * 100.0),
            format!("{:.1}%", result.transition_energy_fraction * 100.0),
            format!("{:.1}%", result.sparse_mid_energy_fraction * 100.0),
            format!("{:.1}%", result.sparse_long_energy_fraction * 100.0),
        ));
    }

    let best_separation = scale_results
        .iter()
        .max_by(|a, b| {
            let score_a = a.dense_energy_fraction + a.sparse_mid_energy_fraction + a.sparse_long_energy_fraction;
            let score_b = b.dense_energy_fraction + b.sparse_mid_energy_fraction + b.sparse_long_energy_fraction;
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap();
    output.push_str(&format!(
        "Best separation: {} (dense={:.1}%, mid={:.1}%, long={:.1}%)\n\n",
        best_separation.wavelet_name,
        best_separation.dense_energy_fraction * 100.0,
        best_separation.sparse_mid_energy_fraction * 100.0,
        best_separation.sparse_long_energy_fraction * 100.0,
    ));

    // ── Per-level detail ──
    output.push_str("DETAIL ENERGY PER DWT LEVEL (% of total):\n");
    output.push_str(&format!(
        "{:<10} {}\n",
        "Wavelet",
        (0..DWT_LEVELS)
            .map(|l| format!("L{:<4}", l))
            .collect::<Vec<_>>()
            .join(" ")
    ));
    for result in scale_results {
        let total: f64 = result.level_energies.iter().sum();
        let fractions: Vec<String> = result
            .level_energies
            .iter()
            .map(|e| {
                if total > 0.0 {
                    format!("{:>5.1}%", e / total * 100.0)
                } else {
                    format!("{:>5.1}%", 0.0)
                }
            })
            .collect();
        output.push_str(&format!(
            "{:<10} {}\n",
            result.wavelet_name,
            fractions.join(" ")
        ));
    }
    output.push('\n');

    // ── Offset Localization ──
    output.push_str("OFFSET LOCALIZATION SCORES (higher=better):\n");
    output.push_str(&format!(
        "{:<10} {:<14} {:<14} {:<14} {:<14}\n",
        "Wavelet", "δ=96", "δ=128", "δ=384", "mean"
    ));
    for result in localization_results {
        output.push_str(&format!(
            "{:<10} {:<14.4} {:<14.4} {:<14.4} {:<14.4}\n",
            result.wavelet_name,
            result.score_at_96,
            result.score_at_128,
            result.score_at_384,
            result.mean_score,
        ));
    }
    output.push('\n');

    // ── Cross-Scale Leakage ──
    output.push_str("CROSS-SCALE LEAKAGE (lower=better):\n");
    output.push_str(&format!(
        "{:<10} {:<16} {:<16} {:<16} {:<14}\n",
        "Wavelet", "leakage@δ=96", "leakage@δ=128", "leakage@δ=384", "mean"
    ));
    for result in leakage_results {
        output.push_str(&format!(
            "{:<10} {:<16.4} {:<16.4} {:<16.4} {:<14.4}\n",
            result.wavelet_name,
            result.leakage_at_96,
            result.leakage_at_128,
            result.leakage_at_384,
            result.mean_leakage,
        ));
    }
    output.push('\n');

    // ── FFN Alignment ──
    output.push_str("FFN OUTPUT ENERGY COMPACTION (top-20% of coefficients, higher=better):\n");
    output.push_str(&format!(
        "{:<10} {:<22} {:<22}\n",
        "Wavelet", "Unweighted signal", "Weighted signal"
    ));
    for result in ffn_results {
        output.push_str(&format!(
            "{:<10} {:<22} {:<22}\n",
            result.wavelet_name,
            format!("{:.1}%", result.compaction_unweighted * 100.0),
            format!("{:.1}%", result.compaction_weighted * 100.0),
        ));
    }
    output.push('\n');

    // ── Vanishing Moments ──
    output.push_str("VANISHING MOMENTS:\n");
    output.push_str("Haar=1, D4=2, D8=4, C6=2, S8=4, Bior2.2=2\n\n");

    // ── Ranking ──
    output.push_str("=== RANKING ===\n");
    output.push_str(&format!(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n",
        "Wavelet", "Sparse", "Scale", "Local", "Leakage", "FFN", "Overall"
    ));
    for ranking in rankings {
        output.push_str(&format!(
            "{:<10} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10.3}\n",
            ranking.wavelet_name,
            ranking.sparse_approx_score,
            ranking.scale_space_score,
            ranking.localization_score,
            ranking.leakage_score,
            ranking.ffn_alignment_score,
            ranking.overall_score,
        ));
    }
    output.push('\n');

    let best_sparse = rankings
        .iter()
        .max_by(|a, b| a.sparse_approx_score.partial_cmp(&b.sparse_approx_score).unwrap())
        .unwrap();
    let best_scale = rankings
        .iter()
        .max_by(|a, b| a.scale_space_score.partial_cmp(&b.scale_space_score).unwrap())
        .unwrap();
    let best_local = rankings
        .iter()
        .max_by(|a, b| a.localization_score.partial_cmp(&b.localization_score).unwrap())
        .unwrap();
    let best_leak = rankings
        .iter()
        .max_by(|a, b| a.leakage_score.partial_cmp(&b.leakage_score).unwrap())
        .unwrap();
    let best_ffn = rankings
        .iter()
        .max_by(|a, b| a.ffn_alignment_score.partial_cmp(&b.ffn_alignment_score).unwrap())
        .unwrap();
    let best_overall = rankings
        .iter()
        .max_by(|a, b| a.overall_score.partial_cmp(&b.overall_score).unwrap())
        .unwrap();

    output.push_str(&format!(
        "Best sparse approximation (K=20):   {}\n",
        best_sparse.wavelet_name
    ));
    output.push_str(&format!(
        "Best scale-space separation:        {}\n",
        best_scale.wavelet_name
    ));
    output.push_str(&format!(
        "Best offset localization:           {}\n",
        best_local.wavelet_name
    ));
    output.push_str(&format!(
        "Least cross-scale leakage:          {}\n",
        best_leak.wavelet_name
    ));
    output.push_str(&format!(
        "Best FFN alignment:                 {}\n",
        best_ffn.wavelet_name
    ));
    output.push_str(&format!(
        "Overall winner (weighted average):  {}\n\n",
        best_overall.wavelet_name
    ));

    // ── Recommendation ──
    output.push_str("=== RECOMMENDATION ===\n");

    let winner = &best_overall.wavelet_name;
    let winner_vanishing = all_wavelets()
        .iter()
        .find(|w| w.name == winner.as_str())
        .map(|w| w.vanishing_moments)
        .unwrap_or(0);

    output.push_str(&format!("Recommended wavelet for DSQG FFN:  {}\n", winner));
    output.push_str(&format!(
        "Reason: {} ({} vanishing moments) achieves the best overall balance across \
        sparse approximation quality, scale-space separation of DSQG's three zones \
        (local δ=1..48, mid δ=96/128, long δ=384), and offset localization. \
        Its filter length provides good time-frequency tradeoff for the multi-scale \
        structure of the d41_35m attention pattern.\n\n",
        winner, winner_vanishing
    ));

    let simpler_candidates: Vec<&RankingScores> = rankings
        .iter()
        .filter(|r| {
            (r.wavelet_name == "Haar" || r.wavelet_name == "D4")
                && r.wavelet_name != *winner
        })
        .collect();
    let simpler = if simpler_candidates.is_empty() {
        rankings
            .iter()
            .filter(|r| r.wavelet_name != *winner)
            .max_by(|a, b| a.overall_score.partial_cmp(&b.overall_score).unwrap())
            .unwrap()
    } else {
        simpler_candidates
            .iter()
            .max_by(|a, b| a.overall_score.partial_cmp(&b.overall_score).unwrap())
            .unwrap()
    };

    output.push_str(&format!(
        "Alternative if implementation complexity matters: {}\n",
        simpler.wavelet_name
    ));
    output.push_str(&format!(
        "Reason: {} has short filters ({} taps) making it efficient for real-time inference.\n\n",
        simpler.wavelet_name,
        if simpler.wavelet_name == "Haar" {
            2
        } else {
            4
        }
    ));

    output.push_str(&format!(
        "For a wavelet FFN: use {} levels of decomposition (captures all three DSQG zones)\n",
        DWT_LEVELS
    ));
    output.push_str(
        "Expected benefit vs standard FFN: medium (natural multi-scale basis alignment \
        reduces the number of parameters needed to represent scale-separated features)\n",
    );
    output.push_str("Confidence: MEDIUM\n");

    output
}

fn main() {
    println!("🌊 Wavelet Basis Matching for DSQG d41_35m");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let unweighted_signal = build_unweighted_signal();
    let weighted_signal = build_weighted_signal();
    let wavelets = all_wavelets();

    println!("▸ Verifying DWT round-trip reconstruction...");
    for wavelet in &wavelets {
        let result = dwt_forward(&unweighted_signal, wavelet, DWT_LEVELS);
        let reconstructed = dwt_inverse(&result, wavelet);
        let round_trip_error: f64 = unweighted_signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / energy(&unweighted_signal);
        let method = if wavelet.use_lifting { "lifting" } else { "filterbank" };
        println!(
            "  {:<10} ({:<11}) round-trip MSE = {:.2e}",
            wavelet.name, method, round_trip_error
        );
        assert!(
            round_trip_error < 1e-10,
            "{} DWT round-trip failed: MSE = {:.2e}",
            wavelet.name,
            round_trip_error
        );
    }
    println!("  All wavelets pass round-trip verification.\n");

    println!("▸ Running Analysis 1: Sparse Approximation Quality...");
    let sparse_results: Vec<SparseApproximationResult> = wavelets
        .iter()
        .map(|w| analysis_sparse_approximation(&unweighted_signal, w))
        .collect();

    println!("▸ Running Analysis 2: Scale-Space Decomposition...");
    let scale_results: Vec<ScaleSpaceResult> = wavelets
        .iter()
        .map(|w| analysis_scale_space(&unweighted_signal, w))
        .collect();

    println!("▸ Running Analysis 3: Cross-Scale Leakage...");
    let leakage_results: Vec<LeakageResult> = wavelets
        .iter()
        .map(|w| analysis_cross_scale_leakage(w))
        .collect();

    println!("▸ Running Analysis 4: Offset Localization...");
    let localization_results: Vec<LocalizationResult> = wavelets
        .iter()
        .map(|w| analysis_offset_localization(w))
        .collect();

    println!("▸ Running Analysis 5: Vanishing Moments (theoretical)...");

    println!("▸ Running Analysis 6: FFN Basis Alignment Score...");
    let ffn_results: Vec<FfnAlignmentResult> = wavelets
        .iter()
        .map(|w| analysis_ffn_alignment(&unweighted_signal, &weighted_signal, w))
        .collect();

    println!("▸ Computing rankings...\n");
    let rankings = compute_rankings(
        &sparse_results,
        &scale_results,
        &localization_results,
        &leakage_results,
        &ffn_results,
    );

    let output = format_results(
        &sparse_results,
        &scale_results,
        &localization_results,
        &leakage_results,
        &ffn_results,
        &rankings,
    );

    print!("{}", output);

    let output_path = "results.txt";
    let mut file = fs::File::create(output_path).expect("Failed to create results.txt");
    file.write_all(output.as_bytes())
        .expect("Failed to write results.txt");
    println!("\n✅ Results saved to {}", output_path);
}
