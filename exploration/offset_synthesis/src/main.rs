use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::fs;
use std::io::Write;

const NUM_OFFSETS: usize = 52;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 32;
const EMBEDDING_DIM: usize = 256;
const FFN_DIM: usize = 1024;
const MONTE_CARLO_SAMPLES: usize = 10_000;
const NOISE_SIGMA: f64 = 0.5;
const IG_THRESHOLD: f64 = 0.05;

fn build_offsets() -> [usize; NUM_OFFSETS] {
    let mut offsets = [0usize; NUM_OFFSETS];
    for i in 0..48 {
        offsets[i] = i + 1;
    }
    offsets[48] = 96;
    offsets[49] = 128;
    offsets[50] = 384;
    offsets[51] = 512;
    offsets
}

fn offset_zone(delta: usize) -> &'static str {
    if delta <= 48 {
        "dense"
    } else if delta <= 128 {
        "mid"
    } else {
        "long"
    }
}

fn compute_pos_bias(offsets: &[usize; NUM_OFFSETS]) -> [[f64; NUM_HEADS]; NUM_OFFSETS] {
    let mut pos_bias = [[0.0f64; NUM_HEADS]; NUM_OFFSETS];
    for (i, &delta) in offsets.iter().enumerate() {
        let log_term = (delta as f64 + 1.0).ln();
        for h in 0..NUM_HEADS {
            if h == 0 {
                pos_bias[i][h] = 0.1 * log_term;
            } else {
                pos_bias[i][h] = -0.3 * log_term;
            }
        }
    }
    pos_bias
}

fn compute_scale_embed_sigma(offsets: &[usize; NUM_OFFSETS]) -> [[f64; NUM_HEADS]; NUM_OFFSETS] {
    let base_sigma = 0.15;
    let mut sigma = [[0.0f64; NUM_HEADS]; NUM_OFFSETS];
    for (i, &delta) in offsets.iter().enumerate() {
        let offset_factor = if delta > 48 { 2.5 } else { 1.0 };
        for h in 0..NUM_HEADS {
            let head_factor = if h == 0 {
                1.5
            } else if h == 7 {
                0.7
            } else {
                1.0
            };
            sigma[i][h] = base_sigma * offset_factor * head_factor;
        }
    }
    sigma
}

fn compute_expected_attention(
    pos_bias: &[[f64; NUM_HEADS]; NUM_OFFSETS],
    scale_sigma: &[[f64; NUM_HEADS]; NUM_OFFSETS],
) -> [[f64; NUM_HEADS]; NUM_OFFSETS] {
    let mut weights = [[0.0f64; NUM_HEADS]; NUM_OFFSETS];

    for h in 0..NUM_HEADS {
        let mut logits = [0.0f64; NUM_OFFSETS];
        for i in 0..NUM_OFFSETS {
            let scale_contribution = scale_sigma[i][h].powi(2) * 0.5 * (HEAD_DIM as f64);
            logits[i] = pos_bias[i][h] + scale_contribution;
        }

        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0;
        for i in 0..NUM_OFFSETS {
            logits[i] = (logits[i] - max_logit).exp();
            sum_exp += logits[i];
        }
        for i in 0..NUM_OFFSETS {
            weights[i][h] = logits[i] / sum_exp;
        }
    }
    weights
}

struct AttentionAnalysis {
    top_pairs: Vec<(usize, usize, f64)>,
    zone_distribution: [f64; 3],
    per_head_dominant: Vec<Vec<usize>>,
}

fn analysis_1(
    offsets: &[usize; NUM_OFFSETS],
    weights: &[[f64; NUM_HEADS]; NUM_OFFSETS],
) -> AttentionAnalysis {
    let mut all_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..NUM_OFFSETS {
        for h in 0..NUM_HEADS {
            all_pairs.push((offsets[i], h, weights[i][h]));
        }
    }
    all_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let top_pairs: Vec<(usize, usize, f64)> = all_pairs.iter().take(5).cloned().collect();

    let mut zone_weight = [0.0f64; 3];
    for i in 0..NUM_OFFSETS {
        let total_head_weight: f64 = (0..NUM_HEADS).map(|h| weights[i][h]).sum();
        let zone_idx = match offset_zone(offsets[i]) {
            "dense" => 0,
            "mid" => 1,
            _ => 2,
        };
        zone_weight[zone_idx] += total_head_weight;
    }
    let total: f64 = zone_weight.iter().sum();
    for z in &mut zone_weight {
        *z = *z / total * 100.0;
    }

    let mut per_head_dominant: Vec<Vec<usize>> = Vec::new();
    for h in 0..NUM_HEADS {
        let mut head_pairs: Vec<(usize, f64)> = (0..NUM_OFFSETS)
            .map(|i| (offsets[i], weights[i][h]))
            .collect();
        head_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let dominant: Vec<usize> = head_pairs.iter().take(3).map(|p| p.0).collect();
        per_head_dominant.push(dominant);
    }

    AttentionAnalysis {
        top_pairs,
        zone_distribution: zone_weight,
        per_head_dominant,
    }
}

struct CoAttendanceAnalysis {
    co_attendance_matrix: Vec<Vec<f64>>,
    lift_matrix: Vec<Vec<f64>>,
    marginal_probabilities: Vec<f64>,
    clusters: Vec<Vec<usize>>,
    isolated: Vec<usize>,
}

fn analysis_2(
    offsets: &[usize; NUM_OFFSETS],
    weights: &[[f64; NUM_HEADS]; NUM_OFFSETS],
) -> CoAttendanceAnalysis {
    let uniform_baseline = 1.0 / NUM_OFFSETS as f64;
    let noise_dist = Normal::new(0.0, NOISE_SIGMA).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    let mut co_attend_count = vec![vec![0.0f64; NUM_OFFSETS]; NUM_OFFSETS];
    let mut marginal_count = vec![0.0f64; NUM_OFFSETS];

    let total_events = (MONTE_CARLO_SAMPLES * NUM_HEADS) as f64;

    for _ in 0..MONTE_CARLO_SAMPLES {
        for h in 0..NUM_HEADS {
            let mut noisy_logits = [0.0f64; NUM_OFFSETS];
            for i in 0..NUM_OFFSETS {
                let epsilon: f64 = noise_dist.sample(&mut rng);
                noisy_logits[i] = (weights[i][h] * epsilon.exp()).max(1e-30);
            }

            let sum: f64 = noisy_logits.iter().sum();

            let mut active_in_head = [false; NUM_OFFSETS];
            for i in 0..NUM_OFFSETS {
                let noisy_weight = noisy_logits[i] / sum;
                active_in_head[i] = noisy_weight > uniform_baseline;
                if active_in_head[i] {
                    marginal_count[i] += 1.0;
                }
            }

            for i in 0..NUM_OFFSETS {
                if !active_in_head[i] {
                    continue;
                }
                for j in (i + 1)..NUM_OFFSETS {
                    if active_in_head[j] {
                        co_attend_count[i][j] += 1.0;
                        co_attend_count[j][i] += 1.0;
                    }
                }
            }
        }
    }

    let mut co_attend = vec![vec![0.0f64; NUM_OFFSETS]; NUM_OFFSETS];
    let mut marginal = vec![0.0f64; NUM_OFFSETS];
    for i in 0..NUM_OFFSETS {
        marginal[i] = marginal_count[i] / total_events;
        for j in 0..NUM_OFFSETS {
            co_attend[i][j] = co_attend_count[i][j] / total_events;
        }
    }

    let mut lift = vec![vec![0.0f64; NUM_OFFSETS]; NUM_OFFSETS];
    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            let expected = marginal[i] * marginal[j];
            if expected > 1e-10 {
                lift[i][j] = co_attend[i][j] / expected;
                lift[j][i] = lift[i][j];
            }
        }
    }

    let cluster_threshold = 1.1;

    let mut visited = vec![false; NUM_OFFSETS];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for start in 0..NUM_OFFSETS {
        if visited[start] {
            continue;
        }

        let mut cluster = vec![start];
        visited[start] = true;
        let mut queue = vec![start];

        while let Some(current) = queue.pop() {
            for neighbor in 0..NUM_OFFSETS {
                if !visited[neighbor] && lift[current][neighbor] > cluster_threshold {
                    visited[neighbor] = true;
                    cluster.push(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        clusters.push(cluster);
    }

    let mut isolated: Vec<usize> = Vec::new();
    let mut real_clusters: Vec<Vec<usize>> = Vec::new();

    for cluster in clusters {
        if cluster.len() == 1 {
            isolated.push(offsets[cluster[0]]);
        } else {
            let mut offset_cluster: Vec<usize> = cluster.iter().map(|&idx| offsets[idx]).collect();
            offset_cluster.sort();
            real_clusters.push(offset_cluster);
        }
    }
    isolated.sort();

    CoAttendanceAnalysis {
        co_attendance_matrix: co_attend,
        lift_matrix: lift,
        marginal_probabilities: marginal,
        clusters: real_clusters,
        isolated,
    }
}

struct InformationGainAnalysis {
    ig_matrix: Vec<Vec<f64>>,
    high_ig_pairs: Vec<(usize, usize, f64)>,
    total_high_ig_count: usize,
    total_ig: f64,
}

fn analysis_3(
    offsets: &[usize; NUM_OFFSETS],
    co_attend: &[Vec<f64>],
    scale_sigma: &[[f64; NUM_HEADS]; NUM_OFFSETS],
) -> InformationGainAnalysis {
    let base_snr = 1.0;

    let mut ig_matrix = vec![vec![0.0f64; NUM_OFFSETS]; NUM_OFFSETS];

    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            let co_attend_prob = co_attend[i][j];

            let sigma_i: f64 = (0..NUM_HEADS).map(|h| scale_sigma[i][h]).sum::<f64>() / NUM_HEADS as f64;
            let sigma_j: f64 = (0..NUM_HEADS).map(|h| scale_sigma[j][h]).sum::<f64>() / NUM_HEADS as f64;

            let correlation = co_attend_prob.sqrt().min(0.99);
            let snr = base_snr * sigma_i * sigma_j;
            let ig = 0.5 * (1.0 + correlation * correlation * snr).ln();

            ig_matrix[i][j] = ig;
            ig_matrix[j][i] = ig;
        }
    }

    let mut high_ig_pairs: Vec<(usize, usize, f64)> = Vec::new();
    let mut total_ig = 0.0;
    let mut total_high_ig_count = 0;

    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            if ig_matrix[i][j] > IG_THRESHOLD {
                high_ig_pairs.push((offsets[i], offsets[j], ig_matrix[i][j]));
                total_ig += ig_matrix[i][j];
                total_high_ig_count += 1;
            }
        }
    }

    high_ig_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    InformationGainAnalysis {
        ig_matrix,
        high_ig_pairs,
        total_high_ig_count,
        total_ig,
    }
}

struct DesignComparison {
    name: &'static str,
    parameters: usize,
    versus_ffn_percent: f64,
    expected_ig: f64,
    flops_relative: f64,
    effective_rank: usize,
}

fn analysis_4(
    offsets: &[usize; NUM_OFFSETS],
    ig_analysis: &InformationGainAnalysis,
) -> Vec<DesignComparison> {
    let ffn_params = 2 * EMBEDDING_DIM * FFN_DIM;
    let ffn_flops = ffn_params;

    let dense_params = NUM_OFFSETS * NUM_OFFSETS * HEAD_DIM;
    let dense_ig: f64 = ig_analysis
        .ig_matrix
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().filter(move |(j, _)| *j > i))
        .map(|(_, &v)| v)
        .sum();
    let dense_flops = (NUM_OFFSETS * NUM_OFFSETS * HEAD_DIM) as f64 / ffn_flops as f64;
    let dense_rank = NUM_OFFSETS;

    let sparse_count = ig_analysis.total_high_ig_count;
    let sparse_params = sparse_count * HEAD_DIM * 2;
    let sparse_ig = ig_analysis.total_ig;
    let sparse_flops = (sparse_count * HEAD_DIM) as f64 / ffn_flops as f64;
    let sparse_rank = sparse_count.min(NUM_OFFSETS);

    let dense_zone_size = 48usize;
    let mid_zone_size = 2usize;
    let zone_params =
        dense_zone_size * dense_zone_size * HEAD_DIM + mid_zone_size * mid_zone_size * HEAD_DIM;
    let mut zone_ig = 0.0f64;
    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            let zi = offset_zone(offsets[i]);
            let zj = offset_zone(offsets[j]);
            if zi == zj {
                zone_ig += ig_analysis.ig_matrix[i][j];
            }
        }
    }
    let zone_flops = ((dense_zone_size * dense_zone_size + mid_zone_size * mid_zone_size)
        * HEAD_DIM) as f64
        / ffn_flops as f64;
    let zone_rank = dense_zone_size;

    let cross_zone_params = 3 * 3 * HEAD_DIM;
    let hier_params = zone_params + cross_zone_params;
    let hier_ig = dense_ig;
    let hier_flops = zone_flops
        + (cross_zone_params as f64 / ffn_flops as f64);
    let hier_rank = dense_zone_size + 3;

    let mut designs = Vec::new();
    designs.push(DesignComparison {
        name: "A(dense)",
        parameters: dense_params,
        versus_ffn_percent: dense_params as f64 / ffn_params as f64 * 100.0,
        expected_ig: dense_ig,
        flops_relative: dense_flops,
        effective_rank: dense_rank,
    });
    designs.push(DesignComparison {
        name: "B(sparse)",
        parameters: sparse_params,
        versus_ffn_percent: sparse_params as f64 / ffn_params as f64 * 100.0,
        expected_ig: sparse_ig,
        flops_relative: sparse_flops,
        effective_rank: sparse_rank,
    });
    designs.push(DesignComparison {
        name: "C(zone)",
        parameters: zone_params,
        versus_ffn_percent: zone_params as f64 / ffn_params as f64 * 100.0,
        expected_ig: zone_ig,
        flops_relative: zone_flops,
        effective_rank: zone_rank,
    });
    designs.push(DesignComparison {
        name: "D(hier)",
        parameters: hier_params,
        versus_ffn_percent: hier_params as f64 / ffn_params as f64 * 100.0,
        expected_ig: hier_ig,
        flops_relative: hier_flops,
        effective_rank: hier_rank,
    });
    designs.push(DesignComparison {
        name: "FFN(ref)",
        parameters: ffn_params,
        versus_ffn_percent: 100.0,
        expected_ig: 0.0,
        flops_relative: 1.0,
        effective_rank: FFN_DIM.min(EMBEDDING_DIM),
    });

    designs
}

fn format_results(
    offsets: &[usize; NUM_OFFSETS],
    attention: &AttentionAnalysis,
    co_attend_analysis: &CoAttendanceAnalysis,
    ig_analysis: &InformationGainAnalysis,
    designs: &[DesignComparison],
) -> String {
    let mut output = String::new();

    output.push_str("=== Offset Synthesis Layer Analysis (J=52, H=8, D=256) ===\n\n");

    output.push_str("ATTENTION DISTRIBUTION MODEL:\n");
    output.push_str("Top-5 (offset, head) pairs by expected weight:\n");
    for &(delta, h, w) in &attention.top_pairs {
        output.push_str(&format!("  δ={} h={}: {:.4}\n", delta, h, w));
    }
    output.push_str(&format!(
        "Zone distribution: dense={:.1}%, mid={:.1}%, long={:.1}%\n",
        attention.zone_distribution[0],
        attention.zone_distribution[1],
        attention.zone_distribution[2]
    ));
    output.push_str("Per-head dominant offsets: ");
    for (h, dominant) in attention.per_head_dominant.iter().enumerate() {
        if h > 0 {
            output.push_str(", ");
        }
        output.push_str(&format!(
            "h{}=[δ={}]",
            h,
            dominant
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    output.push_str("\n\n");

    let mut all_lifts: Vec<f64> = Vec::new();
    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            all_lifts.push(co_attend_analysis.lift_matrix[i][j]);
        }
    }
    all_lifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lift_min = all_lifts[0];
    let lift_max = all_lifts[all_lifts.len() - 1];
    let lift_median = all_lifts[all_lifts.len() / 2];
    let lift_mean = all_lifts.iter().sum::<f64>() / all_lifts.len() as f64;
    let lift_p90 = all_lifts[(all_lifts.len() as f64 * 0.90) as usize];
    let lift_p95 = all_lifts[(all_lifts.len() as f64 * 0.95) as usize];

    output.push_str("CO-ATTENDANCE ANALYSIS:\n");
    output.push_str("Marginal activation probabilities (P(active) per offset zone):\n");
    let dense_marginals: Vec<f64> = (0..48)
        .map(|i| co_attend_analysis.marginal_probabilities[i])
        .collect();
    let sparse_marginals: Vec<f64> = (48..NUM_OFFSETS)
        .map(|i| co_attend_analysis.marginal_probabilities[i])
        .collect();
    let dense_mean = dense_marginals.iter().sum::<f64>() / dense_marginals.len() as f64;
    let sparse_mean = sparse_marginals.iter().sum::<f64>() / sparse_marginals.len() as f64;
    output.push_str(&format!(
        "  Dense offsets (1-48): mean P(active)={:.4}\n",
        dense_mean
    ));
    output.push_str(&format!(
        "  Sparse offsets (96-512): mean P(active)={:.4}\n",
        sparse_mean
    ));

    output.push_str(&format!(
        "\nLift distribution (observed/expected co-attendance, 1326 pairs):\n"
    ));
    output.push_str(&format!(
        "  min={:.4}, median={:.4}, mean={:.4}, p90={:.4}, p95={:.4}, max={:.4}\n",
        lift_min, lift_median, lift_mean, lift_p90, lift_p95, lift_max
    ));

    let above_1_1 = all_lifts.iter().filter(|&&l| l > 1.1).count();
    let above_1_05 = all_lifts.iter().filter(|&&l| l > 1.05).count();
    let below_0_9 = all_lifts.iter().filter(|&&l| l < 0.9).count();
    output.push_str(&format!(
        "  Pairs with lift > 1.10: {} ({:.1}%)\n",
        above_1_1,
        above_1_1 as f64 / 1326.0 * 100.0
    ));
    output.push_str(&format!(
        "  Pairs with lift > 1.05: {} ({:.1}%)\n",
        above_1_05,
        above_1_05 as f64 / 1326.0 * 100.0
    ));
    output.push_str(&format!(
        "  Pairs with lift < 0.90: {} ({:.1}%) [anti-correlated due to softmax competition]\n",
        below_0_9,
        below_0_9 as f64 / 1326.0 * 100.0
    ));

    let mut lift_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..NUM_OFFSETS {
        for j in (i + 1)..NUM_OFFSETS {
            lift_pairs.push((offsets[i], offsets[j], co_attend_analysis.lift_matrix[i][j]));
        }
    }
    lift_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    output.push_str("  Top-10 lift pairs:\n");
    for pair in lift_pairs.iter().take(10) {
        output.push_str(&format!(
            "    (δ={}, δ={}): lift={:.4}\n",
            pair.0, pair.1, pair.2
        ));
    }

    output.push_str(&format!(
        "\nCO-ATTENDANCE CLUSTERS (lift threshold: {:.1}x above independence):\n",
        1.1
    ));
    for (idx, cluster) in co_attend_analysis.clusters.iter().enumerate() {
        let mut pair_count = 0usize;
        let mut mean_lift = 0.0f64;

        for (ci, &oi) in cluster.iter().enumerate() {
            for &oj in cluster.iter().skip(ci + 1) {
                let ii = offsets.iter().position(|&o| o == oi).unwrap();
                let ij = offsets.iter().position(|&o| o == oj).unwrap();
                pair_count += 1;
                mean_lift += co_attend_analysis.lift_matrix[ii][ij];
            }
        }
        if pair_count > 0 {
            mean_lift /= pair_count as f64;
        }

        let display_offsets = if cluster.len() > 10 {
            let first: Vec<String> = cluster.iter().take(5).map(|d| d.to_string()).collect();
            let last: Vec<String> = cluster.iter().rev().take(3).rev().map(|d| d.to_string()).collect();
            format!("{}, ...{} more..., {}", first.join(", "), cluster.len() - 8, last.join(", "))
        } else {
            cluster.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        };

        output.push_str(&format!(
            "Cluster {}: [{}] ({} pairs, mean lift={:.4})\n",
            idx + 1,
            display_offsets,
            pair_count,
            mean_lift
        ));
    }
    if !co_attend_analysis.isolated.is_empty() {
        let isolated_display = if co_attend_analysis.isolated.len() > 15 {
            let count = co_attend_analysis.isolated.len();
            let first: Vec<String> = co_attend_analysis.isolated.iter().take(5).map(|d| d.to_string()).collect();
            let last: Vec<String> = co_attend_analysis.isolated.iter().rev().take(3).rev().map(|d| d.to_string()).collect();
            format!("{}, ...{} more..., {}", first.join(", "), count - 8, last.join(", "))
        } else {
            co_attend_analysis.isolated.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        };
        output.push_str(&format!(
            "Isolated offsets ({} total): [{}]\n",
            co_attend_analysis.isolated.len(),
            isolated_display,
        ));
    }
    output.push_str("\n");

    output.push_str("INFORMATION GAIN SUMMARY:\n");
    output.push_str(&format!(
        "Total pairs with IG > {:.2}: {} / 1326\n",
        IG_THRESHOLD, ig_analysis.total_high_ig_count
    ));
    output.push_str("High-IG pairs (top 10):\n");
    for pair in ig_analysis.high_ig_pairs.iter().take(10) {
        output.push_str(&format!(
            "  (δ={}, δ={}): IG={:.4}\n",
            pair.0, pair.1, pair.2
        ));
    }
    output.push_str(&format!(
        "Expected information from synthesis: {:.4} nats total (if all high-IG pairs mixed)\n\n",
        ig_analysis.total_ig
    ));

    output.push_str("SYNTHESIS ARCHITECTURE COMPARISON:\n");
    output.push_str(&format!(
        "{:<12} {:>10} {:>10} {:>12} {:>12} {:>10}\n",
        "Design", "Params", "vs_FFN", "ExpectedIG", "FLOPs(rel)", "Eff_rank"
    ));
    for d in designs {
        let ig_str = if d.expected_ig == 0.0 {
            "N/A".to_string()
        } else {
            format!("{:.4} nats", d.expected_ig)
        };
        output.push_str(&format!(
            "{:<12} {:>10} {:>9.1}% {:>12} {:>11.2}x {:>10}\n",
            d.name, d.parameters, d.versus_ffn_percent, ig_str, d.flops_relative, d.effective_rank
        ));
    }
    output.push_str("\n");

    let best_design = &designs[0..4]
        .iter()
        .max_by(|a, b| {
            let score_a = a.expected_ig / (a.parameters as f64 + 1.0);
            let score_b = b.expected_ig / (b.parameters as f64 + 1.0);
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap();

    let recommended = if best_design.name.contains("sparse") {
        "B"
    } else if best_design.name.contains("zone") {
        "C"
    } else if best_design.name.contains("hier") {
        "D"
    } else {
        "A"
    };

    output.push_str("=== RECOMMENDATION ===\n");
    output.push_str(&format!(
        "Recommended synthesis design: {} ({})\n",
        recommended, best_design.name
    ));

    let reason = match recommended {
        "B" => "Sparse mixing captures the vast majority of information gain at a fraction of the parameter cost. Most offset pairs contribute negligible mutual information, making dense mixing wasteful.",
        "C" => "Zone-based mixing captures within-zone correlations (which dominate) while avoiding the parameter cost of cross-zone connections that carry little information.",
        "D" => "Hierarchical mixing captures both within-zone and cross-zone correlations. The two-stage design provides nearly full information gain with modest parameter overhead over zone-only mixing.",
        _ => "Dense mixing captures all correlations but at significant parameter cost. The analysis suggests most information gain is concentrated in a subset of pairs.",
    };
    output.push_str(&format!("Reason: {}\n\n", reason));

    output.push_str("Is synthesis worth it vs FiLM-conditioned FFN?\n");
    output.push_str("  FiLM-conditioned FFN: 524,800 params, indirect conditioning\n");
    output.push_str(&format!(
        "  {} ({}): {} params, direct offset synthesis\n",
        recommended, best_design.name, best_design.parameters
    ));
    output.push_str(&format!(
        "  Expected benefit of synthesis over FiLM: {:.4} nats additional mutual information,\n",
        best_design.expected_ig
    ));
    output.push_str(&format!(
        "    at {:.1}% of FFN parameter cost ({:.2}x FLOPs).\n",
        best_design.versus_ffn_percent, best_design.flops_relative
    ));

    let synthesis_density = ig_analysis.total_high_ig_count as f64 / 1326.0;
    if synthesis_density < 0.3 {
        output.push_str("  The co-attendance graph is sparse — synthesis primarily benefits nearby offsets.\n");
        output.push_str("  FiLM conditioning may already capture this implicitly through positional structure.\n");
        output.push_str("  Net advantage of dedicated synthesis: MODERATE — worth trying as a cheap add-on,\n");
        output.push_str("  but unlikely to replace FiLM-conditioned FFN entirely.\n\n");
    } else {
        output.push_str("  The co-attendance graph is dense — many offset pairs carry correlated info.\n");
        output.push_str("  Dedicated synthesis captures cross-offset structure that FiLM cannot.\n");
        output.push_str("  Net advantage of dedicated synthesis: HIGH — should complement FiLM.\n\n");
    }

    output.push_str("Implementation priority: MEDIUM — implement after validating FiLM-conditioned FFN baseline,\n");
    output.push_str("  but before wavelet FFN (synthesis is architecturally simpler and provides a direct test\n");
    output.push_str("  of whether cross-offset mixing improves over position-wise processing).\n");
    output.push_str(&format!(
        "Confidence: MEDIUM (analytical model uses calibrated parameters from ep3 training,\n  but real IG depends on downstream task structure not captured here)\n"
    ));

    output
}

fn main() {
    let offsets = build_offsets();

    eprintln!("⚙ Computing pos_bias model...");
    let pos_bias = compute_pos_bias(&offsets);

    eprintln!("⚙ Computing scale_embed sigma model...");
    let scale_sigma = compute_scale_embed_sigma(&offsets);

    eprintln!("⚙ Computing expected attention weights...");
    let weights = compute_expected_attention(&pos_bias, &scale_sigma);

    eprintln!("📊 Analysis 1: Attention Distribution Model");
    let attention = analysis_1(&offsets, &weights);

    eprintln!("📊 Analysis 2: Co-Attendance Probability Matrix (Monte Carlo N={})...", MONTE_CARLO_SAMPLES);
    let co_attend_analysis = analysis_2(&offsets, &weights);

    eprintln!("📊 Analysis 3: Information Gain from Synthesis...");
    let ig_analysis = analysis_3(&offsets, &co_attend_analysis.co_attendance_matrix, &scale_sigma);

    eprintln!("📊 Analysis 4: Synthesis Layer Architecture Comparison...");
    let designs = analysis_4(&offsets, &ig_analysis);

    eprintln!("📝 Formatting results...");
    let results = format_results(&offsets, &attention, &co_attend_analysis, &ig_analysis, &designs);

    print!("{}", results);

    let output_path = "results.txt";
    let mut file = fs::File::create(output_path).expect("Failed to create results.txt");
    file.write_all(results.as_bytes())
        .expect("Failed to write results.txt");
    eprintln!("✅ Results written to {}", output_path);
}
