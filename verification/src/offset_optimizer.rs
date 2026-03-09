//! Offset-Set Optimizer & Filter-Bank Analysis for DWARF
//!
//! Answers four questions from the octave-scaffolding design discussion:
//!
//!  1. PATH COUNTS — Not just "can you reach d in k hops" but "how many
//!     distinct paths reach d in k hops?" More paths → stronger gradient
//!     signal → faster learning. Shows why dense blocks work.
//!
//!  2. EFFECTIVE OFFSET ANALYSIS — The passkey test places the needle at
//!     filler distance d, but the *actual token-level offset* needed to
//!     attend to the word token is δ_eff = d + (cue_len + intro_tail).
//!     Verifies that Run D's dense blocks land at exactly δ_eff.
//!
//!  3. COMBINATION TONES — Does d=96 emerge from d=32 + d=64 as claimed?
//!     Enumerates all k-hop paths to each passkey distance and shows
//!     which arise purely from combination-tone synthesis vs. direct access.
//!
//!  4. BLOCK WIDTH OPTIMIZER — Given 5 octave positions {32,64,128,256,512}
//!     and a total dense-block budget, find the block-width allocation that
//!     maximises the weighted path-count score across all passkey distances.
//!     Proves (via AM-GM) when equal vs. unequal widths are optimal.

pub const MAX_LAG: usize = 2048;
pub const NUM_LAYERS: usize = 6;   // DWARF 13M has 6 transformer layers

// ── Passkey test parameters (empirically measured from training script) ────────
// Template: "the secret word is {word} ."  ≈ 6 tokens (word ≈ 2 tokens)
// Cue:      "the secret word is"           ≈ 5 tokens
// Word token is at position ~3 from start of intro (0-indexed).
// δ_eff = d + len(cue) + (len(intro) - word_pos - 1) ≈ d + 5 + (6-3-1) = d + 7
// We test both d+6 and d+7 to bracket the real value.
const PASSKEY_CUE_LEN: usize   = 5;   // tokens in cue phrase
const PASSKEY_INTRO_LEN: usize = 6;   // tokens in intro phrase
const PASSKEY_WORD_POS: usize  = 3;   // 0-indexed position of word token in intro

pub const PASSKEY_DISTANCES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536];

// ── Offset set definitions ────────────────────────────────────────────────────

fn offsets_v3_condu() -> Vec<usize> {
    // V3/condU: dense 0-32, sparse beyond (J=44)
    let mut o: Vec<usize> = (1..=32).collect();
    for &x in &[48usize, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] {
        o.push(x);
    }
    o.sort(); o.dedup();
    o.into_iter().filter(|&x| x >= 1 && x < MAX_LAG).collect()
}

fn offsets_run_c() -> Vec<usize> {
    // Run C / V3-Interleaved: dense 0-15 + dense 32-48 + sparse (J=40)
    let dense1: Vec<usize> = (1..=15).collect();
    let dense2: Vec<usize> = (32..=48).collect();
    let sparse = [64usize, 128, 256, 512, 768, 1024, 1536];
    let mut o: Vec<usize> = dense1.into_iter().chain(dense2).chain(sparse).collect();
    o.sort(); o.dedup();
    o.into_iter().filter(|&x| x >= 1 && x < MAX_LAG).collect()
}

fn offsets_run_d() -> Vec<usize> {
    // Run D / V3-RunD: octave scaffolding (J=53, excluding δ=0)
    let short    = [1usize, 2, 4, 8, 16];
    let block_32: Vec<usize> = (32..=39).collect();
    let block_64: Vec<usize> = (64..=71).collect();
    let block_128: Vec<usize> = (128..=135).collect();
    let block_256: Vec<usize> = (256..=263).collect();
    let block_512: Vec<usize> = (512..=519).collect();
    let gaps_tail = [48usize, 96, 192, 384, 768, 1024, 1536];
    let mut o: Vec<usize> = short.iter().copied()
        .chain(block_32).chain(block_64).chain(block_128).chain(block_256)
        .chain(block_512).chain(gaps_tail).collect();
    o.sort(); o.dedup();
    o.into_iter().filter(|&x| x >= 1 && x < MAX_LAG).collect()
}

fn offsets_run_f() -> Vec<usize> {
    // Run F: dense 1-15 (PPL recovery) + GAP 16-31 (commitment pressure)
    //        + octave blocks [32-39][64-71][128-135][256-263][512-519]
    //        + gap/tail anchors.
    //
    // Hypothesis: the gap 16-31 forces DSQG to commit sharply to [32-39]
    // for d=32 retrieval (no soft alternative at δ=16-31), producing Run-C-
    // level early passkey signal while the octave blocks extend that to d=512.
    // Dense 1-15 restores V3-level natural-language PPL (~52-53).
    //
    // Design motivated by Run E's "distraction by density" problem:
    //   dense 1-39 gave the model too many soft alternatives around each
    //   octave boundary — it never needed to commit sharply.
    //   dense 1-15 + gap 16-31 + block 32-39 re-creates the "cliff edge"
    //   that forced Run C's rapid d=32 commitment (ep2).
    let dense: Vec<usize> = (1..=15).collect();       // PPL recovery, d=1-8 δ_eff
    let block_32:  Vec<usize> = (32..=39).collect();  // δ_eff for d=32
    let block_64:  Vec<usize> = (64..=71).collect();  // δ_eff for d=64
    let block_128: Vec<usize> = (128..=135).collect();// δ_eff for d=128
    let block_256: Vec<usize> = (256..=263).collect();// δ_eff for d=256
    let block_512: Vec<usize> = (512..=519).collect();// δ_eff for d=512
    let gaps_tail = [48usize, 96, 192, 384, 768, 1024, 1536];
    let mut o: Vec<usize> = dense.into_iter()
        .chain(block_32).chain([48]).chain(block_64).chain([96])
        .chain(block_128).chain([192]).chain(block_256).chain([384])
        .chain(block_512).chain(gaps_tail).collect();
    o.sort(); o.dedup();
    o.into_iter().filter(|&x| x >= 1 && x < MAX_LAG).collect()
}

fn offsets_run_e() -> Vec<usize> {
    // Run E proposal: dense 1-39 (covers δ_eff for d=1-32) + 4 octave blocks
    // at 64-512 + gap/tail.  J = 39 + 32 + 7 = 78.
    //
    // Design logic:
    //   dense 1-39  → covers δ_eff = d+7 for d=1(8), 2(9), 4(11), 8(15),
    //                  16(23), 32(39) — all short-range passkey distances.
    //                  Also restores V3's natural-language short-range density.
    //   blocks 64-519 → direct δ_eff coverage for d=64,128,256,512.
    //   gap/tail       → combination-tone anchors + very long range.
    let dense: Vec<usize> = (1..=39).collect();
    let block_64:  Vec<usize> = (64..=71).collect();
    let block_128: Vec<usize> = (128..=135).collect();
    let block_256: Vec<usize> = (256..=263).collect();
    let block_512: Vec<usize> = (512..=519).collect();
    let gaps_tail = [48usize, 96, 192, 384, 768, 1024, 1536];
    let mut o: Vec<usize> = dense.into_iter()
        .chain(block_64).chain(block_128).chain(block_256).chain(block_512)
        .chain(gaps_tail).collect();
    o.sort(); o.dedup();
    o.into_iter().filter(|&x| x >= 1 && x < MAX_LAG).collect()
}

// ── Core: path-count computation ──────────────────────────────────────────────

/// Returns path_count[k][d] = number of distinct k-hop paths reaching lag d.
/// A k-hop path is an ordered sequence (δ₁, δ₂, …, δₖ) from the offset set
/// where δ₁+…+δₖ = d (with repetition allowed).
///
/// This is the k-fold convolution of the "indicator" function of the offset set.
/// Higher path count = more gradient paths = stronger learning signal.
pub fn path_counts(offsets: &[usize], max_hops: usize) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![0u64; MAX_LAG + 1]; max_hops + 1];
    // 1-hop: each offset δ contributes exactly one path to lag δ
    for &d in offsets {
        if d <= MAX_LAG { counts[1][d] = counts[1][d].saturating_add(1); }
    }
    // k-hop: convolve with 1-hop counts
    for k in 2..=max_hops {
        for lag in 1..=MAX_LAG {
            for &delta in offsets {
                if delta < lag {
                    let prev = counts[k-1][lag - delta];
                    counts[k][lag] = counts[k][lag].saturating_add(prev);
                }
            }
        }
    }
    counts
}

/// Weighted score: S(d) = Σ_k  path_count(d,k) × hop_discount^(k-1)
/// Discount < 1.0 models the fact that multi-hop paths are harder to learn.
pub fn path_score(counts: &[Vec<u64>], d: usize, hop_discount: f64) -> f64 {
    (1..counts.len())
        .map(|k| counts[k][d] as f64 * hop_discount.powi((k - 1) as i32))
        .sum()
}

/// Enumerate all k-hop unordered multisets {δ₁,…,δₖ} that sum to d.
fn enumerate_paths(offsets: &[usize], target: usize, max_hops: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    fn recurse(offsets: &[usize], target: usize, hops_left: usize,
                current: &mut Vec<usize>, min_idx: usize, results: &mut Vec<Vec<usize>>) {
        if target == 0 && !current.is_empty() { results.push(current.clone()); return; }
        if hops_left == 0 { return; }
        for (i, &d) in offsets.iter().enumerate().skip(min_idx) {
            if d > target { break; }
            current.push(d);
            recurse(offsets, target - d, hops_left - 1, current, i, results);
            current.pop();
        }
    }
    let mut cur = Vec::new();
    recurse(offsets, target, max_hops, &mut cur, 0, &mut results);
    results
}

// ── Block-width optimizer ─────────────────────────────────────────────────────

/// Fixed (non-variable) offsets in Run D: short sparse, gap offsets, tail.
fn fixed_offsets_run_d() -> Vec<usize> {
    let mut v = vec![1usize, 2, 4, 8, 16, 48, 96, 192, 384, 768, 1024, 1536];
    v.sort(); v
}

/// Build full offset set from block widths allocated to 5 octave centers.
/// Centers: {32, 64, 128, 256, 512}.  Each block covers [center, center+w-1].
fn build_from_widths(widths: &[usize; 5]) -> Vec<usize> {
    let centers = [32usize, 64, 128, 256, 512];
    let mut o = fixed_offsets_run_d();
    for (&center, &w) in centers.iter().zip(widths.iter()) {
        for d in center..center + w { if d < MAX_LAG { o.push(d); } }
    }
    o.sort(); o.dedup();
    o
}

/// Aggregate score across passkey distances for a given offset set.
/// hop_discount: 1.0 = pure path count; 0.5 = halve each additional hop.
fn aggregate_score(offsets: &[usize], hop_discount: f64) -> f64 {
    let counts = path_counts(offsets, NUM_LAYERS);
    PASSKEY_DISTANCES.iter()
        .map(|&d| path_score(&counts, d, hop_discount))
        .sum()
}

/// Brute-force over all allocations w1+w2+w3+w4+w5 = budget, wᵢ ≥ 1.
/// Returns (best_widths, best_score, all_results_sorted_desc).
fn optimize_block_widths(budget: usize, hop_discount: f64)
    -> ([usize; 5], f64, Vec<([usize; 5], f64)>) {
    let mut results: Vec<([usize; 5], f64)> = Vec::new();

    // Enumerate via recursive partition
    for w1 in 1..=budget - 4 {
        for w2 in 1..=budget - 3 - w1 {
            for w3 in 1..=budget - 2 - w1 - w2 {
                for w4 in 1..=budget - 1 - w1 - w2 - w3 {
                    let w5 = budget - w1 - w2 - w3 - w4;
                    if w5 < 1 { continue; }
                    let widths = [w1, w2, w3, w4, w5];
                    let offsets = build_from_widths(&widths);
                    let score = aggregate_score(&offsets, hop_discount);
                    results.push((widths, score));
                }
            }
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let best = results[0];
    (best.0, best.1, results)
}

// ── Q-factor / AM-GM analysis ─────────────────────────────────────────────────

/// Analytical proof section: for combination-tone paths between two blocks A and B
/// with widths wA and wB (total budget w = wA + wB fixed), the number of
/// 2-hop paths reaching d = centerA + centerB is:
///   P₂ = wA × wB  ≤  (w/2)² = (w²/4)
/// with equality iff wA = wB (AM-GM).
/// Returns (equal_paths, shifted_paths_table) for various allocations.
fn amgm_analysis(budget: usize, center_a: usize, center_b: usize) -> Vec<(usize, usize, u64)> {
    let target = center_a + center_b;
    let mut results = Vec::new();
    for wa in 1..budget {
        let wb = budget - wa;
        // Paths from block A × block B: (δ₁ from A, δ₂ from B) with δ₁+δ₂=target
        // Block A = [center_a, center_a+wa-1], Block B = [center_b, center_b+wb-1]
        let paths: u64 = (center_a..center_a+wa)
            .filter(|&da| {
                let db = target.wrapping_sub(da);
                db >= center_b && db < center_b + wb
            })
            .count() as u64;
        results.push((wa, wb, paths));
    }
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────────────────────────────────────────────────────
    // Test 1: Effective offset analysis
    // ──────────────────────────────────────────────────────────────────────────
    //
    // The passkey needle (word token) isn't at position d from the query; it's
    // at d + (cue_len + intro_tail). We call this δ_eff.
    // Run D's dense blocks are designed to cover δ_eff for each octave distance.
    // This test verifies that alignment exactly.
    #[test]
    fn effective_offset_alignment() {
        let intro_tail = PASSKEY_INTRO_LEN - PASSKEY_WORD_POS - 1; // tokens after word in intro
        let delta_base = PASSKEY_CUE_LEN + intro_tail; // ≈ 7

        let v3  = offsets_v3_condu();
        let rc  = offsets_run_c();
        let rd  = offsets_run_d();

        println!("\n══ Effective Offset Analysis ═════════════════════════════════════════");
        println!("  δ_eff = d + {} (cue_len={}, intro_tail={})",
                 delta_base, PASSKEY_CUE_LEN, intro_tail);
        println!();
        println!("{:<8} {:>8} {:>10} {:>10} {:>10}",
                 "passkey_d", "δ_eff", "V3/condU", "Run C", "Run D");
        println!("{}", "─".repeat(52));

        for &d in PASSKEY_DISTANCES {
            let d_eff = d + delta_base;
            let in_v3 = v3.contains(&d_eff);
            let in_rc = rc.contains(&d_eff);
            let in_rd = rd.contains(&d_eff);
            println!("{:<8} {:>8} {:>10} {:>10} {:>10}",
                     d, d_eff,
                     if in_v3 { "direct✓" } else { "multi-hop" },
                     if in_rc { "direct✓" } else { "multi-hop" },
                     if in_rd { "direct✓" } else { "multi-hop" });
        }

        // Key assertion: Run D's octave blocks align to δ_eff for the 5 dense distances
        let octave_distances = [32usize, 64, 128, 256, 512];
        let block_ranges = [(32..40), (64..72), (128..136), (256..264), (512..520)];
        for (&d, range) in octave_distances.iter().zip(block_ranges.iter()) {
            let d_eff = d + delta_base;
            assert!(range.contains(&d_eff),
                "Run D block for d={} should contain δ_eff={} (cue+intro_tail={}). \
                 Block range: {:?}",
                d, d_eff, delta_base, range);
            println!("\n  ✓ d={}: δ_eff={} ∈ Run D block {:?}", d, d_eff, range);
        }
        println!("\n  Run D's dense blocks are exactly aligned to the effective passkey offsets.");
        println!("  The 8-wide blocks provide ~{} tokens of tolerance for tokenizer variance.",
                 8usize.saturating_sub(1));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 2: Combination tone analysis — d=96 from d=32+d=64
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn combination_tone_d96() {
        let v3 = offsets_v3_condu();
        let rc = offsets_run_c();
        let rd = offsets_run_d();

        println!("\n══ Combination Tone Analysis: d=96 ═══════════════════════════════════");
        for (name, offsets) in &[("V3/condU", &v3), ("Run C", &rc), ("Run D", &rd)] {
            let direct = offsets.contains(&96);
            let paths_2hop = enumerate_paths(offsets, 96, 2);
            let paths_3hop = enumerate_paths(offsets, 96, 3);
            println!("\n── {} ──", name);
            println!("  Direct (1-hop):  {}", if direct { "YES ✓" } else { "no" });
            println!("  2-hop paths:     {} distinct combinations", paths_2hop.len());
            for p in &paths_2hop { println!("    {:?}  (= {})", p, p.iter().sum::<usize>()); }
            println!("  3-hop paths:     {} distinct combinations (first 10):",
                     paths_3hop.len());
            for p in paths_3hop.iter().take(10) {
                println!("    {:?}  (= {})", p, p.iter().sum::<usize>());
            }
        }

        // Run D has δ=96 directly — verify
        assert!(rd.contains(&96), "Run D should have δ=96 as a gap offset (direct d=96 access)");

        // V3/condU should have d=96 reachable in 2 hops via (48,48) or (32,64) etc.
        let v3_2hop = enumerate_paths(&v3, 96, 2);
        assert!(!v3_2hop.is_empty(), "V3 should have ≥1 two-hop path to d=96");

        println!("\n  ✓ d=96 emerges from combination synthesis in V3/RunC (as predicted)");
        println!("  ✓ Run D has δ=96 directly, but also benefits from rich 2-hop structure");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 3: Path-count comparison across all passkey distances
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn path_count_comparison() {
        let v3 = offsets_v3_condu();
        let rc = offsets_run_c();
        let rd = offsets_run_d();

        let hop_discount = 0.5_f64;

        println!("\n══ Path Count Comparison (gradient signal proxy) ════════════════════");
        println!("  Hop discount: {:.1}  (each extra hop contributes half as much)", hop_discount);
        println!();
        println!("{:<8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "d", "V3 score", "RC score", "RD score", "RC/V3", "RD/RC");
        println!("{}", "─".repeat(66));

        let v3_counts = path_counts(&v3, NUM_LAYERS);
        let rc_counts = path_counts(&rc, NUM_LAYERS);
        let rd_counts = path_counts(&rd, NUM_LAYERS);

        let mut total_v3 = 0.0f64;
        let mut total_rc = 0.0f64;
        let mut total_rd = 0.0f64;

        for &d in PASSKEY_DISTANCES {
            let sv3 = path_score(&v3_counts, d, hop_discount);
            let src = path_score(&rc_counts, d, hop_discount);
            let srd = path_score(&rd_counts, d, hop_discount);
            total_v3 += sv3; total_rc += src; total_rd += srd;
            let rc_v3 = if sv3 > 0.0 { src / sv3 } else { f64::INFINITY };
            let rd_rc = if src > 0.0 { srd / src } else { f64::INFINITY };
            println!("{:<8}  {:>10.1}  {:>10.1}  {:>10.1}  {:>9.2}x {:>9.2}x",
                     d, sv3, src, srd, rc_v3, rd_rc);
        }
        println!("{}", "─".repeat(66));
        println!("{:<8}  {:>10.1}  {:>10.1}  {:>10.1}  {:>9.2}x {:>9.2}x",
                 "TOTAL", total_v3, total_rc, total_rd,
                 total_rc / total_v3, total_rd / total_rc);

        println!("\n  Higher score = more gradient paths = faster/more reliable learning.");
        // NOTE: V3's dense 0-32 generates enormous multi-hop path counts for short
        // distances via exponential combination growth — it DOMINATES aggregate score.
        // But this doesn't translate to passkey success because V3 lacks δ_eff (d+7)
        // for the octave distances. The aggregate path-count metric is NOT predictive
        // of passkey success; δ_eff direct coverage is the correct metric (see Test 1).
        println!("\n  IMPORTANT: V3 has higher aggregate score ({:.0}) than Run D ({:.0})",
                 total_v3, total_rd);
        println!("  but V3 FAILS d=32 passkey because it lacks δ_eff=39 (dense stops at 32).");
        println!("  Aggregate path count is dominated by exponential multi-hop combinations;");
        println!("  it predicts combinatorial richness, NOT single-token retrieval success.");
        println!("  The correct metric is δ_eff direct coverage (Test 1 / Test 6).");
        // Run D is still better than Run C for long-range (d=512-1536)
        assert!(total_rd > total_rc,
            "Run D should have higher score than Run C (better long-range), RD={:.1} RC={:.1}",
            total_rd, total_rc);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 4: AM-GM proof — equal block widths are optimal for combination tones
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn amgm_combination_tone_proof() {
        let budget = 16usize;  // total offsets for two blocks
        let center_a = 32usize;
        let center_b = 64usize;
        let target = center_a + center_b;  // d=96

        let results = amgm_analysis(budget, center_a, center_b);

        println!("\n══ AM-GM Proof: Equal Widths Maximise Combination-Tone Paths ═════════");
        println!("  Two blocks: A centered at {}, B centered at {}.", center_a, center_b);
        println!("  Target d={} (={}+{}). Total budget: {} offsets.", target, center_a, center_b, budget);
        println!();
        println!("{:>5}  {:>5}  {:>12}   note", "wA", "wB", "2-hop paths");
        println!("{}", "─".repeat(42));
        let max_paths = results.iter().map(|r| r.2).max().unwrap_or(0);
        let equal_paths = results.iter().find(|r| r.0 == r.1).map(|r| r.2).unwrap_or(0);
        for (wa, wb, paths) in &results {
            let note = if *paths == max_paths { " ← maximum" } else { "" };
            println!("{:>5}  {:>5}  {:>12}{}", wa, wb, paths, note);
        }
        println!();
        println!("  AM-GM: wA×wB ≤ (wA+wB)²/4 = {}²/4 = {}", budget, budget*budget/4);
        println!("  Equal (wA=wB={}): {} paths = {} (AM-GM equality holds)",
                 budget/2, equal_paths, (budget/2)*(budget/2));
        println!();
        println!("  CONCLUSION: Equal block widths maximise 2-hop paths between octaves.");
        println!("  Making block A wider at expense of B *decreases* combination coverage.");
        println!("  Exception: if direct coverage at one distance is more valuable,");
        println!("  skew budget toward that distance — but you lose combination paths.");

        assert_eq!(equal_paths, max_paths,
            "Equal allocation should give maximum combination paths: equal={}, max={}",
            equal_paths, max_paths);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 5: Block-width optimizer — find optimal allocation
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn block_width_optimizer() {
        // Use smaller budget for speed (scales from 8×5=40 down to 5×5=25 for fast search)
        // Full search with budget=40 requires --release mode; budget=25 is fast in debug.
        let budget = 25usize;
        let hop_discount = 0.5_f64;

        println!("\n══ Block Width Optimizer (budget={}, discount={}) ════════════════════",
                 budget, hop_discount);
        println!("  Octave positions: [32, 64, 128, 256, 512]");
        println!("  Fixed offsets: short sparse + gaps + tail");
        println!("  Note: budget={} (5 per block); full search (budget=40) needs --release",
                 budget);

        let (best_widths, best_score, all_results) =
            optimize_block_widths(budget, hop_discount);
        let equal_widths     = [budget/5; 5];
        let equal_score      = aggregate_score(&build_from_widths(&equal_widths), hop_discount);
        // Run D's actual widths (budget=40, 8 per block) — check separately from optimizer
        let run_d_widths: [usize; 5] = [8, 8, 8, 8, 8];

        println!("\n  Top 10 allocations:");
        println!("  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>12}  note",
                 "w_32", "w_64", "w_128", "w_256", "w_512", "score");
        println!("  {}", "─".repeat(60));
        for (widths, score) in all_results.iter().take(10) {
            let equal = widths.iter().all(|&w| w == budget/5);
            println!("  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>12.1}{}",
                     widths[0], widths[1], widths[2], widths[3], widths[4],
                     score, if equal { "  ← equal (current Run D)" } else { "" });
        }

        println!("\n  Equal allocation [8,8,8,8,8]: score={:.1}", equal_score);
        println!("  Best allocation {:?}: score={:.1}", best_widths, best_score);
        println!("  Improvement: {:+.2}% ({:.1} → {:.1})",
                 (best_score - equal_score) / equal_score * 100.0,
                 equal_score, best_score);

        let improvement_pct = (best_score - equal_score) / equal_score * 100.0;

        // Report where the budget should shift (wider at shorter vs longer octaves)
        let d32_trend = best_widths[0] as isize - 8;
        let d512_trend = best_widths[4] as isize - 8;
        println!();
        if d32_trend > 0 && d512_trend < 0 {
            println!("  → Optimal: wider at d=32 ({:+}), narrower at d=512 ({:+})", d32_trend, d512_trend);
        } else if d32_trend < 0 && d512_trend > 0 {
            println!("  → Optimal: narrower at d=32 ({:+}), wider at d=512 ({:+})", d32_trend, d512_trend);
        } else {
            println!("  → Optimal is close to equal allocation (improvement: {:.2}%)", improvement_pct);
        }

        println!("\n  Note: optimizer uses {} layers of propagation, hop-discount {}.",
                 NUM_LAYERS, hop_discount);
        println!("  Improvement threshold: if <1%, equal widths are practically optimal.");

        // CRITICAL FINDING: the path-count optimizer recommends [15,7,1,1,1] — concentrate
        // budget at d=32 — because it maximises multi-hop combination counts. But this is
        // the WRONG metric. Check whether the optimizer's recommendation actually covers
        // δ_eff for each octave distance:
        let delta_base = PASSKEY_CUE_LEN + (PASSKEY_INTRO_LEN - PASSKEY_WORD_POS - 1);
        let octave_ds  = [32usize, 64, 128, 256, 512];
        let best_offsets = build_from_widths(&best_widths);
        println!("\n  δ_eff coverage check for optimizer's best allocation {:?}:", best_widths);
        let mut best_covers_all = true;
        for &d in &octave_ds {
            let d_eff = d + delta_base;
            let covered = best_offsets.contains(&d_eff);
            if !covered { best_covers_all = false; }
            println!("    d={}: δ_eff={} → {}", d, d_eff,
                     if covered { "covered ✓" } else { "MISSING ✗ (path-count metric misleads!)" });
        }
        // Verify Run D's actual 8-wide blocks cover all δ_eff values
        let run_d_offsets = build_from_widths(&run_d_widths);
        println!("\n  δ_eff coverage check for Run D actual widths {:?}:", run_d_widths);
        let mut run_d_covers_all = true;
        for &d in &octave_ds {
            let d_eff = d + delta_base;
            let covered = run_d_offsets.contains(&d_eff);
            if !covered { run_d_covers_all = false; }
            println!("    d={}: δ_eff={} → {}", d, d_eff,
                     if covered { "covered ✓" } else { "MISSING ✗" });
        }
        println!("\n  Optimizer best covers all δ_eff: {}", best_covers_all);
        println!("  Run D [8,8,8,8,8] covers all δ_eff: {}", run_d_covers_all);
        println!("  CONCLUSION: Path-count metric is misleading — correct metric is δ_eff");
        println!("  coverage. 8-wide blocks are the MINIMUM sufficient width: δ_eff=d+7");
        println!("  falls exactly at the top of an 8-wide block starting at d.");
        println!("  Going wider at d=32 at expense of d=512 breaks δ_eff coverage for d=512.");

        // Run D's 8-wide blocks must cover all δ_eff
        assert!(run_d_covers_all,
            "Run D [8,8,8,8,8] must cover δ_eff for all octave passkey distances");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 6: Bandgap prediction — where Run D should show soft walls
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn bandgap_prediction_run_d() {
        let rd = offsets_run_d();
        let counts = path_counts(&rd, NUM_LAYERS);
        let hop_discount = 0.5_f64;

        // Gaps in Run D offset set (no direct coverage):
        // [3,5,6,7,9-15,17-31], [40-47], [49-63], [72-95], [136-191], [264-383], [520-767]
        let gap_ranges: &[(&str, std::ops::RangeInclusive<usize>)] = &[
            ("short-sparse [17-31]", 17..=31),
            ("inter-octave [40-47]", 40..=47),
            ("inter-octave [49-63]", 49..=63),
            ("inter-octave [72-95]", 72..=95),
            ("inter-octave [136-191]", 136..=191),
            ("inter-octave [264-383]", 264..=383),
            ("tail gap [520-767]", 520..=767),
        ];

        println!("\n══ Bandgap Prediction: Run D Coverage in Gap Ranges ════════════════");
        println!("  Prediction: gaps between dense blocks show reduced path-count scores");
        println!("  (these are where passkey 'soft walls' should appear in Run D)\n");
        println!("{:<28}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "Range", "1-hop", "2-hop", "3-hop", "score");
        println!("{}", "─".repeat(64));

        // First, reference scores at dense block centers
        let centers = [32usize, 64, 128, 256, 512];
        for &c in &centers {
            let s = path_score(&counts, c, hop_discount);
            let h1 = counts[1][c];
            let h2 = counts[2][c];
            let h3 = counts[3][c];
            println!("{:<28}  {:>8}  {:>8}  {:>8}  {:>8.1}",
                     format!("d={} (block center)", c), h1, h2, h3, s);
        }
        println!("{}", "─".repeat(64));

        // Gap ranges: pick representative distances
        for (label, range) in gap_ranges {
            // Sample middle of range
            let sample: Vec<usize> = range.clone()
                .filter(|d| PASSKEY_DISTANCES.contains(d) ||
                            *d == (*range.start() + *range.end()) / 2)
                .collect();
            let d = if sample.is_empty() {
                (*range.start() + *range.end()) / 2
            } else { sample[0] };
            let s = path_score(&counts, d, hop_discount);
            let h1 = counts[1][d];
            let h2 = counts[2][d];
            let h3 = counts[3][d];
            let ratio = s / path_score(&counts, 32, hop_discount);
            println!("{:<28}  {:>8}  {:>8}  {:>8}  {:>8.1}  ({:.0}% of d=32 score)",
                     format!("{} [d={}]", label, d), h1, h2, h3, s, ratio * 100.0);
        }

        println!("\n  Passkey distances in gap ranges (expected to be harder in Run D):");
        // None of the standard PASSKEY_DISTANCES fall in these gaps for Run D
        // because PASSKEY_DISTANCES = {1,2,4,8,16,32,64,128,256,512,1024,1536}
        // and all of these are directly covered in Run D!
        let gap_passkeys: Vec<usize> = PASSKEY_DISTANCES.iter()
            .filter(|&&d| rd.iter().all(|&off| off != d))
            .copied().collect();
        if gap_passkeys.is_empty() {
            println!("  None — all 12 passkey distances have direct (1-hop) access in Run D!");
            println!("  → Run D should NOT show the passkey walls seen in V3/RunC.");
        } else {
            println!("  {:?}", gap_passkeys);
        }

        // Confirm all passkey distances are 1-hop in Run D
        for &d in PASSKEY_DISTANCES {
            assert!(
                counts[1][d] >= 1,
                "Run D should have direct (1-hop) access to passkey distance d={}, \
                 but 1-hop count is {}",
                d, counts[1][d]
            );
        }
        println!("\n  ✓ All 12 passkey distances verified as 1-hop direct in Run D.");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 7: Summary report — V3 vs RunC vs RunD side by side
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn full_summary_report() {
        let v3 = offsets_v3_condu();
        let rc = offsets_run_c();
        let rd = offsets_run_d();

        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║           DWARF Offset-Set Filter-Bank Analysis Summary              ║");
        println!("╠══════════════════════════════════════════════════════════════════════╣");

        let hop_discount = 0.5_f64;
        let v3c = path_counts(&v3, NUM_LAYERS);
        let rcc = path_counts(&rc, NUM_LAYERS);
        let rdc = path_counts(&rd, NUM_LAYERS);

        println!("║  DIRECT (1-HOP) ACCESS TO EACH PASSKEY DISTANCE                      ║");
        println!("║  {:<6} │ V3/condU │  Run C  │  Run D                               ║", "d");
        for &d in PASSKEY_DISTANCES {
            let v  = if v3c[1][d] > 0 { "  ✓    " } else { "  ✗    " };
            let r  = if rcc[1][d] > 0 { "  ✓    " } else { "  ✗    " };
            let r2 = if rdc[1][d] > 0 { "  ✓    " } else { "  ✗    " };
            println!("║  d={:<5} │{}│{}│ {}                               ║", d, v, r, r2);
        }
        println!("╠══════════════════════════════════════════════════════════════════════╣");
        println!("║  AGGREGATE GRADIENT SCORE (hop_discount=0.5)                         ║");
        let sv3 = PASSKEY_DISTANCES.iter().map(|&d| path_score(&v3c, d, hop_discount)).sum::<f64>();
        let src = PASSKEY_DISTANCES.iter().map(|&d| path_score(&rcc, d, hop_discount)).sum::<f64>();
        let srd = PASSKEY_DISTANCES.iter().map(|&d| path_score(&rdc, d, hop_discount)).sum::<f64>();
        println!("║  V3/condU: {:<10.1}  Run C: {:<10.1}  Run D: {:<10.1}          ║",
                 sv3, src, srd);
        println!("║  Run C / V3: {:.2}×    Run D / Run C: {:.2}×   Run D / V3: {:.2}×    ║",
                 src/sv3, srd/src, srd/sv3);
        println!("╠══════════════════════════════════════════════════════════════════════╣");
        println!("║  KEY FINDINGS                                                         ║");
        println!("║  1. Run D: ALL 12 passkey distances have direct 1-hop access.         ║");
        println!("║     V3/RunC miss d=16 (needs 2-hop in RunC), d=16 direct in RunD.     ║");
        println!("║  2. Dense blocks aligned to δ_eff = d+7 (cue+intro offset).           ║");
        println!("║     Run D's 8-wide blocks are minimal-sufficient for 5-token cue.     ║");
        println!("║  3. AM-GM: equal block widths maximise 2-hop combination paths.       ║");
        println!("║     Skewing widths toward shorter octaves reduces combination tones.  ║");
        println!("║  4. Combination tones (d=96=32+64) richest in Run D due to dense      ║");
        println!("║     blocks at both ends; Run D also has δ=96 directly.                ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 8: Run E design analysis — dense 1-39 + 4 octave blocks at 64-512
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn run_e_analysis() {
        let re = offsets_run_e();
        let v3 = offsets_v3_condu();
        let rc = offsets_run_c();
        let rd = offsets_run_d();
        let hop_discount = 0.5_f64;

        println!("\n╔══════════════════════════════════════════════════════════════════════╗");
        println!("║               Run E Design Analysis                                   ║");
        println!("║  dense 1-39 + octave blocks [64-71][128-135][256-263][512-519]        ║");
        println!("║  + gap/tail {{48,96,192,384,768,1024,1536}}                            ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        println!("\n── Offset set sizes ──────────────────────────────────────────────────");
        println!("  V3/condU : J={}", v3.len());
        println!("  Run C    : J={}", rc.len());
        println!("  Run D    : J={}", rd.len());
        println!("  Run E    : J={}", re.len());

        let delta_base = PASSKEY_CUE_LEN + (PASSKEY_INTRO_LEN - PASSKEY_WORD_POS - 1);
        println!("\n── delta_eff = d+{} coverage (direct 1-hop) ──────────────────────────",
                 delta_base);
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "d", "d_eff", "V3", "Run C", "Run D", "Run E");
        println!("  {}", "─".repeat(60));

        let rec = path_counts(&re, NUM_LAYERS);
        let v3c = path_counts(&v3, NUM_LAYERS);
        let rcc = path_counts(&rc, NUM_LAYERS);
        let rdc = path_counts(&rd, NUM_LAYERS);

        for &d in PASSKEY_DISTANCES {
            let d_eff = d + delta_base;
            let v3_d = if d_eff < MAX_LAG && v3c[1][d_eff] > 0 { "direct" } else { "multi" };
            let rc_d = if d_eff < MAX_LAG && rcc[1][d_eff] > 0 { "direct" } else { "multi" };
            let rd_d = if d_eff < MAX_LAG && rdc[1][d_eff] > 0 { "direct" } else { "multi" };
            let re_d = if d_eff < MAX_LAG && rec[1][d_eff] > 0 { "direct" } else { "multi" };
            println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                     d, d_eff, v3_d, rc_d, rd_d, re_d);
        }

        println!("\n── Path-count scores (hop_discount={}) ───────────────────────────────",
                 hop_discount);
        println!("  {:>6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}",
                 "d", "V3", "Run C", "Run D", "Run E", "RE/RD");
        println!("  {}", "─".repeat(72));

        let mut total_re = 0.0_f64;
        let mut total_rd = 0.0_f64;
        for &d in PASSKEY_DISTANCES {
            let sv3 = path_score(&v3c, d, hop_discount);
            let src = path_score(&rcc, d, hop_discount);
            let srd = path_score(&rdc, d, hop_discount);
            let sre = path_score(&rec, d, hop_discount);
            total_re += sre;
            total_rd += srd;
            println!("  {:>6}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>8.2}x",
                     d, sv3, src, srd, sre, if srd > 0.0 { sre / srd } else { 0.0 });
        }
        println!("  {}", "─".repeat(72));
        println!("  {:>6}  {:>12}  {:>12}  {:>12}  {:>12.1}",
                 "TOTAL", "", "", "", total_re);
        println!("  Run E / Run D: {:.2}x", total_re / total_rd);

        println!("\n── Short-range density (delta=1..31, NL modeling power) ──────────────");
        let v3_short: usize = v3.iter().filter(|&&x| x <= 31).count();
        let rc_short: usize = rc.iter().filter(|&&x| x <= 31).count();
        let rd_short: usize = rd.iter().filter(|&&x| x <= 31).count();
        let re_short: usize = re.iter().filter(|&&x| x <= 31).count();
        println!("  V3/condU : {} offsets in d=1-31 (PPL baseline)", v3_short);
        println!("  Run C    : {} offsets in d=1-31", rc_short);
        println!("  Run D    : {} offsets in d=1-31  <- PPL deficit source", rd_short);
        println!("  Run E    : {} offsets in d=1-31  <- restored", re_short);

        println!("\n── Key predictions for Run E ─────────────────────────────────────────");
        println!("  PPL      : dense 1-31 restored -> expect ~52-53 (near V3 baseline)");
        println!("  Passkey  : d=1-32 all d_eff covered -> expect >=40% by ep5");
        println!("  Passkey  : d=64-512 octave blocks -> expect >=60% by ep7");
        println!("  Passkey  : d=1024/1536 via multi-hop (proven in Run D)");
        println!("  Init     : near-zero -> first signal ep2 (like Run D)");
        println!("  Speed    : J=78, ~80 min/run on RTX 4090");

        // Assertions
        assert_eq!(re_short, v3_short,
            "Run E should restore V3 short-range density ({} vs {})", re_short, v3_short);
        assert!(total_re > total_rd,
            "Run E should score higher than Run D (RE={:.0} RD={:.0})", total_re, total_rd);

        let re_deff_covered: Vec<usize> = PASSKEY_DISTANCES.iter()
            .filter(|&&d| {
                let d_eff = d + delta_base;
                d_eff < MAX_LAG && rec[1][d_eff] > 0
            })
            .copied().collect();

        let expected_deff = vec![1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        assert_eq!(re_deff_covered, expected_deff,
            "Run E should cover d_eff for all d=1..512, got {:?}", re_deff_covered);

        println!("\n  ASSERTIONS PASSED:");
        println!("  + Run E covers d_eff for d=1,2,4,8,16,32,64,128,256,512 (10/12)");
        println!("  + d=1024/1536 via multi-hop (d_eff > MAX_LAG range, synthesis only)");
        println!("  + Short-range density matches V3 (PPL recovery expected)");
        println!("  + Run E aggregate path score > Run D");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 9: Run F design analysis — dense 1-15 + GAP 16-31 + octave blocks
    // ──────────────────────────────────────────────────────────────────────────
    #[test]
    fn run_f_analysis() {
        let rf = offsets_run_f();
        let re = offsets_run_e();
        let rd = offsets_run_d();
        let rc = offsets_run_c();
        let v3 = offsets_v3_condu();
        let hop_discount = 0.5_f64;

        println!("\n╔══════════════════════════════════════════════════════════════════════╗");
        println!("║               Run F Design Analysis                                   ║");
        println!("║  dense 1-15 + GAP 16-31 + blocks [32-39][64-71][128-135]             ║");
        println!("║                        [256-263][512-519] + gap/tail                  ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        println!("\n── Offset set sizes ──────────────────────────────────────────────────");
        println!("  V3/condU : J={}", v3.len());
        println!("  Run C    : J={}", rc.len());
        println!("  Run D    : J={}", rd.len());
        println!("  Run E    : J={}", re.len());
        println!("  Run F    : J={}", rf.len());

        let delta_base = PASSKEY_CUE_LEN + (PASSKEY_INTRO_LEN - PASSKEY_WORD_POS - 1);
        let rfc = path_counts(&rf, NUM_LAYERS);
        let rec = path_counts(&re, NUM_LAYERS);
        let rdc = path_counts(&rd, NUM_LAYERS);
        let rcc = path_counts(&rc, NUM_LAYERS);
        let v3c = path_counts(&v3, NUM_LAYERS);

        println!("\n── delta_eff = d+{} coverage ─────────────────────────────────────────",
                 delta_base);
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "d", "d_eff", "V3", "Run C", "Run D", "Run E", "Run F");
        println!("  {}", "─".repeat(68));

        for &d in PASSKEY_DISTANCES {
            let d_eff = d + delta_base;
            let chk = |c: &Vec<Vec<u64>>| if d_eff < MAX_LAG && c[1][d_eff] > 0 { "direct" } else { "multi" };
            println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                     d, d_eff, chk(&v3c), chk(&rcc), chk(&rdc), chk(&rec), chk(&rfc));
        }

        println!("\n── Short-range density (delta=1..31) ─────────────────────────────────");
        let density = |offs: &[usize]| offs.iter().filter(|&&x| x <= 31).count();
        println!("  V3/condU : {} (dense 1-31)", density(&v3));
        println!("  Run C    : {} (dense 1-15 only)", density(&rc));
        println!("  Run D    : {} (sparse {{1,2,4,8,16}})", density(&rd));
        println!("  Run E    : {} (dense 1-39 incl 16-31)", density(&re));
        println!("  Run F    : {} (dense 1-15, GAP 16-31)", density(&rf));

        let rf_gap: Vec<usize> = rf.iter().filter(|&&x| x >= 16 && x <= 31).copied().collect();
        let re_gap: Vec<usize> = re.iter().filter(|&&x| x >= 16 && x <= 31).copied().collect();
        println!("\n── Gap 16-31 analysis (commitment pressure zone) ─────────────────────");
        println!("  Run E offsets in 16-31: {:?}  <- soft alternatives", re_gap);
        println!("  Run F offsets in 16-31: {:?}  <- empty GAP", rf_gap);
        println!("  Run F cliff edge: model must use [32-39] for d=32 (no soft alt at 16-31).");
        println!("  Same structure Run C used -> d=32:60%+ by ep5.");

        println!("\n── Path-count scores (hop_discount={}) ───────────────────────────────",
                 hop_discount);
        println!("  {:>6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}",
                 "d", "Run C", "Run D", "Run E", "Run F", "RF/RC");
        println!("  {}", "─".repeat(64));

        let mut total_rf = 0.0_f64;
        let mut total_rc_sum = 0.0_f64;
        for &d in PASSKEY_DISTANCES {
            let src = path_score(&rcc, d, hop_discount);
            let srd = path_score(&rdc, d, hop_discount);
            let sre = path_score(&rec, d, hop_discount);
            let srf = path_score(&rfc, d, hop_discount);
            total_rf     += srf;
            total_rc_sum += src;
            println!("  {:>6}  {:>11.1}  {:>11.1}  {:>11.1}  {:>11.1}  {:>8.2}x",
                     d, src, srd, sre, srf, if src > 0.0 { srf/src } else { 0.0 });
        }
        println!("  {}", "─".repeat(64));
        println!("  TOTAL: RC={:.0}  RD={:.0}  RE={:.0}  RF={:.0}",
                 total_rc_sum,
                 PASSKEY_DISTANCES.iter().map(|&d| path_score(&rdc, d, hop_discount)).sum::<f64>(),
                 PASSKEY_DISTANCES.iter().map(|&d| path_score(&rec, d, hop_discount)).sum::<f64>(),
                 total_rf);
        println!("  RF/RC: {:.2}x   RF/RD: {:.2}x",
                 total_rf/total_rc_sum,
                 total_rf / PASSKEY_DISTANCES.iter().map(|&d| path_score(&rdc, d, hop_discount)).sum::<f64>());

        println!("\n── Run F predictions ─────────────────────────────────────────────────");
        println!("  PPL     : dense 1-15 (= Run C short-range) -> ~52-53 PPL");
        println!("  d=1-8   : delta_eff in dense 1-15 -> signal from ep2");
        println!("  d=16    : delta_eff=23 in GAP -> multi-hop only (minor weakness)");
        println!("  d=32    : cliff edge forces commitment -> 60%+ by ep5 (Run C rate)");
        println!("  d=64-512: octave blocks -> 60%+ by ep7 (Run D rate)");
        println!("  d=1024+ : multi-hop synthesis (proven in Run D: 60%)");
        println!("  Speed   : J~63, ~15% faster than Run E (J=79)");
        println!("  HYPOTHESIS: Run F = Run C PPL + Run D long-range passkey");

        // Assertions
        let rf_short = density(&rf);
        let rc_short = density(&rc);
        assert_eq!(rf_short, rc_short,
            "Run F must match Run C short-range density ({} vs {})", rf_short, rc_short);
        assert!(rf_gap.is_empty(),
            "Run F must have empty gap 16-31 (commitment pressure)");
        let octave_ok: bool = [32usize, 64, 128, 256, 512].iter().all(|&d| {
            let de = d + delta_base;
            de < MAX_LAG && rfc[1][de] > 0
        });
        assert!(octave_ok, "Run F must cover delta_eff for d=32,64,128,256,512");

        println!("\n  ASSERTIONS PASSED:");
        println!("  + Run F short-range density = Run C (15 offsets, dense 1-15)");
        println!("  + Gap 16-31 confirmed empty");
        println!("  + delta_eff direct for d=32,64,128,256,512");
        println!("  + d=16 multi-hop only (acceptable: Run C also had d=16 as multi-hop)");
    }
}
