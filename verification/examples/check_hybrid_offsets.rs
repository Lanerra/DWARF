use wave_field_verification::offset_space_explorer::{build_offset_set, compute_metrics, offsets_condu, SPARSE_POOL};
use wave_field_verification::offset_optimizer::PASSKEY_DISTANCES;

fn profile(label: &str, offsets: &[usize]) {
    let condu = offsets_condu();
    let cm = compute_metrics(&condu);
    let m = compute_metrics(offsets);

    let max_offset = offsets.iter().cloned().max().unwrap_or(0);
    println!("--- {} ---", label);
    println!("  J={}, max_offset={}, buffer_vs_condU={:.2}x",
        offsets.len(), max_offset, max_offset as f64 / 1536.0);
    println!("  coverage={:.0} ({:.2}x condU)", m.coverage_score, m.coverage_score / cm.coverage_score);
    println!("  reliable_depth={}", m.reliable_retrieval_depth);
    println!("  {:>8}  {:>14}  {:>14}  ratio", "d", "paths", "condU_paths");
    for (&d, (&paths, &cp)) in PASSKEY_DISTANCES.iter().zip(
        m.paths_by_distance.iter().zip(cm.paths_by_distance.iter())
    ) {
        let ratio = if cp > 0 { paths as f64 / cp as f64 } else { 0.0 };
        let flag = if paths == 0 { " ← ZERO" } else if ratio < 0.01 { " ← <1% condU" } else { "" };
        println!("  {:>8}  {:>14}  {:>14}  {:.3}x{}", d, paths, cp, ratio, flag);
    }
    println!();
}

fn main() {
    println!("=== Offset Config Comparison ===\n");

    let condu_offsets = offsets_condu();
    profile("condU (baseline)", &condu_offsets);

    // Dennis's proposed: dense=41 + 3 sparse (best-3)
    profile("dense=41 + [48,128,384] J=44",
        &build_offset_set(41, &[48, 128, 384]));

    // Hybrid: add the two long-range anchors
    profile("dense=41 + [48,128,384,1024,1536] J=46",
        &build_offset_set(41, &[48, 128, 384, 1024, 1536]));

    // Also try: dense=41 + [48,128,384,768,1024,1536] J=47
    profile("dense=41 + [48,128,384,768,1024,1536] J=47",
        &build_offset_set(41, &[48, 128, 384, 768, 1024, 1536]));

    // Middle ground: dense=35 + 8 sparse (best-8)
    profile("dense=35 + [48,64,96,128,192,256,384,768] J=43",
        &build_offset_set(35, &[48, 64, 96, 128, 192, 256, 384, 768]));

    // Quick sanity: what does condU's max_offset mean for buffer
    let condu_max = condu_offsets.iter().cloned().max().unwrap_or(0);
    println!("=== KV Cache (inference circular buffer) ===");
    println!("  condU:              max_offset={}  buffer=1536 tokens (baseline)", condu_max);
    println!("  dense=41+[48,128,384]:     max=384   buffer=384 tokens  (4.0x smaller)");
    println!("  dense=41+[..1024,1536]:    max=1536  buffer=1536 tokens (same)");
    println!("  dense=41+[..768,1024,1536]:max=1536  buffer=1536 tokens (same)");
    println!("  dense=35+8sparse(max=768): max=768   buffer=768 tokens  (2.0x smaller)");
    println!();
    println!("=== Training Speed (J offsets per token per layer) ===");
    for (label, j, max_o) in &[
        ("condU J=43", 43usize, 1536usize),
        ("41D+3S  J=44", 44, 384),
        ("41D+5S  J=46", 46, 1536),
        ("41D+6S  J=47", 47, 1536),
        ("35D+8S  J=43", 43, 768),
    ] {
        let attn_rel = *j as f64 / 43.0;
        println!("  {:<20} J={:<3}  max_offset={:<5}  attn_compute={:.3}x condU",
            label, j, max_o, attn_rel);
    }
}
