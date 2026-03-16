//! Frobenius-Optimal Offset Set Search
//!
//! Searches for offset sets with J < 24 that achieve the same relay chain
//! coverage as the J=24 production set, using Frobenius number theory.
//!
//! Deep Research finding (March 15, 2026): no prior work connects the
//! Frobenius coin problem to attention mechanism design. DWARF's coprime
//! relay chains are the first explicit application.
//!
//! Usage:
//!   cd verification
//!   PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example frobenius_search

fn main() {
    wave_field_verification::frobenius_offset_search::run();
}
