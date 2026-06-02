//! Wave field verification library (legacy)
//!
//! This library is largely historical. The verification logic has been
//! superseded by the `dwarf_filter` binary (src/bin/dwarf_filter.rs).
//!
//! The archived modules are preserved in verification/legacy/ for
//! reference and can be restored if needed.

#[path = "../legacy/offsets_triadic.rs"]
pub mod offsets_triadic;

#[path = "../legacy/offset_space_explorer.rs"]
pub mod offset_space_explorer;

#[path = "../legacy/offset_optimizer.rs"]
pub mod offset_optimizer;

#[path = "../legacy/sweep_engine.rs"]
pub mod sweep_engine;

#[path = "../legacy/layer_placement_explorer.rs"]
pub mod layer_placement_explorer;

#[path = "../legacy/scale_embed_dynamics.rs"]
pub mod scale_embed_dynamics;

#[path = "../legacy/gate_equilibrium.rs"]
pub mod gate_equilibrium;

#[path = "../legacy/knob_interactions.rs"]
pub mod knob_interactions;

#[path = "../legacy/frobenius_offset_search.rs"]
pub mod frobenius_offset_search;

#[path = "../legacy/frobenius_coopt_search.rs"]
pub mod frobenius_coopt_search;

#[path = "../legacy/scale_optimizer.rs"]
pub mod scale_optimizer;

pub const LEGACY_VERSION: &str = env!("CARGO_PKG_VERSION");
