//! Physics-based verification suite for Wave Field Transformer extensions.
//!
//! These modules prove mathematical properties of proposed architectural
//! components *before* committing to full training runs.
//!
//! # Modules
//!
//! - [`rg_init`]          — RG-motivated scale gain initialization
//! - [`soliton`]          — KdV soliton stability (continuous case)
//! - [`nonlinear_field`]  — Discrete KdV field update causality and amplitude
//! - [`dsqg`]             — Dyadic Sparse Q·K Gather (condL candidate) verification
//! - [`cond_d_db4`]       — condD-DB4: D4 kernel + KdV on matrix fields verification
//! - [`condl_ablation`]   — condL design: ELU normaliser variance + RG init persistence
//! - [`rank_bottleneck`]      — outer-product field rank capacity vs head dimension (13M/85M/7B)
//! - [`coverage_attractor`]  — coverage density metrics + collapse/copy attractor dynamics (condP/condM)
//!
//! # Running tests
//!
//! ```bash
//! cd verification && cargo test -- --nocapture
//! ```
//!
//! Original 28 tests + 15 (dsqg/cond_d_db4) + 2 condl_ablation + 4 rank_bottleneck
//! + 3 coverage_attractor + 6 hop_reachability + 3 variance_vanishing = 61 total.
//! (3 known failures in cond_d_db4: KdV instability, correct behavior)

pub mod cond_d_db4;
pub mod cond_m;
pub mod cond_o;
pub mod condm_sawtooth;
pub mod condl_ablation;
pub mod coverage_attractor;
pub mod dsqg;
pub mod fk_norm;
pub mod hop_reachability;
pub mod nonlinear_field;
pub mod rank_bottleneck;
pub mod rg_init;
pub mod soliton;
pub mod variance_vanishing;
