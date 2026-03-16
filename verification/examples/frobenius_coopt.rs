//! Co-optimized offset set search: Coverage + PPL proxy + J budget.
//! Usage: cd verification && PATH="$HOME/.cargo/bin:$PATH" cargo run --release --example frobenius_coopt
fn main() {
    wave_field_verification::frobenius_coopt_search::run();
}
