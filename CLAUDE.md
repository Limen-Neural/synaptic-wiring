# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`synaptic-wiring` is a small Rust library (edition 2024, published to crates.io, pre-1.0) for the connectivity layer of a spiking neural network: topology generation, per-synapse axonal delays, and tick-aligned spike delivery. It deliberately contains no neuron models, learning rules or training loops. Runtime dependencies are only `serde` and `sha2`.

## Commands

The toolchain is pinned by `rust-toolchain.toml` (currently 1.99.0). CI runs these, and `REVIEW.md` lists the local quality gate:

```bash
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
cargo test --locked --release --all-features     # debug_assert! stripped: bounds checks must still reject bad input
RUSTDOCFLAGS="-D warnings" cargo doc --locked --all-features --no-deps --document-private-items
cargo bench --locked --all-features --no-run     # CI only compiles the criterion bench
cargo package --locked --list                    # must match the allowlist in the `package` job of ci.yml
```

- Single test: `cargo test --locked <name_substring>`; one integration binary: `cargo test --locked --test propagate_contract`.
- Doctests matter: the README's Rust blocks are compiled as doctests via `#[cfg(doctest)] include_str!("../README.md")` in `src/lib.rs`, so editing README examples can break `cargo test`.
- Long checkpoint fuzz run: `cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored` (`CHECKPOINT_RESUME_CASES=50000` to scale; persist failing seeds in `REGRESSION_SEEDS`).
- Coverage CI uses `cargo llvm-cov nextest` with the `ci` profile in `.config/nextest.toml`.

## MSRV pin rule

The Rust version must be identical in `Cargo.toml` `rust-version`, `rust-toolchain.toml`, the README's `MSRV` line, and every `toolchain:` / `rust-version:` pin in `ci.yml` and `coverage.yml`. The `validate` job fails on drift. Never bump only one (see `REVIEW.md`).

## Architecture

`SynapticMesh` (`src/mesh.rs`) is the entry point. It owns a `SynapticGraph` (`src/topology/graph.rs`, CSR adjacency with weight, delay and polarity per synapse) plus a `SpikeDelayBuffer` (`src/delay/ring_buffer.rs`, `max_delay + 1` slots of per-neuron current). One tick of `propagate` / `propagate_into`:

1. For each spiking source, inject every outgoing synapse's weight into the slot `delay` ticks ahead.
2. Drain the current slot into the output, then advance the ring and the tick.

Consequences worth knowing before editing:

- `propagate` is **one hop**. It never decides what fires next; `delay = 0` arrives in the same call. `tests/propagate_contract.rs` pins destination, sign, magnitude and delivery tick, and the README and rustdoc state the same contract. Treat a behavior change there as a breaking change.
- Fallible entry points are **error-atomic**: input is validated (length, finiteness) before the tick, delay buffer or output is mutated.
- The `*_into` path must not heap-allocate after `SynapticMesh::new` (the graded variant relies on a pre-reserved `pending_scratch`, skipped by serde). `tests/propagate_into_alloc.rs` enforces this with a counting global allocator, so it must stay its own test binary.
- Polarity follows Dale's law per source neuron. `SynapseDescriptor.weight` is a non-negative magnitude and `effective_weight()` applies the sign from polarity; the graph stores signed weights. Negative or non-finite magnitudes are rejected, not normalized.

### Conventions that span files

- **Validating deserialization.** Checkpointable types (`SynapticMesh`, `SpikeDelayBuffer`, `SynapticGraph`, `RouterConfig`, `ChannelRouter`) deserialize through a private `Raw*` shadow struct that re-checks invariants (neuron counts agree, delay capacity, tick agreement, non-finite values) and rejects rather than repairs. `ChannelRouter` additionally fills legacy snapshots that omit fields (`LegacyField`) but rejects explicit `null`. New serializable state should follow this pattern and be covered by `tests/checkpoint_resume/`.
- **`try_*` vs panicking constructors.** Public APIs keep source-compatible panicking forms (`with_max_delay`, `with_config`, `inject`) that delegate to additive `try_*` forms returning `MeshError`. Add the `try_*` form first.
- **Determinism, no RNG.** Generators in `src/topology/generators.rs` derive everything from index hashing (golden-ratio fractions). Changing the math changes every generated graph: record it in the CHANGELOG and update the golden values in `tests/topology_digest.rs`.
- **Topology digest** (`src/topology/digest.rs`) is a versioned SHA-256 of the canonical logical graph. Changing what is hashed requires bumping `TOPOLOGY_DIGEST_SCHEMA_VERSION`.
- **Two CSR types.** `SynapticGraph` (u32 targets, delays and polarity) is what the mesh uses. `SparseSynapticMap<const N>` in `src/sparse.rs` is a separate GPU-oriented map with `u16` indices capped at `MAX_SPARSE_NEURONS` (65,536). `to_gpu_arrays` exists on both.
- **Router is independent.** `ChannelRouter` (`src/router.rs`, the largest file) neither uses nor is used by the mesh. Bare `f32`/`usize` arguments there are wrapped in newtypes (`SynapticDrive`, `ModulationGain`, `ChannelIndex`) to satisfy CodeScene's primitive-obsession check.
- `MeshError` (`src/error.rs`) is the single error enum for the crate.

## Process

- Commit and PR titles carry a Linear identifier (`LIM-1234`) so Linear Release scanning attaches the issue. GitHub is the source of truth for issue state.
- User-visible changes go under `[Unreleased]` in `CHANGELOG.md`. The crate is pre-1.0, so breaking changes are allowed in minor releases, but document the migration. The unreleased router signature changes already require the next release to be 0.4.0.
- The published `.crate` holds only source, manifests, licenses, README and CHANGELOG. A new top-level file needs an `exclude` entry in `Cargo.toml` or an addition to the allowlist in `ci.yml`. `docs/` is excluded and currently holds only the README logo.
- Third-party GitHub Actions are pinned to commit SHAs.
- Code-quality bots shape file layout: Codacy flags long files and methods (the checkpoint test suite is split into small modules for that reason), so prefer splitting over growing a file.
- `AGENTS.md` is tracked despite a `.gitignore` entry. Its repo map predates `topology/digest.rs`, `tests/` and `benches/`.
