# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Crate rename**: the Cargo package is now `synaptic-wiring` (was
  `synaptic-mesh`) so the first crates.io publish does not collide with
  [ruvnet/Synaptic-Mesh](https://github.com/ruvnet/Synaptic-Mesh). GitHub
  URLs point at [`Limen-Neural/synaptic-wiring`](https://github.com/Limen-Neural/synaptic-wiring)
  (the GitHub repository rename has landed; the old `synaptic-mesh` path
  redirects). The `SynapticMesh` type is unchanged.

### Added

- **Topology digest**: `SynapticGraph::topology_digest` / `SynapticMesh::topology_digest`
  return a printable, schema-versioned SHA-256 of the canonical logical graph
  (neuron count, sorted edges, IEEE weight bit patterns, delay, polarity).
  Insertion order, host endianness, and serde formatting do not affect the
  value. IEEE `+0.0` / `-0.0` are hashed as distinct bit patterns; NaN/Inf
  never appear because graph construction already rejects them
  (LIM-1219).
- **Mesh**: `SynapticMesh::propagate_into` and `propagate_graded_into` write
  this tick's currents into a caller-owned `&mut [f32]`. The allocating
  `propagate` / `propagate_graded` methods remain as source-compatible
  wrappers. Wrong-sized or non-finite input is rejected before tick, delay
  buffer, or output mutation. After `new()`, a successful reuse-path tick
  performs no heap allocations (issue LIM-1222).
- **Delay buffer**: `SpikeDelayBuffer::drain_current_tick_into` drains into
  a caller-owned buffer; the allocating drain is a wrapper around it.
- **Benches**: Criterion cases for 16 / 256 / 4096 neurons with short and
  long delays, comparing allocating vs caller-buffer propagation.
- **Tests**: `tests/checkpoint_resume/` — seeded property/fuzz coverage that a
  restored `SynapticMesh` continues tick-for-tick identically to the live mesh
  with spikes in flight (currents, tick, queued deliveries). Covers generated
  graphs, signed weights, empty ticks, max-delay capacity, checkpoint-before-
  delivery, JSON plus postcard restore, persisted regression seeds, and a
  documented ignored nightly profile (`CHECKPOINT_RESUME_CASES`).
  Invalid checkpoints stay rejected by the existing load-path tests rather
  than being normalized here (LIM-1223).
- **Packaging**: Docker + GHCR container for releases (`ghcr.io/limen-neural/synaptic-mesh`).
  Library crate (no `examples/` / `[[bin]]`), so the image is a rustdoc snapshot
  plus a version stamp rather than a fake binary. PR workflow verifies without
  pushing; `main` publishes SHA tags; `v*` tags also publish version + `latest`
  when the tag matches `Cargo.toml` (including prerelease) and the tagged
  commit is already on `main`. Anonymous GHCR pulls need the package set
  public in GitHub Packages (`packages: write` only pushes).
  (issue #78 / LIM-1178).
- **CI**: Codecov coverage workflow (`.github/workflows/coverage.yml`)
  uploads LCOV from `cargo llvm-cov nextest` on PRs and `main`.
  The MSRV pin-agreement check now also requires `coverage.yml`
  to stay on the same Rust version as `ci.yml`. JUnit results are
  uploaded from `target/nextest/ci/junit.xml` (issue #77 / LIM-1180).

### Fixed

- **Packaging**: Docker rustdoc snapshot path now matches the `synaptic-wiring`
  crate name (`target/doc/synaptic_wiring/`) after the rename (LIM-1219).
- **ChannelRouter**: `route` and `route_modulated` reject NaN/±Inf channel
  signals and neuromodulator fields before mutating neurons, fatigue,
  adaptive weights, or `total_routes`. Errors name the channel index
  (`MeshError::NonFiniteSignal`) or modulator field
  (`MeshError::NonFiniteNeuromodulator`). Finite modulator values outside
  `[0, 1]` are rejected (`MeshError::OutOfRangeNeuromodulator`) rather than
  silently clamped. Finite signed signals keep their existing weighted-sum
  behavior. (LIM-1229)

## [0.3.0] - 2026-09-13

Everything below ships as **0.3.0**, the first crates.io release: packaging,
docs, and the propagate contract aimed at general SNN users depending on the
crate without any Limen context. Issue #37 promotes this section to
`## [0.3.0] - <publish date>` when it cuts the tag.

### Added

- **Packaging**: `documentation = "https://docs.rs/synaptic-wiring"` and an
  explicit `readme = "README.md"` in `Cargo.toml` (issue #34).
- **Tests**: `tests/propagate_contract.rs` — deterministic consumer contract
  for `SynapticMesh::propagate` over a fixed four-neuron graph, asserting the
  destination, sign, magnitude, and delivery tick of every spike, plus
  co-arrival summing, one-hop semantics, and replay after `reset()`
  (issue #52).
- **Docs**: `SynapticMesh::propagate` rustdoc now states the delivery
  contract explicitly and carries a runnable minimal example (issue #52).
- **Docs**: README and crate-level docs are written for general SNN use —
  what the crate is for, a "Where to start" map of the public API, the spike
  delivery contract, an explicit scope boundary, and a "Used by" footnote
  instead of downstream-specific framing (issue #53).
- **Tests**: README Rust examples are compiled as doctests (`cfg(doctest)`
  `include_str!`), so the quickstart cannot drift from the API (issue #53).
- **CI**: the MSRV pin-agreement check also verifies the README's MSRV
  line (issue #53).
- **CI**: `package` job running `cargo package --locked`, asserting the
  packaged file list against an allowlist of consumer-relevant files, and
  building the unpacked `.crate` outside the repository so an over-eager
  `exclude` or a repo-only build dependency fails CI instead of crates.io
  (issue #36).
- **CI**: Linear Release workflow (`.github/workflows/linear-release.yml`)
  syncs GitHub commits/PRs into the Linear synaptic-mesh Releases pipeline
  (`command: sync` on `main`; tag `v*` syncs then `complete`s that version).
  GitHub remains source of truth for issue state (issue #72 / LIM-1150).


### Fixed

- **Sparse maps**: insertion and construction now validate source/target
  indices against `N` with checked `u16` conversion, so column `65_536`
  cannot wrap to target `0`. `N` is capped at 65,536 neurons. `to_gpu_arrays`
  rejects `usize → u32` row-pointer overflow instead of truncating
  (issue #63). Additive `try_*` helpers leave the map unchanged on error.
- **Delay buffer**: `SpikeDelayBuffer::inject` now checks delay and target
  bounds in debug **and** release builds before writing a slot, so an
  oversized delay cannot wrap onto an earlier tick. `try_new` /
  `try_inject` reject `max_delay + 1` overflow and out-of-range inputs
  without mutating the buffer. `SynapticMesh::with_max_delay` (and
  additive `try_with_max_delay`) reject a buffer smaller than the graph's
  maximum delay at construction (issue #60).
- **Topology generators**: small-world now stores `k` directed outgoing
  synapses per source (`k/2` on each side) and rejects odd `k` instead of
  truncating. A requested rewire excludes the original target so `beta = 1`
  cannot reselect it when another unused target exists; a dense ring
  (`k = n - 1`) still falls back to the original synapse. Scale-free growth
  attaches every new node to exactly `m` distinct older nodes via a bounded
  preferential sample, keeping the reciprocal-edge policy. Identical inputs
  remain deterministic, but the generated graphs **change** relative to
  0.2.x (issue #62).
- **Validation**: `SynapticGraph::from_descriptors` now rejects negative,
  NaN, and infinite descriptor magnitudes (including values constructed
  as ordinary Rust structs that bypass serde). Graph deserialization
  rejects signed weights that disagree with stored polarity, while
  accepting IEEE signed zero for both polarities.
  `propagate_graded` rejects non-finite activations, non-finite
  `weight * activation` products, and non-finite per-slot aggregates
  (including existing delay-buffer current) before mutating the delay
  buffer or tick counters. `SynapseDescriptor::effective_weight` panics on invalid
  magnitudes in release builds so a struct-literal negative inhibitory
  weight cannot flip sign. Invalid magnitudes are not silently
  `abs()`-normalized (issue #61).
- **Serde checkpoints**: `SpikeDelayBuffer` and `SynapticMesh` now reject
  structurally inconsistent payloads at deserialize time (empty/wrong-depth
  rings, ragged slot widths, `max_delay + 1` overflow, graph/buffer count
  mismatch, inadequate delay capacity, tick mismatch, `current_tick` values
  that cannot safely advance (including `usize::MAX - 1`, which would land
  on `usize::MAX` on the next tick) or add `max_delay` without overflowing
  `usize` indexing, and non-finite slot currents) instead of panicking or delivering
  currents on the wrong tick later (issue #64). Valid existing checkpoint
  JSON is unchanged.

### Removed

- **CI**: the Qodana workflow (`.github/workflows/qodana_code_quality.yml`)
  and `qodana.yaml`. Qodana Cloud membership expired, so the scan job was
  failing on license token decline. Build & Test, cargo-deny, and the
  crates.io package dry-run are unchanged.

### Changed

- **MSRV**: Rust pin raised from **1.97.1** to **1.98.1** in `Cargo.toml`
  `rust-version`, `rust-toolchain.toml`, and CI (both the toolchain install
  and the `cargo-deny` action) ahead of the first crates.io publish
  (issue #51).
- **CI**: the MSRV pin-agreement check now compares *every* `toolchain:` /
  `rust-version:` pin in `.github/workflows/ci.yml` against `Cargo.toml`,
  instead of only the first toolchain install, so a partially bumped or
  newly added job fails the check (issue #51).
- **CI**: `validate` now runs `cargo test --locked --release --all-features`
  so delay-buffer capacity checks are exercised with `debug_assert!`
  stripped (issue #60).
- **Version**: bumped to **0.3.0**, the version prepared for the first
  crates.io publish (issue #34; the publish itself is issue #37).
- **Metadata**: crates.io keywords are now `snn`, `spiking`, `neuromorphic`,
  `topology`, `routing` — `lif` (no neuron models live here) and `spikenaut`
  (a downstream consumer, not a description of the crate) were dropped
  (issue #53).
- **Packaging**: `exclude` now also drops `/.github/` and
  `/REVIEW.md` from the published `.crate` (issue #34), plus the remaining
  dev-tooling files `/.codacy.yml`, `/.devin/`, `/.gitignore`, `/AGENTS.md`,
  `/deny.toml`, and `/rust-toolchain.toml` (issue #49). The packaged crate
  now holds only source, `Cargo.toml`, `Cargo.lock`, the licenses, README,
  CHANGELOG, and the metadata cargo generates itself (`Cargo.toml.orig`,
  `.cargo_vcs_info.json`).
- **Docs**: README leads with the crates.io install path
  (`synaptic-wiring = "0.3"`), marks the git dependency as the bleeding-edge
  alternative, and describes 0.3.0 as experimental pre-1.0. Repo-relative
  logo and `REVIEW.md` links are absolute so they resolve on crates.io and
  docs.rs, where those files are not packaged (issue #34).

## [0.2.0] - 2026-09-06

Bridges v0.1.0 (API solidify) and v0.3.0 (first crates.io publish). Hardens
the build/test/bench setup ahead of packaging for publish.

### Added

- **Cargo profiles**: explicit `[profile.dev]`, `[profile.test]`,
  `[profile.release]`, and `[profile.bench]` in `Cargo.toml`, each tuned and
  documented with a rationale comment (issue #40).
- **CI**: workflow now builds under `release` and compiles under `bench`
  in addition to the existing `dev` build and `test` run, so a broken or
  reverted profile fails CI instead of only surfacing locally (issue #41).
- **CI**: `cargo-deny` job checking advisories, license allowlist (dual
  MIT/Apache-2.0), bans, and sources ahead of the crates.io publish
  (issue #42).
- **Docs**: "Build profiles" section in `REVIEW.md` listing each profile,
  the command that uses it, and why it's configured that way, cross-linked
  from `Cargo.toml` and `README.md` (issue #43).

[0.2.0]: https://github.com/Limen-Neural/synaptic-wiring/compare/v0.1.0...v0.2.0

## [0.1.0] - 2026-08-14

First versioned history. Nothing before this tag was published or tagged.

### Added

- Neuromodulatory adaptation in `ChannelRouter` — `NeuromodState` (cortisol,
  dopamine, serotonin), `route_modulated()`, `apply_plasticity()`, per-channel
  fatigue, use-it-or-lose-it plasticity.
- Generic `ChannelRouter` with configurable channel count.
- CSR sparse synaptic maps, topology generators (small-world, scale-free,
  random, layered), temporal delay ring-buffer, Dale's law wiring.
- **CI**: GitHub Actions workflow (`.github/workflows/ci.yml`) running
  `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, and
  `cargo test` on every push/PR to `main`.
- MSRV pin **1.97.1** in `Cargo.toml` `rust-version`, `rust-toolchain.toml`,
  and CI (`dtolnay/rust-toolchain` + pin-agreement check).
- `PartialEq` on `SynapticGraph` and `SynapseDescriptor`, plus a JSON
  serde round-trip test.
- `inhibitory_fraction` range validation to all topology generators
  (`generate_random`, `generate_small_world`, `generate_scale_free`,
  `generate_layered`) — rejects values outside `[0, 1]`.

### Changed

- **License**: Dual `MIT OR Apache-2.0`. SPDX identifier added to all source
  files.
- `apply_dale_polarity` (`topology::wiring_rules`) returns
  `Result<Vec<Polarity>, MeshError>` instead of `Vec<Polarity>`. Out-of-range
  `inhibitory_fraction` values (< 0 or > 1) are rejected with
  `MeshError::InvalidConfig`.

### Removed

- **`AhlRouter`** — use `ChannelRouter` with `RouterConfig::default()`.
- **`AHL_NUM_CHANNELS`** — default channel count is `RouterConfig::default().channel_count` (3).
- **`TelemetrySnapshot`** — use `NeuronStateSnapshot`.
- **`NeuronStateSnapshot::quant_bonus`** — use `NeuronStateSnapshot::error_bonus`.
- **`NeuronStateSnapshot::quant_error`** — use the `error` field.
  The serde `alias = "quant_error"` on `error` is kept so older snapshots
  still deserialize.

### Fixed

- **Test**: `mesh::tests::layered_mesh_feed_forward` — the test now models
  multi-hop propagation by thresholding received currents back into spikes
  (matching the documented one-hop `propagate()` semantics).
- **Underflow**: `SynapticGraph::validate_descriptor_indices` used
  `neuron_count - 1` which underflows on `neuron_count = 0`. Now uses
  `saturating_sub(1)`.
- **Consistency**: `apply_feedback` and `sync_baseline_after_feedback` now
  consistently use `self.config.channel_count` (matching all other methods)
  instead of `self.neurons.len()`. `apply_feedback` retains a double-guard
  for Codacy HIGH RISK out-of-bounds protection.
- **Defensive**: `sync_baseline_after_feedback` inner-row length check
  converted to `debug_assert!` documenting the call-ordering invariant
  (`ensure_neuromod_state_synced` rebuilds `baseline_weights` beforehand).
- Zero-size layer validation in `generate_layered` now rejects `&[0, 5]`
  via `layer_sizes.contains(&0)` with a clearer error message.

### Internal

- Resolved clippy warnings (`RangeInclusive::contains`, iterator idioms,
  derivable impls, redundant borrows).
- Fixed `cargo fmt` formatting (module ordering, debug_assert wrapping).

[Unreleased]: https://github.com/Limen-Neural/synaptic-wiring/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Limen-Neural/synaptic-wiring/compare/v0.2.0...v0.3.0
[0.1.0]: https://github.com/Limen-Neural/synaptic-wiring/releases/tag/v0.1.0
