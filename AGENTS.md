# Agent Notes

This file contains workflow and orientation notes for AI agents working on this repository.

## Repo Map

- `src/lib.rs`: crate root, re-exports from all modules
- `src/types.rs`: core type aliases (NeuronId, DelayTicks, Polarity, SynapseDescriptor) and config structs
- `src/error.rs`: `MeshError` unified error enum
- `src/topology/`: network graph construction and wiring rules
  - `graph.rs`: `SynapticGraph` — CSR adjacency with delay + polarity metadata
  - `generators.rs`: deterministic topology generators (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, layered)
  - `wiring_rules.rs`: Dale's law polarity assignment, distance-based delay assignment
- `src/delay/`: temporal delay infrastructure
  - `ring_buffer.rs`: `SpikeDelayBuffer` — ring-buffer delay queue for tick-aligned spike delivery
- `src/mesh.rs`: `SynapticMesh` — top-level orchestrator owning graph + delays, provides `propagate()` 
- `src/router.rs`: `ChannelRouter` — generic multi-channel SNN router (configurable channel count)
- `src/sparse.rs`: `SparseSynapticMap` CSR format, `NeuronStateSnapshot`, `RoutingPolicy`
- `src/tests.rs`: test suite for router + sparse modules

## Entry Order

- Start at `src/lib.rs` for the exported crate surface.
- Read `src/types.rs` for core type definitions.
- Read `src/topology/graph.rs` before `generators.rs` or `wiring_rules.rs`.
- Read `src/delay/ring_buffer.rs` for the temporal delay model.
- Read `src/mesh.rs` for the orchestrator that ties everything together.

## Design Principles

- **Determinism**: all generators use index-based pseudo-random hashing (golden-ratio fractional), no external RNG. Same inputs always produce the same topology.
- **Dale's Law**: neuron polarity is per-neuron (all outgoing synapses share polarity). First `inh_fraction × N` neurons are inhibitory.
- **Temporal Delays**: per-synapse axonal delays stored in CSR alongside weights. Ring-buffer delivers spikes at correct future tick.
- **Public API**: `ChannelRouter` + `RouterConfig` is the router surface. Deprecated aliases (`AhlRouter`, `AHL_NUM_CHANNELS`, `TelemetrySnapshot`, `quant_bonus`, `quant_error`) were removed in 0.1.0.

## Workflow Policy

- All `synaptic-wiring` work stays in `/home/raulmc/synaptic-wiring`.
- Sister crate: `corinth-canal` at `/home/raulmc/corinth-canal` (inspiration for architecture patterns).
- Run `cargo check`, `cargo test` before closing substantial Rust changes.
- Commits and PR titles should keep carrying Linear issue identifiers (Limen-Neural Linear team-number form) so Linear Release scanning can attach issues.
- GitHub is source of truth for issue state.

## GitHub Actions secrets

`.github/workflows/linear-release.yml` uses `linear/linear-release-action@v0` with `LINEAR_ACCESS_KEY`.

- **What it is**: the Linear pipeline access key for this repo's Linear Releases pipeline ([synaptic-mesh Releases](https://linear.app/rpd-34/pipeline/synaptic-mesh/releases)).
- **How to create it**: open that pipeline's settings page in Linear and copy the access key.
- **Where to store it**: GitHub `Settings → Secrets and variables → Actions`. Do not commit the key.
- **Current status**: set (created 2026-09-13). Verified working via a `workflow_dispatch` dry run on 2026-11-24 — it scanned the commit range and resolved release `0.5.0` against the synaptic-mesh pipeline without mutating Linear.

## Repository Context

- **Repo**: `Limen-Neural/synaptic-wiring` (GitHub redirects the former `Limen-Neural/synaptic-mesh` path)
- **Main branch**: `main`
- **Language**: Rust (edition 2024)
- **Key concepts**: SNN wiring, topology generation, axonal delays, CSR sparse maps, Dale's law
