<p align="center">
  <img src="https://raw.githubusercontent.com/Limen-Neural/synaptic-wiring/main/docs/logo.png" width="220" alt="synaptic-wiring">
</p>

<h1 align="center">synaptic-wiring</h1>
<p align="center">SNN wiring, topology generation, and temporal delay infrastructure</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.3.0-informational" alt="version 0.3.0">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="MIT OR Apache-2.0">
</p>

---

`synaptic-wiring` is the **connectivity layer** of a spiking neural network: it
answers *which neuron connects to which*, *how strongly*, and *how long the
spike takes to get there*. You bring the neuron model and the simulation loop;
this crate wires the network and delivers each spike to the right target on the
right tick.

The Cargo package is `synaptic-wiring` (this crate was previously named
`synaptic-mesh`). The GitHub repository is
[`Limen-Neural/synaptic-wiring`](https://github.com/Limen-Neural/synaptic-wiring)
(GitHub redirects the former `Limen-Neural/synaptic-mesh` URL).

It is a plain library with two runtime dependencies (`serde` and `sha2`) — no
framework, no runtime, no GPU requirement, and no assumptions about how your
neurons integrate current.

**Reach for it when you need to:**

- generate a biologically plausible network (small-world, scale-free, layered,
  or Erdős–Rényi) instead of hand-writing an adjacency matrix,
- give synapses **axonal delays** so spikes arrive on a future tick, enabling
  polychronization and coincidence detection,
- respect **Dale's law** — a neuron is excitatory or inhibitory, and all of its
  outgoing synapses follow,
- store a large sparse weight matrix compactly (CSR) and hand it to a GPU,
- get **reproducible** topologies: the generators hash neuron indices instead of
  drawing from an RNG, so the same parameters always produce the same network,
- fingerprint a mesh's logical wiring with `topology_digest()` for manifests,
  replay, and cross-runtime comparison.

**It deliberately does not** implement neuron models (LIF, Izhikevich, …),
learning rules, or training loops — those stay in your code or in a neuron-model
crate. See [Crate boundary](#crate-boundary).

## Core Capabilities

- **Topology Generators** — Deterministic generation of Erdős–Rényi (random), Watts–Strogatz (small-world), Barabási–Albert (scale-free), and Layered feed-forward topologies.
- **Temporal Propagation** — Per-synapse axonal delays stored alongside weights. Spikes are delivered at the correct future tick via a high-performance ring-buffer queue.
- **Biologically Inspired Wiring** — Support for Dale's Law (fixed neuron polarity) and position-based distance-dependent connectivity.
- **Sparse Synaptic Map (CSR)** — Compressed Sparse Row format for memory-efficient weight storage (20× reduction for sparse networks).
- **Topology digest** — Versioned SHA-256 of the canonical logical graph (sorted edges, IEEE weight bits, delay, polarity) for manifests and provenance.
- **Generic Channel Router** *(optional)* — A configurable multi-channel router for sparse signal classification, usable on its own or ignored entirely.

## Where to start

| If you want to… | Use | Section |
|-----------------|-----|---------|
| Wire up a network from a classic model | `topology::generate_small_world` and friends | [Topology Generation](#topology-generation) |
| Run spikes through it, tick by tick | `SynapticMesh::propagate` | [Quick Start](#quick-start-building-a-mesh) |
| Hand-build an exact graph | `SynapticGraph::from_descriptors` with `SynapseDescriptor` | [Spike delivery contract](#spike-delivery-contract) |
| Plan polarity or distance-based delays *before* building a graph | `topology::apply_dale_polarity` (computes a per-neuron polarity vector), `topology::assign_delays` (rewrites descriptors in place) | [Temporal Delays](#temporal-delays--spike-propagation) |
| Store a big sparse weight matrix / upload to a GPU | `SparseSynapticMap`, `SynapticMesh::to_gpu_arrays` | [Core Capabilities](#core-capabilities) |
| Record topology in a manifest / compare two graphs | `SynapticGraph::topology_digest` | [Topology digest](#topology-digest) |
| Pick a few active channels out of many inputs | `ChannelRouter` | [Generic Channel Router](#generic-channel-router) |

## Installation

```toml
[dependencies]
synaptic-wiring = "0.3"
```

`0.3.0` is **experimental, pre-1.0**: the API is usable and tested, but minor
releases may still contain breaking changes until 1.0. Pin an exact version
(`= "0.3.0"`) if you need stability.

For Development / unreleased main — changes that are not in a release yet —
depend on git instead:

```toml
synaptic-wiring = { git = "https://github.com/Limen-Neural/synaptic-wiring" }
```

Add `tag = "v0.3.0"` to pin a git dependency to a specific release instead of tracking `main` — see [releases](https://github.com/Limen-Neural/synaptic-wiring/releases) for the tags that exist today.

**MSRV:** Rust **1.98.1**. Runtime dependencies: `serde`, `sha2`.

Contributors: see
[REVIEW.md](https://github.com/Limen-Neural/synaptic-wiring/blob/main/REVIEW.md#build-profiles)
for which cargo build profile (`dev`, `test`, `release`, `bench`) to use and why.

## Docker (optional)

Published images are a **docs snapshot** for release packaging (not a
substitute for depending on the crate from Cargo). This crate has no
example binaries.

Anonymous `docker pull` works only after the GHCR package is **public**.
The publish workflow has `packages: write` so it can push images; GitHub
still creates a new package as private until an org owner sets visibility
to public under [GitHub Packages](https://github.com/orgs/Limen-Neural/packages).

```bash
docker pull ghcr.io/limen-neural/synaptic-wiring:0.3.0
docker run --rm ghcr.io/limen-neural/synaptic-wiring:0.3.0
```

Build locally from a git checkout:

```bash
docker build -t synaptic-wiring:dev .
docker run --rm synaptic-wiring:dev

docker build --target builder -t synaptic-wiring:builder .
docker run --rm synaptic-wiring:builder   # cargo test --all-features --locked
```

## Quick Start: Building a Mesh

```rust
use synaptic_wiring::topology::generate_small_world;
use synaptic_wiring::mesh::SynapticMesh;

// 1. Build a 1024-neuron small-world network with delays up to 10 ticks
// (N=1024, k=6 neighbors, beta=0.1 rewiring, max_delay=10, inh_fraction=0.2)
let graph = generate_small_world(1024, 6, 0.1, 10, 0.2).unwrap();

// 2. Wrap in a Mesh orchestrator that manages temporal state
let mut mesh = SynapticMesh::new(graph);

// 3. Each tick: provide current spikes -> receive time-delayed synaptic currents
let mut spikes = vec![false; 1024];
spikes[0] = true; // neuron 0 fires

let currents = mesh.propagate(&spikes).unwrap();
// currents[i] = total incoming synaptic current at neuron i this tick,
// potentially including delayed spikes from previous ticks.
```

### Spike delivery contract

`propagate()` is **one hop**: it converts the spikes you hand it into the
currents arriving *this* tick. It never decides what fires next — that is your
neuron model's job. A complete simulation loop is therefore just:

```rust
use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::SynapticGraph;
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

// Hand-built graph: 0 ──(w=0.75, delay 0, excitatory)──▶ 1
//                   0 ──(w=0.50, delay 2, excitatory)──▶ 2
let graph = SynapticGraph::from_descriptors(3, &[
    SynapseDescriptor { source: 0, target: 1, weight: 0.75, delay: 0, polarity: Polarity::Excitatory },
    SynapseDescriptor { source: 0, target: 2, weight: 0.50, delay: 2, polarity: Polarity::Excitatory },
]).unwrap();
let mut mesh = SynapticMesh::new(graph);

let mut spikes = vec![false, false, false];
spikes[0] = true;

for _tick in 0..3 {
    let currents = mesh.propagate(&spikes).unwrap();
    // Your neuron model goes here; this one is a bare threshold.
    spikes = currents.iter().map(|&c| c > 0.3).collect();
}
```

The rules it guarantees:

| Question | Answer |
|----------|--------|
| **Who** receives the current? | Every target of the firing neuron, per the graph. |
| **What sign?** | The synapse's polarity — inhibitory synapses subtract. The generators assign polarity per source neuron (Dale's law); a hand-built graph gets whatever each descriptor declares. |
| **What magnitude?** | The synapse weight, multiplied by the activation when using `propagate_graded` (a negative activation therefore flips the sign). |
| **Which tick?** | `tick_fired + delay`; `delay = 0` arrives in the same call. |
| **What if two spikes land together?** | They sum at the destination. |
| **Is it reproducible?** | Yes — same graph and spikes give the same currents, including after `reset()`. |

[`tests/propagate_contract.rs`](https://github.com/Limen-Neural/synaptic-wiring/blob/main/tests/propagate_contract.rs)
pins every one of these against a fixed four-neuron graph and is a copyable
starting point for your own loop.

A fixed-rate simulation that already owns a current vector can reuse it with
`propagate_into` / `propagate_graded_into` instead of allocating every tick.
`output` is **overwritten** with this tick's currents (not accumulated into).
Length mismatches and non-finite graded input are rejected before the mesh
tick, delay buffers, or `output` change. After `SynapticMesh::new`, a
successful reuse-path tick does not heap-allocate.

```rust
use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::SynapticGraph;
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

let graph = SynapticGraph::from_descriptors(2, &[
    SynapseDescriptor { source: 0, target: 1, weight: 0.75, delay: 0, polarity: Polarity::Excitatory },
]).unwrap();
let mut mesh = SynapticMesh::new(graph);
let spikes = [true, false];
let mut currents = [0.0; 2];
mesh.propagate_into(&spikes, &mut currents).unwrap();
assert_eq!(currents, [0.0, 0.75]);
```

## Topology Generation

`synaptic-wiring` provides several deterministic models for growing network graphs. All generators use golden-ratio fractional hashing for reproducibility across runs without external RNG dependencies.

| Model | Generator | Best For |
|-------|-----------|----------|
| **Small-World** | `generate_small_world` | Local clustering with short path lengths (mimics cortical connectivity). |
| **Scale-Free** | `generate_scale_free` | Networks with "hubs" following a power-law degree distribution. |
| **Random** | `generate_random` | Erdős–Rényi random graphs for baseline comparisons. |
| **Layered** | `generate_layered` | Classical feed-forward structures (Input -> Hidden -> Output). |

## Topology digest

`SynapticGraph::topology_digest()` (and `SynapticMesh::topology_digest()`) returns
a printable, schema-versioned SHA-256 of the **logical** graph: neuron count and
every edge's source, target, IEEE-754 weight bit pattern, delay, and polarity.
Edges are sorted before hashing, so insertion order, CSR layout, host endianness,
and JSON formatting cannot change the value.

The string is safe to store in a manifest:

```text
synaptic-wiring.topology.digest.v1:sha256:<64 hex chars>
```

IEEE `+0.0` and `-0.0` digest differently (bit patterns are hashed as-is).
NaN and infinities cannot appear: graph construction and serde already reject
them. This is an identifier, not a cryptographic signature.

See [`tests/topology_digest.rs`](https://github.com/Limen-Neural/synaptic-wiring/blob/main/tests/topology_digest.rs)
for golden values and an insertion-order permutation test.

## Temporal Delays & Spike Propagation

In biological networks, spikes do not arrive instantly. `synaptic-wiring` implements a temporal logic layer using a **Ring-Buffer Delay Queue**:

1.  Each synapse in the `SynapticGraph` stores a `DelayTicks` value.
2.  When a neuron fires, its spike is projected through its outgoing synapses.
3.  The `SpikeDelayBuffer` schedules delivery at `current_tick + delay`.
4.  At each tick, `propagate()` drains the current slot and returns the accumulated currents.
    `propagate_into` does the same write into a caller-owned buffer.

This enables complex temporal dynamics like polychronization and coincidence detection.

## Generic Channel Router

*Optional — skip this section if you only need wiring and delays.*

`ChannelRouter` is a standalone, configurable multi-channel classifier: it integrates input pulses over a bank of internal integrate-and-fire units and returns a sparse activation mask (which channels won, and at what firing rate). It does not use `SynapticMesh` and `SynapticMesh` does not use it — they are independent halves of the crate.

```rust
use synaptic_wiring::ChannelRouter;

// Default 3-channel router. `route` returns a Result: it errors if the
// signal slice length doesn't match the configured channel count, or if
// any sample is NaN / ±infinity. Finite signed samples are accepted.
let mut router = ChannelRouter::new();
let decision = router.route([0.8, 0.2, 0.1]).unwrap();

assert_eq!(decision.active_channels, vec![0]);
assert_eq!(decision.firing_rates.len(), 3);
```

### Configurable Channel Count

```rust
use synaptic_wiring::{ChannelRouter, RouterConfig};

let config = RouterConfig {
    channel_count: 8,
    ..RouterConfig::default()
};
let mut router = ChannelRouter::try_with_config(config).unwrap();
let decision = router.route([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

assert_eq!(decision.active_channels, vec![3]);
```

## Adaptation-Aware Routing

- **Neuron State Snapshots** — Per-neuron adaptation and error tracking for dynamic routing decisions.
- **Routing Policies** — Configurable scoring equations that balance spike activity, adaptation penalties, and error bonuses.

## Crate boundary

This crate owns connectivity and timing, and nothing else:

| In scope | Out of scope (bring your own) |
|----------|-------------------------------|
| Topology generation and hand-built graphs | Neuron models — LIF, Izhikevich, Hodgkin-Huxley, … |
| Per-synapse weights, polarity, axonal delays | Learning rules, training loops, optimizers |
| Tick-aligned spike delivery | Simulation scheduling, I/O, encoding of stimuli |
| CSR sparse maps and GPU-ready arrays | GPU kernels, hardware backends |
| Optional `ChannelRouter` classifier | — |

Neuron models are a deliberate omission, not a gap: pair this crate with whatever integrator you already use, or with a dedicated crate such as [`neuromod`](https://github.com/Limen-Neural/neuromod) (LIF, Izhikevich, Hodgkin-Huxley, GIF, FitzHugh-Nagumo, Lapicque). `synaptic-wiring` takes **no dependency** on it, so the two evolve independently.

The one neuron-like type here, `NeuromodNeuron` in [`router`](src/router.rs), is an integration primitive internal to `ChannelRouter` — not a general-purpose neuron model.

## Used by

Spikenaut-SNN uses this crate for Dale-polarity wiring and multi-channel
routing. That is a downstream consumer, not a requirement: nothing in the API
assumes it, and depending on `synaptic-wiring` pulls in `serde` and `sha2`.

## Architecture

```text
┌──────────────────────────────────────────────────┐
│  Source spike vector  [bool; N]                   │
└────────────────┬─────────────────────────────────┘
                 │
       ┌─────────▼─────────┐
       │   SynapticGraph   │  CSR adjacency + delays + polarities
       │   (topology)      │  Generators: random, small-world, etc.
       └─────────┬─────────┘
                 │  per-synapse: (target, weight, delay)
       ┌─────────▼─────────┐
       │  SpikeDelayBuffer │  Ring-buffer delay queue
       │   (delay)         │  inject() → advance() → drain()
       └─────────┬─────────┘
                 │  tick-aligned delivery
       ┌─────────▼─────────┐
       │  Synaptic current │  Vec<f32> of length N
       │  per target neuron│
       └───────────────────┘
```

## References

**Network Topology & Dynamics:**
- Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of 'small-world' networks.* Nature.
- Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random networks.* Science.
- Dale, H. H. (1935). *Pharmacology and Nerve-endings.*

**SNN Logic:**
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications... *Journal of Neuroscience*.
- Maass, W. (2000). On the computational power of winner-take-all. *Neural Computation*.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option. Contributions intentionally submitted for inclusion in this
crate by you, as defined in the Apache-2.0 license, shall be dual-licensed as
above, without any additional terms or conditions.
