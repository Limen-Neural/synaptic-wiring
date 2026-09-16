// SPDX-License-Identifier: MIT OR Apache-2.0

//! # synaptic-wiring
//!
//! The **connectivity layer** of a spiking neural network: which neuron
//! connects to which, how strongly, and how long the spike takes to arrive.
//! You bring the neuron model and the simulation loop; this crate wires the
//! network and delivers each spike to the right target on the right tick.
//!
//! Reach for it to generate biologically plausible topologies, give synapses
//! axonal delays, respect Dale's law, or store a large sparse weight matrix
//! compactly — all deterministically, with `serde` and `sha2` as the only
//! runtime dependencies.
//!
//! ## Where to start
//!
//! 1. [`topology::generate_small_world`] (or [`generate_random`],
//!    [`generate_scale_free`], [`generate_layered`]) to build a
//!    [`SynapticGraph`], or [`SynapticGraph::from_descriptors`] to specify
//!    every synapse yourself.
//! 2. [`SynapticMesh::new`] to wrap it, then [`SynapticMesh::propagate`] once
//!    per tick — its docs state the full delivery contract (destination,
//!    sign, magnitude, tick). Reuse a caller-owned buffer with
//!    [`SynapticMesh::propagate_into`] and
//!    [`SynapticMesh::propagate_graded_into`] when a
//!    fixed-rate loop must avoid allocating every tick.
//! 3. [`topology::assign_delays`] to rewrite a descriptor list's delays from
//!    neuron positions, and [`topology::apply_dale_polarity`] to compute a
//!    per-neuron excitatory/inhibitory split — both work on the inputs to
//!    [`SynapticGraph::from_descriptors`], not on an already-built graph.
//! 4. [`SynapticGraph::topology_digest`] (or [`SynapticMesh::topology_digest`])
//!    for a printable, versioned identifier of the logical graph, usable in
//!    manifests and cross-runtime comparisons.
//! 5. [`sparse::SparseSynapticMap`] or [`SynapticMesh::to_gpu_arrays`] when
//!    you need the weights as flat CSR arrays.
//! 6. [`router::ChannelRouter`] only if you also want a sparse channel
//!    classifier; it is independent of the mesh and safe to ignore.
//!
//! [`generate_random`]: topology::generate_random
//! [`generate_scale_free`]: topology::generate_scale_free
//! [`generate_layered`]: topology::generate_layered
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`topology`] | Network graph construction — CSR adjacency with delay & polarity metadata, deterministic generators (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, layered), and a versioned [`TopologyDigest`] |
//! | [`delay`] | Temporal delay infrastructure — ring-buffer spike queues for tick-aligned delivery with configurable axonal propagation delays |
//! | [`mesh`] | [`SynapticMesh`] orchestrator — the top-level struct owning topology + delays, provides `propagate()` / `propagate_into()` for spike → current conversion |
//! | [`sparse`] | Compressed Sparse Row (CSR) synaptic maps for GPU-optimized weight matrices |
//! | [`router`] | Optional multi-channel classifier built on [`NeuromodNeuron`], a router-internal integration primitive |
//!
//! ## Crate boundary
//!
//! This crate owns connectivity and timing. Neuron models (LIF, Izhikevich,
//! Hodgkin-Huxley, …), learning rules, and training loops are deliberately
//! out of scope — pair it with whatever integrator you already use, or with a
//! dedicated crate such as `neuromod`, on which `synaptic-wiring` takes **no**
//! dependency so the two can evolve independently.
//!
//! [`NeuromodNeuron`] is the one neuron-like type here: an integration
//! primitive internal to [`router::ChannelRouter`], not a general-purpose
//! neuron model.
//!
//! ## Quick start
//!
//! ```rust
//! use synaptic_wiring::topology::generate_small_world;
//! use synaptic_wiring::mesh::SynapticMesh;
//!
//! // Build a 256-neuron small-world network with delays up to 5 ticks
//! let graph = generate_small_world(256, 6, 0.2, 5, 0.2).unwrap();
//! let mut mesh = SynapticMesh::new(graph);
//!
//! // Each tick: provide spike vector → receive delayed synaptic currents
//! let mut spikes = vec![false; 256];
//! spikes[0] = true; // neuron 0 fires
//!
//! let currents = mesh.propagate(&spikes).unwrap();
//! // currents[i] = total incoming synaptic current at neuron i this tick
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  Source spike vector  [bool; N]                   │
//! └────────────────┬─────────────────────────────────┘
//!                  │
//!        ┌─────────▼─────────┐
//!        │   SynapticGraph   │  CSR adjacency + delays + polarities
//!        │   (topology)      │  Generators: random, small-world,
//!        │                   │  scale-free, layered
//!        └─────────┬─────────┘
//!                  │  per-synapse: (target, weight, delay)
//!        ┌─────────▼─────────┐
//!        │  SpikeDelayBuffer │  Ring-buffer delay queue
//!        │   (delay)         │  inject() → advance() → drain()
//!        └─────────┬─────────┘
//!                  │  tick-aligned delivery
//!        ┌─────────▼─────────┐
//!        │  Synaptic current │  Vec<f32> of length N
//!        │  per target neuron│
//!        └───────────────────┘
//! ```
//!
//! ## References
//!
//! - Lapicque, L. (1907). *Recherches quantitatives sur l'excitation électrique des
//!   nerfs traitée comme une polarisation.* Journal de Physiologie et de Pathologie
//!   Générale, 9, 620–635.
//! - Stein, R. B. (1967). *Some models of neuronal variability.* Biophysical
//!   Journal, 7(1), 37–68.
//!
//! **Dale's law:**
//! - Dale, H. H. (1935). *Pharmacology and Nerve-endings.* Proceedings of the
//!   Royal Society of Medicine, 28(3), 319–332.
//!
//! **Network topology:**
//! - Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of 'small-world'
//!   networks.* Nature, 393, 440–442.
//! - Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random networks.*
//!   Science, 286, 509–512.
//!
//! **STDP / Hebbian plasticity:**
//! - Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
//! - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
//!   hippocampal neurons: dependence on spike timing, synaptic strength, and
//!   postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464–10472.
//!
//! **Winner-take-all lateral inhibition:**
//! - Maass, W. (2000). On the computational power of winner-take-all.
//!   *Neural Computation*, 12(11), 2519–2535.

// ── New modules: wiring, topology, delays ─────────────────────────────────────
pub mod delay;
pub mod error;
pub mod mesh;
pub mod topology;
pub mod types;

// ── Existing modules: router + sparse maps ────────────────────────────────────
pub mod router;
pub mod sparse;

// ── Public re-exports ─────────────────────────────────────────────────────────

// New mesh infrastructure
pub use delay::SpikeDelayBuffer;
pub use error::{MeshError, Result};
pub use mesh::SynapticMesh;
pub use topology::{
    SynapticGraph, TOPOLOGY_DIGEST_ALGORITHM, TOPOLOGY_DIGEST_DOMAIN,
    TOPOLOGY_DIGEST_SCHEMA_VERSION, TopologyDigest,
};
pub use types::{
    ConnectionModel, DelayModel, DelayTicks, NeuronId, Polarity, SynapseDescriptor, TopologyConfig,
};

// Generic router exports (NeuromodNeuron is a router-internal NIF primitive; see "Crate boundary" above)
pub use router::{
    ChannelRouter, MAX_ROUTER_CHANNELS, MAX_ROUTING_TIMESTEPS, NeuromodNeuron, NeuromodState,
    RouterConfig, RoutingDecision,
};

// Sparse map exports
pub use sparse::{
    MAX_SPARSE_NEURONS, NeuronStateSnapshot, RoutingPolicy, SparseSynapticMap,
    SparseSynapticMapBuilder, Synapse,
};

#[cfg(test)]
mod tests;

/// Compiles the Rust examples in `README.md` as doctests, so the quickstart a
/// new user copies is guaranteed to build against the current API using this
/// crate alone. `cfg(doctest)` keeps the README out of the rendered docs.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
