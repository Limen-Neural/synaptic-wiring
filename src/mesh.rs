// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`SynapticMesh`] — top-level orchestrator owning topology + delays.
//!
//! The `SynapticMesh` is the primary public-facing struct of this crate.
//! It owns a [`SynapticGraph`] (the wiring diagram) and a [`SpikeDelayBuffer`]
//! (the temporal delay infrastructure), and provides a single `propagate()`
//! method that converts source spikes into delayed synaptic currents.
//!
//! # Usage
//!
//! ```rust
//! use synaptic_wiring::mesh::SynapticMesh;
//! use synaptic_wiring::topology::generate_random;
//!
//! let graph = generate_random(64, 0.1, 5, 0.2).unwrap();
//! let mut mesh = SynapticMesh::new(graph);
//!
//! // Each tick: provide binary spike vector, receive synaptic currents
//! let spikes = vec![false; 64];
//! let currents = mesh.propagate(&spikes);
//! ```

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

use crate::delay::{SpikeDelayBuffer, validate_current_tick};
use crate::error::{MeshError, Result};
use crate::topology::{SynapticGraph, TopologyDigest};

/// Top-level synaptic wiring orchestrator.
///
/// Owns the network topology (graph with weights, delays, polarities) and
/// the temporal delay buffer. Converts source spikes into time-delayed
/// synaptic currents delivered to target neurons.
///
/// Deserialization requires the graph and buffer neuron counts to agree,
/// the buffer capacity to be at least the graph's maximum delay, and
/// `tick` to equal `delay_buffer.current_tick`. Both ticks must leave
/// headroom so one `propagate` cannot land on `usize::MAX`. Inconsistent
/// timestamps are rejected rather than repaired.
#[derive(Clone, Debug, Serialize)]
pub struct SynapticMesh {
    /// The wiring diagram.
    graph: SynapticGraph,
    /// Ring-buffer delay queue for spike delivery.
    delay_buffer: SpikeDelayBuffer,
    /// Current simulation tick.
    tick: u64,
}

#[derive(Deserialize)]
struct RawSynapticMesh {
    graph: SynapticGraph,
    delay_buffer: SpikeDelayBuffer,
    tick: u64,
}

impl RawSynapticMesh {
    fn into_mesh(self) -> std::result::Result<SynapticMesh, String> {
        if self.graph.neuron_count() != self.delay_buffer.neuron_count() {
            return Err(format!(
                "mesh graph neuron_count {} does not match delay buffer neuron_count {}",
                self.graph.neuron_count(),
                self.delay_buffer.neuron_count()
            ));
        }
        let graph_max = usize::from(self.graph.max_delay());
        if self.delay_buffer.max_delay() < graph_max {
            return Err(format!(
                "delay buffer max_delay {} is smaller than the graph's maximum delay {graph_max}",
                self.delay_buffer.max_delay()
            ));
        }
        let max_delay = self.delay_buffer.max_delay();
        validate_current_tick(self.tick, max_delay)?;
        validate_current_tick(self.delay_buffer.current_tick(), max_delay)?;
        if self.tick != self.delay_buffer.current_tick() {
            return Err(format!(
                "mesh tick {} does not match delay buffer current_tick {}",
                self.tick,
                self.delay_buffer.current_tick()
            ));
        }
        Ok(SynapticMesh {
            graph: self.graph,
            delay_buffer: self.delay_buffer,
            tick: self.tick,
        })
    }
}

impl<'de> Deserialize<'de> for SynapticMesh {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawSynapticMesh::deserialize(deserializer)?
            .into_mesh()
            .map_err(DeError::custom)
    }
}

impl SynapticMesh {
    /// Create a new mesh from a pre-built graph.
    ///
    /// The delay buffer is sized to the graph's maximum delay.
    pub fn new(graph: SynapticGraph) -> Self {
        let max_delay = usize::from(graph.max_delay());
        let n = graph.neuron_count();
        Self {
            delay_buffer: SpikeDelayBuffer::new(n, max_delay),
            graph,
            tick: 0,
        }
    }

    /// Create a mesh with a custom maximum delay (overriding graph's max).
    ///
    /// Useful when you want headroom for future dynamic delay changes.
    ///
    /// `max_delay` must be at least `graph.max_delay()`. A smaller buffer
    /// is rejected at construction — before any `propagate` call — rather
    /// than wrapping an oversized delay onto an earlier tick. Use
    /// [`SynapticMesh::new`] to size the buffer from the graph automatically,
    /// or [`SynapticMesh::try_with_max_delay`] for a recoverable error.
    ///
    /// # Panics
    ///
    /// Panics if `max_delay` is smaller than the graph's maximum delay, or
    /// if `max_delay + 1` overflows `usize`.
    pub fn with_max_delay(graph: SynapticGraph, max_delay: usize) -> Self {
        Self::try_with_max_delay(graph, max_delay).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SynapticMesh::with_max_delay`].
    ///
    /// Returns [`MeshError::DelayError`] when `max_delay` cannot hold every
    /// synapse delay in `graph`, or when the ring depth would overflow.
    pub fn try_with_max_delay(graph: SynapticGraph, max_delay: usize) -> Result<Self> {
        let graph_max = usize::from(graph.max_delay());
        if max_delay < graph_max {
            return Err(MeshError::DelayError(format!(
                "max_delay {max_delay} is smaller than the graph's maximum delay {graph_max}"
            )));
        }
        let n = graph.neuron_count();
        Ok(Self {
            delay_buffer: SpikeDelayBuffer::try_new(n, max_delay)?,
            graph,
            tick: 0,
        })
    }

    /// Propagate spikes through the mesh for one tick.
    ///
    /// # Arguments
    ///
    /// * `source_spikes` — boolean spike vector: `true` if neuron `i` fired
    ///   this tick. Length must equal `neuron_count()`.
    ///
    /// # Returns
    ///
    /// Synaptic current vector of length `neuron_count()` — the total
    /// incoming current at each neuron for this tick (including delayed
    /// spikes from previous ticks).
    ///
    /// # Delivery contract
    ///
    /// 1. A spike from neuron `s` on tick `t` delivers current to every
    ///    target of `s` on tick `t + delay`. A `delay` of 0 arrives in the
    ///    return value of this very call. This assumes the delay buffer can
    ///    hold the graph's delays, which [`SynapticMesh::new`] guarantees.
    ///    [`with_max_delay`] rejects a buffer smaller than the graph's
    ///    longest delay at construction (debug and release).
    /// 2. The magnitude is the synapse's weight and the sign is that
    ///    synapse's [`Polarity`](crate::types::Polarity), so an inhibitory
    ///    synapse subtracts. The generators assign polarity per source
    ///    neuron (Dale's law), but [`SynapticGraph::from_descriptors`] takes
    ///    each descriptor's polarity as given and does not check that one
    ///    source's synapses agree.
    ///
    /// [`with_max_delay`]: Self::with_max_delay
    /// [`SynapticGraph::from_descriptors`]: crate::topology::SynapticGraph::from_descriptors
    /// 3. Currents arriving at the same neuron on the same tick are summed.
    /// 4. Propagation is **one hop**: received current never becomes a spike
    ///    by itself. You own the neuron model — threshold the returned
    ///    currents and feed the result into the next call to build a
    ///    multi-layer loop.
    /// 5. Delivery is deterministic: the same graph and the same spike
    ///    sequence produce the same currents, including after [`reset`].
    ///
    /// [`reset`]: Self::reset
    ///
    /// ```rust
    /// use synaptic_wiring::mesh::SynapticMesh;
    /// use synaptic_wiring::topology::SynapticGraph;
    /// use synaptic_wiring::types::{Polarity, SynapseDescriptor};
    ///
    /// // 0 ──(w=0.75, delay 0, excitatory)──▶ 1
    /// // 0 ──(w=0.50, delay 2, excitatory)──▶ 2
    /// let graph = SynapticGraph::from_descriptors(
    ///     3,
    ///     &[
    ///         SynapseDescriptor { source: 0, target: 1, weight: 0.75, delay: 0, polarity: Polarity::Excitatory },
    ///         SynapseDescriptor { source: 0, target: 2, weight: 0.50, delay: 2, polarity: Polarity::Excitatory },
    ///     ],
    /// )?;
    /// let mut mesh = SynapticMesh::new(graph);
    ///
    /// // Tick 0: neuron 0 fires — only the delay-0 synapse lands.
    /// assert_eq!(mesh.propagate(&[true, false, false])?, vec![0.0, 0.75, 0.0]);
    /// // Tick 1: nothing in flight arrives yet.
    /// assert_eq!(mesh.propagate(&[false; 3])?, vec![0.0, 0.0, 0.0]);
    /// // Tick 2: the delay-2 synapse arrives at neuron 2.
    /// assert_eq!(mesh.propagate(&[false; 3])?, vec![0.0, 0.0, 0.50]);
    /// # Ok::<(), synaptic_wiring::MeshError>(())
    /// ```
    ///
    /// `tests/propagate_contract.rs` pins this contract — destination, sign,
    /// magnitude, and delivery tick — against a fixed four-neuron graph, and
    /// is a copyable starting point for your own simulation loop.
    pub fn propagate(&mut self, source_spikes: &[bool]) -> Result<Vec<f32>> {
        let n = self.graph.neuron_count();
        if source_spikes.len() != n {
            return Err(MeshError::NeuronCountMismatch {
                expected: n,
                got: source_spikes.len(),
                context: "propagate source_spikes".into(),
            });
        }

        // Inject spikes from all firing neurons into the delay buffer
        for (src, &fired) in source_spikes.iter().enumerate() {
            if !fired {
                continue;
            }
            for (target, weight, delay, _polarity) in self.graph.outgoing(src) {
                self.delay_buffer
                    .inject(target as usize, weight, delay as usize);
            }
        }

        // Drain currents that have arrived at this tick
        let currents = self.delay_buffer.drain_current_tick();

        // Advance the buffer
        self.delay_buffer.advance();
        self.tick += 1;

        Ok(currents)
    }

    /// Propagate with floating-point spike strengths instead of binary.
    ///
    /// # Arguments
    ///
    /// * `source_activations` — activation level for each neuron.
    ///   Non-zero values are treated as spikes; the activation value
    ///   scales the synaptic weight. Finite **signed** activations are
    ///   allowed (a negative activation inverts the delivered current).
    ///   NaN, ±infinity, non-finite `weight * activation` products, and
    ///   non-finite per-slot aggregates (including current already in the
    ///   delay buffer) are rejected before any buffer or tick mutation.
    pub fn propagate_graded(&mut self, source_activations: &[f32]) -> Result<Vec<f32>> {
        let n = self.graph.neuron_count();
        if source_activations.len() != n {
            return Err(MeshError::NeuronCountMismatch {
                expected: n,
                got: source_activations.len(),
                context: "propagate_graded source_activations".into(),
            });
        }

        // One traversal: validate each product, then group additions by
        // (target, delay) so aggregates can be checked before any inject.
        let mut pending: Vec<(usize, usize, f32)> = Vec::new();
        for (src, &activation) in source_activations.iter().enumerate() {
            if !activation.is_finite() {
                return Err(MeshError::InvalidConfig(format!(
                    "propagate_graded source_activations[{src}] must be finite, got {activation}"
                )));
            }
            if activation.abs() < 1e-9 {
                continue;
            }
            for (target, weight, delay, _) in self.graph.outgoing(src) {
                let current = weight * activation;
                if !current.is_finite() {
                    return Err(MeshError::InvalidConfig(format!(
                        "propagate_graded source_activations[{src}] * synapse weight must be finite, got {current}"
                    )));
                }
                accumulate_slot_current(&mut pending, target as usize, delay as usize, current);
            }
        }

        for &(target, delay, additional) in &pending {
            let total = self.delay_buffer.scheduled_current(target, delay) + additional;
            if !additional.is_finite() || !total.is_finite() {
                return Err(MeshError::InvalidConfig(format!(
                    "propagate_graded would produce a non-finite delay-buffer total for target {target}"
                )));
            }
        }

        for (target, delay, additional) in pending {
            self.delay_buffer.inject(target, additional, delay);
        }

        let currents = self.delay_buffer.drain_current_tick();
        self.delay_buffer.advance();
        self.tick += 1;

        Ok(currents)
    }

    /// Current simulation tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Number of neurons in the mesh.
    pub fn neuron_count(&self) -> usize {
        self.graph.neuron_count()
    }

    /// Total number of synapses.
    pub fn synapse_count(&self) -> usize {
        self.graph.synapse_count()
    }

    /// Sparsity of the connectivity matrix.
    pub fn sparsity(&self) -> f32 {
        self.graph.sparsity()
    }

    /// Maximum delay in ticks.
    pub fn max_delay(&self) -> usize {
        self.delay_buffer.max_delay()
    }

    /// Mean out-degree of the network.
    pub fn mean_degree(&self) -> f32 {
        self.graph.mean_degree()
    }

    /// Immutable access to the underlying graph.
    pub fn graph(&self) -> &SynapticGraph {
        &self.graph
    }

    /// Deterministic digest of the logical topology (not tick or delay-buffer state).
    ///
    /// Equivalent to [`SynapticGraph::topology_digest`] on [`Self::graph`].
    #[must_use]
    pub fn topology_digest(&self) -> TopologyDigest {
        self.graph.topology_digest()
    }

    /// Reset delay buffer and tick counter.
    pub fn reset(&mut self) {
        self.delay_buffer.reset();
        self.tick = 0;
    }

    /// Export the CSR arrays + delays for GPU upload.
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u16>) {
        self.graph.to_gpu_arrays()
    }
}

fn accumulate_slot_current(
    pending: &mut Vec<(usize, usize, f32)>,
    target: usize,
    delay: usize,
    current: f32,
) {
    if let Some((_, _, total)) = pending
        .iter_mut()
        .find(|(t, d, _)| *t == target && *d == delay)
    {
        *total += current;
    } else {
        pending.push((target, delay, current));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{generate_layered, generate_random, generate_small_world};

    #[test]
    fn topology_digest_matches_graph_and_ignores_tick() {
        let graph = two_neuron_delay_graph(1);
        let mut mesh = SynapticMesh::new(graph.clone());
        let before = mesh.topology_digest();
        assert_eq!(before, graph.topology_digest());
        let _ = mesh.propagate(&[true, false]).unwrap();
        assert_eq!(
            mesh.topology_digest(),
            before,
            "tick / delay-buffer state must not enter the topology digest"
        );
    }

    fn two_neuron_delay_graph(delay: u16) -> SynapticGraph {
        use crate::types::{Polarity, SynapseDescriptor};
        SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 1.0,
                delay,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap()
    }

    fn mesh_json_with_override(
        mesh: &SynapticMesh,
        patch: impl FnOnce(&mut serde_json::Value),
    ) -> String {
        let mut value = serde_json::to_value(mesh).unwrap();
        patch(&mut value);
        serde_json::to_string(&value).unwrap()
    }

    #[test]
    fn deserialize_rejects_graph_buffer_neuron_count_mismatch() {
        let mesh = SynapticMesh::new(two_neuron_delay_graph(0));
        let json = mesh_json_with_override(&mesh, |v| {
            v["delay_buffer"] = serde_json::to_value(SpikeDelayBuffer::new(3, 0)).unwrap();
        });
        assert!(serde_json::from_str::<SynapticMesh>(&json).is_err());
    }

    #[test]
    fn deserialize_rejects_inadequate_buffer_capacity() {
        let mesh = SynapticMesh::new(two_neuron_delay_graph(2));
        let json = mesh_json_with_override(&mesh, |v| {
            v["delay_buffer"] = serde_json::to_value(SpikeDelayBuffer::new(2, 1)).unwrap();
        });
        assert!(serde_json::from_str::<SynapticMesh>(&json).is_err());
    }

    #[test]
    fn deserialize_rejects_tick_mismatch() {
        let mesh = SynapticMesh::new(two_neuron_delay_graph(0));
        let json = mesh_json_with_override(&mesh, |v| {
            v["tick"] = serde_json::json!(3);
        });
        assert!(serde_json::from_str::<SynapticMesh>(&json).is_err());
    }

    #[test]
    fn deserialize_rejects_tick_that_advances_to_usize_max() {
        let tick = usize::MAX as u64 - 1;
        for delay in [0_u16, 1] {
            let mesh = SynapticMesh::new(two_neuron_delay_graph(delay));
            let json = mesh_json_with_override(&mesh, |v| {
                v["tick"] = serde_json::json!(tick);
                v["delay_buffer"]["current_tick"] = serde_json::json!(tick);
            });
            assert!(
                serde_json::from_str::<SynapticMesh>(&json).is_err(),
                "synchronized tick usize::MAX - 1 with max_delay {delay} must be rejected"
            );
        }
    }

    #[test]
    fn checkpoint_roundtrip_zero_delay_and_reset() {
        let mut mesh = SynapticMesh::new(two_neuron_delay_graph(0));
        let empty: SynapticMesh =
            serde_json::from_str(&serde_json::to_string(&mesh).unwrap()).unwrap();
        assert_eq!(empty.tick(), 0);

        let _ = mesh.propagate(&[true, false]).unwrap();
        mesh.reset();
        let restored: SynapticMesh =
            serde_json::from_str(&serde_json::to_string(&mesh).unwrap()).unwrap();
        assert_eq!(restored.tick(), 0);
        assert_eq!(
            mesh.propagate(&[true, false]).unwrap(),
            restored.clone().propagate(&[true, false]).unwrap()
        );
    }

    #[test]
    fn checkpoint_resume_with_spikes_in_flight_matches_live() {
        let mut live = SynapticMesh::new(two_neuron_delay_graph(2));
        // Inject an in-flight spike, then snapshot before it lands.
        assert_eq!(live.propagate(&[true, false]).unwrap()[1], 0.0);
        let json = serde_json::to_string(&live).unwrap();
        let mut restored: SynapticMesh = serde_json::from_str(&json).unwrap();

        // Continue both copies through more than one ring wrap (depth = 3).
        let inputs = [
            [false, false],
            [true, false],
            [false, false],
            [false, false],
            [true, false],
            [false, false],
            [false, false],
            [false, false],
        ];
        for spikes in inputs {
            let a = live.propagate(&spikes).unwrap();
            let b = restored.propagate(&spikes).unwrap();
            assert_eq!(a, b);
            assert_eq!(live.tick(), restored.tick());
        }
    }

    #[test]
    fn try_with_max_delay_rejects_insufficient_capacity() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 1.0,
                delay: 2,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap();
        let err = SynapticMesh::try_with_max_delay(graph, 1).unwrap_err();
        assert!(
            err.to_string().contains("smaller than the graph"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[should_panic(expected = "smaller than the graph")]
    fn with_max_delay_rejects_insufficient_capacity() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 1.0,
                delay: 2,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap();
        let _ = SynapticMesh::with_max_delay(graph, 1);
    }

    #[test]
    fn with_max_delay_accepts_equal_and_larger_capacity() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 1.0,
                delay: 2,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap();
        let equal = SynapticMesh::try_with_max_delay(graph.clone(), 2).unwrap();
        assert_eq!(equal.max_delay(), 2);
        let larger = SynapticMesh::with_max_delay(graph, 5);
        assert_eq!(larger.max_delay(), 5);
    }

    #[test]
    fn propagate_length_mismatch_rejected() {
        let graph = generate_random(10, 0.5, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        let bad_spikes = vec![false; 5]; // wrong length
        assert!(mesh.propagate(&bad_spikes).is_err());
    }

    #[test]
    fn no_spikes_no_current() {
        let graph = generate_random(10, 0.5, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        let spikes = vec![false; 10];
        let currents = mesh.propagate(&spikes).unwrap();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn instant_delivery_with_zero_delay() {
        // Build a 2-neuron graph with 0-delay connection: 0 → 1
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.8,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        let mut spikes = vec![false; 2];
        spikes[0] = true;
        let currents = mesh.propagate(&spikes).unwrap();
        assert!((currents[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn delayed_delivery_through_mesh() {
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 1.0,
            delay: 2,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Tick 0: neuron 0 fires
        let mut spikes = vec![false; 2];
        spikes[0] = true;
        let c0 = mesh.propagate(&spikes).unwrap();
        assert_eq!(c0[1], 0.0); // not arrived yet

        // Tick 1: no fires
        spikes[0] = false;
        let c1 = mesh.propagate(&spikes).unwrap();
        assert_eq!(c1[1], 0.0); // not arrived yet

        // Tick 2: spike arrives
        let c2 = mesh.propagate(&spikes).unwrap();
        assert!((c2[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn propagate_graded_rejects_non_finite_before_mutation() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 1.0,
                delay: 2,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Put a spike in flight so we can detect a partial tick advance.
        mesh.propagate(&[true, false]).unwrap();
        assert_eq!(mesh.tick(), 1);

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = mesh.propagate_graded(&[0.0, bad]).unwrap_err();
            assert!(
                err.to_string().contains("must be finite"),
                "unexpected error for {bad}: {err}"
            );
            assert_eq!(mesh.tick(), 1, "tick must not advance on rejection");
        }

        // In-flight delay-2 current must still arrive two ticks after inject
        // (tick 2 of the mesh), proving the failed graded calls did not drain.
        assert_eq!(mesh.propagate(&[false, false]).unwrap()[1], 0.0);
        let arrived = mesh.propagate(&[false, false]).unwrap();
        assert!((arrived[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn propagate_graded_prevalidates_later_elements() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            3,
            &[
                SynapseDescriptor {
                    source: 0,
                    target: 1,
                    weight: 1.0,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
                SynapseDescriptor {
                    source: 2,
                    target: 1,
                    weight: 1.0,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
            ],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);
        // A valid first element must not be injected if a later one is NaN.
        assert!(mesh.propagate_graded(&[1.0, 0.0, f32::NAN]).is_err());
        assert_eq!(mesh.tick(), 0);
        let currents = mesh.propagate(&[false, false, false]).unwrap();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn propagate_graded_rejects_non_finite_weight_activation_product() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            3,
            &[
                SynapseDescriptor {
                    source: 0,
                    target: 1,
                    weight: 1.0,
                    delay: 2,
                    polarity: Polarity::Excitatory,
                },
                SynapseDescriptor {
                    source: 2,
                    target: 1,
                    weight: f32::MAX,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
            ],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);

        mesh.propagate(&[true, false, false]).unwrap();
        assert_eq!(mesh.tick(), 1);

        let err = mesh.propagate_graded(&[0.0, 0.0, f32::MAX]).unwrap_err();
        assert!(
            err.to_string().contains("must be finite"),
            "unexpected error: {err}"
        );
        assert_eq!(
            mesh.tick(),
            1,
            "tick must not advance on overflow rejection"
        );

        // A later overflowing product must not inject the earlier finite synapse.
        assert!(mesh.propagate_graded(&[1.0, 0.0, f32::MAX]).is_err());
        assert_eq!(mesh.tick(), 1);

        assert_eq!(mesh.propagate(&[false, false, false]).unwrap()[1], 0.0);
        let arrived = mesh.propagate(&[false, false, false]).unwrap();
        assert!(
            (arrived[1] - 1.0).abs() < 1e-6,
            "in-flight current must be unchanged, got {}",
            arrived[1]
        );
    }

    #[test]
    fn propagate_graded_rejects_aggregate_overflow_before_mutation() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            3,
            &[
                SynapseDescriptor {
                    source: 0,
                    target: 1,
                    weight: f32::MAX,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
                SynapseDescriptor {
                    source: 2,
                    target: 1,
                    weight: f32::MAX,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
            ],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);

        let err = mesh.propagate_graded(&[1.0, 0.0, 1.0]).unwrap_err();
        assert!(
            err.to_string().contains("non-finite delay-buffer total"),
            "unexpected error: {err}"
        );
        assert_eq!(mesh.tick(), 0);
        let currents = mesh.propagate(&[false, false, false]).unwrap();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn propagate_graded_rejects_overflow_with_existing_buffer_current() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[
                SynapseDescriptor {
                    source: 0,
                    target: 1,
                    weight: f32::MAX,
                    delay: 0,
                    polarity: Polarity::Excitatory,
                },
                SynapseDescriptor {
                    source: 0,
                    target: 1,
                    weight: f32::MAX,
                    delay: 1,
                    polarity: Polarity::Excitatory,
                },
            ],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Parks f32::MAX in the delay-1 slot; delay-0 is drained this tick.
        let first = mesh.propagate_graded(&[1.0, 0.0]).unwrap();
        assert_eq!(first[1], f32::MAX);
        assert_eq!(mesh.tick(), 1);

        let err = mesh.propagate_graded(&[1.0, 0.0]).unwrap_err();
        assert!(
            err.to_string().contains("non-finite delay-buffer total"),
            "unexpected error: {err}"
        );
        assert_eq!(
            mesh.tick(),
            1,
            "tick must not advance on aggregate rejection"
        );

        let arrived = mesh.propagate(&[false, false]).unwrap();
        assert_eq!(
            arrived[1],
            f32::MAX,
            "parked delay-1 current must be unchanged"
        );
    }

    #[test]
    fn propagate_graded_preserves_signed_finite_activations() {
        use crate::types::{Polarity, SynapseDescriptor};
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.5,
                delay: 0,
                polarity: Polarity::Inhibitory,
            }],
        )
        .unwrap();
        let mut mesh = SynapticMesh::new(graph);
        // Inhibitory weight is -0.5; a negative activation inverts it to +0.25.
        let currents = mesh.propagate_graded(&[-0.5, 0.0]).unwrap();
        assert!((currents[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn graded_propagation() {
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        let activations = vec![0.6, 0.0];
        let currents = mesh.propagate_graded(&activations).unwrap();
        // 0.5 * 0.6 = 0.3
        assert!((currents[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tick_counter_increments() {
        let graph = generate_random(8, 0.3, 2, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        assert_eq!(mesh.tick(), 0);
        mesh.propagate(&[false; 8]).unwrap();
        mesh.propagate(&[false; 8]).unwrap();
        assert_eq!(mesh.tick(), 2);
    }

    #[test]
    fn reset_clears_state() {
        let graph = generate_random(8, 0.3, 2, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        mesh.propagate(&[true; 8]).unwrap();
        mesh.propagate(&[true; 8]).unwrap();
        mesh.reset();
        assert_eq!(mesh.tick(), 0);
    }

    #[test]
    fn small_world_mesh_propagation() {
        let graph = generate_small_world(32, 4, 0.2, 5, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Fire a single neuron and run for several ticks
        let mut any_current = false;
        for tick in 0..10 {
            let mut spikes = vec![false; 32];
            if tick == 0 {
                spikes[0] = true;
            }
            let currents = mesh.propagate(&spikes).unwrap();
            if currents.iter().any(|&c| c.abs() > 1e-6) {
                any_current = true;
            }
        }
        assert!(any_current, "expected some delayed current delivery");
    }

    #[test]
    fn layered_mesh_feed_forward() {
        // generate_layered(&[4, 8, 2], inter_layer_p, max_delay, inh_fraction)
        // Neuron layout: [0..4) input, [4..12) hidden, [12..14) output
        //
        // NOTE: This test depends on the deterministic hash-based weight
        // generation. The threshold (0.5) and inhibitory fraction (0.2) are
        // chosen so that excitatory contributions dominate for enough hidden
        // neurons to activate the output layer. If the hash function or weight
        // parameters change, the threshold or layer sizes may need adjustment.
        let graph = generate_layered(&[4, 8, 2], 1.0, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        assert_eq!(mesh.neuron_count(), 14);
        let mut spikes = vec![false; 14];
        for spike in spikes.iter_mut().take(4) {
            *spike = true; // fire input layer
        }

        // Run enough ticks for delays to propagate through both hops
        // (max_delay=3 per hop → need at least 6 ticks; run 12 for margin).
        let mut last_layer_activated = false;
        for _ in 0..12 {
            let currents = mesh.propagate(&spikes).unwrap();

            // Threshold-and-fire: neurons that received enough current fire
            // next tick. This mirrors how a downstream consumer would use
            // `propagate()` to build a multi-layer simulation loop.
            spikes = currents.iter().map(|&c| c > 0.5).collect();

            // Check if last-layer neurons (12, 13) received current
            if currents[12].abs() > 1e-6 || currents[13].abs() > 1e-6 {
                last_layer_activated = true;
            }
        }
        assert!(
            last_layer_activated,
            "feed-forward should eventually reach the output layer"
        );
    }
}
