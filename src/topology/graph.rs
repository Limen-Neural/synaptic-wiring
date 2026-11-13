// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`SynapticGraph`] — adjacency structure with delay and polarity metadata.
//!
//! Extends the CSR sparse representation from `sparse.rs` with per-synapse
//! axonal delay and polarity information. This is the core "wiring diagram"
//! that the `SynapticMesh` orchestrator uses for spike propagation.

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};
use crate::types::{
    DelayTicks, NeuronId, Polarity, SynapseDescriptor, invalid_signed_weight_polarity_msg,
    invalid_weight_magnitude_msg, signed_weight_agrees_with_polarity, weight_magnitude_is_valid,
};

/// Adjacency structure for a spiking neural network with delay and polarity metadata.
///
/// Stores connections in Compressed Sparse Row (CSR) format, augmented with
/// per-synapse delay and polarity vectors aligned to the same index space.
///
/// # Layout
///
/// ```text
/// row_ptr:    [0,  3,  7, 10, ...]     — start index per source neuron
/// targets:    [5, 12,  0,  1,  3, ...]  — target neuron IDs
/// weights:    [0.9, -0.15, 0.3, ...]    — effective signed weights
/// delays:     [2, 5, 1, ...]            — axonal delay in ticks
/// polarities: [Exc, Inh, Exc, ...]      — Dale's law polarity
/// ```
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SynapticGraph {
    /// Number of neurons in the graph.
    neuron_count: usize,
    /// Row pointers: `row_ptr[i]` gives the start index in the edge arrays
    /// for neuron `i`. Length is `neuron_count + 1`.
    row_ptr: Vec<usize>,
    /// Target neuron IDs for each synapse.
    targets: Vec<NeuronId>,
    /// Synaptic weights (signed: polarity × |weight|).
    weights: Vec<f32>,
    /// Axonal propagation delay per synapse.
    delays: Vec<DelayTicks>,
    /// Polarity per synapse.
    polarities: Vec<Polarity>,
}

#[derive(Deserialize)]
struct RawSynapticGraph {
    neuron_count: usize,
    row_ptr: Vec<usize>,
    targets: Vec<NeuronId>,
    weights: Vec<f32>,
    delays: Vec<DelayTicks>,
    polarities: Vec<Polarity>,
}

impl RawSynapticGraph {
    /// Re-validates the CSR invariants that [`SynapticGraph::from_descriptors`]
    /// enforces at construction time, since a derived `Deserialize` would
    /// accept arbitrary field values (mismatched `row_ptr` length,
    /// non-monotonic offsets, out-of-range targets) that later panic in
    /// [`SynapticGraph::outgoing`] or [`SynapticGraph::out_degree`].
    fn into_graph(self) -> std::result::Result<SynapticGraph, String> {
        let nnz = validate_row_ptr(&self.row_ptr, self.neuron_count)?;
        validate_edge_array_lengths(
            nnz,
            [
                ("targets", self.targets.len()),
                ("weights", self.weights.len()),
                ("delays", self.delays.len()),
                ("polarities", self.polarities.len()),
            ],
        )?;
        validate_targets_in_bounds(&self.targets, self.neuron_count)?;
        validate_weights_finite(&self.weights)?;
        validate_weights_agree_with_polarities(&self.weights, &self.polarities)?;

        Ok(SynapticGraph {
            neuron_count: self.neuron_count,
            row_ptr: self.row_ptr,
            targets: self.targets,
            weights: self.weights,
            delays: self.delays,
            polarities: self.polarities,
        })
    }
}

/// Checks `row_ptr` has `neuron_count + 1` entries starting at 0 and
/// non-decreasing, returning the total edge count (its final entry).
fn validate_row_ptr(row_ptr: &[usize], neuron_count: usize) -> std::result::Result<usize, String> {
    let expected_len = neuron_count
        .checked_add(1)
        .ok_or_else(|| format!("neuron_count {neuron_count} is too large"))?;
    if row_ptr.len() != expected_len {
        return Err(format!(
            "row_ptr length {} does not match neuron_count + 1 ({})",
            row_ptr.len(),
            expected_len
        ));
    }
    if row_ptr.first().copied() != Some(0) {
        return Err("row_ptr must start at 0".to_string());
    }
    if !row_ptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err("row_ptr must be non-decreasing".to_string());
    }
    Ok(*row_ptr.last().expect("row_ptr is non-empty"))
}

/// Checks that every named edge array has exactly `nnz` entries.
fn validate_edge_array_lengths(
    nnz: usize,
    arrays: [(&str, usize); 4],
) -> std::result::Result<(), String> {
    for (name, len) in arrays {
        if len != nnz {
            return Err(format!(
                "{name} length {len} does not match row_ptr's final offset {nnz}"
            ));
        }
    }
    Ok(())
}

/// Checks that every target neuron id is within `[0, neuron_count)`.
fn validate_targets_in_bounds(
    targets: &[NeuronId],
    neuron_count: usize,
) -> std::result::Result<(), String> {
    if targets.iter().any(|&t| t as usize >= neuron_count) {
        return Err("target neuron id out of bounds".to_string());
    }
    Ok(())
}

/// Checks that every weight is finite, so a non-JSON format (which, unlike
/// JSON, can represent NaN/Inf) can't smuggle in a value that poisons
/// current sums in [`crate::mesh::SynapticMesh::propagate`].
fn validate_weights_finite(weights: &[f32]) -> std::result::Result<(), String> {
    if weights.iter().any(|w| !w.is_finite()) {
        return Err("synapse weight must be finite".to_string());
    }
    Ok(())
}

/// Signed CSR weight must agree with the stored polarity. Zero, including
/// IEEE signed zero, is accepted for both polarities.
fn validate_weights_agree_with_polarities(
    weights: &[f32],
    polarities: &[Polarity],
) -> std::result::Result<(), String> {
    for (&weight, &polarity) in weights.iter().zip(polarities.iter()) {
        if !signed_weight_agrees_with_polarity(weight, polarity) {
            return Err(invalid_signed_weight_polarity_msg(weight, polarity));
        }
    }
    Ok(())
}

impl<'de> Deserialize<'de> for SynapticGraph {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawSynapticGraph::deserialize(deserializer)?
            .into_graph()
            .map_err(DeError::custom)
    }
}

impl SynapticGraph {
    /// Create an empty graph with `n` neurons and no connections.
    pub fn new(n: usize) -> Self {
        Self {
            neuron_count: n,
            row_ptr: vec![0; n + 1],
            targets: Vec::new(),
            weights: Vec::new(),
            delays: Vec::new(),
            polarities: Vec::new(),
        }
    }

    /// Build from a list of synapse descriptors.
    ///
    /// Descriptors need not be sorted; they will be grouped by source neuron.
    /// Each descriptor `weight` must be a finite non-negative magnitude
    /// (IEEE signed zero is accepted). Negative, NaN, and infinite
    /// magnitudes are rejected — they are not `abs()`-normalized.
    pub fn from_descriptors(
        neuron_count: usize,
        descriptors: &[SynapseDescriptor],
    ) -> Result<Self> {
        Self::validate_descriptor_indices(neuron_count, descriptors)?;
        Self::validate_descriptor_weights(descriptors)?;

        // Count edges per source neuron
        let mut counts = vec![0usize; neuron_count];
        for desc in descriptors {
            counts[desc.source as usize] += 1;
        }

        // Build row_ptr
        let mut row_ptr = Vec::with_capacity(neuron_count + 1);
        row_ptr.push(0);
        for &c in &counts {
            row_ptr.push(row_ptr.last().unwrap() + c);
        }

        let nnz = descriptors.len();
        let mut targets = vec![0u32; nnz];
        let mut weights = vec![0.0f32; nnz];
        let mut delays = vec![0u16; nnz];
        let mut polarities = vec![Polarity::Excitatory; nnz];

        // Place each descriptor at the right position
        let mut cursor = counts.clone();
        cursor.fill(0);

        for desc in descriptors {
            let src = desc.source as usize;
            let pos = row_ptr[src] + cursor[src];
            targets[pos] = desc.target;
            weights[pos] = desc.effective_weight();
            delays[pos] = desc.delay;
            polarities[pos] = desc.polarity;
            cursor[src] += 1;
        }

        Ok(Self {
            neuron_count,
            row_ptr,
            targets,
            weights,
            delays,
            polarities,
        })
    }

    /// Validate that all descriptor source/target indices are within bounds.
    fn validate_descriptor_indices(
        neuron_count: usize,
        descriptors: &[SynapseDescriptor],
    ) -> Result<()> {
        if neuron_count == 0 && !descriptors.is_empty() {
            return Err(MeshError::InvalidConfig(
                "descriptors provided for a graph with 0 neurons".into(),
            ));
        }
        for desc in descriptors {
            if desc.source as usize >= neuron_count {
                return Err(MeshError::IndexOutOfBounds {
                    index: desc.source as usize,
                    max: neuron_count.saturating_sub(1),
                });
            }
            if desc.target as usize >= neuron_count {
                return Err(MeshError::IndexOutOfBounds {
                    index: desc.target as usize,
                    max: neuron_count.saturating_sub(1),
                });
            }
        }
        Ok(())
    }

    /// Reject negative, NaN, and infinite descriptor magnitudes. IEEE signed
    /// zero (`-0.0`) is a valid non-negative magnitude.
    fn validate_descriptor_weights(descriptors: &[SynapseDescriptor]) -> Result<()> {
        for desc in descriptors {
            if !weight_magnitude_is_valid(desc.weight) {
                return Err(MeshError::InvalidConfig(invalid_weight_magnitude_msg(
                    desc.weight,
                )));
            }
        }
        Ok(())
    }

    /// Number of neurons in the graph, including neurons with no synapses.
    ///
    /// Does not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use synaptic_wiring::SynapticGraph;
    ///
    /// assert_eq!(SynapticGraph::new(3).neuron_count(), 3);
    /// assert_eq!(SynapticGraph::new(0).neuron_count(), 0);
    /// ```
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Total number of synapses (non-zero entries).
    ///
    /// Parallel synapses between the same pair and self-connections each
    /// count once. Does not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use synaptic_wiring::{Polarity, SynapseDescriptor, SynapticGraph};
    ///
    /// let syn = |source, target| SynapseDescriptor {
    ///     source,
    ///     target,
    ///     weight: 0.5,
    ///     delay: 1,
    ///     polarity: Polarity::Excitatory,
    /// };
    /// let graph = SynapticGraph::from_descriptors(3, &[syn(0, 1), syn(1, 2), syn(2, 2)])?;
    /// assert_eq!(graph.synapse_count(), 3);
    /// assert_eq!(SynapticGraph::new(3).synapse_count(), 0);
    /// # Ok::<(), synaptic_wiring::MeshError>(())
    /// ```
    pub fn synapse_count(&self) -> usize {
        self.targets.len()
    }

    /// Sparsity: fraction of the N×N matrix that is zero.
    pub fn sparsity(&self) -> f32 {
        let total = self.neuron_count * self.neuron_count;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.synapse_count() as f32 / total as f32)
    }

    /// Iterate over all outgoing synapses from `source`.
    /// Returns `(target, weight, delay, polarity)` tuples, where `weight` is
    /// the signed CSR weight.
    ///
    /// # Panics
    ///
    /// Panics if `source >= neuron_count()`. Use
    /// [`outgoing_synapses`](Self::outgoing_synapses) for a non-panicking
    /// iterator over [`SynapseDescriptor`]s.
    pub fn outgoing(
        &self,
        source: usize,
    ) -> impl Iterator<Item = (NeuronId, f32, DelayTicks, Polarity)> + '_ {
        let start = self.row_ptr[source];
        let end = self.row_ptr[source + 1];
        (start..end).map(move |i| {
            (
                self.targets[i],
                self.weights[i],
                self.delays[i],
                self.polarities[i],
            )
        })
    }

    /// Iterate over the outgoing synapses of `neuron_id` as [`SynapseDescriptor`]s.
    ///
    /// Each descriptor is built on the fly from the CSR edge arrays, so this
    /// does not allocate. Descriptors carry the magnitude of the stored signed
    /// weight (`weight >= 0`), so [`SynapseDescriptor::effective_weight`] gives
    /// back the signed weight bit-for-bit (including signed zero) and the
    /// descriptors can be fed straight into
    /// [`from_descriptors`](Self::from_descriptors). Synapses are yielded in
    /// CSR order, which for a graph built by `from_descriptors` is the order
    /// the descriptors for that source were given in.
    ///
    /// A neuron with no outgoing synapses and an out-of-range `neuron_id`
    /// (`neuron_id >= neuron_count()`) both yield an empty iterator; this
    /// method never panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use synaptic_wiring::{Polarity, SynapseDescriptor, SynapticGraph};
    ///
    /// let exc = SynapseDescriptor {
    ///     source: 0,
    ///     target: 1,
    ///     weight: 0.5,
    ///     delay: 2,
    ///     polarity: Polarity::Excitatory,
    /// };
    /// let inh = SynapseDescriptor {
    ///     target: 2,
    ///     weight: 0.25,
    ///     polarity: Polarity::Inhibitory,
    ///     ..exc
    /// };
    /// let graph = SynapticGraph::from_descriptors(3, &[exc, inh])?;
    ///
    /// for syn in graph.outgoing_synapses(0) {
    ///     println!("{} -> {} after {} ticks", syn.source, syn.target, syn.delay);
    /// }
    /// assert!(graph.outgoing_synapses(0).eq([exc, inh]));
    /// assert_eq!(graph.outgoing_synapses(2).count(), 0); // no outgoing synapses
    /// assert_eq!(graph.outgoing_synapses(3).count(), 0); // out of range
    /// # Ok::<(), synaptic_wiring::MeshError>(())
    /// ```
    pub fn outgoing_synapses(
        &self,
        neuron_id: usize,
    ) -> impl Iterator<Item = SynapseDescriptor> + '_ {
        // A descriptor can only name a `NeuronId` source; an id that does not
        // fit one is treated as out of range.
        let (source, edges) = match (NeuronId::try_from(neuron_id), self.row_ptr.get(neuron_id..)) {
            (Ok(source), Some(&[start, end, ..])) => (source, start..end),
            _ => (0, 0..0),
        };
        edges.map(move |i| SynapseDescriptor {
            source,
            target: self.targets[i],
            // Not `abs()`: multiplying by the polarity sign keeps signed zero
            // round-tripping through `effective_weight()` (and the digest).
            weight: self.weights[i] * self.polarities[i].sign(),
            delay: self.delays[i],
            polarity: self.polarities[i],
        })
    }

    /// Get the out-degree of a neuron.
    pub fn out_degree(&self, neuron: usize) -> usize {
        self.row_ptr[neuron + 1] - self.row_ptr[neuron]
    }

    /// Maximum delay present in the graph.
    pub fn max_delay(&self) -> DelayTicks {
        self.delays.iter().copied().max().unwrap_or(0)
    }

    /// Mean out-degree across all neurons.
    #[deprecated(
        since = "0.4.0",
        note = "use `SynapticGraph::mean_out_degree`, which returns `f64`"
    )]
    pub fn mean_degree(&self) -> f32 {
        if self.neuron_count == 0 {
            return 0.0;
        }
        self.synapse_count() as f32 / self.neuron_count as f32
    }

    /// Raw access to CSR arrays for GPU upload.
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u16>) {
        let row_ptr: Vec<u32> = self.row_ptr.iter().map(|&x| x as u32).collect();
        let targets: Vec<u32> = self.targets.to_vec();
        (row_ptr, targets, self.weights.clone(), self.delays.clone())
    }

    /// Row pointer slice (for external iteration).
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Target neuron slice.
    pub fn targets(&self) -> &[NeuronId] {
        &self.targets
    }

    /// Weight slice.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Delay slice.
    pub fn delays_slice(&self) -> &[DelayTicks] {
        &self.delays
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_no_synapses() {
        let g = SynapticGraph::new(100);
        assert_eq!(g.neuron_count(), 100);
        assert_eq!(g.synapse_count(), 0);
        assert!((g.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_descriptors_roundtrip() {
        let descs = vec![
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.9,
                delay: 3,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 0,
                target: 2,
                weight: 0.15,
                delay: 1,
                polarity: Polarity::Inhibitory,
            },
            SynapseDescriptor {
                source: 1,
                target: 0,
                weight: 0.5,
                delay: 2,
                polarity: Polarity::Excitatory,
            },
        ];

        let graph = SynapticGraph::from_descriptors(3, &descs).unwrap();
        assert_eq!(graph.neuron_count(), 3);
        assert_eq!(graph.synapse_count(), 3);

        // Neuron 0 has 2 outgoing
        assert_eq!(graph.out_degree(0), 2);
        // Neuron 1 has 1 outgoing
        assert_eq!(graph.out_degree(1), 1);
        // Neuron 2 has 0 outgoing
        assert_eq!(graph.out_degree(2), 0);

        // Check inhibitory weight is negative
        let edges: Vec<_> = graph.outgoing(0).collect();
        let inh_edge = edges
            .iter()
            .find(|(_, _, _, p)| *p == Polarity::Inhibitory)
            .unwrap();
        assert!(inh_edge.1 < 0.0);
    }

    /// Outgoing synapses are yielded as descriptors in CSR order, with the
    /// descriptor's source set to the queried neuron.
    #[test]
    fn outgoing_synapses_yields_source_descriptors_in_order() {
        let descs = vec![
            SynapseDescriptor {
                source: 1,
                target: 0,
                weight: 0.5,
                delay: 2,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 0,
                target: 2,
                weight: 0.15,
                delay: 1,
                polarity: Polarity::Inhibitory,
            },
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.9,
                delay: 3,
                polarity: Polarity::Inhibitory,
            },
        ];
        let graph = SynapticGraph::from_descriptors(3, &descs).unwrap();

        let from_0: Vec<_> = graph.outgoing_synapses(0).collect();
        assert_eq!(from_0, [descs[1], descs[2]]);
        let from_1: Vec<_> = graph.outgoing_synapses(1).collect();
        assert_eq!(from_1, [descs[0]]);
        for syn in &from_0 {
            assert!(syn.effective_weight() < 0.0);
        }
    }

    /// Unconnected neurons, out-of-range ids and `usize::MAX` all yield an
    /// empty iterator without panicking.
    #[test]
    fn outgoing_synapses_empty_for_unconnected_and_out_of_range() {
        let graph = SynapticGraph::from_descriptors(
            2,
            &[SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.5,
                delay: 1,
                polarity: Polarity::Excitatory,
            }],
        )
        .unwrap();
        let empty = SynapticGraph::new(0);
        for (g, id) in [(&graph, 1), (&graph, 2), (&graph, usize::MAX), (&empty, 0)] {
            assert_eq!(g.outgoing_synapses(id).count(), 0, "neuron {id}");
        }
    }

    /// Descriptors rebuilt via `from_descriptors` produce an equal graph with
    /// the same topology digest, preserving signed-zero weights.
    #[test]
    fn outgoing_synapses_rebuild_an_equal_graph() {
        // Deserialized graphs may store +0.0 for an inhibitory synapse or
        // -0.0 for an excitatory one; the descriptor magnitude must still be
        // valid and the signed zero must survive the rebuild (it is digested).
        let json = r#"{"neuron_count":3,"row_ptr":[0,2,2,4],"targets":[1,2,0,1],"weights":[-0.4,0.0,0.7,-0.0],"delays":[1,4,2,3],"polarities":["Inhibitory","Inhibitory","Excitatory","Excitatory"]}"#;
        let graph: SynapticGraph = serde_json::from_str(json).unwrap();
        let descs: Vec<_> = (0..graph.neuron_count())
            .flat_map(|n| graph.outgoing_synapses(n))
            .collect();
        assert_eq!(descs.len(), graph.synapse_count());
        let rebuilt = SynapticGraph::from_descriptors(graph.neuron_count(), &descs).unwrap();
        assert_eq!(rebuilt, graph);
        assert_eq!(rebuilt.topology_digest(), graph.topology_digest());
    }

    #[test]
    fn synaptic_graph_json_roundtrip() {
        let descs = vec![
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.9,
                delay: 3,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 0,
                target: 2,
                weight: 0.15,
                delay: 1,
                polarity: Polarity::Inhibitory,
            },
            SynapseDescriptor {
                source: 2,
                target: 1,
                weight: 0.4,
                delay: 5,
                polarity: Polarity::Excitatory,
            },
        ];
        let graph = SynapticGraph::from_descriptors(3, &descs)
            .expect("hand-written descriptors must build a graph");
        let json = serde_json::to_string(&graph).expect("serialize SynapticGraph to JSON");
        let restored: SynapticGraph =
            serde_json::from_str(&json).expect("deserialize SynapticGraph from JSON");
        assert_eq!(restored, graph);
    }

    #[test]
    fn max_delay_reports_correctly() {
        let descs = vec![
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.5,
                delay: 7,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 1,
                target: 0,
                weight: 0.5,
                delay: 3,
                polarity: Polarity::Excitatory,
            },
        ];
        let graph = SynapticGraph::from_descriptors(2, &descs).unwrap();
        assert_eq!(graph.max_delay(), 7);
    }

    #[test]
    fn deserialize_rejects_row_ptr_length_mismatch() {
        let json = r#"{"neuron_count":2,"row_ptr":[0,1],"targets":[],"weights":[],"delays":[],"polarities":[]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_non_monotonic_row_ptr() {
        let json = r#"{"neuron_count":2,"row_ptr":[0,3,1],"targets":[0,0,0],"weights":[0.1,0.1,0.1],"delays":[1,1,1],"polarities":["Excitatory","Excitatory","Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_edge_array_length_mismatch() {
        let json = r#"{"neuron_count":1,"row_ptr":[0,2],"targets":[0],"weights":[0.1,0.1],"delays":[1,1],"polarities":["Excitatory","Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn validate_weights_finite_rejects_nan_and_infinite() {
        // JSON has no literal for NaN/Infinity, so this exercises the
        // validator directly rather than through a JSON round-trip; a
        // non-JSON serde format (bincode, postcard, cbor) could otherwise
        // smuggle these values into a deserialized graph.
        assert!(validate_weights_finite(&[0.1, f32::NAN]).is_err());
        assert!(validate_weights_finite(&[0.1, f32::INFINITY]).is_err());
        assert!(validate_weights_finite(&[0.1, f32::NEG_INFINITY]).is_err());
        assert!(validate_weights_finite(&[0.1, -0.4, 2.0]).is_ok());
    }

    #[test]
    fn deserialize_rejects_usize_max_neuron_count_without_overflow_panic() {
        let json = r#"{"neuron_count":18446744073709551615,"row_ptr":[0,1],"targets":[0],"weights":[0.1],"delays":[1],"polarities":["Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_out_of_range_target() {
        let json = r#"{"neuron_count":1,"row_ptr":[0,1],"targets":[5],"weights":[0.1],"delays":[1],"polarities":["Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn out_of_bounds_source_rejected() {
        let descs = vec![SynapseDescriptor {
            source: 5,
            target: 0,
            weight: 0.5,
            delay: 1,
            polarity: Polarity::Excitatory,
        }];
        assert!(SynapticGraph::from_descriptors(3, &descs).is_err());
    }

    fn descriptor(weight: f32, polarity: Polarity) -> SynapseDescriptor {
        SynapseDescriptor {
            source: 0,
            target: 1,
            weight,
            delay: 0,
            polarity,
        }
    }

    #[test]
    fn from_descriptors_rejects_negative_inhibitory_magnitude() {
        // Direct Rust construction bypasses serde; the graph loader must not
        // turn this into an excitatory +0.5 via effective_weight().
        let invalid = descriptor(-0.5, Polarity::Inhibitory);
        assert!(SynapticGraph::from_descriptors(2, &[invalid]).is_err());
    }

    #[test]
    fn from_descriptors_rejects_nan_and_infinite_magnitudes() {
        assert!(
            SynapticGraph::from_descriptors(2, &[descriptor(f32::NAN, Polarity::Excitatory)])
                .is_err()
        );
        assert!(
            SynapticGraph::from_descriptors(2, &[descriptor(f32::INFINITY, Polarity::Excitatory)])
                .is_err()
        );
        assert!(
            SynapticGraph::from_descriptors(
                2,
                &[descriptor(f32::NEG_INFINITY, Polarity::Inhibitory)]
            )
            .is_err()
        );
    }

    #[test]
    fn from_descriptors_accepts_zero_and_positive_magnitudes() {
        let zero = SynapticGraph::from_descriptors(2, &[descriptor(0.0, Polarity::Inhibitory)]);
        assert!(zero.is_ok());
        let signed_zero =
            SynapticGraph::from_descriptors(2, &[descriptor(-0.0, Polarity::Excitatory)]);
        assert!(signed_zero.is_ok());
        let positive = SynapticGraph::from_descriptors(2, &[descriptor(0.5, Polarity::Excitatory)]);
        assert!(positive.is_ok());
        let edges: Vec<_> = positive.unwrap().outgoing(0).collect();
        assert!((edges[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn deserialize_rejects_positive_inhibitory_weight() {
        let json = r#"{"neuron_count":2,"row_ptr":[0,1,1],"targets":[1],"weights":[0.5],"delays":[0],"polarities":["Inhibitory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_negative_excitatory_weight() {
        let json = r#"{"neuron_count":2,"row_ptr":[0,1,1],"targets":[1],"weights":[-0.5],"delays":[0],"polarities":["Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(json).is_err());
    }

    #[test]
    fn deserialize_accepts_zero_weight_for_both_polarities() {
        let inh = r#"{"neuron_count":2,"row_ptr":[0,1,1],"targets":[1],"weights":[0.0],"delays":[0],"polarities":["Inhibitory"]}"#;
        let exc = r#"{"neuron_count":2,"row_ptr":[0,1,1],"targets":[1],"weights":[0.0],"delays":[0],"polarities":["Excitatory"]}"#;
        assert!(serde_json::from_str::<SynapticGraph>(inh).is_ok());
        assert!(serde_json::from_str::<SynapticGraph>(exc).is_ok());
    }

    #[test]
    fn validate_weights_agree_with_polarities_rejects_nan_without_json() {
        // JSON cannot represent NaN/Inf; a binary serde format could.
        assert!(
            validate_weights_agree_with_polarities(&[f32::NAN], &[Polarity::Excitatory]).is_err()
        );
        assert!(
            validate_weights_agree_with_polarities(&[f32::INFINITY], &[Polarity::Inhibitory])
                .is_err()
        );
        assert!(
            validate_weights_agree_with_polarities(&[f32::NEG_INFINITY], &[Polarity::Excitatory])
                .is_err()
        );
        assert!(validate_weights_agree_with_polarities(&[-0.0], &[Polarity::Inhibitory]).is_ok());
        assert!(validate_weights_agree_with_polarities(&[0.0], &[Polarity::Excitatory]).is_ok());
    }
}
