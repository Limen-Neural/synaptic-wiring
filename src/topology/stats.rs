// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only topology statistics for [`SynapticGraph`].
//!
//! Kept separate from `graph.rs` so the CSR container stays small. Every
//! method here is derived from the neuron and synapse counts the graph
//! already holds: no new storage and no heap allocation.

use super::SynapticGraph;

impl SynapticGraph {
    /// Mean number of outgoing synapses per neuron, as `f64`.
    ///
    /// Equal to `synapse_count() / neuron_count()`. Returns `0.0` for a graph
    /// with no neurons. This is the `f64` counterpart of
    /// [`SynapticGraph::mean_degree`], which returns `f32`. Does not allocate.
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
    /// // 4 neurons, 3 synapses: 0→1, 0→2, 1→3.
    /// let graph = SynapticGraph::from_descriptors(4, &[syn(0, 1), syn(0, 2), syn(1, 3)])?;
    /// assert_eq!(graph.mean_out_degree(), 0.75);
    ///
    /// assert_eq!(SynapticGraph::new(0).mean_out_degree(), 0.0);
    /// # Ok::<(), synaptic_wiring::MeshError>(())
    /// ```
    pub fn mean_out_degree(&self) -> f64 {
        let neurons = self.neuron_count();
        if neurons == 0 {
            return 0.0;
        }
        self.synapse_count() as f64 / neurons as f64
    }

    /// Fraction of the `N × N` adjacency matrix that holds a synapse.
    ///
    /// Equal to `synapse_count() / neuron_count()²`. The denominator counts
    /// self-connections because [`SynapticGraph::from_descriptors`] accepts
    /// them, so `density()` matches `1.0 - sparsity()` up to the `f32`
    /// precision of [`SynapticGraph::sparsity`]. That only holds while
    /// `sparsity`'s own `usize` square of the neuron count fits, which on
    /// 32-bit targets stops above 65,535 neurons; `density` squares in `f64`,
    /// so it cannot overflow. Parallel synapses between the same pair each
    /// count, so a multigraph can exceed `1.0`. Returns `0.0` for a graph with
    /// no neurons. Does not allocate.
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
    /// // 4 neurons → 16 possible synapses; 4 present.
    /// let descs = [syn(0, 1), syn(0, 2), syn(1, 3), syn(3, 0)];
    /// let graph = SynapticGraph::from_descriptors(4, &descs)?;
    /// assert_eq!(graph.density(), 0.25);
    ///
    /// assert_eq!(SynapticGraph::new(0).density(), 0.0);
    /// assert_eq!(SynapticGraph::new(10).density(), 0.0);
    /// # Ok::<(), synaptic_wiring::MeshError>(())
    /// ```
    pub fn density(&self) -> f64 {
        let neurons = self.neuron_count() as f64;
        if neurons == 0.0 {
            return 0.0;
        }
        self.synapse_count() as f64 / (neurons * neurons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NeuronId, Polarity, SynapseDescriptor};

    fn syn(source: NeuronId, target: NeuronId) -> SynapseDescriptor {
        SynapseDescriptor {
            source,
            target,
            weight: 0.5,
            delay: 1,
            polarity: Polarity::Excitatory,
        }
    }

    #[test]
    fn empty_graph_with_zero_neurons_reports_zero_stats() {
        let g = SynapticGraph::new(0);
        assert_eq!(g.neuron_count(), 0);
        assert_eq!(g.synapse_count(), 0);
        assert_eq!(g.mean_out_degree(), 0.0);
        assert_eq!(g.density(), 0.0);
    }

    #[test]
    fn unconnected_graph_reports_zero_degree_and_density() {
        let g = SynapticGraph::new(5);
        assert_eq!(g.neuron_count(), 5);
        assert_eq!(g.synapse_count(), 0);
        assert_eq!(g.mean_out_degree(), 0.0);
        assert_eq!(g.density(), 0.0);
    }

    #[test]
    fn hand_built_graph_reports_expected_stats() {
        // Neuron 2 has no outgoing synapses; it still counts in the mean.
        let descs = [syn(0, 1), syn(0, 2), syn(1, 0)];
        let g = SynapticGraph::from_descriptors(3, &descs).expect("valid descriptors");
        assert_eq!(g.neuron_count(), 3);
        assert_eq!(g.synapse_count(), 3);
        assert_eq!(g.out_degree(2), 0);
        assert_eq!(g.mean_out_degree(), 1.0);
        assert_eq!(g.density(), 3.0 / 9.0);
    }

    #[test]
    fn fully_connected_graph_with_self_loops_has_unit_density() {
        let n: NeuronId = 4;
        let descs: Vec<_> = (0..n)
            .flat_map(|s| (0..n).map(move |t| syn(s, t)))
            .collect();
        let g = SynapticGraph::from_descriptors(n as usize, &descs).expect("valid descriptors");
        assert_eq!(g.mean_out_degree(), 4.0);
        assert_eq!(g.density(), 1.0);
    }

    #[test]
    fn density_is_complement_of_sparsity() {
        let descs = [syn(0, 1), syn(1, 2), syn(2, 3), syn(3, 4), syn(4, 0)];
        let g = SynapticGraph::from_descriptors(5, &descs).expect("valid descriptors");
        assert!((g.density() - (1.0 - f64::from(g.sparsity()))).abs() < 1e-6);
    }

    #[test]
    fn mean_out_degree_agrees_with_f32_mean_degree() {
        let descs = [syn(0, 1), syn(0, 2), syn(1, 3), syn(2, 3), syn(3, 0)];
        let g = SynapticGraph::from_descriptors(4, &descs).expect("valid descriptors");
        assert_eq!(g.mean_out_degree(), 1.25);
        assert!((g.mean_out_degree() - f64::from(g.mean_degree())).abs() < 1e-6);
    }

    #[test]
    fn parallel_synapses_each_count() {
        let descs = [syn(0, 1), syn(0, 1)];
        let g = SynapticGraph::from_descriptors(2, &descs).expect("valid descriptors");
        assert_eq!(g.synapse_count(), 2);
        assert_eq!(g.mean_out_degree(), 1.0);
        assert_eq!(g.density(), 0.5);
    }
}
