// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-count guard for `SynapticGraph::outgoing_synapses` (LIM-1558).
//!
//! The iterator reads the CSR edge arrays in place and must not
//! heap-allocate, including for unconnected and out-of-range neurons. This
//! file is its own integration-test binary so the counting
//! `#[global_allocator]` cannot affect other test crates.

#[path = "support/counting_alloc.rs"]
mod counting_alloc;

use std::hint::black_box;

use counting_alloc::allocation_delta;
use synaptic_wiring::topology::{SynapticGraph, generate_small_world};

#[test]
fn outgoing_synapses_does_not_allocate() {
    let generated = generate_small_world(64, 6, 0.2, 8, 0.2).unwrap();
    let empty = SynapticGraph::new(0);
    let unconnected = SynapticGraph::new(16);

    for graph in [&generated, &empty, &unconnected] {
        let (visited, allocations) = allocation_delta(|| {
            (0..=graph.neuron_count())
                .flat_map(|n| graph.outgoing_synapses(n))
                .map(black_box)
                .count()
        });
        assert_eq!(visited, graph.synapse_count());
        assert_eq!(allocations, 0);
    }
}
