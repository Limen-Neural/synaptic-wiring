// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-count guard for the `SynapticGraph` statistics API (#98).
//!
//! `neuron_count`, `synapse_count`, `mean_out_degree` and `density` must not
//! heap-allocate. This file is its own integration-test binary so the
//! counting `#[global_allocator]` cannot affect other test crates.

#[path = "support/counting_alloc.rs"]
mod counting_alloc;

use std::hint::black_box;

use counting_alloc::allocation_delta;
use synaptic_wiring::topology::{SynapticGraph, generate_small_world};

fn allocations_during(graph: &SynapticGraph) -> u64 {
    let ((), allocations) = allocation_delta(|| {
        black_box(graph.neuron_count());
        black_box(graph.synapse_count());
        black_box(graph.mean_out_degree());
        black_box(graph.density());
    });
    allocations
}

#[test]
fn stats_methods_do_not_allocate() {
    let generated = generate_small_world(64, 6, 0.2, 8, 0.2).unwrap();
    let empty = SynapticGraph::new(0);
    let unconnected = SynapticGraph::new(16);

    for graph in [&generated, &empty, &unconnected] {
        assert_eq!(allocations_during(graph), 0);
    }
}
