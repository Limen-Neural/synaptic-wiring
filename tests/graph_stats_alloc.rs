// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-count guard for the `SynapticGraph` statistics API (#98).
//!
//! `neuron_count`, `synapse_count`, `mean_out_degree` and `density` must not
//! heap-allocate. This file is its own integration-test binary so the
//! counting `#[global_allocator]` cannot affect other test crates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use synaptic_wiring::topology::{SynapticGraph, generate_small_world};

struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: `ptr` came from this allocator; layout/new_size are forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator; layout is forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_during(graph: &SynapticGraph) -> u64 {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    black_box(graph.neuron_count());
    black_box(graph.synapse_count());
    black_box(graph.mean_out_degree());
    black_box(graph.density());
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    after.saturating_sub(before)
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
