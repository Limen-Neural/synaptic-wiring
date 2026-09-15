// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-count guard for the caller-buffer propagation path.
//!
//! After [`SynapticMesh::new`], successful `propagate_into` /
//! `propagate_graded_into` ticks must not heap-allocate. This file is its
//! own integration-test binary so the counting `#[global_allocator]` cannot
//! affect other test crates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::generate_small_world;

struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocation_delta<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let value = f();
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    (value, after.saturating_sub(before))
}

#[test]
fn reuse_path_is_allocation_free_after_new() {
    let n = 64;
    let graph = generate_small_world(n, 6, 0.2, 8, 0.2).unwrap();
    let mut boolean_mesh = SynapticMesh::new(graph.clone());
    let mut graded_mesh = SynapticMesh::new(graph);

    let mut spikes = vec![false; n];
    spikes[0] = true;
    spikes[3] = true;
    let mut activations = vec![0.0; n];
    activations[0] = 0.8;
    activations[7] = -0.3;
    let mut boolean_out = vec![0.0; n];
    let mut graded_out = vec![0.0; n];

    for tick in 0..16 {
        if tick > 0 {
            spikes.fill(false);
            activations.fill(0.0);
            if tick % 2 == 0 {
                spikes[tick % n] = true;
                activations[tick % n] = 0.4;
            }
        }

        let (boolean_result, boolean_allocs) =
            allocation_delta(|| boolean_mesh.propagate_into(&spikes, &mut boolean_out));
        boolean_result.expect("boolean propagate_into must succeed");
        assert_eq!(boolean_allocs, 0, "propagate_into allocated on tick {tick}");

        let (graded_result, graded_allocs) =
            allocation_delta(|| graded_mesh.propagate_graded_into(&activations, &mut graded_out));
        graded_result.expect("graded propagate_into must succeed");
        assert_eq!(
            graded_allocs, 0,
            "propagate_graded_into allocated on tick {tick}"
        );
    }
}
