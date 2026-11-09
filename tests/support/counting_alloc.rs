// SPDX-License-Identifier: MIT OR Apache-2.0

//! Counting global allocator shared by the allocation-guard test binaries.
//!
//! Include it with `#[path = "support/counting_alloc.rs"] mod counting_alloc;`
//! from a dedicated integration-test file: it installs a `#[global_allocator]`,
//! so each including file must be its own test binary and the counter cannot
//! affect other test crates. `tests/support/` has no `main.rs`, so Cargo does
//! not build it as a test target on its own.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Runs `f` and returns its result with the number of heap allocations
/// (including reallocations) it made.
pub fn allocation_delta<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let value = f();
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    (value, after.saturating_sub(before))
}
