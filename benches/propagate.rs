// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion cases for allocating vs caller-buffer propagation.
//!
//! Sizes (16 / 256 / 4096) and delay depths (short = 1, long = 16) match
//! LIM-1222. These benches measure throughput; allocation-freedom of the
//! reuse path is asserted in `tests/propagate_into_alloc.rs`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::generate_small_world;

const NEURON_COUNTS: [usize; 3] = [16, 256, 4096];
const SHORT_DELAY: u16 = 1;
const LONG_DELAY: u16 = 16;

fn mesh_for(n: usize, max_delay: u16) -> SynapticMesh {
    let graph =
        generate_small_world(n, 4, 0.2, max_delay, 0.2).expect("deterministic small-world graph");
    SynapticMesh::new(graph)
}

fn bench_propagate(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagate");
    for n in NEURON_COUNTS {
        for (label, max_delay) in [("short", SHORT_DELAY), ("long", LONG_DELAY)] {
            group.throughput(Throughput::Elements(n as u64));
            let mut mesh = mesh_for(n, max_delay);
            let spikes = vec![false; n];
            group.bench_with_input(BenchmarkId::new(format!("alloc/{label}"), n), &n, |b, _| {
                b.iter(|| mesh.propagate(&spikes).expect("propagate"));
            });

            let mut mesh = mesh_for(n, max_delay);
            let spikes = vec![false; n];
            let mut output = vec![0.0; n];
            group.bench_with_input(BenchmarkId::new(format!("into/{label}"), n), &n, |b, _| {
                b.iter(|| {
                    mesh.propagate_into(&spikes, &mut output)
                        .expect("propagate_into");
                    output[0]
                });
            });
        }
    }
    group.finish();
}

fn bench_propagate_graded(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagate_graded");
    for n in NEURON_COUNTS {
        for (label, max_delay) in [("short", SHORT_DELAY), ("long", LONG_DELAY)] {
            group.throughput(Throughput::Elements(n as u64));
            let mut mesh = mesh_for(n, max_delay);
            let activations = vec![0.0; n];
            group.bench_with_input(BenchmarkId::new(format!("alloc/{label}"), n), &n, |b, _| {
                b.iter(|| {
                    mesh.propagate_graded(&activations)
                        .expect("propagate_graded")
                });
            });

            let mut mesh = mesh_for(n, max_delay);
            let activations = vec![0.0; n];
            let mut output = vec![0.0; n];
            group.bench_with_input(BenchmarkId::new(format!("into/{label}"), n), &n, |b, _| {
                b.iter(|| {
                    mesh.propagate_graded_into(&activations, &mut output)
                        .expect("propagate_graded_into");
                    output[0]
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_propagate, bench_propagate_graded);
criterion_main!(benches);
