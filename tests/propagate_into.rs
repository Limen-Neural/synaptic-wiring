// SPDX-License-Identifier: MIT OR Apache-2.0

//! Equivalence and atomicity for caller-buffer propagation.
//!
//! [`SynapticMesh::propagate_into`] and [`SynapticMesh::propagate_graded_into`]
//! must match the allocating APIs on destination, sign, magnitude, and
//! delivery tick, and must leave tick, delay buffers, and caller output
//! unchanged when input is rejected.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::{
    SynapticGraph, generate_layered, generate_random, generate_scale_free, generate_small_world,
};
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

const EPS: f32 = 1e-6;

fn descriptor(
    source: u32,
    target: u32,
    weight: f32,
    delay: u16,
    polarity: Polarity,
) -> SynapseDescriptor {
    SynapseDescriptor {
        source,
        target,
        weight,
        delay,
        polarity,
    }
}

fn boolean_sequence(n: usize, ticks: usize) -> Vec<Vec<bool>> {
    (0..ticks)
        .map(|tick| {
            let mut spikes = vec![false; n];
            if n > 0 {
                spikes[tick % n] = true;
            }
            if n > 1 && tick % 3 == 0 {
                spikes[(tick * 3) % n] = true;
            }
            spikes
        })
        .collect()
}

fn graded_sequence(n: usize, ticks: usize) -> Vec<Vec<f32>> {
    (0..ticks)
        .map(|tick| {
            let mut activations = vec![0.0; n];
            if n > 0 {
                activations[tick % n] = 0.5 + 0.1 * (tick as f32);
            }
            if n > 1 && tick % 2 == 0 {
                activations[(tick * 2) % n] = -0.25;
            }
            activations
        })
        .collect()
}

fn assert_boolean_equivalent(graph: SynapticGraph, ticks: usize) {
    let n = graph.neuron_count();
    let mut alloc = SynapticMesh::new(graph.clone());
    let mut reuse = SynapticMesh::new(graph);
    let mut output = vec![0.0; n];
    for spikes in boolean_sequence(n, ticks) {
        let allocated = alloc.propagate(&spikes).unwrap();
        reuse.propagate_into(&spikes, &mut output).unwrap();
        assert_eq!(
            allocated,
            output,
            "boolean currents diverged at tick {}",
            alloc.tick() - 1
        );
        assert_eq!(alloc.tick(), reuse.tick());
    }
}

fn assert_graded_equivalent(graph: SynapticGraph, ticks: usize) {
    let n = graph.neuron_count();
    let mut alloc = SynapticMesh::new(graph.clone());
    let mut reuse = SynapticMesh::new(graph);
    let mut output = vec![0.0; n];
    for activations in graded_sequence(n, ticks) {
        let allocated = alloc.propagate_graded(&activations).unwrap();
        reuse
            .propagate_graded_into(&activations, &mut output)
            .unwrap();
        assert_eq!(allocated.len(), output.len());
        for (i, (&a, &b)) in allocated.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "graded currents diverged at tick {}, neuron {i}: {a} vs {b}",
                alloc.tick() - 1
            );
        }
        assert_eq!(alloc.tick(), reuse.tick());
    }
}

#[test]
fn zero_delay_boolean_and_graded_match() {
    let graph = SynapticGraph::from_descriptors(
        3,
        &[
            descriptor(0, 1, 0.75, 0, Polarity::Excitatory),
            descriptor(1, 2, 0.40, 0, Polarity::Excitatory),
            descriptor(2, 0, 0.20, 0, Polarity::Inhibitory),
        ],
    )
    .unwrap();
    assert_boolean_equivalent(graph.clone(), 8);
    assert_graded_equivalent(graph, 8);
}

#[test]
fn delayed_boolean_and_graded_match() {
    let graph = SynapticGraph::from_descriptors(
        4,
        &[
            descriptor(0, 1, 0.75, 0, Polarity::Excitatory),
            descriptor(0, 2, 0.50, 1, Polarity::Excitatory),
            descriptor(0, 3, 0.25, 2, Polarity::Excitatory),
            descriptor(1, 3, 0.40, 3, Polarity::Excitatory),
            descriptor(2, 3, 0.60, 1, Polarity::Inhibitory),
        ],
    )
    .unwrap();
    assert_boolean_equivalent(graph.clone(), 12);
    assert_graded_equivalent(graph, 12);
}

#[test]
fn sparse_and_dense_random_graphs_match() {
    let sparse = generate_random(24, 0.05, 4, 0.2).unwrap();
    let dense = generate_random(12, 1.0, 3, 0.2).unwrap();
    assert_boolean_equivalent(sparse.clone(), 16);
    assert_graded_equivalent(sparse, 16);
    assert_boolean_equivalent(dense.clone(), 16);
    assert_graded_equivalent(dense, 16);
}

#[test]
fn excitatory_and_inhibitory_graphs_match() {
    let excitatory = generate_random(16, 0.4, 3, 0.0).unwrap();
    let inhibitory = generate_random(16, 0.4, 3, 1.0).unwrap();
    assert_boolean_equivalent(excitatory.clone(), 12);
    assert_graded_equivalent(excitatory, 12);
    assert_boolean_equivalent(inhibitory.clone(), 12);
    assert_graded_equivalent(inhibitory, 12);
}

#[test]
fn small_world_scale_free_and_layered_match() {
    let small_world = generate_small_world(32, 4, 0.2, 5, 0.2).unwrap();
    let scale_free = generate_scale_free(24, 4, 2, 5, 0.2).unwrap();
    let layered = generate_layered(&[4, 8, 2], 1.0, 3, 0.2).unwrap();
    assert_boolean_equivalent(small_world.clone(), 12);
    assert_graded_equivalent(small_world, 12);
    assert_boolean_equivalent(scale_free.clone(), 12);
    assert_graded_equivalent(scale_free, 12);
    assert_boolean_equivalent(layered.clone(), 12);
    assert_graded_equivalent(layered, 12);
}

#[test]
fn rejected_boolean_input_is_fully_atomic() {
    let graph =
        SynapticGraph::from_descriptors(2, &[descriptor(0, 1, 1.0, 2, Polarity::Excitatory)])
            .unwrap();
    let mut mesh = SynapticMesh::new(graph);
    mesh.propagate(&[true, false]).unwrap();
    let tick = mesh.tick();
    let mut sentinel = [42.0_f32, 42.0];

    assert!(mesh.propagate_into(&[true], &mut sentinel).is_err());
    let mut too_long = [1.0_f32; 3];
    assert!(mesh.propagate_into(&[false, false], &mut too_long).is_err());
    assert_eq!(mesh.tick(), tick);
    assert_eq!(sentinel, [42.0, 42.0]);
    assert_eq!(too_long, [1.0, 1.0, 1.0]);

    assert_eq!(mesh.propagate(&[false, false]).unwrap()[1], 0.0);
    let arrived = mesh.propagate(&[false, false]).unwrap();
    assert!((arrived[1] - 1.0).abs() < EPS);
}

#[test]
fn rejected_graded_input_is_fully_atomic() {
    let graph = SynapticGraph::from_descriptors(
        3,
        &[
            descriptor(0, 1, 1.0, 2, Polarity::Excitatory),
            descriptor(2, 1, f32::MAX, 0, Polarity::Excitatory),
        ],
    )
    .unwrap();
    let mut mesh = SynapticMesh::new(graph);
    mesh.propagate(&[true, false, false]).unwrap();
    let tick = mesh.tick();
    let mut sentinel = [42.0_f32; 3];

    assert!(
        mesh.propagate_graded_into(&[0.0, f32::NAN, 0.0], &mut sentinel)
            .is_err()
    );
    assert!(
        mesh.propagate_graded_into(&[1.0, 0.0, f32::MAX], &mut sentinel)
            .is_err()
    );
    let mut too_short = [42.0_f32; 2];
    assert!(
        mesh.propagate_graded_into(&[0.0, 0.0, 0.0], &mut too_short)
            .is_err()
    );

    assert_eq!(mesh.tick(), tick);
    assert_eq!(sentinel, [42.0; 3]);
    assert_eq!(too_short, [42.0; 2]);

    assert_eq!(mesh.propagate(&[false, false, false]).unwrap()[1], 0.0);
    let arrived = mesh.propagate(&[false, false, false]).unwrap();
    assert!((arrived[1] - 1.0).abs() < EPS);
}

#[test]
fn overwrite_not_accumulate() {
    let graph =
        SynapticGraph::from_descriptors(2, &[descriptor(0, 1, 0.5, 0, Polarity::Excitatory)])
            .unwrap();
    let mut mesh = SynapticMesh::new(graph);
    let mut output = [9.0_f32, 9.0];
    mesh.propagate_into(&[false, false], &mut output).unwrap();
    assert_eq!(output, [0.0, 0.0]);
}
