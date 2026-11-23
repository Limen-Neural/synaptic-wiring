// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DelayModel` generator contract (GH#100 / LIM-1559): the legacy
//! `max_delay: u16` generators are wrappers over the `*_with_model` APIs, so
//! equal inputs must give equal graphs (by `topology_digest()`). Descriptor
//! delays are covered in `delay_model_descriptors.rs`.

use synaptic_wiring::error::MeshError;
use synaptic_wiring::topology::{
    SynapticGraph, generate_layered, generate_layered_with_model, generate_random,
    generate_random_with_model, generate_scale_free, generate_scale_free_with_model,
    generate_small_world, generate_small_world_with_model,
};
use synaptic_wiring::types::DelayModel;

const INH: f32 = 0.2;

fn uniform(min_delay: u16, max_delay: u16) -> DelayModel {
    DelayModel::UniformRandom {
        min_delay,
        max_delay,
    }
}

fn legacy_model(max_delay: u16) -> DelayModel {
    if max_delay == 0 {
        DelayModel::Fixed { delay: 0 }
    } else {
        uniform(1, max_delay)
    }
}

fn assert_same(legacy: SynapticGraph, model: SynapticGraph) {
    assert_eq!(legacy.topology_digest(), model.topology_digest());
}

fn delays(graph: &SynapticGraph) -> Vec<u16> {
    graph.delays_slice().to_vec()
}

#[test]
fn random_legacy_matches_model() {
    for max in [0, 1, 5, 64] {
        assert_same(
            generate_random(48, 0.15, max, INH).unwrap(),
            generate_random_with_model(48, 0.15, &legacy_model(max), INH).unwrap(),
        );
    }
}

#[test]
fn small_world_legacy_matches_model() {
    for max in [0, 1, 5, 64] {
        assert_same(
            generate_small_world(40, 4, 0.2, max, INH).unwrap(),
            generate_small_world_with_model(40, 4, 0.2, &legacy_model(max), INH).unwrap(),
        );
    }
}

#[test]
fn scale_free_legacy_matches_model() {
    for max in [0, 1, 5, 64] {
        assert_same(
            generate_scale_free(40, 3, 2, max, INH).unwrap(),
            generate_scale_free_with_model(40, 3, 2, &legacy_model(max), INH).unwrap(),
        );
    }
}

#[test]
fn layered_legacy_matches_model() {
    for max in [0, 1, 5, 64] {
        assert_same(
            generate_layered(&[12, 10, 8], 0.4, max, INH).unwrap(),
            generate_layered_with_model(&[12, 10, 8], 0.4, &legacy_model(max), INH).unwrap(),
        );
    }
}

#[test]
fn fixed_model_sets_every_generator_delay() {
    let model = DelayModel::Fixed { delay: 7 };
    let graphs = [
        generate_random_with_model(32, 0.2, &model, INH).unwrap(),
        generate_small_world_with_model(32, 4, 0.2, &model, INH).unwrap(),
        generate_scale_free_with_model(32, 3, 2, &model, INH).unwrap(),
        generate_layered_with_model(&[10, 10, 10], 0.5, &model, INH).unwrap(),
    ];
    for graph in &graphs {
        let d = delays(graph);
        assert!(!d.is_empty());
        assert!(d.iter().all(|&x| x == 7));
    }
}

#[test]
fn uniform_spread_respects_shifted_range() {
    for (min, max) in [(0, 6), (3, 9), (5, 5), (0, 0), (65_000, u16::MAX)] {
        let graph = generate_random_with_model(48, 0.2, &uniform(min, max), INH).unwrap();
        let d = delays(&graph);
        assert!(!d.is_empty());
        assert!(d.iter().all(|&x| (min..=max).contains(&x)), "{min}..={max}");
        if min == max {
            assert!(d.iter().all(|&x| x == min));
        }
    }
}

#[test]
fn uniform_shift_is_legacy_spread_offset() {
    // span 6 starting at 4 equals the legacy spread of max_delay = 6, plus 3.
    let legacy = delays(&generate_random(48, 0.2, 6, INH).unwrap());
    let shifted = delays(&generate_random_with_model(48, 0.2, &uniform(4, 9), INH).unwrap());
    let expected: Vec<u16> = legacy.iter().map(|d| d + 3).collect();
    assert_eq!(shifted, expected);
}

#[test]
fn generators_reject_distance_proportional() {
    let model = DelayModel::DistanceProportional {
        speed: 1.0,
        min_delay: 1,
        max_delay: 5,
    };
    let errors = [
        generate_random_with_model(16, 0.2, &model, INH).unwrap_err(),
        generate_small_world_with_model(16, 4, 0.2, &model, INH).unwrap_err(),
        generate_scale_free_with_model(16, 3, 2, &model, INH).unwrap_err(),
        generate_layered_with_model(&[8, 8], 0.5, &model, INH).unwrap_err(),
    ];
    for err in errors {
        assert!(matches!(err, MeshError::UnsupportedModel(_)), "{err:?}");
    }
}

#[test]
fn generators_reject_inverted_range() {
    for model in [
        uniform(5, 4),
        DelayModel::DistanceProportional {
            speed: 1.0,
            min_delay: 5,
            max_delay: 4,
        },
    ] {
        let errors = [
            generate_random_with_model(16, 0.2, &model, INH).unwrap_err(),
            generate_small_world_with_model(16, 4, 0.2, &model, INH).unwrap_err(),
            generate_scale_free_with_model(16, 3, 2, &model, INH).unwrap_err(),
            generate_layered_with_model(&[8, 8], 0.5, &model, INH).unwrap_err(),
        ];
        for err in errors {
            assert!(matches!(err, MeshError::InvalidConfig(_)), "{err:?}");
        }
    }
}
