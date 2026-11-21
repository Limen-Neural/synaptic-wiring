// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dispatch contract for [`synaptic_wiring::topology::generate_from_config`]
//! (GH#101 / LIM-1560): each supported [`ConnectionModel`] variant must
//! produce a graph whose `topology_digest()` equals the digest of calling
//! the underlying generator directly with the mapped arguments. Digest
//! equality — not synapse count or internal layout — is the oracle.

use synaptic_wiring::error::MeshError;
use synaptic_wiring::topology::{
    SynapticGraph, generate_from_config, generate_layered, generate_random, generate_scale_free,
    generate_small_world,
};
use synaptic_wiring::types::{ConnectionModel, TopologyConfig};

fn config(neuron_count: usize, model: ConnectionModel) -> TopologyConfig {
    TopologyConfig {
        neuron_count,
        model,
        ..TopologyConfig::default()
    }
}

fn assert_dispatch_matches(dispatched: SynapticGraph, direct: SynapticGraph) {
    assert_eq!(
        dispatched.topology_digest(),
        direct.topology_digest(),
        "dispatch must produce the generator's exact graph"
    );
}

#[test]
fn dispatch_uniform_matches_generate_random() {
    let cfg = config(64, ConnectionModel::Uniform { p: 0.1 });
    let dispatched = generate_from_config(&cfg).unwrap();
    let direct = generate_random(64, 0.1, cfg.max_delay, cfg.inhibitory_fraction).unwrap();
    assert_dispatch_matches(dispatched, direct);
}

#[test]
fn dispatch_small_world_matches_generate_small_world() {
    let cfg = config(32, ConnectionModel::SmallWorld { k: 4, beta: 0.3 });
    let dispatched = generate_from_config(&cfg).unwrap();
    let direct = generate_small_world(32, 4, 0.3, cfg.max_delay, cfg.inhibitory_fraction).unwrap();
    assert_dispatch_matches(dispatched, direct);
}

#[test]
fn dispatch_scale_free_matches_generate_scale_free() {
    let cfg = config(50, ConnectionModel::ScaleFree { m0: 5, m: 3 });
    let dispatched = generate_from_config(&cfg).unwrap();
    let direct = generate_scale_free(50, 5, 3, cfg.max_delay, cfg.inhibitory_fraction).unwrap();
    assert_dispatch_matches(dispatched, direct);
}

#[test]
fn dispatch_layered_matches_generate_layered() {
    let cfg = config(
        28,
        ConnectionModel::Layered {
            layer_sizes: vec![8, 16, 4],
            inter_layer_p: 0.7,
        },
    );
    let dispatched = generate_from_config(&cfg).unwrap();
    let direct =
        generate_layered(&[8, 16, 4], 0.7, cfg.max_delay, cfg.inhibitory_fraction).unwrap();
    assert_dispatch_matches(dispatched, direct);
}

#[test]
fn dispatch_distance_dependent_is_unsupported() {
    let cfg = config(
        16,
        ConnectionModel::DistanceDependent {
            p_max: 0.5,
            lambda: 10.0,
        },
    );
    let err = generate_from_config(&cfg).unwrap_err();
    assert!(
        matches!(err, MeshError::UnsupportedModel(_)),
        "DistanceDependent has no generator, expected UnsupportedModel, got {err:?}"
    );
}

#[test]
fn dispatch_layered_rejects_size_sum_mismatch() {
    let cfg = config(
        30,
        ConnectionModel::Layered {
            layer_sizes: vec![8, 16, 4],
            inter_layer_p: 0.5,
        },
    );
    let err = generate_from_config(&cfg).unwrap_err();
    assert!(
        matches!(err, MeshError::InvalidConfig(_)),
        "layer_sizes sum != neuron_count must be InvalidConfig, got {err:?}"
    );
}
