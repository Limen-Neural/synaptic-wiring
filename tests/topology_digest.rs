// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden fixtures and insertion-order permutation tests for
//! [`synaptic_wiring::TopologyDigest`].
//!
//! These values are the schema-v1 SHA-256 of the canonical little-endian
//! encoding documented on [`synaptic_wiring::TopologyDigest`]. They are
//! committed so a silent encoder change fails CI.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::{
    SynapticGraph, TOPOLOGY_DIGEST_ALGORITHM, TOPOLOGY_DIGEST_DOMAIN,
    TOPOLOGY_DIGEST_SCHEMA_VERSION, TopologyDigest,
};
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

/// Golden v1 digest for a 0-neuron empty graph.
const GOLDEN_EMPTY_0: &str = "synaptic-wiring.topology.digest.v1:sha256:14f893289199426a03b83570fda63b60872dedfe03c0b52a46a2061445d7a608";

/// Golden v1 digest for a 3-neuron empty graph.
const GOLDEN_EMPTY_3: &str = "synaptic-wiring.topology.digest.v1:sha256:a2c1014ab2a24a342e01878c5dbe69972e8ab4152cb36d2faf88ec210d0b6203";

/// Golden v1 digest for the three-edge fixture (any insertion order).
const GOLDEN_SMALL: &str = "synaptic-wiring.topology.digest.v1:sha256:be1786c690cda6d02d9c6fa06d1844642398ecebab3d19ccec981062711cd36f";

/// Golden v1 digest for excitatory `+0.0` vs `-0.0` (IEEE bit patterns differ).
const GOLDEN_PLUS_ZERO: &str = "synaptic-wiring.topology.digest.v1:sha256:72c7a0686739be822c0dedf46f71dee35a2c154b3debc37019afe9ece19ae9d0";
const GOLDEN_MINUS_ZERO: &str = "synaptic-wiring.topology.digest.v1:sha256:1b9b1789b7aed14363f73ed4f3c5708fc05180166ee7b9f620212f57d2a5812f";

fn desc(
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

fn small_edges(order: &[usize]) -> Vec<SynapseDescriptor> {
    let base = [
        desc(0, 1, 0.9, 3, Polarity::Excitatory),
        desc(0, 2, 0.15, 1, Polarity::Inhibitory),
        desc(1, 0, 0.5, 2, Polarity::Excitatory),
    ];
    order.iter().map(|&i| base[i]).collect()
}

#[test]
fn topology_digest_golden_fixtures() {
    assert_eq!(
        SynapticGraph::new(0).topology_digest().to_string(),
        GOLDEN_EMPTY_0
    );
    assert_eq!(
        SynapticGraph::new(3).topology_digest().to_string(),
        GOLDEN_EMPTY_3
    );
    let graph = SynapticGraph::from_descriptors(3, &small_edges(&[0, 1, 2])).unwrap();
    let digest = graph.topology_digest();
    assert_eq!(digest.to_string(), GOLDEN_SMALL);
    assert_eq!(digest.schema_version(), TOPOLOGY_DIGEST_SCHEMA_VERSION);
    assert_eq!(digest.algorithm(), TOPOLOGY_DIGEST_ALGORITHM);
    assert_eq!(digest.domain(), TOPOLOGY_DIGEST_DOMAIN);
}

#[test]
fn topology_digest_insertion_order_permutations() {
    let orders = [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for order in orders {
        let graph = SynapticGraph::from_descriptors(3, &small_edges(&order)).unwrap();
        assert_eq!(
            graph.topology_digest().to_string(),
            GOLDEN_SMALL,
            "insertion order {order:?}"
        );
        assert_eq!(
            SynapticMesh::new(graph).topology_digest().to_string(),
            GOLDEN_SMALL
        );
    }
}

#[test]
fn topology_digest_field_changes_are_detected() {
    let base = SynapticGraph::from_descriptors(3, &small_edges(&[0, 1, 2]))
        .unwrap()
        .topology_digest();

    let mut endpoint = small_edges(&[0, 1, 2]);
    endpoint[0].target = 0;
    let mut delay = small_edges(&[0, 1, 2]);
    delay[0].delay = 4;
    let mut polarity = small_edges(&[0, 1, 2]);
    polarity[2].polarity = Polarity::Inhibitory;
    let mut weight = small_edges(&[0, 1, 2]);
    weight[1].weight = 0.2;

    for (label, descriptors) in [
        ("endpoint", endpoint.as_slice()),
        ("delay", delay.as_slice()),
        ("polarity", polarity.as_slice()),
        ("weight", weight.as_slice()),
    ] {
        let changed = SynapticGraph::from_descriptors(3, descriptors)
            .unwrap()
            .topology_digest();
        assert_ne!(changed, base, "{label} change must change the digest");
        assert_ne!(changed.to_string(), GOLDEN_SMALL);
    }
}

#[test]
fn topology_digest_signed_zero_goldens() {
    let plus = SynapticGraph::from_descriptors(2, &[desc(0, 1, 0.0, 0, Polarity::Excitatory)])
        .unwrap()
        .topology_digest();
    let minus = SynapticGraph::from_descriptors(2, &[desc(0, 1, -0.0, 0, Polarity::Excitatory)])
        .unwrap()
        .topology_digest();
    assert_eq!(plus.to_string(), GOLDEN_PLUS_ZERO);
    assert_eq!(minus.to_string(), GOLDEN_MINUS_ZERO);
    assert_ne!(GOLDEN_PLUS_ZERO, GOLDEN_MINUS_ZERO);
}

#[test]
fn topology_digest_nan_is_rejected_before_hashing() {
    assert!(
        SynapticGraph::from_descriptors(2, &[desc(0, 1, f32::NAN, 0, Polarity::Excitatory)])
            .is_err()
    );
    assert!(
        SynapticGraph::from_descriptors(2, &[desc(0, 1, f32::INFINITY, 0, Polarity::Excitatory)])
            .is_err()
    );
}

#[test]
fn topology_digest_print_and_graph_json_round_trip() {
    let graph = SynapticGraph::from_descriptors(3, &small_edges(&[2, 0, 1])).unwrap();
    let digest = graph.topology_digest();
    let printed = digest.to_string();
    assert_eq!(printed.parse::<TopologyDigest>().unwrap(), digest);
    let json = serde_json::to_string(&digest).unwrap();
    let restored: TopologyDigest = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, digest);

    let graph_json = serde_json::to_string(&graph).unwrap();
    let restored_graph: SynapticGraph = serde_json::from_str(&graph_json).unwrap();
    assert_eq!(restored_graph.topology_digest(), digest);
    assert_eq!(digest.to_string(), GOLDEN_SMALL);
}
