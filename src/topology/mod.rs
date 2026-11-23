// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology module — network graph construction and wiring.
//!
//! Provides the [`SynapticGraph`] adjacency structure and deterministic
//! topology generators inspired by classic network science models, plus a
//! versioned [`TopologyDigest`] over the canonical logical graph.

mod digest;
mod generators;
mod graph;
mod stats;
mod wiring_rules;

pub use digest::{
    TOPOLOGY_DIGEST_ALGORITHM, TOPOLOGY_DIGEST_DOMAIN, TOPOLOGY_DIGEST_SCHEMA_VERSION,
    TopologyDigest, TopologyDigestParseError,
};
pub use generators::{
    generate_from_config, generate_layered, generate_layered_with_model, generate_random,
    generate_random_with_model, generate_scale_free, generate_scale_free_with_model,
    generate_small_world, generate_small_world_with_model,
};
pub use graph::SynapticGraph;
pub use wiring_rules::{apply_dale_polarity, assign_delays, assign_delays_from_model};
