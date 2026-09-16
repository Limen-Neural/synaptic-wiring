// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology module — network graph construction and wiring.
//!
//! Provides the [`SynapticGraph`] adjacency structure and deterministic
//! topology generators inspired by classic network science models, plus a
//! versioned [`TopologyDigest`] over the canonical logical graph.

mod digest;
mod generators;
mod graph;
mod wiring_rules;

pub use digest::{
    TOPOLOGY_DIGEST_ALGORITHM, TOPOLOGY_DIGEST_DOMAIN, TOPOLOGY_DIGEST_SCHEMA_VERSION,
    TopologyDigest, TopologyDigestParseError,
};
pub use generators::{
    generate_layered, generate_random, generate_scale_free, generate_small_world,
};
pub use graph::SynapticGraph;
pub use wiring_rules::{apply_dale_polarity, assign_delays};
