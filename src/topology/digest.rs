// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic topology digest for [`SynapticGraph`].
//!
//! Replay, checkpoint provenance, and cross-runtime comparisons need a stable
//! identifier of the *logical* graph that does not depend on CSR insertion
//! order, map iteration, pointer layout, host endianness, or JSON formatting.
//!
//! # Schema v1
//!
//! The printable value is:
//!
//! ```text
//! synaptic-wiring.topology.digest.v1:sha256:<64 lowercase hex chars>
//! ```
//!
//! The SHA-256 preimage is the concatenation of:
//!
//! 1. Domain separator [`TOPOLOGY_DIGEST_DOMAIN`] as UTF-8, then a NUL byte
//!    so the separator cannot prefix-collide with the payload.
//! 2. Schema version [`TOPOLOGY_DIGEST_SCHEMA_VERSION`] as little-endian `u16`.
//! 3. Neuron count as little-endian `u64`.
//! 4. Edge count as little-endian `u64`.
//! 5. One 23-byte record per logical edge, sorted in canonical order
//!    `(source, target, delay, polarity_tag, weight_bits)`:
//!    - `source`: little-endian `u64` (CSR row index)
//!    - `target`: little-endian `u64` (zero-extended [`crate::NeuronId`])
//!    - `weight_bits`: little-endian `u32` of [`f32::to_bits`] (IEEE 754
//!      bit pattern of the **signed** CSR weight)
//!    - `delay`: little-endian [`crate::DelayTicks`]
//!    - `polarity_tag`: `0` = [`Polarity::Excitatory`], `1` =
//!      [`Polarity::Inhibitory`]
//!
//! Bumping the schema version or domain separator produces a new printable
//! prefix; v1 values stay comparable forever.
//!
//! # Float canonicalization
//!
//! Schema v1 hashes the IEEE bit pattern. Construction and serde already
//! reject NaN and infinities, so a valid graph never contains them.
//! IEEE signed zero (`+0.0` vs `-0.0`) is **not** collapsed: the two bit
//! patterns digest differently, which is tested.

use std::fmt;
use std::str::FromStr;

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};

use super::graph::SynapticGraph;
use crate::types::Polarity;

/// Schema version mixed into the v1 preimage and reported by
/// [`TopologyDigest::schema_version`].
pub const TOPOLOGY_DIGEST_SCHEMA_VERSION: u16 = 1;

/// Domain separator (UTF-8) that prefixes the SHA-256 preimage and the
/// printable manifest form. Changing this is a new digest family.
pub const TOPOLOGY_DIGEST_DOMAIN: &str = "synaptic-wiring.topology.digest.v1";

/// Hash algorithm name embedded in the printable manifest form.
pub const TOPOLOGY_DIGEST_ALGORITHM: &str = "sha256";

/// NUL terminator after the domain separator in the hash preimage.
const DOMAIN_TERMINATOR: u8 = 0;

/// Printable, versioned digest of a graph's logical topology.
///
/// The [`fmt::Display`] form is stable and safe to store in manifests:
/// `{domain}:{algorithm}:{hex}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopologyDigest {
    schema_version: u16,
    hash: [u8; 32],
}

impl TopologyDigest {
    /// Compute the schema-v1 digest of `graph`.
    ///
    /// Edges are collected from the CSR and sorted into canonical order, so
    /// two graphs built from the same logical synapses in different insertion
    /// orders compare equal. Runtime mesh state (tick, delay buffer) is not
    /// part of the digest.
    #[must_use]
    pub fn from_graph(graph: &SynapticGraph) -> Self {
        let edges = canonical_edges(graph);
        Self {
            schema_version: TOPOLOGY_DIGEST_SCHEMA_VERSION,
            hash: hash_v1(graph.neuron_count() as u64, &edges),
        }
    }

    /// Schema version that produced this digest.
    #[must_use]
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Hash algorithm name (`sha256` for schema v1).
    #[must_use]
    pub fn algorithm(&self) -> &'static str {
        TOPOLOGY_DIGEST_ALGORITHM
    }

    /// Domain separator mixed into this digest family.
    #[must_use]
    pub fn domain(&self) -> &'static str {
        TOPOLOGY_DIGEST_DOMAIN
    }

    /// Raw 32-byte SHA-256 hash.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Lowercase hex encoding of the SHA-256 hash (64 characters).
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex_encode(&self.hash)
    }
}

impl fmt::Display for TopologyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{TOPOLOGY_DIGEST_DOMAIN}:{TOPOLOGY_DIGEST_ALGORITHM}:{}",
            self.to_hex()
        )
    }
}

impl FromStr for TopologyDigest {
    type Err = TopologyDigestParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (domain, algorithm, hex) = split_manifest(s)?;
        if domain != TOPOLOGY_DIGEST_DOMAIN {
            return Err(TopologyDigestParseError(format!(
                "unsupported digest domain {domain:?}"
            )));
        }
        if algorithm != TOPOLOGY_DIGEST_ALGORITHM {
            return Err(TopologyDigestParseError(format!(
                "unsupported digest algorithm {algorithm:?}"
            )));
        }
        Ok(Self {
            schema_version: TOPOLOGY_DIGEST_SCHEMA_VERSION,
            hash: hex_decode_sha256(hex)?,
        })
    }
}

impl Serialize for TopologyDigest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TopologyDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(DeError::custom)
    }
}

/// Error from parsing a printable [`TopologyDigest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyDigestParseError(String);

impl TopologyDigestParseError {
    fn format() -> Self {
        Self(format!(
            "expected {TOPOLOGY_DIGEST_DOMAIN}:{TOPOLOGY_DIGEST_ALGORITHM}:<64 hex chars>"
        ))
    }
}

impl fmt::Display for TopologyDigestParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TopologyDigestParseError {}

impl SynapticGraph {
    /// Deterministic, versioned digest of this graph's logical topology.
    ///
    /// See [`TopologyDigest`] for the schema, domain separator, and float
    /// bit-pattern rules. Equivalent graphs built through different insertion
    /// orders produce the same value.
    ///
    /// ```
    /// use synaptic_wiring::topology::SynapticGraph;
    /// use synaptic_wiring::types::{Polarity, SynapseDescriptor};
    ///
    /// let a = [
    ///     SynapseDescriptor {
    ///         source: 0,
    ///         target: 1,
    ///         weight: 0.5,
    ///         delay: 2,
    ///         polarity: Polarity::Excitatory,
    ///     },
    ///     SynapseDescriptor {
    ///         source: 1,
    ///         target: 0,
    ///         weight: 0.25,
    ///         delay: 1,
    ///         polarity: Polarity::Inhibitory,
    ///     },
    /// ];
    /// let mut b = a;
    /// b.swap(0, 1);
    ///
    /// let ga = SynapticGraph::from_descriptors(2, &a).unwrap();
    /// let gb = SynapticGraph::from_descriptors(2, &b).unwrap();
    /// assert_eq!(ga.topology_digest(), gb.topology_digest());
    /// ```
    #[must_use]
    pub fn topology_digest(&self) -> TopologyDigest {
        TopologyDigest::from_graph(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalEdge {
    source: u64,
    target: u64,
    delay: u16,
    polarity: u8,
    weight_bits: u32,
}

fn canonical_edges(graph: &SynapticGraph) -> Vec<CanonicalEdge> {
    let mut edges = Vec::with_capacity(graph.synapse_count());
    for src in 0..graph.neuron_count() {
        let source = src as u64;
        for (target, weight, delay, polarity) in graph.outgoing(src) {
            debug_assert!(
                weight.is_finite(),
                "valid graphs reject non-finite weights before digest"
            );
            edges.push(CanonicalEdge {
                source,
                target: u64::from(target),
                delay,
                polarity: polarity_tag(polarity),
                weight_bits: weight.to_bits(),
            });
        }
    }
    edges.sort_unstable();
    edges
}

fn hash_v1(neuron_count: u64, edges: &[CanonicalEdge]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(TOPOLOGY_DIGEST_DOMAIN.as_bytes());
    hasher.update([DOMAIN_TERMINATOR]);
    hasher.update(TOPOLOGY_DIGEST_SCHEMA_VERSION.to_le_bytes());
    hasher.update(neuron_count.to_le_bytes());
    hasher.update((edges.len() as u64).to_le_bytes());
    for edge in edges {
        hasher.update(edge.source.to_le_bytes());
        hasher.update(edge.target.to_le_bytes());
        hasher.update(edge.weight_bits.to_le_bytes());
        hasher.update(edge.delay.to_le_bytes());
        hasher.update([edge.polarity]);
    }
    hasher.finalize().into()
}

fn polarity_tag(polarity: Polarity) -> u8 {
    match polarity {
        Polarity::Excitatory => 0,
        Polarity::Inhibitory => 1,
    }
}

fn split_manifest(s: &str) -> Result<(&str, &str, &str), TopologyDigestParseError> {
    let (domain, rest) = s
        .split_once(':')
        .ok_or_else(TopologyDigestParseError::format)?;
    let (algorithm, hex) = rest
        .split_once(':')
        .ok_or_else(TopologyDigestParseError::format)?;
    Ok((domain, algorithm, hex))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode_sha256(hex: &str) -> Result<[u8; 32], TopologyDigestParseError> {
    if hex.len() != 64 {
        return Err(TopologyDigestParseError(format!(
            "digest hex must be 64 characters, got {}",
            hex.len()
        )));
    }
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, TopologyDigestParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(TopologyDigestParseError(format!(
            "invalid hex digit {:?}",
            b as char
        ))),
    }
}
