// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core types for `synaptic-wiring`.
//!
//! Shared scalar types, measurement units, and configuration structs used
//! across the topology, delay, and mesh modules.

use alloc::{format, string::String, vec::Vec};

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

// ── Scalar aliases ────────────────────────────────────────────────────────────

/// Unique identifier for a neuron within a mesh.
pub type NeuronId = u32;

/// Axonal propagation delay in discrete simulation ticks.
/// A delay of 0 means same-tick delivery (instantaneous).
pub type DelayTicks = u16;

// ── Polarity ──────────────────────────────────────────────────────────────────

/// Synaptic polarity following Dale's principle: a neuron's outgoing
/// synapses are either all excitatory or all inhibitory.
///
/// # References
///
/// Dale, H. H. (1935). *Pharmacology and Nerve-endings.* Proceedings of
/// the Royal Society of Medicine, 28(3), 319–332.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Polarity {
    /// Positive synaptic weight → depolarises the postsynaptic neuron.
    #[default]
    Excitatory,
    /// Negative synaptic weight → hyperpolarises the postsynaptic neuron.
    Inhibitory,
}

impl Polarity {
    /// Sign multiplier: +1.0 for excitatory, −1.0 for inhibitory.
    pub fn sign(self) -> f32 {
        match self {
            Self::Excitatory => 1.0,
            Self::Inhibitory => -1.0,
        }
    }
}

// ── Synapse descriptor ────────────────────────────────────────────────────────

/// A fully-described synaptic connection with weight, delay, and polarity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SynapseDescriptor {
    /// Source neuron.
    pub source: NeuronId,
    /// Target neuron.
    pub target: NeuronId,
    /// Absolute synaptic weight (always ≥ 0; sign comes from polarity).
    #[serde(deserialize_with = "deserialize_nonnegative_weight")]
    pub weight: f32,
    /// Axonal propagation delay in ticks.
    pub delay: DelayTicks,
    /// Excitatory or inhibitory.
    pub polarity: Polarity,
}

impl SynapseDescriptor {
    /// Effective signed weight: `weight × polarity.sign()`.
    ///
    /// `weight` is a non-negative magnitude. Invalid magnitudes are rejected
    /// by [`crate::topology::SynapticGraph::from_descriptors`] and by
    /// descriptor deserialization rather than being silently `abs()`-normalized
    /// here.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is negative, NaN, or infinite. Ordinary Rust struct
    /// literals can still construct those values; this check runs in release
    /// builds so a negative inhibitory magnitude cannot flip sign.
    pub fn effective_weight(&self) -> f32 {
        assert!(
            weight_magnitude_is_valid(self.weight),
            "synapse weight must be finite and non-negative, got {}",
            self.weight
        );
        self.weight * self.polarity.sign()
    }
}

/// `true` when `weight` is a finite magnitude, including IEEE `+0.0` and `-0.0`.
///
/// `-0.0 < 0.0` is false, so signed zero is accepted rather than rejected as
/// a negative magnitude.
pub(crate) fn weight_magnitude_is_valid(weight: f32) -> bool {
    weight.is_finite() && weight >= 0.0
}

/// Error text for an invalid descriptor magnitude (negative, NaN, or infinite).
pub(crate) fn invalid_weight_magnitude_msg(weight: f32) -> String {
    format!("synapse weight must be finite and non-negative, got {weight}")
}

/// CSR graphs store **signed** weights. Zero (including signed zero) is valid
/// for both polarities; a strictly positive inhibitory weight or a strictly
/// negative excitatory weight is not.
pub(crate) fn signed_weight_agrees_with_polarity(weight: f32, polarity: Polarity) -> bool {
    if !weight.is_finite() {
        return false;
    }
    match polarity {
        Polarity::Excitatory => weight >= 0.0,
        Polarity::Inhibitory => weight <= 0.0,
    }
}

/// Error text for a signed CSR weight that disagrees with its polarity.
pub(crate) fn invalid_signed_weight_polarity_msg(weight: f32, polarity: Polarity) -> String {
    if !weight.is_finite() {
        return format!("synapse weight must be finite, got {weight}");
    }
    match polarity {
        Polarity::Excitatory => {
            format!("excitatory synapse weight must be >= 0 (signed zero allowed), got {weight}")
        }
        Polarity::Inhibitory => {
            format!("inhibitory synapse weight must be <= 0 (signed zero allowed), got {weight}")
        }
    }
}

/// Rejects negative or non-finite weights so deserialized descriptors keep
/// the documented `weight >= 0` invariant that [`SynapseDescriptor::effective_weight`]
/// relies on to apply polarity correctly.
fn deserialize_nonnegative_weight<'de, D>(deserializer: D) -> core::result::Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let weight = f32::deserialize(deserializer)?;
    if !weight_magnitude_is_valid(weight) {
        return Err(DeError::custom(invalid_weight_magnitude_msg(weight)));
    }
    Ok(weight)
}

// ── Topology parameters ──────────────────────────────────────────────────────

/// Configuration for network topology generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Number of neurons in the network.
    pub neuron_count: usize,
    /// Fraction of neurons that are inhibitory (Dale's law).
    /// Typical cortical value: ~0.20 (80% excitatory, 20% inhibitory).
    pub inhibitory_fraction: f32,
    /// Maximum axonal delay in ticks.
    pub max_delay: DelayTicks,
    /// Minimum absolute weight to retain (sparsity pruning threshold).
    pub sparsity_threshold: f32,
    /// Connection probability model dispatched on by
    /// [`crate::topology::generate_from_config`].
    ///
    /// Defaults to `ConnectionModel::default()` (`Uniform { p: 0.05 }`), both
    /// in [`TopologyConfig::default()`] and when deserializing self-describing
    /// formats (for example JSON) that omit this field. Positional binary
    /// formats such as postcard do not load configurations written before
    /// this field existed.
    #[serde(default)]
    pub model: ConnectionModel,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            neuron_count: 2048,
            inhibitory_fraction: 0.20,
            max_delay: 20,
            sparsity_threshold: 0.01,
            model: ConnectionModel::default(),
        }
    }
}

// ── Connection probability models ─────────────────────────────────────────────

/// Strategy for determining connection probability between neuron pairs.
///
/// Marked `#[non_exhaustive]` so new variants are additive:
/// [`crate::topology::generate_from_config`] returns
/// [`crate::error::MeshError::UnsupportedModel`] for any variant that has no
/// generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConnectionModel {
    /// Erdős–Rényi: each pair connected independently with probability `p`.
    Uniform { p: f32 },
    /// Distance-dependent: probability decays with Euclidean distance.
    /// `p(d) = p_max × exp(−d / lambda)`.
    DistanceDependent { p_max: f32, lambda: f32 },
    /// Watts–Strogatz small-world: each neuron has `k` **outgoing** directed
    /// synapses to its nearest ring neighbours (`k/2` clockwise, `k/2`
    /// counterclockwise). `k` must be even. Each outgoing synapse is then
    /// rewired with probability `beta` to a different non-self target when
    /// one is available (a dense ring `k = n - 1` has none).
    SmallWorld { k: usize, beta: f32 },
    /// Barabási–Albert preferential attachment: start with `m0` nodes that
    /// are fully connected in both directions (`m0 * (m0 - 1)` directed
    /// synapses). Each new node attaches to exactly `m` distinct older
    /// nodes, and each attachment is stored as a reciprocal directed pair.
    ScaleFree { m0: usize, m: usize },
    /// Feed-forward layered: neurons arranged in layers, each layer
    /// fully connected to the next with given probability.
    Layered {
        layer_sizes: Vec<usize>,
        inter_layer_p: f32,
    },
}

impl Default for ConnectionModel {
    fn default() -> Self {
        Self::Uniform { p: 0.05 }
    }
}

// ── Delay model ───────────────────────────────────────────────────────────────

/// Strategy for assigning axonal delays to synapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelayModel {
    /// All synapses have the same fixed delay.
    Fixed { delay: DelayTicks },
    /// Delay is proportional to Euclidean distance between neurons.
    /// `delay = clamp(round(distance / speed), min_delay, max_delay)`.
    DistanceProportional {
        speed: f32,
        min_delay: DelayTicks,
        max_delay: DelayTicks,
    },
    /// Delay drawn uniformly from `[min_delay, max_delay]`.
    UniformRandom {
        min_delay: DelayTicks,
        max_delay: DelayTicks,
    },
}

impl Default for DelayModel {
    fn default() -> Self {
        Self::Fixed { delay: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_rejects_negative_weight() {
        let json = r#"{"source":0,"target":1,"weight":-0.5,"delay":1,"polarity":"Excitatory"}"#;
        assert!(serde_json::from_str::<SynapseDescriptor>(json).is_err());
    }

    #[test]
    fn deserialize_accepts_valid_weight() {
        let json = r#"{"source":0,"target":1,"weight":0.5,"delay":1,"polarity":"Inhibitory"}"#;
        let desc: SynapseDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(desc.weight, 0.5);
        assert!((desc.effective_weight() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn deserialize_accepts_zero_weight() {
        let json = r#"{"source":0,"target":1,"weight":0.0,"delay":1,"polarity":"Excitatory"}"#;
        let desc: SynapseDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(desc.weight, 0.0);
        assert_eq!(desc.effective_weight(), 0.0);
    }

    #[test]
    #[should_panic(expected = "synapse weight must be finite and non-negative")]
    fn effective_weight_panics_on_negative_inhibitory_magnitude() {
        let desc = SynapseDescriptor {
            source: 0,
            target: 1,
            weight: -0.5,
            delay: 1,
            polarity: Polarity::Inhibitory,
        };
        let _ = desc.effective_weight();
    }

    #[test]
    fn topology_config_deserializes_without_model_field() {
        // Configs serialized before `TopologyConfig.model` existed must still
        // load; `#[serde(default)]` fills in `ConnectionModel::default()`.
        let json = r#"{"neuron_count":128,"inhibitory_fraction":0.2,"max_delay":8,"sparsity_threshold":0.01}"#;
        let config: TopologyConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.neuron_count, 128);
        match config.model {
            ConnectionModel::Uniform { p } => assert_eq!(p, 0.05),
            ref other => panic!("expected Uniform {{ p: 0.05 }}, got {other:?}"),
        }
        assert!(matches!(
            TopologyConfig::default().model,
            ConnectionModel::Uniform { p } if p == 0.05
        ));
    }

    #[test]
    fn topology_config_deserializes_explicit_model() {
        let json = r#"{"neuron_count":64,"inhibitory_fraction":0.2,"max_delay":8,"sparsity_threshold":0.01,"model":{"SmallWorld":{"k":4,"beta":0.1}}}"#;
        let config: TopologyConfig = serde_json::from_str(json).unwrap();
        match config.model {
            ConnectionModel::SmallWorld { k, beta } => {
                assert_eq!(k, 4);
                assert_eq!(beta, 0.1);
            }
            ref other => panic!("expected SmallWorld, got {other:?}"),
        }
    }

    #[test]
    fn weight_magnitude_accepts_signed_zero_and_rejects_non_finite() {
        assert!(weight_magnitude_is_valid(0.0));
        assert!(weight_magnitude_is_valid(-0.0));
        assert!(weight_magnitude_is_valid(0.5));
        assert!(!weight_magnitude_is_valid(-0.5));
        assert!(!weight_magnitude_is_valid(f32::NAN));
        assert!(!weight_magnitude_is_valid(f32::INFINITY));
        assert!(!weight_magnitude_is_valid(f32::NEG_INFINITY));
    }

    #[test]
    fn signed_weight_polarity_accepts_signed_zero_for_both() {
        assert!(signed_weight_agrees_with_polarity(
            0.0,
            Polarity::Excitatory
        ));
        assert!(signed_weight_agrees_with_polarity(
            -0.0,
            Polarity::Excitatory
        ));
        assert!(signed_weight_agrees_with_polarity(
            0.0,
            Polarity::Inhibitory
        ));
        assert!(signed_weight_agrees_with_polarity(
            -0.0,
            Polarity::Inhibitory
        ));
        assert!(signed_weight_agrees_with_polarity(
            0.5,
            Polarity::Excitatory
        ));
        assert!(signed_weight_agrees_with_polarity(
            -0.5,
            Polarity::Inhibitory
        ));
        assert!(!signed_weight_agrees_with_polarity(
            -0.5,
            Polarity::Excitatory
        ));
        assert!(!signed_weight_agrees_with_polarity(
            0.5,
            Polarity::Inhibitory
        ));
        assert!(!signed_weight_agrees_with_polarity(
            f32::NAN,
            Polarity::Excitatory
        ));
        assert!(!signed_weight_agrees_with_polarity(
            f32::INFINITY,
            Polarity::Inhibitory
        ));
        assert!(!signed_weight_agrees_with_polarity(
            f32::NEG_INFINITY,
            Polarity::Excitatory
        ));
    }
}
