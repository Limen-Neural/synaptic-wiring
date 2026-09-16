// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for `synaptic-wiring`.
//!
//! Follows the `corinth-canal` pattern of a single unified error enum
//! using `thiserror` for ergonomic `Display` and `From` implementations.

use std::fmt;

/// Unified error type for synaptic-wiring operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    /// A required parameter was out of range or invalid.
    InvalidConfig(String),

    /// [`crate::RouterConfig`] or a [`crate::ChannelRouter`] checkpoint failed
    /// validation.
    ///
    /// `field` is the config or state field that failed (`"leak"`,
    /// `"neurons"`, `"baseline_weights"`, …) so constructor and serde paths
    /// can be compared by category without parsing the reason string.
    InvalidRouterConfig {
        /// Config or checkpoint field that failed validation.
        field: &'static str,
        /// Why the value is rejected.
        reason: String,
    },

    /// Neuron count mismatch between components.
    NeuronCountMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Attempted to access a neuron or synapse that doesn't exist.
    IndexOutOfBounds { index: usize, max: usize },

    /// Topology generation failed (e.g., impossible wiring constraints).
    TopologyError(String),

    /// Delay buffer overflow or misconfiguration.
    DelayError(String),

    /// A [`ChannelRouter`](crate::router::ChannelRouter) channel signal was
    /// NaN or ±infinity. `index` is the channel of the rejected sample;
    /// `context` names the public entry point (`"route signals"` or
    /// `"route_modulated signals"`).
    NonFiniteSignal { index: usize, context: String },

    /// A [`NeuromodState`](crate::router::NeuromodState) field was NaN or
    /// ±infinity. `field` is `"cortisol"`, `"dopamine"`, or `"serotonin"`.
    NonFiniteNeuromodulator { field: &'static str },

    /// A [`NeuromodState`](crate::router::NeuromodState) field was finite but
    /// outside the documented `[0, 1]` interval. Out-of-range values are
    /// rejected rather than silently clamped.
    OutOfRangeNeuromodulator { field: &'static str, value: f32 },
}

impl MeshError {
    pub(crate) fn invalid_router_config(field: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidRouterConfig {
            field,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            MeshError::InvalidRouterConfig { field, reason } => {
                write!(f, "invalid router config ({field}): {reason}")
            }
            MeshError::NeuronCountMismatch {
                expected,
                got,
                context,
            } => write!(
                f,
                "neuron count mismatch in {context}: expected {expected}, got {got}"
            ),
            MeshError::IndexOutOfBounds { index, max } => {
                write!(f, "index {index} out of bounds (max {max})")
            }
            MeshError::TopologyError(msg) => write!(f, "topology error: {msg}"),
            MeshError::DelayError(msg) => write!(f, "delay error: {msg}"),
            MeshError::NonFiniteSignal { index, context } => {
                write!(f, "non-finite signal in {context}[{index}]")
            }
            MeshError::NonFiniteNeuromodulator { field } => {
                write!(f, "non-finite neuromodulator field {field}")
            }
            MeshError::OutOfRangeNeuromodulator { field, value } => {
                write!(
                    f,
                    "neuromodulator field {field} must be in [0, 1], got {value}"
                )
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, MeshError>;
