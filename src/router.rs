// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic multi-channel SNN router with neuromodulatory adaptation.
//!
//! A domain-agnostic SNN router that integrates signal pulses across a bank
//! of neuromodulatory neurons to produce a sparse routing mask.
//!
//! The router is generic over channel count and supports adaptive
//! neuromodulatory routing — channels strengthen with use (dopamine-gated)
//! and weaken when idle (use-it-or-lose-it plasticity).
//!
//! This module is **optional and self-contained**: it neither uses nor is
//! used by [`SynapticMesh`](crate::mesh::SynapticMesh). Reach for it when you
//! need to pick a few active channels out of many inputs; ignore it entirely
//! if you only need wiring, topology, and delays.

use crate::error::{MeshError, Result};
use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

/// Integration timesteps per routing decision (more → more stable).
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / `ROUTING_TIMESTEPS`) to activate a channel.
const MIN_FIRE_RATE: f32 = 0.1875;

/// Largest channel count [`ChannelRouter`] will allocate.
///
/// The router stores a dense `N × N` weight matrix plus a baseline copy, so
/// this cap is checked in [`RouterConfig::validate`] *before* any of those
/// vectors are allocated.
pub const MAX_ROUTER_CHANNELS: usize = 1024;

/// Largest `routing_timesteps` accepted by [`RouterConfig::validate`].
///
/// The routing loop is `O(timesteps × channel_count)`; this cap rejects
/// values that would hang a restore/route on untrusted input.
pub const MAX_ROUTING_TIMESTEPS: usize = 4096;

/// Neuromodulatory Integrative Fixed-threshold (NIF) neuron.
///
/// This is a **router-internal integration primitive** for [`ChannelRouter`],
/// not a general-purpose neuron model — canonical neuron models (LIF,
/// Izhikevich, Hodgkin-Huxley, GIF, FitzHugh-Nagumo, Lapicque) live in the
/// separate `neuromod` crate, which `synaptic-wiring` intentionally does not
/// depend on. See the crate-level docs for the full boundary rationale.
///
/// $V_{t+1} = V_t + (G \cdot I_{syn}) - \lambda(V_t - V_{rest})$
/// where $G$ is the modulation gain and $\lambda$ is the leak rate.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct NeuromodNeuron {
    /// Current membrane potential.
    pub v: f32,
    /// Resting membrane potential.
    pub v_rest: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Passive leak rate per timestep.
    pub leak: f32,
    /// Firing threshold.
    pub threshold: f32,

    /// Neuromodulatory gain (scales incoming stimulus).
    pub gain: f32,

    /// Synaptic weights — one per input channel.
    pub weights: Vec<f32>,
    /// Whether the neuron fired in the last timestep.
    pub last_spike: bool,
}

impl Default for NeuromodNeuron {
    fn default() -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            leak: 0.12,
            threshold: 0.25,
            gain: 1.0,
            weights: Vec::new(),
            last_spike: false,
        }
    }
}

impl NeuromodNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance neuron dynamics by one timestep.
    ///
    /// The `stimulus` is scaled by the neuron's current `gain`.
    pub fn integrate(&mut self, stimulus: f32) {
        // Apply modulated integration
        self.v += stimulus * self.gain;
        // Apply leak towards resting potential
        self.v -= (self.v - self.v_rest) * self.leak;
    }

    /// Check if the neuron spikes. Resets V on fire.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.v >= self.threshold {
            let peak = self.v;
            self.v = self.v_reset;
            self.last_spike = true;
            return Some(peak);
        }
        self.last_spike = false;
        None
    }

    /// Update the modulation gain.
    pub fn set_gain(&mut self, new_gain: f32) {
        self.gain = new_gain;
    }
}

/// Configuration for a generic channel router.
///
/// Direct struct literals can hold out-of-range values; they are rejected by
/// [`RouterConfig::validate`], [`ChannelRouter::try_with_config`], and by
/// deserialization of both this type and [`ChannelRouter`].
///
/// # Valid ranges
///
/// | Field | Constraint |
/// |-------|------------|
/// | `channel_count` | `1..=`[`MAX_ROUTER_CHANNELS`] |
/// | `routing_timesteps` | `1..=`[`MAX_ROUTING_TIMESTEPS`] |
/// | `self_weight`, `cross_weight`, `threshold` | finite; **signed weights are allowed** |
/// | `leak`, `min_fire_rate`, `plasticity_decay`, `plasticity_speed`, `fatigue_accumulation`, `fatigue_recovery` | finite, in `0.0..=1.0` |
/// | `plasticity_potentiate` | finite, `>= 0` (scale factor, not a probability) |
///
/// `cross_weight` is typically negative (lateral inhibition). That sign is
/// allowed and intended; it is not required. Positive self-weights are the
/// usual self-affinity pattern, but a signed self-weight is also accepted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RouterConfig {
    /// Number of input/output channels (`1..=`[`MAX_ROUTER_CHANNELS`]).
    pub channel_count: usize,
    /// Self-affinity weight (diagonal of the dense channel matrix).
    ///
    /// Must be finite. Signed values are allowed.
    pub self_weight: f32,
    /// Cross-channel weight (off-diagonal).
    ///
    /// Must be finite. Signed values are allowed; **negative values are the
    /// intended lateral-inhibition pattern** and are not rejected.
    pub cross_weight: f32,
    /// Firing threshold for neuromodulatory neurons.
    ///
    /// Must be finite. Typically positive; zero and negative values are
    /// accepted here (modulated routing later clamps the *effective*
    /// threshold).
    pub threshold: f32,
    /// Passive leak rate per timestep.
    ///
    /// Fraction of `(V − V_rest)` removed each tick. Valid range: `0.0..=1.0`.
    /// `0.0` = no leak; `1.0` = membrane snaps to rest in one tick.
    pub leak: f32,
    /// Integration timesteps per routing decision.
    ///
    /// Must be in `1..=`[`MAX_ROUTING_TIMESTEPS`]. More timesteps → more
    /// stable firing-rate estimates.
    pub routing_timesteps: usize,
    /// Minimum firing rate to activate a channel.
    ///
    /// Fraction of `routing_timesteps` that must spike. Valid range: `0.0..=1.0`.
    pub min_fire_rate: f32,
    /// Weight decay rate for inactive channels (use-it-or-lose-it).
    ///
    /// Mix toward baseline: `current + (baseline − current) * decay`.
    /// Valid range: `0.0..=1.0`.
    pub plasticity_decay: f32,
    /// Weight potentiation rate for active channels (dopamine-gated).
    ///
    /// Non-negative finite scale factor, not a probability. The amplified
    /// target is `baseline * (1 + potentiate * (1 + dopamine))`.
    pub plasticity_potentiate: f32,
    /// Smoothing factor for active-channel weight potentiation
    /// (`0.0` = no change, `1.0` = snap to amplified target). Tunable so
    /// callers can trade off adaptation speed vs. numerical stability.
    /// Valid range: `0.0..=1.0`.
    pub plasticity_speed: f32,
    /// Fatigue accumulation per activation.
    ///
    /// Added to per-channel fatigue (itself in `0.0..=1.0`). Valid range:
    /// `0.0..=1.0`.
    pub fatigue_accumulation: f32,
    /// Fatigue recovery per idle routing decision.
    ///
    /// Subtracted from per-channel fatigue. Valid range: `0.0..=1.0`.
    pub fatigue_recovery: f32,
}

#[derive(Deserialize)]
struct RawRouterConfig {
    channel_count: usize,
    self_weight: f32,
    cross_weight: f32,
    threshold: f32,
    leak: f32,
    routing_timesteps: usize,
    min_fire_rate: f32,
    #[serde(default = "default_plasticity_decay")]
    plasticity_decay: f32,
    #[serde(default = "default_plasticity_potentiate")]
    plasticity_potentiate: f32,
    #[serde(default = "default_plasticity_speed")]
    plasticity_speed: f32,
    #[serde(default = "default_fatigue_accumulation")]
    fatigue_accumulation: f32,
    #[serde(default = "default_fatigue_recovery")]
    fatigue_recovery: f32,
}

impl RawRouterConfig {
    fn into_config(self) -> Result<RouterConfig> {
        let config = RouterConfig {
            channel_count: self.channel_count,
            self_weight: self.self_weight,
            cross_weight: self.cross_weight,
            threshold: self.threshold,
            leak: self.leak,
            routing_timesteps: self.routing_timesteps,
            min_fire_rate: self.min_fire_rate,
            plasticity_decay: self.plasticity_decay,
            plasticity_potentiate: self.plasticity_potentiate,
            plasticity_speed: self.plasticity_speed,
            fatigue_accumulation: self.fatigue_accumulation,
            fatigue_recovery: self.fatigue_recovery,
        };
        config.validate()?;
        Ok(config)
    }
}

impl<'de> Deserialize<'de> for RouterConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawRouterConfig::deserialize(deserializer)?
            .into_config()
            .map_err(DeError::custom)
    }
}

fn default_plasticity_decay() -> f32 {
    0.02
}
fn default_plasticity_potentiate() -> f32 {
    0.05
}
fn default_plasticity_speed() -> f32 {
    0.1
}
fn default_fatigue_accumulation() -> f32 {
    0.15
}
fn default_fatigue_recovery() -> f32 {
    0.05
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            channel_count: 3,
            self_weight: 0.9,
            cross_weight: -0.15,
            threshold: 0.22,
            leak: 0.12,
            routing_timesteps: ROUTING_TIMESTEPS,
            min_fire_rate: MIN_FIRE_RATE,
            plasticity_decay: 0.02,
            plasticity_potentiate: 0.05,
            plasticity_speed: 0.1,
            fatigue_accumulation: 0.15,
            fatigue_recovery: 0.05,
        }
    }
}

impl RouterConfig {
    /// Check that every field is in its documented range.
    ///
    /// This is the single validation path used by
    /// [`ChannelRouter::try_with_config`], [`ChannelRouter::with_config`],
    /// and by `Deserialize` for both [`RouterConfig`] and [`ChannelRouter`].
    /// It does not allocate.
    pub fn validate(&self) -> Result<()> {
        if self.channel_count == 0 || self.channel_count > MAX_ROUTER_CHANNELS {
            return Err(MeshError::invalid_router_config(
                "channel_count",
                format!(
                    "must be in 1..={MAX_ROUTER_CHANNELS}, got {}",
                    self.channel_count
                ),
            ));
        }
        if self.routing_timesteps == 0 || self.routing_timesteps > MAX_ROUTING_TIMESTEPS {
            return Err(MeshError::invalid_router_config(
                "routing_timesteps",
                format!(
                    "must be in 1..={MAX_ROUTING_TIMESTEPS}, got {}",
                    self.routing_timesteps
                ),
            ));
        }
        require_finite("self_weight", self.self_weight)?;
        require_finite("cross_weight", self.cross_weight)?;
        require_finite("threshold", self.threshold)?;
        require_unit_interval("leak", self.leak)?;
        require_unit_interval("min_fire_rate", self.min_fire_rate)?;
        require_unit_interval("plasticity_decay", self.plasticity_decay)?;
        require_non_negative("plasticity_potentiate", self.plasticity_potentiate)?;
        require_unit_interval("plasticity_speed", self.plasticity_speed)?;
        require_unit_interval("fatigue_accumulation", self.fatigue_accumulation)?;
        require_unit_interval("fatigue_recovery", self.fatigue_recovery)?;
        Ok(())
    }
}

fn require_finite(field: &'static str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite, got {value}"),
        ))
    }
}

fn require_unit_interval(field: &'static str, value: f32) -> Result<()> {
    require_finite(field, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite and in 0.0..=1.0, got {value}"),
        ))
    }
}

fn require_non_negative(field: &'static str, value: f32) -> Result<()> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite and >= 0, got {value}"),
        ))
    }
}

/// Inclusive unit interval required of every [`NeuromodState`] field.
const NEUROMOD_MIN: f32 = 0.0;
const NEUROMOD_MAX: f32 = 1.0;

/// Neuromodulatory state for adaptive routing.
///
/// Cortisol (stress) increases resistance — channels become harder to activate.
/// Dopamine (reward) increases conductance — channels become easier to activate.
/// Serotonin (patience) reduces persistence — faster decay of activation.
///
/// Every field must be **finite** and in `[0.0, 1.0]` (IEEE signed zero is
/// accepted). [`ChannelRouter::route_modulated`] rejects NaN, ±infinity, and
/// finite out-of-range values **before** any routing timestep; they are not
/// sanitized to zero and not silently clamped. Derived thresholds and leaks
/// may still clamp internally — that is a stability bound on the neuron
/// parameters, not an ingress policy for these fields.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NeuromodState {
    /// Stress level (0.0 = calm, 1.0 = max stress).
    /// Raises firing thresholds (always — even on fresh / zero-fatigue channels)
    /// and additionally amplifies the effect of accumulated fatigue.
    pub cortisol: f32,
    /// Reward level (0.0 = no reward, 1.0 = high reward).
    /// Lowers thresholds, strengthens active synapses, counteracts fatigue.
    pub dopamine: f32,
    /// Patience/risk-aversion level (0.0 = impulsive, 1.0 = patient).
    /// Increases leak/decay rate, making activations less persistent.
    pub serotonin: f32,
}

impl NeuromodState {
    /// Create a balanced neuromodulatory state (no modulation).
    pub fn balanced() -> Self {
        Self {
            cortisol: 0.0,
            dopamine: 0.0,
            serotonin: 0.0,
        }
    }

    /// Create a stressed state (high cortisol).
    pub fn stressed() -> Self {
        Self {
            cortisol: 0.8,
            dopamine: 0.0,
            serotonin: 0.0,
        }
    }

    /// Create a rewarded state (high dopamine).
    pub fn rewarded() -> Self {
        Self {
            cortisol: 0.0,
            dopamine: 0.8,
            serotonin: 0.0,
        }
    }

    /// Reject non-finite fields and values outside `[0, 1]`.
    ///
    /// Finite out-of-range values are **rejected**, not clamped, so a caller
    /// cannot smuggle `cortisol = 2.0` (or a negative field) past ingress and
    /// rely on the downstream threshold/leak clamp to hide it.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("cortisol", self.cortisol),
            ("dopamine", self.dopamine),
            ("serotonin", self.serotonin),
        ] {
            if !value.is_finite() {
                return Err(MeshError::NonFiniteNeuromodulator { field });
            }
            if !(NEUROMOD_MIN..=NEUROMOD_MAX).contains(&value) {
                return Err(MeshError::OutOfRangeNeuromodulator { field, value });
            }
        }
        Ok(())
    }
}

/// Sparse activation decision from the SNN router.
#[derive(Debug, Clone, Default)]
pub struct RoutingDecision {
    /// Indices of the channels that were activated.
    pub active_channels: Vec<usize>,
    /// Per-channel firing rates (for diagnostics and feedback).
    pub firing_rates: Vec<f32>,
    /// Raw input signals fed into the router.
    pub input_signals: Vec<f32>,
}

impl RoutingDecision {
    pub fn is_active(&self, channel: usize) -> bool {
        self.active_channels.contains(&channel)
    }

    /// True when no channel was activated.
    pub fn is_empty(&self) -> bool {
        self.active_channels.is_empty()
    }
}

/// Shared ingress checks for [`ChannelRouter::route`] and
/// [`ChannelRouter::route_modulated`]. Length, signal finiteness, and
/// neuromodulator validity are all decided before any router field changes.
fn validate_route_ingress(
    signals: &[f32],
    mods: &NeuromodState,
    expected_len: usize,
    error_context: &str,
) -> Result<()> {
    if signals.len() != expected_len {
        return Err(MeshError::NeuronCountMismatch {
            expected: expected_len,
            got: signals.len(),
            context: error_context.into(),
        });
    }
    for (index, &value) in signals.iter().enumerate() {
        if !value.is_finite() {
            return Err(MeshError::NonFiniteSignal {
                index,
                context: error_context.into(),
            });
        }
    }
    mods.validate()
}

/// Generic multi-channel SNN Router.
///
/// Integrates multi-channel signals over `ROUTING_TIMESTEPS` to produce
/// a sparse activation mask. The number of channels is configurable at
/// construction time via [`RouterConfig`].
///
/// Supports adaptive neuromodulatory routing via [`ChannelRouter::route_modulated`]:
/// - Channels strengthen with use (dopamine-gated potentiation)
/// - Channels weaken when idle (use-it-or-lose-it decay)
/// - Fatigue accumulates with activation, cortisol amplifies it
/// - The router naturally seeks the least-resistance pathway
///
/// Deserialization re-runs [`RouterConfig::validate`] and requires neuron,
/// weight, fatigue, and baseline-weight vector shapes to match
/// `channel_count`. Missing `config` / `channel_fatigue` / `baseline_weights`
/// (legacy snapshots) are filled from the neuron bank rather than rejected.
///
/// Ingress is atomic: non-finite signals or invalid [`NeuromodState`] values
/// return a structured [`MeshError`] and leave neurons, fatigue, adaptive
/// weights, and `total_routes` unchanged.
#[derive(Clone, Debug, Serialize)]
pub struct ChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    config: RouterConfig,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
    /// Per-channel fatigue (0.0 = fresh, 1.0 = fully exhausted).
    pub channel_fatigue: Vec<f32>,
    /// Baseline weights for plasticity decay reference.
    baseline_weights: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum LegacyField<T> {
    #[default]
    Omitted,
    Null,
    Value(T),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for LegacyField<T> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct LegacyFieldVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for LegacyFieldVisitor<T> {
            type Value = LegacyField<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a value or null")
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(LegacyField::Null)
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(LegacyField::Null)
            }

            fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                T::deserialize(deserializer).map(LegacyField::Value)
            }
        }

        deserializer.deserialize_option(LegacyFieldVisitor(std::marker::PhantomData))
    }
}

#[derive(Deserialize)]
struct RawChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    #[serde(default)]
    config: LegacyField<RouterConfig>,
    #[serde(default)]
    total_routes: u64,
    #[serde(default)]
    channel_fatigue: LegacyField<Vec<f32>>,
    #[serde(default)]
    baseline_weights: LegacyField<Vec<Vec<f32>>>,
}

impl RawChannelRouter {
    fn into_router(self) -> Result<ChannelRouter> {
        let n_neurons = self.neurons.len();
        let config = match self.config {
            LegacyField::Omitted => RouterConfig {
                channel_count: n_neurons,
                ..RouterConfig::default()
            },
            LegacyField::Null => {
                return Err(MeshError::invalid_router_config(
                    "config",
                    "null value is not allowed; omit field for legacy format",
                ));
            }
            LegacyField::Value(config) => config,
        };
        // Present configs were already validated by `RouterConfig`'s
        // `Deserialize`. Re-run the same path so a `Raw` built in tests, or a
        // future constructor that skips serde, cannot bypass it.
        config.validate()?;

        let n = config.channel_count;
        validate_neuron_bank(n, &self.neurons)?;
        let channel_fatigue = match self.channel_fatigue {
            LegacyField::Omitted => vec![0.0; n],
            LegacyField::Null => {
                return Err(MeshError::invalid_router_config(
                    "channel_fatigue",
                    "null value is not allowed; omit field for legacy format",
                ));
            }
            LegacyField::Value(fatigue) => fatigue,
        };
        let baseline_weights = match self.baseline_weights {
            LegacyField::Omitted => self.neurons.iter().map(|neu| neu.weights.clone()).collect(),
            LegacyField::Null => {
                return Err(MeshError::invalid_router_config(
                    "baseline_weights",
                    "null value is not allowed; omit field for legacy format",
                ));
            }
            LegacyField::Value(weights) => weights,
        };
        validate_fatigue_and_baseline(n, &channel_fatigue, &baseline_weights)?;
        Ok(ChannelRouter {
            neurons: self.neurons,
            config,
            total_routes: self.total_routes,
            channel_fatigue,
            baseline_weights,
        })
    }
}

/// Neuron bank must be length `n`, each row length `n`, with finite
/// membrane / gain / weight parameters. Called before cloning a missing
/// baseline table so malformed snapshots cannot double peak memory.
fn validate_neuron_bank(n: usize, neurons: &[NeuromodNeuron]) -> Result<()> {
    if neurons.len() != n {
        return Err(MeshError::invalid_router_config(
            "neurons",
            format!("length {} does not match channel_count {n}", neurons.len()),
        ));
    }
    for (i, neu) in neurons.iter().enumerate() {
        require_finite_named("v", neu.v, i)?;
        require_finite_named("v_rest", neu.v_rest, i)?;
        require_finite_named("v_reset", neu.v_reset, i)?;
        require_finite_named("leak", neu.leak, i)?;
        require_finite_named("threshold", neu.threshold, i)?;
        require_finite_named("gain", neu.gain, i)?;
        if neu.weights.len() != n {
            return Err(MeshError::invalid_router_config(
                "weights",
                format!(
                    "neuron {i} weights length {} does not match channel_count {n}",
                    neu.weights.len()
                ),
            ));
        }
        if neu.weights.iter().any(|w| !w.is_finite()) {
            return Err(MeshError::invalid_router_config(
                "weights",
                format!("neuron {i} weights contain a non-finite value"),
            ));
        }
    }
    Ok(())
}

fn require_finite_named(field: &'static str, value: f32, neuron: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("neuron {neuron} {field} must be finite, got {value}"),
        ))
    }
}

fn validate_fatigue_and_baseline(
    n: usize,
    channel_fatigue: &[f32],
    baseline_weights: &[Vec<f32>],
) -> Result<()> {
    if channel_fatigue.len() != n {
        return Err(MeshError::invalid_router_config(
            "channel_fatigue",
            format!(
                "length {} does not match channel_count {n}",
                channel_fatigue.len()
            ),
        ));
    }
    for (i, &fatigue) in channel_fatigue.iter().enumerate() {
        if !fatigue.is_finite() || !(0.0..=1.0).contains(&fatigue) {
            return Err(MeshError::invalid_router_config(
                "channel_fatigue",
                format!("index {i} must be finite and in 0.0..=1.0, got {fatigue}"),
            ));
        }
    }
    if baseline_weights.len() != n {
        return Err(MeshError::invalid_router_config(
            "baseline_weights",
            format!(
                "length {} does not match channel_count {n}",
                baseline_weights.len()
            ),
        ));
    }
    for (i, row) in baseline_weights.iter().enumerate() {
        if row.len() != n {
            return Err(MeshError::invalid_router_config(
                "baseline_weights",
                format!(
                    "row {i} length {} does not match channel_count {n}",
                    row.len()
                ),
            ));
        }
        if row.iter().any(|w| !w.is_finite()) {
            return Err(MeshError::invalid_router_config(
                "baseline_weights",
                format!("row {i} contains a non-finite value"),
            ));
        }
    }
    Ok(())
}

impl<'de> Deserialize<'de> for ChannelRouter {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawChannelRouter::deserialize(deserializer)?
            .into_router()
            .map_err(DeError::custom)
    }
}

impl Default for ChannelRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelRouter {
    /// Create a new router with default configuration (3 channels).
    pub fn new() -> Self {
        Self::try_with_config(RouterConfig::default()).expect("default RouterConfig is valid")
    }

    /// Create a new router with a custom configuration.
    ///
    /// Invalid configs panic. Prefer [`ChannelRouter::try_with_config`] when
    /// the caller needs a recoverable error.
    ///
    /// # Panics
    ///
    /// Panics if [`RouterConfig::validate`] fails (zero `routing_timesteps`,
    /// out-of-range channel count, non-finite parameters, or rates outside
    /// their documented interval).
    pub fn with_config(config: RouterConfig) -> Self {
        Self::try_with_config(config).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`ChannelRouter::with_config`].
    ///
    /// Validates `config` before allocating the dense weight tables, so an
    /// oversized `channel_count` cannot exhaust memory on the way to an error.
    pub fn try_with_config(config: RouterConfig) -> Result<Self> {
        config.validate()?;
        let n = config.channel_count;
        let neurons: Vec<NeuromodNeuron> = (0..n)
            .map(|i| {
                let mut neu = NeuromodNeuron::new();
                // Strong self-affinity; weak cross-channel inhibition.
                neu.weights = vec![config.cross_weight; n];
                neu.weights[i] = config.self_weight;
                neu.threshold = config.threshold;
                neu.leak = config.leak;
                neu
            })
            .collect();

        let baseline_weights = neurons.iter().map(|neu| neu.weights.clone()).collect();

        Ok(Self {
            neurons,
            config,
            total_routes: 0,
            channel_fatigue: vec![0.0; n],
            baseline_weights,
        })
    }

    /// Route raw channel signals through the SNN (non-modulated).
    ///
    /// `signals` must have length equal to `config.channel_count`. Every
    /// sample must be **finite**; NaN and ±infinity are rejected with
    /// [`MeshError::NonFiniteSignal`] naming the channel index. Finite
    /// **signed** samples are accepted and participate in the weighted sum
    /// (a negative input inhibits).
    ///
    /// Validation runs **before** any neuron, fatigue, weight, or
    /// `total_routes` mutation, including the deserialized-state self-heal.
    ///
    /// Backward-compatible thin wrapper around [`ChannelRouter::route_modulated`]. The error
    /// context reported on a signal-length mismatch is `"route signals"`,
    /// matching the original pre-neuromodulation API — callers using this
    /// public method see the same error message they did before, even though
    /// the implementation now delegates to `route_modulated` internally.
    pub fn route<S: AsRef<[f32]>>(&mut self, signals: S) -> Result<RoutingDecision> {
        self.route_modulated_with_context(signals, &NeuromodState::balanced(), "route signals")
    }

    /// Route with neuromodulatory modulation.
    ///
    /// Seeks the least-resistance pathway by dynamically adjusting thresholds
    /// and applying use-it-or-lose-it plasticity:
    /// - Cortisol raises effective thresholds (resistance)
    /// - Dopamine lowers thresholds and strengthens active channels (conductance)
    /// - Serotonin increases leak (reduces persistence)
    /// - Inactive channels decay toward baseline weights
    /// - Active channels potentiate (dopamine-gated)
    ///
    /// `signals` follow the same finite/signed rules as [`ChannelRouter::route`].
    /// `mods` is validated via [`NeuromodState::validate`] before any routing
    /// timestep: non-finite fields become [`MeshError::NonFiniteNeuromodulator`],
    /// and finite values outside `[0, 1]` become
    /// [`MeshError::OutOfRangeNeuromodulator`]. Both this method and
    /// [`ChannelRouter::route`] share that ingress check.
    pub fn route_modulated<S: AsRef<[f32]>>(
        &mut self,
        signals: S,
        mods: &NeuromodState,
    ) -> Result<RoutingDecision> {
        self.route_modulated_with_context(signals, mods, "route_modulated signals")
    }

    /// Internal routing implementation. The `error_context` argument is the
    /// string used in length-mismatch and non-finite-signal errors so each
    /// public entry point can report the method the caller actually invoked.
    fn route_modulated_with_context<S: AsRef<[f32]>>(
        &mut self,
        signals: S,
        mods: &NeuromodState,
        error_context: &str,
    ) -> Result<RoutingDecision> {
        let signals = signals.as_ref();
        let n = self.config.channel_count;
        validate_route_ingress(signals, mods, n, error_context)?;

        // Self-heal: keep neuromod state vectors aligned with the current
        // channel count if a caller mutated the public `channel_fatigue`
        // field (serde restore already rejects inconsistent shapes).
        self.ensure_neuromod_state_synced();
        let (effective_thresholds, effective_leaks) = self.compute_effective_params(mods);

        // Reset membrane potentials for a fresh routing decision.
        for neu in &mut self.neurons {
            neu.v = 0.0;
        }

        let timesteps = self.config.routing_timesteps;
        let min_rate = self.config.min_fire_rate;
        let spike_counts =
            self.integrate_signals(signals, &effective_thresholds, &effective_leaks, timesteps);

        let mut firing_rates = vec![0.0f32; n];
        let mut active_channels = Vec::new();
        for i in 0..n {
            firing_rates[i] = spike_counts[i] as f32 / timesteps as f32;
            if firing_rates[i] >= min_rate {
                active_channels.push(i);
            }
        }

        // Apply use-it-or-lose-it plasticity.
        self.apply_plasticity(&active_channels, mods);

        self.total_routes += 1;
        Ok(RoutingDecision {
            active_channels,
            firing_rates,
            input_signals: signals.to_vec(),
        })
    }

    /// Run the per-timestep integration loop and return spike counts per channel.
    ///
    /// For each timestep, every neuron computes its stimulus from the signal
    /// vector, applies the serotonin-modulated leak, integrates the stimulus,
    /// and checks against the dopamine/cortisol-modulated threshold. Neuron
    /// membrane potentials (`v`) are updated in place; per-channel spike counts
    /// are incremented on each fire.
    fn integrate_signals(
        &mut self,
        signals: &[f32],
        effective_thresholds: &[f32],
        effective_leaks: &[f32],
        timesteps: usize,
    ) -> Vec<u32> {
        let n = self.config.channel_count;
        let mut spike_counts = vec![0u32; n];
        for _ in 0..timesteps {
            // Iterate only up to n (channel_count) so we never index beyond
            // spike_counts, effective_thresholds, or effective_leaks. If
            // neurons.len() > n (malformed state), the extra neurons are
            // skipped. If neurons.len() < n, those channels produce no spikes.
            for (i, neu) in self.neurons.iter_mut().enumerate().take(n) {
                debug_assert_eq!(
                    neu.weights.len(),
                    signals.len(),
                    "Neuron weights length mismatch"
                );
                let stimulus: f32 = signals
                    .iter()
                    .zip(neu.weights.iter())
                    .map(|(sig, w)| sig * w)
                    .sum();
                neu.leak = effective_leaks[i];
                neu.integrate(stimulus);
                neu.threshold = effective_thresholds[i];
                if neu.check_fire().is_some() {
                    spike_counts[i] += 1;
                }
            }
        }
        spike_counts
    }

    /// Lazily (re)initialize `channel_fatigue`, `baseline_weights`, and
    /// individual neuron weight vectors so that their lengths match the
    /// current channel count. Called at the top of `route_modulated` so
    /// that mutating the public `channel_fatigue` field (or any other
    /// post-construction length drift) cannot trigger out-of-bounds
    /// indexing. Deserialization already rejects inconsistent shapes.
    ///
    /// Repairs three layers of state:
    /// 1. `channel_fatigue` — resized to `n` (zero-filled).
    /// 2. Each neuron's `weights` vector — truncated or zero-padded to `n`
    ///    so that `integrate_signals`, `apply_plasticity`, and
    ///    `apply_feedback` can safely index `neurons[i].weights[j]`.
    /// 3. `baseline_weights` — rebuilt from the (now-repaired) neuron
    ///    weights if any row is the wrong size.
    fn ensure_neuromod_state_synced(&mut self) {
        let n = self.config.channel_count;

        // 1. Repair channel_fatigue length.
        if self.channel_fatigue.len() != n {
            self.channel_fatigue.resize(n, 0.0);
        }

        // 2. Repair individual neuron weight vectors.
        //    A deserialized neuron may have weights.len() != n (e.g.
        //    serialized with an older channel_count). Truncate or
        //    zero-pad each to exactly n.
        //
        //    Zero-padding gives new channels weight 0.0, which differs
        //    from the constructor's self_weight/cross_weight pattern.
        //    This degrades routing on new channels until weights are
        //    explicitly set — acceptable as a self-heal path (the
        //    alternative is a panic). Callers who need proper weights
        //    after a channel_count change should reconstruct via
        //    with_config rather than relying on this repair.
        let mut weights_repaired = false;
        for neu in &mut self.neurons {
            if neu.weights.len() != n {
                weights_repaired = true;
                neu.weights.resize(n, 0.0);
            }
        }

        // 3. Repair baseline_weights (rebuild from neuron weights if
        //    any row is the wrong size, or if weights were repaired
        //    in step 2).
        let baseline_ok = !weights_repaired
            && self.baseline_weights.len() == n
            && self.baseline_weights.iter().all(|row| row.len() == n);
        if !baseline_ok {
            self.baseline_weights = self.neurons.iter().map(|neu| neu.weights.clone()).collect();
        }
    }

    /// Per-channel effective thresholds and leaks under neuromodulation.
    ///
    /// - Cortisol: baseline stress component (always raises threshold, even on
    ///   fresh channels) + fatigue amplification (further raises it on
    ///   fatigued channels).
    /// - Dopamine: lowers threshold (conductance).
    /// - Serotonin: raises leak (faster decay).
    fn compute_effective_params(&self, mods: &NeuromodState) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.channel_count;
        let mut thresholds = vec![0.0f32; n];
        let mut leaks = vec![0.0f32; n];
        for i in 0..n {
            let baseline_stress = 1.0 + mods.cortisol * 0.5;
            let fatigue_amplification = 1.0 + mods.cortisol * self.channel_fatigue[i];
            let fatigue_factor = baseline_stress * fatigue_amplification;
            let dopamine_factor = 1.0 - mods.dopamine * 0.5;
            thresholds[i] =
                (self.config.threshold * fatigue_factor * dopamine_factor).clamp(0.05, 2.0);
            leaks[i] = (self.config.leak * (1.0 + mods.serotonin)).clamp(0.0, 1.0);
        }
        (thresholds, leaks)
    }

    /// Apply use-it-or-lose-it plasticity.
    ///
    /// - Active channels: strengthen (dopamine-gated), accumulate fatigue
    /// - Inactive channels: decay toward baseline weights, recover fatigue
    ///
    /// Weights are clamped to a unified range that covers both the
    /// self-affinity range ([0.1, 2.0]) and the cross-channel range
    /// ([-1.0, 1.5]) used by `apply_feedback`. This prevents the
    /// use-it-or-lose-it decay from drifting into a regime where the
    /// downstream `apply_feedback` clamp would suddenly snap a weight.
    fn apply_plasticity(&mut self, active_channels: &[usize], mods: &NeuromodState) {
        let n = self.config.channel_count;
        // Guard against malformed deserialized state: check both outer lengths
        // AND inner row lengths of neurons and baseline_weights, matching the
        // belt-and-suspenders pattern in sync_baseline_after_feedback.
        debug_assert!(
            self.neurons.len() >= n
                && self.neurons.iter().all(|neu| neu.weights.len() >= n)
                && self.baseline_weights.len() >= n
                && self.baseline_weights.iter().all(|row| row.len() >= n),
            "apply_plasticity invariant violation — ensure_neuromod_state_synced should have rebuilt"
        );
        if n > self.neurons.len()
            || self.neurons.iter().any(|neu| neu.weights.len() < n)
            || self.baseline_weights.len() < n
            || self.baseline_weights.iter().any(|row| row.len() < n)
        {
            return;
        }
        let decay = self.config.plasticity_decay;
        let potentiate = self.config.plasticity_potentiate;
        let plasticity_speed = self.config.plasticity_speed;
        let fatigue_acc = self.config.fatigue_accumulation;
        let fatigue_rec = self.config.fatigue_recovery;

        for i in 0..n {
            if active_channels.contains(&i) {
                // Active channel: strengthen (dopamine-gated), accumulate fatigue.
                let strengthen = potentiate * (1.0 + mods.dopamine);
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    // Move toward amplified baseline.
                    let next_weight = if plasticity_speed == 0.0 {
                        current
                    } else {
                        let target = baseline * (1.0 + strengthen);
                        let update = if !target.is_finite() {
                            let extreme = if (baseline > 0.0 && strengthen >= -1.0)
                                || (baseline < 0.0 && strengthen < -1.0)
                            {
                                2.0
                            } else {
                                -1.5
                            };
                            current + (extreme - current) * plasticity_speed
                        } else {
                            current + (target - current) * plasticity_speed
                        };
                        if update.is_finite() {
                            update.clamp(-1.5, 2.0)
                        } else if update.is_sign_positive() {
                            2.0
                        } else {
                            -1.5
                        }
                    };
                    self.neurons[i].weights[j] = next_weight;
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] + fatigue_acc).min(1.0);
            } else {
                // Inactive channel: decay toward baseline, recover fatigue.
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    let update = if decay == 0.0 || current == baseline {
                        current
                    } else {
                        let diff = baseline - current;
                        if !diff.is_finite() {
                            if baseline > current { 2.0 } else { -1.5 }
                        } else {
                            current + diff * decay
                        }
                    };
                    self.neurons[i].weights[j] = if update.is_finite() {
                        update.clamp(-1.5, 2.0)
                    } else if update.is_sign_positive() {
                        2.0
                    } else {
                        -1.5
                    };
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] - fatigue_rec).max(0.0);
            }
        }
    }

    /// Apply feedback to adjust synaptic weights for a specific channel.
    ///
    /// Self-heals `channel_fatigue` and `baseline_weights` before any indexing,
    /// so calling `apply_feedback` on a freshly deserialized router (where
    /// the lazy repair in `route_modulated` has not yet run) is safe. See
    /// `ensure_neuromod_state_synced` for the exact shape check.
    pub fn apply_feedback(&mut self, channel_idx: usize, reward: f32) {
        self.ensure_neuromod_state_synced();
        let n = self.config.channel_count;
        // Guard all indexing: channel_idx bounds, neuron vector length,
        // and individual neuron weight vector lengths. A malformed
        // deserialized state could have short weight rows even when
        // neurons.len() >= n.
        debug_assert!(
            channel_idx < n
                && self.neurons.len() >= n
                && self.neurons.iter().all(|neu| neu.weights.len() >= n),
            "apply_feedback invariant violation — ensure_neuromod_state_synced should have rebuilt"
        );
        if channel_idx >= n
            || n > self.neurons.len()
            || self.neurons.iter().any(|neu| neu.weights.len() < n)
        {
            return;
        }

        let delta = reward * 0.01;

        self.neurons[channel_idx].weights[channel_idx] =
            (self.neurons[channel_idx].weights[channel_idx] + delta).clamp(0.1, 2.0);

        if reward > 0.0 {
            for j in 0..n {
                if j != channel_idx {
                    self.neurons[j].weights[channel_idx] =
                        (self.neurons[j].weights[channel_idx] - delta * 0.3).clamp(-1.0, 1.5);
                }
            }
        }

        // Keep plasticity baseline in sync with feedback-driven learning.
        self.sync_baseline_after_feedback(channel_idx, reward);
    }

    /// Sync `baseline_weights` for the rows affected by a feedback call so that
    /// the new feedback-adjusted weights become the reference point for future
    /// use-it-or-lose-it decay. Skipped if `baseline_weights` hasn't been
    /// initialized yet (e.g. before the first `route_modulated` call).
    fn sync_baseline_after_feedback(&mut self, channel_idx: usize, reward: f32) {
        let n = self.config.channel_count;
        // Belt-and-suspenders: ensure_neuromod_state_synced() in apply_feedback
        // should have already rebuilt baseline_weights if rows were malformed,
        // but guard anyway to avoid a panic if the call ordering invariant
        // is ever violated.
        debug_assert!(
            self.baseline_weights.len() == n
                && self.baseline_weights.iter().all(|row| row.len() == n),
            "baseline_weights shape mismatch — ensure_neuromod_state_synced should have rebuilt"
        );
        if self.baseline_weights.len() != n
            || self.baseline_weights.iter().any(|row| row.len() != n)
        {
            return;
        }
        self.baseline_weights[channel_idx][channel_idx] =
            self.neurons[channel_idx].weights[channel_idx];
        if reward > 0.0 {
            for j in 0..n {
                if j != channel_idx {
                    self.baseline_weights[j][channel_idx] = self.neurons[j].weights[channel_idx];
                }
            }
        }
    }

    /// Apply global neuromodulatory gain to all neurons.
    pub fn set_global_gain(&mut self, gain: f32) {
        for neu in &mut self.neurons {
            neu.set_gain(gain);
        }
    }

    /// Current routing weight matrix (row = neuron, col = input channel).
    pub fn weight_matrix(&self) -> Vec<Vec<f32>> {
        self.neurons.iter().map(|neu| neu.weights.clone()).collect()
    }

    /// Access the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Access per-channel fatigue levels.
    pub fn fatigue(&self) -> &[f32] {
        &self.channel_fatigue
    }
}

#[cfg(test)]
mod validate_tests {
    use super::*;
    use serde_json::json;

    fn router_field(err: &MeshError) -> &'static str {
        match err {
            MeshError::InvalidRouterConfig { field, .. } => field,
            other => panic!("expected InvalidRouterConfig, got {other:?}"),
        }
    }

    fn serde_field_from_config_json(value: serde_json::Value) -> String {
        let err = serde_json::from_value::<RouterConfig>(value).expect_err("config should fail");
        err.to_string()
    }

    fn serde_field_from_router_json(value: serde_json::Value) -> String {
        let err = serde_json::from_value::<ChannelRouter>(value).expect_err("router should fail");
        err.to_string()
    }

    enum BadValue {
        Count(usize),
        Float(f32),
    }

    struct InvalidCase {
        name: &'static str,
        field: &'static str,
        value: BadValue,
    }

    fn apply_bad(config: &mut RouterConfig, field: &str, value: &BadValue) {
        match (field, value) {
            ("channel_count", BadValue::Count(v)) => config.channel_count = *v,
            ("routing_timesteps", BadValue::Count(v)) => config.routing_timesteps = *v,
            ("self_weight", BadValue::Float(v)) => config.self_weight = *v,
            ("cross_weight", BadValue::Float(v)) => config.cross_weight = *v,
            ("threshold", BadValue::Float(v)) => config.threshold = *v,
            ("leak", BadValue::Float(v)) => config.leak = *v,
            ("min_fire_rate", BadValue::Float(v)) => config.min_fire_rate = *v,
            ("plasticity_decay", BadValue::Float(v)) => config.plasticity_decay = *v,
            ("plasticity_potentiate", BadValue::Float(v)) => config.plasticity_potentiate = *v,
            ("plasticity_speed", BadValue::Float(v)) => config.plasticity_speed = *v,
            ("fatigue_accumulation", BadValue::Float(v)) => config.fatigue_accumulation = *v,
            ("fatigue_recovery", BadValue::Float(v)) => config.fatigue_recovery = *v,
            _ => panic!("unhandled invalid-case field {field}"),
        }
    }

    fn json_representable(value: &BadValue) -> bool {
        match *value {
            BadValue::Count(_) => true,
            BadValue::Float(v) => v.is_finite(),
        }
    }

    fn invalid_cases() -> Vec<InvalidCase> {
        let counts = [
            ("channel_count_zero", "channel_count", 0usize),
            (
                "channel_count_over_max",
                "channel_count",
                MAX_ROUTER_CHANNELS + 1,
            ),
            ("channel_count_usize_max", "channel_count", usize::MAX),
            ("routing_timesteps_zero", "routing_timesteps", 0),
            (
                "routing_timesteps_over_max",
                "routing_timesteps",
                MAX_ROUTING_TIMESTEPS + 1,
            ),
        ];
        let floats = [
            ("self_weight_nan", "self_weight", f32::NAN),
            ("self_weight_inf", "self_weight", f32::INFINITY),
            ("cross_weight_neg_inf", "cross_weight", f32::NEG_INFINITY),
            ("threshold_nan", "threshold", f32::NAN),
            ("leak_below", "leak", -0.01),
            ("leak_above", "leak", 1.01),
            ("leak_nan", "leak", f32::NAN),
            ("leak_inf", "leak", f32::INFINITY),
            ("min_fire_rate_below", "min_fire_rate", -0.01),
            ("min_fire_rate_above", "min_fire_rate", 1.01),
            ("min_fire_rate_nan", "min_fire_rate", f32::NAN),
            ("plasticity_decay_below", "plasticity_decay", -0.01),
            ("plasticity_decay_above", "plasticity_decay", 1.01),
            ("plasticity_decay_nan", "plasticity_decay", f32::NAN),
            (
                "plasticity_potentiate_negative",
                "plasticity_potentiate",
                -0.01,
            ),
            (
                "plasticity_potentiate_nan",
                "plasticity_potentiate",
                f32::NAN,
            ),
            (
                "plasticity_potentiate_inf",
                "plasticity_potentiate",
                f32::INFINITY,
            ),
            ("plasticity_speed_below", "plasticity_speed", -0.01),
            ("plasticity_speed_above", "plasticity_speed", 1.01),
            ("plasticity_speed_nan", "plasticity_speed", f32::NAN),
            ("fatigue_accumulation_below", "fatigue_accumulation", -0.01),
            ("fatigue_accumulation_above", "fatigue_accumulation", 1.01),
            ("fatigue_accumulation_nan", "fatigue_accumulation", f32::NAN),
            ("fatigue_recovery_below", "fatigue_recovery", -0.01),
            ("fatigue_recovery_above", "fatigue_recovery", 1.01),
            ("fatigue_recovery_inf", "fatigue_recovery", f32::INFINITY),
        ];
        counts
            .into_iter()
            .map(|(name, field, value)| InvalidCase {
                name,
                field,
                value: BadValue::Count(value),
            })
            .chain(floats.into_iter().map(|(name, field, value)| InvalidCase {
                name,
                field,
                value: BadValue::Float(value),
            }))
            .collect()
    }
    #[test]
    fn default_config_is_valid() {
        RouterConfig::default().validate().unwrap();
    }

    #[test]
    fn table_rejects_every_invalid_config_field() {
        for case in invalid_cases() {
            let mut config = RouterConfig::default();
            apply_bad(&mut config, case.field, &case.value);
            let err = match config.validate() {
                Err(err) => err,
                Ok(()) => panic!("{}: validate should fail", case.name),
            };
            assert_eq!(
                router_field(&err),
                case.field,
                "{}: unexpected field in {err}",
                case.name
            );

            let ctor_err = ChannelRouter::try_with_config(config.clone()).expect_err(case.name);
            assert_eq!(
                router_field(&ctor_err),
                case.field,
                "{}: constructor field mismatch",
                case.name
            );
            assert_eq!(
                ctor_err, err,
                "{}: validate and try_with_config must return the same error",
                case.name
            );

            if !json_representable(&case.value) {
                continue;
            }
            let mut mutated = RouterConfig::default();
            apply_bad(&mut mutated, case.field, &case.value);
            let config_msg = serde_field_from_config_json(serde_json::to_value(&mutated).unwrap());
            assert!(
                config_msg.contains(case.field),
                "{}: RouterConfig serde missing field name: {config_msg}",
                case.name
            );
            assert!(
                config_msg.contains(&err.to_string()),
                "{}: RouterConfig serde should carry the same MeshError Display, got {config_msg}, expected {}",
                case.name,
                err
            );

            let mut router_json = serde_json::to_value(ChannelRouter::default()).unwrap();
            router_json["config"] = serde_json::to_value(&mutated).unwrap();
            let router_msg = serde_field_from_router_json(router_json);
            assert!(
                router_msg.contains(case.field),
                "{}: ChannelRouter serde missing field name: {router_msg}",
                case.name
            );
            assert!(
                router_msg.contains(&err.to_string()),
                "{}: ChannelRouter serde should carry the same MeshError Display, got {router_msg}, expected {}",
                case.name,
                err
            );
        }
    }

    type BoundaryMutator = fn(&mut RouterConfig);

    fn valid_boundary_mutators() -> Vec<(&'static str, BoundaryMutator)> {
        vec![
            ("min_channels", |c| c.channel_count = 1),
            ("max_channels_validate_only", |c| {
                c.channel_count = MAX_ROUTER_CHANNELS
            }),
            ("leak_0", |c| c.leak = 0.0),
            ("leak_1", |c| c.leak = 1.0),
            ("min_fire_0", |c| c.min_fire_rate = 0.0),
            ("min_fire_1", |c| c.min_fire_rate = 1.0),
            ("decay_0", |c| c.plasticity_decay = 0.0),
            ("decay_1", |c| c.plasticity_decay = 1.0),
            ("potentiate_0", |c| c.plasticity_potentiate = 0.0),
            ("speed_0", |c| c.plasticity_speed = 0.0),
            ("speed_1", |c| c.plasticity_speed = 1.0),
            ("fatigue_acc_0", |c| c.fatigue_accumulation = 0.0),
            ("fatigue_acc_1", |c| c.fatigue_accumulation = 1.0),
            ("fatigue_rec_0", |c| c.fatigue_recovery = 0.0),
            ("fatigue_rec_1", |c| c.fatigue_recovery = 1.0),
            ("signed_cross_inhibition", |c| c.cross_weight = -1.0),
            ("positive_cross_weight", |c| c.cross_weight = 0.25),
            ("signed_self_weight", |c| c.self_weight = -0.3),
            ("zero_threshold", |c| c.threshold = 0.0),
            ("one_timestep", |c| c.routing_timesteps = 1),
            ("max_timesteps", |c| {
                c.routing_timesteps = MAX_ROUTING_TIMESTEPS
            }),
        ]
    }

    #[test]
    fn valid_boundaries_round_trip() {
        for (name, mutate) in valid_boundary_mutators() {
            let mut config = RouterConfig::default();
            mutate(&mut config);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{name}: valid boundary rejected: {err}"));
            let json = serde_json::to_value(&config).unwrap();
            let restored: RouterConfig = serde_json::from_value(json)
                .unwrap_or_else(|err| panic!("{name}: valid config failed to deserialize: {err}"));
            assert_eq!(restored, config, "{name}: config round-trip");

            if config.channel_count == MAX_ROUTER_CHANNELS {
                continue;
            }
            let router = ChannelRouter::try_with_config(config.clone())
                .unwrap_or_else(|err| panic!("{name}: try_with_config: {err}"));
            let router_json = serde_json::to_value(&router).unwrap();
            let restored_router: ChannelRouter = serde_json::from_value(router_json)
                .unwrap_or_else(|err| panic!("{name}: router deserialize: {err}"));
            assert_eq!(restored_router.config(), router.config(), "{name}");
            assert_eq!(
                restored_router.weight_matrix(),
                router.weight_matrix(),
                "{name}"
            );
            assert_eq!(restored_router.fatigue(), router.fatigue(), "{name}");
            assert_eq!(restored_router.total_routes, router.total_routes, "{name}");
        }
    }

    #[test]
    fn custom_and_default_routers_round_trip() {
        let default_router = ChannelRouter::default();
        let json = serde_json::to_value(&default_router).unwrap();
        let restored: ChannelRouter = serde_json::from_value(json).unwrap();
        assert_eq!(restored.config(), default_router.config());
        assert_eq!(restored.weight_matrix(), default_router.weight_matrix());

        let custom = RouterConfig {
            channel_count: 4,
            self_weight: 1.1,
            cross_weight: -0.2,
            threshold: 0.3,
            leak: 0.05,
            routing_timesteps: 8,
            min_fire_rate: 0.25,
            plasticity_decay: 0.03,
            plasticity_potentiate: 0.2,
            plasticity_speed: 0.4,
            fatigue_accumulation: 0.2,
            fatigue_recovery: 0.1,
        };
        let router = ChannelRouter::try_with_config(custom.clone()).unwrap();
        let json = serde_json::to_value(&router).unwrap();
        let restored: ChannelRouter = serde_json::from_value(json).unwrap();
        assert_eq!(restored.config(), &custom);
        assert_eq!(restored.weight_matrix(), router.weight_matrix());
    }

    #[test]
    fn try_with_config_rejects_oversized_channel_count_without_allocating() {
        // If validation ran after `vec![0.0; n]`, usize::MAX would OOM/hang
        // this test rather than returning.
        let config = RouterConfig {
            channel_count: usize::MAX,
            ..RouterConfig::default()
        };
        let err = ChannelRouter::try_with_config(config).unwrap_err();
        assert_eq!(router_field(&err), "channel_count");
    }

    #[test]
    fn max_channel_count_is_accepted_by_validate_and_constructor() {
        let config = RouterConfig {
            channel_count: MAX_ROUTER_CHANNELS,
            ..RouterConfig::default()
        };
        config.validate().unwrap();
        let router = ChannelRouter::try_with_config(config).unwrap();
        assert_eq!(router.config().channel_count, MAX_ROUTER_CHANNELS);
        assert_eq!(router.weight_matrix().len(), MAX_ROUTER_CHANNELS);
        assert_eq!(router.weight_matrix()[0].len(), MAX_ROUTER_CHANNELS);
        assert_eq!(router.fatigue().len(), MAX_ROUTER_CHANNELS);
    }

    #[test]
    fn malformed_shapes_are_rejected_with_matching_fields() {
        let router = ChannelRouter::default();
        let base = serde_json::to_value(&router).unwrap();

        let mut empty_inner = base.clone();
        empty_inner["baseline_weights"] = json!([[], [], []]);
        let msg = serde_field_from_router_json(empty_inner);
        assert!(msg.contains("baseline_weights"), "empty inner rows: {msg}");

        let mut short_fatigue = base.clone();
        short_fatigue["channel_fatigue"] = json!([0.0, 0.0]);
        let msg = serde_field_from_router_json(short_fatigue);
        assert!(msg.contains("channel_fatigue"), "short fatigue: {msg}");

        let mut extra_neurons = base.clone();
        extra_neurons["neurons"] = json!([]);
        let msg = serde_field_from_router_json(extra_neurons);
        assert!(msg.contains("neurons"), "empty neurons: {msg}");

        let mut missing_weights = base.clone();
        missing_weights["neurons"][0]
            .as_object_mut()
            .unwrap()
            .remove("weights");
        let msg = serde_field_from_router_json(missing_weights);
        assert!(msg.contains("weights"), "missing neuron weights: {msg}");

        let mut ragged_baseline = base.clone();
        ragged_baseline["baseline_weights"] = json!([[0.9, -0.15, -0.15], [0.0], [0.0, 0.0, 0.0]]);
        let msg = serde_field_from_router_json(ragged_baseline);
        assert!(msg.contains("baseline_weights"), "ragged baseline: {msg}");

        let mut out_of_range_fatigue = base;
        out_of_range_fatigue["channel_fatigue"] = json!([0.0, 1.5, 0.0]);
        let msg = serde_field_from_router_json(out_of_range_fatigue);
        assert!(
            msg.contains("channel_fatigue"),
            "fatigue out of range: {msg}"
        );
    }

    #[test]
    fn legacy_snapshot_without_optional_fields_deserializes() {
        let router = ChannelRouter::default();
        let mut json = serde_json::to_value(&router).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("config");
        obj.remove("channel_fatigue");
        obj.remove("baseline_weights");

        let restored: ChannelRouter =
            serde_json::from_value(json).expect("legacy snapshot must deserialize");
        assert_eq!(restored.config().channel_count, 3);
        assert_eq!(restored.weight_matrix().len(), 3);
        assert_eq!(restored.fatigue(), &[0.0, 0.0, 0.0]);
        let mut restored = restored;
        restored
            .route_modulated([0.5, 0.0, 0.0], &NeuromodState::balanced())
            .unwrap();
    }

    #[test]
    fn signed_zero_rates_are_accepted() {
        let config = RouterConfig {
            leak: -0.0,
            min_fire_rate: -0.0,
            plasticity_decay: -0.0,
            plasticity_potentiate: -0.0,
            plasticity_speed: -0.0,
            fatigue_accumulation: -0.0,
            fatigue_recovery: -0.0,
            ..RouterConfig::default()
        };
        config.validate().unwrap();
        ChannelRouter::try_with_config(config).unwrap();
    }

    #[test]
    fn explicit_null_fields_are_rejected() {
        for field in ["config", "channel_fatigue", "baseline_weights"] {
            let router = ChannelRouter::default();
            let mut json = serde_json::to_value(&router).unwrap();
            json[field] = serde_json::Value::Null;
            let err = serde_json::from_value::<ChannelRouter>(json)
                .expect_err("explicit null field must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(field),
                "error should mention field {field}, got: {msg}"
            );
        }
    }

    #[test]
    fn extreme_weights_and_potentiate_with_zero_speed_do_not_produce_nan() {
        let config = RouterConfig {
            self_weight: 1.0,
            cross_weight: -0.5,
            plasticity_potentiate: f32::MAX,
            plasticity_speed: 0.0,
            ..RouterConfig::default()
        };
        config.validate().unwrap();
        let mut router = ChannelRouter::try_with_config(config).unwrap();
        let initial_weights = router.weight_matrix();
        let res = router.route_modulated([1.0, 0.0, 0.0], &NeuromodState::balanced());
        assert!(res.is_ok());
        for (row_idx, row) in router.weight_matrix().iter().enumerate() {
            for (col_idx, &w) in row.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "weight at [{row_idx}][{col_idx}] must remain finite, got {w}"
                );
                assert_eq!(
                    w, initial_weights[row_idx][col_idx],
                    "weight with plasticity_speed 0.0 must remain unchanged"
                );
            }
        }

        // Also verify non-zero plasticity_speed with extreme potentiation clamps without NaN
        let config_speed = RouterConfig {
            self_weight: 1.0,
            cross_weight: -0.5,
            plasticity_potentiate: f32::MAX,
            plasticity_speed: 0.5,
            ..RouterConfig::default()
        };
        let mut router_speed = ChannelRouter::try_with_config(config_speed).unwrap();
        let res_speed = router_speed.route_modulated([1.0, 0.0, 0.0], &NeuromodState::balanced());
        assert!(res_speed.is_ok());
        for (row_idx, row) in router_speed.weight_matrix().iter().enumerate() {
            for (col_idx, &w) in row.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "weight at [{row_idx}][{col_idx}] must remain finite, got {w}"
                );
                assert!(
                    (-1.5..=2.0).contains(&w),
                    "weight {w} at [{row_idx}][{col_idx}] must stay within [-1.5, 2.0]"
                );
            }
        }
    }
}
