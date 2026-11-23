// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wiring rules for synaptic graph construction.
//!
//! These functions operate on an existing [`SynapticGraph`] to enforce
//! biological constraints like Dale's law or to assign realistic
//! axonal propagation delays.

use alloc::{format, vec::Vec};

use crate::error::MeshError;
use crate::math;
use crate::topology::graph::SynapticGraph;
use crate::types::{DelayModel, DelayTicks, Polarity, SynapseDescriptor};

/// Apply Dale's law to a graph: all outgoing synapses from each neuron
/// share a single polarity (excitatory or inhibitory).
///
/// The first `inh_count` neurons (by ID) are assigned inhibitory polarity;
/// the rest are excitatory.
///
/// # References
///
/// Dale, H. H. (1935). *Pharmacology and Nerve-endings.*
pub fn apply_dale_polarity(
    graph: &SynapticGraph,
    inhibitory_fraction: f32,
) -> Result<Vec<Polarity>, MeshError> {
    if !(0.0..=1.0).contains(&inhibitory_fraction) {
        return Err(MeshError::InvalidConfig(format!(
            "inhibitory fraction={inhibitory_fraction} must be in [0, 1]"
        )));
    }
    let n = graph.neuron_count();
    let inh_count = (n as f32 * inhibitory_fraction) as usize;
    Ok((0..n)
        .map(|i| {
            if i < inh_count {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            }
        })
        .collect())
}

/// Assign axonal propagation delays based on distance between neurons.
///
/// If the graph has position data, delay is proportional to Euclidean distance:
///   `delay = clamp(round(distance / speed), 1, max_delay)`
///
/// When `max_delay == 0`, all delays are set to 0 (same-tick delivery),
/// matching the `SpikeDelayBuffer` zero-cost behavior.
///
/// If no positions are available, delay is assigned deterministically from
/// source/target indices.
///
/// # Arguments
///
/// * `descriptors` — synapse descriptors to update
/// * `positions` — optional 3D positions per neuron
/// * `speed` — propagation speed (distance units per tick)
/// * `max_delay` — maximum delay in ticks
pub fn assign_delays(
    descriptors: &mut [SynapseDescriptor],
    positions: Option<&[[f32; 3]]>,
    speed: f32,
    max_delay: DelayTicks,
) {
    // max_delay == 0 means all synapses deliver same-tick (delay = 0).
    //
    // Historically, the position-based path used `clamp(1, max_delay)`
    // which would panic in debug mode (min > max), while the index-based
    // path produced delay=1 via `% 1 = 0` then `.max(1) = 1`. The early
    // return below (predating this PR) makes both paths consistent when
    // max_delay == 0.
    //
    // Callers that previously relied on max_delay=0 producing delay=1
    // should update to expect delay=0 (same-tick delivery).
    if max_delay == 0 {
        for desc in descriptors.iter_mut() {
            desc.delay = 0;
        }
        return;
    }

    for desc in descriptors.iter_mut() {
        if let Some(pos) = positions {
            let src_pos = &pos[desc.source as usize];
            let tgt_pos = &pos[desc.target as usize];
            let (dx, dy, dz) = (
                src_pos[0] - tgt_pos[0],
                src_pos[1] - tgt_pos[1],
                src_pos[2] - tgt_pos[2],
            );
            let dist = math::sqrt(dx * dx + dy * dy + dz * dz);
            let delay = math::round(dist / speed.max(1e-6)) as u16;
            desc.delay = delay.clamp(1, max_delay);
        } else {
            // Deterministic delay from indices
            let mixed = (desc.source as usize * 71 + desc.target as usize * 37 + 13)
                % (max_delay as usize + 1);
            desc.delay = (mixed as u16).max(1);
        }
    }
}

/// Assign axonal propagation delays from a [`DelayModel`].
///
/// - [`DelayModel::Fixed`] sets every delay to `delay` and ignores
///   `positions`; `Fixed { delay: 0 }` is same-tick delivery.
/// - [`DelayModel::DistanceProportional`] with positions uses
///   `clamp(round(distance / speed), min_delay, max_delay)`. Without positions
///   it falls back to the index spread below over `[min_delay, max_delay]`.
/// - [`DelayModel::UniformRandom`] uses a deterministic index spread, never
///   an RNG: `min_delay - 1 + max((s * 71 + t * 37 + 13) % (span + 1), 1)`
///   with `span = max_delay - min_delay + 1`.
///
/// With `min_delay == 1` and `max_delay >= 1`, `DistanceProportional` and
/// `UniformRandom` reproduce [`assign_delays`] with the same `speed` and
/// `max_delay` exactly (with and without positions respectively).
///
/// This function is infallible: when `min_delay > max_delay` the model
/// behaves as `Fixed { delay: min_delay }`. That includes
/// `{ min_delay: 1, max_delay: 0 }`, which sets every delay to 1, whereas
/// `assign_delays(.., max_delay = 0)` sets every delay to 0; use
/// `Fixed { delay: 0 }` for same-tick delivery.
///
/// # Panics
///
/// Like [`assign_delays`], panics if `positions` is shorter than a
/// descriptor's `source + 1` or `target + 1` on the distance path.
pub fn assign_delays_from_model(
    descriptors: &mut [SynapseDescriptor],
    positions: Option<&[[f32; 3]]>,
    model: &DelayModel,
) {
    match *model {
        DelayModel::Fixed { delay } => set_all_delays(descriptors, delay),
        DelayModel::UniformRandom {
            min_delay,
            max_delay,
        }
        | DelayModel::DistanceProportional {
            min_delay,
            max_delay,
            ..
        } if min_delay > max_delay => set_all_delays(descriptors, min_delay),
        DelayModel::UniformRandom {
            min_delay,
            max_delay,
        } => index_spread_delays(descriptors, DelayRange::new(min_delay, max_delay)),
        DelayModel::DistanceProportional {
            speed,
            min_delay,
            max_delay,
        } => {
            let range = DelayRange::new(min_delay, max_delay);
            match positions {
                Some(pos) => distance_delays(descriptors, pos, speed, range),
                None => index_spread_delays(descriptors, range),
            }
        }
    }
}

fn set_all_delays(descriptors: &mut [SynapseDescriptor], delay: DelayTicks) {
    for desc in descriptors.iter_mut() {
        desc.delay = delay;
    }
}

/// Inclusive delay range with `min <= max`.
#[derive(Debug, Clone, Copy)]
struct DelayRange {
    min: DelayTicks,
    max: DelayTicks,
}

impl DelayRange {
    fn new(min: DelayTicks, max: DelayTicks) -> Self {
        debug_assert!(min <= max);
        Self { min, max }
    }
}

fn index_spread_delays(descriptors: &mut [SynapseDescriptor], range: DelayRange) {
    let span = usize::from(range.max) - usize::from(range.min) + 1;
    for desc in descriptors.iter_mut() {
        let mixed = (desc.source as usize * 71 + desc.target as usize * 37 + 13) % (span + 1);
        desc.delay = (usize::from(range.min) + mixed.max(1) - 1) as DelayTicks;
    }
}

fn distance_delays(
    descriptors: &mut [SynapseDescriptor],
    positions: &[[f32; 3]],
    speed: f32,
    range: DelayRange,
) {
    for desc in descriptors.iter_mut() {
        let src_pos = &positions[desc.source as usize];
        let tgt_pos = &positions[desc.target as usize];
        let (dx, dy, dz) = (
            src_pos[0] - tgt_pos[0],
            src_pos[1] - tgt_pos[1],
            src_pos[2] - tgt_pos[2],
        );
        let dist = math::sqrt(dx * dx + dy * dy + dz * dz);
        let delay = math::round(dist / speed.max(1e-6)) as u16;
        desc.delay = delay.clamp(range.min, range.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::generators::generate_random;

    #[test]
    fn dale_polarity_splits_correctly() {
        let graph = generate_random(100, 0.1, 5, 0.2).unwrap();
        let polarities = apply_dale_polarity(&graph, 0.2).unwrap();
        assert_eq!(polarities.len(), 100);
        assert_eq!(
            polarities
                .iter()
                .filter(|&&p| p == Polarity::Inhibitory)
                .count(),
            20
        );
        assert_eq!(
            polarities
                .iter()
                .filter(|&&p| p == Polarity::Excitatory)
                .count(),
            80
        );
    }

    #[test]
    fn distance_based_delays() {
        let mut descs = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]; // distance = 5
        assign_delays(&mut descs, Some(&positions), 2.5, 10);
        // 5.0 / 2.5 = 2.0 → delay = 2
        assert_eq!(descs[0].delay, 2);
    }

    #[test]
    fn delay_clamped_to_max() {
        let mut descs = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let positions = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]; // distance = 100
        assign_delays(&mut descs, Some(&positions), 1.0, 5);
        assert_eq!(descs[0].delay, 5);
    }
}
