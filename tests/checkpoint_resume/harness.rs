// SPDX-License-Identifier: MIT OR Apache-2.0

//! Types, RNG, and live-vs-restored comparison for the checkpoint suite.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::SynapticGraph;
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

/// Seeded cases executed in normal CI (`cargo test`).
pub(crate) const CI_CASES: u64 = 512;

/// Default seed count for the ignored nightly profile.
pub(crate) const NIGHTLY_CASES_DEFAULT: u64 = 10_000;

/// Env var that overrides [`NIGHTLY_CASES_DEFAULT`] for ignored nightly runs.
pub(crate) const NIGHTLY_CASES_ENV: &str = "CHECKPOINT_RESUME_CASES";

/// Once-failing seeds discovered outside the CI range (`0..CI_CASES`).
pub(crate) const REGRESSION_SEEDS: &[u64] = &[];

const WEIGHTS: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 0.125, 1.5, 0.375, 0.625];

pub(crate) const ACTIVATIONS: [f32; 7] = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recipe {
    EmptyGraph,
    SingleNeuron,
    DelayZero,
    DelayCapacity,
    MultiSpikeSameSlot,
    CheckpointBeforeDelivery,
    SignedWeights,
    EmptyTicks,
    GeneratedRandom,
    GeneratedSmallWorld,
}

impl Recipe {
    pub(crate) const ALL: [Recipe; 10] = [
        Self::EmptyGraph,
        Self::SingleNeuron,
        Self::DelayZero,
        Self::DelayCapacity,
        Self::MultiSpikeSameSlot,
        Self::CheckpointBeforeDelivery,
        Self::SignedWeights,
        Self::EmptyTicks,
        Self::GeneratedRandom,
        Self::GeneratedSmallWorld,
    ];

    pub(crate) fn from_seed(seed: u64) -> Self {
        Self::ALL[(seed % Self::ALL.len() as u64) as usize]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TickEvent {
    Binary(Vec<bool>),
    Graded(Vec<f32>),
}

impl TickEvent {
    pub(crate) fn idle(n: usize) -> Self {
        Self::Binary(vec![false; n])
    }

    pub(crate) fn fire(n: usize, neurons: &[usize]) -> Self {
        let mut spikes = vec![false; n];
        for &i in neurons {
            spikes[i] = true;
        }
        Self::Binary(spikes)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Scenario {
    pub(crate) seed: u64,
    pub(crate) recipe: Recipe,
    pub(crate) neuron_count: usize,
    pub(crate) descriptors: Vec<SynapseDescriptor>,
    pub(crate) buffer_max_delay: usize,
    pub(crate) prefix: Vec<TickEvent>,
    pub(crate) suffix: Vec<TickEvent>,
}

impl Scenario {
    pub(crate) fn mesh(&self) -> SynapticMesh {
        let graph = SynapticGraph::from_descriptors(self.neuron_count, &self.descriptors)
            .expect("generator must emit valid descriptors");
        SynapticMesh::try_with_max_delay(graph, self.buffer_max_delay)
            .expect("buffer capacity must hold every synapse delay")
    }
}

pub(crate) fn descriptor(
    source: usize,
    target: usize,
    weight: f32,
    delay: u16,
    polarity: Polarity,
) -> SynapseDescriptor {
    SynapseDescriptor {
        source: source as u32,
        target: target as u32,
        weight,
        delay,
        polarity,
    }
}

pub(crate) fn apply_event(mesh: &mut SynapticMesh, event: &TickEvent) -> Vec<f32> {
    match event {
        TickEvent::Binary(spikes) => mesh
            .propagate(spikes)
            .expect("generated spike vector matches neuron_count"),
        TickEvent::Graded(activations) => mesh
            .propagate_graded(activations)
            .expect("generated activations are finite and match neuron_count"),
    }
}

pub(crate) fn checkpoint_snapshot(mesh: &SynapticMesh) -> serde_json::Value {
    serde_json::to_value(mesh).expect("valid mesh must serialize")
}

pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn bounded(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }

    pub(crate) fn inclusive(&mut self, max: usize) -> usize {
        self.bounded(max.saturating_add(1))
    }

    pub(crate) fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub(crate) fn f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    pub(crate) fn weight(&mut self) -> f32 {
        WEIGHTS[self.bounded(WEIGHTS.len())]
    }
}
