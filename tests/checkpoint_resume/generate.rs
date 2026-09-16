// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic scenario generation for checkpoint/resume property tests.

use crate::compare::{check_resume, shrink};
use crate::harness::{Recipe, Scenario, SplitMix64, TickEvent};
use crate::recipes::{
    checkpoint_before_delivery, delay_capacity, delay_zero, empty_ticks, generated_random_scenario,
    generated_small_world_scenario, multi_spike_same_slot, signed_weights, single_neuron,
};

impl Scenario {
    pub(crate) fn from_seed(seed: u64) -> Self {
        let recipe = Recipe::from_seed(seed);
        let mut rng = SplitMix64::new(seed ^ 0xC0FF_EE11_D15C_A5E5);
        match recipe {
            Recipe::EmptyGraph => empty_graph(seed),
            Recipe::SingleNeuron => single_neuron(seed, &mut rng),
            Recipe::DelayZero => delay_zero(seed, &mut rng),
            Recipe::DelayCapacity => delay_capacity(seed, &mut rng),
            Recipe::MultiSpikeSameSlot => multi_spike_same_slot(seed, &mut rng),
            Recipe::CheckpointBeforeDelivery => checkpoint_before_delivery(seed, &mut rng),
            Recipe::SignedWeights => signed_weights(seed, &mut rng),
            Recipe::EmptyTicks => empty_ticks(seed, &mut rng),
            Recipe::GeneratedRandom => generated_random_scenario(seed, &mut rng),
            Recipe::GeneratedSmallWorld => generated_small_world_scenario(seed, &mut rng),
        }
    }
}

pub(crate) fn check_seed(seed: u64) {
    let scenario = Scenario::from_seed(seed);
    if let Err(err) = check_resume(&scenario) {
        let minimized = shrink(&scenario);
        panic!(
            "{err}\n\noriginal scenario: {scenario:?}\n\nminimized scenario: {minimized:?}\n\n\
             Persist seed {seed} in REGRESSION_SEEDS. If the minimized case is smaller, \
             add it as a named fixture."
        );
    }
}

pub(crate) fn empty_graph(seed: u64) -> Scenario {
    Scenario {
        seed,
        recipe: Recipe::EmptyGraph,
        neuron_count: 0,
        descriptors: Vec::new(),
        buffer_max_delay: 0,
        prefix: vec![TickEvent::idle(0), TickEvent::idle(0)],
        suffix: vec![TickEvent::idle(0), TickEvent::Graded(Vec::new())],
    }
}
