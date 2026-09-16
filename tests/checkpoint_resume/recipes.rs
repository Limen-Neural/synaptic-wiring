// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-built and generated recipe constructors.

use synaptic_wiring::topology::{SynapticGraph, generate_random, generate_small_world};
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

use crate::harness::{ACTIVATIONS, Recipe, Scenario, SplitMix64, TickEvent, descriptor};

pub(crate) fn single_neuron(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let delay = rng.inclusive(4) as u16;
    let extra = rng.inclusive(3);
    let descriptors = if rng.bool() {
        vec![descriptor(0, 0, rng.weight(), delay, Polarity::Excitatory)]
    } else {
        Vec::new()
    };
    let n = 1;
    let buffer_max_delay = usize::from(delay) + extra;
    let prefix_len = rng.inclusive(4);
    let suffix_len = rng.inclusive(4).max(1);
    Scenario {
        seed,
        recipe: Recipe::SingleNeuron,
        neuron_count: n,
        descriptors,
        buffer_max_delay,
        prefix: random_events(rng, n, prefix_len),
        suffix: random_events(rng, n, suffix_len),
    }
}

pub(crate) fn delay_zero(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(6).max(2);
    let mut descriptors = Vec::new();
    for src in 0..n {
        let src_polarity = random_polarity(rng);
        let tgt = (src + 1) % n;
        descriptors.push(descriptor(src, tgt, rng.weight(), 0, src_polarity));
        if rng.bool() {
            let tgt2 = rng.bounded(n);
            descriptors.push(descriptor(src, tgt2, rng.weight(), 0, src_polarity));
        }
    }
    let prefix_len = rng.inclusive(5);
    let suffix_len = rng.inclusive(5).max(1);
    Scenario {
        seed,
        recipe: Recipe::DelayZero,
        neuron_count: n,
        descriptors,
        buffer_max_delay: rng.inclusive(3),
        prefix: random_events(rng, n, prefix_len),
        suffix: random_events(rng, n, suffix_len),
    }
}

pub(crate) fn delay_capacity(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(5).max(2);
    let delay = rng.inclusive(8).max(1) as u16;
    let extra = rng.inclusive(2);
    let descriptors = vec![
        descriptor(0, 1, rng.weight(), delay, random_polarity(rng)),
        descriptor(1, 0, rng.weight(), delay, random_polarity(rng)),
    ];
    let buffer_max_delay = usize::from(delay) + extra;
    let extra_suffix = rng.inclusive(3);
    Scenario {
        seed,
        recipe: Recipe::DelayCapacity,
        neuron_count: n,
        descriptors,
        buffer_max_delay,
        prefix: vec![TickEvent::fire(n, &[0, 1])],
        suffix: {
            let mut suffix = vec![TickEvent::idle(n); usize::from(delay) + 1];
            suffix.extend(random_events(rng, n, extra_suffix));
            suffix
        },
    }
}

pub(crate) fn multi_spike_same_slot(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = 3;
    let delay = rng.inclusive(5).max(1) as u16;
    let descriptors = vec![
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(1, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
    ];
    Scenario {
        seed,
        recipe: Recipe::MultiSpikeSameSlot,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay),
        prefix: vec![TickEvent::fire(n, &[0, 1])],
        suffix: {
            let mut suffix = vec![TickEvent::idle(n); usize::from(delay)];
            suffix.push(TickEvent::idle(n));
            suffix.push(TickEvent::fire(n, &[0, 1]));
            suffix.extend(vec![TickEvent::idle(n); usize::from(delay)]);
            suffix
        },
    }
}

pub(crate) fn checkpoint_before_delivery(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(4).max(2);
    let delay = rng.inclusive(6).max(1) as u16;
    let descriptors = vec![descriptor(0, 1, rng.weight(), delay, Polarity::Excitatory)];
    let mut prefix = vec![TickEvent::fire(n, &[0])];
    prefix.extend(vec![TickEvent::idle(n); usize::from(delay) - 1]);
    Scenario {
        seed,
        recipe: Recipe::CheckpointBeforeDelivery,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay) + rng.inclusive(2),
        prefix,
        suffix: {
            let extra = rng.inclusive(4);
            let mut suffix = vec![TickEvent::idle(n)];
            suffix.extend(random_events(rng, n, extra));
            suffix
        },
    }
}

pub(crate) fn signed_weights(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = 3;
    let delay = rng.inclusive(4) as u16;
    let descriptors = vec![
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(1, 2, rng.weight(), delay.max(1), Polarity::Inhibitory),
    ];
    Scenario {
        seed,
        recipe: Recipe::SignedWeights,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay.max(1)) + rng.inclusive(2),
        prefix: vec![TickEvent::fire(n, &[0, 1]), TickEvent::idle(n)],
        suffix: vec![
            TickEvent::idle(n),
            TickEvent::Graded(vec![0.5, -0.5, 0.0]),
            TickEvent::idle(n),
            TickEvent::idle(n),
        ],
    }
}

pub(crate) fn empty_ticks(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(5).max(2);
    let delay = rng.inclusive(4) as u16;
    let descriptors = vec![descriptor(0, 1, rng.weight(), delay, random_polarity(rng))];
    let prefix_len = rng.inclusive(6);
    let suffix_len = rng.inclusive(6).max(2);
    Scenario {
        seed,
        recipe: Recipe::EmptyTicks,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay) + rng.inclusive(2),
        prefix: (0..prefix_len).map(|_| TickEvent::idle(n)).collect(),
        suffix: (0..suffix_len).map(|_| TickEvent::idle(n)).collect(),
    }
}

pub(crate) fn generated_random_scenario(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(8).max(2);
    let max_delay = rng.inclusive(6) as u16;
    let p = 0.2 + rng.f32() * 0.6;
    let inh = rng.f32() * 0.4;
    let graph = generate_random(n, p, max_delay, inh).expect("bounded generate_random");
    from_graph(seed, Recipe::GeneratedRandom, graph, rng)
}

pub(crate) fn generated_small_world_scenario(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = [4, 6, 8][rng.bounded(3)];
    let k = 2;
    let max_delay = rng.inclusive(5) as u16;
    let graph =
        generate_small_world(n, k, 0.2, max_delay, 0.25).expect("bounded generate_small_world");
    from_graph(seed, Recipe::GeneratedSmallWorld, graph, rng)
}

fn from_graph(seed: u64, recipe: Recipe, graph: SynapticGraph, rng: &mut SplitMix64) -> Scenario {
    let n = graph.neuron_count();
    let graph_max = usize::from(graph.max_delay());
    let extra = rng.inclusive(3);
    let descriptors = descriptors_from_graph(&graph);
    Scenario {
        seed,
        recipe,
        neuron_count: n,
        descriptors,
        buffer_max_delay: graph_max + extra,
        prefix: {
            let prefix_len = rng.inclusive(8);
            random_events(rng, n, prefix_len)
        },
        suffix: {
            let suffix_len = rng.inclusive(8).max(1);
            random_events(rng, n, suffix_len)
        },
    }
}

fn descriptors_from_graph(graph: &SynapticGraph) -> Vec<SynapseDescriptor> {
    let mut descriptors = Vec::new();
    for src in 0..graph.neuron_count() {
        for (target, weight, delay, polarity) in graph.outgoing(src) {
            descriptors.push(SynapseDescriptor {
                source: src as u32,
                target,
                weight: weight.abs(),
                delay,
                polarity,
            });
        }
    }
    descriptors
}

fn random_polarity(rng: &mut SplitMix64) -> Polarity {
    if rng.bool() {
        Polarity::Inhibitory
    } else {
        Polarity::Excitatory
    }
}

fn random_events(rng: &mut SplitMix64, n: usize, len: usize) -> Vec<TickEvent> {
    (0..len).map(|_| random_event(rng, n)).collect()
}

fn random_event(rng: &mut SplitMix64, n: usize) -> TickEvent {
    match rng.bounded(5) {
        0 | 1 => TickEvent::idle(n),
        2 | 3 => {
            let spikes = (0..n).map(|_| rng.bounded(4) == 0).collect();
            TickEvent::Binary(spikes)
        }
        _ => {
            let activations = (0..n)
                .map(|_| ACTIVATIONS[rng.bounded(ACTIVATIONS.len())])
                .collect();
            TickEvent::Graded(activations)
        }
    }
}
