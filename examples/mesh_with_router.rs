//! Wire a [`SynapticMesh`] to a [`ChannelRouter`] for a few ticks.
//!
//! The two types do not call each other. This example is the glue a new
//! user writes: build a graph, propagate spikes, then hand the resulting
//! currents to the router (and to the same [`NeuromodNeuron`] primitive the
//! router is built from).
//!
//! The graph is input → hidden → output. `propagate` is one hop, so the
//! hidden layer's spikes (chosen here by a threshold on this tick's
//! currents) arrive at the output on the next tick. The output layer has
//! no outgoing synapses.
//!
//! Run with `cargo run --example mesh_with_router`.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::generate_layered;
use synaptic_wiring::{ChannelRouter, NeuromodNeuron, RouterConfig};

/// Output-layer width, and the router's channel count.
const CHANNELS: usize = 4;
/// Neurons that only emit the first spike volley.
const INPUTS: usize = 8;
/// Middle layer. Its spikes are what reach the router on the next tick.
const HIDDEN: usize = 6;
/// Stand-in for a neuron model: a current above this becomes a spike.
const SPIKE_THRESHOLD: f32 = 0.2;

fn main() {
    // Fully connected, all excitatory, delay 0 so a spike arrives in the
    // same `propagate` call that injects it.
    let graph = generate_layered(&[INPUTS, HIDDEN, CHANNELS], 1.0, 0, 0.0)
        .expect("layered generator arguments are in range");
    let mut mesh = SynapticMesh::new(graph);
    let neuron_count = mesh.neuron_count();
    let output_start = neuron_count - CHANNELS;

    let mut router = ChannelRouter::try_with_config(RouterConfig {
        channel_count: CHANNELS,
        ..RouterConfig::default()
    })
    .expect("channel count is within the router limit");

    // One integrate-and-fire unit per output channel, driven by that
    // channel's mesh current. `ChannelRouter` owns a separate bank built
    // from `RouterConfig`; these are the caller's copy of the same primitive.
    let mut readouts: Vec<NeuromodNeuron> = (0..CHANNELS)
        .map(|_| {
            let mut neuron = NeuromodNeuron::new();
            neuron.threshold = 0.25;
            neuron
        })
        .collect();

    let mut spikes = vec![false; neuron_count];
    for spike in spikes.iter_mut().take(INPUTS) {
        *spike = true;
    }

    for tick in 0..3 {
        let currents = mesh
            .propagate(&spikes)
            .expect("spike vector matches the neuron count");

        let signals = &currents[output_start..];
        let decision = router
            .route(signals)
            .expect("one finite sample per router channel");

        let mut fired = Vec::new();
        for (channel, neuron) in readouts.iter_mut().enumerate() {
            neuron.integrate(signals[channel].into());
            if neuron.check_fire().is_some() {
                fired.push(channel);
            }
        }

        let hidden_spikes = spikes[INPUTS..output_start].iter().filter(|s| **s).count();
        println!(
            "tick {tick}: hidden_spikes_in={hidden_spikes} active_channels={:?} firing_rates={:?} readout_spikes={fired:?}",
            decision.active_channels, decision.firing_rates,
        );

        // Next volley is whatever crossed the threshold. Input is not
        // clamped back on, so the wave moves hidden → output and then stops.
        spikes = currents
            .iter()
            .map(|&current| current > SPIKE_THRESHOLD)
            .collect();
    }
}
