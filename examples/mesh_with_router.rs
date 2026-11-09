//! Wire a [`SynapticMesh`] to a [`ChannelRouter`] for a few ticks.
//!
//! The two types do not call each other. This example is the glue a new
//! user writes: build a graph, propagate spikes, then hand the resulting
//! currents to the router (and to the same [`NeuromodNeuron`] primitive the
//! router is built from).
//!
//! Run with `cargo run --example mesh_with_router`.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::generate_layered;
use synaptic_wiring::{ChannelRouter, NeuromodNeuron, RouterConfig};

/// Output-layer width, and the router's channel count.
const CHANNELS: usize = 4;
/// Neurons that only emit spikes into the mesh.
const INPUTS: usize = 8;

fn main() {
    // Fully connected feed-forward layer, all excitatory, delay 0 so the
    // current arrives in the same `propagate` call.
    let graph = generate_layered(&[INPUTS, CHANNELS], 1.0, 0, 0.0)
        .expect("layered generator arguments are in range");
    let mut mesh = SynapticMesh::new(graph);
    let neuron_count = mesh.neuron_count();

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

    // One feed-forward hop: the output layer has no outgoing synapses, so
    // spikes there never come back into `propagate`. The same input is
    // presented on every tick.
    let mut spikes = vec![false; neuron_count];
    for spike in spikes.iter_mut().take(INPUTS) {
        *spike = true;
    }

    for tick in 0..3 {
        let currents = mesh
            .propagate(&spikes)
            .expect("spike vector matches the neuron count");

        let signals = &currents[neuron_count - CHANNELS..];
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

        println!(
            "tick {tick}: active_channels={:?} firing_rates={:?} readout_spikes={fired:?}",
            decision.active_channels, decision.firing_rates,
        );
    }
}
