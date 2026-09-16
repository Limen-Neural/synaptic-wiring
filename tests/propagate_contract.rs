// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consumer contract for [`SynapticMesh::propagate`].
//!
//! Everything a simulation loop built on this crate depends on is pinned
//! here against a hand-built graph: **who** receives current, with which
//! **sign**, at which **magnitude**, on which **tick**.
//!
//! # The contract
//!
//! 1. A spike from neuron `s` on tick `t` delivers `weight` to every target
//!    of `s` on tick `t + delay`. `delay = 0` means same-tick delivery, so
//!    the current appears in the return value of the very call that fed the
//!    spike in.
//! 2. The delivered magnitude is the descriptor's `weight` and the sign is
//!    that descriptor's [`Polarity`], so an inhibitory synapse subtracts.
//!    This fixture follows Dale's law (a source's synapses share a
//!    polarity) because the generators do, but `from_descriptors` applies
//!    each descriptor's polarity as given rather than enforcing it.
//! 3. Currents arriving at the same neuron on the same tick are summed.
//! 4. `propagate` is **one hop**. Received current never becomes a spike on
//!    its own — the caller owns the neuron model and decides what fires next
//!    (see [`threshold_and_fire_loop_is_deterministic`]).
//! 5. Everything is deterministic: same graph plus same spike sequence gives
//!    bit-identical currents, run after run and after [`SynapticMesh::reset`].
//!
//! `propagate_graded` keeps rules 1, 3, and 4 and multiplies each delivery by
//! the source's activation — including its sign, so a negative activation
//! inverts the synapse's polarity (see
//! [`negative_graded_activation_inverts_polarity`]).
//!
//! # Fixture
//!
//! Four neurons, five synapses, obeying Dale's law (neuron 2 is inhibitory,
//! so all of its outgoing synapses are negative):
//!
//! ```text
//!            w=0.75 d=0                w=0.40 d=1
//!    (0) ──────────────────▶ (1) ──────────────────▶ (3)
//!     │  exc                   exc                    ▲
//!     │                                               │
//!     │  w=0.50 d=1                       w=0.60 d=1  │  inhibitory
//!     ├──────────────────────▶ (2) ───────────────────┤
//!     │  exc                   inh                    │
//!     │              w=0.25 d=2                       │
//!     └───────────────────────────────────────────────┘
//!                    exc
//! ```

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::SynapticGraph;
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

/// Neuron count of the fixture graph.
const N: usize = 4;

/// Absolute tolerance for comparing `f32` currents. Weights such as 0.40 and
/// 0.60 are not exactly representable in `f32`, so sums land within ~1e-8 of
/// the decimal values written here; this tolerance absorbs that while staying
/// far below the differences the assertions are meant to catch.
const EPS: f32 = 1e-6;

/// Build the fixture described in the module docs.
///
/// Deterministic by construction: no generator, no hashing, no RNG.
fn fixture() -> SynapticMesh {
    let descriptors = [
        // source, target, weight, delay, polarity
        (0, 1, 0.75, 0, Polarity::Excitatory),
        (0, 2, 0.50, 1, Polarity::Excitatory),
        (0, 3, 0.25, 2, Polarity::Excitatory),
        (1, 3, 0.40, 1, Polarity::Excitatory),
        (2, 3, 0.60, 1, Polarity::Inhibitory),
    ]
    .map(
        |(source, target, weight, delay, polarity)| SynapseDescriptor {
            source,
            target,
            weight,
            delay,
            polarity,
        },
    );

    let graph = SynapticGraph::from_descriptors(N, &descriptors)
        .expect("fixture descriptors are in bounds");
    SynapticMesh::new(graph)
}

/// Spike vector with exactly the listed neurons firing.
fn spikes(firing: &[usize]) -> Vec<bool> {
    let mut v = vec![false; N];
    for &neuron in firing {
        v[neuron] = true;
    }
    v
}

/// Assert a full per-neuron current vector, naming the tick on failure.
fn assert_currents(tick: u64, actual: &[f32], expected: [f32; N]) {
    assert_eq!(actual.len(), N, "tick {tick}: wrong current vector length");
    for (neuron, (&got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < EPS,
            "tick {tick}, neuron {neuron}: expected {want}, got {got}"
        );
    }
}

/// One spike into neuron 0 — every delivery lands on a known neuron, with a
/// known sign and magnitude, on a known tick, and nothing arrives early or
/// twice.
#[test]
fn single_spike_delivers_to_known_destination_at_known_tick() {
    let mut mesh = fixture();

    // Tick 0: neuron 0 fires. Only its delay-0 synapse (0 → 1) lands now.
    let t0 = mesh.propagate(&spikes(&[0])).unwrap();
    assert_currents(0, &t0, [0.0, 0.75, 0.0, 0.0]);

    // Tick 1: nothing fires; the delay-1 synapse 0 → 2 arrives.
    let t1 = mesh.propagate(&spikes(&[])).unwrap();
    assert_currents(1, &t1, [0.0, 0.0, 0.50, 0.0]);

    // Tick 2: the delay-2 synapse 0 → 3 arrives.
    let t2 = mesh.propagate(&spikes(&[])).unwrap();
    assert_currents(2, &t2, [0.0, 0.0, 0.0, 0.25]);

    // Tick 3: the buffer is drained — no phantom re-delivery on wrap-around.
    let t3 = mesh.propagate(&spikes(&[])).unwrap();
    assert_currents(3, &t3, [0.0; N]);

    assert_eq!(mesh.tick(), 4, "one tick advanced per propagate() call");
}

/// An inhibitory source delivers `-weight`, and co-arriving excitatory and
/// inhibitory currents sum at the destination.
#[test]
fn polarity_signs_the_current_and_co_arrivals_sum() {
    let mut mesh = fixture();

    // Tick 0: neurons 1 and 2 fire together. Both project to neuron 3 with
    // delay 1, so their currents arrive on the same tick and are summed:
    // +0.40 (excitatory neuron 1) − 0.60 (inhibitory neuron 2) = −0.20.
    let t0 = mesh.propagate(&spikes(&[1, 2])).unwrap();
    assert_currents(0, &t0, [0.0; N]);

    let t1 = mesh.propagate(&spikes(&[])).unwrap();
    assert_currents(1, &t1, [0.0, 0.0, 0.0, -0.20]);
    assert!(t1[3] < 0.0, "net inhibition must stay negative");
}

/// The multi-hop pattern a consumer writes: threshold the returned currents
/// back into spikes. Pinning the whole tick-by-tick trace documents that
/// `propagate` itself is one hop and that the loop is reproducible.
#[test]
fn threshold_and_fire_loop_is_deterministic() {
    /// Currents above this fire the neuron on the next tick.
    const THRESHOLD: f32 = 0.3;

    fn run() -> Vec<Vec<f32>> {
        let mut mesh = fixture();
        let mut firing = spikes(&[0]);
        let mut trace = Vec::new();
        for _ in 0..4 {
            let currents = mesh.propagate(&firing).unwrap();
            firing = currents.iter().map(|&c| c > THRESHOLD).collect();
            trace.push(currents);
        }
        trace
    }

    let trace = run();

    // Tick 0: neuron 0 fires; 0 → 1 (delay 0) lands immediately. 0.75 is
    // over threshold, so neuron 1 fires on tick 1.
    assert_currents(0, &trace[0], [0.0, 0.75, 0.0, 0.0]);
    // Tick 1: 0 → 2 (delay 1) arrives. 0.50 is over threshold, so the
    // inhibitory neuron 2 fires on tick 2.
    assert_currents(1, &trace[1], [0.0, 0.0, 0.50, 0.0]);
    // Tick 2: neuron 3 collects 0 → 3 (delay 2, +0.25) and 1 → 3 (delay 1
    // from neuron 1's tick-1 spike, +0.40) = +0.65.
    assert_currents(2, &trace[2], [0.0, 0.0, 0.0, 0.65]);
    // Tick 3: neuron 2's tick-2 spike arrives as inhibition, −0.60.
    assert_currents(3, &trace[3], [0.0, 0.0, 0.0, -0.60]);

    assert_eq!(trace, run(), "same graph + same spikes ⇒ same currents");
}

/// `reset()` returns the mesh to its initial state: the tick counter goes
/// back to 0 and in-flight spikes are dropped rather than delivered late.
#[test]
fn reset_replays_identically_and_drops_in_flight_spikes() {
    let mut mesh = fixture();
    let first = mesh.propagate(&spikes(&[0])).unwrap();

    // Neuron 0's delay-1 and delay-2 deliveries are still queued here.
    mesh.reset();
    assert_eq!(mesh.tick(), 0);

    // Run past the longest delay: if reset had only rewound the tick
    // counter, those queued deliveries would surface on one of these ticks.
    for tick in 0..=mesh.max_delay() as u64 {
        let idle = mesh.propagate(&spikes(&[])).unwrap();
        assert_currents(tick, &idle, [0.0; N]);
    }

    mesh.reset();
    let replay = mesh.propagate(&spikes(&[0])).unwrap();
    assert_eq!(first, replay);
}

/// `propagate_graded` keeps the same destinations and delivery ticks, and
/// multiplies each delivery by the source activation.
#[test]
fn graded_activation_scales_the_delivered_current() {
    let mut mesh = fixture();

    // Neuron 0 at half strength: 0.5 × 0.75 = 0.375 on neuron 1 this tick.
    let t0 = mesh.propagate_graded(&[0.5, 0.0, 0.0, 0.0]).unwrap();
    assert_currents(0, &t0, [0.0, 0.375, 0.0, 0.0]);

    // Delays are unchanged by scaling: 0.5 × 0.50 = 0.25 on neuron 2 next tick.
    let t1 = mesh.propagate_graded(&[0.0; N]).unwrap();
    assert_currents(1, &t1, [0.0, 0.0, 0.25, 0.0]);
}

/// A negative activation multiplies the already-signed weight, so it flips
/// the synapse's polarity. That is a genuine multiplication, not a bug — but
/// it is a sharp edge worth pinning: pass activations ≥ 0 unless you mean it.
#[test]
fn negative_graded_activation_inverts_polarity() {
    // Excitatory 0 → 1 (delay 0) at −0.5 delivers −0.375, not +0.375.
    let mut mesh = fixture();
    let t0 = mesh.propagate_graded(&[-0.5, 0.0, 0.0, 0.0]).unwrap();
    assert_currents(0, &t0, [0.0, -0.375, 0.0, 0.0]);

    // Symmetrically, inhibitory neuron 2 turns excitatory:
    // −0.60 × −0.5 = +0.30 on neuron 3, one tick later.
    let mut mesh = fixture();
    let t0 = mesh.propagate_graded(&[0.0, 0.0, -0.5, 0.0]).unwrap();
    assert_currents(0, &t0, [0.0; N]);
    let t1 = mesh.propagate_graded(&[0.0; N]).unwrap();
    assert_currents(1, &t1, [0.0, 0.0, 0.0, 0.30]);
}

/// Guard rails a consumer can rely on: the spike vector must match the
/// neuron count, and the mesh reports the fixture's shape.
#[test]
fn mesh_reports_fixture_shape_and_rejects_mismatched_input() {
    let mut mesh = fixture();

    assert_eq!(mesh.neuron_count(), N);
    assert_eq!(mesh.synapse_count(), 5);
    assert_eq!(
        mesh.max_delay(),
        2,
        "buffer is sized from the graph's delays"
    );

    assert!(mesh.propagate(&[false; N - 1]).is_err());
    assert!(mesh.propagate(&[false; N + 1]).is_err());
}

/// Caller-buffer APIs are a drop-in reuse of the allocating contract: same
/// destinations, signs, magnitudes, and delivery ticks on this fixture.
#[test]
fn propagate_into_matches_allocating_on_fixture() {
    let mut alloc = fixture();
    let mut reuse = fixture();
    let sequence = [spikes(&[0]), spikes(&[]), spikes(&[]), spikes(&[1, 2])];
    let mut output = [0.0; N];
    for firing in &sequence {
        let allocated = alloc.propagate(firing).unwrap();
        reuse.propagate_into(firing, &mut output).unwrap();
        assert_eq!(allocated.as_slice(), output.as_slice());
        assert_eq!(alloc.tick(), reuse.tick());
    }

    let mut alloc = fixture();
    let mut reuse = fixture();
    let graded = [[0.5, 0.0, 0.0, 0.0], [0.0; N], [0.0, 0.0, -0.5, 0.0]];
    for activations in graded {
        let allocated = alloc.propagate_graded(&activations).unwrap();
        reuse
            .propagate_graded_into(&activations, &mut output)
            .unwrap();
        assert_eq!(allocated.as_slice(), output.as_slice());
        assert_eq!(alloc.tick(), reuse.tick());
    }
}
