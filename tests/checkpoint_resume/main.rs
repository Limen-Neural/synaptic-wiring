// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint/resume equivalence: a valid restored [`SynapticMesh`] must
//! continue **tick-for-tick identically** to the uninterrupted live mesh.
//!
//! Malformed checkpoints are rejected by the load-path tests in `src/mesh.rs`
//! and `src/delay/ring_buffer.rs` (LIM-1106 / LIM-1151). This suite never
//! repairs invalid state; it only serializes meshes that the constructors
//! and `propagate` already consider valid.
//!
//! # What is compared
//!
//! After the prefix, and again after every suffix tick, live and restored
//! copies must match on:
//!
//! - returned synaptic currents
//! - `tick`
//! - queued delay-buffer deliveries (the checkpoint's `delay_buffer` slots)
//! - the rest of the checkpoint snapshot (graph + buffer metadata)
//!
//! Restore is exercised through JSON and one non-JSON serde format (`postcard`,
//! a dev-only dependency).
//!
//! # CI vs nightly
//!
//! Default `cargo test` runs [`resume_equivalence_seeded_ci`]: 512 generated
//! cases from deterministic seeds, plus named boundary and regression
//! fixtures. That is the bounded CI profile.
//!
//! A longer ignored/nightly profile lives in [`resume_equivalence_nightly`]:
//!
//! ```text
//! cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
//! CHECKPOINT_RESUME_CASES=50000 cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
//! ```
//!
//! The default nightly length is 10_000 seeds. Override with
//! `CHECKPOINT_RESUME_CASES`. Failing seeds belong in [`REGRESSION_SEEDS`];
//! if the on-failure shrinker prints a smaller scenario, persist that as a
//! named fixture beside the other boundary tests.

mod compare;
mod generate;
mod harness;
mod recipes;

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::SynapticGraph;
use synaptic_wiring::types::Polarity;

use compare::{SerdeFormat, check_resume, meshes_equivalent};
use generate::{check_seed, empty_graph};
use harness::{
    CI_CASES, NIGHTLY_CASES_DEFAULT, NIGHTLY_CASES_ENV, REGRESSION_SEEDS, Recipe, Scenario,
    TickEvent, apply_event, checkpoint_snapshot, descriptor,
};

/// Seeded property suite for default CI: hundreds of generated graphs,
/// delays, signed weights, checkpoint locations, and suffix sequences.
#[test]
fn resume_equivalence_seeded_ci() {
    for seed in 0..CI_CASES {
        check_seed(seed);
    }
}

/// Replays persisted seeds so a once-failing case cannot silently disappear
/// from the generator stream.
#[test]
fn resume_equivalence_regression_seeds() {
    for &seed in REGRESSION_SEEDS {
        check_seed(seed);
    }
}

/// Narrow in-flight regression matching `checkpoint_resume_with_spikes_in_flight_matches_live`.
#[test]
fn regression_in_flight_delay2_json_and_postcard() {
    let scenario = Scenario {
        seed: u64::MAX,
        recipe: Recipe::CheckpointBeforeDelivery,
        neuron_count: 2,
        descriptors: vec![descriptor(0, 1, 1.0, 2, Polarity::Excitatory)],
        buffer_max_delay: 2,
        prefix: vec![TickEvent::fire(2, &[0])],
        suffix: vec![
            TickEvent::idle(2),
            TickEvent::fire(2, &[0]),
            TickEvent::idle(2),
            TickEvent::idle(2),
            TickEvent::fire(2, &[0]),
            TickEvent::idle(2),
            TickEvent::idle(2),
            TickEvent::idle(2),
        ],
    };
    check_resume(&scenario).unwrap();
}

/// Frozen JSON checkpoint captured immediately before the delay-2 delivery.
/// If serde field names or ring layout change, this fixture must be updated
/// deliberately rather than silently accepted.
#[test]
fn regression_frozen_json_in_flight_before_delivery() {
    let json = r#"{
        "graph": {
            "neuron_count": 2,
            "row_ptr": [0, 1, 1],
            "targets": [1],
            "weights": [1.0],
            "delays": [2],
            "polarities": ["Excitatory"]
        },
        "delay_buffer": {
            "slots": [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            "neuron_count": 2,
            "max_delay": 2,
            "current_tick": 1
        },
        "tick": 1
    }"#;
    let mut restored: SynapticMesh =
        serde_json::from_str(json).expect("fixture is a valid checkpoint");
    assert_eq!(restored.tick(), 1);

    let mut live = SynapticMesh::new(
        SynapticGraph::from_descriptors(2, &[descriptor(0, 1, 1.0, 2, Polarity::Excitatory)])
            .unwrap(),
    );
    assert_eq!(live.propagate(&[true, false]).unwrap()[1], 0.0);
    assert_eq!(checkpoint_snapshot(&live), checkpoint_snapshot(&restored));

    for spikes in [[false, false], [true, false], [false, false]] {
        let a = live.propagate(&spikes).unwrap();
        let b = restored.propagate(&spikes).unwrap();
        assert_eq!(a, b);
        assert_eq!(live.tick(), restored.tick());
        assert_eq!(checkpoint_snapshot(&live), checkpoint_snapshot(&restored));
    }
}

#[test]
fn boundary_empty_graph() {
    check_resume(&empty_graph(0)).unwrap();
}

#[test]
fn boundary_single_neuron_no_synapses() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::SingleNeuron,
        neuron_count: 1,
        descriptors: Vec::new(),
        buffer_max_delay: 0,
        prefix: vec![TickEvent::fire(1, &[0]), TickEvent::idle(1)],
        suffix: vec![TickEvent::Graded(vec![0.5]), TickEvent::idle(1)],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_single_neuron_self_loop_delay_zero() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::DelayZero,
        neuron_count: 1,
        descriptors: vec![descriptor(0, 0, 0.75, 0, Polarity::Excitatory)],
        buffer_max_delay: 0,
        prefix: vec![TickEvent::fire(1, &[0])],
        suffix: vec![TickEvent::idle(1), TickEvent::fire(1, &[0])],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_delay_zero_with_inhibitory_and_empty_ticks() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::DelayZero,
        neuron_count: 3,
        descriptors: vec![
            descriptor(0, 1, 0.5, 0, Polarity::Excitatory),
            descriptor(1, 2, 0.4, 0, Polarity::Inhibitory),
        ],
        buffer_max_delay: 0,
        prefix: vec![TickEvent::idle(3), TickEvent::fire(3, &[0, 1])],
        suffix: vec![TickEvent::idle(3), TickEvent::fire(3, &[1])],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_delay_capacity_exact_and_headroom() {
    for extra in [0usize, 3] {
        let delay = 5u16;
        let scenario = Scenario {
            seed: extra as u64,
            recipe: Recipe::DelayCapacity,
            neuron_count: 2,
            descriptors: vec![descriptor(0, 1, 1.0, delay, Polarity::Excitatory)],
            buffer_max_delay: usize::from(delay) + extra,
            prefix: vec![TickEvent::fire(2, &[0])],
            suffix: vec![TickEvent::idle(2); usize::from(delay) + 2],
        };
        check_resume(&scenario).unwrap();
    }
}

#[test]
fn boundary_multiple_spikes_same_slot_and_checkpoint_before_delivery() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::MultiSpikeSameSlot,
        neuron_count: 3,
        descriptors: vec![
            descriptor(0, 2, 0.3, 2, Polarity::Excitatory),
            descriptor(1, 2, 0.7, 2, Polarity::Inhibitory),
        ],
        buffer_max_delay: 2,
        prefix: vec![TickEvent::fire(3, &[0, 1]), TickEvent::idle(3)],
        suffix: vec![TickEvent::idle(3), TickEvent::idle(3)],
    };
    check_resume(&scenario).unwrap();

    let mut live = scenario.mesh();
    let _ = apply_event(&mut live, &scenario.prefix[0]);
    let _ = apply_event(&mut live, &scenario.prefix[1]);
    let mut restored = SerdeFormat::Json.restore(&live);
    let live_delivery = apply_event(&mut live, &TickEvent::idle(3));
    let restored_delivery = apply_event(&mut restored, &TickEvent::idle(3));
    assert_eq!(live_delivery, restored_delivery);
    assert!((live_delivery[2] - (0.3 - 0.7)).abs() < 1e-6);
}

#[test]
fn cross_format_restore_equivalence() {
    let seeds = [0u64, 1, 3, 4, 5, 8, 42, 255];
    for seed in seeds {
        let scenario = Scenario::from_seed(seed);
        let mut live = scenario.mesh();
        for event in &scenario.prefix {
            let _ = apply_event(&mut live, event);
        }
        let from_json = SerdeFormat::Json.restore(&live);
        let from_postcard = SerdeFormat::Postcard.restore(&live);
        meshes_equivalent(&live, &from_json).unwrap();
        meshes_equivalent(&live, &from_postcard).unwrap();
        meshes_equivalent(&from_json, &from_postcard).unwrap();

        // JSON → postcard → JSON must not drift for a valid checkpoint.
        let via_postcard = SerdeFormat::Postcard.restore(&from_json);
        let via_json = SerdeFormat::Json.restore(&from_postcard);
        meshes_equivalent(&live, &via_postcard).unwrap();
        meshes_equivalent(&live, &via_json).unwrap();
    }
}

/// Deeper seeded run, skipped by default CI.
///
/// ```text
/// cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
/// CHECKPOINT_RESUME_CASES=50000 cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
/// ```
#[test]
#[ignore = "nightly profile: CHECKPOINT_RESUME_CASES (default 10000) seeded resume-equivalence run"]
fn resume_equivalence_nightly() {
    let cases = std::env::var(NIGHTLY_CASES_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(NIGHTLY_CASES_DEFAULT);
    let start = CI_CASES;
    for seed in start..start.saturating_add(cases) {
        check_seed(seed);
    }
}
