// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DelayModel` descriptor contract (GH#100 / LIM-1559):
//! `assign_delays_from_model` must reproduce `assign_delays` for equivalent
//! models and document where it differs.

use synaptic_wiring::topology::{assign_delays, assign_delays_from_model};
use synaptic_wiring::types::{DelayModel, Polarity, SynapseDescriptor};

fn uniform(min_delay: u16, max_delay: u16) -> DelayModel {
    DelayModel::UniformRandom {
        min_delay,
        max_delay,
    }
}

fn descriptors(n: u32) -> Vec<SynapseDescriptor> {
    let mut out = Vec::new();
    for s in 0..n {
        for t in 0..n {
            if s != t {
                out.push(SynapseDescriptor {
                    source: s,
                    target: t,
                    weight: 1.0,
                    delay: 99,
                    polarity: Polarity::Excitatory,
                });
            }
        }
    }
    out
}

fn positions(n: u32) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| [i as f32 * 1.7, (i % 3) as f32 * 2.5, (i % 5) as f32])
        .collect()
}

fn legacy_delays(pos: Option<&[[f32; 3]]>, speed: f32, max: u16) -> Vec<u16> {
    let mut d = descriptors(12);
    assign_delays(&mut d, pos, speed, max);
    d.iter().map(|x| x.delay).collect()
}

fn model_delays(pos: Option<&[[f32; 3]]>, model: &DelayModel) -> Vec<u16> {
    let mut d = descriptors(12);
    assign_delays_from_model(&mut d, pos, model);
    d.iter().map(|x| x.delay).collect()
}

#[test]
fn distance_model_with_positions_matches_assign_delays() {
    let pos = positions(12);
    for (speed, max) in [(1.0, 1), (1.0, 8), (0.5, 20), (3.0, 4)] {
        let model = DelayModel::DistanceProportional {
            speed,
            min_delay: 1,
            max_delay: max,
        };
        assert_eq!(
            model_delays(Some(&pos), &model),
            legacy_delays(Some(&pos), speed, max)
        );
    }
}

#[test]
fn index_path_matches_assign_delays_without_positions() {
    for max in [1, 2, 8, 200] {
        let legacy = legacy_delays(None, 1.0, max);
        assert_eq!(model_delays(None, &uniform(1, max)), legacy);
        let distance = DelayModel::DistanceProportional {
            speed: 1.0,
            min_delay: 1,
            max_delay: max,
        };
        assert_eq!(model_delays(None, &distance), legacy);
    }
}

#[test]
fn index_path_respects_shifted_range() {
    for (min, max) in [(0, 6), (3, 9), (5, 5), (0, 0)] {
        let d = model_delays(None, &uniform(min, max));
        assert!(d.iter().all(|&x| (min..=max).contains(&x)), "{min}..={max}");
    }
    let legacy = legacy_delays(None, 1.0, 6);
    let expected: Vec<u16> = legacy.iter().map(|d| d + 3).collect();
    assert_eq!(model_delays(None, &uniform(4, 9)), expected);
}

#[test]
fn fixed_model_ignores_positions() {
    let pos = positions(12);
    let model = DelayModel::Fixed { delay: 4 };
    assert!(model_delays(Some(&pos), &model).iter().all(|&d| d == 4));
    assert!(model_delays(None, &model).iter().all(|&d| d == 4));
}

#[test]
fn inverted_range_falls_back_to_fixed_min_delay() {
    let pos = positions(12);
    let distance = DelayModel::DistanceProportional {
        speed: 1.0,
        min_delay: 6,
        max_delay: 2,
    };
    assert!(model_delays(Some(&pos), &distance).iter().all(|&d| d == 6));
    assert!(model_delays(None, &uniform(6, 2)).iter().all(|&d| d == 6));
}

#[test]
fn distance_model_max_zero_differs_from_assign_delays() {
    // Documented edge case: { min: 1, max: 0 } is an inverted range, so it
    // falls back to Fixed { delay: 1 }, while assign_delays(max = 0) gives 0.
    let pos = positions(12);
    let model = DelayModel::DistanceProportional {
        speed: 1.0,
        min_delay: 1,
        max_delay: 0,
    };
    assert!(model_delays(Some(&pos), &model).iter().all(|&d| d == 1));
    assert!(model_delays(None, &model).iter().all(|&d| d == 1));
    assert!(legacy_delays(Some(&pos), 1.0, 0).iter().all(|&d| d == 0));
    assert!(legacy_delays(None, 1.0, 0).iter().all(|&d| d == 0));
}

#[test]
fn short_positions_panic_leaves_descriptors_unchanged() {
    let mut d = descriptors(12);
    let pos = positions(11);
    let model = DelayModel::DistanceProportional {
        speed: 1.0,
        min_delay: 1,
        max_delay: 10,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assign_delays_from_model(&mut d, Some(&pos), &model);
    }));
    assert!(result.is_err());
    assert!(d.iter().all(|x| x.delay == 99));
}
