// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::MeshError;
use crate::router::{ChannelRouter, NeuromodNeuron, NeuromodState, RouterConfig};

#[test]
fn channel_0_pulse_activates_channel_0() {
    let mut router = ChannelRouter::new();
    let d = router.route([1.0, 0.0, 0.0]).unwrap();
    assert!(
        d.is_active(0),
        "Channel 0 should be active, firing rate was {:?}",
        d.firing_rates[0]
    );
    assert!(!d.is_active(1));
    assert!(!d.is_active(2));
}

#[test]
fn channel_1_pulse_activates_channel_1() {
    let mut router = ChannelRouter::new();
    let d = router.route([0.0, 1.0, 0.0]).unwrap();
    assert!(d.is_active(1));
    assert!(!d.is_active(0));
}

#[test]
fn background_noise_routes_nowhere() {
    let mut router = ChannelRouter::new();
    let d = router.route([0.05, 0.05, 0.05]).unwrap();
    assert!(d.is_empty());
}

#[test]
fn firing_rates_in_range() {
    let mut router = ChannelRouter::new();
    let d = router.route([0.8, 0.2, 0.1]).unwrap();
    for &rate in &d.firing_rates {
        assert!(
            (0.0..=1.0).contains(&rate),
            "firing rate out of range: {rate}"
        );
    }
}

#[test]
fn positive_feedback_increases_weight() {
    let mut router = ChannelRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, 1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after > w_before);
}

#[test]
fn negative_feedback_decreases_weight() {
    let mut router = ChannelRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, -1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after < w_before);
}

#[test]
fn global_gain_inhibits_firing() {
    let mut router = ChannelRouter::new();
    let d1 = router.route([0.5, 0.0, 0.0]).unwrap();
    assert!(d1.is_active(0));

    router.set_global_gain(0.1);
    let d2 = router.route([0.5, 0.0, 0.0]).unwrap();
    assert!(d2.is_empty(), "Reduced gain should have inhibited firing");
}

#[test]
fn total_routes_increments() {
    let mut router = ChannelRouter::new();
    assert_eq!(router.total_routes, 0);
    router.route([0.0, 0.0, 0.0]).unwrap();
    router.route([0.0, 0.0, 0.0]).unwrap();
    assert_eq!(router.total_routes, 2);
}

#[test]
fn neuromod_neuron_fires_above_threshold() {
    let mut n = NeuromodNeuron::new();
    n.threshold = 0.1;
    n.leak = 0.0;
    n.integrate(0.5);
    assert!(n.check_fire().is_some());
    assert_eq!(n.v, 0.0);
}

#[test]
fn neuromod_neuron_no_fire_below_threshold() {
    let mut n = NeuromodNeuron::new();
    n.threshold = 1.0;
    n.integrate(0.05);
    assert!(n.check_fire().is_none());
    assert!(n.v > 0.0);
}

// ── Generic channel count tests ───────────────────────────────────────────────

#[test]
fn five_channel_router_routes_correctly() {
    let config = RouterConfig {
        channel_count: 5,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router.route([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
    assert!(!d.is_active(1));
    assert!(!d.is_active(4));
    assert_eq!(d.firing_rates.len(), 5);
}

#[test]
fn eight_channel_router_routes_correctly() {
    let config = RouterConfig {
        channel_count: 8,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router
        .route([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        .unwrap();
    assert!(d.is_active(3));
    assert!(!d.is_active(0));
    assert_eq!(d.firing_rates.len(), 8);
}

#[test]
fn single_channel_router_always_routes() {
    let config = RouterConfig {
        channel_count: 1,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router.route([1.0]).unwrap();
    assert!(d.is_active(0));
    assert_eq!(d.firing_rates.len(), 1);
}

#[test]
fn custom_config_weights_applied() {
    let config = RouterConfig {
        channel_count: 3,
        self_weight: 1.2,
        cross_weight: -0.2,
        ..RouterConfig::default()
    };
    let router = ChannelRouter::with_config(config);
    let m = router.weight_matrix();
    assert!((m[0][0] - 1.2).abs() < 1e-6);
    assert!((m[0][1] - (-0.2)).abs() < 1e-6);
}

#[test]
fn route_accepts_array_by_value() {
    let mut router = ChannelRouter::new();
    let d = router.route([1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
}

#[test]
fn route_rejects_mismatched_length() {
    let mut router = ChannelRouter::new();
    let result = router.route([1.0, 0.0]);
    assert!(result.is_err());
}

#[test]
fn route_error_context_reports_public_method_name() {
    // The error returned by `route()` on a signal-length mismatch must use the
    // "route signals" context (matching the original pre-neuromodulation API),
    // not "route_modulated signals" — because callers of `route()` never
    // invoked `route_modulated` and would be confused by that string.
    let mut router = ChannelRouter::new();
    let err = router.route([1.0, 0.0]).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("route signals"),
        "route() error must reference \"route signals\", got: {msg}"
    );
    assert!(
        !msg.contains("route_modulated"),
        "route() error must NOT leak the internal method name, got: {msg}"
    );
}

#[test]
fn route_modulated_error_context_reports_internal_method_name() {
    // Conversely, `route_modulated()` reports its own name in the error.
    let mut router = ChannelRouter::new();
    let err = router
        .route_modulated([1.0, 0.0], &NeuromodState::balanced())
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("route_modulated signals"),
        "route_modulated() error must reference \"route_modulated signals\", got: {msg}"
    );
}

#[test]
fn deserialized_router_with_empty_inner_baseline_weights_recovers() {
    // Regression test: a router deserialized from a payload that was produced
    // with the wrong inner-row shape for `baseline_weights` (e.g. `[[], [], []]`)
    // used to panic at the first `apply_plasticity` call. The lazy repair in
    // `ensure_neuromod_state_synced` must now rebuild the full 2D table from
    // the current neuron weights when ANY row is the wrong size.
    //
    // We drive the test through the public serde API: build a valid router,
    // serialize it, mutate the JSON to drop the inner rows of `baseline_weights`,
    // and confirm the router self-heals on the first modulated route instead
    // of panicking at `apply_plasticity`.
    let router = ChannelRouter::new();
    let mut json: serde_json::Value =
        serde_json::to_value(&router).expect("Fresh router must serialize");
    // Force the malformed-payload case: outer length matches channel count (3)
    // but each row is empty.
    json["baseline_weights"] = serde_json::json!([[], [], []]);

    let mut router: ChannelRouter = serde_json::from_value(json)
        .expect("Malformed-payload router should still deserialize (lazy repair on first route)");

    // The first modulated route must self-heal instead of panicking on
    // `apply_plasticity` indexing `baseline_weights[i][j]`.
    let result = router.route_modulated([0.5, 0.0, 0.0], &NeuromodState::balanced());
    assert!(
        result.is_ok(),
        "Malformed baseline_weights must self-heal on first route: {:?}",
        result.err()
    );
}

#[test]
fn deserialized_router_with_missing_neuron_fields_recovers() {
    // Regression test: older or hand-authored payloads may omit fields from a
    // NeuromodNeuron. Deserialization must use the neuron's defaults so the
    // router's existing lazy state repair can run on the first route.
    let router = ChannelRouter::new();
    let mut json: serde_json::Value =
        serde_json::to_value(&router).expect("Fresh router must serialize");
    json["neurons"][0]
        .as_object_mut()
        .unwrap()
        .remove("weights");

    let mut router: ChannelRouter = serde_json::from_value(json)
        .expect("Missing neuron fields should deserialize using defaults");

    let result = router.route_modulated([0.5, 0.0, 0.0], &NeuromodState::balanced());
    assert!(
        result.is_ok(),
        "Router with missing neuron fields must self-heal on first route: {:?}",
        result.err()
    );
}

#[test]
fn apply_feedback_on_deserialized_router_with_malformed_baseline_does_not_panic() {
    // Regression test for chatgpt-codex P2 thread #NyuBl / devin-ai BUG #NytR0.
    // Calling `apply_feedback` on a deserialized router whose `baseline_weights`
    // outer length matches the channel count but whose inner rows are empty
    // (the same shape that `route_modulated` already repairs) used to panic at
    // `baseline_weights[channel_idx][channel_idx]` inside `sync_baseline_after_feedback`.
    //
    // The fix: `apply_feedback` now calls `ensure_neuromod_state_synced` at the
    // top, which rebuilds the full 2D table from the current neuron weights
    // before any indexing happens. This test exercises the path *without* going
    // through `route_modulated` first.
    let router = ChannelRouter::new();
    let mut json: serde_json::Value =
        serde_json::to_value(&router).expect("Fresh router must serialize");
    // Force the malformed-payload case: outer length matches channel count (3)
    // but each row is empty.
    json["baseline_weights"] = serde_json::json!([[], [], []]);

    let mut router: ChannelRouter =
        serde_json::from_value(json).expect("Malformed-payload router should still deserialize");

    // `apply_feedback` must self-heal instead of panicking on
    // `baseline_weights[channel_idx][channel_idx]`.
    router.apply_feedback(0, 1.0);

    // And the self-heal must leave a well-formed `baseline_weights` table behind.
    let w = router.weight_matrix();
    assert_eq!(w.len(), 3, "channel count must be intact after feedback");
    assert_eq!(w[0].len(), 3, "weights[0] must be intact after feedback");
}

#[test]
#[should_panic(expected = "routing_timesteps must be > 0")]
fn zero_routing_timesteps_panics() {
    let config = RouterConfig {
        routing_timesteps: 0,
        ..RouterConfig::default()
    };
    let _router = ChannelRouter::with_config(config);
}

// ── Neuromodulatory routing tests ─────────────────────────────────────────────

#[test]
fn dopamine_increases_channel_conductance() {
    let mut router = ChannelRouter::new();
    // Baseline: weak signal should not activate.
    let d1 = router.route([0.15, 0.0, 0.0]).unwrap();
    let baseline_active = d1.is_active(0);

    // With dopamine: same weak signal should have lower threshold.
    let mods = NeuromodState {
        dopamine: 0.8,
        ..NeuromodState::default()
    };
    let d2 = router.route_modulated([0.15, 0.0, 0.0], &mods).unwrap();
    // Dopamine makes it easier to fire, so if it wasn't active before,
    // it might be now. If it was active before, it should still be.
    if !baseline_active {
        // Dopamine should help weak signal cross threshold.
        assert!(
            d2.is_active(0) || d2.firing_rates[0] > d1.firing_rates[0],
            "Dopamine should increase conductance"
        );
    }
}

#[test]
fn cortisol_increases_channel_resistance() {
    let mut router = ChannelRouter::new();
    // Baseline: moderate signal should activate.
    let d1 = router.route([0.5, 0.0, 0.0]).unwrap();
    let baseline_rate = d1.firing_rates[0];

    // With cortisol: same signal should have higher threshold.
    let mods = NeuromodState {
        cortisol: 0.8,
        ..NeuromodState::default()
    };
    let d2 = router.route_modulated([0.5, 0.0, 0.0], &mods).unwrap();
    // Cortisol should reduce firing rate.
    assert!(
        d2.firing_rates[0] <= baseline_rate,
        "Cortisol should increase resistance (reduce firing rate)"
    );
}

#[test]
fn serotonin_increases_leak_reduces_persistence() {
    let mut router = ChannelRouter::new();
    // Baseline firing rate.
    let d1 = router.route([0.8, 0.0, 0.0]).unwrap();
    let baseline_rate = d1.firing_rates[0];

    // With serotonin: higher leak should reduce firing.
    let mods = NeuromodState {
        serotonin: 0.8,
        ..NeuromodState::default()
    };
    let d2 = router.route_modulated([0.8, 0.0, 0.0], &mods).unwrap();
    // Serotonin should reduce firing rate due to higher leak.
    assert!(
        d2.firing_rates[0] <= baseline_rate,
        "Serotonin should increase leak (reduce persistence)"
    );
}

#[test]
fn active_channel_strengthens_with_dopamine() {
    let config = RouterConfig {
        plasticity_potentiate: 0.1,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let w_before = router.weight_matrix()[0][0];

    // Route with high dopamine — channel 0 should activate and strengthen.
    let mods = NeuromodState {
        dopamine: 1.0,
        ..NeuromodState::default()
    };
    let d = router.route_modulated([1.0, 0.0, 0.0], &mods).unwrap();
    assert!(d.is_active(0), "Channel 0 should be active");

    let w_after = router.weight_matrix()[0][0];
    assert!(
        w_after > w_before,
        "Active channel should strengthen with dopamine: {w_before} -> {w_after}"
    );
}

#[test]
fn inactive_channel_weakens_over_time() {
    let config = RouterConfig {
        plasticity_decay: 0.1,
        plasticity_potentiate: 0.2,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);

    // First activate channel 1 to strengthen it above baseline.
    let _ = router.route([0.0, 1.0, 0.0]).unwrap();
    let w_strengthened = router.weight_matrix()[1][1];
    assert!(
        w_strengthened > 0.9,
        "Channel 1 should strengthen after activation"
    );

    // Now route multiple times with signal only on channel 0.
    // Channel 1 is inactive and should decay toward baseline.
    for _ in 0..10 {
        let _ = router.route([1.0, 0.0, 0.0]).unwrap();
    }

    let w_after = router.weight_matrix()[1][1];
    assert!(
        w_after < w_strengthened,
        "Inactive channel should weaken (use-it-or-lose-it): {w_strengthened} -> {w_after}"
    );
}

#[test]
fn fatigue_accumulates_with_use() {
    let mut router = ChannelRouter::new();
    let fatigue_before = router.channel_fatigue[0];

    // Route with strong signal on channel 0.
    let _ = router.route([1.0, 0.0, 0.0]).unwrap();

    let fatigue_after = router.channel_fatigue[0];
    assert!(
        fatigue_after > fatigue_before,
        "Fatigue should accumulate with activation: {fatigue_before} -> {fatigue_after}"
    );
}

#[test]
fn fatigue_recovery_when_inactive() {
    let mut router = ChannelRouter::new();
    // Activate channel 0.
    let _ = router.route([1.0, 0.0, 0.0]).unwrap();
    let fatigue_after_active = router.channel_fatigue[0];
    assert!(fatigue_after_active > 0.0);

    // Route with signal on channel 1 (channel 0 inactive).
    let _ = router.route([0.0, 1.0, 0.0]).unwrap();
    let fatigue_after_inactive = router.channel_fatigue[0];
    assert!(
        fatigue_after_inactive < fatigue_after_active,
        "Fatigue should recover when inactive: {fatigue_after_active} -> {fatigue_after_inactive}"
    );
}

#[test]
fn cortisol_amplifies_fatigue_effect() {
    let mut router = ChannelRouter::new();
    // Build up some fatigue on channel 0.
    for _ in 0..3 {
        let _ = router.route([1.0, 0.0, 0.0]).unwrap();
    }
    let fatigue = router.channel_fatigue[0];
    assert!(fatigue > 0.0);

    // Baseline rate without cortisol.
    let d1 = router.route([0.5, 0.0, 0.0]).unwrap();
    let rate_no_stress = d1.firing_rates[0];

    // With cortisol: fatigue should have stronger effect.
    let mods = NeuromodState {
        cortisol: 1.0,
        ..NeuromodState::default()
    };
    let d2 = router.route_modulated([0.5, 0.0, 0.0], &mods).unwrap();
    let rate_stressed = d2.firing_rates[0];

    assert!(
        rate_stressed <= rate_no_stress,
        "Cortisol should amplify fatigue effect: {rate_no_stress} -> {rate_stressed}"
    );
}

#[test]
fn dopamine_counteracts_fatigue() {
    let mut router = ChannelRouter::new();
    // Build up fatigue on channel 0.
    for _ in 0..5 {
        let _ = router.route([1.0, 0.0, 0.0]).unwrap();
    }
    let fatigue = router.channel_fatigue[0];
    assert!(fatigue > 0.3, "Should have significant fatigue");

    // Baseline rate with fatigue.
    let d1 = router.route([0.5, 0.0, 0.0]).unwrap();
    let rate_no_dopamine = d1.firing_rates[0];

    // With dopamine: should counteract fatigue.
    let mods = NeuromodState {
        dopamine: 1.0,
        ..NeuromodState::default()
    };
    let d2 = router.route_modulated([0.5, 0.0, 0.0], &mods).unwrap();
    let rate_dopamine = d2.firing_rates[0];

    assert!(
        rate_dopamine >= rate_no_dopamine,
        "Dopamine should counteract fatigue: {rate_no_dopamine} -> {rate_dopamine}"
    );
}

#[test]
fn least_resistance_pathway_routing() {
    // The router should prefer channels with less fatigue. Heavily-fatigued
    // channels should have a measurably lower firing rate than fresh ones
    // under the same input, because the effective threshold scales up with
    // fatigue.
    //
    // The previous version of this test built fatigue up by running 10 routes
    // of `[1.0, 0.0, 0.0]`, but that ALSO strengthened `weights[0][0]` via
    // dopamine-independent potentiation in `apply_plasticity`. The stronger
    // self-weight compensated the fatigue-induced threshold bump, so the
    // measured rate did not actually drop. We now inject fatigue directly via
    // the public `channel_fatigue` field and keep the weight matrix at its
    // baseline — isolating the fatigue → threshold → rate mechanism.
    //
    // Signal / threshold tuned so the equilibrium membrane potential hovers
    // right at the fresh-router threshold. With fatigue the equilibrium stays
    // strictly sub-threshold, driving the fatigued rate to 0 while the fresh
    // rate remains > 0. This produces a clean, unambiguous gap.
    let config = RouterConfig {
        channel_count: 3,
        threshold: 0.25,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);

    let low_cortisol = NeuromodState {
        cortisol: 0.3,
        ..NeuromodState::default()
    };

    // Baseline: fresh router, channel 0 fatigue = 0.
    // Effective threshold ≈ 0.25 * 1.15 = 0.2875. Stimulus = 0.18 → fire.
    let d_baseline = router
        .route_modulated([0.3, 0.3, 0.3], &low_cortisol)
        .unwrap();
    let rate_baseline = d_baseline.firing_rates[0];

    // Inject high fatigue on channel 0 directly. Weights are unchanged from
    // baseline, so any rate change is purely due to the fatigue-driven
    // threshold increase.
    // Effective threshold ≈ 0.25 * 1.15 * 1.3 = 0.374. Stimulus = 0.18 < 0.374
    // → no firing.
    router.channel_fatigue[0] = 1.0;
    let d_fatigued = router
        .route_modulated([0.3, 0.3, 0.3], &low_cortisol)
        .unwrap();
    let rate_fatigued = d_fatigued.firing_rates[0];

    assert!(
        rate_fatigued < rate_baseline,
        "Fatigued channel 0 should have lower firing rate: {rate_baseline} -> {rate_fatigued}"
    );
}

#[test]
fn cortisol_increases_resistance_on_fresh_router() {
    // Regression test: on a brand-new router (all fatigue = 0), high cortisol
    // must still raise the effective threshold and reduce firing. Prior to the
    // fix, `fatigue_factor = 1 + cortisol * fatigue == 1.0` collapsed to the
    // balanced baseline and cortisol had no effect at all.
    //
    // Signal 0.05 picked so the equilibrium membrane potential under the
    // balanced threshold (0.22) drives sustained firing, but with the
    // stressed threshold (0.22 * 1.5 = 0.33) it stays sub-threshold for the
    // full 16-timestep integration window. This gives a clear, reproducible
    // gap between balanced and stressed rates on a fresh router.
    //
    // Two independent fresh routers are used so that the calm path's
    // `apply_plasticity` side effects (channel 0 fatigue accumulation and
    // dopamine-independent potentiation of `weights[0][0]`) cannot leak into
    // the stressed path. The stressed call must stand on its own as a
    // "brand-new router with zero fatigue" measurement.
    let mut calm_router = ChannelRouter::new();
    let mut stressed_router = ChannelRouter::new();
    assert!(
        calm_router.channel_fatigue.iter().all(|&f| f == 0.0)
            && stressed_router.channel_fatigue.iter().all(|&f| f == 0.0),
        "Pre-condition: both fresh routers have zero fatigue on every channel"
    );

    let d_calm = calm_router
        .route_modulated([0.05, 0.0, 0.0], &NeuromodState::balanced())
        .unwrap();
    let rate_calm = d_calm.firing_rates[0];

    let stressed = NeuromodState {
        cortisol: 1.0,
        ..NeuromodState::default()
    };
    let d_stressed = stressed_router
        .route_modulated([0.05, 0.0, 0.0], &stressed)
        .unwrap();
    let rate_stressed = d_stressed.firing_rates[0];

    assert!(
        rate_stressed < rate_calm,
        "Cortisol must raise the threshold on a fresh router: \
         calm={rate_calm}, stressed={rate_stressed}"
    );
}

#[test]
fn router_config_backward_compatible_serde() {
    // Regression test: a config serialized before the plasticity/fatigue
    // fields were added must still deserialize. Prior to the fix, the
    // missing fields caused `serde_json` / `bincode` to error.
    let old_json = r#"{
        "channel_count": 3,
        "self_weight": 0.9,
        "cross_weight": -0.15,
        "threshold": 0.22,
        "leak": 0.12,
        "routing_timesteps": 16,
        "min_fire_rate": 0.1875
    }"#;
    let cfg: RouterConfig = serde_json::from_str(old_json)
        .expect("Old RouterConfig JSON must deserialize with new defaults");
    assert_eq!(cfg.channel_count, 3);
    assert!((cfg.plasticity_decay - 0.02).abs() < 1e-6);
    assert!((cfg.plasticity_potentiate - 0.05).abs() < 1e-6);
    assert!((cfg.plasticity_speed - 0.1).abs() < 1e-6);
    assert!((cfg.fatigue_accumulation - 0.15).abs() < 1e-6);
    assert!((cfg.fatigue_recovery - 0.05).abs() < 1e-6);
}

#[test]
fn plasticity_speed_tunes_adaptation_rate() {
    // Higher plasticity_speed → faster convergence to the amplified target.
    let cfg_slow = RouterConfig {
        plasticity_potentiate: 0.2,
        plasticity_speed: 0.05,
        ..RouterConfig::default()
    };
    let cfg_fast = RouterConfig {
        plasticity_potentiate: 0.2,
        plasticity_speed: 0.5,
        ..RouterConfig::default()
    };

    let mut slow = ChannelRouter::with_config(cfg_slow);
    let mut fast = ChannelRouter::with_config(cfg_fast);

    let _ = slow.route([1.0, 0.0, 0.0]).unwrap();
    let _ = fast.route([1.0, 0.0, 0.0]).unwrap();

    let w_slow = slow.weight_matrix()[0][0];
    let w_fast = fast.weight_matrix()[0][0];

    assert!(
        w_fast > w_slow,
        "Faster plasticity_speed should produce a larger weight after one active route: \
         slow={w_slow}, fast={w_fast}"
    );
}

// ── Non-finite ingress (LIM-1229) ─────────────────────────────────────────────

fn router_snapshot(router: &ChannelRouter) -> serde_json::Value {
    serde_json::to_value(router).expect("router must serialize")
}

fn assert_internal_state_eq(left: &ChannelRouter, right: &ChannelRouter) {
    assert_eq!(left.total_routes, right.total_routes);
    assert_eq!(left.fatigue(), right.fatigue());
    assert_eq!(left.weight_matrix(), right.weight_matrix());
    assert_eq!(
        router_snapshot(left),
        router_snapshot(right),
        "serialized internal state (neurons, baseline weights, config) must match"
    );
}

fn non_finite_values() -> [f32; 3] {
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
}

fn assert_non_finite_signal(err: MeshError, expected_index: usize, expected_context: &str) {
    let msg = format!("{err}");
    match err {
        MeshError::NonFiniteSignal { index, context } => {
            assert_eq!(index, expected_index, "rejected channel index");
            assert_eq!(context, expected_context);
            assert!(
                msg.contains(&format!("[{expected_index}]")),
                "Display must name the channel index, got: {msg}"
            );
            assert!(
                msg.contains(expected_context),
                "Display must name the entry point, got: {msg}"
            );
        }
        other => panic!("expected NonFiniteSignal, got {other}"),
    }
}

#[test]
fn route_rejects_non_finite_signals_at_first_middle_and_last_channel() {
    let n = 3;
    for bad in non_finite_values() {
        for index in [0usize, n / 2, n - 1] {
            let mut signals = vec![0.5f32; n];
            signals[index] = bad;
            let mut router = ChannelRouter::new();
            let before = router.clone();
            let err = router.route(&signals).unwrap_err();
            assert_non_finite_signal(err, index, "route signals");
            assert_internal_state_eq(&router, &before);
        }
    }
}

#[test]
fn route_modulated_rejects_non_finite_signals_at_first_middle_and_last_channel() {
    let n = 5;
    let config = RouterConfig {
        channel_count: n,
        ..RouterConfig::default()
    };
    for bad in non_finite_values() {
        for index in [0usize, n / 2, n - 1] {
            let mut signals = vec![0.25f32; n];
            signals[index] = bad;
            let mut router = ChannelRouter::with_config(config.clone());
            let before = router.clone();
            let err = router
                .route_modulated(&signals, &NeuromodState::balanced())
                .unwrap_err();
            assert_non_finite_signal(err, index, "route_modulated signals");
            assert_internal_state_eq(&router, &before);
        }
    }
}

#[test]
fn each_neuromodulator_field_independently_rejected_when_non_finite() {
    let fields: [(&str, fn(f32) -> NeuromodState); 3] = [
        ("cortisol", |v| NeuromodState {
            cortisol: v,
            ..NeuromodState::balanced()
        }),
        ("dopamine", |v| NeuromodState {
            dopamine: v,
            ..NeuromodState::balanced()
        }),
        ("serotonin", |v| NeuromodState {
            serotonin: v,
            ..NeuromodState::balanced()
        }),
    ];
    for (field, make) in fields {
        for bad in non_finite_values() {
            let mut router = ChannelRouter::new();
            // Advance once so a mutation would be visible against a clone.
            router.route([1.0, 0.0, 0.0]).unwrap();
            let before = router.clone();
            let err = router
                .route_modulated([0.5, 0.0, 0.0], &make(bad))
                .unwrap_err();
            let msg = format!("{err}");
            match err {
                MeshError::NonFiniteNeuromodulator { field: got } => {
                    assert_eq!(got, field);
                    assert!(
                        msg.contains(field),
                        "Display must name the field, got: {msg}"
                    );
                }
                other => panic!("expected NonFiniteNeuromodulator for {field}, got {other}"),
            }
            assert_internal_state_eq(&router, &before);
        }
    }
}

#[test]
fn each_neuromodulator_field_independently_rejected_when_out_of_range() {
    let fields: [(&str, fn(f32) -> NeuromodState); 3] = [
        ("cortisol", |v| NeuromodState {
            cortisol: v,
            ..NeuromodState::balanced()
        }),
        ("dopamine", |v| NeuromodState {
            dopamine: v,
            ..NeuromodState::balanced()
        }),
        ("serotonin", |v| NeuromodState {
            serotonin: v,
            ..NeuromodState::balanced()
        }),
    ];
    for (field, make) in fields {
        for value in [-0.1f32, 1.01, 2.0, -1.0] {
            let err = make(value).validate().unwrap_err();
            let msg = format!("{err}");
            match err {
                MeshError::OutOfRangeNeuromodulator {
                    field: got,
                    value: got_value,
                } => {
                    assert_eq!(got, field);
                    assert_eq!(got_value, value);
                    assert!(
                        msg.contains(field) && msg.contains("[0, 1]"),
                        "Display must name field and range, got: {msg}"
                    );
                }
                other => panic!("expected OutOfRangeNeuromodulator for {field}, got {other}"),
            }

            let mut router = ChannelRouter::new();
            let before = router.clone();
            let route_err = router
                .route_modulated([0.5, 0.0, 0.0], &make(value))
                .unwrap_err();
            assert!(
                matches!(
                    route_err,
                    MeshError::OutOfRangeNeuromodulator { field: got, .. } if got == field
                ),
                "route_modulated must reject out-of-range {field}"
            );
            assert_internal_state_eq(&router, &before);
        }
    }
}

#[test]
fn unit_interval_endpoints_and_signed_zero_are_accepted() {
    for value in [-0.0f32, 0.0, 1.0] {
        NeuromodState {
            cortisol: value,
            dopamine: value,
            serotonin: value,
        }
        .validate()
        .expect("documented [0, 1] endpoints must be accepted");
    }
}

#[test]
fn rejection_leaves_all_internal_state_unchanged() {
    let mut router = ChannelRouter::new();
    router.route([1.0, 0.0, 0.0]).unwrap();
    router.route([0.0, 1.0, 0.0]).unwrap();
    let before = router.clone();

    assert!(router.route([0.8, f32::NAN, 0.1]).is_err());
    assert_internal_state_eq(&router, &before);

    let mods = NeuromodState {
        dopamine: f32::INFINITY,
        ..NeuromodState::balanced()
    };
    assert!(router.route_modulated([0.8, 0.1, 0.0], &mods).is_err());
    assert_internal_state_eq(&router, &before);
}

#[test]
fn valid_call_after_rejection_matches_untouched_control() {
    let mut rejected = ChannelRouter::new();
    let mut control = ChannelRouter::new();
    rejected.route([1.0, 0.0, 0.0]).unwrap();
    control.route([1.0, 0.0, 0.0]).unwrap();

    assert!(rejected.route([f32::NAN, 0.0, 0.0]).is_err());
    assert!(
        rejected
            .route_modulated(
                [0.0, 0.0, 0.0],
                &NeuromodState {
                    cortisol: f32::NAN,
                    ..NeuromodState::balanced()
                }
            )
            .is_err()
    );

    let d_rej = rejected.route([0.8, 0.1, 0.0]).unwrap();
    let d_ctl = control.route([0.8, 0.1, 0.0]).unwrap();
    assert_eq!(d_rej.active_channels, d_ctl.active_channels);
    assert_eq!(d_rej.firing_rates, d_ctl.firing_rates);
    assert_eq!(d_rej.input_signals, d_ctl.input_signals);
    assert_internal_state_eq(&rejected, &control);

    let mods = NeuromodState::rewarded();
    let d_rej = rejected.route_modulated([0.4, 0.0, 0.2], &mods).unwrap();
    let d_ctl = control.route_modulated([0.4, 0.0, 0.2], &mods).unwrap();
    assert_eq!(d_rej.active_channels, d_ctl.active_channels);
    assert_eq!(d_rej.firing_rates, d_ctl.firing_rates);
    assert_internal_state_eq(&rejected, &control);
}

#[test]
fn serde_restored_router_follows_the_same_ingress_rules() {
    let original = ChannelRouter::new();
    let json = serde_json::to_value(&original).expect("fresh router must serialize");
    let mut restored: ChannelRouter =
        serde_json::from_value(json).expect("fresh router must deserialize");
    let before = restored.clone();

    let err = restored.route([0.0, f32::INFINITY, 0.0]).unwrap_err();
    assert_non_finite_signal(err, 1, "route signals");
    assert_internal_state_eq(&restored, &before);

    let err = restored
        .route_modulated(
            [0.5, 0.0, 0.0],
            &NeuromodState {
                serotonin: f32::NEG_INFINITY,
                ..NeuromodState::balanced()
            },
        )
        .unwrap_err();
    match err {
        MeshError::NonFiniteNeuromodulator { field } => assert_eq!(field, "serotonin"),
        other => panic!("expected NonFiniteNeuromodulator, got {other}"),
    }
    assert_internal_state_eq(&restored, &before);

    restored.route([1.0, 0.0, 0.0]).unwrap();
    let mut control = ChannelRouter::new();
    control.route([1.0, 0.0, 0.0]).unwrap();
    assert_internal_state_eq(&restored, &control);
}

#[test]
fn rejection_does_not_self_heal_malformed_serde_state() {
    let router = ChannelRouter::new();
    let mut json = serde_json::to_value(&router).expect("fresh router must serialize");
    json["baseline_weights"] = serde_json::json!([[], [], []]);
    let mut router: ChannelRouter =
        serde_json::from_value(json).expect("malformed payload should still deserialize");
    let before = router_snapshot(&router);
    assert!(before["baseline_weights"] == serde_json::json!([[], [], []]));

    assert!(router.route([f32::NAN, 0.0, 0.0]).is_err());
    assert_eq!(
        router_snapshot(&router)["baseline_weights"],
        before["baseline_weights"],
        "failed ingress must not run ensure_neuromod_state_synced"
    );
}

#[test]
fn finite_signed_signals_retain_current_behavior() {
    let mut positive = ChannelRouter::new();
    let mut negative = ChannelRouter::new();
    let pos = positive.route([0.8, 0.2, -0.1]).unwrap();
    let neg = negative.route([-0.8, 0.2, -0.1]).unwrap();

    assert!(pos.firing_rates.iter().all(|r| r.is_finite()));
    assert!(neg.firing_rates.iter().all(|r| r.is_finite()));
    assert!(pos.is_active(0), "positive pulse on channel 0 should fire");
    assert!(
        !neg.is_active(0),
        "negative pulse on channel 0 should inhibit rather than activate"
    );

    let mut modulated = ChannelRouter::new();
    let d = modulated
        .route_modulated([-0.5, 0.0, 0.9], &NeuromodState::balanced())
        .unwrap();
    assert!(d.firing_rates.iter().all(|r| r.is_finite()));
    assert!(d.is_active(2));
    assert!(!d.is_active(0));
}

#[test]
fn later_non_finite_signal_does_not_apply_earlier_valid_channels() {
    let mut router = ChannelRouter::new();
    let before = router.clone();
    let err = router
        .route_modulated([1.0, 0.0, f32::NAN], &NeuromodState::stressed())
        .unwrap_err();
    assert_non_finite_signal(err, 2, "route_modulated signals");
    assert_eq!(router.total_routes, 0);
    assert_internal_state_eq(&router, &before);
}
