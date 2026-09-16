// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint restore and live-vs-restored comparison.

use synaptic_wiring::mesh::SynapticMesh;

use crate::harness::{Scenario, apply_event};

#[derive(Clone, Copy)]
pub(crate) enum SerdeFormat {
    Json,
    Postcard,
}

impl SerdeFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Postcard => "postcard",
        }
    }

    pub(crate) fn restore(self, mesh: &SynapticMesh) -> SynapticMesh {
        match self {
            Self::Json => {
                let json = serde_json::to_string(mesh).expect("valid mesh must serialize to JSON");
                serde_json::from_str(&json).expect("valid JSON checkpoint must deserialize")
            }
            Self::Postcard => {
                let bytes =
                    postcard::to_allocvec(mesh).expect("valid mesh must serialize to postcard");
                postcard::from_bytes(&bytes).expect("valid postcard checkpoint must deserialize")
            }
        }
    }
}

fn queued_deliveries(snapshot: &serde_json::Value) -> &serde_json::Value {
    &snapshot["delay_buffer"]["slots"]
}

pub(crate) fn meshes_equivalent(
    live: &SynapticMesh,
    restored: &SynapticMesh,
) -> Result<(), String> {
    let live_snap = crate::harness::checkpoint_snapshot(live);
    let restored_snap = crate::harness::checkpoint_snapshot(restored);
    if live_snap != restored_snap {
        let mut diffs = Vec::new();
        if let (Some(l_obj), Some(r_obj)) = (live_snap.as_object(), restored_snap.as_object()) {
            for (k, v) in l_obj {
                if r_obj.get(k) != Some(v) {
                    diffs.push(format!("  field '{k}': live={v} != restored={:?}", r_obj.get(k)));
                }
            }
            for k in r_obj.keys() {
                if !l_obj.contains_key(k) {
                    diffs.push(format!("  field '{k}' missing in live"));
                }
            }
        }
        let diff_summary = if diffs.is_empty() {
            String::new()
        } else {
            format!("\ndiffering fields:\n{}", diffs.join("\n"))
        };
        return Err(format!(
            "checkpoint snapshot mismatch (tick: live={} restored={}){diff_summary}\nlive queued={}\nrestored queued={}\nfull live snapshot={live_snap}\nfull restored snapshot={restored_snap}",
            live.tick(),
            restored.tick(),
            queued_deliveries(&live_snap),
            queued_deliveries(&restored_snap)
        ));
    }
    Ok(())
}

pub(crate) fn check_resume(scenario: &Scenario) -> Result<(), String> {
    check_resume_format(scenario, SerdeFormat::Json)?;
    check_resume_format(scenario, SerdeFormat::Postcard)?;
    Ok(())
}

fn check_resume_format(scenario: &Scenario, format: SerdeFormat) -> Result<(), String> {
    let mut live = scenario.mesh();
    for event in &scenario.prefix {
        let _ = apply_event(&mut live, event);
    }

    let mut restored = format.restore(&live);
    meshes_equivalent(&live, &restored).map_err(|err| {
        format!(
            "{} immediately after restore (seed {}, {:?}): {err}",
            format.name(),
            scenario.seed,
            scenario.recipe
        )
    })?;

    for (i, event) in scenario.suffix.iter().enumerate() {
        let live_currents = apply_event(&mut live, event);
        let restored_currents = apply_event(&mut restored, event);
        if live_currents != restored_currents {
            return Err(format!(
                "{} suffix tick {i} currents mismatch (seed {}, {:?}): live={live_currents:?} restored={restored_currents:?}",
                format.name(),
                scenario.seed,
                scenario.recipe
            ));
        }
        meshes_equivalent(&live, &restored).map_err(|err| {
            format!(
                "{} after suffix tick {i} (seed {}, {:?}): {err}",
                format.name(),
                scenario.seed,
                scenario.recipe
            )
        })?;
    }
    Ok(())
}

/// Greedy shrinker used only on failure so the panic carries a smaller
/// counterexample. It never runs on the passing CI path.
pub(crate) fn shrink(scenario: &Scenario) -> Scenario {
    let mut best = scenario.clone();
    while let Some(smaller) = shrink_step(&best) {
        best = smaller;
    }
    best
}

fn shrink_step(best: &Scenario) -> Option<Scenario> {
    shrink_remove_one_event(best, true)
        .or_else(|| shrink_remove_one_event(best, false))
        .or_else(|| shrink_remove_one_descriptor(best))
}

fn shrink_remove_one_event(best: &Scenario, prefix: bool) -> Option<Scenario> {
    let events = if prefix { &best.prefix } else { &best.suffix };
    if events.len() <= 1 {
        return None;
    }
    for i in 0..events.len() {
        let mut candidate = best.clone();
        if prefix {
            candidate.prefix.remove(i);
        } else {
            candidate.suffix.remove(i);
        }
        if check_resume(&candidate).is_err() {
            return Some(candidate);
        }
    }
    None
}

fn shrink_remove_one_descriptor(best: &Scenario) -> Option<Scenario> {
    if best.descriptors.len() <= 1 {
        return None;
    }
    for i in 0..best.descriptors.len() {
        let mut candidate = best.clone();
        candidate.descriptors.remove(i);
        let graph_max = candidate
            .descriptors
            .iter()
            .map(|d| usize::from(d.delay))
            .max()
            .unwrap_or(0);
        candidate.buffer_max_delay = candidate.buffer_max_delay.max(graph_max);
        if check_resume(&candidate).is_err() {
            return Some(candidate);
        }
    }
    None
}
