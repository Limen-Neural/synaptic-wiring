// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic topology generators.
//!
//! All generators use index-based pseudo-random hashing (no external RNG)
//! following the corinth-canal pattern — deterministic from neuron index
//! alone, reproducible across runs.
//!
//! # References
//!
//! - Erdős, P. & Rényi, A. (1959). *On random graphs.*
//! - Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of
//!   'small-world' networks.* Nature, 393, 440–442.
//! - Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in
//!   random networks.* Science, 286, 509–512.

use std::f64::consts::GOLDEN_RATIO;

use crate::error::{MeshError, Result};
use crate::topology::graph::SynapticGraph;
use crate::types::{Polarity, SynapseDescriptor};

// ── Deterministic hash helpers ────────────────────────────────────────────────

/// Simple deterministic hash from two indices — produces a value in [0, 1).
/// Uses a combination of golden-ratio fractional hashing and bit mixing.
fn hash_pair(a: usize, b: usize) -> f32 {
    const SILVER: f64 = 2.414_213_562_373_095;
    let mixed = (a as f64 * GOLDEN_RATIO + b as f64 * SILVER) % 1.0;
    mixed.abs() as f32
}

/// Deterministic weight from neuron pair.
fn hash_weight(src: usize, tgt: usize, base: f32, spread: f32) -> f32 {
    base + hash_pair(src * 97 + 13, tgt * 53 + 7) * spread
}

/// Deterministic delay from neuron pair.
fn hash_delay(src: usize, tgt: usize, max_delay: u16) -> u16 {
    if max_delay == 0 {
        return 0;
    }
    let h = hash_pair(src * 71 + 3, tgt * 37 + 11);
    (h * max_delay as f32)
        .round()
        .min(max_delay as f32)
        .max(1.0) as u16
}

// ── Generators ────────────────────────────────────────────────────────────────

/// Erdős–Rényi random graph.
///
/// Each directed edge `(i, j)` with `i ≠ j` exists independently with
/// probability `p`. Self-connections are not created.
///
/// Deterministic: the same `(n, p)` always produces the same graph.
pub fn generate_random(
    n: usize,
    p: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if !(0.0..=1.0).contains(&p) {
        return Err(MeshError::InvalidConfig(format!(
            "connection probability p={p} must be in [0, 1]"
        )));
    }
    if !(0.0..=1.0).contains(&inhibitory_fraction) {
        return Err(MeshError::InvalidConfig(format!(
            "inhibitory fraction={inhibitory_fraction} must be in [0, 1]"
        )));
    }
    if n == 0 {
        return Err(MeshError::InvalidConfig("neuron count must be ≥ 1".into()));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();

    for src in 0..n {
        let polarity = if src < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        for tgt in 0..n {
            if src == tgt {
                continue;
            }
            if hash_pair(src, tgt) < p {
                descriptors.push(SynapseDescriptor {
                    source: src as u32,
                    target: tgt as u32,
                    weight: hash_weight(src, tgt, 0.3, 0.6),
                    delay: hash_delay(src, tgt, max_delay),
                    polarity,
                });
            }
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Watts–Strogatz small-world graph.
///
/// Start with a **directed** ring lattice: each neuron has `k` outgoing
/// synapses to its nearest neighbours — `k/2` clockwise and `k/2`
/// counterclockwise, with no self-loop. `k` must be even and in `[2, n-1]`.
/// Odd `k` is rejected rather than silently truncated by integer division.
///
/// Then each outgoing synapse is rewired independently with probability
/// `beta` to a different non-self target when one is available. Rewiring
/// never introduces a self-loop or a duplicate outgoing target, so a valid
/// graph always has exactly `n * k` directed synapses (including `beta = 1`
/// and a dense ring `k = n - 1` when that value is even; a dense ring has
/// no unused target, so the original synapse is kept).
///
/// **Changed graphs:** the previous implementation only stored the
/// clockwise half of the lattice, so identical `(n, k, beta, …)` inputs
/// now produce a different (still deterministic) topology.
///
/// # References
///
/// Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of
/// 'small-world' networks.* Nature, 393, 440–442.
pub fn generate_small_world(
    n: usize,
    k: usize,
    beta: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if n < 3 {
        return Err(MeshError::InvalidConfig(
            "small-world requires n ≥ 3".into(),
        ));
    }
    if k < 2 || k >= n || !k.is_multiple_of(2) {
        return Err(MeshError::InvalidConfig(format!(
            "k={k} must be even and in [2, n-1]"
        )));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(MeshError::InvalidConfig(format!(
            "beta={beta} must be in [0, 1]"
        )));
    }
    if !(0.0..=1.0).contains(&inhibitory_fraction) {
        return Err(MeshError::InvalidConfig(format!(
            "inhibitory fraction={inhibitory_fraction} must be in [0, 1]"
        )));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let half_k = k / 2;
    let mut descriptors = Vec::new();

    for src in 0..n {
        let polarity = if src < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        let mut targets = Vec::with_capacity(k);
        for offset in 1..=half_k {
            targets.push((src + offset) % n);
            targets.push((src + n - offset) % n);
        }

        for i in 0..targets.len() {
            let salt = i + 1;
            if hash_pair(src * 131 + salt, targets[i] * 79) < beta {
                targets[i] = rewire_small_world_target(src, &targets, targets[i], n, salt);
            }
        }

        for tgt in targets {
            debug_assert_ne!(tgt, src, "small-world must not create a self-loop");
            descriptors.push(SynapseDescriptor {
                source: src as u32,
                target: tgt as u32,
                weight: hash_weight(src, tgt, 0.4, 0.5),
                delay: hash_delay(src, tgt, max_delay),
                polarity,
            });
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Replace `replacing` with a deterministic unused target, or keep it when
/// the ring is already a complete directed graph (`k = n - 1`).
fn rewire_small_world_target(
    src: usize,
    current_targets: &[usize],
    replacing: usize,
    n: usize,
    salt: usize,
) -> usize {
    let mut occupied = vec![false; n];
    occupied[src] = true;
    occupied[replacing] = true;
    for &tgt in current_targets {
        if tgt != replacing {
            occupied[tgt] = true;
        }
    }
    let candidates: Vec<usize> = (0..n).filter(|&idx| !occupied[idx]).collect();
    if candidates.is_empty() {
        return replacing;
    }
    let h = hash_pair(src * 173 + salt * 41, n * 29);
    let idx = ((h * candidates.len() as f32) as usize).min(candidates.len() - 1);
    candidates[idx]
}

/// Barabási–Albert scale-free graph.
///
/// Start with `m0` fully connected neurons (`m0 * (m0 - 1)` directed
/// synapses). Each new neuron attaches to **exactly** `m` distinct older
/// neurons, chosen with a bounded weighted sample proportional to current
/// degree (preferential attachment). Each attachment is stored as a
/// reciprocal directed pair, so the final directed synapse count is
/// `m0 * (m0 - 1) + 2 * m * (n - m0)`.
///
/// Later nodes may attach to an earlier node, so a node's **final**
/// out-degree is not `m`. The contract is: every new node `v >= m0` has
/// exactly `m` outgoing synapses whose targets are `< v`.
///
/// **Changed graphs:** the previous scan could stop with fewer than `m`
/// attachments; identical `(n, m0, m, …)` inputs now produce a different
/// (still deterministic) topology.
///
/// # References
///
/// Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random
/// networks.* Science, 286, 509–512.
pub fn generate_scale_free(
    n: usize,
    m0: usize,
    m: usize,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if m0 < 2 || m0 > n {
        return Err(MeshError::InvalidConfig(format!(
            "m0={m0} must be in [2, n]"
        )));
    }
    if m == 0 || m > m0 {
        return Err(MeshError::InvalidConfig(format!(
            "m={m} must be in [1, m0]"
        )));
    }
    if !(0.0..=1.0).contains(&inhibitory_fraction) {
        return Err(MeshError::InvalidConfig(format!(
            "inhibitory fraction={inhibitory_fraction} must be in [0, 1]"
        )));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();
    let mut degree = vec![0usize; n];

    // Seed: fully connect the first m0 nodes
    for (i, deg) in degree.iter_mut().enumerate().take(m0) {
        let polarity = if i < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };
        for j in 0..m0 {
            if i != j {
                descriptors.push(SynapseDescriptor {
                    source: i as u32,
                    target: j as u32,
                    weight: hash_weight(i, j, 0.3, 0.6),
                    delay: hash_delay(i, j, max_delay),
                    polarity,
                });
                *deg += 1;
            }
        }
    }

    // Growth phase: add nodes m0..n
    for new_node in m0..n {
        let polarity = if new_node < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        let older_targets = select_preferential_targets(new_node, m, &degree);
        for target_cursor in older_targets {
            descriptors.push(SynapseDescriptor {
                source: new_node as u32,
                target: target_cursor as u32,
                weight: hash_weight(new_node, target_cursor, 0.3, 0.6),
                delay: hash_delay(new_node, target_cursor, max_delay),
                polarity,
            });
            // Bidirectional attachment; polarity follows the source neuron.
            let reverse_polarity = if target_cursor < inhibitory_cutoff {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            };
            descriptors.push(SynapseDescriptor {
                source: target_cursor as u32,
                target: new_node as u32,
                weight: hash_weight(target_cursor, new_node, 0.3, 0.6),
                delay: hash_delay(target_cursor, new_node, max_delay),
                polarity: reverse_polarity,
            });
            degree[new_node] += 1;
            degree[target_cursor] += 1;
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Bounded weighted sample of `m` distinct older nodes (`0..new_node`).
///
/// Uses Efraimidis–Spirakis keys `u^(1/w)` so high-degree nodes are
/// preferred without an unbounded retry loop. Ties break on neuron id.
fn select_preferential_targets(new_node: usize, m: usize, degree: &[usize]) -> Vec<usize> {
    debug_assert!(new_node >= m);
    let mut scored: Vec<(f32, usize)> = (0..new_node)
        .map(|j| {
            let weight = degree[j].max(1) as f32;
            let u = hash_pair(new_node * 113 + 59, j * 83).clamp(1e-9, 1.0 - 1e-9);
            (u.powf(1.0 / weight), j)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().take(m).map(|(_, j)| j).collect()
}

/// Feed-forward layered network.
///
/// Neurons are arranged in layers; each neuron in layer `l` connects to each
/// neuron in layer `l+1` with probability `inter_layer_p`.
pub fn generate_layered(
    layer_sizes: &[usize],
    inter_layer_p: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if layer_sizes.is_empty() || layer_sizes.contains(&0) {
        return Err(MeshError::InvalidConfig(
            "all layer sizes must be greater than 0".into(),
        ));
    }
    if !(0.0..=1.0).contains(&inter_layer_p) {
        return Err(MeshError::InvalidConfig(format!(
            "inter_layer_p={inter_layer_p} must be in [0, 1]"
        )));
    }
    if !(0.0..=1.0).contains(&inhibitory_fraction) {
        return Err(MeshError::InvalidConfig(format!(
            "inhibitory fraction={inhibitory_fraction} must be in [0, 1]"
        )));
    }

    let n: usize = layer_sizes.iter().sum();

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();

    // Compute layer offsets
    let mut offsets = Vec::with_capacity(layer_sizes.len());
    let mut offset = 0usize;
    for &size in layer_sizes {
        offsets.push(offset);
        offset += size;
    }

    for layer_idx in 0..layer_sizes.len().saturating_sub(1) {
        let src_start = offsets[layer_idx];
        let src_end = src_start + layer_sizes[layer_idx];
        let tgt_start = offsets[layer_idx + 1];
        let tgt_end = tgt_start + layer_sizes[layer_idx + 1];

        for src in src_start..src_end {
            let polarity = if src < inhibitory_cutoff {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            };

            for tgt in tgt_start..tgt_end {
                if hash_pair(src * 67 + layer_idx, tgt * 43) < inter_layer_p {
                    descriptors.push(SynapseDescriptor {
                        source: src as u32,
                        target: tgt as u32,
                        weight: hash_weight(src, tgt, 0.3, 0.6),
                        delay: hash_delay(src, tgt, max_delay),
                        polarity,
                    });
                }
            }
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_graph_deterministic() {
        let g1 = generate_random(64, 0.1, 5, 0.2).unwrap();
        let g2 = generate_random(64, 0.1, 5, 0.2).unwrap();
        assert_eq!(g1.synapse_count(), g2.synapse_count());
    }

    #[test]
    fn random_graph_respects_probability() {
        let g = generate_random(100, 0.0, 5, 0.2).unwrap();
        assert_eq!(g.synapse_count(), 0);

        let g_full = generate_random(10, 1.0, 5, 0.2).unwrap();
        // 10 neurons, no self-connections → 10 * 9 = 90 edges
        assert_eq!(g_full.synapse_count(), 90);
    }

    #[test]
    fn small_world_produces_connected_graph() {
        let g = generate_small_world(32, 4, 0.1, 5, 0.2).unwrap();
        assert!(g.synapse_count() > 0);
        assert_eq!(g.neuron_count(), 32);
    }

    #[test]
    fn small_world_deterministic() {
        let g1 = generate_small_world(32, 4, 0.3, 5, 0.2).unwrap();
        let g2 = generate_small_world(32, 4, 0.3, 5, 0.2).unwrap();
        assert_eq!(g1, g2);
    }

    #[test]
    fn small_world_ring_has_k_outgoing_neighbors() {
        let graph = generate_small_world(8, 4, 0.0, 1, 0.0).unwrap();
        assert_eq!(graph.synapse_count(), 32);
        for source in 0..8 {
            assert_eq!(graph.out_degree(source), 4);
            let mut targets: Vec<usize> = graph
                .outgoing(source)
                .map(|(t, _, _, _)| t as usize)
                .collect();
            targets.sort_unstable();
            let unique = targets.clone();
            targets.dedup();
            assert_eq!(targets, unique, "source {source} has duplicate targets");
            assert!(!targets.contains(&source), "self-loop at {source}");
            let expected = {
                let mut e = vec![
                    (source + 1) % 8,
                    (source + 2) % 8,
                    (source + 8 - 1) % 8,
                    (source + 8 - 2) % 8,
                ];
                e.sort_unstable();
                e
            };
            assert_eq!(targets, expected);
        }
    }

    #[test]
    fn small_world_rejects_odd_k() {
        let err = generate_small_world(8, 3, 0.0, 1, 0.0).unwrap_err();
        assert!(err.to_string().contains("even"), "{err}");
        assert!(generate_small_world(8, 1, 0.0, 1, 0.0).is_err());
    }

    #[test]
    fn small_world_min_valid_size() {
        let g = generate_small_world(3, 2, 0.0, 1, 0.0).unwrap();
        assert_eq!(g.synapse_count(), 6);
        for src in 0..3 {
            assert_eq!(g.out_degree(src), 2);
        }
    }

    #[test]
    fn small_world_rewiring_preserves_edge_count_and_no_loops() {
        for beta in [0.0, 0.5, 1.0] {
            let g = generate_small_world(8, 4, beta, 1, 0.0).unwrap();
            assert_eq!(g.synapse_count(), 32, "beta={beta}");
            for src in 0..8 {
                let mut targets: Vec<usize> =
                    g.outgoing(src).map(|(t, _, _, _)| t as usize).collect();
                assert_eq!(targets.len(), 4, "beta={beta} src={src}");
                assert!(!targets.contains(&src));
                targets.sort_unstable();
                let before = targets.len();
                targets.dedup();
                assert_eq!(targets.len(), before, "duplicate at beta={beta} src={src}");
            }
        }
    }

    #[test]
    fn small_world_dense_ring() {
        let g = generate_small_world(5, 4, 1.0, 1, 0.0).unwrap();
        assert_eq!(g.synapse_count(), 20);
        for src in 0..5 {
            assert_eq!(g.out_degree(src), 4);
        }
    }

    #[test]
    fn small_world_beta_1_rewires_every_non_dense_ring_slot() {
        let n = 8;
        let k = 4;
        let half_k = k / 2;
        let g = generate_small_world(n, k, 1.0, 1, 0.0).unwrap();
        assert_eq!(g.synapse_count(), n * k);
        for src in 0..n {
            let mut original = Vec::with_capacity(k);
            for offset in 1..=half_k {
                original.push((src + offset) % n);
                original.push((src + n - offset) % n);
            }
            let final_targets: Vec<usize> =
                g.outgoing(src).map(|(t, _, _, _)| t as usize).collect();
            assert_eq!(final_targets.len(), k, "src={src}");
            for (slot, &tgt) in final_targets.iter().enumerate() {
                assert_ne!(
                    tgt, original[slot],
                    "src={src} slot={slot} kept original ring target {tgt}"
                );
            }
        }
    }

    #[test]
    fn scale_free_hub_structure() {
        let g = generate_scale_free(50, 5, 3, 5, 0.2).unwrap();
        assert!(g.synapse_count() > 0);

        // The seed nodes should have higher degree than late arrivals
        let seed_degree: usize = (0..5).map(|i| g.out_degree(i)).sum();
        let late_degree: usize = (45..50).map(|i| g.out_degree(i)).sum();
        assert!(
            seed_degree >= late_degree,
            "seed degree {seed_degree} should be ≥ late degree {late_degree}"
        );
    }

    fn assert_scale_free_contract(n: usize, m0: usize, m: usize) {
        let g = generate_scale_free(n, m0, m, 5, 0.2).unwrap();
        assert_eq!(
            g.synapse_count(),
            m0 * (m0 - 1) + 2 * m * (n - m0),
            "directed edge count"
        );
        for v in m0..n {
            let mut older: Vec<usize> = g
                .outgoing(v)
                .map(|(t, _, _, _)| t as usize)
                .filter(|&t| t < v)
                .collect();
            older.sort_unstable();
            let before = older.len();
            older.dedup();
            assert_eq!(older.len(), before, "duplicate older targets at {v}");
            assert_eq!(older.len(), m, "older attachments at {v}");
            assert!(!older.contains(&v));
        }
        let g2 = generate_scale_free(n, m0, m, 5, 0.2).unwrap();
        assert_eq!(g, g2);
    }

    #[test]
    fn scale_free_exact_older_attachments() {
        assert_scale_free_contract(50, 5, 3);
    }

    #[test]
    fn scale_free_m_equals_one() {
        assert_scale_free_contract(10, 3, 1);
    }

    #[test]
    fn scale_free_m_equals_m0() {
        assert_scale_free_contract(12, 4, 4);
    }

    #[test]
    fn scale_free_min_valid_size() {
        assert_scale_free_contract(2, 2, 1);
    }

    #[test]
    fn scale_free_rejects_endpoint_bounds() {
        assert!(generate_scale_free(5, 1, 1, 1, 0.0).is_err());
        assert!(generate_scale_free(5, 6, 2, 1, 0.0).is_err());
        assert!(generate_scale_free(5, 3, 0, 1, 0.0).is_err());
        assert!(generate_scale_free(5, 3, 4, 1, 0.0).is_err());
    }

    #[test]
    fn layered_feed_forward() {
        let g = generate_layered(&[8, 16, 4], 1.0, 5, 0.2).unwrap();
        assert_eq!(g.neuron_count(), 28);

        // Layer 0→1: 8×16 = 128 edges, Layer 1→2: 16×4 = 64 edges
        // With p=1.0, all inter-layer connections present
        assert_eq!(g.synapse_count(), 128 + 64);
    }

    #[test]
    fn layered_no_back_connections() {
        let g = generate_layered(&[4, 4, 4], 1.0, 5, 0.0).unwrap();
        // No neuron in layer 1 or 2 should connect back to layer 0
        for src in 4..12 {
            for (tgt, _, _, _) in g.outgoing(src) {
                assert!(tgt as usize >= 4, "back-connection from {src} to {tgt}");
            }
        }
    }

    #[test]
    fn dale_law_polarity_applied() {
        let g = generate_random(20, 0.5, 3, 0.3).unwrap();
        // First 6 neurons (30% of 20) should be inhibitory
        for src in 0..6 {
            for (_, weight, _, polarity) in g.outgoing(src) {
                assert_eq!(polarity, Polarity::Inhibitory);
                assert!(
                    weight <= 0.0,
                    "inhibitory neuron {src} has positive weight {weight}"
                );
            }
        }
    }
}
