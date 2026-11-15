// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compressed Sparse Row (CSR) synaptic map for GPU-optimized execution.
//!
//! Replaces dense $N \times N$ weight matrices with adjacency lists stored in CSR format,
//! reducing VRAM pressure and enabling warp-optimized shared memory pulls.
//!
//! # Layout
//!
//! ```text
//! row_ptr:    [0, 3, 7, 10, ...]        — start index in col_indices/values per neuron
//! col_indices:[0, 5, 12, 1, 3, 8, 15, ...] — target neuron indices
//! values:     [0.9, -0.15, 0.3, ...]     — synaptic weights
//! ```
//!
//! For a 2048-neuron network with ~5% connectivity, this reduces storage from
//! ~16 MB (dense f32) to ~800 KB (sparse), a 20× reduction.

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};

/// Largest neuron count a `u16` CSR target can address (`0..=u16::MAX`).
pub const MAX_SPARSE_NEURONS: usize = u16::MAX as usize + 1;

/// A single sparse synaptic connection.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Synapse {
    /// Target neuron index (column in the weight matrix).
    pub target: u16,
    /// Synaptic weight strength.
    pub weight: f32,
}

/// Compressed Sparse Row representation of the synaptic weight matrix.
///
/// Generic over `N` (number of neurons) to support both small channel routers
/// and full 2048-neuron routing fabrics.
///
/// `N` must be at most [`MAX_SPARSE_NEURONS`] (65,536): target index
/// `65,535` is representable as `u16`, target `65,536` is not. Constructors
/// and insertion APIs reject out-of-range or non-`u16` indices rather than
/// truncating them into a different valid connection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseSynapticMap<const N: usize> {
    /// Row pointers: `row_ptr[i]` gives the start index in `col_indices`/`values`
    /// for neuron `i`. Length is `N + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices: target neuron for each non-zero entry.
    pub col_indices: Vec<u16>,
    /// Non-zero weight values.
    pub values: Vec<f32>,
}

impl<const N: usize> Default for SparseSynapticMap<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> SparseSynapticMap<N> {
    /// Create an empty sparse synaptic map with `N` neurons.
    ///
    /// # Panics
    ///
    /// Panics if `N` exceeds [`MAX_SPARSE_NEURONS`]. Prefer
    /// [`SparseSynapticMap::try_new`] for a recoverable error.
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible constructor that rejects an `N` larger than the `u16`
    /// target address space.
    pub fn try_new() -> Result<Self> {
        validate_neuron_count::<N>()?;
        Ok(Self {
            row_ptr: vec![0; N + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Build from a dense weight matrix (for migration from dense representations).
    ///
    /// # Panics
    ///
    /// Panics if `N` exceeds [`MAX_SPARSE_NEURONS`].
    pub fn from_dense(matrix: &[[f32; N]; N], sparsity_threshold: f32) -> Self {
        Self::try_from_dense(matrix, sparsity_threshold).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SparseSynapticMap::from_dense`].
    pub fn try_from_dense(matrix: &[[f32; N]; N], sparsity_threshold: f32) -> Result<Self> {
        validate_neuron_count::<N>()?;
        let mut row_ptr = Vec::with_capacity(N + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for row in matrix.iter() {
            for (col, &w) in row.iter().enumerate() {
                if w.abs() > sparsity_threshold {
                    col_indices.push(index_as_u16::<N>(col, "column")?);
                    values.push(w);
                }
            }
            row_ptr.push(col_indices.len());
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
        })
    }

    /// Build from explicit adjacency lists.
    ///
    /// # Panics
    ///
    /// Panics if `N` is unsupported, `adjacency.len()` is not `N`, or any
    /// stored target is `>= N`. Prefer [`SparseSynapticMap::try_from_adjacency`].
    pub fn from_adjacency(adjacency: &[Vec<Synapse>]) -> Self {
        Self::try_from_adjacency(adjacency).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SparseSynapticMap::from_adjacency`].
    pub fn try_from_adjacency(adjacency: &[Vec<Synapse>]) -> Result<Self> {
        validate_neuron_count::<N>()?;
        if adjacency.len() != N {
            return Err(MeshError::NeuronCountMismatch {
                expected: N,
                got: adjacency.len(),
                context: "from_adjacency".into(),
            });
        }

        let mut row_ptr = Vec::with_capacity(N + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for synapses in adjacency {
            for syn in synapses {
                if syn.target as usize >= N {
                    return Err(MeshError::IndexOutOfBounds {
                        index: syn.target as usize,
                        max: N.saturating_sub(1),
                    });
                }
                col_indices.push(syn.target);
                values.push(syn.weight);
            }
            row_ptr.push(col_indices.len());
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
        })
    }

    /// Get all synapses for a given neuron (row).
    pub fn get_row(&self, row: usize) -> impl Iterator<Item = (u16, f32)> + '_ {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        (start..end).map(move |i| (self.col_indices[i], self.values[i]))
    }

    /// Get the weight from neuron `row` to neuron `col`. Returns 0.0 if not connected.
    pub fn get_weight(&self, row: usize, col: usize) -> f32 {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for i in start..end {
            if self.col_indices[i] as usize == col {
                return self.values[i];
            }
        }
        0.0
    }

    /// Update a single synaptic weight. Creates the connection if it doesn't exist
    /// (when weight exceeds threshold) or removes it (when weight drops below).
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` is `>= N`, or if `col` cannot be represented
    /// as `u16`. Prefer [`SparseSynapticMap::try_set_weight`].
    pub fn set_weight(&mut self, row: usize, col: usize, weight: f32, sparsity_threshold: f32) {
        self.try_set_weight(row, col, weight, sparsity_threshold)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SparseSynapticMap::set_weight`].
    ///
    /// On error the map is left unchanged.
    pub fn try_set_weight(
        &mut self,
        row: usize,
        col: usize,
        weight: f32,
        sparsity_threshold: f32,
    ) -> Result<()> {
        let _ = index_as_u16::<N>(row, "row")?;
        let col_u16 = index_as_u16::<N>(col, "column")?;
        if let Some(i) = self.find_connection(row, col) {
            self.update_or_remove(i, row, weight, sparsity_threshold);
            return Ok(());
        }
        if weight.abs() > sparsity_threshold {
            self.insert_connection(row, col_u16, weight);
        }
        Ok(())
    }

    fn find_connection(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        (start..end).find(|&i| self.col_indices[i] as usize == col)
    }

    fn update_or_remove(&mut self, index: usize, row: usize, weight: f32, sparsity_threshold: f32) {
        if weight.abs() > sparsity_threshold {
            self.values[index] = weight;
            return;
        }
        self.col_indices.remove(index);
        self.values.remove(index);
        self.shift_row_ptr(row, -1);
    }

    fn insert_connection(&mut self, row: usize, col: u16, weight: f32) {
        let end = self.row_ptr[row + 1];
        self.col_indices.insert(end, col);
        self.values.insert(end, weight);
        self.shift_row_ptr(row, 1);
    }

    fn shift_row_ptr(&mut self, row: usize, delta: isize) {
        for ptr in &mut self.row_ptr[(row + 1)..=N] {
            *ptr = ptr
                .checked_add_signed(delta)
                .expect("row_ptr stays in range");
        }
    }

    /// Number of non-zero synapses.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio: fraction of zero entries in the full $N \times N$ matrix.
    ///
    /// The denominator is computed in `u64` so `N = 65_536` does not overflow
    /// `usize` on 32-bit targets (`65_536² = 2³²`).
    pub fn sparsity(&self) -> f32 {
        let total = (N as u64).saturating_mul(N as u64);
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / total as f32)
    }

    /// Convert back to dense matrix (for debugging / interoperability).
    pub fn to_dense(&self) -> [[f32; N]; N] {
        let mut matrix = [[0.0; N]; N];
        for (row, row_data) in matrix.iter_mut().enumerate().take(N) {
            for (col, w) in self.get_row(row) {
                row_data[col as usize] = w;
            }
        }
        matrix
    }

    /// Export as GPU-ready flat arrays for kernel launch.
    /// Returns (row_ptr, col_indices, values) as owned vectors.
    ///
    /// # Panics
    ///
    /// Panics if a row-pointer offset exceeds `u32::MAX`. Prefer
    /// [`SparseSynapticMap::try_to_gpu_arrays`].
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        self.try_to_gpu_arrays()
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SparseSynapticMap::to_gpu_arrays`].
    pub fn try_to_gpu_arrays(&self) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        let row_ptr = usizes_to_u32(&self.row_ptr)?;
        let col_indices: Vec<u32> = self.col_indices.iter().map(|&x| u32::from(x)).collect();
        Ok((row_ptr, col_indices, self.values.clone()))
    }
}

fn validate_neuron_count<const N: usize>() -> Result<()> {
    if N > MAX_SPARSE_NEURONS {
        return Err(MeshError::InvalidConfig(format!(
            "SparseSynapticMap neuron count N={N} exceeds the u16 target address space ({MAX_SPARSE_NEURONS})"
        )));
    }
    Ok(())
}

fn index_as_u16<const N: usize>(index: usize, what: &str) -> Result<u16> {
    if index >= N {
        return Err(MeshError::IndexOutOfBounds {
            index,
            max: N.saturating_sub(1),
        });
    }
    u16::try_from(index).map_err(|_| {
        MeshError::InvalidConfig(format!("{what} {index} is not representable as u16"))
    })
}

/// Checked `usize → u32` conversion for CSR row pointers.
///
/// Exposed for tests so overflow can be checked without allocating a map
/// with more than `u32::MAX` synapses.
pub(crate) fn usizes_to_u32(values: &[usize]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&offset| {
            u32::try_from(offset).map_err(|_| {
                MeshError::InvalidConfig(format!("row pointer offset {offset} exceeds u32::MAX"))
            })
        })
        .collect()
}

/// Builder for constructing a sparse synaptic map with a fluent API.
pub struct SparseSynapticMapBuilder<const N: usize> {
    adjacency: Vec<Vec<Synapse>>,
    default_weight: f32,
    sparsity_threshold: f32,
}

impl<const N: usize> SparseSynapticMapBuilder<N> {
    /// # Panics
    ///
    /// Panics if `N` exceeds [`MAX_SPARSE_NEURONS`].
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`SparseSynapticMapBuilder::new`].
    pub fn try_new() -> Result<Self> {
        validate_neuron_count::<N>()?;
        Ok(Self {
            adjacency: vec![Vec::new(); N],
            default_weight: 0.0,
            sparsity_threshold: 0.01,
        })
    }

    /// Set the default weight for self-connections.
    pub fn with_self_weight(mut self, weight: f32) -> Self {
        self.default_weight = weight;
        self
    }

    /// Set the sparsity threshold below which connections are pruned.
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Add a synaptic connection.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is `>= N`. Prefer
    /// [`SparseSynapticMapBuilder::try_connect`].
    pub fn connect(mut self, from: usize, to: usize, weight: f32) -> Self {
        self.try_connect(from, to, weight)
            .unwrap_or_else(|err| panic!("{err}"));
        self
    }

    /// Fallible counterpart of [`SparseSynapticMapBuilder::connect`].
    /// On error the builder is left unchanged.
    pub fn try_connect(&mut self, from: usize, to: usize, weight: f32) -> Result<()> {
        let _ = index_as_u16::<N>(from, "source")?;
        let target = index_as_u16::<N>(to, "target")?;
        if weight.abs() > self.sparsity_threshold {
            self.adjacency[from].push(Synapse { target, weight });
        }
        Ok(())
    }

    /// Add self-connections for all neurons with the default weight.
    pub fn with_self_connections(self) -> Self {
        let w = self.default_weight;
        let mut result = self;
        for i in 0..N {
            result = result.connect(i, i, w);
        }
        result
    }

    /// Add lateral inhibition between all pairs with the given weight.
    pub fn with_lateral_inhibition(self, weight: f32) -> Self {
        let mut result = self;
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    result = result.connect(i, j, weight);
                }
            }
        }
        result
    }

    /// Build the CSR map.
    pub fn build(self) -> SparseSynapticMap<N> {
        SparseSynapticMap::from_adjacency(&self.adjacency)
    }
}

impl<const N: usize> Default for SparseSynapticMapBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-neuron state snapshot for adaptation-aware routing.
///
/// Captures the state needed for dynamic routing: per-neuron
/// adaptation levels, spike counts, and error estimates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronStateSnapshot {
    /// Per-neuron adaptation state (0.0 = fresh, 1.0 = fully adapted/exhausted).
    pub adaptation: Vec<f32>,
    /// Per-neuron spike count from the last routing window.
    pub spike_counts: Vec<u32>,
    /// Estimated error per neuron (e.g. quantization error from external calibration).
    #[serde(alias = "quant_error")]
    pub error: Vec<f32>,
    /// Global routing step index.
    pub step: u64,
}

impl NeuronStateSnapshot {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            adaptation: vec![0.0; num_neurons],
            spike_counts: vec![0; num_neurons],
            error: vec![0.0; num_neurons],
            step: 0,
        }
    }

    /// Get a routing penalty for a neuron based on its adaptation state.
    /// Higher adaptation → higher penalty → less likely to be selected.
    pub fn adaptation_penalty(&self, neuron: usize, alpha: f32) -> f32 {
        alpha * self.adaptation.get(neuron).copied().unwrap_or(0.0)
    }

    /// Get a routing bonus for a neuron based on low error.
    /// Lower error → higher bonus → preferred for routing.
    pub fn error_bonus(&self, neuron: usize, beta: f32) -> f32 {
        let err = self.error.get(neuron).copied().unwrap_or(0.0);
        beta * (1.0 - err.min(1.0))
    }
}

/// A routing policy equation for scoring neurons.
///
/// The policy computes a routing score for each neuron:
/// `score = α·spikes - β·adaptation - γ·error + δ·base_weight`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingPolicy {
    /// Weight for spike count contribution.
    pub alpha: f32,
    /// Weight for adaptation penalty.
    pub beta: f32,
    /// Weight for error penalty.
    pub gamma: f32,
    /// Weight for base synaptic strength.
    pub delta: f32,
    /// Minimum score threshold to activate a neuron.
    pub threshold: f32,
    /// Human-readable description of the policy origin.
    pub description: String,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.3,
            delta: 0.8,
            threshold: 0.1,
            description: "default policy".to_string(),
        }
    }
}

impl RoutingPolicy {
    /// Compute the routing score for a single neuron.
    pub fn score(&self, neuron: usize, snapshot: &NeuronStateSnapshot, base_weight: f32) -> f32 {
        let spikes = snapshot.spike_counts.get(neuron).copied().unwrap_or(0) as f32;
        let adapt = snapshot.adaptation.get(neuron).copied().unwrap_or(0.0);
        let err = snapshot.error.get(neuron).copied().unwrap_or(0.0);

        self.alpha * spikes - self.beta * adapt - self.gamma * err + self.delta * base_weight
    }

    /// Check if a neuron should be activated based on its score.
    pub fn should_activate(&self, score: f32) -> bool {
        score >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_weight_rejects_truncated_u16_column() {
        let mut map = SparseSynapticMap::<2>::new();
        assert!(map.try_set_weight(0, 65_536, 1.0, 0.0).is_err());
        assert_eq!(map.nnz(), 0);
        assert_eq!(map.get_weight(0, 0), 0.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_weight_panics_on_column_equal_to_n() {
        let mut map = SparseSynapticMap::<2>::new();
        map.set_weight(0, 2, 1.0, 0.0);
    }

    #[test]
    fn set_weight_rejects_column_equal_to_n_without_mutation() {
        let mut map = SparseSynapticMap::<2>::new();
        map.try_set_weight(0, 1, 0.5, 0.0).unwrap();
        assert!(map.try_set_weight(0, 2, 1.0, 0.0).is_err());
        assert_eq!(map.nnz(), 1);
        assert!((map.get_weight(0, 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn set_weight_rejects_invalid_source_row() {
        let mut map = SparseSynapticMap::<2>::new();
        assert!(map.try_set_weight(2, 0, 1.0, 0.0).is_err());
        assert_eq!(map.nnz(), 0);
    }

    #[test]
    fn set_weight_insert_update_remove_roundtrip() {
        let mut map = SparseSynapticMap::<2>::new();
        map.try_set_weight(0, 1, 1.0, 0.1).unwrap();
        assert!((map.get_weight(0, 1) - 1.0).abs() < 1e-6);
        map.try_set_weight(0, 1, 0.5, 0.1).unwrap();
        assert!((map.get_weight(0, 1) - 0.5).abs() < 1e-6);
        map.try_set_weight(0, 1, 0.01, 0.1).unwrap();
        assert_eq!(map.get_weight(0, 1), 0.0);
        assert_eq!(map.nnz(), 0);
    }

    #[test]
    fn from_adjacency_rejects_target_equal_to_n() {
        let adjacency = vec![
            vec![Synapse {
                target: 2,
                weight: 1.0,
            }],
            vec![],
        ];
        assert!(SparseSynapticMap::<2>::try_from_adjacency(&adjacency).is_err());
    }

    #[test]
    fn builder_rejects_invalid_target() {
        let mut builder = SparseSynapticMapBuilder::<2>::new();
        assert!(builder.try_connect(0, 2, 1.0).is_err());
        assert!(builder.try_connect(0, 65_536, 1.0).is_err());
        let map = builder.build();
        assert_eq!(map.nnz(), 0);
    }

    #[test]
    fn empty_and_single_neuron_maps() {
        let mut empty = SparseSynapticMap::<0>::try_new().unwrap();
        assert_eq!(empty.nnz(), 0);
        assert!(empty.try_set_weight(0, 0, 1.0, 0.0).is_err());

        let mut single = SparseSynapticMap::<1>::new();
        single.try_set_weight(0, 0, 0.8, 0.0).unwrap();
        assert!((single.get_weight(0, 0) - 0.8).abs() < 1e-6);
        assert!(single.try_set_weight(0, 1, 0.8, 0.0).is_err());
    }

    #[test]
    fn highest_representable_target_is_accepted() {
        // N = 65_536 allocates only the CSR row_ptr (~512 KiB), not an N×N matrix.
        let mut map = SparseSynapticMap::<65_536>::try_new().unwrap();
        map.try_set_weight(0, 65_535, 1.0, 0.0).unwrap();
        assert!((map.get_weight(0, 65_535) - 1.0).abs() < 1e-6);
        assert!((map.sparsity() - 1.0).abs() < 1e-6);
        let (_row_ptr, cols, values) = map.try_to_gpu_arrays().unwrap();
        assert_eq!(cols, vec![65_535]);
        assert!((values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn try_new_rejects_n_above_u16_address_space() {
        // Allocates only a small row_ptr probe via try_new, not an N×N matrix.
        assert!(SparseSynapticMap::<65_537>::try_new().is_err());
        assert!(SparseSynapticMapBuilder::<65_537>::try_new().is_err());
    }

    #[test]
    fn usizes_to_u32_accepts_u32_max() {
        assert_eq!(usizes_to_u32(&[0, u32::MAX as usize]).unwrap().len(), 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn usizes_to_u32_rejects_unrepresentable_offset() {
        let too_big = (u32::MAX as usize) + 1;
        assert!(usizes_to_u32(&[0, too_big]).is_err());
    }
}
