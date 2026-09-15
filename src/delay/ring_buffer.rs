// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ring-buffer delay queue for tick-aligned spike delivery.
//!
//! The [`SpikeDelayBuffer`] implements a fixed-size circular buffer where
//! spikes are injected at the current tick plus a per-synapse delay, and
//! delivered (drained) at each tick advance.
//!
//! # Design
//!
//! ```text
//! tick 0:  inject spike at delay=3  →  buffer[3] += weight
//! tick 1:  ...
//! tick 2:  ...
//! tick 3:  drain buffer[3]  →  deliver accumulated current to target neuron
//! ```
//!
//! The ring buffer has `max_delay + 1` slots, each slot is a vector of
//! length `neuron_count` accumulating incoming synaptic current.

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};

/// Ring-buffer delay queue for spike delivery.
///
/// At each simulation tick:
/// 1. Call [`SpikeDelayBuffer::inject`] for each spiking synapse to schedule future delivery.
/// 2. Call [`SpikeDelayBuffer::drain_current_tick`] or
///    [`SpikeDelayBuffer::drain_current_tick_into`] to collect all currents
///    that have arrived.
/// 3. Call [`SpikeDelayBuffer::advance`] to move the tick forward.
///
/// With `max_delay == 0` the buffer holds a single slot and every spike is
/// delivered in the same tick it is injected, behaving as if there were no
/// delay layer — it still allocates that one slot, one `f32` per neuron.
///
/// Deserialization re-validates the ring invariants: depth is
/// `max_delay + 1` (checked, nonzero), every slot width equals
/// `neuron_count`, every stored current is finite, `current_tick + max_delay`
/// fits in `usize` so inject/drain slot arithmetic cannot overflow, and
/// `current_tick` leaves one-tick headroom so `advance` cannot land on
/// `usize::MAX`. Empty-neuron buffers (`neuron_count == 0`) remain legal
/// when each slot is an empty vector.
#[derive(Clone, Debug, Serialize)]
pub struct SpikeDelayBuffer {
    /// Ring buffer: `slots[slot_index][neuron_id]` → accumulated current.
    slots: Vec<Vec<f32>>,
    /// Number of neurons (width of each slot).
    neuron_count: usize,
    /// Maximum delay in ticks (depth of the ring buffer minus 1).
    max_delay: usize,
    /// Current simulation tick.
    current_tick: u64,
}

#[derive(Deserialize)]
struct RawSpikeDelayBuffer {
    slots: Vec<Vec<f32>>,
    neuron_count: usize,
    max_delay: usize,
    current_tick: u64,
}

impl RawSpikeDelayBuffer {
    fn into_buffer(self) -> std::result::Result<SpikeDelayBuffer, String> {
        validate_delay_buffer_shape(&self.slots, self.neuron_count, self.max_delay)?;
        validate_current_tick(self.current_tick, self.max_delay)?;
        Ok(SpikeDelayBuffer {
            slots: self.slots,
            neuron_count: self.neuron_count,
            max_delay: self.max_delay,
            current_tick: self.current_tick,
        })
    }
}

/// Ring depth must be `max_delay + 1` (nonzero, no overflow) and every
/// slot must have width `neuron_count` with only finite currents.
fn validate_delay_buffer_shape(
    slots: &[Vec<f32>],
    neuron_count: usize,
    max_delay: usize,
) -> std::result::Result<(), String> {
    let expected_depth = max_delay.checked_add(1).ok_or_else(|| {
        format!("max_delay {max_delay} + 1 overflows usize; delay buffer depth is invalid")
    })?;
    if slots.is_empty() {
        return Err("delay buffer slots must be non-empty".into());
    }
    if slots.len() != expected_depth {
        return Err(format!(
            "delay buffer depth {} does not match max_delay + 1 ({expected_depth})",
            slots.len()
        ));
    }
    for (i, slot) in slots.iter().enumerate() {
        if slot.len() != neuron_count {
            return Err(format!(
                "delay buffer slot {i} width {} does not match neuron_count {neuron_count}",
                slot.len()
            ));
        }
        if slot.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "delay buffer slot {i} contains a non-finite current; refusing to poison later drain_current_tick sums"
            ));
        }
    }
    Ok(())
}

/// `advance` (`+= 1`) must not land on `usize::MAX`, and
/// `current_tick + max_delay` must fit in `usize` so inject slot arithmetic
/// cannot wrap.
pub(crate) fn validate_current_tick(
    current_tick: u64,
    max_delay: usize,
) -> std::result::Result<(), String> {
    if current_tick >= usize::MAX as u64 {
        return Err("current_tick is too large for safe advancement and indexing".into());
    }
    let max_safe_tick = (usize::MAX as u64).saturating_sub(max_delay as u64);
    if current_tick > max_safe_tick {
        return Err(format!(
            "current_tick {current_tick} is too large to add max_delay {max_delay} without overflowing usize indexing"
        ));
    }
    // One subsequent `+= 1` must stay strictly below `usize::MAX`, even when
    // `max_delay` is 0 or 1 and inject arithmetic would otherwise allow
    // `current_tick == usize::MAX - 1`.
    if current_tick >= (usize::MAX as u64).saturating_sub(1) {
        return Err("current_tick is too large for safe advancement and indexing".into());
    }
    Ok(())
}

impl<'de> Deserialize<'de> for SpikeDelayBuffer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawSpikeDelayBuffer::deserialize(deserializer)?
            .into_buffer()
            .map_err(DeError::custom)
    }
}

impl SpikeDelayBuffer {
    /// Create a new delay buffer.
    ///
    /// The ring has `max_delay + 1` slots so a spike injected with
    /// `delay == max_delay` lands on a future tick rather than wrapping
    /// onto the current slot.
    ///
    /// # Arguments
    ///
    /// * `neuron_count` — number of target neurons (slot width)
    /// * `max_delay` — maximum axonal delay in ticks
    ///
    /// # Panics
    ///
    /// Panics if `max_delay + 1` overflows `usize`. Prefer
    /// [`SpikeDelayBuffer::try_new`] when the caller needs a recoverable
    /// error.
    pub fn new(neuron_count: usize, max_delay: usize) -> Self {
        Self::try_new(neuron_count, max_delay).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible constructor that rejects a `max_delay` whose ring depth
    /// (`max_delay + 1`) would overflow `usize`.
    pub fn try_new(neuron_count: usize, max_delay: usize) -> Result<Self> {
        let depth = max_delay.checked_add(1).ok_or_else(|| {
            MeshError::DelayError(format!(
                "max_delay {max_delay} + 1 overflows usize; ring depth cannot be represented"
            ))
        })?;
        Ok(Self {
            slots: vec![vec![0.0; neuron_count]; depth],
            neuron_count,
            max_delay,
            current_tick: 0,
        })
    }

    /// Inject a spike from a source neuron through a synapse.
    ///
    /// The synaptic current `weight` will be delivered to `target` neuron
    /// after `delay` ticks from the current tick.
    ///
    /// Bounds are checked in **both** debug and release builds before any
    /// slot is modified. An oversized delay is rejected rather than
    /// wrapping onto an earlier tick via modulo arithmetic.
    ///
    /// # Panics
    ///
    /// Panics if `delay > max_delay` or `target >= neuron_count`. Prefer
    /// [`SpikeDelayBuffer::try_inject`] when the caller needs a recoverable
    /// error.
    #[inline]
    pub fn inject(&mut self, target: usize, weight: f32, delay: usize) {
        self.try_inject(target, weight, delay)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible inject that leaves the buffer unchanged when `delay` or
    /// `target` is out of range.
    #[inline]
    pub fn try_inject(&mut self, target: usize, weight: f32, delay: usize) -> Result<()> {
        if delay > self.max_delay {
            return Err(MeshError::DelayError(format!(
                "delay {delay} exceeds max_delay {}",
                self.max_delay
            )));
        }
        if target >= self.neuron_count {
            return Err(MeshError::IndexOutOfBounds {
                index: target,
                max: self.neuron_count.saturating_sub(1),
            });
        }
        // Constructors keep depth == max_delay + 1, but a deserialized
        // ring can still be shorter. Refuse delay >= depth so we never
        // wrap onto an earlier tick.
        let depth = self.slots.len();
        if depth == 0 || delay >= depth {
            return Err(MeshError::DelayError(format!(
                "delay {delay} exceeds buffer depth {}",
                depth.saturating_sub(1)
            )));
        }
        let slot_idx = self.slot_index(delay);
        self.slots[slot_idx][target] += weight;
        Ok(())
    }

    /// Current scheduled at `target` for `delay` ticks from now.
    #[inline]
    pub(crate) fn scheduled_current(&self, target: usize, delay: usize) -> f32 {
        debug_assert!(
            delay <= self.max_delay,
            "delay {delay} > max_delay {}",
            self.max_delay
        );
        debug_assert!(
            target < self.neuron_count,
            "target {target} >= neuron_count {}",
            self.neuron_count
        );
        self.slots[self.slot_index(delay)][target]
    }

    /// Reduce `current_tick` modulo ring depth before adding `delay` so
    /// `current_tick + delay` cannot overflow `usize`.
    #[inline]
    fn slot_index(&self, delay: usize) -> usize {
        let depth = self.slots.len();
        debug_assert!(depth > 0);
        let tick_mod = (self.current_tick % depth as u64) as usize;
        tick_mod.wrapping_add(delay) % depth
    }

    /// Drain the current tick's accumulated synaptic currents.
    ///
    /// Returns a vector of length `neuron_count` with the total synaptic
    /// current arriving at each neuron in this tick. The slot is zeroed
    /// after draining.
    ///
    /// Prefer [`SpikeDelayBuffer::drain_current_tick_into`] when the caller
    /// can reuse an output buffer.
    pub fn drain_current_tick(&mut self) -> Vec<f32> {
        let mut currents = vec![0.0; self.neuron_count];
        self.drain_current_tick_into(&mut currents)
            .expect("output length matches neuron_count");
        currents
    }

    /// Drain the current tick into a caller-owned buffer.
    ///
    /// `output` is **overwritten** with this tick's currents and must have
    /// length `neuron_count()`. On success the drained slot is zeroed. On
    /// a length mismatch the buffer is left unchanged — `output` is not
    /// written and no slot is cleared.
    pub fn drain_current_tick_into(&mut self, output: &mut [f32]) -> Result<()> {
        if output.len() != self.neuron_count {
            return Err(MeshError::NeuronCountMismatch {
                expected: self.neuron_count,
                got: output.len(),
                context: "drain_current_tick_into output".into(),
            });
        }
        let slot_idx = self.slot_index(0);
        output.copy_from_slice(&self.slots[slot_idx]);
        self.slots[slot_idx].fill(0.0);
        Ok(())
    }

    /// Advance to the next tick.
    pub fn advance(&mut self) {
        self.current_tick += 1;
    }

    /// Current simulation tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Maximum delay supported by this buffer.
    pub fn max_delay(&self) -> usize {
        self.max_delay
    }

    /// Number of neurons (slot width).
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Reset all slots and the tick counter.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.fill(0.0);
        }
        self.current_tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_delay_delivers_same_tick() {
        let mut buf = SpikeDelayBuffer::new(4, 0);
        buf.inject(2, 0.75, 0);
        let currents = buf.drain_current_tick();
        assert!((currents[2] - 0.75).abs() < 1e-6);
        assert_eq!(currents[0], 0.0);
    }

    #[test]
    fn delayed_delivery() {
        let mut buf = SpikeDelayBuffer::new(4, 5);

        // Inject at tick 0 with delay 3 → should arrive at tick 3
        buf.inject(1, 0.5, 3);

        // Tick 0: nothing delivered to neuron 1
        let c0 = buf.drain_current_tick();
        assert_eq!(c0[1], 0.0);
        buf.advance();

        // Tick 1: nothing
        let c1 = buf.drain_current_tick();
        assert_eq!(c1[1], 0.0);
        buf.advance();

        // Tick 2: nothing
        let c2 = buf.drain_current_tick();
        assert_eq!(c2[1], 0.0);
        buf.advance();

        // Tick 3: delivered!
        let c3 = buf.drain_current_tick();
        assert!((c3[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn multiple_spikes_accumulate() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 0.3, 2);
        buf.inject(0, 0.7, 2);

        buf.advance();
        buf.advance();

        let currents = buf.drain_current_tick();
        assert!((currents[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ring_buffer_wraps_correctly() {
        let mut buf = SpikeDelayBuffer::new(2, 3);

        // Run for more ticks than the ring buffer depth
        for tick in 0..10 {
            buf.inject(0, 1.0, 2);
            let currents = buf.drain_current_tick();
            if tick >= 2 {
                // After tick 2, we should receive the spike injected 2 ticks ago
                assert!(
                    (currents[0] - 1.0).abs() < 1e-6,
                    "tick {tick}: expected 1.0, got {}",
                    currents[0]
                );
            }
            buf.advance();
        }
    }

    #[test]
    fn reset_clears_everything() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 0.5, 3);
        buf.advance();
        buf.advance();
        buf.reset();

        assert_eq!(buf.current_tick(), 0);
        let currents = buf.drain_current_tick();
        assert!(currents.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn inject_delay_zero_is_accepted() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(1, 1.0, 0);
        let currents = buffer.drain_current_tick();
        assert!((currents[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inject_delay_equal_to_capacity_is_accepted() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(0, 1.0, 1);
        let c0 = buffer.drain_current_tick();
        assert_eq!(c0[0], 0.0);
        buffer.advance();
        let c1 = buffer.drain_current_tick();
        assert!((c1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "delay 2 exceeds max_delay 1")]
    fn inject_excessive_delay_is_rejected() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(1, 1.0, 2);
    }

    #[test]
    fn try_inject_excessive_delay_leaves_buffer_unchanged() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        assert!(buffer.try_inject(1, 1.0, 2).is_err());
        let currents = buffer.drain_current_tick();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn inject_out_of_range_target_is_rejected() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(2, 1.0, 0);
    }

    #[test]
    fn try_inject_out_of_range_target_leaves_buffer_unchanged() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        assert!(buffer.try_inject(2, 1.0, 0).is_err());
        let currents = buffer.drain_current_tick();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn try_new_rejects_max_delay_plus_one_overflow() {
        let err = SpikeDelayBuffer::try_new(1, usize::MAX).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("overflows usize") && msg.contains("ring depth cannot be represented"),
            "expected overflow error, got {msg}"
        );
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn new_rejects_max_delay_plus_one_overflow() {
        let _ = SpikeDelayBuffer::new(1, usize::MAX);
    }

    #[test]
    fn deserialize_rejects_empty_slots() {
        let json = r#"{"slots":[],"neuron_count":2,"max_delay":1,"current_tick":0}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_wrong_depth() {
        let json = r#"{"slots":[[0.0,0.0]],"neuron_count":2,"max_delay":1,"current_tick":0}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_ragged_slot_widths() {
        let json = r#"{"slots":[[0.0,0.0],[0.0]],"neuron_count":2,"max_delay":1,"current_tick":0}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
    }

    #[test]
    fn deserialize_rejects_max_delay_plus_one_overflow() {
        let json =
            r#"{"slots":[],"neuron_count":1,"max_delay":18446744073709551615,"current_tick":0}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
        assert!(validate_delay_buffer_shape(&[], 1, usize::MAX).is_err());
    }

    #[test]
    fn deserialize_rejects_current_tick_at_u64_max() {
        let json = r#"{"slots":[[0.0],[0.0]],"neuron_count":1,"max_delay":1,"current_tick":18446744073709551615}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
        assert!(validate_current_tick(u64::MAX, 1).is_err());
        assert!(validate_current_tick(usize::MAX as u64, 1).is_err());
        assert!(validate_current_tick(0, 1).is_ok());
    }

    #[test]
    fn deserialize_rejects_current_tick_that_advances_to_usize_max() {
        let tick = usize::MAX as u64 - 1;
        let json_delay0 =
            format!(r#"{{"slots":[[0.0]],"neuron_count":1,"max_delay":0,"current_tick":{tick}}}"#);
        let json_delay1 = format!(
            r#"{{"slots":[[0.0],[0.0]],"neuron_count":1,"max_delay":1,"current_tick":{tick}}}"#
        );
        assert!(serde_json::from_str::<SpikeDelayBuffer>(&json_delay0).is_err());
        assert!(serde_json::from_str::<SpikeDelayBuffer>(&json_delay1).is_err());
        assert!(validate_current_tick(tick, 0).is_err());
        assert!(validate_current_tick(tick, 1).is_err());
        assert!(validate_current_tick(usize::MAX as u64 - 2, 0).is_ok());
        assert!(validate_current_tick(usize::MAX as u64 - 2, 1).is_ok());
    }

    #[test]
    fn deserialize_rejects_current_tick_that_overflows_inject_arithmetic() {
        let tick = usize::MAX as u64 - 1;
        let json = format!(
            r#"{{"slots":[[0.0],[0.0],[0.0]],"neuron_count":1,"max_delay":2,"current_tick":{tick}}}"#
        );
        let err = serde_json::from_str::<SpikeDelayBuffer>(&json).unwrap_err();
        assert!(
            err.to_string().contains("overflowing usize indexing"),
            "unexpected error: {err}"
        );
        assert!(validate_current_tick(tick, 2).is_err());
        // Boundary: current_tick + max_delay == usize::MAX is still safe.
        assert!(validate_current_tick(usize::MAX as u64 - 2, 2).is_ok());
    }

    #[test]
    fn deserialize_accepts_current_tick_at_inject_arithmetic_limit() {
        let tick = usize::MAX as u64 - 2;
        let json = format!(
            r#"{{"slots":[[0.0],[0.0],[0.0]],"neuron_count":1,"max_delay":2,"current_tick":{tick}}}"#
        );
        let mut buf: SpikeDelayBuffer = serde_json::from_str(&json).unwrap();
        buf.inject(0, 1.0, 2);
        assert_eq!(buf.current_tick(), tick);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn deserialize_rejects_current_tick_that_wraps_usize() {
        let json =
            r#"{"slots":[[0.0],[0.0]],"neuron_count":1,"max_delay":1,"current_tick":4294967296}"#;
        assert!(serde_json::from_str::<SpikeDelayBuffer>(json).is_err());
        assert!(validate_current_tick((usize::MAX as u64) + 1, 1).is_err());
    }

    #[test]
    fn validate_delay_buffer_shape_rejects_non_finite_currents() {
        // JSON has no NaN/Infinity literal; a non-JSON serde format could
        // otherwise smuggle these values into drain_current_tick sums.
        assert!(validate_delay_buffer_shape(&[vec![0.1], vec![f32::NAN]], 1, 1).is_err());
        assert!(validate_delay_buffer_shape(&[vec![f32::INFINITY], vec![0.0]], 1, 1).is_err());
        assert!(validate_delay_buffer_shape(&[vec![0.0], vec![f32::NEG_INFINITY]], 1, 1).is_err());
        assert!(validate_delay_buffer_shape(&[vec![0.1, -0.0], vec![2.0, 0.0]], 2, 1).is_ok());
    }

    #[test]
    fn deserialize_accepts_zero_neuron_and_zero_delay() {
        let json = r#"{"slots":[[]],"neuron_count":0,"max_delay":0,"current_tick":0}"#;
        let buf: SpikeDelayBuffer = serde_json::from_str(json).unwrap();
        assert_eq!(buf.neuron_count(), 0);
        assert_eq!(buf.max_delay(), 0);
    }

    #[test]
    fn serialize_roundtrip_preserves_in_flight_current() {
        let mut buf = SpikeDelayBuffer::new(2, 1);
        buf.inject(1, 0.5, 1);
        let json = serde_json::to_string(&buf).unwrap();
        let mut restored: SpikeDelayBuffer = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.current_tick(), 0);
        assert_eq!(restored.drain_current_tick()[1], 0.0);
        restored.advance();
        assert!((restored.drain_current_tick()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn drain_into_matches_allocating_drain() {
        let mut a = SpikeDelayBuffer::new(4, 2);
        let mut b = SpikeDelayBuffer::new(4, 2);
        a.inject(1, 0.5, 0);
        a.inject(3, 1.25, 0);
        b.inject(1, 0.5, 0);
        b.inject(3, 1.25, 0);

        let allocated = a.drain_current_tick();
        let mut reused = vec![42.0; 4];
        b.drain_current_tick_into(&mut reused).unwrap();
        assert_eq!(allocated, reused);
    }

    #[test]
    fn drain_into_length_mismatch_leaves_buffer_and_output_unchanged() {
        let mut buf = SpikeDelayBuffer::new(2, 1);
        buf.inject(0, 1.0, 0);
        let mut output = [42.0_f32, 42.0, 42.0];
        let err = buf.drain_current_tick_into(&mut output).unwrap_err();
        assert!(err.to_string().contains("drain_current_tick_into output"));
        assert_eq!(output, [42.0, 42.0, 42.0]);
        let currents = buf.drain_current_tick();
        assert!((currents[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn different_delays_different_arrival() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 1.0, 1); // arrives tick 1
        buf.inject(1, 2.0, 3); // arrives tick 3

        buf.advance(); // tick 1
        let c1 = buf.drain_current_tick();
        assert!((c1[0] - 1.0).abs() < 1e-6);
        assert_eq!(c1[1], 0.0);

        buf.advance(); // tick 2
        let c2 = buf.drain_current_tick();
        assert_eq!(c2[0], 0.0);
        assert_eq!(c2[1], 0.0);

        buf.advance(); // tick 3
        let c3 = buf.drain_current_tick();
        assert_eq!(c3[0], 0.0);
        assert!((c3[1] - 2.0).abs() < 1e-6);
    }
}
