// SPDX-License-Identifier: MIT OR Apache-2.0

//! Float math that is not in `core` at the 1.85 MSRV.
//!
//! With `std` these are the inherent `f32` methods, so the default build is
//! unchanged. Without `std` (the `alloc` feature) they come from `libm`.
//! `round` and `sqrt` are exact / correctly rounded in both providers, so
//! generated delays agree bit-for-bit; `powf` may differ in the last ulp
//! between the platform library and `libm` (it already differs between
//! platforms), which only reorders near-tied preferential-attachment keys.

#[cfg(feature = "std")]
#[inline]
pub(crate) fn round(x: f32) -> f32 {
    x.round()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn round(x: f32) -> f32 {
    libm::roundf(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_matches_ties_away_from_zero() {
        assert_eq!(round(0.5), 1.0);
        assert_eq!(round(1.5), 2.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(-0.5), -1.0);
        assert_eq!(round(0.49), 0.0);
    }

    #[test]
    fn sqrt_is_exact_on_squares() {
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt(2.25), 1.5);
    }

    #[test]
    fn powf_matches_known_values() {
        assert_eq!(powf(2.0, 3.0), 8.0);
        assert_eq!(powf(0.25, 0.5), 0.5);
        assert!((powf(0.3, 1.0 / 7.0) - 0.841_982_4).abs() < 1e-6);
    }
}
