//! Multirate helpers analogous to `scipy.signal` multirate APIs.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use num_traits::{Float, FromPrimitive};

fn convolve_full<F>(x: &[F], h: &[F]) -> Vec<F>
where
    F: Float + Copy,
{
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let mut out = vec![F::zero(); x.len() + h.len() - 1];
    for (i, &xv) in x.iter().enumerate() {
        for (j, &hv) in h.iter().enumerate() {
            out[i + j] = out[i + j] + xv * hv;
        }
    }
    out
}

/// Upsample by `up`, apply FIR `h`, and downsample by `down`.
pub fn upfirdn<F>(h: &[F], x: &[F], up: usize, down: usize) -> Vec<F>
where
    F: Float + Copy,
{
    if h.is_empty() || x.is_empty() || up == 0 || down == 0 {
        return Vec::new();
    }

    let mut upsampled = vec![F::zero(); x.len() * up];
    for (i, &v) in x.iter().enumerate() {
        upsampled[i * up] = v;
    }

    let filtered = convolve_full(&upsampled, h);
    filtered.into_iter().step_by(down).collect()
}

/// Polyphase-like resampling using linear interpolation.
///
/// This implementation targets deterministic embedded-friendly behavior and
/// supports the common `up/down` ratio contract from SciPy's `resample_poly`.
pub fn resample_poly<F>(x: &[F], up: usize, down: usize) -> Vec<F>
where
    F: Float + Copy + FromPrimitive,
{
    if x.is_empty() || up == 0 || down == 0 {
        return Vec::new();
    }
    if up == down {
        return x.to_vec();
    }

    let out_len = (x.len() * up).div_ceil(down);
    let up_f = F::from_usize(up).expect("ratio conversion");
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let pos_num = i * down;
        let i0 = pos_num / up;
        let frac = F::from_usize(pos_num % up).expect("fraction conversion") / up_f;

        let y = if i0 + 1 < x.len() {
            x[i0] * (F::one() - frac) + x[i0 + 1] * frac
        } else {
            x[x.len() - 1]
        };
        out.push(y);
    }

    out
}

/// Decimate by integer factor `q`.
pub fn decimate<F>(x: &[F], q: usize) -> Vec<F>
where
    F: Float + Copy + FromPrimitive,
{
    if q == 0 {
        return Vec::new();
    }
    resample_poly(x, 1, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn upfirdn_matches_simple_reference() {
        let x = [1.0f64, 2.0, 3.0];
        let h = [1.0f64, 1.0];
        let y = upfirdn(&h, &x, 2, 1);
        let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0];
        assert_eq!(y, expected);
    }

    #[test]
    fn resample_poly_interpolates_to_expected_length() {
        let x = [1.0f64, 2.0, 3.0];
        let y = resample_poly(&x, 2, 1);
        let expected = [1.0, 1.5, 2.0, 2.5, 3.0, 3.0];
        assert_eq!(y.len(), expected.len());
        for (a, b) in y.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn decimate_reduces_length() {
        let x = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let y = decimate(&x, 2);
        assert_eq!(y, vec![0.0, 2.0, 4.0]);
    }
}
