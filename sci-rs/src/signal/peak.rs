//! Peak-finding and wavelet helpers analogous to `scipy.signal` peak APIs.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::cmp::Ordering;
use num_traits::{Float, FromPrimitive};

/// Options for [`find_peaks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FindPeaksOptions<F>
where
    F: PartialOrd + Copy,
{
    /// Minimum peak height.
    pub height: Option<F>,
    /// Minimum index distance between retained peaks.
    pub distance: Option<usize>,
}

impl<F> Default for FindPeaksOptions<F>
where
    F: PartialOrd + Copy,
{
    fn default() -> Self {
        Self {
            height: None,
            distance: None,
        }
    }
}

/// Prominence result bundle for [`peak_prominences`].
#[derive(Debug, Clone, PartialEq)]
pub struct PeakProminencesResult<F>
where
    F: Float + Copy,
{
    /// Prominence value for each input peak index.
    pub prominences: Vec<F>,
    /// Left base index for each peak.
    pub left_bases: Vec<usize>,
    /// Right base index for each peak.
    pub right_bases: Vec<usize>,
}

/// Width result bundle for [`peak_widths`].
#[derive(Debug, Clone, PartialEq)]
pub struct PeakWidthsResult<F>
where
    F: Float + Copy,
{
    /// Width of each peak at the evaluated height.
    pub widths: Vec<F>,
    /// Height level used for each width.
    pub width_heights: Vec<F>,
    /// Left interpolated intersection point (index domain).
    pub left_ips: Vec<F>,
    /// Right interpolated intersection point (index domain).
    pub right_ips: Vec<F>,
}

/// Return indices of relative extrema according to `comparator`.
pub fn argrelextrema<F, C>(x: &[F], comparator: C, order: usize) -> Vec<usize>
where
    F: PartialOrd + Copy,
    C: Fn(F, F) -> bool,
{
    if x.is_empty() || order == 0 || x.len() < (2 * order + 1) {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in order..(x.len() - order) {
        let center = x[i];
        let mut is_extremum = true;
        for k in 1..=order {
            if !(comparator(center, x[i - k]) && comparator(center, x[i + k])) {
                is_extremum = false;
                break;
            }
        }
        if is_extremum {
            out.push(i);
        }
    }
    out
}

/// Return indices of relative maxima.
pub fn argrelmax<F>(x: &[F], order: usize) -> Vec<usize>
where
    F: PartialOrd + Copy,
{
    argrelextrema(x, |a, b| a > b, order)
}

/// Return indices of relative minima.
pub fn argrelmin<F>(x: &[F], order: usize) -> Vec<usize>
where
    F: PartialOrd + Copy,
{
    argrelextrema(x, |a, b| a < b, order)
}

/// Find local peaks with optional height and distance filtering.
pub fn find_peaks<F>(x: &[F], options: FindPeaksOptions<F>) -> Vec<usize>
where
    F: PartialOrd + Copy,
{
    let mut peaks = argrelmax(x, 1);

    if let Some(height) = options.height {
        peaks.retain(|&idx| x[idx] >= height);
    }

    if let Some(distance) = options.distance {
        if distance > 1 && !peaks.is_empty() {
            let mut ranked = peaks.clone();
            ranked.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap_or(Ordering::Equal));

            let mut selected = Vec::new();
            for cand in ranked {
                if selected
                    .iter()
                    .all(|&kept: &usize| kept.abs_diff(cand) >= distance)
                {
                    selected.push(cand);
                }
            }
            selected.sort_unstable();
            peaks = selected;
        }
    }

    peaks
}

/// Compute peak prominences and base indices.
pub fn peak_prominences<F>(x: &[F], peaks: &[usize]) -> PeakProminencesResult<F>
where
    F: Float + Copy,
{
    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &peak in peaks {
        if peak >= x.len() {
            continue;
        }
        let peak_val = x[peak];

        let mut left_min = peak_val;
        let mut left_base = peak;
        let mut i = peak;
        while i > 0 {
            i -= 1;
            let v = x[i];
            if v > peak_val {
                break;
            }
            if v < left_min {
                left_min = v;
                left_base = i;
            }
        }

        let mut right_min = peak_val;
        let mut right_base = peak;
        i = peak;
        while i + 1 < x.len() {
            i += 1;
            let v = x[i];
            if v > peak_val {
                break;
            }
            if v < right_min {
                right_min = v;
                right_base = i;
            }
        }

        let base_level = if left_min > right_min {
            left_min
        } else {
            right_min
        };
        prominences.push(peak_val - base_level);
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    PeakProminencesResult {
        prominences,
        left_bases,
        right_bases,
    }
}

/// Compute peak widths at relative height.
pub fn peak_widths<F>(x: &[F], peaks: &[usize], rel_height: F) -> PeakWidthsResult<F>
where
    F: Float + Copy + FromPrimitive,
{
    let prom = peak_prominences(x, peaks);
    let valid_peaks: Vec<usize> = peaks.iter().copied().filter(|&p| p < x.len()).collect();
    let mut widths = Vec::with_capacity(prom.prominences.len());
    let mut width_heights = Vec::with_capacity(prom.prominences.len());
    let mut left_ips = Vec::with_capacity(prom.prominences.len());
    let mut right_ips = Vec::with_capacity(prom.prominences.len());

    for (i, peak) in valid_peaks
        .iter()
        .copied()
        .enumerate()
        .take(prom.prominences.len())
    {
        let peak_val = x[peak];
        let width_height = peak_val - prom.prominences[i] * rel_height;
        let left_base = prom.left_bases[i];
        let right_base = prom.right_bases[i];

        let mut l = peak;
        while l > left_base && x[l] > width_height {
            l -= 1;
        }
        let l_ip = if l < peak && x[l + 1] != x[l] {
            let denom = x[l + 1] - x[l];
            F::from_usize(l).expect("index conversion") + (width_height - x[l]) / denom
        } else {
            F::from_usize(l).expect("index conversion")
        };

        let mut r = peak;
        while r < right_base && x[r] > width_height {
            r += 1;
        }
        let r_ip = if r > peak && x[r - 1] != x[r] {
            let denom = x[r - 1] - x[r];
            F::from_usize(r).expect("index conversion") - (width_height - x[r]) / denom
        } else {
            F::from_usize(r).expect("index conversion")
        };

        widths.push(r_ip - l_ip);
        width_heights.push(width_height);
        left_ips.push(l_ip);
        right_ips.push(r_ip);
    }

    PeakWidthsResult {
        widths,
        width_heights,
        left_ips,
        right_ips,
    }
}

fn ricker_wavelet<F>(points: usize, a: F) -> Vec<F>
where
    F: Float + Copy + FromPrimitive,
{
    if points == 0 {
        return Vec::new();
    }
    let pi = F::from_f64(core::f64::consts::PI).expect("pi conversion");
    let two = F::from_f64(2.0).expect("scalar conversion");
    let three = F::from_f64(3.0).expect("scalar conversion");
    let quarter = F::from_f64(0.25).expect("scalar conversion");
    let half = F::from_f64(0.5).expect("scalar conversion");
    let one = F::one();

    let norm = two / ((three * a).sqrt() * pi.powf(quarter));
    let center = F::from_usize(points - 1).expect("scalar conversion") * half;

    (0..points)
        .map(|i| {
            let x = F::from_usize(i).expect("scalar conversion") - center;
            let xa = x / a;
            norm * (one - xa * xa) * (-(x * x) / (two * a * a)).exp()
        })
        .collect()
}

fn convolve_same<F>(x: &[F], h: &[F]) -> Vec<F>
where
    F: Float + Copy,
{
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let n = x.len();
    let m = h.len();
    let half = m / 2;
    let mut out = vec![F::zero(); n];
    for (i, out_i) in out.iter_mut().enumerate().take(n) {
        let mut acc = F::zero();
        for j in 0..m {
            let idx = i + j;
            if idx >= half {
                let x_idx = idx - half;
                if x_idx < n {
                    acc = acc + x[x_idx] * h[m - 1 - j];
                }
            }
        }
        *out_i = acc;
    }
    out
}

/// Continuous wavelet transform using a Ricker wavelet family.
pub fn cwt<F>(x: &[F], widths: &[usize]) -> Vec<Vec<F>>
where
    F: Float + Copy + FromPrimitive,
{
    if x.is_empty() || widths.is_empty() {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(widths.len());
    for &w in widths {
        let w = w.max(1);
        let n_points = (10 * w).max(3).min(x.len().max(3));
        let wavelet = ricker_wavelet(n_points, F::from_usize(w).expect("scalar conversion"));
        rows.push(convolve_same(x, &wavelet));
    }
    rows
}

/// Find peaks from a wavelet-enhanced score map.
pub fn find_peaks_cwt<F>(x: &[F], widths: &[usize]) -> Vec<usize>
where
    F: Float + Copy + FromPrimitive,
{
    if x.is_empty() || widths.is_empty() {
        return Vec::new();
    }
    let cwt_map = cwt(x, widths);
    let mut score = vec![F::zero(); x.len()];
    for row in &cwt_map {
        for (i, val) in row.iter().enumerate() {
            score[i] = score[i] + val.abs();
        }
    }
    let min_distance = widths.iter().copied().min().unwrap_or(1).max(1);
    find_peaks(
        &score,
        FindPeaksOptions {
            height: None,
            distance: Some(min_distance),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn argrel_variants_find_expected_indices() {
        let x = [0.0f64, 1.0, 0.0, -1.0, 0.0, 2.0, 1.0];
        assert_eq!(argrelmax(&x, 1), vec![1, 5]);
        assert_eq!(argrelmin(&x, 1), vec![3]);
        assert_eq!(argrelextrema(&x, |a, b| a > b, 1), vec![1, 5]);
    }

    #[test]
    fn find_peaks_applies_height_and_distance() {
        let x = [0.0f64, 1.0, 0.1, 0.9, 0.0, 2.0, 0.0];
        let peaks = find_peaks(
            &x,
            FindPeaksOptions {
                height: Some(0.5),
                distance: Some(3),
            },
        );
        assert_eq!(peaks, vec![1, 5]);
    }

    #[test]
    fn prominences_and_widths_return_valid_shapes() {
        let x = [0.0f64, 1.0, 0.2, 0.8, 0.1, 2.0, 0.0];
        let peaks = vec![1, 5];
        let prom = peak_prominences(&x, &peaks);
        assert_eq!(prom.prominences.len(), 2);
        assert_eq!(prom.left_bases.len(), 2);
        assert_eq!(prom.right_bases.len(), 2);
        assert!(prom.prominences[0] > 0.0);
        assert!(prom.prominences[1] > 0.0);

        let widths = peak_widths(&x, &peaks, 0.5);
        assert_eq!(widths.widths.len(), 2);
        assert_eq!(widths.width_heights.len(), 2);
        assert!(widths.widths[0] >= 0.0);
        assert!(widths.widths[1] >= 0.0);
    }

    #[test]
    fn cwt_shape_matches_widths_and_input_len() {
        let x = [0.0f64, 1.0, 0.0, -1.0, 0.0];
        let widths = [1usize, 2, 3];
        let m = cwt(&x, &widths);
        assert_eq!(m.len(), widths.len());
        for row in &m {
            assert_eq!(row.len(), x.len());
        }
    }

    #[test]
    fn find_peaks_cwt_detects_main_peak() {
        let x = [0.0f64, 0.3, 1.2, 0.2, 0.1, 0.0];
        let peaks = find_peaks_cwt(&x, &[1, 2, 3]);
        assert!(peaks.contains(&2));
    }

    #[test]
    fn peak_width_matches_simple_triangle() {
        let x = [0.0f64, 1.0, 0.0];
        let widths = peak_widths(&x, &[1], 0.5);
        assert_eq!(widths.widths.len(), 1);
        assert_abs_diff_eq!(widths.widths[0], 1.0, epsilon = 1e-12);
    }
}
