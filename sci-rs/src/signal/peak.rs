//! Peak-finding and wavelet helpers analogous to `scipy.signal` peak APIs.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D};
use crate::signal::traits::{
    ArgRelExtrema1D, Cwt1D, FindPeaks1D, FindPeaksCwt1D, PeakProminence1D, PeakWidths1D,
};
use core::cmp::Ordering;
use num_traits::{Float, FromPrimitive};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

impl<F> Default for PeakProminencesResult<F>
where
    F: Float + Copy,
{
    fn default() -> Self {
        Self {
            prominences: Vec::new(),
            left_bases: Vec::new(),
            right_bases: Vec::new(),
        }
    }
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

impl<F> Default for PeakWidthsResult<F>
where
    F: Float + Copy,
{
    fn default() -> Self {
        Self {
            widths: Vec::new(),
            width_heights: Vec::new(),
            left_ips: Vec::new(),
            right_ips: Vec::new(),
        }
    }
}

fn argrelextrema_impl<F, C>(x: &[F], comparator: C, order: usize) -> Vec<usize>
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

fn find_peaks_impl<F>(x: &[F], options: FindPeaksOptions<F>) -> Vec<usize>
where
    F: PartialOrd + Copy,
{
    let mut peaks = argrelextrema_impl(x, |a, b| a > b, 1);

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

fn peak_prominences_impl<F>(x: &[F], peaks: &[usize]) -> PeakProminencesResult<F>
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

fn peak_widths_impl<F>(
    x: &[F],
    peaks: &[usize],
    rel_height: F,
) -> Result<PeakWidthsResult<F>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let prom = peak_prominences_impl(x, peaks);
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
            F::from_usize(l).ok_or(ExecInvariantViolation::InvalidState {
                reason: "peak_widths left index conversion failed",
            })? + (width_height - x[l]) / denom
        } else {
            F::from_usize(l).ok_or(ExecInvariantViolation::InvalidState {
                reason: "peak_widths left index conversion failed",
            })?
        };

        let mut r = peak;
        while r < right_base && x[r] > width_height {
            r += 1;
        }
        let r_ip = if r > peak && x[r - 1] != x[r] {
            let denom = x[r - 1] - x[r];
            F::from_usize(r).ok_or(ExecInvariantViolation::InvalidState {
                reason: "peak_widths right index conversion failed",
            })? - (width_height - x[r]) / denom
        } else {
            F::from_usize(r).ok_or(ExecInvariantViolation::InvalidState {
                reason: "peak_widths right index conversion failed",
            })?
        };

        widths.push(r_ip - l_ip);
        width_heights.push(width_height);
        left_ips.push(l_ip);
        right_ips.push(r_ip);
    }

    Ok(PeakWidthsResult {
        widths,
        width_heights,
        left_ips,
        right_ips,
    })
}

fn ricker_wavelet<F>(points: usize, a: F) -> Result<Vec<F>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    if points == 0 {
        return Ok(Vec::new());
    }
    let pi = F::from_f64(core::f64::consts::PI).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet pi conversion failed",
    })?;
    let two = F::from_f64(2.0).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet scalar conversion failed",
    })?;
    let three = F::from_f64(3.0).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet scalar conversion failed",
    })?;
    let quarter = F::from_f64(0.25).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet scalar conversion failed",
    })?;
    let half = F::from_f64(0.5).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet scalar conversion failed",
    })?;
    let one = F::one();

    let norm = two / ((three * a).sqrt() * pi.powf(quarter));
    let center = F::from_usize(points - 1).ok_or(ExecInvariantViolation::InvalidState {
        reason: "ricker_wavelet scalar conversion failed",
    })? * half;

    (0..points)
        .map(|i| {
            let x = F::from_usize(i).ok_or(ExecInvariantViolation::InvalidState {
                reason: "ricker_wavelet scalar conversion failed",
            })? - center;
            let xa = x / a;
            Ok(norm * (one - xa * xa) * (-(x * x) / (two * a * a)).exp())
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

fn cwt_impl<F>(x: &[F], widths: &[usize]) -> Result<Vec<Vec<F>>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    if x.is_empty() || widths.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::with_capacity(widths.len());
    for &w in widths {
        let n_points = (10 * w).max(3).min(x.len().max(3));
        let width_f = F::from_usize(w).ok_or(ExecInvariantViolation::InvalidState {
            reason: "cwt width conversion failed",
        })?;
        let wavelet = ricker_wavelet(n_points, width_f)?;
        rows.push(convolve_same(x, &wavelet));
    }
    Ok(rows)
}

/// Constructor config for [`ArgRelExtremaKernel`].
#[derive(Debug, Clone, Copy)]
pub struct ArgRelExtremaConfig<F>
where
    F: PartialOrd + Copy,
{
    /// Neighbor count considered on each side.
    pub order: usize,
    /// Comparator used to declare local extrema.
    pub comparator: fn(F, F) -> bool,
}

/// Trait-first extrema kernel.
#[derive(Debug, Clone, Copy)]
pub struct ArgRelExtremaKernel<F>
where
    F: PartialOrd + Copy,
{
    order: usize,
    comparator: fn(F, F) -> bool,
}

impl<F> KernelLifecycle for ArgRelExtremaKernel<F>
where
    F: PartialOrd + Copy,
{
    type Config = ArgRelExtremaConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.order == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be > 0",
            });
        }
        Ok(Self {
            order: config.order,
            comparator: config.comparator,
        })
    }
}

impl<F> ArgRelExtrema1D<F> for ArgRelExtremaKernel<F>
where
    F: PartialOrd + Copy,
{
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let result = self.run_alloc(input)?;
        out.clear();
        out.extend(result);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "argrelextrema input must be non-empty",
            });
        }
        Ok(argrelextrema_impl(input, self.comparator, self.order))
    }
}

/// Constructor config for [`FindPeaksKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FindPeaksConfig<F>
where
    F: PartialOrd + Copy,
{
    /// Minimum peak height.
    pub height: Option<F>,
    /// Minimum spacing between retained peaks.
    pub distance: Option<usize>,
}

impl<F> From<FindPeaksOptions<F>> for FindPeaksConfig<F>
where
    F: PartialOrd + Copy,
{
    fn from(value: FindPeaksOptions<F>) -> Self {
        Self {
            height: value.height,
            distance: value.distance,
        }
    }
}

/// Trait-first `find_peaks` kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FindPeaksKernel<F>
where
    F: PartialOrd + Copy,
{
    height: Option<F>,
    distance: Option<usize>,
}

impl<F> KernelLifecycle for FindPeaksKernel<F>
where
    F: PartialOrd + Copy,
{
    type Config = FindPeaksConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if matches!(config.distance, Some(0)) {
            return Err(ConfigError::InvalidArgument {
                arg: "distance",
                reason: "distance must be >= 1 when provided",
            });
        }
        Ok(Self {
            height: config.height,
            distance: config.distance,
        })
    }
}

impl<F> FindPeaks1D<F> for FindPeaksKernel<F>
where
    F: PartialOrd + Copy,
{
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let result = self.run_alloc(input)?;
        out.clear();
        out.extend(result);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "find_peaks input must be non-empty",
            });
        }
        Ok(find_peaks_impl(
            input,
            FindPeaksOptions {
                height: self.height,
                distance: self.distance,
            },
        ))
    }
}

/// Constructor config for [`PeakProminencesKernel`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeakProminencesConfig;

/// Trait-first `peak_prominences` kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeakProminencesKernel;

impl KernelLifecycle for PeakProminencesKernel {
    type Config = PeakProminencesConfig;

    fn try_new(_config: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<F> PeakProminence1D<F> for PeakProminencesKernel
where
    F: Float + Copy,
{
    type Output = PeakProminencesResult<F>;

    fn run_into<I, P>(
        &self,
        input: &I,
        peaks: &P,
        out: &mut Self::Output,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        P: Read1D<usize> + ?Sized,
    {
        *out = self.run_alloc(input, peaks)?;
        Ok(())
    }

    fn run_alloc<I, P>(&self, input: &I, peaks: &P) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        P: Read1D<usize> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "peak_prominences input must be non-empty",
            });
        }
        let peaks = peaks.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(peak_prominences_impl(input, peaks))
    }
}

/// Constructor config for [`PeakWidthsKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakWidthsConfig<F>
where
    F: Float + Copy,
{
    /// Relative prominence height used for width calculation.
    pub rel_height: F,
}

/// Trait-first `peak_widths` kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakWidthsKernel<F>
where
    F: Float + Copy,
{
    rel_height: F,
}

impl<F> KernelLifecycle for PeakWidthsKernel<F>
where
    F: Float + Copy,
{
    type Config = PeakWidthsConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.rel_height.is_finite() || config.rel_height < F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "rel_height",
                reason: "rel_height must be finite and >= 0",
            });
        }
        Ok(Self {
            rel_height: config.rel_height,
        })
    }
}

impl<F> PeakWidths1D<F> for PeakWidthsKernel<F>
where
    F: Float + Copy + FromPrimitive,
{
    type Output = PeakWidthsResult<F>;

    fn run_into<I, P>(
        &self,
        input: &I,
        peaks: &P,
        out: &mut Self::Output,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        P: Read1D<usize> + ?Sized,
    {
        *out = self.run_alloc(input, peaks)?;
        Ok(())
    }

    fn run_alloc<I, P>(&self, input: &I, peaks: &P) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        P: Read1D<usize> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "peak_widths input must be non-empty",
            });
        }
        let peaks = peaks.read_slice().map_err(ExecInvariantViolation::from)?;
        peak_widths_impl(input, peaks, self.rel_height)
    }
}

/// Constructor config for [`CwtKernel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CwtConfig {
    /// Positive wavelet widths.
    pub widths: Vec<usize>,
}

/// Trait-first CWT kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CwtKernel {
    widths: Vec<usize>,
}

impl KernelLifecycle for CwtKernel {
    type Config = CwtConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.widths.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "widths" });
        }
        if config.widths.contains(&0) {
            return Err(ConfigError::InvalidArgument {
                arg: "widths",
                reason: "all widths must be > 0",
            });
        }
        Ok(Self {
            widths: config.widths,
        })
    }
}

impl<F> Cwt1D<F> for CwtKernel
where
    F: Float + Copy + FromPrimitive,
{
    type Output = Vec<Vec<F>>;

    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        *out = self.run_alloc(input)?;
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "cwt input must be non-empty",
            });
        }
        cwt_impl(input, &self.widths)
    }
}

/// Constructor config for [`FindPeaksCwtKernel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindPeaksCwtConfig {
    /// Positive wavelet widths.
    pub widths: Vec<usize>,
}

/// Trait-first `find_peaks_cwt` kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindPeaksCwtKernel {
    widths: Vec<usize>,
}

impl KernelLifecycle for FindPeaksCwtKernel {
    type Config = FindPeaksCwtConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.widths.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "widths" });
        }
        if config.widths.contains(&0) {
            return Err(ConfigError::InvalidArgument {
                arg: "widths",
                reason: "all widths must be > 0",
            });
        }
        Ok(Self {
            widths: config.widths,
        })
    }
}

impl<F> FindPeaksCwt1D<F> for FindPeaksCwtKernel
where
    F: Float + Copy + FromPrimitive,
{
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let result = self.run_alloc(input)?;
        out.clear();
        out.extend(result);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "find_peaks_cwt input must be non-empty",
            });
        }

        let cwt_map = cwt_impl(input, &self.widths)?;
        let mut score = vec![F::zero(); input.len()];
        for row in &cwt_map {
            for (i, val) in row.iter().enumerate() {
                score[i] = score[i] + val.abs();
            }
        }

        let min_distance = self.widths.iter().copied().min().unwrap_or(1).max(1);
        let peaks_kernel = FindPeaksKernel::try_new(FindPeaksConfig {
            height: None,
            distance: Some(min_distance),
        })
        .map_err(ExecInvariantViolation::from)?;
        peaks_kernel.run_alloc(&score)
    }
}

/// Return indices of relative extrema according to `comparator`.
pub(crate) fn argrelextrema<F>(
    x: &[F],
    comparator: fn(F, F) -> bool,
    order: usize,
) -> Result<Vec<usize>, ExecInvariantViolation>
where
    F: PartialOrd + Copy,
{
    let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig { order, comparator })
        .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Return indices of relative maxima.
pub(crate) fn argrelmax<F>(x: &[F], order: usize) -> Result<Vec<usize>, ExecInvariantViolation>
where
    F: PartialOrd + Copy,
{
    let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
        order,
        comparator: |a, b| a > b,
    })
    .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Return indices of relative minima.
pub(crate) fn argrelmin<F>(x: &[F], order: usize) -> Result<Vec<usize>, ExecInvariantViolation>
where
    F: PartialOrd + Copy,
{
    let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
        order,
        comparator: |a, b| a < b,
    })
    .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Find local peaks with optional height and distance filtering.
pub(crate) fn find_peaks<F>(
    x: &[F],
    options: FindPeaksOptions<F>,
) -> Result<Vec<usize>, ExecInvariantViolation>
where
    F: PartialOrd + Copy,
{
    let kernel = FindPeaksKernel::try_new(options.into()).map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Compute peak prominences and base indices.
pub(crate) fn peak_prominences<F>(
    x: &[F],
    peaks: &[usize],
) -> Result<PeakProminencesResult<F>, ExecInvariantViolation>
where
    F: Float + Copy,
{
    let kernel = PeakProminencesKernel::try_new(PeakProminencesConfig)
        .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x, peaks)
}

/// Compute peak widths at relative height.
pub(crate) fn peak_widths<F>(
    x: &[F],
    peaks: &[usize],
    rel_height: F,
) -> Result<PeakWidthsResult<F>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let kernel = PeakWidthsKernel::try_new(PeakWidthsConfig { rel_height })
        .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x, peaks)
}

/// Continuous wavelet transform using a Ricker wavelet family.
pub(crate) fn cwt<F>(x: &[F], widths: &[usize]) -> Result<Vec<Vec<F>>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let kernel = CwtKernel::try_new(CwtConfig {
        widths: widths.to_vec(),
    })
    .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Find peaks from a wavelet-enhanced score map.
pub(crate) fn find_peaks_cwt<F>(
    x: &[F],
    widths: &[usize],
) -> Result<Vec<usize>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let kernel = FindPeaksCwtKernel::try_new(FindPeaksCwtConfig {
        widths: widths.to_vec(),
    })
    .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn argrel_variants_find_expected_indices() {
        let x = [0.0f64, 1.0, 0.0, -1.0, 0.0, 2.0, 1.0];
        assert_eq!(
            argrelmax(&x, 1).expect("argrelmax should succeed"),
            vec![1, 5]
        );
        assert_eq!(argrelmin(&x, 1).expect("argrelmin should succeed"), vec![3]);
        assert_eq!(
            argrelextrema(&x, |a, b| a > b, 1).expect("argrelextrema should succeed"),
            vec![1, 5]
        );
    }

    #[test]
    fn argrel_kernel_contracts_validate_order() {
        let bad = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order: 0,
            comparator: |a: f64, b: f64| a > b,
        });
        assert!(bad.is_err());

        let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order: 1,
            comparator: |a: f64, b: f64| a > b,
        })
        .expect("valid config");
        let x = [0.0f64, 1.0, 0.0];
        let mut out = vec![];
        kernel
            .run_into(&x, &mut out)
            .expect("argrelextrema run_into should succeed");
        assert_eq!(out, vec![1]);
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
        )
        .expect("find_peaks should succeed");
        assert_eq!(peaks, vec![1, 5]);
    }

    #[test]
    fn find_peaks_kernel_contracts_validate_distance() {
        let bad = FindPeaksKernel::<f64>::try_new(FindPeaksConfig {
            height: None,
            distance: Some(0),
        });
        assert!(bad.is_err());
    }

    #[test]
    fn prominences_and_widths_return_valid_shapes() {
        let x = [0.0f64, 1.0, 0.2, 0.8, 0.1, 2.0, 0.0];
        let peaks = vec![1, 5];
        let prom = peak_prominences(&x, &peaks).expect("peak_prominences should succeed");
        assert_eq!(prom.prominences.len(), 2);
        assert_eq!(prom.left_bases.len(), 2);
        assert_eq!(prom.right_bases.len(), 2);
        assert!(prom.prominences[0] > 0.0);
        assert!(prom.prominences[1] > 0.0);

        let widths = peak_widths(&x, &peaks, 0.5).expect("peak_widths should succeed");
        assert_eq!(widths.widths.len(), 2);
        assert_eq!(widths.width_heights.len(), 2);
        assert!(widths.widths[0] >= 0.0);
        assert!(widths.widths[1] >= 0.0);
    }

    #[test]
    fn peak_prominence_kernel_contract_round_trip() {
        let kernel = PeakProminencesKernel::try_new(PeakProminencesConfig).expect("valid config");
        let x = [0.0f64, 1.0, 0.2, 0.8, 0.1, 2.0, 0.0];
        let peaks = [1usize, 5usize];
        let mut out = PeakProminencesResult::<f64>::default();
        kernel
            .run_into(&x, &peaks, &mut out)
            .expect("peak prominence run_into should succeed");
        assert_eq!(out.prominences.len(), 2);
    }

    #[test]
    fn cwt_shape_matches_widths_and_input_len() {
        let x = [0.0f64, 1.0, 0.0, -1.0, 0.0];
        let widths = [1usize, 2, 3];
        let m = cwt(&x, &widths).expect("cwt should succeed");
        assert_eq!(m.len(), widths.len());
        for row in &m {
            assert_eq!(row.len(), x.len());
        }
    }

    #[test]
    fn cwt_kernel_contracts_validate_widths() {
        assert!(CwtKernel::try_new(CwtConfig { widths: vec![] }).is_err());
        assert!(CwtKernel::try_new(CwtConfig {
            widths: vec![1, 0, 3]
        })
        .is_err());
    }

    #[test]
    fn find_peaks_cwt_detects_main_peak() {
        let x = [0.0f64, 0.3, 1.2, 0.2, 0.1, 0.0];
        let peaks = find_peaks_cwt(&x, &[1, 2, 3]).expect("find_peaks_cwt should succeed");
        assert!(peaks.contains(&2));
    }

    #[test]
    fn find_peaks_cwt_kernel_contracts_validate_widths() {
        assert!(FindPeaksCwtKernel::try_new(FindPeaksCwtConfig { widths: vec![] }).is_err());
        assert!(FindPeaksCwtKernel::try_new(FindPeaksCwtConfig {
            widths: vec![1, 0, 3]
        })
        .is_err());
    }

    #[test]
    fn peak_width_matches_simple_triangle() {
        let x = [0.0f64, 1.0, 0.0];
        let widths = peak_widths(&x, &[1], 0.5).expect("peak_widths should succeed");
        assert_eq!(widths.widths.len(), 1);
        assert_abs_diff_eq!(widths.widths[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn peak_width_kernel_rejects_negative_rel_height() {
        let bad = PeakWidthsKernel::try_new(PeakWidthsConfig {
            rel_height: -0.1f64,
        });
        assert!(bad.is_err());
    }
}
