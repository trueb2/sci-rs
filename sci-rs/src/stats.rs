use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use core::{borrow::Borrow, iter::Sum, ops::Add};
use itertools::Itertools;
use num_traits::{Float, Num, NumCast, Signed};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// 1D mean reduction capability.
pub trait MeanReduce1D<T> {
    /// Compute the mean and sample count.
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D variance reduction capability.
pub trait VarianceReduce1D<T> {
    /// Compute the variance and sample count.
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D standard-deviation reduction capability.
pub trait StdevReduce1D<T> {
    /// Compute the standard deviation and sample count.
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D median reduction capability.
#[cfg(feature = "alloc")]
pub trait MedianReduce1D<T> {
    /// Compute the median and sample count.
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D median absolute deviation reduction capability.
#[cfg(feature = "alloc")]
pub trait MadReduce1D<T> {
    /// Compute the median absolute deviation and sample count.
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D z-score normalization capability.
pub trait ZScoreNormalize1D<T> {
    /// Normalize into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Normalize and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D modified z-score normalization capability.
#[cfg(feature = "alloc")]
pub trait ModZScoreNormalize1D<T> {
    /// Normalize into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Normalize and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Empty config for stateless kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatsConfig;

/// Trait-first mean kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanKernel;

impl KernelLifecycle for MeanKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<T> MeanReduce1D<T> for MeanKernel
where
    T: Num + NumCast + Default + Copy + Add,
{
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(mean(input.iter()))
    }
}

/// Trait-first variance kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct VarianceKernel;

impl KernelLifecycle for VarianceKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<T> VarianceReduce1D<T> for VarianceKernel
where
    T: Float + Default + Sum,
{
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(variance(input.iter()))
    }
}

/// Trait-first standard deviation kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdevKernel;

impl KernelLifecycle for StdevKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<T> StdevReduce1D<T> for StdevKernel
where
    T: Float + Default + Sum,
{
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(stdev(input.iter()))
    }
}

/// Trait-first median kernel.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MedianKernel;

#[cfg(feature = "alloc")]
impl KernelLifecycle for MedianKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

#[cfg(feature = "alloc")]
impl<T> MedianReduce1D<T> for MedianKernel
where
    T: Num + NumCast + PartialOrd + Copy + Default,
{
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(median(input.iter()))
    }
}

/// Trait-first MAD kernel.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MadKernel;

#[cfg(feature = "alloc")]
impl KernelLifecycle for MadKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

#[cfg(feature = "alloc")]
impl<T> MadReduce1D<T> for MadKernel
where
    T: Float + Default + Sum,
{
    fn run<I>(&self, input: &I) -> Result<(T, usize), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(median_abs_deviation(input.iter()))
    }
}

/// Trait-first z-score kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZScoreKernel;

impl KernelLifecycle for ZScoreKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<T> ZScoreNormalize1D<T> for ZScoreKernel
where
    T: Float + Default + Copy + Add + Sum,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        let out = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out.len() != input.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: input.len(),
                got: out.len(),
            });
        }

        let mean = mean(input.iter()).0;
        let standard_deviation = stdev(input.iter()).0;
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, yi)| *out = (*yi - mean) / standard_deviation);
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(zscore(input.iter()).collect())
    }
}

/// Trait-first modified z-score kernel.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ModZScoreKernel;

#[cfg(feature = "alloc")]
impl KernelLifecycle for ModZScoreKernel {
    type Config = StatsConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

#[cfg(feature = "alloc")]
impl<T> ModZScoreNormalize1D<T> for ModZScoreKernel
where
    T: Float + Default + Copy + Add + Sum,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized,
    {
        let normalized = self.run_alloc(input)?;
        let out = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out.len() != normalized.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: normalized.len(),
                got: out.len(),
            });
        }
        out.copy_from_slice(&normalized);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(mod_zscore(input.iter()).collect())
    }
}

// Quick select finds the `i`th smallest element with 2N comparisons
#[cfg(feature = "alloc")]
fn quickselect<B, T>(y: &[B], k: usize) -> T
where
    B: Borrow<T>,
    T: Num + NumCast + PartialOrd + Copy,
{
    use num_traits::{Num, NumCast};

    let n = y.len();
    if n == 1 {
        return *y[0].borrow();
    }

    let pivot = y.get(n / 2).unwrap().borrow();
    let lower = y
        .iter()
        .filter(|yi| *(*yi).borrow() < *pivot)
        .map(|yi| *yi.borrow())
        .collect::<Vec<_>>();
    let lowers = lower.len();
    let upper = y
        .iter()
        .filter(|yi| *(*yi).borrow() > *pivot)
        .map(|yi| *yi.borrow())
        .collect::<Vec<_>>();
    let uppers = upper.len();
    let pivots = n - lowers - uppers;

    if k < lowers {
        quickselect(&lower, k)
    } else if k < lowers + pivots {
        *pivot
    } else {
        quickselect(&upper, k - lowers - pivots)
    }
}

///
/// Compute the median of the signal, `y`
///
/// Return the median and the number of points averaged
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::median;
///
/// let y: [f64; 5] = [1.,2.,3.,4.,5.];
/// assert_relative_eq!(3f64, median(y.iter()).0);
///
/// let y: [i32; 5] = [1,2,3,4,5];
/// assert_eq!(3, median(y.iter()).0);
///
/// let y: [f64; 4] = [1.,2.,3.,4.];
/// assert_relative_eq!(2.5f64, median(y.iter()).0);
///
/// let y: [f32; 5] = [3.,1.,4.,2.,5.];
/// assert_relative_eq!(3f32, median(y.iter()).0);
///
/// let y: [f64; 6] = [3.,1.,4.,2.,3.,5.];
/// assert_relative_eq!(3f64, median(y.iter()).0);
///
/// let y: &[f32] = &[];
/// assert_eq!((0f32, 0), median(y.iter()));
///
/// let y: &[f32] = &[1.];
/// assert_eq!((1f32, 1), median(y.iter()));
///
/// let y: [i64; 4] = [1,2,3,4];
/// assert_eq!(2i64, median(y.iter()).0);
///
/// ```
///
#[cfg(feature = "alloc")]
pub fn median<YI, T>(y: YI) -> (T, usize)
where
    T: Num + NumCast + PartialOrd + Copy + Default,
    YI: Iterator,
    YI::Item: Borrow<T>,
{
    // Materialize the values in the iterator in order to run O(n) quick select

    use num_traits::NumCast;
    let y = y.collect::<Vec<_>>();
    let n = y.len();

    if n == 0 {
        Default::default()
    } else if n == 1 {
        (*y[0].borrow(), 1)
    } else if n % 2 == 1 {
        (quickselect(&y, n / 2), n)
    } else {
        (
            (quickselect(&y, n / 2 - 1) + quickselect(&y, n / 2)) / T::from(2).unwrap(),
            n,
        )
    }
}

///
/// Compute the mean of the signal, `y`
///
/// Return the mean and the number of points averaged
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::mean;
///
/// let y: [f64; 5] = [1.,2.,3.,4.,5.];
/// assert_relative_eq!(3f64, mean(y.iter()).0);
///
/// let y: [i64; 5] = [1,2,3,4,5];
/// assert_eq!(3i64, mean(y.iter()).0);
///
/// let y: &[f32] = &[];
/// assert_eq!((0f32, 0), mean(y.iter()));
///
/// ```
///
pub fn mean<YI, F>(y: YI) -> (F, usize)
where
    F: Num + NumCast + Default + Copy + Add,
    YI: Iterator,
    YI::Item: Borrow<F>,
{
    let (sum, count) = y.fold(Default::default(), |acc: (F, usize), yi| {
        (acc.0 + *yi.borrow(), acc.1 + 1)
    });
    if count > 0 {
        (sum / F::from(count).unwrap(), count)
    } else {
        Default::default()
    }
}

///
/// Compute the variance of the signal, `y`
///
/// Return the variance and the number of points averaged
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::variance;
///
/// let y: [f64; 5] = [1.,2.,3.,4.,5.];
/// assert_relative_eq!(2f64, variance(y.iter()).0);
///
/// let y: &[f32] = &[];
/// assert_eq!((0f32, 0), variance(y.iter()));
///
/// ```
///
pub fn variance<YI, F>(y: YI) -> (F, usize)
where
    F: Float + Default + Sum,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    let (avg, n) = mean(y.clone());
    let sum: F = y
        .map(|f| {
            let delta = *f.borrow() - avg;
            delta * delta
        })
        .sum::<F>();
    if n > 0 {
        (sum / F::from(n).unwrap(), n)
    } else {
        Default::default()
    }
}

///
/// Compute the standard deviation of the signal, `y`
///
/// Return the standard deviation and the number of points averaged
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::stdev;
///
/// let y: [f64; 5] = [1.,2.,3.,4.,5.];
/// assert_relative_eq!(1.41421356237, stdev(y.iter()).0, max_relative = 1e-8);
///
/// let y: &[f32] = &[];
/// assert_eq!((0f32, 0), stdev(y.iter()));
///
/// ```
pub fn stdev<YI, F>(y: YI) -> (F, usize)
where
    F: Float + Default + Sum,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    match variance(y) {
        (_, 0) => Default::default(),
        (v, n) => (v.sqrt(), n),
    }
}

///
/// Autocorrelate the signal `y` with lag `k`,
/// using 1/N formulation
///
/// <https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm>
///
pub fn autocorr<YI, F>(y: YI, k: usize) -> F
where
    F: Float + Add + Sum + Default,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    let (avg, n) = mean(y.clone());
    let n = F::from(n).unwrap();
    let (var, _) = variance(y.clone());
    let autocovariance: F = y
        .clone()
        .zip(y.skip(k))
        .map(|(fi, fik)| (*fi.borrow() - avg) * (*fik.borrow() - avg))
        .sum::<F>()
        / n;
    autocovariance / var
}

///
/// Unscaled tiled autocorrelation of signal `y` with itself into `x`.
///
/// This skips variance normalization and only computes lags in `SKIP..SKIP+x.len()`
///
/// The autocorrelation is not normalized by 1/y.len() or variance. The variance of the signal
/// is returned. The returned variance may be used to normalize lags of interest after the fact.
///
/// <https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm>
///
pub fn autocorr_fast32<const N: usize, const M: usize, const SKIP: usize>(
    y: &mut [f32; N],
    x: &mut [f32; M],
) -> f32 {
    assert!(N >= M + SKIP);

    // Subtract the mean
    let sum = y.iter().sum::<f32>();
    let avg = sum / y.len() as f32;
    y.iter_mut().for_each(|yi| *yi -= avg);

    // Compute the variance of the signal
    let var = y.iter().map(|yi| yi * yi).sum::<f32>() / y.len() as f32;

    // Compute the autocorrelation for lag 1 to lag n
    let lag_skip = y.len() - x.len();
    for (h, xi) in (SKIP..y.len()).zip(x.iter_mut()) {
        let left = &y[..y.len() - h];
        let right = &y[h..];
        const TILE: usize = 4;
        let left = left.chunks_exact(TILE);
        let right = right.chunks_exact(TILE);
        *xi = left
            .remainder()
            .iter()
            .zip(right.remainder().iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        *xi = left
            .zip(right)
            .map(|(left, right)| {
                left.iter()
                    .zip(right.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .sum();
    }

    var
}

///
/// Root Mean Square (RMS) of signal `y`.
///
/// It is assumed that the mean of the signal is zero.
///
pub fn rms_fast32<const N: usize>(y: &[f32; N]) -> f32 {
    const TILE: usize = 4;
    let tiles = y.chunks_exact(TILE);
    let sum = tiles.remainder().iter().map(|yi| yi * yi).sum::<f32>()
        + tiles
            .map(|yi| yi.iter().map(|yi| yi * yi).sum::<f32>())
            .sum::<f32>();
    (sum / y.len() as f32).sqrt()
}

///
/// Produce an iterator yielding the lag difference, yi1 - yi0,
///
/// <https://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm>
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::lag_diff;
///
/// // Flat signal perfectly correlates with itself
/// let y: [f64; 4] = [1.,2.,4.,7.];
/// let z = lag_diff(y.iter()).collect::<Vec<_>>();
/// for i in 0..3 {
///     assert_relative_eq!(i as f64 + 1f64, z[i]);
/// }
/// ```
///
pub fn lag_diff<'a, YI, F>(y: YI) -> impl Iterator<Item = F>
where
    F: Float + 'a,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    y.clone()
        .zip(y.skip(1))
        .map(|(yi0, yi1)| *yi1.borrow() - *yi0.borrow())
}

///
/// Compute the root mean square of successive differences
///
/// ```
/// use approx::assert_relative_eq;
/// use sci_rs::stats::rmssd;
///
/// // Differences are 1, 2, 3
/// // Square differences are 1, 4, 9
/// // Mean is 4.666666666666667
/// // RMSSD is 2.1602468995
/// let y: [f64; 4] = [1.,2.,4.,7.];
/// assert_relative_eq!(2.1602468995, rmssd(y.iter()), max_relative = 1e-8);
/// ```
///
pub fn rmssd<YI, F>(y: YI) -> F
where
    F: Float + Add + Sum + Default,
    YI: Iterator + Clone,
    YI::Item: Borrow<F> + Copy,
{
    let square_diffs = y
        .tuple_windows()
        .map(|(yi0, yi1)| (*yi1.borrow() - *yi0.borrow()).powi(2));
    let (sum, n): (F, usize) = mean(square_diffs);
    sum.sqrt()
}

///
/// Compute the z score of each value in the sample, relative to the sample mean and standard deviation.
///
/// <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html>
///
/// # Arguments
///
/// * `y` - An array of floating point values
///
/// # Examples
///
/// ```
/// use sci_rs::stats::zscore;
/// use approx::assert_relative_eq;
///
/// let y: [f32; 5] = [1.,2.,3.,4.,5.];
/// let z : Vec<f32> = zscore(y.iter()).collect::<Vec<_>>();
/// let answer: [f32; 5] = [-1.4142135, -0.70710677, 0.,  0.70710677,  1.4142135];
/// for i in 0..5 {
///     assert_relative_eq!(answer[i], z[i], epsilon = 1e-6);
/// }
///
/// // Example from scipy docs
/// let a: [f32; 10] = [ 0.7972,  0.0767,  0.4383,  0.7866,  0.8091, 0.1954,  0.6307,  0.6599,  0.1065,  0.0508];
/// let z : Vec<f32> = zscore(a.iter()).collect::<Vec<_>>();
/// let answer: [f32; 10] =[ 1.12724554, -1.2469956 , -0.05542642,  1.09231569,  1.16645923, -0.8558472 ,  0.57858329,  0.67480514, -1.14879659, -1.33234306];
/// for i in 0..10 {
///    assert_relative_eq!(answer[i], z[i], epsilon = 1e-6);
/// }
/// ```
pub fn zscore<YI, F>(y: YI) -> impl Iterator<Item = F>
where
    F: Float + Default + Copy + Add + Sum,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    let mean = mean(y.clone()).0;
    let standard_deviation = stdev(y.clone()).0;
    y.map(move |yi| ((*yi.borrow() - mean) / standard_deviation))
}

///
/// Compute the modified Z-score of each value in the sample, relative to the sample median over the mean absolute deviation.
///
/// <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm>
///
/// # Arguments
///
/// * `y` - An array of floating point values
///
/// # Examples
///
/// ```
/// use sci_rs::stats::mod_zscore;
/// use approx::assert_relative_eq;
///
/// let y: [f32; 5] = [1.,2.,3.,4.,5.];
/// let z : Vec<f32> = mod_zscore(y.iter()).collect::<Vec<_>>();
/// let answer: [f32; 5] = [-1.349, -0.6745, 0.,  0.6745,  1.349];
/// for i in 0..5 {
///     assert_relative_eq!(answer[i], z[i], epsilon = 1e-5);
/// }
/// ```
pub fn mod_zscore<YI, F>(y: YI) -> impl Iterator<Item = F>
where
    F: Float + Default + Copy + Add + Sum,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    let median = median(y.clone()).0;

    let mad = median_abs_deviation(y.clone()).0;
    y.map(move |yi| ((*yi.borrow() - median) * F::from(0.6745).unwrap() / mad))
}

/// The median absolute deviation ([MAD](https://en.wikipedia.org/wiki/Median_absolute_deviation))
/// computes the median over the absolute deviations from the median. It is a measure of dispersion
/// similar to the standard deviation but more robust to outliers
///
/// # Arguments
///
/// * `y` - An array of floating point values
///
/// # Examples
///
/// ```
/// use sci_rs::stats::median_abs_deviation;
/// use approx::assert_relative_eq;
///
/// let y: [f64; 16] = [6., 7., 7., 8., 12., 14., 15., 16., 16., 19., 22., 24., 26., 26., 29., 46.];
/// let z = median_abs_deviation(y.iter());
///
/// assert_relative_eq!(8., z.0);
/// ```
pub fn median_abs_deviation<YI, F>(y: YI) -> (F, usize)
where
    F: Float + Default + Sum,
    YI: Iterator + Clone,
    YI::Item: Borrow<F>,
{
    let med = median(y.clone()).0;

    let abs_vals = y.map(|yi| (*yi.borrow() - med).abs());

    median(abs_vals.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelLifecycle;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[cfg(feature = "std")]
    use {std::f64::consts::PI, std::vec::Vec};

    #[cfg(feature = "std")]
    #[test]
    fn can_median() {
        let y: [f64; 4] = [1., 2., 3., 4.];
        println!("y = {:?}", y);
        println!("y = {:?}", median::<_, f64>(y.iter()));
        assert_relative_eq!(2.5, median::<_, f64>(y.iter()).0);
        let y: [f64; 5] = [1., 2., 3., 4., 5.];
        println!("y = {:?}", y);
        println!("y = {:?}", median::<_, f64>(y.iter()));
        assert_relative_eq!(3.0, median::<_, f64>(y.iter()).0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn can_autocorrelate() {
        // sin wave w/ multiple periods
        let periods = 1.;
        let points = 100;
        let radians_per_pt = (periods * 2. * PI) / points as f64;
        let sin_wave = (0..points)
            .map(|i| (i as f64 * radians_per_pt).sin())
            .collect::<Vec<_>>();
        // println!("sin_wave = {:?}", sin_wave);

        let _correlations: Vec<f64> = (0..points)
            .map(|i| autocorr(sin_wave.iter(), i))
            .collect::<Vec<_>>();
        let correlations: Vec<f32> = (0..points)
            .map(|i| autocorr(sin_wave.iter().map(|f| *f as f32), i))
            .collect::<Vec<_>>();
        println!("correlations = {:?}", correlations);
    }

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn mean_variance_stdev_kernels_match_reference() {
        let input = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mean_kernel = MeanKernel::try_new(StatsConfig).expect("mean kernel");
        let var_kernel = VarianceKernel::try_new(StatsConfig).expect("variance kernel");
        let stdev_kernel = StdevKernel::try_new(StatsConfig).expect("stdev kernel");

        let (m, n) = mean_kernel.run(&input).expect("mean run");
        let (v, _) = var_kernel.run(&input).expect("variance run");
        let (s, _) = stdev_kernel.run(&input).expect("stdev run");

        assert_eq!(n, input.len());
        assert_relative_eq!(m, 3.0, epsilon = 1e-12);
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
        assert_relative_eq!(s, 2.0f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn zscore_kernel_run_into_ndarray_output() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = ZScoreKernel::try_new(StatsConfig).expect("zscore kernel");
        let mut out = Array1::from(vec![0.0f32; input.len()]);
        kernel.run_into(&input, &mut out).expect("run_into");

        let expected: Vec<f32> = zscore(input.iter()).collect();
        out.iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_relative_eq!(*a, *b, epsilon = 1e-6));
    }

    #[test]
    fn zscore_kernel_validates_output_length() {
        let input = [1.0f32, 2.0, 3.0];
        let kernel = ZScoreKernel::try_new(StatsConfig).expect("zscore kernel");
        let mut out = [0.0f32; 2];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("length mismatch should fail");
        assert!(matches!(
            err,
            ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 3,
                got: 2
            }
        ));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn median_mad_and_modzscore_kernels_match_reference() {
        let input = [6.0f64, 7.0, 7.0, 8.0, 12.0, 14.0, 15.0, 16.0, 16.0, 19.0];
        let median_kernel = MedianKernel::try_new(StatsConfig).expect("median kernel");
        let mad_kernel = MadKernel::try_new(StatsConfig).expect("mad kernel");
        let modz_kernel = ModZScoreKernel::try_new(StatsConfig).expect("modz kernel");

        let (median_v, _) = median_kernel.run(&input).expect("median");
        let (mad_v, _) = mad_kernel.run(&input).expect("mad");
        let modz = modz_kernel.run_alloc(&input).expect("mod-zscore");
        let expected_modz: Vec<f64> = mod_zscore(input.iter()).collect();

        assert_relative_eq!(median_v, median(input.iter()).0, epsilon = 1e-12);
        assert_relative_eq!(mad_v, median_abs_deviation(input.iter()).0, epsilon = 1e-12);
        modz.iter()
            .zip(expected_modz.iter())
            .for_each(|(a, b)| assert_relative_eq!(*a, *b, epsilon = 1e-12));
    }
}
