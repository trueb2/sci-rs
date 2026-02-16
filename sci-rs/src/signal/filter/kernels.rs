//! Trait-first kernel wrappers for filtering primitives.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::{FiltFilt1D, LFilter1D, SavgolFilter1D, SosFilt1D, SosFiltFilt1D};
use alloc::vec::Vec;
use core::iter::Sum;
use core::ops::{Add, Sub};
use nalgebra::RealField;
use ndarray::{Array1, ArrayView1};
use num_traits::{FromPrimitive, NumAssign, One};

use super::design::Sos;
use super::{FiltFilt, FiltFiltPad, LFilter};

/// Constructor config for [`SosFiltKernel`].
#[derive(Debug, Clone)]
pub struct SosFiltConfig<F>
where
    F: RealField + Copy,
{
    /// Second-order sections with mutable filter state.
    pub sos: Vec<Sos<F>>,
}

/// Stateful 1D `sosfilt` kernel.
#[derive(Debug, Clone)]
pub struct SosFiltKernel<F>
where
    F: RealField + Copy,
{
    sos: Vec<Sos<F>>,
}

impl<F> KernelLifecycle for SosFiltKernel<F>
where
    F: RealField + Copy,
{
    type Config = SosFiltConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.sos.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "sos" });
        }
        Ok(Self { sos: config.sos })
    }
}

impl<F> SosFilt1D<F> for SosFiltKernel<F>
where
    F: RealField + Copy + Sum,
{
    fn run_into<I, O>(&mut self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != input.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: input.len(),
                got: out_slice.len(),
            });
        }
        let y = super::sosfilt_dyn(input.iter(), &mut self.sos);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&mut self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(super::sosfilt_dyn(input.iter(), &mut self.sos))
    }
}

/// Constructor config for [`SosFiltFiltKernel`].
#[derive(Debug, Clone)]
pub struct SosFiltFiltConfig<F>
where
    F: RealField + Copy,
{
    /// Second-order sections used for forward-backward filtering.
    pub sos: Vec<Sos<F>>,
}

/// Stateless 1D `sosfiltfilt` kernel.
#[derive(Debug, Clone)]
pub struct SosFiltFiltKernel<F>
where
    F: RealField + Copy,
{
    sos: Vec<Sos<F>>,
}

impl<F> KernelLifecycle for SosFiltFiltKernel<F>
where
    F: RealField + Copy,
{
    type Config = SosFiltFiltConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.sos.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "sos" });
        }
        Ok(Self { sos: config.sos })
    }
}

impl<F> SosFiltFilt1D<F> for SosFiltFiltKernel<F>
where
    F: RealField + Copy + Sum,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != input.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: input.len(),
                got: out_slice.len(),
            });
        }
        let y = super::sosfiltfilt_dyn(input.iter(), &self.sos);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(super::sosfiltfilt_dyn(input.iter(), &self.sos))
    }
}

/// Constructor config for [`LFilterKernel`].
#[derive(Debug, Clone)]
pub struct LFilterConfig<T> {
    /// Numerator coefficients.
    pub b: Vec<T>,
    /// Denominator coefficients.
    pub a: Vec<T>,
    /// Optional axis for ndarray-backed execution.
    pub axis: Option<isize>,
}

/// 1D `lfilter` kernel wrapper over ndarray-backed implementation.
#[derive(Debug, Clone)]
pub struct LFilterKernel<T> {
    b: Vec<T>,
    a: Vec<T>,
    axis: Option<isize>,
}

impl<T> KernelLifecycle for LFilterKernel<T>
where
    T: NumAssign + FromPrimitive + Copy,
{
    type Config = LFilterConfig<T>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.b.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "b" });
        }
        if config.a.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "a" });
        }
        Ok(Self {
            b: config.b,
            a: config.a,
            axis: config.axis,
        })
    }
}

impl<T> LFilter1D<T> for LFilterKernel<T>
where
    T: NumAssign + FromPrimitive + Copy,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized,
    {
        let y = self.run_alloc(input)?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != y.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: y.len(),
                got: out_slice.len(),
            });
        }
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        let b = Array1::from_vec(self.b.clone());
        let a = Array1::from_vec(self.a.clone());
        let x = Array1::from_vec(input.to_vec());
        let (y, _) = Array1::lfilter(
            ArrayView1::from(b.as_slice().unwrap_or_default()),
            ArrayView1::from(a.as_slice().unwrap_or_default()),
            x,
            self.axis,
            None,
        )
        .map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "lfilter kernel execution failed",
        })?;
        Ok(y.to_vec())
    }
}

/// Constructor config for [`FiltFiltKernel`].
#[derive(Debug, Clone)]
pub struct FiltFiltConfig<T> {
    /// Numerator coefficients.
    pub b: Vec<T>,
    /// Denominator coefficients.
    pub a: Vec<T>,
    /// Optional axis for ndarray-backed execution.
    pub axis: Option<isize>,
    /// Optional padding policy.
    pub padding: Option<FiltFiltPad>,
}

/// 1D `filtfilt` kernel wrapper over ndarray-backed implementation.
#[derive(Debug, Clone)]
pub struct FiltFiltKernel<T> {
    b: Vec<T>,
    a: Vec<T>,
    axis: Option<isize>,
    padding: Option<FiltFiltPad>,
}

impl<T> KernelLifecycle for FiltFiltKernel<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + One + RealField + Copy + Sum,
{
    type Config = FiltFiltConfig<T>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.b.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "b" });
        }
        if config.a.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "a" });
        }
        Ok(Self {
            b: config.b,
            a: config.a,
            axis: config.axis,
            padding: config.padding,
        })
    }
}

impl<T> FiltFilt1D<T> for FiltFiltKernel<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + One + RealField + Copy + Sum,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized,
    {
        let y = self.run_alloc(input)?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != y.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: y.len(),
                got: out_slice.len(),
            });
        }
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        let b = Array1::from_vec(self.b.clone());
        let a = Array1::from_vec(self.a.clone());
        let x = Array1::from_vec(input.to_vec());
        let y = Array1::filtfilt(
            ArrayView1::from(b.as_slice().unwrap_or_default()),
            ArrayView1::from(a.as_slice().unwrap_or_default()),
            x,
            self.axis,
            self.padding,
        )
        .map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "filtfilt kernel execution failed",
        })?;
        Ok(y.to_vec())
    }
}

/// Constructor config for [`SavgolFilterKernel`].
#[derive(Debug, Clone)]
pub struct SavgolFilterConfig<T> {
    /// Odd window length.
    pub window_length: usize,
    /// Polynomial order.
    pub polyorder: usize,
    /// Derivative order.
    pub deriv: Option<usize>,
    /// Sample spacing.
    pub delta: Option<T>,
}

/// Trait-first Savitzky-Golay filtering kernel.
#[derive(Debug, Clone)]
pub struct SavgolFilterKernel<T> {
    window_length: usize,
    polyorder: usize,
    deriv: Option<usize>,
    delta: Option<T>,
}

impl<T> KernelLifecycle for SavgolFilterKernel<T>
where
    T: RealField + Copy + Sum,
{
    type Config = SavgolFilterConfig<T>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.window_length == 0 || config.window_length.is_multiple_of(2) {
            return Err(ConfigError::InvalidArgument {
                arg: "window_length",
                reason: "window_length must be odd and greater than zero",
            });
        }
        if config.polyorder >= config.window_length {
            return Err(ConfigError::InvalidArgument {
                arg: "polyorder",
                reason: "polyorder must be less than window_length",
            });
        }
        if config.window_length < config.polyorder + 2 {
            return Err(ConfigError::InvalidArgument {
                arg: "window_length/polyorder",
                reason: "window_length is too small for the polynomial order",
            });
        }

        Ok(Self {
            window_length: config.window_length,
            polyorder: config.polyorder,
            deriv: config.deriv,
            delta: config.delta,
        })
    }
}

impl<T> SavgolFilter1D<T> for SavgolFilterKernel<T>
where
    T: RealField + Copy + Sum,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized,
    {
        let filtered = self.run_alloc(input)?;
        let out = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out.len() != filtered.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: filtered.len(),
                got: out.len(),
            });
        }
        out.copy_from_slice(&filtered);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(super::savgol_filter_dyn(
            input.iter(),
            self.window_length,
            self.polyorder,
            self.deriv,
            self.delta,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ConfigError;
    use crate::signal::filter::{
        design::Sos, savgol_filter_dyn, sosfilt_dyn, sosfiltfilt_dyn, FiltFiltPad,
    };
    use ndarray::Array1;

    #[test]
    fn sosfilt_kernel_matches_function() {
        let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
            sos: Sos::from_scipy_dyn(1, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        })
        .expect("kernel should initialize");
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 4];
        kernel
            .run_into(&x, &mut y)
            .expect("sosfilt kernel should run");

        let mut sos = Sos::from_scipy_dyn(1, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let expected = sosfilt_dyn(x.iter(), &mut sos);
        assert_eq!(y, expected.as_slice());
    }

    #[test]
    fn sosfiltfilt_kernel_matches_function() {
        let kernel = SosFiltFiltKernel::try_new(SosFiltFiltConfig {
            sos: Sos::from_scipy_dyn(1, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        })
        .expect("kernel should initialize");
        let x: Vec<f64> = (0..64).map(|i| i as f64).collect();

        let actual = kernel.run_alloc(&x).expect("sosfiltfilt should run");
        let sos = Sos::from_scipy_dyn(1, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let expected = sosfiltfilt_dyn(x.iter(), &sos);
        assert_eq!(actual, expected);
    }

    #[test]
    fn lfilter_kernel_matches_lfilter_reference_and_length_checks() {
        let kernel = LFilterKernel::try_new(LFilterConfig {
            b: vec![0.5f64, 0.5],
            a: vec![1.0f64],
            axis: None,
        })
        .expect("kernel should initialize");

        let input = [1.0f64, 2.0, 3.0, 4.0];
        let actual = kernel.run_alloc(&input).expect("lfilter should run");

        let b = Array1::from(vec![0.5f64, 0.5]);
        let a = Array1::from(vec![1.0f64]);
        let x = Array1::from(input.to_vec());
        let (expected, _) = Array1::lfilter(b.view(), a.view(), x, None, None)
            .expect("reference lfilter should run");
        assert_eq!(actual, expected.to_vec());

        let mut too_short = vec![0.0f64; input.len() - 1];
        let err = kernel
            .run_into(&input, &mut too_short)
            .expect_err("output size mismatch must fail");
        assert!(matches!(
            err,
            ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn filtfilt_kernel_matches_reference_and_validates_lengths() {
        let kernel = FiltFiltKernel::try_new(FiltFiltConfig {
            b: vec![0.5f64, 0.5],
            a: vec![1.0f64],
            axis: None,
            padding: Some(FiltFiltPad::default()),
        })
        .expect("kernel should initialize");

        let input = vec![
            0.0f64, 0.6389613, 0.890577, 0.9830277, 0.9992535, 0.9756868, 0.9304659, 0.8734051,
        ];
        let actual = kernel.run_alloc(&input).expect("filtfilt should run");

        let b = Array1::from(vec![0.5f64, 0.5]);
        let a = Array1::from(vec![1.0f64]);
        let x = Array1::from(input.clone());
        let expected = Array1::filtfilt(b.view(), a.view(), x, None, Some(FiltFiltPad::default()))
            .expect("reference filtfilt should run");
        assert_eq!(actual, expected.to_vec());

        let mut too_short = vec![0.0f64; input.len() - 1];
        let err = kernel
            .run_into(&input, &mut too_short)
            .expect_err("output size mismatch must fail");
        assert!(matches!(
            err,
            ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 8,
                got: 7
            }
        ));
    }

    #[test]
    fn constructors_reject_empty_coefficients() {
        let err = SosFiltKernel::<f64>::try_new(SosFiltConfig { sos: Vec::new() })
            .expect_err("empty sos must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "sos" });

        let err = SosFiltFiltKernel::<f64>::try_new(SosFiltFiltConfig { sos: Vec::new() })
            .expect_err("empty sos must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "sos" });

        let err = LFilterKernel::<f64>::try_new(LFilterConfig {
            b: Vec::new(),
            a: vec![1.0],
            axis: None,
        })
        .expect_err("empty b must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "b" });

        let err = FiltFiltKernel::<f64>::try_new(FiltFiltConfig {
            b: vec![1.0],
            a: Vec::new(),
            axis: None,
            padding: None,
        })
        .expect_err("empty a must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "a" });
    }

    #[test]
    fn savgol_kernel_matches_reference_and_validates_lengths() {
        let kernel = SavgolFilterKernel::try_new(SavgolFilterConfig {
            window_length: 5,
            polyorder: 2,
            deriv: None,
            delta: None::<f64>,
        })
        .expect("kernel should initialize");
        let input = [2.0f64, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];

        let actual = kernel.run_alloc(&input).expect("savgol should run");
        let expected = savgol_filter_dyn(input.iter(), 5, 2, None, None);
        assert_eq!(actual, expected);

        let mut too_short = vec![0.0f64; input.len() - 1];
        let err = kernel
            .run_into(&input, &mut too_short)
            .expect_err("output size mismatch should fail");
        assert!(matches!(
            err,
            ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 9,
                got: 8
            }
        ));
    }

    #[test]
    fn savgol_kernel_constructor_validation() {
        let err = SavgolFilterKernel::<f64>::try_new(SavgolFilterConfig {
            window_length: 4,
            polyorder: 2,
            deriv: None,
            delta: None,
        })
        .expect_err("even window must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "window_length",
                reason: "window_length must be odd and greater than zero",
            }
        );
    }
}
