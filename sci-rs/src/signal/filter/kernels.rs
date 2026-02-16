//! Trait-first kernel wrappers for filtering primitives.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::{FiltFilt1D, LFilter1D, SosFilt1D, SosFiltFilt1D};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::filter::{design::Sos, sosfilt_dyn};

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
}
