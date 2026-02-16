//! Trait interfaces for signal-processing capabilities.
//!
//! These traits define the trait-first API shape used by refactored kernels.

use crate::kernel::{ExecInvariantViolation, Read1D, Write1D};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// 1D convolution capability.
pub trait Convolve1D<T> {
    /// Run convolution into a caller-provided output buffer.
    fn run_into<I1, I2, O>(
        &self,
        in1: &I1,
        in2: &I2,
        out: &mut O,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<T> + ?Sized,
        I2: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run convolution and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I1, I2>(&self, in1: &I1, in2: &I2) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I1: Read1D<T> + ?Sized,
        I2: Read1D<T> + ?Sized;
}

/// 1D correlation capability.
pub trait Correlate1D<T> {
    /// Run correlation into a caller-provided output buffer.
    fn run_into<I1, I2, O>(
        &self,
        in1: &I1,
        in2: &I2,
        out: &mut O,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<T> + ?Sized,
        I2: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run correlation and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I1, I2>(&self, in1: &I1, in2: &I2) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I1: Read1D<T> + ?Sized,
        I2: Read1D<T> + ?Sized;
}

/// 1D resampling capability.
pub trait Resample1D<T> {
    /// Run resampling into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run resampling and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D square-wave generation capability.
pub trait SquareWave1D<T> {
    /// Generate a square wave from phase/time input into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Generate a square wave and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D `lfilter` capability.
pub trait LFilter1D<T> {
    /// Run filtering into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run filtering and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D `filtfilt` capability.
pub trait FiltFilt1D<T> {
    /// Run zero-phase filtering into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run zero-phase filtering and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D Savitzky-Golay filtering capability.
pub trait SavgolFilter1D<T> {
    /// Run Savitzky-Golay filtering into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run Savitzky-Golay filtering and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D `sosfilt` capability.
pub trait SosFilt1D<T> {
    /// Run second-order-sections filtering into a caller-provided output buffer.
    fn run_into<I, O>(&mut self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run second-order-sections filtering and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&mut self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D `sosfiltfilt` capability.
pub trait SosFiltFilt1D<T> {
    /// Run zero-phase SOS filtering into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run zero-phase SOS filtering and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// FIR design capability.
#[cfg(feature = "alloc")]
pub trait FirWinDesign<T> {
    /// Run FIR design into a caller-provided output buffer.
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized;

    /// Run FIR design and allocate output coefficients.
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation>;
}

/// FIR design capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait FirWinDesign<T> {}

/// IIR design capability.
#[cfg(feature = "alloc")]
pub trait IirDesign<T> {
    /// Output representation produced by the design kernel.
    type Output;

    /// Run IIR design and allocate output representation.
    fn run_alloc(&self) -> Result<Self::Output, ExecInvariantViolation>;
}

/// IIR design capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait IirDesign<T> {}

/// Window generation capability.
#[cfg(feature = "alloc")]
pub trait WindowGenerate<T> {
    /// Run window generation into a caller-provided output buffer.
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized;

    /// Run window generation and allocate output samples.
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation>;
}

/// Window generation capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait WindowGenerate<T> {}
