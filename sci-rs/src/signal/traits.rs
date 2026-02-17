//! Trait interfaces for signal-processing capabilities.
//!
//! These traits define the trait-first API shape used by refactored kernels.

use crate::kernel::{ExecInvariantViolation, Read1D, Write1D};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use nalgebra::Complex;

/// Output tuple for splitting roots into conjugate-paired complex and real sets.
#[cfg(feature = "alloc")]
pub type ComplexSplit<T> = (Vec<Complex<T>>, Vec<Complex<T>>);

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

/// 1D sawtooth-wave generation capability.
pub trait SawtoothWave1D<T> {
    /// Generate a sawtooth wave from phase/time input into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Generate a sawtooth wave and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D chirp-wave generation capability.
pub trait ChirpWave1D<T> {
    /// Generate a chirp wave from phase/time input into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Generate a chirp wave and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D unit-impulse generation capability.
pub trait UnitImpulse1D<T> {
    /// Generate a unit impulse into a caller-provided output buffer.
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized;

    /// Generate a unit impulse and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation>;
}

/// 1D Gaussian-pulse generation capability.
pub trait GaussPulseWave1D<T> {
    /// Generate a Gaussian-modulated sinusoid into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Generate a Gaussian-modulated sinusoid and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D polynomial-frequency sweep generation capability.
pub trait SweepPolyWave1D<T> {
    /// Generate a polynomial sweep into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Generate a polynomial sweep and allocate output.
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

/// `lfilter_zi` design capability.
pub trait LFilterZiDesign1D<T> {
    /// Compute initial state into a caller-provided output buffer.
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized;

    /// Compute initial state and allocate output.
    #[cfg(feature = "alloc")]
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation>;
}

/// `sosfilt_zi` design capability.
#[cfg(feature = "alloc")]
pub trait SosFiltZiDesign1D<T>
where
    T: nalgebra::RealField + Copy,
{
    /// Compute SOS initial states and allocate output sections.
    fn run_alloc(
        &self,
    ) -> Result<Vec<crate::signal::filter::design::Sos<T>>, ExecInvariantViolation>;
}

/// `sosfilt_zi` design capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait SosFiltZiDesign1D<T> {}

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

/// 1D Savitzky-Golay coefficient design capability.
#[cfg(feature = "alloc")]
pub trait SavgolCoeffsDesign<T> {
    /// Design coefficients into a caller-provided output buffer.
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized;

    /// Design coefficients and allocate output.
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation>;
}

/// 1D Savitzky-Golay coefficient design capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait SavgolCoeffsDesign<T> {}

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

/// ZPK-domain transform capability.
#[cfg(feature = "alloc")]
pub trait ZpkTransform<T>
where
    T: nalgebra::RealField + Copy,
{
    /// Run a ZPK-domain transform and allocate the transformed representation.
    fn run_alloc(
        &self,
        zpk: crate::signal::filter::design::ZpkFormatFilter<T>,
    ) -> Result<crate::signal::filter::design::ZpkFormatFilter<T>, ExecInvariantViolation>;
}

/// ZPK-domain transform capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait ZpkTransform<T> {}

/// ZPK to transfer-function conversion capability.
#[cfg(feature = "alloc")]
pub trait ZpkToTfDesign<T>
where
    T: nalgebra::RealField,
{
    /// Convert poles/zeros/gain into transfer-function coefficients.
    fn run_alloc(
        &self,
        zeros: &[Complex<T>],
        poles: &[Complex<T>],
        gain: T,
    ) -> Result<crate::signal::filter::design::BaFormatFilter<T>, ExecInvariantViolation>;
}

/// ZPK to transfer-function conversion capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait ZpkToTfDesign<T> {}

/// ZPK to SOS conversion capability.
#[cfg(feature = "alloc")]
pub trait ZpkToSosDesign<T>
where
    T: nalgebra::RealField + Copy,
{
    /// Convert poles/zeros/gain into second-order sections.
    fn run_alloc(
        &self,
        zpk: crate::signal::filter::design::ZpkFormatFilter<T>,
    ) -> Result<crate::signal::filter::design::SosFormatFilter<T>, ExecInvariantViolation>;
}

/// ZPK to SOS conversion capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait ZpkToSosDesign<T> {}

/// Relative degree computation capability.
#[cfg(feature = "alloc")]
pub trait RelativeDegreeDesign<T> {
    /// Compute `len(poles) - len(zeros)` after validation.
    fn run_alloc(
        &self,
        zeros: &[Complex<T>],
        poles: &[Complex<T>],
    ) -> Result<usize, ExecInvariantViolation>;
}

/// Relative degree computation capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait RelativeDegreeDesign<T> {}

/// Complex root pairing/splitting capability.
#[cfg(feature = "alloc")]
pub trait ComplexPairSplit<T> {
    /// Split roots into conjugate-paired complex roots and real roots.
    fn run_alloc(&self, roots: Vec<Complex<T>>) -> Result<ComplexSplit<T>, ExecInvariantViolation>;
}

/// Complex root pairing/splitting capability in no-alloc mode.
#[cfg(not(feature = "alloc"))]
pub trait ComplexPairSplit<T> {}

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
