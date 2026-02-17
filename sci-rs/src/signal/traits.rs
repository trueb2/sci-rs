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

/// 1D upsample-filter-downsample capability (`upfirdn`).
#[cfg(feature = "alloc")]
pub trait UpFirDn1D<T> {
    /// Run `upfirdn` into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run `upfirdn` and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D rational resampling capability (`resample_poly`).
#[cfg(feature = "alloc")]
pub trait ResamplePoly1D<T> {
    /// Run rational resampling into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run rational resampling and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// 1D integer downsampling capability (`decimate`).
#[cfg(feature = "alloc")]
pub trait Decimate1D<T> {
    /// Run decimation into a caller-provided output buffer.
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        O: Write1D<T> + ?Sized;

    /// Run decimation and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<T>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Relative-extrema detection capability.
#[cfg(feature = "alloc")]
pub trait ArgRelExtrema1D<T> {
    /// Compute extrema indices into a caller-provided vector.
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;

    /// Compute extrema indices and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Local peak detection capability.
#[cfg(feature = "alloc")]
pub trait FindPeaks1D<T> {
    /// Compute peak indices into a caller-provided vector.
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;

    /// Compute peak indices and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Peak prominence computation capability.
#[cfg(feature = "alloc")]
pub trait PeakProminence1D<T> {
    /// Output bundle type.
    type Output;

    /// Compute prominences into a caller-provided output bundle.
    fn run_into<I, P>(
        &self,
        input: &I,
        peaks: &P,
        out: &mut Self::Output,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        P: Read1D<usize> + ?Sized;

    /// Compute prominences and allocate output.
    fn run_alloc<I, P>(&self, input: &I, peaks: &P) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        P: Read1D<usize> + ?Sized;
}

/// Peak width computation capability.
#[cfg(feature = "alloc")]
pub trait PeakWidths1D<T> {
    /// Output bundle type.
    type Output;

    /// Compute widths into a caller-provided output bundle.
    fn run_into<I, P>(
        &self,
        input: &I,
        peaks: &P,
        out: &mut Self::Output,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        P: Read1D<usize> + ?Sized;

    /// Compute widths and allocate output.
    fn run_alloc<I, P>(&self, input: &I, peaks: &P) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
        P: Read1D<usize> + ?Sized;
}

/// Continuous wavelet transform capability.
#[cfg(feature = "alloc")]
pub trait Cwt1D<T> {
    /// Output bundle type.
    type Output;

    /// Compute CWT into a caller-provided output bundle.
    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;

    /// Compute CWT and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Wavelet-guided peak detection capability.
#[cfg(feature = "alloc")]
pub trait FindPeaksCwt1D<T> {
    /// Compute peak indices into a caller-provided vector.
    fn run_into<I>(&self, input: &I, out: &mut Vec<usize>) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;

    /// Compute peak indices and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<usize>, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Periodogram capability.
#[cfg(feature = "std")]
pub trait Periodogram1D {
    /// Compute one-sided PSD into caller-provided output buffers.
    fn run_into<I, OF, OP>(
        &self,
        input: &I,
        freqs: &mut OF,
        pxx: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<f64> + ?Sized;

    /// Compute one-sided PSD and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;
}

/// Welch PSD capability.
#[cfg(feature = "std")]
pub trait WelchPsd1D {
    /// Compute Welch PSD into caller-provided output buffers.
    fn run_into<I, OF, OP>(
        &self,
        input: &I,
        freqs: &mut OF,
        pxx: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<f64> + ?Sized;

    /// Compute Welch PSD and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;
}

/// Cross power spectral density capability.
#[cfg(feature = "std")]
pub trait Csd1D {
    /// Compute cross PSD into caller-provided output buffers.
    fn run_into<I1, I2, OF, OP>(
        &self,
        x: &I1,
        y: &I2,
        freqs: &mut OF,
        pxy: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<Complex<f64>> + ?Sized;

    /// Compute cross PSD and allocate output.
    fn run_alloc<I1, I2>(
        &self,
        x: &I1,
        y: &I2,
    ) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized;
}

/// Magnitude-squared coherence capability.
#[cfg(feature = "std")]
pub trait Coherence1D {
    /// Compute coherence into caller-provided output buffers.
    fn run_into<I1, I2, OF, OC>(
        &self,
        x: &I1,
        y: &I2,
        freqs: &mut OF,
        coherence: &mut OC,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OC: Write1D<f64> + ?Sized;

    /// Compute coherence and allocate output.
    fn run_alloc<I1, I2>(
        &self,
        x: &I1,
        y: &I2,
    ) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized;
}

/// STFT capability.
#[cfg(feature = "std")]
pub trait Stft1D {
    /// Output bundle type.
    type Output;

    /// Compute STFT into a caller-provided output bundle.
    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;

    /// Compute STFT and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;
}

/// Inverse STFT capability.
#[cfg(feature = "std")]
pub trait Istft1D {
    /// Compute inverse STFT into caller-provided output buffers.
    fn run_into<O1, O2>(
        &self,
        zxx: &[Vec<Complex<f64>>],
        t: &mut O1,
        y: &mut O2,
    ) -> Result<(), ExecInvariantViolation>
    where
        O1: Write1D<f64> + ?Sized,
        O2: Write1D<f64> + ?Sized;

    /// Compute inverse STFT and allocate output.
    fn run_alloc(
        &self,
        zxx: &[Vec<Complex<f64>>],
    ) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>;
}

/// Spectrogram capability.
#[cfg(feature = "std")]
pub trait Spectrogram1D {
    /// Output bundle type.
    type Output;

    /// Compute spectrogram into a caller-provided output bundle.
    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;

    /// Compute spectrogram and allocate output.
    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized;
}

/// Digital filter frequency response capability.
#[cfg(feature = "std")]
pub trait Freqz1D {
    /// Compute digital frequency response into caller-provided output buffers.
    fn run_into<I1, I2, OW, OH>(
        &self,
        b: &I1,
        a: &I2,
        w: &mut OW,
        h: &mut OH,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OW: Write1D<f64> + ?Sized,
        OH: Write1D<Complex<f64>> + ?Sized;

    /// Compute digital frequency response and allocate output.
    fn run_alloc<I1, I2>(
        &self,
        b: &I1,
        a: &I2,
    ) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized;
}

/// SOS cascade frequency response capability.
#[cfg(feature = "std")]
pub trait SosFreqz1D {
    /// Compute SOS frequency response into caller-provided output buffers.
    fn run_into<I, OW, OH>(
        &self,
        sos: &I,
        w: &mut OW,
        h: &mut OH,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<crate::signal::filter::design::Sos<f64>> + ?Sized,
        OW: Write1D<f64> + ?Sized,
        OH: Write1D<Complex<f64>> + ?Sized;

    /// Compute SOS frequency response and allocate output.
    fn run_alloc<I>(
        &self,
        sos: &I,
    ) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I: Read1D<crate::signal::filter::design::Sos<f64>> + ?Sized;
}
