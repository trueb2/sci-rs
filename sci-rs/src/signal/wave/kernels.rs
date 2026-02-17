//! Trait-first kernels for waveform generation.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::{
    ChirpWave1D, GaussPulseWave1D, SawtoothWave1D, SquareWave1D, SweepPolyWave1D, UnitImpulse1D,
};
use nalgebra::RealField;
use num_traits::{FromPrimitive, One, Zero};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

fn wrap_phase<F>(t: F) -> F
where
    F: RealField + Copy,
{
    let two_pi = F::two_pi();
    let mut x = t % two_pi;
    if x < F::zero() {
        x += two_pi;
    }
    x
}

/// Constructor config for [`SquareWaveKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareWaveConfig<F>
where
    F: RealField + Copy,
{
    /// Duty cycle in the interval `[0, 1]`.
    pub duty: F,
}

/// Trait-first 1D square-wave generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareWaveKernel<F>
where
    F: RealField + Copy,
{
    duty: F,
}

impl<F> SquareWaveKernel<F>
where
    F: RealField + Copy,
{
    /// Return configured duty cycle.
    pub fn duty(&self) -> F {
        self.duty
    }

    pub(super) fn sample(&self, t: F) -> F {
        let duty_threshold = F::two_pi() * self.duty;
        let x = wrap_phase(t);
        if x < duty_threshold {
            F::one()
        } else {
            -F::one()
        }
    }
}

impl<F> KernelLifecycle for SquareWaveKernel<F>
where
    F: RealField + Copy,
{
    type Config = SquareWaveConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.duty < F::zero() || config.duty > F::one() {
            return Err(ConfigError::InvalidArgument {
                arg: "duty",
                reason: "duty must be in [0, 1]",
            });
        }
        Ok(Self { duty: config.duty })
    }
}

impl<F> SquareWave1D<F> for SquareWaveKernel<F>
where
    F: RealField + Copy,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
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
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, t)| *out = self.sample(*t));
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(input.iter().map(|t| self.sample(*t)).collect())
    }
}

/// Constructor config for [`SawtoothWaveKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SawtoothWaveConfig<F>
where
    F: RealField + Copy,
{
    /// Width of the rising ramp in `[0, 1]`.
    pub width: F,
}

/// Trait-first 1D sawtooth-wave generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SawtoothWaveKernel<F>
where
    F: RealField + Copy,
{
    width: F,
}

impl<F> SawtoothWaveKernel<F>
where
    F: RealField + Copy,
{
    /// Return configured width.
    pub fn width(&self) -> F {
        self.width
    }

    pub(super) fn sample(&self, t: F) -> F {
        let x = wrap_phase(t);
        let pi = F::pi();

        if self.width == F::zero() {
            return F::one() - x / pi;
        }
        if self.width == F::one() {
            return x / pi - F::one();
        }

        let threshold = F::two_pi() * self.width;
        if x < threshold {
            x / (pi * self.width) - F::one()
        } else {
            (pi * (self.width + F::one()) - x) / (pi * (F::one() - self.width))
        }
    }
}

impl<F> KernelLifecycle for SawtoothWaveKernel<F>
where
    F: RealField + Copy,
{
    type Config = SawtoothWaveConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.width < F::zero() || config.width > F::one() {
            return Err(ConfigError::InvalidArgument {
                arg: "width",
                reason: "width must be in [0, 1]",
            });
        }
        Ok(Self {
            width: config.width,
        })
    }
}

impl<F> SawtoothWave1D<F> for SawtoothWaveKernel<F>
where
    F: RealField + Copy,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
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
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, t)| *out = self.sample(*t));
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(input.iter().map(|t| self.sample(*t)).collect())
    }
}

/// Chirp sweep method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChirpMethod {
    /// Linear frequency sweep.
    Linear,
    /// Quadratic frequency sweep.
    Quadratic,
    /// Logarithmic frequency sweep.
    Logarithmic,
    /// Hyperbolic frequency sweep.
    Hyperbolic,
}

/// Constructor config for [`ChirpKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChirpConfig<F>
where
    F: RealField + Copy,
{
    /// Frequency at `t=0`.
    pub f0: F,
    /// Time at which `f1` is specified.
    pub t1: F,
    /// Frequency at `t=t1`.
    pub f1: F,
    /// Sweep method.
    pub method: ChirpMethod,
    /// Phase offset in degrees.
    pub phi_deg: F,
    /// When `method` is quadratic, choose parabola vertex at `t=0` if true and `t=t1` if false.
    pub vertex_zero: bool,
}

/// Trait-first 1D chirp-wave generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChirpKernel<F>
where
    F: RealField + Copy,
{
    f0: F,
    t1: F,
    f1: F,
    method: ChirpMethod,
    phi_rad: F,
    vertex_zero: bool,
}

impl<F> ChirpKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    /// Return configured chirp method.
    pub fn method(&self) -> ChirpMethod {
        self.method
    }

    fn phase_at(&self, t: F) -> F {
        let two_pi = F::two_pi();
        let half = F::one() / (F::one() + F::one());
        let third = F::one() / (F::one() + F::one() + F::one());

        match self.method {
            ChirpMethod::Linear => {
                let beta = (self.f1 - self.f0) / self.t1;
                two_pi * (self.f0 * t + half * beta * t * t)
            }
            ChirpMethod::Quadratic => {
                let beta = (self.f1 - self.f0) / (self.t1 * self.t1);
                if self.vertex_zero {
                    two_pi * (self.f0 * t + beta * t * t * t * third)
                } else {
                    let t1_minus_t = self.t1 - t;
                    two_pi
                        * (self.f1 * t
                            + beta
                                * ((t1_minus_t * t1_minus_t * t1_minus_t)
                                    - self.t1 * self.t1 * self.t1)
                                * third)
                }
            }
            ChirpMethod::Logarithmic => {
                if self.f0 == self.f1 {
                    two_pi * self.f0 * t
                } else {
                    let ratio = self.f1 / self.f0;
                    let beta = self.t1 / ratio.ln();
                    two_pi * beta * self.f0 * (ratio.powf(t / self.t1) - F::one())
                }
            }
            ChirpMethod::Hyperbolic => {
                if self.f0 == self.f1 {
                    two_pi * self.f0 * t
                } else {
                    let sing = -self.f1 * self.t1 / (self.f0 - self.f1);
                    two_pi * (-sing * self.f0) * (F::one() - t / sing).abs().ln()
                }
            }
        }
    }

    pub(super) fn sample(&self, t: F) -> F {
        (self.phase_at(t) + self.phi_rad).cos()
    }
}

impl<F> KernelLifecycle for ChirpKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    type Config = ChirpConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.t1 == F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "t1",
                reason: "t1 must be nonzero",
            });
        }

        match config.method {
            ChirpMethod::Logarithmic => {
                if config.f0 * config.f1 <= F::zero() {
                    return Err(ConfigError::InvalidArgument {
                        arg: "f0/f1",
                        reason: "logarithmic chirp requires nonzero f0/f1 with same sign",
                    });
                }
            }
            ChirpMethod::Hyperbolic => {
                if config.f0 == F::zero() || config.f1 == F::zero() {
                    return Err(ConfigError::InvalidArgument {
                        arg: "f0/f1",
                        reason: "hyperbolic chirp requires nonzero f0 and f1",
                    });
                }
            }
            ChirpMethod::Linear | ChirpMethod::Quadratic => {}
        }

        let deg = F::from_u8(180).ok_or(ConfigError::InvalidArgument {
            arg: "phi_deg",
            reason: "phi conversion failed",
        })?;
        Ok(Self {
            f0: config.f0,
            t1: config.t1,
            f1: config.f1,
            method: config.method,
            phi_rad: config.phi_deg * F::pi() / deg,
            vertex_zero: config.vertex_zero,
        })
    }
}

impl<F> ChirpWave1D<F> for ChirpKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
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
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, t)| *out = self.sample(*t));
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(input.iter().map(|t| self.sample(*t)).collect())
    }
}

/// Constructor config for [`GaussPulseKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussPulseConfig<F>
where
    F: RealField + Copy,
{
    /// Center frequency.
    pub fc: F,
    /// Fractional bandwidth.
    pub bw: F,
    /// Reference dB level for bandwidth.
    pub bwr: F,
}

/// Trait-first 1D Gaussian-modulated sinusoid generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussPulseKernel<F>
where
    F: RealField + Copy,
{
    fc: F,
    a: F,
}

impl<F> GaussPulseKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    fn envelope(&self, t: F) -> F {
        (-(self.a * t * t)).exp()
    }

    /// Compute in-phase sample.
    pub(super) fn sample(&self, t: F) -> F {
        let two_pi = F::two_pi();
        let yenv = self.envelope(t);
        yenv * (two_pi * self.fc * t).cos()
    }

    /// Compute quadrature sample.
    pub(super) fn sample_quadrature(&self, t: F) -> F {
        let two_pi = F::two_pi();
        let yenv = self.envelope(t);
        yenv * (two_pi * self.fc * t).sin()
    }

    /// Compute envelope sample.
    pub(super) fn sample_envelope(&self, t: F) -> F {
        self.envelope(t)
    }

    /// Compute cutoff time for a target reference level in dB.
    pub fn cutoff_time(&self, tpr: F) -> Result<F, ConfigError> {
        if tpr >= F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "tpr",
                reason: "reference level for time cutoff must be < 0 dB",
            });
        }
        let twenty = F::from_u8(20).ok_or(ConfigError::InvalidArgument {
            arg: "tpr",
            reason: "scalar conversion failed",
        })?;
        let ten = F::from_u8(10).ok_or(ConfigError::InvalidArgument {
            arg: "tpr",
            reason: "scalar conversion failed",
        })?;
        let tref = ten.powf(tpr / twenty);
        Ok((-(tref.ln()) / self.a).sqrt())
    }
}

impl<F> KernelLifecycle for GaussPulseKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    type Config = GaussPulseConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.fc < F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "fc",
                reason: "center frequency must be >= 0",
            });
        }
        if config.bw <= F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "bw",
                reason: "fractional bandwidth must be > 0",
            });
        }
        if config.bwr >= F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "bwr",
                reason: "reference level for bandwidth must be < 0 dB",
            });
        }

        let twenty = F::from_u8(20).ok_or(ConfigError::InvalidArgument {
            arg: "bwr",
            reason: "scalar conversion failed",
        })?;
        let four = F::from_u8(4).ok_or(ConfigError::InvalidArgument {
            arg: "bwr",
            reason: "scalar conversion failed",
        })?;
        let ten = F::from_u8(10).ok_or(ConfigError::InvalidArgument {
            arg: "bwr",
            reason: "scalar conversion failed",
        })?;
        let ref_level = ten.powf(config.bwr / twenty);
        let num = (F::pi() * config.fc * config.bw).powi(2);
        let a = -num / (four * ref_level.ln());

        Ok(Self { fc: config.fc, a })
    }
}

impl<F> GaussPulseWave1D<F> for GaussPulseKernel<F>
where
    F: RealField + Copy + FromPrimitive,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
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
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, t)| *out = self.sample(*t));
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(input.iter().map(|t| self.sample(*t)).collect())
    }
}

/// Constructor config for [`SweepPolyKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepPolyConfig<'a, F>
where
    F: RealField + Copy,
{
    /// Polynomial coefficients in descending power order.
    pub poly: &'a [F],
    /// Phase offset in degrees.
    pub phi_deg: F,
}

/// Trait-first 1D polynomial-frequency sweep generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepPolyKernel<'a, F>
where
    F: RealField + Copy,
{
    poly: &'a [F],
    phi_rad: F,
}

impl<'a, F> SweepPolyKernel<'a, F>
where
    F: RealField + Copy + FromPrimitive,
{
    fn integrated_phase(&self, t: F) -> F {
        // `poly` stores coefficients in descending power. We iterate in reverse
        // to evaluate the integral term-by-term as sum(c_k * t^(k+1)/(k+1)).
        let mut integral = F::zero();
        let mut t_pow = t;
        let mut denom = F::one();
        for coeff in self.poly.iter().rev() {
            integral += (*coeff * t_pow) / denom;
            t_pow *= t;
            denom += F::one();
        }
        F::two_pi() * integral
    }

    /// Compute sweep sample.
    pub(super) fn sample(&self, t: F) -> F {
        (self.integrated_phase(t) + self.phi_rad).cos()
    }
}

impl<'a, F> KernelLifecycle for SweepPolyKernel<'a, F>
where
    F: RealField + Copy + FromPrimitive,
{
    type Config = SweepPolyConfig<'a, F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.poly.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "poly" });
        }
        let deg = F::from_u8(180).ok_or(ConfigError::InvalidArgument {
            arg: "phi_deg",
            reason: "phi conversion failed",
        })?;
        Ok(Self {
            poly: config.poly,
            phi_rad: config.phi_deg * F::pi() / deg,
        })
    }
}

impl<'a, F> SweepPolyWave1D<F> for SweepPolyKernel<'a, F>
where
    F: RealField + Copy + FromPrimitive,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
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
        out.iter_mut()
            .zip(input.iter())
            .for_each(|(out, t)| *out = self.sample(*t));
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        Ok(input.iter().map(|t| self.sample(*t)).collect())
    }
}

/// Constructor config for [`UnitImpulseKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitImpulseConfig {
    /// Number of output samples.
    pub len: usize,
    /// Index at which the impulse is set to one.
    pub idx: usize,
}

/// Trait-first 1D unit-impulse generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitImpulseKernel {
    len: usize,
    idx: usize,
}

impl UnitImpulseKernel {
    /// Return configured impulse length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether configured impulse length is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return configured impulse index.
    pub fn idx(&self) -> usize {
        self.idx
    }
}

impl KernelLifecycle for UnitImpulseKernel {
    type Config = UnitImpulseConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.len == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "len",
                reason: "impulse length must be > 0",
            });
        }
        if config.idx >= config.len {
            return Err(ConfigError::InvalidArgument {
                arg: "idx",
                reason: "impulse index must be within [0, len)",
            });
        }
        Ok(Self {
            len: config.len,
            idx: config.idx,
        })
    }
}

impl<T> UnitImpulse1D<T> for UnitImpulseKernel
where
    T: Zero + One + Copy,
{
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<T> + ?Sized,
    {
        let out = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out.len() != self.len {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: self.len,
                got: out.len(),
            });
        }
        out.fill(T::zero());
        out[self.idx] = T::one();
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc(&self) -> Result<Vec<T>, ExecInvariantViolation> {
        let mut out = vec![T::zero(); self.len];
        out[self.idx] = T::one();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChirpConfig, ChirpKernel, ChirpMethod, GaussPulseConfig, GaussPulseKernel,
        SawtoothWaveConfig, SawtoothWaveKernel, SquareWaveConfig, SquareWaveKernel,
        SweepPolyConfig, SweepPolyKernel, UnitImpulseConfig, UnitImpulseKernel,
    };
    use crate::kernel::{ConfigError, KernelLifecycle};
    use crate::signal::traits::{
        ChirpWave1D, GaussPulseWave1D, SawtoothWave1D, SquareWave1D, SweepPolyWave1D, UnitImpulse1D,
    };
    use crate::signal::wave::{
        chirp, gausspulse, gausspulse_cutoff, sawtooth, square, sweep_poly, unit_impulse,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, Array1};

    #[test]
    fn square_wave_kernel_matches_ndarray_square() {
        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty: 0.67f32 })
            .expect("kernel should initialize");
        let input = arr1(&[
            -4.452f32, -4.182, -3.663, -3.307, -2.995, -2.482, -2.46, -1.929, -1.823, -1.44,
        ]);
        let expected = square(&input, 0.67f32).expect("square reference should succeed");
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-6));
    }

    #[test]
    fn sawtooth_wave_kernel_matches_ndarray_sawtooth() {
        let kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width: 0.3f32 })
            .expect("kernel should initialize");
        let input = arr1(&[-4.0f32, -1.2, -0.1, 0.0, 0.3, 1.7, 3.14, 4.2, 6.1, 7.9]);
        let expected = sawtooth(&input, 0.3f32).expect("sawtooth reference should succeed");
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-6));
    }

    #[test]
    fn chirp_kernel_matches_ndarray_chirp() {
        let kernel = ChirpKernel::try_new(ChirpConfig {
            f0: 2.0f64,
            t1: 2.0,
            f1: 5.0,
            method: ChirpMethod::Quadratic,
            phi_deg: 15.0,
            vertex_zero: false,
        })
        .expect("kernel should initialize");

        let input = arr1(&[0.0f64, 0.25, 0.5, 1.0, 1.5, 2.0]);
        let expected = chirp(
            &input,
            2.0f64,
            2.0,
            5.0,
            ChirpMethod::Quadratic,
            15.0,
            false,
        )
        .expect("chirp reference should succeed");
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-9));
    }

    #[test]
    fn gausspulse_kernel_matches_ndarray_gausspulse() {
        let kernel = GaussPulseKernel::try_new(GaussPulseConfig {
            fc: 5.0f64,
            bw: 0.5,
            bwr: -6.0,
        })
        .expect("kernel should initialize");
        let input = arr1(&[-1.0f64, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]);
        let expected =
            gausspulse(&input, 5.0f64, 0.5, -6.0).expect("gausspulse reference should succeed");
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-12));
    }

    #[test]
    fn sweep_poly_kernel_matches_ndarray_sweep_poly() {
        let poly = [0.025f64, -0.36, 1.25, 2.0];
        let kernel = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &poly,
            phi_deg: 15.0,
        })
        .expect("kernel should initialize");
        let input = arr1(&[0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let expected =
            sweep_poly(&input, &poly, 15.0).expect("sweep_poly reference should succeed");
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-12));
    }

    #[test]
    fn unit_impulse_kernel_matches_ndarray_unit_impulse() {
        let kernel = UnitImpulseKernel::try_new(UnitImpulseConfig { len: 7, idx: 2 })
            .expect("kernel should initialize");
        let expected =
            unit_impulse::<f64>(7, Some(2)).expect("unit_impulse reference should succeed");
        let actual: Vec<f64> = kernel.run_alloc().expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-12));
    }

    #[test]
    fn square_wave_kernel_run_into_supports_ndarray_output() {
        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty: 0.5f64 })
            .expect("kernel should initialize");
        let input = [
            0.0f64,
            core::f64::consts::PI * 0.25,
            core::f64::consts::PI,
            7.0,
        ];
        let mut out = Array1::from(vec![0.0f64; input.len()]);
        kernel
            .run_into(&input, &mut out)
            .expect("run_into should succeed");

        let expected = [1.0f64, 1.0, -1.0, 1.0];
        out.iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-12));
    }

    #[test]
    fn sawtooth_kernel_run_into_supports_ndarray_output() {
        let kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width: 0.5f64 })
            .expect("kernel should initialize");
        let input = [0.0f64, 0.5, 1.0, 1.5];
        let mut out = Array1::from(vec![0.0f64; input.len()]);
        kernel
            .run_into(&input, &mut out)
            .expect("run_into should succeed");
        assert!(out.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn chirp_kernel_run_into_supports_ndarray_output() {
        let kernel = ChirpKernel::try_new(ChirpConfig {
            f0: 3.0f64,
            t1: 2.0,
            f1: 7.0,
            method: ChirpMethod::Linear,
            phi_deg: 0.0,
            vertex_zero: true,
        })
        .expect("kernel should initialize");
        let input = [0.0f64, 0.5, 1.0, 1.5];
        let mut out = Array1::from(vec![0.0f64; input.len()]);
        kernel
            .run_into(&input, &mut out)
            .expect("run_into should succeed");
        assert!(out.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn gausspulse_kernel_run_into_supports_ndarray_output() {
        let kernel = GaussPulseKernel::try_new(GaussPulseConfig {
            fc: 5.0f64,
            bw: 0.5,
            bwr: -6.0,
        })
        .expect("kernel should initialize");
        let input = [0.0f64, 0.25, 0.5, 0.75];
        let mut out = Array1::from(vec![0.0f64; input.len()]);
        kernel
            .run_into(&input, &mut out)
            .expect("run_into should succeed");
        assert!(out.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn sweep_poly_kernel_run_into_supports_ndarray_output() {
        let poly = [0.025f64, -0.36, 1.25, 2.0];
        let kernel = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &poly,
            phi_deg: 15.0,
        })
        .expect("kernel should initialize");
        let input = [0.0f64, 0.5, 1.0, 1.5];
        let mut out = Array1::from(vec![0.0f64; input.len()]);
        kernel
            .run_into(&input, &mut out)
            .expect("run_into should succeed");
        assert!(out.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn unit_impulse_kernel_run_into_supports_ndarray_output() {
        let kernel = UnitImpulseKernel::try_new(UnitImpulseConfig { len: 5, idx: 3 })
            .expect("kernel should initialize");
        let mut out = Array1::from(vec![0.0f64; 5]);
        kernel.run_into(&mut out).expect("run_into should succeed");
        assert_eq!(out.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn square_wave_kernel_validates_duty_and_output_length() {
        let err = SquareWaveKernel::try_new(SquareWaveConfig { duty: 1.2f32 })
            .expect_err("duty above 1 must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "duty",
                reason: "duty must be in [0, 1]",
            }
        );

        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty: 0.5f32 })
            .expect("kernel should initialize");
        let input = [0.0f32, 1.0, 2.0];
        let mut out = [0.0f32; 2];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("short output should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn sawtooth_kernel_validates_width() {
        let err = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width: -0.1f32 })
            .expect_err("width below zero must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "width",
                reason: "width must be in [0, 1]",
            }
        );
    }

    #[test]
    fn chirp_kernel_validates_constructor_rules() {
        let err = ChirpKernel::try_new(ChirpConfig {
            f0: -2.0f64,
            t1: 2.0,
            f1: 5.0,
            method: ChirpMethod::Logarithmic,
            phi_deg: 0.0,
            vertex_zero: true,
        })
        .expect_err("log chirp with mixed-sign endpoints must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "f0/f1",
                reason: "logarithmic chirp requires nonzero f0/f1 with same sign",
            }
        );

        let err = ChirpKernel::try_new(ChirpConfig {
            f0: 0.0f64,
            t1: 2.0,
            f1: 3.0,
            method: ChirpMethod::Hyperbolic,
            phi_deg: 0.0,
            vertex_zero: true,
        })
        .expect_err("hyperbolic chirp with zero f0 must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "f0/f1",
                reason: "hyperbolic chirp requires nonzero f0 and f1",
            }
        );
    }

    #[test]
    fn gausspulse_kernel_validates_constructor_rules_and_cutoff() {
        let err = GaussPulseKernel::try_new(GaussPulseConfig {
            fc: -1.0f64,
            bw: 0.5,
            bwr: -6.0,
        })
        .expect_err("negative fc must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "fc",
                reason: "center frequency must be >= 0",
            }
        );

        let kernel = GaussPulseKernel::try_new(GaussPulseConfig {
            fc: 5.0f64,
            bw: 0.5,
            bwr: -6.0,
        })
        .expect("kernel should initialize");
        let cutoff = kernel.cutoff_time(-60.0).expect("cutoff should compute");
        let expected = gausspulse_cutoff(5.0f64, 0.5, -6.0, -60.0)
            .expect("gausspulse cutoff reference should succeed");
        assert_abs_diff_eq!(cutoff, expected, epsilon = 1e-12);
    }

    #[test]
    fn sweep_poly_kernel_validates_constructor_and_output_length() {
        let err = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &[] as &[f64],
            phi_deg: 0.0,
        })
        .expect_err("empty poly must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "poly" });

        let poly = [1.0f64, 2.0];
        let kernel = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &poly,
            phi_deg: 0.0,
        })
        .expect("kernel should initialize");
        let input = [0.0f64, 0.5, 1.0];
        let mut out = [0.0f64; 2];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("short output should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn unit_impulse_kernel_validates_length_index_and_output_length() {
        let err = UnitImpulseKernel::try_new(UnitImpulseConfig { len: 0, idx: 0 })
            .expect_err("zero length must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "len",
                reason: "impulse length must be > 0",
            }
        );

        let err = UnitImpulseKernel::try_new(UnitImpulseConfig { len: 4, idx: 4 })
            .expect_err("idx at len must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "idx",
                reason: "impulse index must be within [0, len)",
            }
        );

        let kernel = UnitImpulseKernel::try_new(UnitImpulseConfig { len: 4, idx: 1 })
            .expect("kernel should initialize");
        let mut out = [0.0f64; 3];
        let err = kernel
            .run_into(&mut out)
            .expect_err("short output should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: 4,
                got: 3
            }
        ));
    }
}
