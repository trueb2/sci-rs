//! Trait-first kernels for waveform generation.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::SquareWave1D;
use nalgebra::RealField;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

    fn sample(&self, t: F) -> F {
        let two_pi = F::two_pi();
        let duty_threshold = two_pi * self.duty;
        let mut x = t % two_pi;
        if x < F::zero() {
            x += two_pi;
        }
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

#[cfg(test)]
mod tests {
    use super::{SquareWaveConfig, SquareWaveKernel};
    use crate::kernel::{ConfigError, KernelLifecycle};
    use crate::signal::traits::SquareWave1D;
    use crate::signal::wave::square;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, Array1};

    #[test]
    fn square_wave_kernel_matches_ndarray_square() {
        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty: 0.67f32 })
            .expect("kernel should initialize");
        let input = arr1(&[
            -4.452f32, -4.182, -3.663, -3.307, -2.995, -2.482, -2.46, -1.929, -1.823, -1.44,
        ]);
        let expected = square(&input, 0.67f32);
        let actual = kernel.run_alloc(&input).expect("kernel should run");
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(*a, *b, epsilon = 1e-6));
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
}
