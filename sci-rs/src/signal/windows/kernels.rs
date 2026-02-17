//! Trait-first window generation kernels.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Write1D};
use crate::signal::traits::WindowGenerate;
use crate::special::Bessel;
use alloc::vec::Vec;
use nalgebra::RealField;
use num_traits::{real::Real, Float};

use super::{get_window, GetWindow, GetWindowBuilder, Window};

/// Owned window builder specification suitable for kernel configs.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowBuilderOwned<F>
where
    F: Real,
{
    /// Boxcar window.
    Boxcar,
    /// Triangle window.
    Triangle,
    /// Blackman window.
    Blackman,
    /// Hamming window.
    Hamming,
    /// Nuttall window.
    Nuttall,
    /// Kaiser window.
    Kaiser {
        /// Shape parameter `beta`.
        beta: F,
    },
    /// General cosine window.
    GeneralCosine {
        /// Coefficients for weighted cosine terms.
        weights: Vec<F>,
    },
    /// General gaussian window.
    GeneralGaussian {
        /// Shape parameter.
        p: F,
        /// Width parameter.
        width: F,
    },
    /// General hamming window.
    GeneralHamming {
        /// Alpha coefficient.
        coefficient: F,
    },
}

impl<F> WindowBuilderOwned<F>
where
    F: Real + Copy,
{
    fn as_builder(&self) -> GetWindowBuilder<'_, F> {
        match self {
            WindowBuilderOwned::Boxcar => GetWindowBuilder::Boxcar,
            WindowBuilderOwned::Triangle => GetWindowBuilder::Triangle,
            WindowBuilderOwned::Blackman => GetWindowBuilder::Blackman,
            WindowBuilderOwned::Hamming => GetWindowBuilder::Hamming,
            WindowBuilderOwned::Nuttall => GetWindowBuilder::Nuttall,
            WindowBuilderOwned::Kaiser { beta } => GetWindowBuilder::Kaiser { beta: *beta },
            WindowBuilderOwned::GeneralCosine { weights } => GetWindowBuilder::GeneralCosine {
                weights: weights.as_slice(),
            },
            WindowBuilderOwned::GeneralGaussian { p, width } => GetWindowBuilder::GeneralGaussian {
                p: *p,
                width: *width,
            },
            WindowBuilderOwned::GeneralHamming { coefficient } => {
                GetWindowBuilder::GeneralHamming {
                    coefficient: *coefficient,
                }
            }
        }
    }

    /// Build a concrete [`Window`] with the given length and symmetry mode.
    pub fn build_window(&self, nx: usize, fftbins: Option<bool>) -> Window<F> {
        get_window(self.as_builder(), nx, fftbins)
    }
}

/// Constructor config for [`WindowKernel`].
#[derive(Debug, Clone, PartialEq)]
pub struct WindowConfig<F>
where
    F: Real,
{
    /// Window family and parameters.
    pub builder: WindowBuilderOwned<F>,
    /// Output length.
    pub nx: usize,
    /// FFT-bin mode (`Some(true)` periodic, `Some(false)` symmetric).
    pub fftbins: Option<bool>,
}

/// Trait-first window generation kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowKernel<F>
where
    F: Real,
{
    builder: WindowBuilderOwned<F>,
    nx: usize,
    fftbins: Option<bool>,
}

impl<F> KernelLifecycle for WindowKernel<F>
where
    F: Real + Copy,
{
    type Config = WindowConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.nx <= 1 {
            return Err(ConfigError::InvalidArgument {
                arg: "nx",
                reason: "window length must be greater than 1",
            });
        }
        if matches!(
            &config.builder,
            WindowBuilderOwned::GeneralCosine { weights } if weights.is_empty()
        ) {
            return Err(ConfigError::EmptyInput { arg: "weights" });
        }

        Ok(Self {
            builder: config.builder,
            nx: config.nx,
            fftbins: config.fftbins,
        })
    }
}

impl<F> WindowGenerate<F> for WindowKernel<F>
where
    F: Real + Float + RealField + Bessel + Copy,
{
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<F> + ?Sized,
    {
        let generated = self.run_alloc()?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != generated.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: generated.len(),
                got: out_slice.len(),
            });
        }
        out_slice.copy_from_slice(&generated);
        Ok(())
    }

    fn run_alloc(&self) -> Result<Vec<F>, ExecInvariantViolation> {
        let window = self.builder.build_window(self.nx, self.fftbins);
        Ok(window.get_window())
    }
}

#[cfg(test)]
mod tests {
    use super::{WindowBuilderOwned, WindowConfig, WindowKernel};
    use crate::kernel::{ConfigError, KernelLifecycle};
    use crate::signal::traits::WindowGenerate;
    use crate::signal::windows::{GetWindow, Hamming};
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn window_kernel_alloc_matches_hamming_reference() {
        let kernel = WindowKernel::try_new(WindowConfig {
            builder: WindowBuilderOwned::Hamming,
            nx: 17,
            fftbins: Some(false),
        })
        .expect("window kernel should initialize");

        let actual = kernel.run_alloc().expect("window run_alloc should succeed");
        let expected: Vec<f64> = Hamming::new(17, true).get_window();
        actual
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 1e-10));
    }

    #[test]
    fn window_kernel_run_into_ndarray() {
        let kernel = WindowKernel::try_new(WindowConfig {
            builder: WindowBuilderOwned::Boxcar,
            nx: 8,
            fftbins: Some(false),
        })
        .expect("window kernel should initialize");

        let mut out = Array1::from(vec![0.0f64; 8]);
        kernel
            .run_into(&mut out)
            .expect("window run_into should succeed");
        out.iter()
            .for_each(|v| assert_abs_diff_eq!(*v, 1.0f64, epsilon = 1e-12));
    }

    #[test]
    fn window_kernel_constructor_rejects_invalid_config() {
        let err = WindowKernel::<f64>::try_new(WindowConfig {
            builder: WindowBuilderOwned::GeneralCosine {
                weights: Vec::new(),
            },
            nx: 32,
            fftbins: None,
        })
        .expect_err("empty general cosine weights must fail");
        assert_eq!(err, ConfigError::EmptyInput { arg: "weights" });

        let err = WindowKernel::<f64>::try_new(WindowConfig {
            builder: WindowBuilderOwned::Hamming,
            nx: 1,
            fftbins: None,
        })
        .expect_err("short windows must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "nx",
                reason: "window length must be greater than 1",
            }
        );
    }
}
