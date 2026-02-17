//! Multirate helpers analogous to `scipy.signal` multirate APIs.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::{Decimate1D, ResamplePoly1D, UpFirDn1D};
use num_traits::{Float, FromPrimitive};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

fn convolve_full<F>(x: &[F], h: &[F]) -> Vec<F>
where
    F: Float + Copy,
{
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let mut out = vec![F::zero(); x.len() + h.len() - 1];
    for (i, &xv) in x.iter().enumerate() {
        for (j, &hv) in h.iter().enumerate() {
            out[i + j] = out[i + j] + xv * hv;
        }
    }
    out
}

fn upfirdn_impl<F>(h: &[F], x: &[F], up: usize, down: usize) -> Vec<F>
where
    F: Float + Copy,
{
    if h.is_empty() || x.is_empty() || up == 0 || down == 0 {
        return Vec::new();
    }

    let mut upsampled = vec![F::zero(); x.len() * up];
    for (i, &v) in x.iter().enumerate() {
        upsampled[i * up] = v;
    }

    let filtered = convolve_full(&upsampled, h);
    filtered.into_iter().step_by(down).collect()
}

fn resample_poly_impl<F>(x: &[F], up: usize, down: usize) -> Vec<F>
where
    F: Float + Copy + FromPrimitive,
{
    if x.is_empty() || up == 0 || down == 0 {
        return Vec::new();
    }
    if up == down {
        return x.to_vec();
    }

    let out_len = (x.len() * up).div_ceil(down);
    let up_f = F::from_usize(up).expect("ratio conversion");
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let pos_num = i * down;
        let i0 = pos_num / up;
        let frac = F::from_usize(pos_num % up).expect("fraction conversion") / up_f;

        let y = if i0 + 1 < x.len() {
            x[i0] * (F::one() - frac) + x[i0 + 1] * frac
        } else {
            x[x.len() - 1]
        };
        out.push(y);
    }

    out
}

fn decimate_impl<F>(x: &[F], q: usize) -> Vec<F>
where
    F: Float + Copy + FromPrimitive,
{
    if q == 0 {
        return Vec::new();
    }
    resample_poly_impl(x, 1, q)
}

/// Constructor config for [`UpFirDnKernel`].
#[derive(Debug, Clone)]
pub struct UpFirDnConfig<F>
where
    F: Float + Copy,
{
    /// FIR coefficients.
    pub h: Vec<F>,
    /// Integer upsampling factor.
    pub up: usize,
    /// Integer downsampling factor.
    pub down: usize,
}

/// Trait-first `upfirdn` kernel.
#[derive(Debug, Clone)]
pub struct UpFirDnKernel<F>
where
    F: Float + Copy,
{
    h: Vec<F>,
    up: usize,
    down: usize,
}

impl<F> UpFirDnKernel<F>
where
    F: Float + Copy,
{
    /// Return configured upsampling factor.
    pub fn up(&self) -> usize {
        self.up
    }

    /// Return configured downsampling factor.
    pub fn down(&self) -> usize {
        self.down
    }

    /// Return expected output length for a given input length.
    pub fn expected_len(&self, input_len: usize) -> usize {
        if input_len == 0 {
            0
        } else {
            (input_len * self.up + self.h.len() - 1).div_ceil(self.down)
        }
    }
}

impl<F> KernelLifecycle for UpFirDnKernel<F>
where
    F: Float + Copy,
{
    type Config = UpFirDnConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.h.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "h" });
        }
        if config.up == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "up",
                reason: "up must be > 0",
            });
        }
        if config.down == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "down",
                reason: "down must be > 0",
            });
        }
        Ok(Self {
            h: config.h,
            up: config.up,
            down: config.down,
        })
    }
}

impl<F> UpFirDn1D<F> for UpFirDnKernel<F>
where
    F: Float + Copy,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "upfirdn input must be non-empty",
            });
        }

        let expected = self.expected_len(input.len());
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected,
                got: out_slice.len(),
            });
        }

        let y = upfirdn_impl(&self.h, input, self.up, self.down);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "upfirdn input must be non-empty",
            });
        }
        Ok(upfirdn_impl(&self.h, input, self.up, self.down))
    }
}

/// Constructor config for [`ResamplePolyKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResamplePolyConfig {
    /// Integer upsampling factor.
    pub up: usize,
    /// Integer downsampling factor.
    pub down: usize,
}

/// Trait-first `resample_poly` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResamplePolyKernel {
    up: usize,
    down: usize,
}

impl ResamplePolyKernel {
    /// Return configured upsampling factor.
    pub fn up(&self) -> usize {
        self.up
    }

    /// Return configured downsampling factor.
    pub fn down(&self) -> usize {
        self.down
    }

    /// Return expected output length for a given input length.
    pub fn expected_len(&self, input_len: usize) -> usize {
        if input_len == 0 {
            0
        } else {
            (input_len * self.up).div_ceil(self.down)
        }
    }
}

impl KernelLifecycle for ResamplePolyKernel {
    type Config = ResamplePolyConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.up == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "up",
                reason: "up must be > 0",
            });
        }
        if config.down == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "down",
                reason: "down must be > 0",
            });
        }
        Ok(Self {
            up: config.up,
            down: config.down,
        })
    }
}

impl<F> ResamplePoly1D<F> for ResamplePolyKernel
where
    F: Float + Copy + FromPrimitive,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "resample_poly input must be non-empty",
            });
        }
        let expected = self.expected_len(input.len());
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected,
                got: out_slice.len(),
            });
        }

        let y = resample_poly_impl(input, self.up, self.down);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "resample_poly input must be non-empty",
            });
        }
        Ok(resample_poly_impl(input, self.up, self.down))
    }
}

/// Constructor config for [`DecimateKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimateConfig {
    /// Integer decimation factor.
    pub q: usize,
}

/// Trait-first `decimate` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimateKernel {
    q: usize,
}

impl DecimateKernel {
    /// Return configured decimation factor.
    pub fn q(&self) -> usize {
        self.q
    }

    /// Return expected output length for a given input length.
    pub fn expected_len(&self, input_len: usize) -> usize {
        if input_len == 0 {
            0
        } else {
            input_len.div_ceil(self.q)
        }
    }
}

impl KernelLifecycle for DecimateKernel {
    type Config = DecimateConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.q == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "q",
                reason: "q must be > 0",
            });
        }
        Ok(Self { q: config.q })
    }
}

impl<F> Decimate1D<F> for DecimateKernel
where
    F: Float + Copy + FromPrimitive,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "decimate input must be non-empty",
            });
        }
        let expected = self.expected_len(input.len());
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected,
                got: out_slice.len(),
            });
        }

        let y = decimate_impl(input, self.q);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "decimate input must be non-empty",
            });
        }
        Ok(decimate_impl(input, self.q))
    }
}

/// Upsample by `up`, apply FIR `h`, and downsample by `down`.
pub(crate) fn upfirdn<F>(
    h: &[F],
    x: &[F],
    up: usize,
    down: usize,
) -> Result<Vec<F>, ExecInvariantViolation>
where
    F: Float + Copy,
{
    let kernel = UpFirDnKernel::try_new(UpFirDnConfig {
        h: h.to_vec(),
        up,
        down,
    })
    .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Polyphase-like resampling using linear interpolation.
///
/// This implementation targets deterministic embedded-friendly behavior and
/// supports the common `up/down` ratio contract from SciPy's `resample_poly`.
pub(crate) fn resample_poly<F>(
    x: &[F],
    up: usize,
    down: usize,
) -> Result<Vec<F>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let kernel = ResamplePolyKernel::try_new(ResamplePolyConfig { up, down })
        .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

/// Decimate by integer factor `q`.
pub(crate) fn decimate<F>(x: &[F], q: usize) -> Result<Vec<F>, ExecInvariantViolation>
where
    F: Float + Copy + FromPrimitive,
{
    let kernel =
        DecimateKernel::try_new(DecimateConfig { q }).map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::traits::{Decimate1D, ResamplePoly1D, UpFirDn1D};
    use approx::assert_abs_diff_eq;

    #[test]
    fn upfirdn_matches_simple_reference() {
        let x = [1.0f64, 2.0, 3.0];
        let h = [1.0f64, 1.0];
        let y = upfirdn(&h, &x, 2, 1).expect("upfirdn should succeed");
        let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0];
        assert_eq!(y, expected);
    }

    #[test]
    fn upfirdn_kernel_contracts_validate_and_check_output_shape() {
        assert!(UpFirDnKernel::<f64>::try_new(UpFirDnConfig {
            h: vec![],
            up: 2,
            down: 1,
        })
        .is_err());
        assert!(UpFirDnKernel::<f64>::try_new(UpFirDnConfig {
            h: vec![1.0],
            up: 0,
            down: 1,
        })
        .is_err());
        assert!(UpFirDnKernel::<f64>::try_new(UpFirDnConfig {
            h: vec![1.0],
            up: 1,
            down: 0,
        })
        .is_err());

        let kernel = UpFirDnKernel::try_new(UpFirDnConfig {
            h: vec![1.0f64, 1.0],
            up: 2,
            down: 1,
        })
        .expect("valid config");
        let input = [1.0f64, 2.0, 3.0];
        let mut out = vec![0.0f64; 6];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn resample_poly_interpolates_to_expected_length() {
        let x = [1.0f64, 2.0, 3.0];
        let y = resample_poly(&x, 2, 1).expect("resample_poly should succeed");
        let expected = [1.0, 1.5, 2.0, 2.5, 3.0, 3.0];
        assert_eq!(y.len(), expected.len());
        for (a, b) in y.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn resample_poly_kernel_contracts_validate_and_check_output_shape() {
        assert!(ResamplePolyKernel::try_new(ResamplePolyConfig { up: 0, down: 1 }).is_err());
        assert!(ResamplePolyKernel::try_new(ResamplePolyConfig { up: 1, down: 0 }).is_err());

        let kernel = ResamplePolyKernel::try_new(ResamplePolyConfig { up: 2, down: 1 })
            .expect("valid config");
        let input = [1.0f64, 2.0, 3.0];
        let mut out = vec![0.0f64; 5];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn decimate_reduces_length() {
        let x = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let y = decimate(&x, 2).expect("decimate should succeed");
        assert_eq!(y, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn decimate_kernel_contracts_validate_and_check_output_shape() {
        assert!(DecimateKernel::try_new(DecimateConfig { q: 0 }).is_err());

        let kernel = DecimateKernel::try_new(DecimateConfig { q: 2 }).expect("valid config");
        let input = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f64; 2];
        let err = kernel
            .run_into(&input, &mut out)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }
}
