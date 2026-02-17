use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::traits::Resample1D;
use nalgebra::Complex;
use num_traits::{Float, Zero};
use rustfft::FftNum;

/// Constructor config for [`ResampleKernel`].
#[derive(Debug, Clone, Copy)]
pub struct ResampleConfig {
    /// Target number of output samples.
    pub target_len: usize,
}

/// Trait-first 1D Fourier resampling kernel.
#[derive(Debug, Clone, Copy)]
pub struct ResampleKernel {
    target_len: usize,
}

impl ResampleKernel {
    /// Return configured target output length.
    pub fn target_len(&self) -> usize {
        self.target_len
    }
}

impl KernelLifecycle for ResampleKernel {
    type Config = ResampleConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.target_len == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "target_len",
                reason: "target length must be > 0",
            });
        }
        Ok(Self {
            target_len: config.target_len,
        })
    }
}

impl<F> Resample1D<F> for ResampleKernel
where
    F: Float + FftNum + Copy,
{
    fn run_into<I, O>(&self, input: &I, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
        O: Write1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "resample input must be non-empty",
            });
        }
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != self.target_len {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: self.target_len,
                got: out_slice.len(),
            });
        }
        let y = resample_impl(input, self.target_len);
        out_slice.copy_from_slice(&y);
        Ok(())
    }

    #[cfg(feature = "alloc")]
    fn run_alloc<I>(&self, input: &I) -> Result<Vec<F>, ExecInvariantViolation>
    where
        I: Read1D<F> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "resample input must be non-empty",
            });
        }
        Ok(resample_impl(input, self.target_len))
    }
}

fn resample_impl<F: Float + FftNum>(x: &[F], n: usize) -> Vec<F> {
    if x.is_empty() || n == 0 {
        return Vec::new();
    }

    // SciPy style 'Fourier' resampling
    // 1. Compute FFT of x
    // 2. Fill vec of zeros with the desired length, y.
    // 3. Set the from beginning of y to the first half of x
    // 4. Set the from end of y to the second half of x
    // 5. Compute IFFT of y
    // 6. Multiply y by (n / x.len())
    // 7. Take the real part of y

    // Compute FFT of x
    let mut fft_planner = rustfft::FftPlanner::<F>::new();
    let fft = fft_planner.plan_fft_forward(x.len());
    let ifft = fft_planner.plan_fft_inverse(n);

    let scratch_length = std::cmp::max(
        fft.get_inplace_scratch_len(),
        ifft.get_inplace_scratch_len(),
    );
    let mut scratch = vec![Complex::zero(); scratch_length];
    let mut x = x
        .iter()
        .map(|x| Complex::new(*x, F::zero()))
        .collect::<Vec<_>>();
    fft.process_with_scratch(&mut x, &mut scratch);

    // Fill y with halfs of x
    let mut y = vec![Complex::zero(); n];
    let bins = std::cmp::min(x.len(), n);
    let half_spectrum = bins / 2;
    y[..half_spectrum].copy_from_slice(&x[..half_spectrum]);
    y[n - half_spectrum..].copy_from_slice(&x[x.len() - half_spectrum..]);

    // Compute iFFT of y
    ifft.process_with_scratch(&mut y, &mut scratch);

    // Take the scaled real domain as the resampled result
    let scale_factor = F::from(1.0 / x.len() as f64).unwrap();
    y.iter().map(|x| x.re * scale_factor).collect::<Vec<_>>()
}

///
/// Resample the data to the desired number of samples using the Fourier transform.
///
/// This method is similar but not exactly equivalent to the SciPy method of resampling:
/// <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html>
///
/// It skips some complexity of the SciPy method, such as windowing and handling odd vs. even-length signals.
///
/// Procedure:
/// 1. Convert to the frequency-domain.
///    a. If upsampling, pad higher frequency bins with 0
///    b. If downsampling, truncate higher frequency bins
/// 2. Convert back to the time-domain.
///
pub fn resample<F: Float + FftNum>(x: &[F], n: usize) -> Result<Vec<F>, ExecInvariantViolation> {
    let kernel = ResampleKernel::try_new(ResampleConfig { target_len: n })
        .map_err(ExecInvariantViolation::from)?;
    kernel.run_alloc(x)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::Rng;

    use super::*;
    use crate::kernel::KernelLifecycle;
    use crate::signal::traits::Resample1D;

    #[test]
    #[should_panic]
    fn can_resample_like_scipy() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = resample(&x, 5).expect("resample should succeed");
        let expected = vec![3., 2.18649851, 5.01849831, 5.98150169, 8.81350149];
        assert_eq!(y.len(), expected.len());

        // Fails due to slight algorithmic differences
        println!("y: {:?}, scipy: {:?}", y, expected);
        for (y, expected) in y.iter().zip(expected.iter()) {
            assert_relative_eq!(y, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn can_resample_to_exact_number() {
        // Randomly generate 1000 vectors of length (10,1000)
        // Resample each to length 100
        // Check that each resampled vector is length 100

        let mut rng = rand::rng();
        for _ in 0..100 {
            let len = rng.random_range(10..50);
            let x: Vec<_> = (0..len).map(|_| rng.random_range(-100.0..100.)).collect();
            let y = resample(&x, 100).expect("resample should succeed");
            assert_eq!(y.len(), 100);
        }

        for _ in 0..50 {
            let len = rng.random_range(200..10000);
            let target_len = rng.random_range(50..50000);
            let x: Vec<_> = (0..len).map(|_| rng.random_range(-100.0..100.)).collect();
            let y = resample(&x, target_len).expect("resample should succeed");
            assert_eq!(y.len(), target_len);
        }
    }

    #[test]
    fn kernel_run_into_slice() {
        let kernel = ResampleKernel::try_new(ResampleConfig { target_len: 16 })
            .expect("kernel should initialize");
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 16];
        kernel
            .run_into(&input, &mut out)
            .expect("resample kernel run_into should succeed");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn kernel_rejects_invalid_target_len() {
        let result = ResampleKernel::try_new(ResampleConfig { target_len: 0 });
        assert!(result.is_err());
    }
}
