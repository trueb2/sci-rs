use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D};
use ::core::{
    borrow::Borrow,
    ops::{Div, Neg},
};
use nalgebra::{allocator::Allocator, *};
use num_traits::{One, Zero};

/// 1D companion-matrix construction capability.
#[cfg(feature = "alloc")]
pub trait CompanionBuild1D<T> {
    /// Output matrix type.
    type Output;

    /// Build companion matrix from polynomial coefficients.
    fn run<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized;
}

/// Constructor config for [`CompanionKernel`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompanionConfig {
    /// Optional expected coefficient length.
    pub expected_len: Option<usize>,
}

/// Trait-first companion-matrix kernel.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompanionKernel {
    expected_len: Option<usize>,
}

#[cfg(feature = "alloc")]
impl KernelLifecycle for CompanionKernel {
    type Config = CompanionConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(expected_len) = config.expected_len {
            if expected_len < 2 {
                return Err(ConfigError::InvalidArgument {
                    arg: "expected_len",
                    reason: "companion requires at least 2 coefficients",
                });
            }
        }
        Ok(Self {
            expected_len: config.expected_len,
        })
    }
}

#[cfg(feature = "alloc")]
impl<T> CompanionBuild1D<T> for CompanionKernel
where
    T: Scalar + One + Zero + Div<Output = T> + Neg<Output = T> + Copy + PartialEq,
    DefaultAllocator: Allocator<Dyn, Dyn>,
{
    type Output = OMatrix<T, Dyn, Dyn>;

    fn run<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<T> + ?Sized,
    {
        let coeffs = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if coeffs.len() < 2 {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "companion requires at least 2 coefficients",
            });
        }
        if let Some(expected_len) = self.expected_len {
            if coeffs.len() != expected_len {
                return Err(ExecInvariantViolation::LengthMismatch {
                    arg: "coeffs",
                    expected: expected_len,
                    got: coeffs.len(),
                });
            }
        }
        if coeffs[0] == T::zero() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "leading coefficient must be non-zero",
            });
        }
        Ok(companion_from_slice(coeffs))
    }
}

#[cfg(feature = "alloc")]
fn companion_from_slice<T>(coeffs: &[T]) -> OMatrix<T, Dyn, Dyn>
where
    T: Scalar + One + Zero + Div<Output = T> + Neg<Output = T> + Copy,
    DefaultAllocator: Allocator<Dyn, Dyn>,
{
    let m = coeffs.len();
    let a0 = coeffs[0];
    let itr = coeffs
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, ai)| ((0, i), -*ai / a0))
        .chain((0..(m - 2)).map(|i| (((i + 1), i), T::one())));
    let mut matrix = Matrix::<
        T,
        Dyn,
        Dyn,
        <DefaultAllocator as allocator::Allocator<Dyn, Dyn>>::Buffer<T>,
    >::zeros(m - 1, m - 1);
    for (i, t) in itr {
        unsafe {
            *matrix.get_unchecked_mut(i) = t;
        }
    }
    matrix
}

///
/// Create a companion matrix.
///
#[cfg(feature = "alloc")]
pub fn companion_dyn<T, B, I>(itr: I, m: usize) -> OMatrix<T, Dyn, Dyn>
where
    T: Scalar + One + Zero + Div<Output = T> + Neg<Output = T> + Copy + PartialEq,
    B: Borrow<T>,
    I: Iterator<Item = B>,
    DefaultAllocator: Allocator<Dyn, Dyn>,
{
    let coeffs = itr
        .take(m)
        .map(|b| *b.borrow())
        .collect::<alloc::vec::Vec<_>>();
    if coeffs.len() < 2 {
        return Matrix::<
            T,
            Dyn,
            Dyn,
            <DefaultAllocator as allocator::Allocator<Dyn, Dyn>>::Buffer<T>,
        >::zeros(0, 0);
    }
    if coeffs[0] == T::zero() {
        Matrix::<
            T,
            Dyn,
            Dyn,
            <DefaultAllocator as allocator::Allocator<Dyn, Dyn>>::Buffer<T>,
        >::zeros(coeffs.len() - 1, coeffs.len() - 1)
    } else {
        companion_checked_dyn(coeffs.iter().copied(), coeffs.len())
            .unwrap_or_else(|_| companion_from_slice(&coeffs))
    }
}

///
/// Checked companion matrix construction from iterator input.
///
#[cfg(feature = "alloc")]
pub fn companion_checked_dyn<T, B, I>(itr: I, m: usize) -> Result<OMatrix<T, Dyn, Dyn>, ConfigError>
where
    T: Scalar + One + Zero + Div<Output = T> + Neg<Output = T> + Copy + PartialEq,
    B: Borrow<T>,
    I: Iterator<Item = B>,
    DefaultAllocator: Allocator<Dyn, Dyn>,
{
    let coeffs = itr
        .take(m)
        .map(|b| *b.borrow())
        .collect::<alloc::vec::Vec<_>>();
    if coeffs.len() < 2 {
        return Err(ConfigError::InvalidArgument {
            arg: "coeffs",
            reason: "companion requires at least 2 coefficients",
        });
    }
    if coeffs[0] == T::zero() {
        return Err(ConfigError::InvalidArgument {
            arg: "coeffs",
            reason: "leading coefficient must be non-zero",
        });
    }
    let kernel = CompanionKernel::try_new(CompanionConfig {
        expected_len: Some(coeffs.len()),
    })?;
    kernel
        .run(&coeffs)
        .map_err(|_| ConfigError::InvalidArgument {
            arg: "coeffs",
            reason: "companion kernel execution failed",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelLifecycle;

    #[cfg(feature = "std")]
    #[test]
    fn scipy_example_dyn() {
        const M: usize = 4;
        let data: [_; M] = [1, -10, 31, -30];
        let matrix: DMatrix<_> = companion_dyn(data.iter().map(|i| *i as f32), data.len());

        let expected = matrix!(
            10., -31.,  30.;
            1.,   0.,   0.;
            0.,   1.,   0.;
        );

        assert_eq!(expected, matrix);
    }

    #[cfg(feature = "std")]
    #[test]
    fn companion_kernel_matches_function_and_validates() {
        let coeffs = [1.0f32, -10.0, 31.0, -30.0];
        let kernel = CompanionKernel::try_new(CompanionConfig {
            expected_len: Some(coeffs.len()),
        })
        .expect("kernel should initialize");
        let actual = kernel.run(&coeffs).expect("kernel should run");
        let expected = companion_dyn(coeffs.iter().copied(), coeffs.len());
        assert_eq!(actual, expected);

        let bad_len = CompanionKernel::try_new(CompanionConfig {
            expected_len: Some(1),
        })
        .expect_err("short expected_len should fail");
        assert_eq!(
            bad_len,
            ConfigError::InvalidArgument {
                arg: "expected_len",
                reason: "companion requires at least 2 coefficients",
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn companion_checked_dyn_matches_kernel() {
        let coeffs = [1.0f32, -10.0, 31.0, -30.0];
        let checked =
            companion_checked_dyn(coeffs.iter().copied(), coeffs.len()).expect("checked companion");
        let legacy = companion_dyn(coeffs.iter().copied(), coeffs.len());
        assert_eq!(checked, legacy);
    }
}
