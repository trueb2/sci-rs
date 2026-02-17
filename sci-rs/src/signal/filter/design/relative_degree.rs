use nalgebra::Complex;

#[cfg(feature = "alloc")]
use crate::error::Error;

#[cfg(feature = "alloc")]
pub(crate) fn relative_degree_dyn<F>(zeros: &[Complex<F>], poles: &[Complex<F>]) -> usize {
    relative_degree_checked(zeros, poles).expect("invalid relative degree configuration")
}

/// Checked relative-degree helper returning a deterministic argument error.
#[cfg(feature = "alloc")]
pub(crate) fn relative_degree_checked<F>(
    zeros: &[Complex<F>],
    poles: &[Complex<F>],
) -> Result<usize, Error> {
    let degree = poles.len() as isize - zeros.len() as isize;
    if degree < 0 {
        return Err(Error::InvalidArg {
            arg: "zpk".into(),
            reason: "improper transfer function; poles must be >= zeros".into(),
        });
    }
    Ok(degree as usize)
}
