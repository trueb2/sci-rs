use nalgebra::Complex;

#[cfg(feature = "alloc")]
use crate::error::Error;

#[cfg(feature = "alloc")]
pub(crate) fn relative_degree_dyn<F>(zeros: &[Complex<F>], poles: &[Complex<F>]) -> usize {
    debug_assert!(
        poles.len() >= zeros.len(),
        "improper transfer function; poles must be >= zeros"
    );
    poles.len().saturating_sub(zeros.len())
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
