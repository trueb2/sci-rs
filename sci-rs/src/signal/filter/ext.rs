use core::ops::Sub;

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, Scalar};
use num_traits::{One, Zero};
use sci_rs_core::{Error, Result};

/// Pad types.
///
/// Used by [super::sosfiltfilt].  
/// This differs from [super::FiltFiltPadType] which has different semantics.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Pad {
    /// No padding.
    None,

    /// Even extension of the first and last values.
    Even,

    /// Odd extension of the first and last values.
    Odd,

    /// Repeat of first and last values.
    Constant,
}

fn to_dyn_matrix<T, M, N>(x: OMatrix<T, M, N>) -> OMatrix<T, Dyn, Dyn>
where
    T: Scalar + Copy,
    M: Dim,
    N: Dim,
    DefaultAllocator: Allocator<M, N> + Allocator<Dyn, Dyn>,
{
    nalgebra::Matrix::<
        T,
        nalgebra::Dyn,
        nalgebra::Dyn,
        <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
            nalgebra::Dyn,
            nalgebra::Dyn,
        >>::Buffer<T>,
    >::from_iterator(x.shape().0, x.shape().1, x.into_iter().cloned())
}

/// Pad an [nalgebra] array.
pub(crate) fn pad<T, M, N>(
    padtype: Pad,
    mut padlen: Option<usize>,
    x: OMatrix<T, M, N>,
    axis: usize,
    ntaps: usize,
) -> Result<(usize, OMatrix<T, Dyn, Dyn>)>
where
    T: Scalar + Copy + Zero + One + Sub<Output = T>,
    M: Dim,
    N: Dim,
    DefaultAllocator: Allocator<M, N> + Allocator<Dyn, Dyn>,
{
    if matches!(padtype, Pad::None) {
        padlen = Some(0);
    }

    let edge = match padlen {
        Some(padlen) => padlen,
        None => ntaps * 3,
    };

    if axis >= 2 {
        return Err(Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        });
    }
    let shape = x.shape();
    let axis_len = if axis == 0 { shape.0 } else { shape.1 };
    if axis_len <= edge {
        return Err(Error::InvalidArg {
            arg: "padlen".into(),
            reason: "edge extension cannot exceed the selected axis length.".into(),
        });
    }

    if matches!(padtype, Pad::None) || edge == 0 {
        return Ok((edge, to_dyn_matrix(x)));
    }

    match padtype {
        Pad::Odd => Ok((edge, odd_ext_dyn(x, edge, axis)?)),
        Pad::Even | Pad::Constant => Err(Error::InvalidArg {
            arg: "padtype".into(),
            reason: "only odd extension padding is currently implemented.".into(),
        }),
        Pad::None => unreachable!(),
    }
}

/// Pad an [nalgebra] array with odd extension.
///
// This differs from [super::FiltFiltPadType]'s ext that acts on [ndarray].
pub(crate) fn odd_ext_dyn<T, M, N>(
    x: OMatrix<T, M, N>,
    n: usize,
    axis: usize,
) -> Result<OMatrix<T, Dyn, Dyn>>
where
    T: Scalar + Copy + Zero + One + Sub<Output = T>,
    M: Dim,
    N: Dim,
    DefaultAllocator: Allocator<M, N> + Allocator<Dyn, Dyn>,
{
    //TODO: Figure out the ndarray situation
    if axis >= 2 {
        return Err(Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        });
    }

    if n < 1 {
        return Ok(to_dyn_matrix(x));
    }

    let axis_len = if axis == 0 { x.shape().0 } else { x.shape().1 };
    if n >= axis_len {
        return Err(Error::InvalidArg {
            arg: "n".into(),
            reason: "extension length must be less than the selected axis length.".into(),
        });
    }

    let two = T::one() + T::one();
    match axis {
        0 => {
            // extend rows
            let (old_rows, old_columns) = x.shape();
            let (new_rows, new_columns) = (old_rows + 2 * n, old_columns);
            let mut m = nalgebra::Matrix::<
                T,
                nalgebra::Dyn,
                nalgebra::Dyn,
                <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
                    nalgebra::Dyn,
                    nalgebra::Dyn,
                >>::Buffer<T>,
            >::zeros(new_rows, new_columns);

            // 2 * left_end - left_ext
            let first_row = x.row(0);
            for j in 0..old_columns {
                for i in 0..n {
                    unsafe {
                        *m.get_unchecked_mut((i, j)) =
                            two * *first_row.index(j) - *x.get_unchecked((n - i, j));
                    }
                }
            }

            // x
            for j in 0..old_columns {
                for i in 0..old_rows {
                    unsafe {
                        *m.get_unchecked_mut((i + n, j)) = *x.get_unchecked((i, j));
                    }
                }
            }

            // 2 * right_end - right_ext
            let last_row_idx = old_rows - 1;
            let last_row = x.row(last_row_idx);
            for j in 0..old_columns {
                for i in 0..n {
                    unsafe {
                        *m.get_unchecked_mut((old_rows + 2 * n - 1 - i, j)) =
                            two * *last_row.index(j) - *x.get_unchecked((last_row_idx - n + i, j));
                    }
                }
            }

            Ok(m)
        }
        1 => {
            // extend columns
            let shape = x.shape();
            let (old_rows, old_columns) = x.shape();
            let (new_rows, new_columns) = (old_rows, shape.1 + 2 * n);
            let mut m = nalgebra::Matrix::<
                T,
                nalgebra::Dyn,
                nalgebra::Dyn,
                <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
                    nalgebra::Dyn,
                    nalgebra::Dyn,
                >>::Buffer<T>,
            >::zeros(new_rows, new_columns);

            // 2 * left_end - left_ext
            let first_col = x.column(0);
            for j in 0..n {
                for i in 0..old_rows {
                    unsafe {
                        *m.get_unchecked_mut((i, j)) =
                            two * *first_col.index(i) - *x.get_unchecked((i, n - j));
                    }
                }
            }

            // x
            for j in 0..old_columns {
                for i in 0..old_rows {
                    unsafe {
                        *m.get_unchecked_mut((i, j + n)) = *x.get_unchecked((i, j));
                    }
                }
            }

            // 2 * right_end - right_ext
            let last_col_idx = old_columns - 1;
            let last_col = x.column(last_col_idx);
            for j in 0..n {
                for i in 0..old_rows {
                    unsafe {
                        *m.get_unchecked_mut((i, old_columns + 2 * n - 1 - j)) =
                            two * *last_col.index(i) - *x.get_unchecked((i, last_col_idx - n + j));
                    }
                }
            }

            Ok(m)
        }
        _ => Err(Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn scipy_example_dyn() {
        // a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
        let a = nalgebra::matrix!(
            1,2,3,4,5;
            0,1,4,9,16;
        );

        // odd_ext(a, 1, axis = 0)
        let scipy_rows = nalgebra::matrix!(
            2,  3 , 2, -1, -6;
            1,  2 , 3,  4,  5;
            0,  1 , 4,  9, 16;
            -1,  0 , 5, 14, 27;
        );
        let row_ext = odd_ext_dyn(a, 1, 0).unwrap();
        assert_eq!(scipy_rows, row_ext);

        // odd_ext(a, 2)
        let scipy_cols = nalgebra::matrix!(
            -1,  0,  1,  2 , 3  ,4  ,5  ,6  ,7;
            -4, -1,  0,  1 , 4  ,9 ,16 ,23 ,28
        );
        let col_ext = odd_ext_dyn(a, 2, 1).unwrap();
        assert_eq!(scipy_cols, col_ext);

        let a = nalgebra::matrix!(
            1,2,3,4,5;
            0,1,4,9,16;
            0,1,8,27,64;
        );

        // odd_ext(a, 1, axis = 0)
        let scipy_rows = nalgebra::matrix!(
            2   ,3  ,-2 ,-19 ,-54;
            2   ,3   ,2  ,-1  ,-6;
            1   ,2   ,3   ,4   ,5;
            0   ,1   ,4   ,9  ,16;
            0   ,1   ,8  ,27  ,64;
            0   ,1  ,12  ,45 ,112;
        -1 ,  0 , 13 , 50, 123;
        );
        let row_ext = odd_ext_dyn(a, 2, 0).unwrap();
        assert_eq!(scipy_rows, row_ext);
    }
}
