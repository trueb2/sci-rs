use super::ConfigError;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Adapter trait for reading contiguous 1D input.
pub trait Read1D<T> {
    /// Borrow the underlying input as a contiguous slice.
    fn read_slice(&self) -> Result<&[T], ConfigError>;
}

/// Adapter trait for writing contiguous 1D output.
pub trait Write1D<T> {
    /// Borrow the underlying output as a mutable contiguous slice.
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError>;
}

/// Stream adapter for iterator-like sample sources.
pub trait SampleStream<T> {
    /// Get the next sample from the stream.
    fn next_sample(&mut self) -> Option<T>;
}

impl<T> Read1D<T> for [T] {
    fn read_slice(&self) -> Result<&[T], ConfigError> {
        Ok(self)
    }
}

impl<T> Write1D<T> for [T] {
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError> {
        Ok(self)
    }
}

impl<T, const N: usize> Read1D<T> for [T; N] {
    fn read_slice(&self) -> Result<&[T], ConfigError> {
        Ok(self)
    }
}

impl<T, const N: usize> Write1D<T> for [T; N] {
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError> {
        Ok(self)
    }
}

#[cfg(feature = "alloc")]
impl<T> Read1D<T> for Vec<T> {
    fn read_slice(&self) -> Result<&[T], ConfigError> {
        Ok(self.as_slice())
    }
}

#[cfg(feature = "alloc")]
impl<T> Write1D<T> for Vec<T> {
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError> {
        Ok(self.as_mut_slice())
    }
}

#[cfg(feature = "alloc")]
impl<T> Read1D<T> for Array1<T> {
    fn read_slice(&self) -> Result<&[T], ConfigError> {
        self.as_slice()
            .ok_or(ConfigError::NonContiguous { arg: "array" })
    }
}

#[cfg(feature = "alloc")]
impl<T> Write1D<T> for Array1<T> {
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError> {
        self.as_slice_mut()
            .ok_or(ConfigError::NonContiguous { arg: "array" })
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> Read1D<T> for ArrayView1<'a, T> {
    fn read_slice(&self) -> Result<&[T], ConfigError> {
        self.as_slice()
            .ok_or(ConfigError::NonContiguous { arg: "array_view" })
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> Write1D<T> for ArrayViewMut1<'a, T> {
    fn write_slice_mut(&mut self) -> Result<&mut [T], ConfigError> {
        self.as_slice_mut().ok_or(ConfigError::NonContiguous {
            arg: "array_view_mut",
        })
    }
}

impl<I, T> SampleStream<T> for I
where
    I: Iterator<Item = T>,
{
    fn next_sample(&mut self) -> Option<T> {
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::{Read1D, Write1D};

    #[test]
    fn slice_and_array_adapters() {
        let a = [1.0f32, 2.0, 3.0];
        assert_eq!(a.read_slice().expect("array adapter").len(), 3);

        let s: &[f32] = &a;
        assert_eq!(s.read_slice().expect("slice adapter")[1], 2.0);
    }

    #[test]
    fn vec_write_adapter() {
        let mut out = vec![0.0f32; 4];
        let slice = out.write_slice_mut().expect("vec write adapter");
        slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn ndarray_adapters() {
        use ndarray::Array1;

        let arr = Array1::from(vec![1.0f64, 2.0, 3.0]);
        assert_eq!(arr.read_slice().expect("array1 read")[2], 3.0);

        let mut out = Array1::from(vec![0.0f64, 0.0, 0.0]);
        out.write_slice_mut()
            .expect("array1 write")
            .copy_from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(out.as_slice().expect("slice"), &[4.0, 5.0, 6.0]);
    }
}
