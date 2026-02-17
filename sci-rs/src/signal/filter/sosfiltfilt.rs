use core::{borrow::Borrow, cmp::min, iter::Sum, ops::SubAssign};
use nalgebra::{DVector, RealField, Scalar};
use num_traits::{Float, One, Zero};
use sci_rs_core::{Error, Result};

use super::{design::Sos, pad, sosfilt_checked_slice, sosfilt_zi_checked_slice, Pad};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

///
/// A forward-backward digital filter using cascaded second-order sections
///
/// <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt>
///
///
#[inline]
pub fn sosfiltfilt_checked_slice<F>(y: &[F], sos: &[Sos<F>]) -> Result<Vec<F>>
where
    F: RealField + Copy + PartialEq + Scalar + Zero + One + Sum + SubAssign,
{
    let n = sos.len();
    let ntaps = 2 * n + 1;
    let bzeros = sos.iter().filter(|s| s.b[2] == F::zero()).count();
    let azeros = sos.iter().filter(|s| s.a[2] == F::zero()).count();
    let ntaps = ntaps - min(bzeros, azeros);
    if y.is_empty() {
        return Err(Error::InvalidArg {
            arg: "y".into(),
            reason: "input must be non-empty.".into(),
        });
    }
    let y_len = y.len();
    let x = DVector::<F>::from_column_slice(y);
    let (edge, ext) = pad(Pad::Odd, None, x, 0, ntaps)?;

    let mut init_sos = sos.to_vec();
    sosfilt_zi_checked_slice(init_sos.as_mut_slice())?;

    let x0 = *ext.index(0);
    let mut sos_x = init_sos.clone();
    for s in sos_x.iter_mut() {
        s.zi0 *= x0;
        s.zi1 *= x0;
    }
    let y = sosfilt_checked_slice(ext.as_slice(), sos_x.as_mut_slice())?;

    let y0 = *y.last().ok_or(Error::InvalidArg {
        arg: "y".into(),
        reason: "input must be non-empty.".into(),
    })?;
    let mut sos_y = init_sos;
    for s in sos_y.iter_mut() {
        s.zi0 *= y0;
        s.zi1 *= y0;
    }
    let mut y_rev = y;
    y_rev.reverse();
    let mut z = sosfilt_checked_slice(y_rev.as_slice(), sos_y.as_mut_slice())?;
    z = z.into_iter().skip(edge).take(y_len).collect::<Vec<_>>();
    z.reverse();
    Ok(z)
}

///
/// Checked `sosfiltfilt` adapter for iterator-like inputs.
///
#[inline]
pub fn sosfiltfilt_checked<YI, F>(y: YI, sos: &[Sos<F>]) -> Result<Vec<F>>
where
    F: RealField + Copy + PartialEq + Scalar + Zero + One + Sum + SubAssign,
    YI: IntoIterator,
    YI::Item: Borrow<F>,
{
    let y = y.into_iter().map(|yi| *yi.borrow()).collect::<Vec<F>>();
    sosfiltfilt_checked_slice(&y, sos)
}

///
/// A forward-backward digital filter using cascaded second-order sections.
///
#[inline]
pub fn sosfiltfilt_dyn<YI, F>(y: YI, sos: &[Sos<F>]) -> Result<Vec<F>>
where
    F: RealField + Copy + PartialEq + Scalar + Zero + One + Sum + SubAssign,
    YI: IntoIterator,
    YI::Item: Borrow<F>,
{
    sosfiltfilt_checked(y, sos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dasp_signal::{rate, Signal};

    #[cfg(feature = "std")]
    #[test]
    fn can_sosfiltfilt() {
        // 4th order butterworth bandpass 10 to 50 at 1666Hz
        let filter: [f64; 24] = [
            2.677_576_738_259_783_5e-5,
            5.355_153_476_519_567e-5,
            2.677_576_738_259_783_5e-5,
            1.0,
            -1.7991202154617734,
            0.8162578614819005,
            1.0,
            2.0,
            1.0,
            1.0,
            -1.8774769894419825,
            0.9094302413068086,
            1.0,
            -2.0,
            1.0,
            1.0,
            -1.9237959892866103,
            0.9263794671616161,
            1.0,
            -2.0,
            1.0,
            1.0,
            -1.978497311228862,
            0.9799894886973378,
        ];
        let sos = Sos::from_scipy_dyn(4, filter.to_vec());
        assert_eq!(sos.len(), 4);

        // A signal with a frequency that we can recover
        let sample_hz = 1666.;
        let seconds = 10;
        let mut signal = rate(sample_hz).const_hz(25.).sine();
        let sin_wave: Vec<f64> = (0..seconds * sample_hz as usize)
            .map(|_| signal.next())
            .collect::<Vec<_>>();
        // println!("{:?}", &sin_wave);

        let bp_wave = sosfiltfilt_dyn(sin_wave.iter(), &sos).unwrap();
        println!("{:?}", bp_wave);
        assert_eq!(sin_wave.len(), bp_wave.len());

        println!("{:?}", &bp_wave[..10]);
        println!("{:?}", &sin_wave[..10]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn can_sosfiltfilt_f32() {
        // 4th order butterworth bandpass 10 to 50 at 1666Hz
        let filter: [f32; 24] = [
            2.677_576_8e-5_f32,
            5.355_153_6e-5_f32,
            2.677_576_8e-5_f32,
            1.0_f32,
            -1.799_120_2_f32,
            0.816_257_83_f32,
            1.0_f32,
            2.0_f32,
            1.0_f32,
            1.0_f32,
            -1.877_476_9_f32,
            0.909_430_27_f32,
            1.0_f32,
            -2.0_f32,
            1.0_f32,
            1.0_f32,
            -1.923_795_9_f32,
            0.926_379_44_f32,
            1.0_f32,
            -2.0_f32,
            1.0_f32,
            1.0_f32,
            -1.978_497_3_f32,
            0.979_989_47_f32,
        ];
        let sos = Sos::<f32>::from_scipy_dyn(4, filter.to_vec());
        assert_eq!(sos.len(), 4);

        // A signal with a frequency that we can recover
        let sample_hz = 1666.;
        let seconds = 10;
        let mut signal = rate(sample_hz).const_hz(25.).sine();
        let sin_wave: Vec<f32> = (0..seconds * sample_hz as usize)
            .map(|_| signal.next() as f32)
            .collect::<Vec<_>>();
        // println!("{:?}", &sin_wave);

        let bp_wave = sosfiltfilt_dyn(sin_wave.iter(), &sos).unwrap();
        assert_eq!(sin_wave.len(), bp_wave.len());
        println!("{:?}", bp_wave);

        println!("{:?}", &bp_wave[..10]);
        println!("{:?}", &sin_wave[..10]);
    }
}
