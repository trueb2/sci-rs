use crate::kernel::KernelLifecycle;
use crate::signal::traits::{ChirpWave1D, SawtoothWave1D, SquareWave1D, UnitImpulse1D};
use nalgebra::RealField;
use ndarray::{Array, Array1, ArrayBase, Data, Dimension};
use num_traits::FromPrimitive;

mod kernels;
pub use kernels::*;

/// Return a periodic square-wave waveform.
///
/// The square wave has period `2*pi`, has value `+1` from `0` to
/// `2*pi*duty`, and `-1` from `2*pi*duty` to `2*pi`.
pub fn square<F, S, D>(t: &ArrayBase<S, D>, duty: F) -> Array<F, D>
where
    F: RealField + Copy,
    S: Data<Elem = F>,
    D: Dimension,
{
    #[cfg(feature = "alloc")]
    {
        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty })
            .expect("duty must be in [0, 1] for square wave generation");
        let flat_t = t.iter().copied().collect::<alloc::vec::Vec<_>>();
        let flat_y = kernel
            .run_alloc(&flat_t)
            .expect("square wave generation failed");
        Array::from_shape_vec(t.raw_dim(), flat_y)
            .expect("square wave output shape conversion failed")
    }

    #[cfg(not(feature = "alloc"))]
    {
        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty })
            .expect("duty must be in [0, 1] for square wave generation");
        t.mapv(|v| kernel.sample(v))
    }
}

/// Return a periodic sawtooth waveform.
///
/// The waveform has period `2*pi`, rises from `-1` to `1` over
/// `[0, width*2*pi)`, and falls from `1` to `-1` over `[width*2*pi, 2*pi)`.
pub fn sawtooth<F, S, D>(t: &ArrayBase<S, D>, width: F) -> Array<F, D>
where
    F: RealField + Copy,
    S: Data<Elem = F>,
    D: Dimension,
{
    #[cfg(feature = "alloc")]
    {
        let kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width })
            .expect("width must be in [0, 1] for sawtooth generation");
        let flat_t = t.iter().copied().collect::<alloc::vec::Vec<_>>();
        let flat_y = kernel
            .run_alloc(&flat_t)
            .expect("sawtooth wave generation failed");
        Array::from_shape_vec(t.raw_dim(), flat_y)
            .expect("sawtooth wave output shape conversion failed")
    }

    #[cfg(not(feature = "alloc"))]
    {
        let kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width })
            .expect("width must be in [0, 1] for sawtooth generation");
        t.mapv(|v| kernel.sample(v))
    }
}

/// Return a cosine chirp waveform.
///
/// The output is `cos(phase + phi)`, where `phase` is the integral of
/// instantaneous frequency according to `method`.
pub fn chirp<F, S, D>(
    t: &ArrayBase<S, D>,
    f0: F,
    t1: F,
    f1: F,
    method: ChirpMethod,
    phi_deg: F,
    vertex_zero: bool,
) -> Array<F, D>
where
    F: RealField + Copy + FromPrimitive,
    S: Data<Elem = F>,
    D: Dimension,
{
    #[cfg(feature = "alloc")]
    {
        let kernel = ChirpKernel::try_new(ChirpConfig {
            f0,
            t1,
            f1,
            method,
            phi_deg,
            vertex_zero,
        })
        .expect("invalid chirp config");
        let flat_t = t.iter().copied().collect::<alloc::vec::Vec<_>>();
        let flat_y = kernel.run_alloc(&flat_t).expect("chirp generation failed");
        Array::from_shape_vec(t.raw_dim(), flat_y).expect("chirp output shape conversion failed")
    }

    #[cfg(not(feature = "alloc"))]
    {
        let kernel = ChirpKernel::try_new(ChirpConfig {
            f0,
            t1,
            f1,
            method,
            phi_deg,
            vertex_zero,
        })
        .expect("invalid chirp config");
        t.mapv(|v| kernel.sample(v))
    }
}

/// Return a 1D unit impulse (Kronecker delta).
///
/// `len` defines output length. `idx` selects the index whose value is `1`;
/// when omitted, index `0` is used.
pub fn unit_impulse<F>(len: usize, idx: Option<usize>) -> Array1<F>
where
    F: RealField + Copy,
{
    let kernel = UnitImpulseKernel::try_new(UnitImpulseConfig {
        len,
        idx: idx.unwrap_or(0),
    })
    .expect("invalid unit impulse config");

    let mut out = Array1::from_elem(len, F::zero());
    kernel
        .run_into(&mut out)
        .expect("unit impulse generation failed");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr3};

    #[test]
    fn test_square_zero_duty() {
        let t = arr1(&[
            -4.821, -4.15, -3.394, -3.386, -2.966, -2.735, -2.464, -2.277, -2.094, -0.8963,
            0.03853, 1.432, 2.384, 2.522, 2.732, 3.125, 3.297, 3.517, 3.602, 4.908,
        ]);
        let expected = arr1(&[
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ]);
        let result = square(&t, 0.0);
        assert_vec_eq_f32(result, expected);
    }

    #[test]
    fn test_square_one_duty() {
        let t = arr1(&[
            -3.521, -3.284, -3.257, -2.367, -1.965, -1.933, -0.5399, -0.4277, -0.3761, 0.3024,
            0.3624, 0.6161, 0.784, 1.42, 1.869, 1.934, 4.0, 4.298, 4.389, 4.807,
        ]);
        let expected = arr1(&[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        let result = square(&t, 1.0);
        assert_vec_eq_f32(result, expected);
    }

    #[test]
    fn test_square_duty_03_05_07() {
        let t = arr1(&[
            -4.991, -4.973, -3.988, -3.084, -2.562, -2.378, -1.618, -1.449, -0.8056, -0.6883,
            -0.5677, -0.5353, -0.1377, -0.1142, 0.8072, 0.821, 1.836, 2.722, 4.189, 4.384,
        ]);

        let expected_03 = arr1(&[
            1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, 1.0, -1.0, -1.0, -1.0,
        ]);
        let expected_05 = arr1(&[
            1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, 1.0, 1.0, -1.0, -1.0,
        ]);
        let expected_07 = arr1(&[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]);
        let result_03 = square(&t, 0.3);
        let result_05 = square(&t, 0.5);
        let result_07 = square(&t, 0.7);
        assert_vec_eq_f32(result_03, expected_03);
        assert_vec_eq_f32(result_05, expected_05);
        assert_vec_eq_f32(result_07, expected_07);
    }

    #[test]
    fn test_square_3d() {
        let t = arr3(&[
            [
                [-4.452, -4.182, -3.663, -3.307, -2.995],
                [-2.482, -2.46, -1.929, -1.823, -1.44],
            ],
            [
                [-0.8743, 0.5359, 0.9073, 2.101, 2.161],
                [2.582, 2.977, 3.966, 4.298, 4.659],
            ],
        ]);
        let expected = arr3(&[
            [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0, -1.0]],
            [[-1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0, -1.0]],
        ]);
        let result = square(&t, 0.67);
        assert_vec_eq_f32(result, expected);
    }

    #[test]
    fn test_sawtooth_width_0_03_05_10() {
        let t = arr1(&[-4.0, -1.2, -0.1, 0.0, 0.3, 1.7, 3.14, 4.2, 6.1, 7.9]);
        let expected_w0 = arr1(&[
            0.2732395447352,
            -0.6180281365795,
            -0.9681690113816,
            1.0,
            0.9045070341449,
            0.4588731934876,
            0.0005069573829,
            -0.3369015219719,
            -0.9416903057211,
            0.4853518991481,
        ]);
        let expected_w03 = arr1(&[
            0.818913635336,
            -0.454325909399,
            -0.954527159117,
            -1.0,
            -0.681690113816,
            0.803756021708,
            0.429295653404,
            -0.05271645996,
            -0.916700436744,
            0.715493669506,
        ]);
        let expected_w05 = arr1(&[
            0.45352091053,
            -0.236056273159,
            -0.936338022763,
            -1.0,
            -0.80901406829,
            0.082253613025,
            0.998986085234,
            0.326196956056,
            -0.883380611442,
            0.029296201704,
        ]);
        let expected_w1 = arr1(&[
            -0.2732395447352,
            0.6180281365795,
            0.9681690113816,
            -1.0,
            -0.9045070341449,
            -0.4588731934876,
            -0.0005069573829,
            0.3369015219719,
            0.9416903057211,
            -0.4853518991481,
        ]);

        assert_vec_eq_f64(sawtooth(&t, 0.0), expected_w0, 1e-9);
        assert_vec_eq_f64(sawtooth(&t, 0.3), expected_w03, 1e-9);
        assert_vec_eq_f64(sawtooth(&t, 0.5), expected_w05, 1e-9);
        assert_vec_eq_f64(sawtooth(&t, 1.0), expected_w1, 1e-9);
    }

    #[test]
    fn test_chirp_methods_against_known_reference_values() {
        let t = arr1(&[0.0, 0.25, 0.5, 1.0, 1.5, 2.0]);

        let expected_linear = arr1(&[
            0.965925826289,
            -0.849202181527,
            0.13052619222,
            0.258819045103,
            -0.13052619222,
            0.965925826289,
        ]);
        let expected_quadratic = arr1(&[
            0.965925826289,
            -0.677598304996,
            -0.751839807479,
            -0.258819045103,
            -0.896872741533,
            0.965925826289,
        ]);
        let expected_log = arr1(&[
            0.965925826289,
            -0.900975876087,
            0.506478779795,
            -0.880553288116,
            -0.615287915617,
            -0.844978171921,
        ]);
        let expected_hyp = arr1(&[
            0.965925826289,
            -0.926478064699,
            0.706545882009,
            -0.874787746575,
            0.985381739166,
            0.586404074679,
        ]);

        assert_vec_eq_f64(
            chirp(&t, 2.0, 2.0, 5.0, ChirpMethod::Linear, 15.0, false),
            expected_linear,
            1e-9,
        );
        assert_vec_eq_f64(
            chirp(&t, 2.0, 2.0, 5.0, ChirpMethod::Quadratic, 15.0, false),
            expected_quadratic,
            1e-9,
        );
        assert_vec_eq_f64(
            chirp(&t, 2.0, 2.0, 5.0, ChirpMethod::Logarithmic, 15.0, false),
            expected_log,
            1e-9,
        );
        assert_vec_eq_f64(
            chirp(&t, 2.0, 2.0, 5.0, ChirpMethod::Hyperbolic, 15.0, false),
            expected_hyp,
            1e-9,
        );
    }

    #[test]
    fn test_unit_impulse_default_and_offset_index() {
        let default = unit_impulse::<f64>(7, None);
        assert_eq!(default.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let shifted = unit_impulse::<f64>(7, Some(2));
        assert_eq!(shifted.to_vec(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let centered = unit_impulse::<f64>(8, Some(8 / 2));
        assert_eq!(
            centered.to_vec(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    #[track_caller]
    fn assert_vec_eq_f32<D: Dimension>(a: Array<f32, D>, b: Array<f32, D>) {
        assert_eq!(a.shape(), b.shape());
        for (a, b) in a.into_iter().zip(b) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[track_caller]
    fn assert_vec_eq_f64<D: Dimension>(a: Array<f64, D>, b: Array<f64, D>, epsilon: f64) {
        assert_eq!(a.shape(), b.shape());
        for (a, b) in a.into_iter().zip(b) {
            assert_abs_diff_eq!(a, b, epsilon = epsilon);
        }
    }
}
