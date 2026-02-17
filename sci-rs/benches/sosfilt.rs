use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dasp_signal::{rate, Signal};
use sci_rs::kernel::KernelLifecycle;
use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::{SosFiltConfig, SosFiltKernel};
use sci_rs::signal::traits::SosFilt1D;

fn butter_sosfilt_100x_dyn(c: &mut Criterion) {
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
    let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
        sos: Sos::from_scipy_dyn(4, filter.to_vec()),
    })
    .expect("valid sosfilt kernel config");

    // A signal with a frequency that we can recover
    let sample_hz = 1666.;
    let seconds = 10;
    let mut signal = rate(sample_hz).const_hz(25.).sine();
    let sin_wave: Vec<f64> = (0..seconds * sample_hz as usize)
        .map(|_| signal.next())
        .collect::<Vec<_>>();
    let sin_wave = (0..100).flat_map(|_| sin_wave.clone()).collect::<Vec<_>>();

    c.bench_function("sosfilt_100x_dyn", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .run_alloc(sin_wave.as_slice())
                    .expect("benchmark input should satisfy sosfilt preconditions"),
            );
        });
    });
}

fn butter_sosfilt_f64(c: &mut Criterion) {
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
    let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
        sos: Sos::from_scipy_dyn(4, filter.to_vec()),
    })
    .expect("valid sosfilt kernel config");

    // A signal with a frequency that we can recover
    let sample_hz = 1666.;
    let seconds = 10;
    let mut signal = rate(sample_hz).const_hz(25.).sine();
    let sin_wave: Vec<f64> = (0..seconds * sample_hz as usize)
        .map(|_| signal.next())
        .collect::<Vec<_>>();

    c.bench_function("sosfilt_f64", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .run_alloc(sin_wave.as_slice())
                    .expect("benchmark input should satisfy sosfilt preconditions"),
            );
        });
    });
}

fn butter_sosfilt_f32(c: &mut Criterion) {
    // 4th order butterworth bandpass 10 to 50 at 1666Hz
    let filter: [f32; 24] = [
        2.677_576_8e-5,
        5.355_153_6e-5,
        2.677_576_8e-5,
        1.0,
        -1.799_120_2,
        0.816_257_83,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.877_476_9,
        0.909_430_27,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.923_795_9,
        0.926_379_44,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.978_497_3,
        0.979_989_47,
    ];
    let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
        sos: Sos::from_scipy_dyn(4, filter.to_vec()),
    })
    .expect("valid sosfilt kernel config");

    // A signal with a frequency that we can recover
    let sample_hz = 1666.;
    let seconds = 10;
    let mut signal = rate(sample_hz).const_hz(25.).sine();
    let sin_wave: Vec<f32> = (0..seconds * sample_hz as usize)
        .map(|_| signal.next() as f32)
        .collect::<Vec<_>>();

    c.bench_function("sosfilt_f32", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .run_alloc(sin_wave.as_slice())
                    .expect("benchmark input should satisfy sosfilt preconditions"),
            );
        });
    });
}

fn butter_sosfilt_f32_order4_trait(c: &mut Criterion) {
    // 4th order butterworth bandpass 10 to 50 at 1666Hz
    let filter: [f32; 24] = [
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
    let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
        sos: Sos::from_scipy_dyn(4, filter.to_vec()),
    })
    .expect("valid sosfilt kernel config");

    // A signal with a frequency that we can recover
    let sample_hz = 1666.;
    let seconds = 10;
    let mut signal = rate(sample_hz).const_hz(25.).sine();
    let sin_wave: Vec<f32> = (0..seconds * sample_hz as usize)
        .map(|_| signal.next() as f32)
        .collect::<Vec<_>>();

    c.bench_function("sosfilt_f32_order4_trait", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .run_alloc(sin_wave.as_slice())
                    .expect("benchmark input should satisfy sosfilt preconditions"),
            );
        });
    });
}

fn butter_sosfilt_f32_order8_trait(c: &mut Criterion) {
    // 8th order butterworth bandpass 10 to 50 at 1666Hz
    let filter: [f32; 48] = [
        7.223657016655901e-10,
        1.4447314033311803e-09,
        7.223657016655901e-10,
        1.0,
        -1.8117367715812775,
        0.82390242865626,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.8027320089797896,
        0.8243217191895463,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.8436432946399057,
        0.8727119112095707,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.898284650388357,
        0.9018968456559892,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.9415418844285526,
        0.943617934088245,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.918327282768592,
        0.9524499384015938,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.967653721895735,
        0.9692546426434141,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.9886761584321624,
        0.9901117398066808,
    ];
    let mut kernel = SosFiltKernel::try_new(SosFiltConfig {
        sos: Sos::from_scipy_dyn(8, filter.to_vec()),
    })
    .expect("valid sosfilt kernel config");

    // A signal with a frequency that we can recover
    let sample_hz = 1666.;
    let seconds = 10;
    let mut signal = rate(sample_hz).const_hz(25.).sine();
    let sin_wave: Vec<f32> = (0..seconds * sample_hz as usize)
        .map(|_| signal.next() as f32)
        .collect::<Vec<_>>();

    c.bench_function("sosfilt_f32_order8_trait", |b| {
        b.iter(|| {
            black_box(
                kernel
                    .run_alloc(sin_wave.as_slice())
                    .expect("benchmark input should satisfy sosfilt preconditions"),
            );
        });
    });
}

criterion_group!(
    benches,
    butter_sosfilt_100x_dyn,
    butter_sosfilt_f64,
    butter_sosfilt_f32,
    butter_sosfilt_f32_order4_trait,
    butter_sosfilt_f32_order8_trait,
);
criterion_main!(benches);
