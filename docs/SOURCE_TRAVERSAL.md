# Source Traversal Log

Last updated: 2026-02-17

Status legend:
- `refactored`: trait-first/kernel-first path is implemented and tested.
- `partial`: mixed state; trait-first path exists but legacy surface still primary in places.
- `pending`: not yet migrated to trait-first conventions.
- `support`: module is utility/export/error infrastructure.

## sci-rs/src

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs/src/lib.rs` | support | crate exports and feature gating |
| `sci-rs/src/error/mod.rs` | support | shared error type behavior updated |
| `sci-rs/src/plot.rs` | support | non-blocking Agg plotting path |
| `sci-rs/src/kernel/mod.rs` | refactored | kernel substrate module |
| `sci-rs/src/kernel/errors.rs` | refactored | `ConfigError` / `ExecInvariantViolation` |
| `sci-rs/src/kernel/io.rs` | refactored | adapter traits + tests |
| `sci-rs/src/kernel/lifecycle.rs` | refactored | constructor lifecycle + tests |
| `sci-rs/src/linalg/mod.rs` | partial | companion trait path exported |
| `sci-rs/src/linalg/companion.rs` | refactored | `CompanionKernel` + checked companion wrapper path |
| `sci-rs/src/stats.rs` | partial | trait-first kernels added; free functions retained |
| `sci-rs/src/special/mod.rs` | support | special-function exports |
| `sci-rs/src/special/bessel.rs` | support | existing trait surface |
| `sci-rs/src/special/combinatorics.rs` | support | existing trait surface |
| `sci-rs/src/special/factorial.rs` | support | existing trait surface |
| `sci-rs/src/special/xsf/mod.rs` | support | special internals |
| `sci-rs/src/special/xsf/chbevl.rs` | support | numeric helper |
| `sci-rs/src/special/xsf/i0.rs` | support | bessel implementations |

## sci-rs/src/signal

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs/src/signal/mod.rs` | partial | mixed legacy exports with trait-first kernels |
| `sci-rs/src/signal/traits.rs` | refactored | signal trait-first contract surface |
| `sci-rs/src/signal/convolve.rs` | refactored | kernel-first + legacy shim routing |
| `sci-rs/src/signal/resample.rs` | refactored | kernel-first + legacy shim routing |
| `sci-rs/src/signal/wave/mod.rs` | refactored | `square`/`sawtooth`/`chirp`/`gausspulse`/`sweep_poly` route through waveform kernels; 1D `unit_impulse` shim added |
| `sci-rs/src/signal/wave/kernels.rs` | refactored | waveform kernels: `SquareWaveKernel`, `SawtoothWaveKernel`, `ChirpKernel`, `GaussPulseKernel`, `SweepPolyKernel`, `UnitImpulseKernel` |

## sci-rs/src/signal/windows

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs/src/signal/windows/mod.rs` | partial | legacy builders + kernel exports |
| `sci-rs/src/signal/windows/kernels.rs` | refactored | `WindowKernel` + owned builder |
| `sci-rs/src/signal/windows/boxcar.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/triangle.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/blackman.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/hamming.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/nuttall.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/kaiser.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/general_cosine.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/general_gaussian.rs` | partial | legacy get_window implementation |
| `sci-rs/src/signal/windows/general_hamming.rs` | partial | legacy get_window implementation |

## sci-rs/src/signal/filter

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs/src/signal/filter/mod.rs` | partial | mixed exports |
| `sci-rs/src/signal/filter/kernels.rs` | refactored | main filtering kernels and zi/savgol kernels |
| `sci-rs/src/signal/filter/lfilter.rs` | partial | free-function shim has kernel-first 1D fast path |
| `sci-rs/src/signal/filter/filtfilt.rs` | partial | added kernel-backed `filtfilt_dyn` compatibility wrapper |
| `sci-rs/src/signal/filter/sosfilt.rs` | partial | checked wrappers (`sosfilt_checked`, `sosfilt_item_checked`) |
| `sci-rs/src/signal/filter/sosfiltfilt.rs` | partial | legacy free function still primary |
| `sci-rs/src/signal/filter/lfilter_zi.rs` | partial | checked wrapper (`lfilter_zi_checked`) + kernel path |
| `sci-rs/src/signal/filter/sosfilt_zi.rs` | partial | checked wrapper (`sosfilt_zi_checked`) + kernel path |
| `sci-rs/src/signal/filter/savgol_filter.rs` | partial | checked wrappers + trait kernels for filter/coeff design |
| `sci-rs/src/signal/filter/ext.rs` | partial | moved to checked `Result` API for pad/odd extension; no trait kernel yet |
| `sci-rs/src/signal/filter/arraytools.rs` | partial | checked `AxisSliceKernel`/`AxisReverseKernel` landed; legacy helpers retained |

## sci-rs/src/signal/filter/design

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs/src/signal/filter/design/mod.rs` | partial | mixed exports + kernel exports |
| `sci-rs/src/signal/filter/design/kernels.rs` | refactored | trait-first kernels for `firwin/iirfilter/butter` + zpk helpers; checked execution routing |
| `sci-rs/src/signal/filter/design/firwin.rs` | partial | legacy function retained |
| `sci-rs/src/signal/filter/design/iirfilter.rs` | partial | checked `iirfilter_checked` added; `iirfilter_dyn` is compatibility wrapper |
| `sci-rs/src/signal/filter/design/butter.rs` | partial | checked `butter_checked` added; `butter_dyn` is compatibility wrapper |
| `sci-rs/src/signal/filter/design/filter_output.rs` | support | model types |
| `sci-rs/src/signal/filter/design/filter_type.rs` | support | enum types |
| `sci-rs/src/signal/filter/design/sos.rs` | support | SOS representation |
| `sci-rs/src/signal/filter/design/kaiser.rs` | partial | legacy helpers |
| `sci-rs/src/signal/filter/design/bilinear_zpk.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/cplx.rs` | partial | checked `cplxreal_checked` added; kernel uses checked path |
| `sci-rs/src/signal/filter/design/lp2bp_zpk.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/lp2bs_zpk.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/lp2hp_zpk.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/lp2lp_zpk.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/relative_degree.rs` | partial | checked `relative_degree_checked` added; kernel uses checked path |
| `sci-rs/src/signal/filter/design/zpk2sos.rs` | partial | legacy helper retained; validated kernel wrapper landed |
| `sci-rs/src/signal/filter/design/zpk2tf.rs` | partial | legacy helper retained; validated kernel wrapper landed |

## sci-rs-core/src

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs-core/src/lib.rs` | support | core error/result |
| `sci-rs-core/src/num_rs/mod.rs` | support | module exports |
| `sci-rs-core/src/num_rs/convolve/mod.rs` | partial | foundational convolution impl |
| `sci-rs-core/src/num_rs/convolve/ndarray_conv_binds.rs` | support | ndarray bindings |

## sci-rs-test/src

| File | Status | Notes |
| --- | --- | --- |
| `sci-rs-test/src/lib.rs` | support | smoke test crate |
