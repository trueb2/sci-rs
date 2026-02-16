# sci-rs Trait-First Refactor Checklist

This checklist tracks interface migration status for the breaking refactor.

Legend:
- `planned`: interface is on the refactor path.
- `refactored`: trait/config/kernel implementation exists.
- `tested`: interface has explicit tests for behavior and adapter coverage.

| Area | Interface | planned | refactored | tested | Notes |
| --- | --- | --- | --- | --- | --- |
| substrate | `Read1D<T>` | yes | yes | yes | slice/array/vec/ndarray adapters |
| substrate | `Write1D<T>` | yes | yes | yes | slice/array/vec/ndarray adapters |
| substrate | `SampleStream<T>` | yes | yes | yes | iterator blanket impl + stream tests |
| substrate | `KernelLifecycle` | yes | yes | yes | constructor contract tests |
| signal | `Convolve1D<T>` | yes | yes | yes | `ConvolveKernel` |
| signal | `Correlate1D<T>` | yes | yes | yes | `CorrelateKernel` |
| signal | `Resample1D<T>` | yes | yes | yes | `ResampleKernel` |
| signal | `SquareWave1D<T>` | yes | yes | yes | `SquareWaveKernel` |
| signal | `SavgolFilter1D<T>` | yes | yes | yes | `SavgolFilterKernel` |
| signal | `SosFilt1D<T>` | yes | yes | yes | `SosFiltKernel` |
| signal | `SosFiltFilt1D<T>` | yes | yes | yes | kernel parity and invariant tests |
| signal | `LFilter1D<T>` | yes | yes | yes | reference parity + length checks |
| signal | `FiltFilt1D<T>` | yes | yes | yes | reference parity + length checks |
| design | `FirWinDesign<T>` | yes | yes | yes | `FirWinKernel` with constructor validation |
| design | `IirDesign<T>` | yes | yes | yes | `IirFilterKernel` and `ButterKernel` |
| windows | `WindowGenerate<T>` | yes | yes | yes | `WindowKernel` and owned window builder |
| migration | legacy free functions cleanup | yes | partial | partial | `convolve`/`correlate`/`resample` shimmed |
| contract | local contract runner (`xtask`) | yes | yes | yes | local-only artifacts under `target/contracts` |

## Legacy Shim Status

| Legacy API | Trait-first replacement | shimmed | tested |
| --- | --- | --- | --- |
| `convolve` | `ConvolveKernel` + `Convolve1D` | yes | yes |
| `correlate` | `CorrelateKernel` + `Correlate1D` | yes | yes |
| `resample` | `ResampleKernel` + `Resample1D` | yes | yes |
| `lfilter` | `LFilterKernel` + `LFilter1D` | partial | yes |
| `filtfilt` | `FiltFiltKernel` + `FiltFilt1D` | partial | yes |
| `sosfilt` | `SosFiltKernel` + `SosFilt1D` | partial | yes |
| `sosfiltfilt` | `SosFiltFiltKernel` + `SosFiltFilt1D` | partial | yes |

## Remaining Public Interface Sweep

This is the active line-by-line sweep list for remaining public free-function surfaces.

| Module | Public API | trait-first status | Notes |
| --- | --- | --- | --- |
| `signal/filter` | `lfilter` | partial | kernel exists; ndarray multi-axis API still legacy-first |
| `signal/filter` | `sosfilt_dyn` / `sosfilt_item` / `sosfilt_st` | partial | kernels exist; low-level helpers still direct |
| `signal/filter` | `sosfiltfilt_dyn` | partial | kernel exists; free function still primary helper |
| `signal/filter` | `savgol_filter_dyn` / `savgol_coeffs_dyn` | partial | `SavgolFilterKernel` landed; coeffs API still legacy |
| `signal/filter` | `lfilter_zi_dyn` / `sosfilt_zi_dyn` | pending | no trait-first wrappers yet |
| `signal/filter` | `pad` / `odd_ext_dyn` / `axis_slice` / `axis_reverse` | pending | helper utilities pending trait policy |
| `signal/filter/design` | `cheby1_dyn` / `cheby2_dyn` + zpk transforms | partial | kernels for `firwin/iirfilter/butter` landed |
| `signal/wave` | `square` (ndarray N-D) | partial | 1D trait kernel landed; N-D API remains legacy |

## Acceptance Gate Tracker

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --all-features -- -D warnings`
- [x] `cargo test --no-default-features`
- [x] `cargo test --no-default-features --features alloc`
- [x] `cargo test --all-features`
- [x] `cargo bench --all-features --no-run`
- [x] `cargo run -p xtask -- contracts`

## Next Interfaces In Flight

1. Legacy free-function cleanup and quarantine/deprecation policy.
2. Convert remaining panic-based validation paths into deterministic config errors where feasible.
3. Add trait-first coverage for remaining helper utilities (`arraytools`, extension helpers).
4. Expand allocation/perf assertions for hot paths in benchmark suites.
