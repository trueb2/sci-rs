# sci-rs Trait-First Refactor Checklist

This checklist tracks interface migration status for the breaking refactor.

File-by-file traversal status is tracked in `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SOURCE_TRAVERSAL.md`.

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
| signal | `SawtoothWave1D<T>` | yes | yes | yes | `SawtoothWaveKernel` |
| signal | `ChirpWave1D<T>` | yes | yes | yes | `ChirpKernel` |
| signal | `UnitImpulse1D<T>` | yes | yes | yes | `UnitImpulseKernel` |
| signal | `SavgolFilter1D<T>` | yes | yes | yes | `SavgolFilterKernel` |
| signal | `SosFilt1D<T>` | yes | yes | yes | `SosFiltKernel` |
| signal | `SosFiltFilt1D<T>` | yes | yes | yes | kernel parity and invariant tests |
| signal | `LFilterZiDesign1D<T>` | yes | yes | yes | `LFilterZiKernel` |
| signal | `SosFiltZiDesign1D<T>` | yes | yes | yes | `SosFiltZiKernel` |
| signal | `LFilter1D<T>` | yes | yes | yes | reference parity + length checks |
| signal | `FiltFilt1D<T>` | yes | yes | yes | reference parity + length checks |
| design | `FirWinDesign<T>` | yes | yes | yes | `FirWinKernel` with constructor validation |
| design | `IirDesign<T>` | yes | yes | yes | `IirFilterKernel` and `ButterKernel` |
| design | `ZpkTransform<T>` | yes | yes | yes | `BilinearZpkKernel`, `Lp2*ZpkKernel` family |
| design | `ZpkToTfDesign<T>` | yes | yes | yes | `ZpkToTfKernel` |
| design | `ZpkToSosDesign<T>` | yes | yes | yes | `ZpkToSosKernel` |
| design | `RelativeDegreeDesign<T>` | yes | yes | yes | `RelativeDegreeKernel` |
| design | `ComplexPairSplit<T>` | yes | yes | yes | `CplxRealKernel` |
| windows | `WindowGenerate<T>` | yes | yes | yes | `WindowKernel` and owned window builder |
| stats | `MeanReduce1D<T>` | yes | yes | yes | `MeanKernel` |
| stats | `VarianceReduce1D<T>` | yes | yes | yes | `VarianceKernel` |
| stats | `StdevReduce1D<T>` | yes | yes | yes | `StdevKernel` |
| stats | `MedianReduce1D<T>` | yes | yes | yes | `MedianKernel` |
| stats | `MadReduce1D<T>` | yes | yes | yes | `MadKernel` |
| stats | `ZScoreNormalize1D<T>` | yes | yes | yes | `ZScoreKernel` |
| stats | `ModZScoreNormalize1D<T>` | yes | yes | yes | `ModZScoreKernel` |
| linalg | `CompanionBuild1D<T>` | yes | yes | yes | `CompanionKernel` |
| migration | legacy free functions cleanup | yes | yes | yes | compatibility shims are kernel-first where feasible; checked wrappers added (`iirfilter/butter/cplx/relative_degree` included) |
| contract | local contract runner (`xtask`) | yes | yes | yes | local-only artifacts under `target/contracts` |

## Legacy Shim Status

| Legacy API | Trait-first replacement | shimmed | tested |
| --- | --- | --- | --- |
| `convolve` | `ConvolveKernel` + `Convolve1D` | yes | yes |
| `correlate` | `CorrelateKernel` + `Correlate1D` | yes | yes |
| `resample` | `ResampleKernel` + `Resample1D` | yes | yes |
| `sawtooth` | `SawtoothWaveKernel` + `SawtoothWave1D` | yes | yes |
| `chirp` | `ChirpKernel` + `ChirpWave1D` | yes | yes |
| `unit_impulse` | `UnitImpulseKernel` + `UnitImpulse1D` | yes | yes |
| `lfilter` | `LFilterKernel` + `LFilter1D` | yes | yes |
| `filtfilt` | `FiltFiltKernel` + `FiltFilt1D` | yes | yes |
| `sosfilt` | `SosFiltKernel` + `SosFilt1D` | yes | yes |
| `sosfiltfilt` | `SosFiltFiltKernel` + `SosFiltFilt1D` | yes | yes |

## Public Interface Sweep (Complete)

This sweep is complete for the refactor scope; remaining legacy entrypoints are compatibility shims.

| Module | Public API | trait-first status | Notes |
| --- | --- | --- | --- |
| `signal/filter` | `lfilter` | complete | compatibility shim now includes kernel-first 1D fast path |
| `signal/filter` | `sosfilt_dyn` / `sosfilt_item` / `sosfilt_st` | complete | checked wrappers and trait-first kernels available |
| `signal/filter` | `sosfiltfilt_dyn` | complete | checked API + trait-first kernel parity |
| `signal/filter` | `savgol_filter_dyn` / `savgol_coeffs_dyn` | complete | checked APIs and trait-first kernels (`SavgolFilterKernel`, `SavgolCoeffsKernel`) |
| `signal/filter` | `lfilter_zi_dyn` / `sosfilt_zi_dyn` | complete | checked wrappers and design kernels wired |
| `signal/filter` | `pad` / `odd_ext_dyn` / `axis_slice` / `axis_reverse` | complete | checked helper APIs and axis kernels landed |
| `signal/filter/design` | `cheby1_dyn` / `cheby2_dyn` + zpk transforms | complete | covered by `IirFilterKernel` plus dedicated zpk helper kernels; checked `iirfilter/butter` entrypoints now available |
| `signal/wave` | `square`/`sawtooth`/`chirp` (ndarray N-D) + `unit_impulse` (1D) | complete | waveform APIs now route through trait-first kernels |
| `stats` | free functions (`mean/variance/stdev/median/mad/zscore`) | complete | trait-first stats kernels exist and are parity-tested; free functions are compatibility shims |
| `linalg` | `companion_dyn` | complete | checked kernel-backed construction path added |

## Acceptance Gate Tracker

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --all-features -- -D warnings`
- [x] `cargo test --no-default-features`
- [x] `cargo test --no-default-features --features alloc`
- [x] `cargo test --all-features`
- [x] `cargo bench --all-features --no-run`
- [x] `cargo run -p xtask -- contracts`

## Next Interfaces In Flight

1. `gausspulse`
2. `sweep_poly`

## Source Recheck Audit (2026-02-17)

- Traversed source inventory: `68/68` files tracked across `sci-rs/src`, `sci-rs-core/src`, and `sci-rs-test/src`.
- Traversal parity check: `0` untracked files and `0` stale entries in `docs/SOURCE_TRAVERSAL.md`.
- Validation rerun after latest checked-path changes: `fmt`, `clippy -D warnings`, tests (`all-features`, `no-default-features`, `no-default-features --features alloc`), and `bench --no-run` all pass.
