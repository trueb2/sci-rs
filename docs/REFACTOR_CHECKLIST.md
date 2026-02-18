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
| signal | `GaussPulseWave1D<T>` | yes | yes | yes | `GaussPulseKernel` |
| signal | `SweepPolyWave1D<T>` | yes | yes | yes | `SweepPolyKernel` |
| signal | `UnitImpulse1D<T>` | yes | yes | yes | `UnitImpulseKernel` |
| signal | `SavgolFilter1D<T>` | yes | yes | yes | `SavgolFilterKernel` |
| signal | `SosFilt1D<T>` | yes | yes | yes | `SosFiltKernel` |
| signal | `SosFiltFilt1D<T>` | yes | yes | yes | kernel parity and invariant tests |
| signal | `LFilterZiDesign1D<T>` | yes | yes | yes | `LFilterZiKernel` |
| signal | `SosFiltZiDesign1D<T>` | yes | yes | yes | `SosFiltZiKernel` |
| signal | `LFilter1D<T>` | yes | yes | yes | reference parity + length checks |
| signal | `FiltFilt1D<T>` | yes | yes | yes | reference parity + length checks |
| signal | `UpFirDn1D<T>` | yes | yes | yes | `UpFirDnKernel` |
| signal | `ResamplePoly1D<T>` | yes | yes | yes | `ResamplePolyKernel` |
| signal | `Decimate1D<T>` | yes | yes | yes | `DecimateKernel` |
| signal | `ArgRelExtrema1D<T>` | yes | yes | yes | `ArgRelExtremaKernel` |
| signal | `FindPeaks1D<T>` | yes | yes | yes | `FindPeaksKernel` |
| signal | `PeakProminence1D<T>` | yes | yes | yes | `PeakProminencesKernel` |
| signal | `PeakWidths1D<T>` | yes | yes | yes | `PeakWidthsKernel` |
| signal | `Cwt1D<T>` | yes | yes | yes | `CwtKernel` |
| signal | `FindPeaksCwt1D<T>` | yes | yes | yes | `FindPeaksCwtKernel` |
| signal | `Periodogram1D` | yes | yes | yes | `PeriodogramKernel` |
| signal | `WelchPsd1D` | yes | yes | yes | `WelchKernel` |
| signal | `Csd1D` | yes | yes | yes | `CsdKernel` |
| signal | `Coherence1D` | yes | yes | yes | `CoherenceKernel` |
| signal | `Stft1D` | yes | yes | yes | `StftKernel` |
| signal | `Istft1D` | yes | yes | yes | `IstftKernel` |
| signal | `Spectrogram1D` | yes | yes | yes | `SpectrogramKernel` |
| signal | `Freqz1D` | yes | yes | yes | `FreqzKernel` |
| signal | `SosFreqz1D` | yes | yes | yes | `SosFreqzKernel` |
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
| migration | legacy free functions cleanup | yes | yes | yes | public entrypoints are kernel-backed checked APIs; silent fallback wrappers removed from migrated paths |
| contract | local contract runner (`xtask`) | yes | yes | yes | local-only artifacts under `target/contracts`; Python parity coverage expanded for all currently implemented SciPy checklist interfaces |

## Function Entrypoint Status

| Legacy API | Trait-first replacement | checked | tested |
| --- | --- | --- | --- |
| `convolve` | `ConvolveKernel` + `Convolve1D` | yes | yes |
| `correlate` | `CorrelateKernel` + `Correlate1D` | yes | yes |
| `resample` | `ResampleKernel` + `Resample1D` | yes | yes |
| `sawtooth` | `SawtoothWaveKernel` + `SawtoothWave1D` | yes | yes |
| `chirp` | `ChirpKernel` + `ChirpWave1D` | yes | yes |
| `gausspulse` | `GaussPulseKernel` + `GaussPulseWave1D` | yes | yes |
| `sweep_poly` | `SweepPolyKernel` + `SweepPolyWave1D` | yes | yes |
| `unit_impulse` | `UnitImpulseKernel` + `UnitImpulse1D` | yes | yes |
| `lfilter` | `LFilterKernel` + `LFilter1D` | yes | yes |
| `filtfilt` | `FiltFiltKernel` + `FiltFilt1D` | yes | yes |
| `sosfilt` | `SosFiltKernel` + `SosFilt1D` | yes | yes |
| `sosfiltfilt` | `SosFiltFiltKernel` + `SosFiltFilt1D` | yes | yes |

## Public Interface Sweep (Complete)

This sweep is complete for the refactor scope: the public signal interface is trait/config/kernel-first.
Legacy function-style helpers are internal-only (`pub(crate)`) and are not part of the public API contract.

| Module | Public trait/config/kernel surface | trait-first status | Notes |
| --- | --- | --- | --- |
| `signal/convolve` | `Convolve1D` / `ConvolveConfig` / `ConvolveKernel` | complete | kernel-backed checked execution |
| `signal/convolve` | `Correlate1D` / `CorrelateConfig` / `CorrelateKernel` | complete | kernel-backed checked execution |
| `signal/resample` | `Resample1D` / `ResampleConfig` / `ResampleKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `LFilter1D` / `LFilterConfig` / `LFilterKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `FiltFilt1D` / `FiltFiltConfig` / `FiltFiltKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `SosFilt1D` / `SosFiltConfig` / `SosFiltKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `SosFiltFilt1D` / `SosFiltFiltConfig` / `SosFiltFiltKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `LFilterZiDesign1D` / `LFilterZiConfig` / `LFilterZiKernel` | complete | design kernel path |
| `signal/filter` | `SosFiltZiDesign1D` / `SosFiltZiConfig` / `SosFiltZiKernel` | complete | design kernel path |
| `signal/filter` | `SavgolFilter1D` / `SavgolFilterConfig` / `SavgolFilterKernel` | complete | kernel-backed checked execution |
| `signal/filter` | `SavgolCoeffsDesign` / `SavgolCoeffsConfig` / `SavgolCoeffsKernel` | complete | design kernel path |
| `signal/filter/design` | `FirWinDesign` / `FirWinConfig` / `FirWinKernel` | complete | constructor validation + kernel execution |
| `signal/filter/design` | `IirDesign` / `IirFilterConfig` / `IirFilterKernel` | complete | constructor validation + kernel execution |
| `signal/filter/design` | `IirDesign` / `ButterConfig` / `ButterKernel` | complete | constructor validation + kernel execution |
| `signal/windows` | `WindowGenerate` / `WindowConfig` / `WindowKernel` | complete | owned builder + kernel execution |
| `signal/wave` | `SquareWave1D` / `SquareWaveConfig` / `SquareWaveKernel` | complete | kernel-backed waveform generation |
| `signal/wave` | `SawtoothWave1D` / `SawtoothWaveConfig` / `SawtoothWaveKernel` | complete | kernel-backed waveform generation |
| `signal/wave` | `ChirpWave1D` / `ChirpConfig` / `ChirpKernel` | complete | kernel-backed waveform generation |
| `signal/wave` | `GaussPulseWave1D` / `GaussPulseConfig` / `GaussPulseKernel` | complete | kernel-backed waveform generation |
| `signal/wave` | `SweepPolyWave1D` / `SweepPolyConfig` / `SweepPolyKernel` | complete | kernel-backed waveform generation |
| `signal/wave` | `UnitImpulse1D` / `UnitImpulseConfig` / `UnitImpulseKernel` | complete | kernel-backed waveform generation |
| `signal/multirate` | `UpFirDn1D` / `UpFirDnConfig` / `UpFirDnKernel` | complete | checked trait-first kernel |
| `signal/multirate` | `ResamplePoly1D` / `ResamplePolyConfig` / `ResamplePolyKernel` | complete | checked trait-first kernel |
| `signal/multirate` | `Decimate1D` / `DecimateConfig` / `DecimateKernel` | complete | checked trait-first kernel |
| `signal/peak` | `ArgRelExtrema1D` / `ArgRelExtremaConfig` / `ArgRelExtremaKernel` | complete | checked trait-first kernel |
| `signal/peak` | `FindPeaks1D` / `FindPeaksConfig` / `FindPeaksKernel` | complete | checked trait-first kernel |
| `signal/peak` | `PeakProminence1D` / `PeakProminencesConfig` / `PeakProminencesKernel` | complete | checked trait-first kernel |
| `signal/peak` | `PeakWidths1D` / `PeakWidthsConfig` / `PeakWidthsKernel` | complete | checked trait-first kernel |
| `signal/peak` | `Cwt1D` / `CwtConfig` / `CwtKernel` | complete | checked trait-first kernel |
| `signal/peak` | `FindPeaksCwt1D` / `FindPeaksCwtConfig` / `FindPeaksCwtKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Periodogram1D` / `PeriodogramConfig` / `PeriodogramKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `WelchPsd1D` / `WelchConfig` / `WelchKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Csd1D` / `CsdConfig` / `CsdKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Coherence1D` / `CoherenceConfig` / `CoherenceKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Stft1D` / `StftConfig` / `StftKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Istft1D` / `IstftConfig` / `IstftKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Spectrogram1D` / `SpectrogramConfig` / `SpectrogramKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `Freqz1D` / `FreqzConfig` / `FreqzKernel` | complete | checked trait-first kernel |
| `signal/spectral` | `SosFreqz1D` / `SosFreqzConfig` / `SosFreqzKernel` | complete | checked trait-first kernel |

## Acceptance Gate Tracker

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --all-features -- -D warnings`
- [x] `cargo test --no-default-features`
- [x] `cargo test --no-default-features --features alloc`
- [x] `cargo test --all-features`
- [x] `cargo bench --all-features --no-run`
- [x] `cargo run -p xtask -- contracts`

## Next Interfaces In Flight

1. none (`Signal-First Priority Queue` 20/20 implemented)

## Source Recheck Audit (2026-02-18)

- Traversed source inventory: `71/71` files tracked across `sci-rs/src`, `sci-rs-core/src`, and `sci-rs-test/src`.
- Traversal parity check: `0` untracked files and `0` stale entries in `docs/SOURCE_TRAVERSAL.md`.
- Validation rerun after latest checked-path changes: `fmt`, `clippy -D warnings`, tests (`all-features`, `no-default-features`, `no-default-features --features alloc`), and `bench --no-run` all pass.
- Contract parity audit: all `26` checklist `impl` rows now have `contract_case_ids`; latest local bundle includes `49` cases in `target/contracts/1771376659/`.
