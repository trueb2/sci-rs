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
| substrate | `SampleStream<T>` | yes | yes | partial | iterator blanket impl present |
| substrate | `KernelLifecycle` | yes | yes | partial | constructor validation pattern in place |
| signal | `Convolve1D<T>` | yes | yes | yes | `ConvolveKernel` |
| signal | `Correlate1D<T>` | yes | yes | yes | `CorrelateKernel` |
| signal | `Resample1D<T>` | yes | yes | yes | `ResampleKernel` |
| signal | `SosFilt1D<T>` | yes | yes | yes | `SosFiltKernel` |
| signal | `SosFiltFilt1D<T>` | yes | yes | partial | kernel tests extending |
| signal | `LFilter1D<T>` | yes | yes | partial | kernel wrappers in place |
| signal | `FiltFilt1D<T>` | yes | yes | partial | kernel wrappers in place |
| design | `FirWinDesign<T>` | yes | yes | yes | `FirWinKernel` with constructor validation |
| design | `IirDesign<T>` | yes | yes | yes | `IirFilterKernel` and `ButterKernel` |
| windows | `WindowGenerate<T>` | yes | yes | yes | `WindowKernel` and owned window builder |
| migration | legacy free functions cleanup | yes | partial | pending | compatibility shim policy active |
| contract | local contract runner (`xtask`) | yes | yes | yes | local-only artifacts under `target/contracts` |

## Acceptance Gate Tracker

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --all-features -- -D warnings`
- [x] `cargo test --no-default-features`
- [x] `cargo test --no-default-features --features alloc`
- [x] `cargo test --all-features`
- [x] `cargo bench --all-features --no-run`
- [x] `cargo run -p xtask -- contracts`

## Next Interfaces In Flight

1. Finalize concrete trait contracts for design/windows.
2. Land `firwin`, `iirfilter`, `butter`, and `get_window` kernels with constructor-time validation.
3. Add adapter-focused tests for design/window kernels.
4. Expand no-allocation assertions where execution paths support caller buffers.
