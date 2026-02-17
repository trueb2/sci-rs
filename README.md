[![CI](https://github.com/qsib-cbie/sci-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/qsib-cbie/sci-rs/actions/workflows/rust.yml)
[![Crate](https://img.shields.io/crates/v/sci-rs)](https://crates.io/crates/sci-rs)
[![Crate Downloads](https://img.shields.io/crates/d/sci-rs?label=downloads)](https://crates.io/crates/sci-rs)

# sci-rs

sci-rs is a Rust scientific computing library inspired by SciPy, built for embedded (`no_std` + `alloc`), mobile, and server deployments.

It covers signal processing, statistics, special functions, and linear algebra helpers with a trait-first kernel style for refactored interfaces.

## What sci-rs Covers

- Signal processing:
  filtering, filter design, resampling, multirate processing, waveforms, spectral analysis, windows, convolution, and peak routines.
- Statistics and time-series utilities:
  mean/median/variance/stdev, z-score and modified z-score, autocorrelation, lag differences, and RMSSD.
- Special math:
  factorial variants, combinatorics, and Bessel support.
- Linear algebra helpers:
  companion matrix construction and shared kernel adapters.

## Feature Flags

- `alloc` (default):
  enables allocating paths and modules that require dynamic memory.
- `std`:
  enables FFT-backed and std-only routines (including parts of spectral/convolution/resampling).
- `plot`:
  local debug plotting helpers.

## API Style (Trait-First)

The project standard is `trait + config + kernel`:

1. Import the capability trait for the module you are using.
2. Construct the kernel with `KernelLifecycle::try_new(config)`.
3. Execute with:
   - `run_into(...)` for caller-provided output.
   - `run_alloc(...)` for allocating convenience paths (`alloc`).

## Quick Start

### Stats kernel usage

```rust
use sci_rs::kernel::KernelLifecycle;
use sci_rs::stats::{
    MeanKernel, MeanReduce1D, StatsConfig, VarianceKernel, VarianceReduce1D, ZScoreKernel,
    ZScoreNormalize1D,
};

let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];

let mean_kernel = MeanKernel::try_new(StatsConfig).expect("mean kernel");
let var_kernel = VarianceKernel::try_new(StatsConfig).expect("variance kernel");
let z_kernel = ZScoreKernel::try_new(StatsConfig).expect("z-score kernel");

let (mean, n) = mean_kernel.run(&x).expect("mean run");
let (variance, _) = var_kernel.run(&x).expect("variance run");
let mut z = vec![0.0_f64; x.len()];
z_kernel.run_into(&x, &mut z).expect("z-score run");

assert_eq!(n, x.len());
assert_eq!(mean, 3.0);
assert_eq!(variance, 2.0);
```

### Signal design and zero-phase filtering

```rust
use sci_rs::kernel::KernelLifecycle;
use sci_rs::signal::filter::{
    design::{ButterConfig, ButterKernel, DigitalFilter, FilterBandType, FilterOutputType},
    SosFiltFiltConfig, SosFiltFiltKernel,
};
use sci_rs::signal::traits::{IirDesign, SosFiltFilt1D};

let design = ButterKernel::try_new(ButterConfig {
    order: 4,
    wn: vec![10.0_f64, 50.0],
    btype: Some(FilterBandType::Bandpass),
    analog: Some(false),
    output: Some(FilterOutputType::Sos),
    fs: Some(1666.0),
})
.expect("butter kernel");

let sos = match design.run_alloc().expect("design run") {
    DigitalFilter::Sos(s) => s.sos,
    _ => panic!("expected SOS output"),
};

let kernel = SosFiltFiltKernel::try_new(SosFiltFiltConfig { sos }).expect("sosfiltfilt kernel");
let x: Vec<f64> = (0..1024)
    .map(|i| ((i as f64) * 0.01).sin() + 0.1 * ((i as f64) * 0.37).sin())
    .collect();

let y = kernel.run_alloc(&x).expect("sosfiltfilt run");
assert_eq!(x.len(), y.len());
```

### Preallocated multirate execution (`run_into`)

```rust
use sci_rs::kernel::KernelLifecycle;
use sci_rs::signal::multirate::{ResamplePolyConfig, ResamplePolyKernel};
use sci_rs::signal::traits::ResamplePoly1D;

let kernel = ResamplePolyKernel::try_new(ResamplePolyConfig { up: 2, down: 1 })
    .expect("resample_poly kernel");
let x = vec![0.0_f32, 1.0, 0.0, -1.0, 0.0];
let mut y = vec![0.0_f32; kernel.expected_len(x.len())];

kernel.run_into(&x, &mut y).expect("resample_poly run");
assert_eq!(y.len(), kernel.expected_len(x.len()));
```

## Coverage and Parity Tracking

Detailed SciPy parity/backlog status is tracked in `docs/SCIPY_CHECKLIST.md`.

## SciPy Comparison and Contracts

SciPy is the local behavior oracle for parity checks.

Generate local contract artifacts:

```bash
cargo run -p xtask -- contracts
```

Artifacts are written to `target/contracts/<timestamp>/` and include:

- `summary.csv`
- `summary.json`
- `report.pdf`
- `plots/*.png`

Required parity metrics per case:

- Pearson `r`
- MAE
- RMSE
- max-abs

Required performance metrics per case:

- `rust_candidate_ns`
- `rust_baseline_ns`
- `python_ns`
- `speedup_vs_baseline`
- `speedup_vs_python`

## Contributing

Refactor and SciPy-process docs:

- `AGENTS.md`
- `docs/REFACTOR_SOUL.md`
- `docs/SCIPY_CHECKLIST.md`
- `docs/SCIPY_CHECKLIST_PROCESS.md`
- `docs/SCIPY_AGENT_PACKET_TEMPLATE.md`
- `docs/SCIPY_TRACE_PROTOCOL.md`

Checklist sync commands:

```bash
python scripts/generate_scipy_checklist_inventory.py
python scripts/verify_scipy_checklist.py
```

## MSRV

Until the project matures, the target is the latest stable Rust.
