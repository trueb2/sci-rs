# sci-rs Trait-First Refactor Soul Document

## Mission

Refactor sci-rs into a trait-first, kernel-oriented scientific computing library that:

1. Prioritizes embedded determinism and low allocation.
2. Scales to mobile/server workloads via optional parallel paths.
3. Preserves correctness by validating against NumPy/SciPy reference behavior.

## Non-Goals (Current Cycle)

1. Preserving all legacy free-function APIs.
2. Introducing proc-macro-heavy API generation.
3. Merging large unrelated upstream PRs during refactor execution.

## Design Principles

1. Constructor-time validation:
   - `try_new(config) -> Result<Kernel, ConfigError>`
2. Hot path execution:
   - no per-sample `Result` overhead.
3. Output control:
   - `run_into` for caller-provided buffers.
   - optional allocating convenience methods under `alloc`.
4. Explicit type adapters:
   - slices, arrays, vecs, ndarray 1D, and stream-like iterators.

## Target API Shape

Each migrated capability follows the same structure:

1. A trait describing the capability (for example `Convolve1D`, `Resample1D`).
2. A kernel struct implementing the trait.
3. A config struct with all validation done once in `try_new`.
4. Execution methods:
   - `run_into(...)`
   - optional `run_alloc(...)` behind `alloc`.

## Migration Milestones

## M0: Guardrails and Hygiene

1. Finalize `AGENTS.md`.
2. Remove blocking/interactive plotting behavior in tests and CI paths.
3. Fix all clippy warnings blocking `-D warnings`.

## M1: Trait Substrate

1. Introduce shared I/O adapter traits:
   - `Read1D<T>`
   - `Write1D<T>`
   - `SampleStream<T>`
2. Introduce shared config/execution error types.
3. Add base kernel lifecycle conventions and docs.

## M2: Signal Filtering Core

1. Migrate:
   - `sosfilt`
   - `sosfiltfilt`
   - `lfilter`
   - `filtfilt`
2. Replace panic/TODO execution paths with deterministic policy.
3. Add adapter coverage tests across container types.

## M3: Design + Windows

1. Migrate:
   - `iirfilter`
   - `butter`
   - `firwin`
   - window builders/generators
2. Standardize constructor validation and run API style.

## M4: Convolution + Resample

1. Migrate:
   - `convolve`
   - `correlate`
   - `resample`
2. Keep legacy free functions as thin compatibility wrappers temporarily.

## M5: Contract System

1. Add local contract runner command:
   - `cargo run -p xtask -- contracts`
2. Generate local artifact bundle under:
   - `target/contracts/<timestamp>/`
3. Include:
   - `summary.csv`
   - `summary.json`
   - `report.pdf`
   - per-case PNG plots

## Quality Gates

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-features -- -D warnings`
3. `cargo test --no-default-features`
4. `cargo test --no-default-features --features alloc`
5. `cargo test --all-features`
6. `cargo bench --all-features --no-run`
7. Contract bundle generation succeeds locally.

## Python Gold-Reference Baseline

Pinned local baseline:

1. Python `3.12.6`
2. NumPy `2.1.0`
3. SciPy `1.14.1`
4. Matplotlib `3.9.2`
5. Python executable comes from active environment as `python`.

Required contract metrics:

1. Pearson `r`
2. MAE
3. RMSE
4. Max absolute error

Required performance columns:

1. `rust_candidate_ns`
2. `rust_baseline_ns`
3. `python_ns`
4. `speedup_vs_baseline`
5. `speedup_vs_python`

## Work Packet Template (for Delegated Agents)

Each delegated task must include:

1. Scope:
   - module/functionality covered.
2. Interfaces:
   - trait/config/kernel types added or changed.
3. Validation:
   - constructor checks added.
4. Tests:
   - adapter matrix and behavior checks.
5. Evidence:
   - benchmark summary and contract deltas if behavior changes.
6. Risks:
   - known TODOs and unsupported paths.

## Communication Standard

1. Be respectful, factual, and reproducible.
2. Keep commentary technical and evidence-based.
3. Use clear absolute paths and exact commands for reproducibility.

## Refactor Cadence

1. Refactor one interface group.
2. Run formatting, clippy, tests, bench compile, and contract generation.
3. Commit locally with a focused message.
4. Re-scan interfaces and update `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/REFACTOR_CHECKLIST.md`.
5. Repeat until all planned interfaces are marked refactored and tested.
