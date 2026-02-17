# SciPy Checklist Process

This document defines how contributors and agents update `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_CHECKLIST.md` while implementing SciPy features in `sci-rs`.

The goal is repeatable, auditable feature delivery:

1. Claim one function.
2. Implement trait-first kernel API.
3. Prove correctness against Python/SciPy.
4. Prove performance against local Rust baseline.
5. Mark merge readiness in the checklist row.

## 1. Canonical Row Lifecycle

Valid lifecycle statuses:

1. `wishlist`
2. `assigned`
3. `impl`
4. `parity`
5. `perf`
6. `ready`
7. `merged`
8. `blocked`
9. `deferred_pythonism`

`deferred_pythonism` is only for non-computational symbols (currently `BadCoefficients` and `test`).

## 2. Claim Protocol

1. Pick exactly one `wishlist` row from `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_CHECKLIST.md`.
2. Set:
   - `status = assigned`
   - `owner = <github-handle-or-name>`
   - `branch = codex/<short-feature-branch>` or `user/<name>/<short-feature-branch>`
3. Confirm dependencies in the `notes` field if the row is blocked by another symbol.

## 3. Implementation Protocol

For each claimed symbol:

1. Add or extend trait-first interfaces:
   - `try_new(config) -> Result<Kernel, ConfigError>`
   - `run_into(...)` first
   - optional `run_alloc(...)` under `alloc`
2. Do not add public free-function shims; any helper wrappers must stay internal (`pub(crate)`).
3. Add deterministic constructor validation for invalid argument/config shape cases.
4. Move checklist status to `impl` once code is in progress.

## 4. Evidence Protocol

### Tests

Add tests for:

1. Correctness against expected reference behavior.
2. Constructor/config failure modes (`ConfigError`).
3. Adapter/container behavior where applicable (slice/array/vec/ndarray views).

### Contracts

1. Add or extend `xtask` contract case IDs for the symbol.
2. Generate local contract artifacts:

```bash
cargo run -p xtask -- contracts
```

3. Fill row fields:
   - `contract_case_ids`
   - `parity_threshold_profile`

### Benchmarks

1. Add or extend a benchmark ID/case for the symbol.
2. Fill row fields:
   - `benchmark_ids`
   - `perf_gate_result`

## 5. Row Update Requirements By Status

Required fields for each status:

1. `assigned` / `impl`: `owner`, `branch`
2. `parity`: `owner`, `branch`, `contract_case_ids`, `parity_threshold_profile`
3. `perf`: `owner`, `branch`, `contract_case_ids`, `parity_threshold_profile`, `benchmark_ids`, `perf_gate_result`
4. `ready`: all `perf` fields complete and all local checks pass
5. `merged`: all `ready` fields plus `rust_trait_config_kernel`, `rust_paths`, `notes` (include merge commit hash)
6. `blocked`: `owner`, `branch`, `notes` describing blocker and unblock condition

## 6. Merge-Ready Protocol

A row can move to `ready` only if all local checks pass:

```bash
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
cargo test --no-default-features
cargo test --no-default-features --features alloc
cargo test --all-features
cargo bench --all-features --no-run
cargo run -p xtask -- contracts
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/verify_scipy_checklist.py
```

Perf gate policy for `ready`:

1. No canonical benchmark case regresses by more than 10% versus Rust baseline.
2. At least one canonical case improves by at least 10%.

## 7. Post-Merge Protocol

After merge into the active integration branch:

1. Set `status = merged`.
2. Record commit hash and merge note in `notes`.
3. Update `rust_trait_config_kernel` and `rust_paths` with concrete identifiers/paths.
4. Re-run checklist verification:

```bash
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/verify_scipy_checklist.py
```

## 8. Inventory Sync

When SciPy version or inventory changes:

```bash
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/generate_scipy_checklist_inventory.py
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/verify_scipy_checklist.py
```

The generator refreshes symbol/source/docs fields while preserving manual progress/evidence fields.
