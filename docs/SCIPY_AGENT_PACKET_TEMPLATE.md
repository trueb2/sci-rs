# SciPy Agent Packet Template

Use this template when claiming a symbol from `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_CHECKLIST.md`.

## 1. Packet Header

- `symbol`:
- `scipy_namespace`:
- `owner`:
- `branch`:
- `checklist_row_status`:

## 2. Scope

- Supported arguments/modes in this packet:
- Deferred arguments/modes in this packet:
- Out-of-scope behavior notes:

## 3. SciPy Source Trace Evidence

- Docs URL:
- Source module and line anchor:
- Helper-call flow summary:
- Why this mapping matches SciPy behavior:

## 4. Rust Interface Mapping

- Trait(s):
- Config type(s):
- Kernel type(s):
- Constructor validation rules added:

## 5. Tests Added

- Behavior tests:
- Adapter/container compatibility tests:
- Config failure (`ConfigError`) tests:
- Edge-case tests:

## 6. Contract Evidence (Python/SciPy)

- `contract_case_ids`:
- Parity metrics summary:
  - Pearson `r`:
  - MAE:
  - RMSE:
  - max-abs:
- Threshold profile used:
- Contract artifact path (`target/contracts/<timestamp>`):

## 7. Benchmark Evidence

- `benchmark_ids`:
- Candidate vs baseline ns summary:
- Candidate vs Python ns summary:
- `perf_gate_result`:
- Performance caveats:

## 8. Checklist Row Update

- `status` transition:
- `rust_trait_config_kernel` value:
- `rust_paths` value:
- `notes` update:

## 9. Final Merge Checklist

Run from repository root:

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

If all commands pass and checklist row is complete, set status to `ready`.
