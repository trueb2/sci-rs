# sci-rs Agent Contract

This file defines mandatory contributor and agent behavior for the ongoing trait-first refactor.

## 1. Git Safety Rules (Hard Requirements)

1. Never push directly to `main`.
2. Never push to upstream remotes owned outside the fork.
3. Perform work on feature/refactor branches only.
4. Recommended branch names:
   - `codex/refactor-*`
   - `user/<name>/refactor-*`
5. Before any push, verify destination and branch:
   - `git remote -v`
   - `git branch --show-current`
   - `git push --dry-run <remote> <branch>`
6. Enable local protected push hook in this repo:
   - `git config core.hooksPath .githooks`

## 2. Required Local Checks Before Push

Run all commands from repository root:

```bash
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
cargo test --no-default-features
cargo test --no-default-features --features alloc
cargo test --all-features
cargo bench --all-features --no-run
cargo run -p xtask -- contracts
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/generate_scipy_checklist_inventory.py
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/verify_scipy_checklist.py
```

If any command fails, do not push.

## 3. Benchmark Policy

Use two benchmark modes:

1. Quick local smoke benchmark (during development):
   - `cargo bench --all-features --bench sosfilt -- --sample-size 10`
2. Full benchmark compile gate (before push/PR):
   - `cargo bench --all-features --no-run`

Benchmark output policy:

1. Keep criterion output local.
2. If publishing benchmark evidence, export summarized tables to:
   - `target/contracts/<timestamp>/summary.csv`
   - `target/contracts/<timestamp>/summary.json`

## 4. Plotting Policy (Non-Blocking Only)

1. Tests and CI must never open interactive plot windows.
2. `plt.show()` is prohibited in test/CI paths.
3. Use non-interactive backend and file output only:
   - `matplotlib.use("Agg")`
   - `plt.savefig(...)`
4. Plot artifacts are local-only and written under:
   - `target/contracts/<timestamp>/plots/`

## 5. Python Comparison Policy

Pinned local reference environment:

1. Python `3.12.6`
2. NumPy `2.1.0`
3. SciPy `1.14.1`
4. Matplotlib `3.9.2`
5. Python executable is resolved from active environment via command name `python`.

Required parity metrics per contract case:

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

## 6. Contract Artifact Policy (Local Only)

1. Contract bundles are generated locally only.
2. Output location:
   - `target/contracts/<timestamp>/`
3. Required files:
   - `summary.csv`
   - `summary.json`
   - `report.pdf`
   - per-case PNG plots
4. Contract artifacts must not be committed.

## 7. Public OSS Collaboration Policy

1. Be respectful and precise in technical discussions.
2. Keep review feedback evidence-first:
   - include failing test, metric, benchmark, or contract deltas.
3. Prefer small, auditable PRs over large mixed changes.
4. Do not dismiss prior work; supersede with reproducible evidence.

## 8. Refactor Execution Priorities

1. Trait-first kernel APIs with constructor-time validation.
2. Embedded-first deterministic execution as default.
3. Optional parallel execution under explicit feature flags.
4. Free functions are compatibility shims or deprecated after trait migration.
5. Avoid proc macros for API shape; use traits and explicit adapters.

## 9. Standard Commands

Contract generation:

```bash
cargo run -p xtask -- contracts
```

Check for banned interactive plotting calls:

```bash
if rg -n "plt\\.show\\(" sci-rs/src sci-rs-test/src .github/workflows -S; then
  echo "Interactive plotting calls detected."
  exit 1
fi
```

## 10. Refactor Cadence Rule

For the trait-first refactor, contributors and agents must work in short verified loops:

1. Refactor one interface group.
2. Run local checks (format, clippy, tests, bench compile, contracts).
3. Commit locally.
4. Update `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/REFACTOR_CHECKLIST.md`.
5. Re-scan remaining interfaces and repeat.

## 11. SciPy Checklist Operating Model

For any feature PR targeting SciPy parity, contributors must update one row in:

- `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_CHECKLIST.md`

Process and handoff docs:

1. `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_CHECKLIST_PROCESS.md`
2. `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_AGENT_PACKET_TEMPLATE.md`
3. `/Users/jacobtrueb/Desktop/workspace/sci-rs/docs/SCIPY_TRACE_PROTOCOL.md`

Checklist sync and validation commands:

```bash
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/generate_scipy_checklist_inventory.py
python /Users/jacobtrueb/Desktop/workspace/sci-rs/scripts/verify_scipy_checklist.py
```

Rules:

1. Do not merge if checklist row status/evidence fields are stale.
2. Keep `BadCoefficients` and `test` as `deferred_pythonism` unless maintainers explicitly change policy.
3. Keep existing git safety and non-interactive plotting rules unchanged.
