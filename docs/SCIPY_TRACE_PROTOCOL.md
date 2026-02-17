# SciPy Trace Protocol

This protocol defines how `sci-rs` contributors trace and validate SciPy behavior while implementing feature parity.

## 1. Pinned Local Baseline

Use this local Python baseline when generating evidence:

1. Python `3.12.6`
2. NumPy `2.1.0`
3. SciPy `1.14.1`
4. Matplotlib `3.9.2`
5. Interpreter command: `python`

## 2. Required Static Trace (Every Packet)

For every claimed symbol, record:

1. SciPy docs URL.
2. Source module path and function/class line anchor.
3. Helper-call flow summary that describes which internal functions shape behavior.

This static trace evidence must be present in the packet and checklist row.

## 3. Runtime Intermediate-State Tracing (As Needed)

Runtime tracing is allowed and encouraged when discrepancies remain after static tracing.

Use runtime tracing when:

1. Output mismatch exists but surface API mapping appears correct.
2. Internal helper stages (padding/state/normalization/ordering) may diverge.
3. Numerical behavior differs due to edge handling or axis conventions.

Runtime tracing is optional when:

1. The same algorithm is clearly implemented.
2. Contract metrics pass with accepted thresholds.

## 4. Runtime Trace Artifact Policy

1. Store local trace artifacts under:
   - `target/contracts/<timestamp>/traces/<symbol>/...`
2. Include stage names, key scalar parameters, array lengths/shapes, and short summary stats.
3. Do not commit trace artifacts to git.

## 5. Non-Blocking Plot Policy

Any diagnostic plots for trace/debug must be non-interactive:

1. Use `Agg` backend.
2. Save files only.
3. Never call `plt.show()` in tests/CI workflows.

## 6. Minimum Evidence To Mark Parity

A symbol cannot move to `parity`/`perf`/`ready` without:

1. Static trace fields completed.
2. Contract case IDs recorded.
3. Metrics recorded: Pearson `r`, MAE, RMSE, max-abs.
4. Threshold profile documented in checklist row.

## 7. Suggested Trace Snippet

```python
import inspect
import scipy.signal as signal

fn = signal.find_peaks
print(inspect.getsourcefile(fn))
print(inspect.getsourcelines(fn)[1])
```

This gives module path and line anchor for checklist trace fields.
