#!/usr/bin/env python3
"""Verify docs/SCIPY_CHECKLIST.md is synchronized and structurally valid."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import scipy.signal

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKLIST_PATH = REPO_ROOT / "docs" / "SCIPY_CHECKLIST.md"

MATRIX_BEGIN = "<!-- CHECKLIST_MATRIX_BEGIN -->"
MATRIX_END = "<!-- CHECKLIST_MATRIX_END -->"

DEFERRED_SYMBOLS = {"BadCoefficients", "test"}
STATUS_VALUES = {
    "wishlist",
    "assigned",
    "impl",
    "parity",
    "perf",
    "ready",
    "merged",
    "blocked",
    "deferred_pythonism",
}
REQUIRED_CLASSIFICATIONS = {"required_now", "deferred_pythonism"}

MATRIX_COLUMNS = [
    "symbol",
    "scipy_namespace",
    "kind",
    "required_classification",
    "domain",
    "status",
    "owner",
    "branch",
    "scipy_docs_url",
    "scipy_source_anchor",
    "rust_trait_config_kernel",
    "rust_paths",
    "contract_case_ids",
    "benchmark_ids",
    "parity_threshold_profile",
    "perf_gate_result",
    "notes",
]


def split_markdown_row(line: str) -> List[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def load_expected_symbols() -> set[str]:
    symbols = set()
    for symbol in dir(scipy.signal):
        if symbol.startswith("_"):
            continue
        obj = getattr(scipy.signal, symbol)
        if callable(obj):
            symbols.add(symbol)
    return symbols


def parse_matrix(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"checklist missing: {path}")

    text = path.read_text(encoding="utf-8")
    if MATRIX_BEGIN not in text or MATRIX_END not in text:
        raise RuntimeError("missing checklist matrix markers")

    matrix_block = text.split(MATRIX_BEGIN, 1)[1].split(MATRIX_END, 1)[0]
    table_lines = [line for line in matrix_block.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        raise RuntimeError("checklist matrix table is empty")

    headers = split_markdown_row(table_lines[0])
    if headers != MATRIX_COLUMNS:
        raise RuntimeError(
            "checklist columns do not match expected schema\n"
            f"expected: {MATRIX_COLUMNS}\n"
            f"actual:   {headers}"
        )

    rows: Dict[str, Dict[str, str]] = {}
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            raise RuntimeError(f"malformed table row: {line}")
        row = dict(zip(headers, cells, strict=True))
        symbol = row["symbol"]
        if symbol in rows:
            raise RuntimeError(f"duplicate symbol row: {symbol}")
        rows[symbol] = row

    return rows


def require_nonempty(errors: List[str], row: Dict[str, str], fields: List[str]) -> None:
    symbol = row["symbol"]
    for field in fields:
        if not row[field].strip():
            errors.append(f"{symbol}: field `{field}` is required for status `{row['status']}`")


def verify_rows(rows: Dict[str, Dict[str, str]], expected_symbols: set[str]) -> List[str]:
    errors: List[str] = []

    actual_symbols = set(rows)
    missing = sorted(expected_symbols - actual_symbols)
    extra = sorted(actual_symbols - expected_symbols)

    if missing:
        errors.append(f"missing symbols: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected symbols: {', '.join(extra)}")

    for symbol in sorted(actual_symbols & expected_symbols):
        row = rows[symbol]
        status = row["status"].strip()
        classification = row["required_classification"].strip()

        if row["scipy_namespace"] != "scipy.signal":
            errors.append(f"{symbol}: scipy_namespace must be `scipy.signal`")

        if classification not in REQUIRED_CLASSIFICATIONS:
            errors.append(f"{symbol}: invalid required_classification `{classification}`")

        if status not in STATUS_VALUES:
            errors.append(f"{symbol}: invalid status `{status}`")

        if not row["scipy_docs_url"].strip():
            errors.append(f"{symbol}: missing scipy_docs_url")

        if not row["scipy_source_anchor"].strip():
            errors.append(f"{symbol}: missing scipy_source_anchor")

        if classification == "deferred_pythonism":
            if symbol not in DEFERRED_SYMBOLS:
                errors.append(
                    f"{symbol}: deferred_pythonism not allowed; only {sorted(DEFERRED_SYMBOLS)}"
                )
            if status != "deferred_pythonism":
                errors.append(
                    f"{symbol}: deferred_pythonism classification must use status deferred_pythonism"
                )

        if symbol in DEFERRED_SYMBOLS and classification != "deferred_pythonism":
            errors.append(f"{symbol}: expected deferred_pythonism classification")

        if status == "deferred_pythonism" and classification != "deferred_pythonism":
            errors.append(f"{symbol}: status deferred_pythonism requires deferred classification")

        if status in {"assigned", "impl", "parity", "perf", "ready", "merged", "blocked"}:
            require_nonempty(errors, row, ["owner", "branch"])

        if status in {"parity", "perf", "ready", "merged"}:
            require_nonempty(errors, row, ["contract_case_ids", "parity_threshold_profile"])

        if status in {"perf", "ready", "merged"}:
            require_nonempty(errors, row, ["benchmark_ids", "perf_gate_result"])

        if status in {"merged", "blocked"}:
            require_nonempty(errors, row, ["notes"])

        if status == "merged":
            require_nonempty(errors, row, ["rust_trait_config_kernel", "rust_paths"])

    return errors


def main() -> int:
    try:
        rows = parse_matrix(CHECKLIST_PATH)
    except RuntimeError as err:
        print(f"ERROR: {err}")
        return 1

    expected_symbols = load_expected_symbols()
    errors = verify_rows(rows, expected_symbols)

    if errors:
        print("ERROR: checklist verification failed")
        for error in errors:
            print(f" - {error}")
        return 1

    print(
        "OK: checklist synchronized and valid "
        f"({len(rows)} symbols, deferred={sorted(DEFERRED_SYMBOLS)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
