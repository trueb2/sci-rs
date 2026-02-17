#!/usr/bin/env python3
"""Generate docs/SCIPY_CHECKLIST.md from local scipy.signal inventory.

The script preserves manual progress/evidence fields from existing checklist rows,
while refreshing inventory-driven fields (symbol/source/docs/domain/classification).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Iterable, List

import scipy
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

MANUAL_COLUMNS = [
    "status",
    "owner",
    "branch",
    "rust_trait_config_kernel",
    "rust_paths",
    "contract_case_ids",
    "benchmark_ids",
    "parity_threshold_profile",
    "perf_gate_result",
    "notes",
]

NEXT_20 = [
    "find_peaks",
    "peak_prominences",
    "peak_widths",
    "argrelextrema",
    "argrelmax",
    "argrelmin",
    "find_peaks_cwt",
    "cwt",
    "upfirdn",
    "resample_poly",
    "decimate",
    "periodogram",
    "welch",
    "csd",
    "coherence",
    "stft",
    "istft",
    "spectrogram",
    "freqz",
    "sosfreqz",
]

CONVOLUTION = {
    "choose_conv_method",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "correlation_lags",
    "deconvolve",
    "fftconvolve",
    "oaconvolve",
    "sepfir2d",
}

FILTERING = {
    "filtfilt",
    "hilbert",
    "hilbert2",
    "lfilter",
    "lfilter_zi",
    "lfiltic",
    "medfilt",
    "medfilt2d",
    "order_filter",
    "savgol_coeffs",
    "savgol_filter",
    "sosfilt",
    "sosfilt_zi",
    "sosfiltfilt",
    "symiirorder1",
    "symiirorder2",
    "vectorstrength",
    "wiener",
}

MULTIRATE = {"decimate", "resample", "resample_poly", "upfirdn"}

PEAK = {
    "argrelextrema",
    "argrelmax",
    "argrelmin",
    "find_peaks",
    "find_peaks_cwt",
    "peak_prominences",
    "peak_widths",
}

SPECTRAL = {
    "check_COLA",
    "check_NOLA",
    "coherence",
    "csd",
    "istft",
    "lombscargle",
    "periodogram",
    "spectrogram",
    "stft",
    "welch",
}

FILTER_DESIGN = {
    "band_stop_obj",
    "bessel",
    "besselap",
    "bilinear",
    "bilinear_zpk",
    "buttap",
    "butter",
    "buttord",
    "cheb1ap",
    "cheb1ord",
    "cheb2ap",
    "cheb2ord",
    "cheby1",
    "cheby2",
    "ellip",
    "ellipap",
    "ellipord",
    "findfreqs",
    "firls",
    "firwin",
    "firwin2",
    "freqs",
    "freqs_zpk",
    "freqz",
    "freqz_zpk",
    "gammatone",
    "group_delay",
    "iircomb",
    "iirdesign",
    "iirfilter",
    "iirnotch",
    "iirpeak",
    "kaiser_atten",
    "kaiser_beta",
    "kaiserord",
    "lp2bp",
    "lp2bp_zpk",
    "lp2bs",
    "lp2bs_zpk",
    "lp2hp",
    "lp2hp_zpk",
    "lp2lp",
    "lp2lp_zpk",
    "minimum_phase",
    "normalize",
    "remez",
    "sos2tf",
    "sos2zpk",
    "sosfreqz",
    "tf2sos",
    "tf2zpk",
    "zpk2sos",
    "zpk2tf",
}

LTI = {
    "StateSpace",
    "TransferFunction",
    "ZerosPolesGain",
    "abcd_normalize",
    "bode",
    "cont2discrete",
    "dbode",
    "dfreqresp",
    "dimpulse",
    "dlsim",
    "dlti",
    "dstep",
    "freqresp",
    "impulse",
    "lsim",
    "lti",
    "place_poles",
    "ss2tf",
    "ss2zpk",
    "step",
    "tf2ss",
    "zpk2ss",
}

WAVEFORMS = {"chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"}

TRANSFORMS = {"CZT", "ShortTimeFFT", "ZoomFFT", "czt", "czt_points", "zoom_fft"}

WAVELETS = {"cascade", "cwt", "daub", "morlet", "morlet2", "qmf", "ricker"}

SPLINES = {
    "cspline1d",
    "cspline1d_eval",
    "cspline2d",
    "gauss_spline",
    "qspline1d",
    "qspline1d_eval",
    "qspline2d",
    "spline_filter",
}

RESIDUE = {"invres", "invresz", "residue", "residuez", "unique_roots"}


def markdown_escape(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", "<br>")


def split_markdown_row(line: str) -> List[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def parse_existing_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    if MATRIX_BEGIN not in text or MATRIX_END not in text:
        return {}

    matrix_block = text.split(MATRIX_BEGIN, 1)[1].split(MATRIX_END, 1)[0]
    table_lines = [line for line in matrix_block.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return {}

    headers = split_markdown_row(table_lines[0])
    rows: Dict[str, Dict[str, str]] = {}

    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        symbol = row.get("symbol", "")
        if symbol:
            rows[symbol] = row

    return rows


def classify_domain(symbol: str, source_file: str) -> str:
    if symbol in CONVOLUTION:
        return "convolution"
    if symbol in FILTERING:
        return "filtering"
    if symbol in MULTIRATE:
        return "multirate"
    if symbol in PEAK:
        return "peak"
    if symbol in SPECTRAL:
        return "spectral"
    if symbol in FILTER_DESIGN:
        return "filter_design"
    if symbol in LTI:
        return "lti"
    if symbol in WAVEFORMS:
        return "waveforms"
    if symbol in TRANSFORMS:
        return "transforms"
    if symbol in WAVELETS:
        return "wavelets"
    if symbol in SPLINES:
        return "splines"
    if symbol in RESIDUE:
        return "residue"

    if source_file in {"_filter_design.py", "_fir_filter_design.py"}:
        return "filter_design"
    if source_file == "_spectral_py.py":
        return "spectral"
    if source_file == "_peak_finding.py":
        return "peak"
    if source_file in {"_ltisys.py", "_lti_conversion.py"}:
        return "lti"
    if source_file == "_signaltools.py":
        return "filtering"
    if source_file == "_upfirdn.py":
        return "multirate"
    if source_file == "_waveforms.py":
        return "waveforms"
    if source_file == "_wavelets.py":
        return "wavelets"
    if source_file == "_bsplines.py":
        return "splines"
    if source_file in {"_czt.py", "_short_time_fft.py"}:
        return "transforms"

    return "helpers"


def callable_inventory() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for symbol in sorted(dir(scipy.signal)):
        if symbol.startswith("_"):
            continue

        obj = getattr(scipy.signal, symbol)
        if not callable(obj):
            continue

        kind = "class" if inspect.isclass(obj) else "function"

        try:
            source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        except (OSError, TypeError):
            source_file = "builtin"

        source_name = Path(source_file).name if source_file != "builtin" else "builtin"

        try:
            _, line = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            line = None

        if source_name == "builtin":
            source_anchor = "builtin"
        elif line is None:
            source_anchor = source_name
        else:
            source_anchor = f"{source_name}:L{line}"

        required_classification = (
            "deferred_pythonism" if symbol in DEFERRED_SYMBOLS else "required_now"
        )

        rows.append(
            {
                "symbol": symbol,
                "scipy_namespace": "scipy.signal",
                "kind": kind,
                "required_classification": required_classification,
                "domain": classify_domain(symbol, source_name),
                "scipy_docs_url": (
                    f"https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.{symbol}.html"
                ),
                "scipy_source_anchor": source_anchor,
            }
        )

    return rows


def merge_rows(
    inventory_rows: Iterable[Dict[str, str]],
    existing_rows: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []

    for inventory_row in inventory_rows:
        symbol = inventory_row["symbol"]
        existing = existing_rows.get(symbol, {})

        row = {key: "" for key in MATRIX_COLUMNS}
        row.update(inventory_row)

        for column in MANUAL_COLUMNS:
            row[column] = existing.get(column, "").strip()

        if row["required_classification"] == "deferred_pythonism":
            row["status"] = "deferred_pythonism"
            row["owner"] = ""
            row["branch"] = ""
        else:
            if not row["status"]:
                row["status"] = "wishlist"
            if row["status"] == "deferred_pythonism":
                row["status"] = "wishlist"

        merged.append(row)

    return merged


def build_counters(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    rows_list = list(rows)
    active_statuses = {"assigned", "impl", "parity", "perf", "ready", "blocked"}

    return {
        "total": len(rows_list),
        "required_now": sum(1 for row in rows_list if row["required_classification"] == "required_now"),
        "deferred_pythonism": sum(
            1 for row in rows_list if row["required_classification"] == "deferred_pythonism"
        ),
        "merged": sum(1 for row in rows_list if row["status"] == "merged"),
        "active": sum(1 for row in rows_list if row["status"] in active_statuses),
    }


def render_table(columns: List[str], rows: Iterable[Dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, sep]
    for row in rows:
        cells = [markdown_escape(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_checklist(rows: List[Dict[str, str]]) -> str:
    rows_by_symbol = {row["symbol"]: row for row in rows}
    counters = build_counters(rows)

    next_rows = [rows_by_symbol[symbol] for symbol in NEXT_20 if symbol in rows_by_symbol]
    next_table_rows = [
        {
            "symbol": row["symbol"],
            "domain": row["domain"],
            "status": row["status"],
            "scipy_source_anchor": row["scipy_source_anchor"],
            "notes": row["notes"],
        }
        for row in next_rows
    ]

    next_columns = ["symbol", "domain", "status", "scipy_source_anchor", "notes"]

    return "\n".join(
        [
            "# SciPy Checklist",
            "",
            "SciPy-wide feature completion backlog for `sci-rs`, starting with `scipy.signal`.",
            "This checklist is the canonical source of truth for assignment, verification, and merge readiness.",
            "",
            "## Mission and Scope",
            "",
            "1. Complete useful computational SciPy APIs in `sci-rs` with trait-first kernels.",
            "2. Start with `scipy.signal` coverage, then continue through remaining SciPy domains.",
            "3. Keep Python/SciPy as the local correctness and behavior reference oracle.",
            "4. Defer only non-computational pythonisms by default (`BadCoefficients`, `test`).",
            "",
            f"Generated from local SciPy `{scipy.__version__}` by `scripts/generate_scipy_checklist_inventory.py`.",
            "",
            "## Counters",
            "",
            f"- total callables tracked: `{counters['total']}`",
            f"- required-now count: `{counters['required_now']}`",
            f"- deferred-pythonism count: `{counters['deferred_pythonism']}`",
            f"- merged count: `{counters['merged']}`",
            f"- active count: `{counters['active']}`",
            "",
            "## Status Legend",
            "",
            "- `wishlist`: not yet claimed",
            "- `assigned`: claimed and branch opened",
            "- `impl`: implementation in progress",
            "- `parity`: implementation done; parity evidence in progress",
            "- `perf`: parity passed; performance evidence in progress",
            "- `ready`: all checks passed; merge-ready",
            "- `merged`: merged into active integration branch",
            "- `blocked`: waiting on dependency/decision",
            "- `deferred_pythonism`: intentionally deferred non-computational item",
            "",
            "## Signal-First Priority Queue (Next 20)",
            "",
            render_table(next_columns, next_table_rows),
            "",
            "## Full Callable Matrix",
            "",
            MATRIX_BEGIN,
            render_table(MATRIX_COLUMNS, rows),
            MATRIX_END,
            "",
        ]
    )


def main() -> int:
    existing_rows = parse_existing_rows(CHECKLIST_PATH)
    inventory_rows = callable_inventory()
    merged_rows = merge_rows(inventory_rows, existing_rows)

    checklist_markdown = render_checklist(merged_rows)
    CHECKLIST_PATH.write_text(checklist_markdown, encoding="utf-8")

    print(f"wrote {CHECKLIST_PATH}")
    print(f"rows: {len(merged_rows)}")
    print(f"scipy: {scipy.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
