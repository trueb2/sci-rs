use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array1;
use sci_rs::kernel::KernelLifecycle;
use sci_rs::signal::convolve::{ConvolveConfig, ConvolveKernel, ConvolveMode};
use sci_rs::signal::resample::{resample as resample_baseline, ResampleConfig, ResampleKernel};
use sci_rs::signal::traits::{Convolve1D, Resample1D};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PYTHON_BIN: &str = "python";

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PythonEval {
    output: Vec<f64>,
    avg_ns: f64,
    python_version: String,
    numpy_version: String,
    scipy_version: Option<String>,
    matplotlib_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ContractRow {
    case_id: String,
    pearson_r: f64,
    mae: f64,
    rmse: f64,
    max_abs: f64,
    rust_candidate_ns: f64,
    rust_baseline_ns: f64,
    python_ns: f64,
    speedup_vs_baseline: f64,
    speedup_vs_python: f64,
    overlay_plot: String,
    residual_plot: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContractBundle {
    generated_epoch_seconds: u64,
    python_executable: String,
    python_version: String,
    numpy_version: String,
    scipy_version: String,
    matplotlib_version: String,
    rows: Vec<ContractRow>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("contracts") => run_contracts(),
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo run -p xtask -- contracts");
            Ok(())
        }
    }
}

fn run_contracts() -> Result<()> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let out_dir = PathBuf::from(format!("target/contracts/{ts}"));
    let plots_dir = out_dir.join("plots");
    fs::create_dir_all(&plots_dir).context("creating contract output directories")?;

    let python_bin = detect_python_bin();

    let mut rows = Vec::new();
    let mut case_plot_payload = Vec::new();

    // Case 1: convolution against NumPy.
    {
        let case_id = "convolve_full_f64".to_string();
        let in1: Vec<f64> = (0..512)
            .map(|i| {
                let x = i as f64 / 32.0;
                x.sin() + 0.25 * (2.0 * x).cos()
            })
            .collect();
        let in2: Vec<f64> = (0..63)
            .map(|i| {
                let x = i as f64 / 8.0;
                (-(x * x) / 8.0).exp()
            })
            .collect();

        let kernel = ConvolveKernel::try_new(ConvolveConfig {
            mode: ConvolveMode::Full,
        })
        .context("initializing ConvolveKernel")?;

        let candidate = kernel
            .run_alloc(in1.as_slice(), in2.as_slice())
            .map_err(|e| anyhow!("convolve candidate execution failed: {e}"))?;
        let _ = rust_baseline_convolve(&in1, &in2, ConvolveMode::Full)
            .context("running rust baseline convolution")?;
        let py = python_convolve(&python_bin, &in1, &in2, "full", 500)?;

        ensure_same_length(&case_id, &candidate, &py.output)?;

        let candidate_ns = benchmark_avg_ns(200, || {
            kernel
                .run_alloc(in1.as_slice(), in2.as_slice())
                .map(|_| ())
                .map_err(|e| anyhow!("convolve candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(200, || {
            rust_baseline_convolve(&in1, &in2, ConvolveMode::Full).map(|_| ())
        })?;

        let overlay = plots_dir.join(format!("{case_id}_overlay.png"));
        let residual = plots_dir.join(format!("{case_id}_residual.png"));
        rows.push(build_row(RowBuildInput {
            case_id: &case_id,
            rust_candidate: &candidate,
            python_reference: &py.output,
            rust_candidate_ns: candidate_ns,
            rust_baseline_ns: baseline_ns,
            python_ns: py.avg_ns,
            overlay_plot: &overlay,
            residual_plot: &residual,
        }));
        case_plot_payload.push(json!({
            "case_id": case_id,
            "rust_candidate": candidate,
            "python_reference": py.output,
            "overlay_plot": overlay.to_string_lossy(),
            "residual_plot": residual.to_string_lossy()
        }));
    }

    // Case 2: resample against SciPy.
    {
        let case_id = "resample_f64".to_string();
        let input: Vec<f64> = (0..256)
            .map(|i| {
                let x = i as f64 / 25.0;
                x.sin() + 0.2 * (4.0 * x).sin() + 0.05 * (7.5 * x).cos()
            })
            .collect();
        let target_len = 384usize;

        let kernel = ResampleKernel::try_new(ResampleConfig { target_len })
            .context("initializing ResampleKernel")?;
        let candidate = kernel
            .run_alloc(input.as_slice())
            .map_err(|e| anyhow!("resample candidate execution failed: {e}"))?;
        let _ = resample_baseline(&input, target_len);
        let py = python_resample(&python_bin, &input, target_len, 300)?;

        ensure_same_length(&case_id, &candidate, &py.output)?;

        let candidate_ns = benchmark_avg_ns(150, || {
            kernel
                .run_alloc(input.as_slice())
                .map(|_| ())
                .map_err(|e| anyhow!("resample candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(150, || {
            let _ = resample_baseline(&input, target_len);
            Ok(())
        })?;

        let overlay = plots_dir.join(format!("{case_id}_overlay.png"));
        let residual = plots_dir.join(format!("{case_id}_residual.png"));
        rows.push(build_row(RowBuildInput {
            case_id: &case_id,
            rust_candidate: &candidate,
            python_reference: &py.output,
            rust_candidate_ns: candidate_ns,
            rust_baseline_ns: baseline_ns,
            python_ns: py.avg_ns,
            overlay_plot: &overlay,
            residual_plot: &residual,
        }));
        case_plot_payload.push(json!({
            "case_id": case_id,
            "rust_candidate": candidate,
            "python_reference": py.output,
            "overlay_plot": overlay.to_string_lossy(),
            "residual_plot": residual.to_string_lossy()
        }));
    }

    let version_probe = python_versions(&python_bin)?;
    let report_pdf = out_dir.join("report.pdf");
    generate_plots_and_pdf(&python_bin, &case_plot_payload, &report_pdf)?;

    let bundle = ContractBundle {
        generated_epoch_seconds: ts,
        python_executable: python_bin.to_string_lossy().into_owned(),
        python_version: version_probe.python_version,
        numpy_version: version_probe.numpy_version,
        scipy_version: version_probe
            .scipy_version
            .unwrap_or_else(|| "unknown".to_string()),
        matplotlib_version: version_probe
            .matplotlib_version
            .unwrap_or_else(|| "unknown".to_string()),
        rows,
    };
    write_summary_csv(&out_dir.join("summary.csv"), &bundle.rows)?;
    fs::write(
        out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&bundle).context("serializing summary bundle")?,
    )
    .context("writing summary.json")?;

    println!("Contract artifacts generated in: {}", out_dir.display());
    println!("  - {}", out_dir.join("summary.csv").display());
    println!("  - {}", out_dir.join("summary.json").display());
    println!("  - {}", report_pdf.display());
    println!("  - {}", plots_dir.display());

    Ok(())
}

fn detect_python_bin() -> PathBuf {
    PathBuf::from(DEFAULT_PYTHON_BIN)
}

fn rust_baseline_convolve(in1: &[f64], in2: &[f64], mode: ConvolveMode) -> Result<Vec<f64>> {
    let a = Array1::from_vec(in1.to_vec());
    let b = Array1::from_vec(in2.to_vec());
    let out = sci_rs_core::num_rs::convolve(a.view(), b.view(), mode)
        .map_err(|e| anyhow!("sci-rs-core baseline convolution failed: {e}"))?;
    Ok(out.to_vec())
}

fn python_versions(python_bin: &Path) -> Result<PythonEval> {
    run_python_eval(
        python_bin,
        r#"
import json, sys
import numpy
import scipy
import matplotlib
payload = json.loads(sys.stdin.read())
print(json.dumps({
    "output": [],
    "avg_ns": 0.0,
    "python_version": sys.version.split()[0],
    "numpy_version": numpy.__version__,
    "scipy_version": scipy.__version__,
    "matplotlib_version": matplotlib.__version__
}))
"#,
        json!({}),
    )
}

fn python_convolve(
    python_bin: &Path,
    in1: &[f64],
    in2: &[f64],
    mode: &str,
    iters: usize,
) -> Result<PythonEval> {
    run_python_eval(
        python_bin,
        r#"
import json, sys, time
import numpy as np
payload = json.loads(sys.stdin.read())
a = np.asarray(payload["in1"], dtype=float)
b = np.asarray(payload["in2"], dtype=float)
mode = payload["mode"]
y = np.convolve(a, b, mode=mode)
iters = int(payload["iters"])
t0 = time.perf_counter_ns()
for _ in range(iters):
    np.convolve(a, b, mode=mode)
t1 = time.perf_counter_ns()
print(json.dumps({
    "output": y.tolist(),
    "avg_ns": (t1 - t0) / iters,
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "scipy_version": None,
    "matplotlib_version": None
}))
"#,
        json!({
            "in1": in1,
            "in2": in2,
            "mode": mode,
            "iters": iters
        }),
    )
}

fn python_resample(
    python_bin: &Path,
    input: &[f64],
    target_len: usize,
    iters: usize,
) -> Result<PythonEval> {
    run_python_eval(
        python_bin,
        r#"
import json, sys, time
import numpy as np
import scipy.signal
payload = json.loads(sys.stdin.read())
x = np.asarray(payload["input"], dtype=float)
n = int(payload["target_len"])
y = scipy.signal.resample(x, n)
iters = int(payload["iters"])
t0 = time.perf_counter_ns()
for _ in range(iters):
    scipy.signal.resample(x, n)
t1 = time.perf_counter_ns()
print(json.dumps({
    "output": y.tolist(),
    "avg_ns": (t1 - t0) / iters,
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "matplotlib_version": None
}))
"#,
        json!({
            "input": input,
            "target_len": target_len,
            "iters": iters
        }),
    )
}

fn run_python_eval(
    python_bin: &Path,
    script: &str,
    payload: serde_json::Value,
) -> Result<PythonEval> {
    let mut child = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning python interpreter at {}", python_bin.display()))?;

    {
        let stdin = child.stdin.as_mut().context("opening python stdin")?;
        let payload_bytes = serde_json::to_vec(&payload).context("serializing python payload")?;
        stdin
            .write_all(&payload_bytes)
            .context("writing payload to python stdin")?;
    }

    let output = child
        .wait_with_output()
        .context("waiting for python process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("python execution failed: {stderr}");
    }
    let stdout = String::from_utf8(output.stdout).context("parsing python stdout utf8")?;
    let parsed: PythonEval = serde_json::from_str(stdout.trim()).context("parsing python json")?;
    Ok(parsed)
}

fn ensure_same_length(case_id: &str, a: &[f64], b: &[f64]) -> Result<()> {
    if a.len() != b.len() {
        bail!(
            "case {case_id} has mismatched output lengths: rust={}, python={}",
            a.len(),
            b.len()
        );
    }
    Ok(())
}

fn benchmark_avg_ns<F>(iters: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(start.elapsed().as_nanos() as f64 / iters as f64)
}

struct RowBuildInput<'a> {
    case_id: &'a str,
    rust_candidate: &'a [f64],
    python_reference: &'a [f64],
    rust_candidate_ns: f64,
    rust_baseline_ns: f64,
    python_ns: f64,
    overlay_plot: &'a Path,
    residual_plot: &'a Path,
}

fn build_row(args: RowBuildInput<'_>) -> ContractRow {
    let pearson_r = pearson(args.rust_candidate, args.python_reference);
    let mae = mean_abs_error(args.rust_candidate, args.python_reference);
    let rmse = root_mean_squared_error(args.rust_candidate, args.python_reference);
    let max_abs = max_abs_error(args.rust_candidate, args.python_reference);
    ContractRow {
        case_id: args.case_id.to_string(),
        pearson_r,
        mae,
        rmse,
        max_abs,
        rust_candidate_ns: args.rust_candidate_ns,
        rust_baseline_ns: args.rust_baseline_ns,
        python_ns: args.python_ns,
        speedup_vs_baseline: args.rust_baseline_ns / args.rust_candidate_ns,
        speedup_vs_python: args.python_ns / args.rust_candidate_ns,
        overlay_plot: args.overlay_plot.to_string_lossy().into_owned(),
        residual_plot: args.residual_plot.to_string_lossy().into_owned(),
    }
}

fn mean_abs_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn root_mean_squared_error(a: &[f64], b: &[f64]) -> f64 {
    (a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = *x - mean_a;
        let db = *y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a == 0.0 || var_b == 0.0 {
        if a == b {
            1.0
        } else {
            0.0
        }
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

fn write_summary_csv(path: &Path, rows: &[ContractRow]) -> Result<()> {
    let mut out = String::new();
    out.push_str("case_id,pearson_r,mae,rmse,max_abs,rust_candidate_ns,rust_baseline_ns,python_ns,speedup_vs_baseline,speedup_vs_python,overlay_plot,residual_plot\n");
    for row in rows {
        out.push_str(&format!(
            "{},{:.12},{:.12},{:.12},{:.12},{:.3},{:.3},{:.3},{:.6},{:.6},{},{}\n",
            row.case_id,
            row.pearson_r,
            row.mae,
            row.rmse,
            row.max_abs,
            row.rust_candidate_ns,
            row.rust_baseline_ns,
            row.python_ns,
            row.speedup_vs_baseline,
            row.speedup_vs_python,
            row.overlay_plot,
            row.residual_plot
        ));
    }
    fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

fn generate_plots_and_pdf(
    python_bin: &Path,
    case_payload: &[serde_json::Value],
    report_pdf: &Path,
) -> Result<()> {
    let payload = json!({
        "cases": case_payload,
        "report_pdf": report_pdf.to_string_lossy()
    });
    let script = r#"
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

payload = json.loads(sys.stdin.read())
cases = payload["cases"]
report_pdf = payload["report_pdf"]

with PdfPages(report_pdf) as pdf:
    for case in cases:
        case_id = case["case_id"]
        rust = case["rust_candidate"]
        py = case["python_reference"]
        x = list(range(len(py)))
        residual = [ri - pi for ri, pi in zip(rust, py)]

        fig_overlay = plt.figure(figsize=(10, 4))
        ax_overlay = fig_overlay.add_subplot(1, 1, 1)
        ax_overlay.plot(x, py, label="Python reference", linewidth=1.6)
        ax_overlay.plot(x, rust, label="Rust candidate", linewidth=1.2, alpha=0.8)
        ax_overlay.set_title(f"{case_id} :: overlay")
        ax_overlay.set_xlabel("index")
        ax_overlay.set_ylabel("value")
        ax_overlay.legend()
        fig_overlay.tight_layout()
        fig_overlay.savefig(case["overlay_plot"], dpi=150)
        pdf.savefig(fig_overlay)
        plt.close(fig_overlay)

        fig_residual = plt.figure(figsize=(10, 4))
        ax_residual = fig_residual.add_subplot(1, 1, 1)
        ax_residual.plot(x, residual, label="Rust - Python", linewidth=1.2, color="tab:red")
        ax_residual.set_title(f"{case_id} :: residual")
        ax_residual.set_xlabel("index")
        ax_residual.set_ylabel("error")
        ax_residual.legend()
        fig_residual.tight_layout()
        fig_residual.savefig(case["residual_plot"], dpi=150)
        pdf.savefig(fig_residual)
        plt.close(fig_residual)
"#;

    let mut child = Command::new(python_bin)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning python interpreter at {}", python_bin.display()))?;

    {
        let stdin = child.stdin.as_mut().context("opening python stdin")?;
        let payload_bytes = serde_json::to_vec(&payload).context("serializing plot payload")?;
        stdin
            .write_all(&payload_bytes)
            .context("writing plot payload to python stdin")?;
    }

    let output = child
        .wait_with_output()
        .context("waiting for python plot process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("plot/pdf generation failed: {stderr}");
    }

    Ok(())
}
