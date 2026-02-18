use anyhow::{anyhow, bail, Context, Result};
use sci_rs::kernel::KernelLifecycle;
use sci_rs::linalg::{companion_dyn, CompanionBuild1D, CompanionConfig, CompanionKernel};
use sci_rs::signal::convolve::{
    ConvolveConfig, ConvolveKernel, ConvolveMode, CorrelateConfig, CorrelateKernel,
};
use sci_rs::signal::filter::design::{
    ButterConfig, ButterKernel, DigitalFilter, FilterBandType, FilterOutputType, FilterType,
    FirWinConfig, FirWinKernel, IirFilterConfig, IirFilterKernel, Sos,
};
use sci_rs::signal::filter::{
    FiltFiltConfig, FiltFiltKernel, FiltFiltPad, LFilterConfig, LFilterKernel, LFilterZiConfig,
    LFilterZiKernel, SavgolCoeffsConfig, SavgolCoeffsKernel, SavgolFilterConfig,
    SavgolFilterKernel, SosFiltConfig, SosFiltFiltConfig, SosFiltFiltKernel, SosFiltKernel,
    SosFiltZiConfig, SosFiltZiKernel,
};
use sci_rs::signal::multirate::{
    DecimateConfig, DecimateKernel, ResamplePolyConfig, ResamplePolyKernel, UpFirDnConfig,
    UpFirDnKernel,
};
use sci_rs::signal::peak::{
    ArgRelExtremaConfig, ArgRelExtremaKernel, CwtConfig, CwtKernel, FindPeaksConfig,
    FindPeaksCwtConfig, FindPeaksCwtKernel, FindPeaksKernel, PeakProminencesConfig,
    PeakProminencesKernel, PeakWidthsConfig, PeakWidthsKernel,
};
use sci_rs::signal::resample::{ResampleConfig, ResampleKernel};
use sci_rs::signal::spectral::{
    CoherenceConfig, CoherenceKernel, CsdConfig, CsdKernel, FreqzConfig, FreqzKernel, IstftConfig,
    IstftKernel, PeriodogramConfig, PeriodogramKernel, SosFreqzConfig, SosFreqzKernel,
    SpectrogramConfig, SpectrogramKernel, SpectrogramResult, StftConfig, StftKernel, StftResult,
    WelchConfig, WelchKernel,
};
use sci_rs::signal::traits::{
    ArgRelExtrema1D, ChirpWave1D, Coherence1D, Convolve1D, Correlate1D, Csd1D, Cwt1D, Decimate1D,
    FiltFilt1D, FindPeaks1D, FindPeaksCwt1D, FirWinDesign, Freqz1D, GaussPulseWave1D, IirDesign,
    Istft1D, LFilter1D, LFilterZiDesign1D, PeakProminence1D, PeakWidths1D, Periodogram1D,
    Resample1D, ResamplePoly1D, SavgolCoeffsDesign, SavgolFilter1D, SawtoothWave1D, SosFilt1D,
    SosFiltFilt1D, SosFiltZiDesign1D, SosFreqz1D, Spectrogram1D, SquareWave1D, Stft1D,
    SweepPolyWave1D, UnitImpulse1D, UpFirDn1D, WelchPsd1D, WindowGenerate,
};
use sci_rs::signal::wave::{
    ChirpConfig, ChirpKernel, ChirpMethod, GaussPulseConfig, GaussPulseKernel, SawtoothWaveConfig,
    SawtoothWaveKernel, SquareWaveConfig, SquareWaveKernel, SweepPolyConfig, SweepPolyKernel,
    UnitImpulseConfig, UnitImpulseKernel,
};
use sci_rs::signal::windows::{WindowBuilderOwned, WindowConfig, WindowKernel};
use sci_rs::stats::{
    mean as mean_baseline, median as median_baseline, median_abs_deviation as mad_baseline,
    mod_zscore as mod_zscore_baseline, stdev as stdev_baseline, variance as variance_baseline,
    zscore as zscore_baseline, MadKernel, MadReduce1D, MeanKernel, MeanReduce1D, MedianKernel,
    MedianReduce1D, ModZScoreKernel, ModZScoreNormalize1D, StatsConfig, StdevKernel, StdevReduce1D,
    VarianceKernel, VarianceReduce1D, ZScoreKernel, ZScoreNormalize1D,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PYTHON_BIN: &str = "python";

const PY_SIGNAL_SCRIPT: &str = r#"
import json
import sys
import time
import numpy as np
import scipy.signal
import scipy.linalg
import scipy.stats

env = json.loads(sys.stdin.read())
op = env["op"]
iters = int(env["iters"])
p = env["payload"]

def _as_array(key):
    return np.asarray(p[key], dtype=float)

def _flat(v):
    return np.asarray(v, dtype=float).reshape(-1)

def _flat_complex(v):
    z = np.asarray(v, dtype=np.complex128).reshape(-1)
    out = np.empty(z.size * 2, dtype=float)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out

def _compute():
    if op == "convolve":
        return np.convolve(_as_array("in1"), _as_array("in2"), mode=p["mode"])
    if op == "correlate":
        return np.correlate(_as_array("in1"), _as_array("in2"), mode=p["mode"])
    if op == "resample":
        return scipy.signal.resample(_as_array("input"), int(p["target_len"]))
    if op == "upfirdn":
        return scipy.signal.upfirdn(
            _as_array("h"),
            _as_array("x"),
            up=int(p["up"]),
            down=int(p["down"]),
        )
    if op == "resample_poly":
        return scipy.signal.resample_poly(_as_array("x"), int(p["up"]), int(p["down"]))
    if op == "decimate":
        return scipy.signal.decimate(
            _as_array("x"),
            int(p["q"]),
            ftype="fir",
            zero_phase=False,
        )
    if op == "square":
        return scipy.signal.square(_as_array("t"), duty=float(p["duty"]))
    if op == "sawtooth":
        return scipy.signal.sawtooth(_as_array("t"), width=float(p["width"]))
    if op == "chirp":
        return scipy.signal.chirp(
            _as_array("t"),
            f0=float(p["f0"]),
            t1=float(p["t1"]),
            f1=float(p["f1"]),
            method=p["method"],
            phi=float(p["phi_deg"]),
            vertex_zero=bool(p["vertex_zero"]),
        )
    if op == "gausspulse":
        return scipy.signal.gausspulse(
            _as_array("t"),
            fc=float(p["fc"]),
            bw=float(p["bw"]),
            bwr=float(p["bwr"]),
        )
    if op == "sweep_poly":
        return scipy.signal.sweep_poly(
            _as_array("t"),
            np.asarray(p["poly"], dtype=float),
            phi=float(p["phi_deg"]),
        )
    if op == "unit_impulse":
        idx = p.get("idx")
        return scipy.signal.unit_impulse(int(p["len"]), idx if idx is None else int(idx))
    if op == "lfilter":
        return scipy.signal.lfilter(_as_array("b"), _as_array("a"), _as_array("x"))
    if op == "filtfilt":
        pad = p.get("pad")
        if pad is None:
            return scipy.signal.filtfilt(_as_array("b"), _as_array("a"), _as_array("x"), padtype=None, padlen=0)
        return scipy.signal.filtfilt(
            _as_array("b"),
            _as_array("a"),
            _as_array("x"),
            padtype=pad["pad_type"],
            padlen=int(pad["pad_len"]),
        )
    if op == "sosfilt":
        sos = _as_array("sos").reshape((-1, 6))
        return scipy.signal.sosfilt(sos, _as_array("x"))
    if op == "sosfiltfilt":
        sos = _as_array("sos").reshape((-1, 6))
        return scipy.signal.sosfiltfilt(sos, _as_array("x"))
    if op == "savgol_filter":
        return scipy.signal.savgol_filter(
            _as_array("x"),
            int(p["window_length"]),
            int(p["polyorder"]),
            deriv=int(p["deriv"]),
            delta=float(p["delta"]),
            mode="nearest",
        )
    if op == "savgol_coeffs":
        return scipy.signal.savgol_coeffs(
            int(p["window_length"]),
            int(p["polyorder"]),
            deriv=int(p["deriv"]),
            delta=float(p["delta"]),
        )
    if op == "lfilter_zi":
        return scipy.signal.lfilter_zi(_as_array("b"), _as_array("a"))
    if op == "sosfilt_zi":
        sos = _as_array("sos").reshape((-1, 6))
        return scipy.signal.sosfilt_zi(sos)
    if op == "argrelextrema":
        comparator = p["comparator"]
        if comparator == "gt":
            cmp = np.greater
        elif comparator == "lt":
            cmp = np.less
        else:
            raise RuntimeError(f"unsupported argrelextrema comparator: {comparator}")
        return scipy.signal.argrelextrema(_as_array("x"), cmp, order=int(p["order"]))[0]
    if op == "argrelmax":
        return scipy.signal.argrelmax(_as_array("x"), order=int(p["order"]))[0]
    if op == "argrelmin":
        return scipy.signal.argrelmin(_as_array("x"), order=int(p["order"]))[0]
    if op == "find_peaks":
        peaks, _ = scipy.signal.find_peaks(
            _as_array("x"),
            height=(None if p["height"] is None else float(p["height"])),
            distance=(None if p["distance"] is None else int(p["distance"])),
        )
        return peaks
    if op == "peak_prominences":
        peaks = np.asarray(p["peaks"], dtype=int)
        prominences, left_bases, right_bases = scipy.signal.peak_prominences(_as_array("x"), peaks)
        return np.concatenate([prominences, left_bases.astype(float), right_bases.astype(float)])
    if op == "peak_widths":
        peaks = np.asarray(p["peaks"], dtype=int)
        widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            _as_array("x"),
            peaks,
            rel_height=float(p["rel_height"]),
        )
        return np.concatenate([widths, width_heights, left_ips, right_ips])
    if op == "cwt":
        widths = np.asarray(p["widths"], dtype=float)
        return scipy.signal.cwt(_as_array("x"), scipy.signal.ricker, widths).reshape(-1)
    if op == "find_peaks_cwt":
        widths = np.asarray(p["widths"], dtype=float)
        return np.asarray(scipy.signal.find_peaks_cwt(_as_array("x"), widths), dtype=float)
    if op == "periodogram":
        freqs, pxx = scipy.signal.periodogram(
            _as_array("x"),
            fs=float(p["fs"]),
            window="boxcar",
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        return np.concatenate([freqs, pxx])
    if op == "welch":
        freqs, pxx = scipy.signal.welch(
            _as_array("x"),
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        return np.concatenate([freqs, pxx])
    if op == "csd":
        freqs, pxy = scipy.signal.csd(
            _as_array("x"),
            _as_array("y"),
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        return np.concatenate([freqs, _flat_complex(pxy)])
    if op == "coherence":
        freqs, cxy = scipy.signal.coherence(
            _as_array("x"),
            _as_array("y"),
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            detrend=False,
        )
        return np.concatenate([freqs, cxy])
    if op == "stft":
        freqs, times, zxx = scipy.signal.stft(
            _as_array("x"),
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
        )
        return np.concatenate([freqs, times, _flat_complex(zxx)])
    if op == "istft":
        zxx_flat = np.asarray(p["zxx_re_im"], dtype=float)
        zxx_vec = zxx_flat[0::2] + 1j * zxx_flat[1::2]
        zxx = zxx_vec.reshape((int(p["n_freq"]), int(p["n_frames"])))
        times, y = scipy.signal.istft(
            zxx,
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            input_onesided=True,
            boundary=False,
        )
        return np.concatenate([times, y])
    if op == "spectrogram":
        freqs, times, zxx = scipy.signal.stft(
            _as_array("x"),
            fs=float(p["fs"]),
            window="hann",
            nperseg=int(p["nperseg"]),
            noverlap=int(p["noverlap"]),
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
        )
        sxx = np.abs(zxx) ** 2
        return np.concatenate([freqs, times, sxx.reshape(-1)])
    if op == "freqz":
        w, h = scipy.signal.freqz(_as_array("b"), _as_array("a"), worN=int(p["wor_n"]))
        return np.concatenate([w, _flat_complex(h)])
    if op == "sosfreqz":
        sos = _as_array("sos").reshape((-1, 6))
        w, h = scipy.signal.sosfreqz(sos, worN=int(p["wor_n"]))
        return np.concatenate([w, _flat_complex(h)])
    if op == "window":
        return scipy.signal.get_window(p["window"], int(p["nx"]), fftbins=bool(p["fftbins"]))
    if op == "mean":
        return np.asarray([np.mean(_as_array("x"))], dtype=float)
    if op == "variance":
        return np.asarray([np.var(_as_array("x"), ddof=0)], dtype=float)
    if op == "stdev":
        return np.asarray([np.std(_as_array("x"), ddof=0)], dtype=float)
    if op == "median":
        return np.asarray([np.median(_as_array("x"))], dtype=float)
    if op == "mad":
        x = _as_array("x")
        med = np.median(x)
        return np.asarray([np.median(np.abs(x - med))], dtype=float)
    if op == "zscore":
        return scipy.stats.zscore(_as_array("x"), axis=None, ddof=0)
    if op == "mod_zscore":
        x = _as_array("x")
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return (x - med) * 0.6745 / mad
    if op == "companion":
        return scipy.linalg.companion(_as_array("coeffs")).reshape(-1)
    if op == "firwin":
        return scipy.signal.firwin(
            int(p["numtaps"]),
            _as_array("cutoff"),
            pass_zero=bool(p["pass_zero"]),
            scale=bool(p["scale"]),
            fs=float(p["fs"]),
            window=p["window"],
        )
    if op == "butter_ba":
        b, a = scipy.signal.butter(
            int(p["order"]),
            _as_array("wn"),
            btype=p["btype"],
            analog=bool(p["analog"]),
            output="ba",
            fs=float(p["fs"]),
        )
        return np.concatenate([b, a])
    if op == "iirfilter_ba":
        b, a = scipy.signal.iirfilter(
            int(p["order"]),
            _as_array("wn"),
            rp=(None if p["rp"] is None else float(p["rp"])),
            rs=(None if p["rs"] is None else float(p["rs"])),
            btype=p["btype"],
            analog=bool(p["analog"]),
            ftype=p["ftype"],
            output="ba",
            fs=(None if p["fs"] is None else float(p["fs"])),
        )
        return np.concatenate([b, a])

    raise RuntimeError(f"unsupported op: {op}")

y = _flat(_compute())

t0 = time.perf_counter_ns()
for _ in range(iters):
    _compute()
t1 = time.perf_counter_ns()

print(json.dumps({
    "output": y.tolist(),
    "avg_ns": (t1 - t0) / max(iters, 1),
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "matplotlib_version": None
}))
"#;

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

    // Shared synthetic input for many 1D cases.
    let signal: Vec<f64> = (0..512)
        .map(|i| {
            let x = i as f64 / 27.0;
            x.sin() + 0.35 * (2.3 * x).cos() + 0.1 * (7.0 * x).sin()
        })
        .collect();

    // Convolution
    {
        let case_id = "convolve_full_f64";
        let in1 = signal.iter().copied().take(256).collect::<Vec<_>>();
        let in2: Vec<f64> = (0..63)
            .map(|i| {
                let x = i as f64 / 8.0;
                (-(x * x) / 8.0).exp()
            })
            .collect();

        let kernel = ConvolveKernel::try_new(ConvolveConfig {
            mode: ConvolveMode::Full,
        })?;
        let baseline_kernel = ConvolveKernel::try_new(ConvolveConfig {
            mode: ConvolveMode::Full,
        })?;
        let candidate = kernel
            .run_alloc(in1.as_slice(), in2.as_slice())
            .map_err(|e| anyhow!("convolve candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(in1.as_slice(), in2.as_slice())
            .map_err(|e| anyhow!("convolve baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "convolve",
            json!({ "in1": in1, "in2": in2, "mode": "full" }),
            200,
        )?;

        let candidate_ns = benchmark_avg_ns(120, || {
            kernel
                .run_alloc(&in1, &in2)
                .map(|_| ())
                .map_err(|e| anyhow!("convolve candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(120, || {
            baseline_kernel
                .run_alloc(&in1, &in2)
                .map(|_| ())
                .map_err(|e| anyhow!("convolve baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Correlation
    {
        let case_id = "correlate_full_f64";
        let in1 = signal.iter().copied().take(256).collect::<Vec<_>>();
        let in2 = signal.iter().copied().skip(50).take(63).collect::<Vec<_>>();

        let kernel = CorrelateKernel::try_new(CorrelateConfig {
            mode: ConvolveMode::Full,
        })?;
        let baseline_kernel = CorrelateKernel::try_new(CorrelateConfig {
            mode: ConvolveMode::Full,
        })?;
        let candidate = kernel
            .run_alloc(in1.as_slice(), in2.as_slice())
            .map_err(|e| anyhow!("correlate candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(in1.as_slice(), in2.as_slice())
            .map_err(|e| anyhow!("correlate baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "correlate",
            json!({ "in1": in1, "in2": in2, "mode": "full" }),
            200,
        )?;

        let candidate_ns = benchmark_avg_ns(120, || {
            kernel
                .run_alloc(&in1, &in2)
                .map(|_| ())
                .map_err(|e| anyhow!("correlate candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(120, || {
            baseline_kernel
                .run_alloc(&in1, &in2)
                .map(|_| ())
                .map_err(|e| anyhow!("correlate baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Resample
    {
        let case_id = "resample_f64";
        let input = signal.iter().copied().take(256).collect::<Vec<_>>();
        let target_len = 384usize;

        let kernel = ResampleKernel::try_new(ResampleConfig { target_len })?;
        let baseline_kernel = ResampleKernel::try_new(ResampleConfig { target_len })?;
        let candidate = kernel
            .run_alloc(input.as_slice())
            .map_err(|e| anyhow!("resample candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(input.as_slice())
            .map_err(|e| anyhow!("resample baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "resample",
            json!({ "input": input, "target_len": target_len }),
            100,
        )?;

        let candidate_ns = benchmark_avg_ns(80, || {
            kernel
                .run_alloc(&input)
                .map(|_| ())
                .map_err(|e| anyhow!("resample candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(80, || {
            baseline_kernel
                .run_alloc(&input)
                .map(|_| ())
                .map_err(|e| anyhow!("resample baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // upfirdn
    {
        let case_id = "upfirdn_f64";
        let x = signal.iter().copied().take(192).collect::<Vec<_>>();
        let h = vec![0.125f64, 0.5, 0.75, 0.5, 0.125];
        let up = 2usize;
        let down = 3usize;

        let kernel = UpFirDnKernel::try_new(UpFirDnConfig {
            h: h.clone(),
            up,
            down,
        })?;
        let baseline_kernel = UpFirDnKernel::try_new(UpFirDnConfig {
            h: h.clone(),
            up,
            down,
        })?;
        let mut candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("upfirdn candidate execution failed: {e}"))?;
        let mut baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("upfirdn baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "upfirdn",
            json!({ "h": h, "x": x, "up": up, "down": down }),
            180,
        )?;
        if candidate.len() != py.output.len() {
            candidate.truncate(py.output.len());
            baseline.truncate(py.output.len());
        }

        let candidate_ns = benchmark_avg_ns(140, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("upfirdn candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("upfirdn baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // resample_poly
    {
        let case_id = "resample_poly_f64";
        let x = signal.iter().copied().take(192).collect::<Vec<_>>();
        let up = 3usize;
        let down = 2usize;

        let kernel = ResamplePolyKernel::try_new(ResamplePolyConfig { up, down })?;
        let baseline_kernel = ResamplePolyKernel::try_new(ResamplePolyConfig { up, down })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("resample_poly candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("resample_poly baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "resample_poly",
            json!({ "x": x, "up": up, "down": down }),
            160,
        )?;

        let candidate_ns = benchmark_avg_ns(120, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("resample_poly candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(120, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("resample_poly baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // decimate
    {
        let case_id = "decimate_q3_f64";
        let x = signal.iter().copied().take(240).collect::<Vec<_>>();
        let q = 3usize;

        let kernel = DecimateKernel::try_new(DecimateConfig { q })?;
        let baseline_kernel = DecimateKernel::try_new(DecimateConfig { q })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("decimate candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("decimate baseline execution failed: {e}"))?;
        let py = python_signal_eval(&python_bin, "decimate", json!({ "x": x, "q": q }), 150)?;

        let candidate_ns = benchmark_avg_ns(120, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("decimate candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(120, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("decimate baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // argrelextrema
    {
        let case_id = "argrelextrema_order1_gt_f64";
        let x = vec![0.0f64, 1.0, 0.0, -1.0, 0.0, 2.0, 1.0, 0.0, 1.4, 0.2, 0.0];
        let order = 1usize;

        let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a > b,
        })?;
        let baseline_kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a > b,
        })?;
        let candidate = usize_vec_to_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelextrema candidate execution failed: {e}"))?,
        );
        let baseline = usize_vec_to_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelextrema baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "argrelextrema",
            json!({ "x": x, "order": order, "comparator": "gt" }),
            300,
        )?;

        let candidate_ns = benchmark_avg_ns(220, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelextrema candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(220, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelextrema baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // argrelmax
    {
        let case_id = "argrelmax_order1_f64";
        let x = vec![0.0f64, 1.0, 0.0, -1.0, 0.0, 2.0, 1.0, 0.0, 1.4, 0.2, 0.0];
        let order = 1usize;

        let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a > b,
        })?;
        let baseline_kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a > b,
        })?;
        let candidate = usize_vec_to_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelmax candidate execution failed: {e}"))?,
        );
        let baseline = usize_vec_to_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelmax baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "argrelmax",
            json!({ "x": x, "order": order }),
            300,
        )?;

        let candidate_ns = benchmark_avg_ns(220, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelmax candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(220, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelmax baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // argrelmin
    {
        let case_id = "argrelmin_order1_f64";
        let x = vec![0.0f64, 1.0, 0.0, -1.0, 0.0, 2.0, 1.0, 0.0, 1.4, 0.2, 0.0];
        let order = 1usize;

        let kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a < b,
        })?;
        let baseline_kernel = ArgRelExtremaKernel::try_new(ArgRelExtremaConfig {
            order,
            comparator: |a: f64, b: f64| a < b,
        })?;
        let candidate = usize_vec_to_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelmin candidate execution failed: {e}"))?,
        );
        let baseline = usize_vec_to_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("argrelmin baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "argrelmin",
            json!({ "x": x, "order": order }),
            300,
        )?;

        let candidate_ns = benchmark_avg_ns(220, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelmin candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(220, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("argrelmin baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // find_peaks
    {
        let case_id = "find_peaks_height_distance_f64";
        let x = vec![0.0f64, 1.0, 0.1, 0.9, 0.0, 2.0, 0.0, 1.5, 0.2, 0.0];
        let height = Some(0.5f64);
        let distance = Some(3usize);

        let kernel = FindPeaksKernel::try_new(FindPeaksConfig { height, distance })?;
        let baseline_kernel = FindPeaksKernel::try_new(FindPeaksConfig { height, distance })?;
        let candidate = usize_vec_to_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("find_peaks candidate execution failed: {e}"))?,
        );
        let baseline = usize_vec_to_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("find_peaks baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "find_peaks",
            json!({ "x": x, "height": height, "distance": distance }),
            260,
        )?;

        let candidate_ns = benchmark_avg_ns(200, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("find_peaks candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(200, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("find_peaks baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // peak_prominences
    {
        let case_id = "peak_prominences_f64";
        let x = vec![0.0f64, 1.0, 0.2, 0.8, 0.1, 2.0, 0.0, 1.5, 0.0];
        let peaks = vec![1usize, 5usize, 7usize];

        let kernel = PeakProminencesKernel::try_new(PeakProminencesConfig)?;
        let baseline_kernel = PeakProminencesKernel::try_new(PeakProminencesConfig)?;
        let candidate_out = kernel
            .run_alloc(&x, &peaks)
            .map_err(|e| anyhow!("peak_prominences candidate execution failed: {e}"))?;
        let baseline_out = baseline_kernel
            .run_alloc(&x, &peaks)
            .map_err(|e| anyhow!("peak_prominences baseline execution failed: {e}"))?;

        let mut candidate = candidate_out.prominences.clone();
        candidate.extend(candidate_out.left_bases.iter().map(|&v| v as f64));
        candidate.extend(candidate_out.right_bases.iter().map(|&v| v as f64));

        let mut baseline = baseline_out.prominences.clone();
        baseline.extend(baseline_out.left_bases.iter().map(|&v| v as f64));
        baseline.extend(baseline_out.right_bases.iter().map(|&v| v as f64));

        let py = python_signal_eval(
            &python_bin,
            "peak_prominences",
            json!({ "x": x, "peaks": peaks }),
            220,
        )?;

        let candidate_ns = benchmark_avg_ns(180, || {
            kernel
                .run_alloc(&x, &peaks)
                .map(|_| ())
                .map_err(|e| anyhow!("peak_prominences candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            baseline_kernel
                .run_alloc(&x, &peaks)
                .map(|_| ())
                .map_err(|e| anyhow!("peak_prominences baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // peak_widths
    {
        let case_id = "peak_widths_rel_height_f64";
        let x = vec![0.0f64, 1.0, 0.2, 0.8, 0.1, 2.0, 0.0, 1.5, 0.0];
        let peaks = vec![1usize, 5usize, 7usize];
        let rel_height = 0.5f64;

        let kernel = PeakWidthsKernel::try_new(PeakWidthsConfig { rel_height })?;
        let baseline_kernel = PeakWidthsKernel::try_new(PeakWidthsConfig { rel_height })?;
        let candidate_out = kernel
            .run_alloc(&x, &peaks)
            .map_err(|e| anyhow!("peak_widths candidate execution failed: {e}"))?;
        let baseline_out = baseline_kernel
            .run_alloc(&x, &peaks)
            .map_err(|e| anyhow!("peak_widths baseline execution failed: {e}"))?;

        let mut candidate = candidate_out.widths.clone();
        candidate.extend_from_slice(&candidate_out.width_heights);
        candidate.extend_from_slice(&candidate_out.left_ips);
        candidate.extend_from_slice(&candidate_out.right_ips);

        let mut baseline = baseline_out.widths.clone();
        baseline.extend_from_slice(&baseline_out.width_heights);
        baseline.extend_from_slice(&baseline_out.left_ips);
        baseline.extend_from_slice(&baseline_out.right_ips);

        let py = python_signal_eval(
            &python_bin,
            "peak_widths",
            json!({ "x": x, "peaks": peaks, "rel_height": rel_height }),
            220,
        )?;

        let candidate_ns = benchmark_avg_ns(180, || {
            kernel
                .run_alloc(&x, &peaks)
                .map(|_| ())
                .map_err(|e| anyhow!("peak_widths candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            baseline_kernel
                .run_alloc(&x, &peaks)
                .map(|_| ())
                .map_err(|e| anyhow!("peak_widths baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // cwt
    {
        let case_id = "cwt_ricker_f64";
        let x = signal.iter().copied().take(128).collect::<Vec<_>>();
        let widths = vec![1usize, 2, 3, 4, 6];

        let kernel = CwtKernel::try_new(CwtConfig {
            widths: widths.clone(),
        })?;
        let baseline_kernel = CwtKernel::try_new(CwtConfig {
            widths: widths.clone(),
        })?;
        let candidate = flatten_nested_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("cwt candidate execution failed: {e}"))?,
        );
        let baseline = flatten_nested_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("cwt baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(&python_bin, "cwt", json!({ "x": x, "widths": widths }), 80)?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("cwt candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("cwt baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // find_peaks_cwt
    {
        let case_id = "find_peaks_cwt_f64";
        let x = signal.iter().copied().take(128).collect::<Vec<_>>();
        let widths = vec![1usize, 2, 3, 4];

        let kernel = FindPeaksCwtKernel::try_new(FindPeaksCwtConfig {
            widths: widths.clone(),
        })?;
        let baseline_kernel = FindPeaksCwtKernel::try_new(FindPeaksCwtConfig {
            widths: widths.clone(),
        })?;
        let mut candidate = usize_vec_to_f64(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("find_peaks_cwt candidate execution failed: {e}"))?,
        );
        let mut baseline = usize_vec_to_f64(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("find_peaks_cwt baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "find_peaks_cwt",
            json!({ "x": x, "widths": widths }),
            120,
        )?;
        if candidate.len() != py.output.len() {
            if candidate.len() < py.output.len() {
                candidate.resize(py.output.len(), -1.0);
                baseline.resize(py.output.len(), -1.0);
            } else {
                candidate.truncate(py.output.len());
                baseline.truncate(py.output.len());
            }
        }

        let candidate_ns = benchmark_avg_ns(100, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("find_peaks_cwt candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(100, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("find_peaks_cwt baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Square wave
    {
        let case_id = "square_wave_f64";
        let duty = 0.37f64;
        let t: Vec<f64> = (0..512).map(|i| (i as f64 - 128.0) / 16.0).collect();

        let kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty })?;
        let baseline_kernel = SquareWaveKernel::try_new(SquareWaveConfig { duty })?;
        let candidate = kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("square candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("square baseline execution failed: {e}"))?;
        let py = python_signal_eval(&python_bin, "square", json!({ "t": t, "duty": duty }), 250)?;

        let candidate_ns = benchmark_avg_ns(200, || {
            kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("square candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(200, || {
            baseline_kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("square baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Sawtooth wave
    {
        let case_id = "sawtooth_wave_f64";
        let width = 0.37f64;
        let t: Vec<f64> = (0..512).map(|i| (i as f64 - 128.0) / 16.0).collect();

        let kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width })?;
        let baseline_kernel = SawtoothWaveKernel::try_new(SawtoothWaveConfig { width })?;
        let candidate = kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("sawtooth candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("sawtooth baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "sawtooth",
            json!({ "t": t, "width": width }),
            250,
        )?;

        let candidate_ns = benchmark_avg_ns(200, || {
            kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("sawtooth candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(200, || {
            baseline_kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("sawtooth baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Chirp wave (linear)
    {
        let case_id = "chirp_linear_f64";
        let f0 = 2.0f64;
        let t1 = 4.0f64;
        let f1 = 12.0f64;
        let phi_deg = 15.0f64;
        let vertex_zero = true;
        let method = ChirpMethod::Linear;
        let py_method = "linear";
        let t: Vec<f64> = (0..512).map(|i| i as f64 / 128.0).collect();

        let kernel = ChirpKernel::try_new(ChirpConfig {
            f0,
            t1,
            f1,
            method,
            phi_deg,
            vertex_zero,
        })?;
        let baseline_kernel = ChirpKernel::try_new(ChirpConfig {
            f0,
            t1,
            f1,
            method,
            phi_deg,
            vertex_zero,
        })?;
        let candidate = kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("chirp candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("chirp baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "chirp",
            json!({
                "t": t,
                "f0": f0,
                "t1": t1,
                "f1": f1,
                "method": py_method,
                "phi_deg": phi_deg,
                "vertex_zero": vertex_zero,
            }),
            180,
        )?;

        let candidate_ns = benchmark_avg_ns(140, || {
            kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("chirp candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            baseline_kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("chirp baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Gaussian pulse (in-phase)
    {
        let case_id = "gausspulse_f64";
        let fc = 5.0f64;
        let bw = 0.5f64;
        let bwr = -6.0f64;
        let t: Vec<f64> = (0..512).map(|i| (i as f64 - 256.0) / 128.0).collect();

        let kernel = GaussPulseKernel::try_new(GaussPulseConfig { fc, bw, bwr })?;
        let baseline_kernel = GaussPulseKernel::try_new(GaussPulseConfig { fc, bw, bwr })?;
        let candidate = kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("gausspulse candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("gausspulse baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "gausspulse",
            json!({
                "t": t,
                "fc": fc,
                "bw": bw,
                "bwr": bwr,
            }),
            220,
        )?;

        let candidate_ns = benchmark_avg_ns(180, || {
            kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("gausspulse candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            baseline_kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("gausspulse baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Polynomial sweep
    {
        let case_id = "sweep_poly_f64";
        let phi_deg = 15.0f64;
        let poly = vec![0.025f64, -0.36, 1.25, 2.0];
        let t: Vec<f64> = (0..512).map(|i| i as f64 / 128.0).collect();

        let kernel = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &poly,
            phi_deg,
        })?;
        let baseline_kernel = SweepPolyKernel::try_new(SweepPolyConfig {
            poly: &poly,
            phi_deg,
        })?;
        let candidate = kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("sweep_poly candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&t)
            .map_err(|e| anyhow!("sweep_poly baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "sweep_poly",
            json!({
                "t": t,
                "poly": poly,
                "phi_deg": phi_deg,
            }),
            180,
        )?;

        let candidate_ns = benchmark_avg_ns(140, || {
            kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("sweep_poly candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            baseline_kernel
                .run_alloc(&t)
                .map(|_| ())
                .map_err(|e| anyhow!("sweep_poly baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Unit impulse
    {
        let case_id = "unit_impulse_1d_f64";
        let len = 1024usize;
        let idx = 257usize;

        let kernel = UnitImpulseKernel::try_new(UnitImpulseConfig { len, idx })?;
        let baseline_kernel = UnitImpulseKernel::try_new(UnitImpulseConfig { len, idx })?;
        let candidate: Vec<f64> = kernel
            .run_alloc()
            .map_err(|e| anyhow!("unit_impulse candidate execution failed: {e}"))?;
        let baseline: Vec<f64> = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("unit_impulse baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "unit_impulse",
            json!({ "len": len, "idx": idx }),
            500,
        )?;

        let candidate_ns = benchmark_avg_ns(300, || {
            let _: Vec<f64> = kernel
                .run_alloc()
                .map_err(|e| anyhow!("unit_impulse candidate benchmark failed: {e}"))?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(300, || {
            let _: Vec<f64> = baseline_kernel
                .run_alloc()
                .map_err(|e| anyhow!("unit_impulse baseline benchmark failed: {e}"))?;
            Ok(())
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // lfilter
    {
        let case_id = "lfilter_f64";
        let b = vec![0.25f64, 0.5, 0.25];
        let a = vec![1.0f64];
        let x = signal.iter().copied().take(320).collect::<Vec<_>>();

        let kernel = LFilterKernel::try_new(LFilterConfig {
            b: b.clone(),
            a: a.clone(),
            axis: Some(0),
        })?;
        let baseline_kernel = LFilterKernel::try_new(LFilterConfig {
            b: b.clone(),
            a: a.clone(),
            axis: Some(0),
        })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("lfilter candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("lfilter baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "lfilter",
            json!({ "b": b, "a": a, "x": x }),
            180,
        )?;

        let candidate_ns = benchmark_avg_ns(120, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("lfilter candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(120, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("lfilter baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // filtfilt
    {
        let case_id = "filtfilt_f64";
        let b = vec![0.25f64, 0.5, 0.25];
        let a = vec![1.0f64];
        let x = signal.iter().copied().take(320).collect::<Vec<_>>();

        let padding = Some(FiltFiltPad::default());
        let kernel = FiltFiltKernel::try_new(FiltFiltConfig {
            b: b.clone(),
            a: a.clone(),
            axis: Some(0),
            padding,
        })?;
        let baseline_kernel = FiltFiltKernel::try_new(FiltFiltConfig {
            b: b.clone(),
            a: a.clone(),
            axis: Some(0),
            padding,
        })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("filtfilt candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("filtfilt baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "filtfilt",
            json!({
                "b": b,
                "a": a,
                "x": x,
                "pad": {"pad_type": "odd", "pad_len": 9}
            }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(50, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("filtfilt candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(50, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("filtfilt baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    let sos = reference_sos();
    let sos_flat = flatten_sos_coeffs(&sos);

    // sosfilt
    {
        let case_id = "sosfilt_f64";
        let x = signal.iter().copied().take(320).collect::<Vec<_>>();

        let mut kernel = SosFiltKernel::try_new(SosFiltConfig { sos: sos.clone() })?;
        let mut baseline_kernel = SosFiltKernel::try_new(SosFiltConfig { sos: sos.clone() })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("sosfilt candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("sosfilt baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "sosfilt",
            json!({ "sos": sos_flat, "x": x }),
            120,
        )?;

        let candidate_ns = benchmark_avg_ns(80, || {
            let mut bench_kernel = SosFiltKernel::try_new(SosFiltConfig { sos: sos.clone() })?;
            bench_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfilt candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(80, || {
            let mut bench_kernel = SosFiltKernel::try_new(SosFiltConfig { sos: sos.clone() })?;
            bench_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfilt baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // sosfiltfilt
    {
        let case_id = "sosfiltfilt_f64";
        let x = signal.iter().copied().take(320).collect::<Vec<_>>();

        let kernel = SosFiltFiltKernel::try_new(SosFiltFiltConfig { sos: sos.clone() })?;
        let baseline_kernel = SosFiltFiltKernel::try_new(SosFiltFiltConfig { sos: sos.clone() })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("sosfiltfilt candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("sosfiltfilt baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "sosfiltfilt",
            json!({ "sos": sos_flat, "x": x }),
            60,
        )?;

        let candidate_ns = benchmark_avg_ns(40, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfiltfilt candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(40, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfiltfilt baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // savgol filter
    {
        let case_id = "savgol_filter_f64";
        let x = signal.iter().copied().take(240).collect::<Vec<_>>();
        let kernel = SavgolFilterKernel::try_new(SavgolFilterConfig {
            window_length: 11,
            polyorder: 3,
            deriv: Some(0),
            delta: Some(1.0f64),
        })?;
        let baseline_kernel = SavgolFilterKernel::try_new(SavgolFilterConfig {
            window_length: 11,
            polyorder: 3,
            deriv: Some(0),
            delta: Some(1.0f64),
        })?;
        let candidate = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("savgol filter candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("savgol filter baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "savgol_filter",
            json!({
                "x": x,
                "window_length": 11,
                "polyorder": 3,
                "deriv": 0,
                "delta": 1.0
            }),
            150,
        )?;

        let candidate_ns = benchmark_avg_ns(100, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("savgol filter candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(100, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("savgol filter baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // savgol coeffs
    {
        let case_id = "savgol_coeffs_f64";
        let kernel = SavgolCoeffsKernel::try_new(SavgolCoeffsConfig {
            window_length: 11,
            polyorder: 3,
            deriv: Some(0),
            delta: Some(1.0f64),
        })?;
        let baseline_kernel = SavgolCoeffsKernel::try_new(SavgolCoeffsConfig {
            window_length: 11,
            polyorder: 3,
            deriv: Some(0),
            delta: Some(1.0f64),
        })?;
        let candidate = kernel
            .run_alloc()
            .map_err(|e| anyhow!("savgol coeffs candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("savgol coeffs baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "savgol_coeffs",
            json!({
                "window_length": 11,
                "polyorder": 3,
                "deriv": 0,
                "delta": 1.0
            }),
            400,
        )?;

        let candidate_ns = benchmark_avg_ns(250, || {
            kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("savgol coeffs candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(250, || {
            baseline_kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("savgol coeffs baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // lfilter_zi
    {
        let case_id = "lfilter_zi_f64";
        let b = vec![0.25f64, 0.5, 0.25];
        let a = vec![1.0f64];
        let kernel = LFilterZiKernel::try_new(LFilterZiConfig {
            b: b.clone(),
            a: a.clone(),
        })?;
        let baseline_kernel = LFilterZiKernel::try_new(LFilterZiConfig {
            b: b.clone(),
            a: a.clone(),
        })?;
        let candidate = kernel
            .run_alloc()
            .map_err(|e| anyhow!("lfilter_zi candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("lfilter_zi baseline execution failed: {e}"))?;
        let py = python_signal_eval(&python_bin, "lfilter_zi", json!({ "b": b, "a": a }), 300)?;

        let candidate_ns = benchmark_avg_ns(200, || {
            kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("lfilter_zi candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(200, || {
            baseline_kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("lfilter_zi baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // sosfilt_zi
    {
        let case_id = "sosfilt_zi_f64";
        let kernel = SosFiltZiKernel::try_new(SosFiltZiConfig { sos: sos.clone() })?;
        let baseline_kernel = SosFiltZiKernel::try_new(SosFiltZiConfig { sos: sos.clone() })?;
        let candidate_sections = kernel
            .run_alloc()
            .map_err(|e| anyhow!("sosfilt_zi candidate execution failed: {e}"))?;
        let candidate = flatten_sos_state(&candidate_sections);

        let baseline_sections = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("sosfilt_zi baseline execution failed: {e}"))?;
        let baseline = flatten_sos_state(&baseline_sections);

        let py = python_signal_eval(&python_bin, "sosfilt_zi", json!({ "sos": sos_flat }), 220)?;

        let candidate_ns = benchmark_avg_ns(150, || {
            let bench_kernel = SosFiltZiKernel::try_new(SosFiltZiConfig { sos: sos.clone() })?;
            let _ = bench_kernel
                .run_alloc()
                .map_err(|e| anyhow!("sosfilt_zi candidate benchmark failed: {e}"))?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(150, || {
            let bench_kernel = SosFiltZiKernel::try_new(SosFiltZiConfig { sos: sos.clone() })?;
            let _ = bench_kernel
                .run_alloc()
                .map_err(|e| anyhow!("sosfilt_zi baseline benchmark failed: {e}"))?;
            Ok(())
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // periodogram
    {
        let case_id = "periodogram_f64";
        let x = signal.clone();
        let fs = 100.0f64;

        let kernel = PeriodogramKernel::try_new(PeriodogramConfig { fs })?;
        let baseline_kernel = PeriodogramKernel::try_new(PeriodogramConfig { fs })?;
        let (cand_f, cand_p) = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("periodogram candidate execution failed: {e}"))?;
        let candidate = flatten_two_real(&cand_f, &cand_p);
        let (base_f, base_p) = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("periodogram baseline execution failed: {e}"))?;
        let baseline = flatten_two_real(&base_f, &base_p);
        let py = python_signal_eval(&python_bin, "periodogram", json!({ "x": x, "fs": fs }), 120)?;

        let candidate_ns = benchmark_avg_ns(80, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("periodogram candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(80, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("periodogram baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // welch
    {
        let case_id = "welch_f64";
        let x = signal.clone();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = nperseg / 2;

        let kernel = WelchKernel::try_new(WelchConfig { fs, nperseg })?;
        let baseline_kernel = WelchKernel::try_new(WelchConfig { fs, nperseg })?;
        let (cand_f, cand_p) = kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("welch candidate execution failed: {e}"))?;
        let candidate = flatten_two_real(&cand_f, &cand_p);
        let (base_f, base_p) = baseline_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("welch baseline execution failed: {e}"))?;
        let baseline = flatten_two_real(&base_f, &base_p);
        let py = python_signal_eval(
            &python_bin,
            "welch",
            json!({ "x": x, "fs": fs, "nperseg": nperseg, "noverlap": noverlap }),
            100,
        )?;

        let candidate_ns = benchmark_avg_ns(70, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("welch candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(70, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("welch baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // csd
    {
        let case_id = "csd_f64";
        let x = signal.clone();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.9 + 0.1 * (i as f64 * 0.07).sin())
            .collect::<Vec<_>>();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = nperseg / 2;

        let kernel = CsdKernel::try_new(CsdConfig { fs, nperseg })?;
        let baseline_kernel = CsdKernel::try_new(CsdConfig { fs, nperseg })?;
        let (cand_f, cand_p) = kernel
            .run_alloc(&x, &y)
            .map_err(|e| anyhow!("csd candidate execution failed: {e}"))?;
        let candidate = flatten_freqs_complex(&cand_f, &cand_p, |z| (z.re, z.im));
        let (base_f, base_p) = baseline_kernel
            .run_alloc(&x, &y)
            .map_err(|e| anyhow!("csd baseline execution failed: {e}"))?;
        let baseline = flatten_freqs_complex(&base_f, &base_p, |z| (z.re, z.im));
        let py = python_signal_eval(
            &python_bin,
            "csd",
            json!({
                "x": x,
                "y": y,
                "fs": fs,
                "nperseg": nperseg,
                "noverlap": noverlap
            }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&x, &y)
                .map(|_| ())
                .map_err(|e| anyhow!("csd candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&x, &y)
                .map(|_| ())
                .map_err(|e| anyhow!("csd baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // coherence
    {
        let case_id = "coherence_f64";
        let x = signal.clone();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.9 + 0.1 * (i as f64 * 0.07).sin())
            .collect::<Vec<_>>();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = nperseg / 2;

        let kernel = CoherenceKernel::try_new(CoherenceConfig { fs, nperseg })?;
        let baseline_kernel = CoherenceKernel::try_new(CoherenceConfig { fs, nperseg })?;
        let (cand_f, cand_c) = kernel
            .run_alloc(&x, &y)
            .map_err(|e| anyhow!("coherence candidate execution failed: {e}"))?;
        let candidate = flatten_two_real(&cand_f, &cand_c);
        let (base_f, base_c) = baseline_kernel
            .run_alloc(&x, &y)
            .map_err(|e| anyhow!("coherence baseline execution failed: {e}"))?;
        let baseline = flatten_two_real(&base_f, &base_c);
        let py = python_signal_eval(
            &python_bin,
            "coherence",
            json!({
                "x": x,
                "y": y,
                "fs": fs,
                "nperseg": nperseg,
                "noverlap": noverlap
            }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&x, &y)
                .map(|_| ())
                .map_err(|e| anyhow!("coherence candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&x, &y)
                .map(|_| ())
                .map_err(|e| anyhow!("coherence baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // stft
    {
        let case_id = "stft_f64";
        let x = signal.clone();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = 64usize;

        let kernel = StftKernel::try_new(StftConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let baseline_kernel = StftKernel::try_new(StftConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let candidate = flatten_stft_result(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("stft candidate execution failed: {e}"))?,
        );
        let baseline = flatten_stft_result(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("stft baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "stft",
            json!({ "x": x, "fs": fs, "nperseg": nperseg, "noverlap": noverlap }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("stft candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("stft baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // istft
    {
        let case_id = "istft_f64";
        let x = signal.clone();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = 64usize;
        let stft_kernel = StftKernel::try_new(StftConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let stft_output = stft_kernel
            .run_alloc(&x)
            .map_err(|e| anyhow!("istft source stft execution failed: {e}"))?;
        let n_freq = stft_output.zxx.len();
        let n_frames = stft_output.zxx.first().map_or(0, Vec::len);
        let zxx_re_im = flatten_complex_rows(&stft_output.zxx, |z| (z.re, z.im));

        let kernel = IstftKernel::try_new(IstftConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let baseline_kernel = IstftKernel::try_new(IstftConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let (cand_t, cand_y) = kernel
            .run_alloc(&stft_output.zxx)
            .map_err(|e| anyhow!("istft candidate execution failed: {e}"))?;
        let candidate = flatten_two_real(&cand_t, &cand_y);
        let (base_t, base_y) = baseline_kernel
            .run_alloc(&stft_output.zxx)
            .map_err(|e| anyhow!("istft baseline execution failed: {e}"))?;
        let baseline = flatten_two_real(&base_t, &base_y);
        let py = python_signal_eval(
            &python_bin,
            "istft",
            json!({
                "zxx_re_im": zxx_re_im,
                "n_freq": n_freq,
                "n_frames": n_frames,
                "fs": fs,
                "nperseg": nperseg,
                "noverlap": noverlap
            }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&stft_output.zxx)
                .map(|_| ())
                .map_err(|e| anyhow!("istft candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&stft_output.zxx)
                .map(|_| ())
                .map_err(|e| anyhow!("istft baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // spectrogram
    {
        let case_id = "spectrogram_f64";
        let x = signal.clone();
        let fs = 100.0f64;
        let nperseg = 128usize;
        let noverlap = 64usize;

        let kernel = SpectrogramKernel::try_new(SpectrogramConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let baseline_kernel = SpectrogramKernel::try_new(SpectrogramConfig {
            fs,
            nperseg,
            noverlap,
        })?;
        let candidate = flatten_spectrogram_result(
            &kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("spectrogram candidate execution failed: {e}"))?,
        );
        let baseline = flatten_spectrogram_result(
            &baseline_kernel
                .run_alloc(&x)
                .map_err(|e| anyhow!("spectrogram baseline execution failed: {e}"))?,
        );
        let py = python_signal_eval(
            &python_bin,
            "spectrogram",
            json!({ "x": x, "fs": fs, "nperseg": nperseg, "noverlap": noverlap }),
            80,
        )?;

        let candidate_ns = benchmark_avg_ns(60, || {
            kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("spectrogram candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(60, || {
            baseline_kernel
                .run_alloc(&x)
                .map(|_| ())
                .map_err(|e| anyhow!("spectrogram baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // freqz
    {
        let case_id = "freqz_f64";
        let b = vec![0.25f64, 0.5, 0.25];
        let a = vec![1.0f64];
        let wor_n = 64usize;

        let kernel = FreqzKernel::try_new(FreqzConfig { wor_n })?;
        let baseline_kernel = FreqzKernel::try_new(FreqzConfig { wor_n })?;
        let (cand_w, cand_h) = kernel
            .run_alloc(&b, &a)
            .map_err(|e| anyhow!("freqz candidate execution failed: {e}"))?;
        let candidate = flatten_freqs_complex(&cand_w, &cand_h, |z| (z.re, z.im));
        let (base_w, base_h) = baseline_kernel
            .run_alloc(&b, &a)
            .map_err(|e| anyhow!("freqz baseline execution failed: {e}"))?;
        let baseline = flatten_freqs_complex(&base_w, &base_h, |z| (z.re, z.im));
        let py = python_signal_eval(
            &python_bin,
            "freqz",
            json!({ "b": b, "a": a, "wor_n": wor_n }),
            140,
        )?;

        let candidate_ns = benchmark_avg_ns(110, || {
            kernel
                .run_alloc(&b, &a)
                .map(|_| ())
                .map_err(|e| anyhow!("freqz candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(110, || {
            baseline_kernel
                .run_alloc(&b, &a)
                .map(|_| ())
                .map_err(|e| anyhow!("freqz baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // sosfreqz
    {
        let case_id = "sosfreqz_f64";
        let wor_n = 64usize;

        let kernel = SosFreqzKernel::try_new(SosFreqzConfig { wor_n })?;
        let baseline_kernel = SosFreqzKernel::try_new(SosFreqzConfig { wor_n })?;
        let (cand_w, cand_h) = kernel
            .run_alloc(&sos)
            .map_err(|e| anyhow!("sosfreqz candidate execution failed: {e}"))?;
        let candidate = flatten_freqs_complex(&cand_w, &cand_h, |z| (z.re, z.im));
        let (base_w, base_h) = baseline_kernel
            .run_alloc(&sos)
            .map_err(|e| anyhow!("sosfreqz baseline execution failed: {e}"))?;
        let baseline = flatten_freqs_complex(&base_w, &base_h, |z| (z.re, z.im));
        let py = python_signal_eval(
            &python_bin,
            "sosfreqz",
            json!({ "sos": sos_flat, "wor_n": wor_n }),
            140,
        )?;

        let candidate_ns = benchmark_avg_ns(110, || {
            kernel
                .run_alloc(&sos)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfreqz candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(110, || {
            baseline_kernel
                .run_alloc(&sos)
                .map(|_| ())
                .map_err(|e| anyhow!("sosfreqz baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Window (hamming)
    {
        let case_id = "window_hamming_f64";
        let kernel = WindowKernel::try_new(WindowConfig {
            builder: WindowBuilderOwned::Hamming,
            nx: 128,
            fftbins: Some(false),
        })?;
        let baseline_kernel = WindowKernel::try_new(WindowConfig {
            builder: WindowBuilderOwned::Hamming,
            nx: 128,
            fftbins: Some(false),
        })?;
        let candidate = kernel
            .run_alloc()
            .map_err(|e| anyhow!("window candidate execution failed: {e}"))?;
        let baseline = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("window baseline execution failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "window",
            json!({ "window": "hamming", "nx": 128, "fftbins": false }),
            200,
        )?;

        let candidate_ns = benchmark_avg_ns(140, || {
            kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("window candidate benchmark failed: {e}"))
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            baseline_kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("window baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    let stats_input = signal.iter().copied().take(241).collect::<Vec<_>>();

    // Stats: mean
    {
        let case_id = "stats_mean_f64";
        let kernel = MeanKernel::try_new(StatsConfig)?;
        let candidate = vec![kernel.run(&stats_input)?.0];
        let baseline = vec![mean_baseline::<_, f64>(stats_input.iter()).0];
        let py = python_signal_eval(&python_bin, "mean", json!({ "x": stats_input }), 300)?;
        let candidate_ns = benchmark_avg_ns(250, || {
            let _ = kernel.run(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(250, || {
            let _ = mean_baseline::<_, f64>(stats_input.iter());
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: variance
    {
        let case_id = "stats_variance_f64";
        let kernel = VarianceKernel::try_new(StatsConfig)?;
        let candidate = vec![kernel.run(&stats_input)?.0];
        let baseline = vec![variance_baseline::<_, f64>(stats_input.iter()).0];
        let py = python_signal_eval(&python_bin, "variance", json!({ "x": stats_input }), 280)?;
        let candidate_ns = benchmark_avg_ns(220, || {
            let _ = kernel.run(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(220, || {
            let _ = variance_baseline::<_, f64>(stats_input.iter());
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: stdev
    {
        let case_id = "stats_stdev_f64";
        let kernel = StdevKernel::try_new(StatsConfig)?;
        let candidate = vec![kernel.run(&stats_input)?.0];
        let baseline = vec![stdev_baseline::<_, f64>(stats_input.iter()).0];
        let py = python_signal_eval(&python_bin, "stdev", json!({ "x": stats_input }), 280)?;
        let candidate_ns = benchmark_avg_ns(220, || {
            let _ = kernel.run(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(220, || {
            let _ = stdev_baseline::<_, f64>(stats_input.iter());
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: median
    {
        let case_id = "stats_median_f64";
        let kernel = MedianKernel::try_new(StatsConfig)?;
        let candidate = vec![kernel.run(&stats_input)?.0];
        let baseline = vec![median_baseline::<_, f64>(stats_input.iter()).0];
        let py = python_signal_eval(&python_bin, "median", json!({ "x": stats_input }), 220)?;
        let candidate_ns = benchmark_avg_ns(180, || {
            let _ = kernel.run(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            let _ = median_baseline::<_, f64>(stats_input.iter());
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: MAD
    {
        let case_id = "stats_mad_f64";
        let kernel = MadKernel::try_new(StatsConfig)?;
        let candidate = vec![kernel.run(&stats_input)?.0];
        let baseline = vec![mad_baseline::<_, f64>(stats_input.iter()).0];
        let py = python_signal_eval(&python_bin, "mad", json!({ "x": stats_input }), 220)?;
        let candidate_ns = benchmark_avg_ns(180, || {
            let _ = kernel.run(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            let _ = mad_baseline::<_, f64>(stats_input.iter());
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: zscore
    {
        let case_id = "stats_zscore_f64";
        let kernel = ZScoreKernel::try_new(StatsConfig)?;
        let candidate = kernel.run_alloc(&stats_input)?;
        let baseline = zscore_baseline::<_, f64>(stats_input.iter()).collect::<Vec<_>>();
        let py = python_signal_eval(&python_bin, "zscore", json!({ "x": stats_input }), 200)?;
        let candidate_ns = benchmark_avg_ns(140, || {
            let _ = kernel.run_alloc(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            let _ = zscore_baseline::<_, f64>(stats_input.iter()).collect::<Vec<_>>();
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Stats: mod_zscore
    {
        let case_id = "stats_mod_zscore_f64";
        let kernel = ModZScoreKernel::try_new(StatsConfig)?;
        let candidate = kernel.run_alloc(&stats_input)?;
        let baseline = mod_zscore_baseline::<_, f64>(stats_input.iter()).collect::<Vec<_>>();
        let py = python_signal_eval(&python_bin, "mod_zscore", json!({ "x": stats_input }), 200)?;
        let candidate_ns = benchmark_avg_ns(140, || {
            let _ = kernel.run_alloc(&stats_input)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            let _ = mod_zscore_baseline::<_, f64>(stats_input.iter()).collect::<Vec<_>>();
            Ok(())
        })?;
        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Companion matrix
    {
        let case_id = "companion_f64";
        let coeffs = vec![1.0f64, -10.0, 31.0, -30.0];
        let kernel = CompanionKernel::try_new(CompanionConfig {
            expected_len: Some(coeffs.len()),
        })?;
        let candidate_mat = kernel.run(&coeffs)?;
        let candidate = flatten_matrix_row_major(&candidate_mat);
        let baseline_mat = companion_dyn(coeffs.iter().copied(), coeffs.len());
        let baseline = flatten_matrix_row_major(&baseline_mat);
        let py = python_signal_eval(&python_bin, "companion", json!({ "coeffs": coeffs }), 250)?;

        let candidate_ns = benchmark_avg_ns(180, || {
            let _ = kernel.run(&coeffs)?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(180, || {
            let _ = companion_dyn(coeffs.iter().copied(), coeffs.len());
            Ok(())
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // FIR design
    {
        let case_id = "firwin_lowpass_f64";
        let kernel = FirWinKernel::try_new(FirWinConfig {
            numtaps: 31,
            cutoff: vec![0.2f64],
            width: None,
            window: Some(WindowBuilderOwned::Hamming),
            pass_zero: FilterBandType::Lowpass,
            scale: Some(true),
            fs: Some(2.0),
        })?;
        let baseline_kernel = FirWinKernel::try_new(FirWinConfig {
            numtaps: 31,
            cutoff: vec![0.2f64],
            width: None,
            window: Some(WindowBuilderOwned::Hamming),
            pass_zero: FilterBandType::Lowpass,
            scale: Some(true),
            fs: Some(2.0),
        })?;
        let candidate = kernel.run_alloc()?;
        let baseline = baseline_kernel
            .run_alloc()
            .map_err(|e| anyhow!("firwin baseline failed: {e}"))?;
        let py = python_signal_eval(
            &python_bin,
            "firwin",
            json!({
                "numtaps": 31,
                "cutoff": [0.2],
                "pass_zero": true,
                "scale": true,
                "fs": 2.0,
                "window": "hamming"
            }),
            180,
        )?;

        let candidate_ns = benchmark_avg_ns(140, || {
            let _ = kernel.run_alloc()?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(140, || {
            baseline_kernel
                .run_alloc()
                .map(|_| ())
                .map_err(|e| anyhow!("firwin baseline benchmark failed: {e}"))
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // Butter design (BA output flattened)
    {
        let case_id = "butter_ba_f64";
        let kernel = ButterKernel::try_new(ButterConfig {
            order: 4,
            wn: vec![10.0f64, 50.0],
            btype: Some(FilterBandType::Bandpass),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: Some(1666.0),
        })?;
        let baseline_kernel = ButterKernel::try_new(ButterConfig {
            order: 4,
            wn: vec![10.0f64, 50.0],
            btype: Some(FilterBandType::Bandpass),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: Some(1666.0),
        })?;
        let candidate = flatten_digital_filter_ba(
            kernel
                .run_alloc()
                .map_err(|e| anyhow!("butter candidate failed: {e}"))?,
        )?;
        let baseline = flatten_digital_filter_ba(
            baseline_kernel
                .run_alloc()
                .map_err(|e| anyhow!("butter baseline failed: {e}"))?,
        )?;
        let py = python_signal_eval(
            &python_bin,
            "butter_ba",
            json!({
                "order": 4,
                "wn": [10.0, 50.0],
                "btype": "bandpass",
                "analog": false,
                "fs": 1666.0
            }),
            140,
        )?;

        let candidate_ns = benchmark_avg_ns(100, || {
            let _ = kernel.run_alloc()?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(100, || {
            let _ = baseline_kernel
                .run_alloc()
                .map_err(|e| anyhow!("butter baseline benchmark failed: {e}"))?;
            Ok(())
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
    }

    // IIR design (Chebyshev II, BA output flattened)
    {
        let case_id = "iirfilter_cheby2_ba_f64";
        let kernel = IirFilterKernel::try_new(IirFilterConfig {
            order: 4,
            wn: vec![10.0f64, 50.0],
            rp: None,
            rs: Some(20.0),
            btype: Some(FilterBandType::Bandpass),
            ftype: Some(FilterType::ChebyshevII),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: Some(1666.0),
        })?;
        let baseline_kernel = IirFilterKernel::try_new(IirFilterConfig {
            order: 4,
            wn: vec![10.0f64, 50.0],
            rp: None,
            rs: Some(20.0),
            btype: Some(FilterBandType::Bandpass),
            ftype: Some(FilterType::ChebyshevII),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: Some(1666.0),
        })?;
        let candidate = flatten_digital_filter_ba(
            kernel
                .run_alloc()
                .map_err(|e| anyhow!("iirfilter candidate failed: {e}"))?,
        )?;
        let baseline = flatten_digital_filter_ba(
            baseline_kernel
                .run_alloc()
                .map_err(|e| anyhow!("iirfilter baseline failed: {e}"))?,
        )?;
        let py = python_signal_eval(
            &python_bin,
            "iirfilter_ba",
            json!({
                "order": 4,
                "wn": [10.0, 50.0],
                "rp": null,
                "rs": 20.0,
                "btype": "bandpass",
                "analog": false,
                "ftype": "cheby2",
                "fs": 1666.0
            }),
            120,
        )?;

        let candidate_ns = benchmark_avg_ns(90, || {
            let _ = kernel.run_alloc()?;
            Ok(())
        })?;
        let baseline_ns = benchmark_avg_ns(90, || {
            let _ = baseline_kernel
                .run_alloc()
                .map_err(|e| anyhow!("iirfilter baseline benchmark failed: {e}"))?;
            Ok(())
        })?;

        record_case(
            &mut rows,
            &mut case_plot_payload,
            &plots_dir,
            case_id,
            candidate,
            baseline,
            py,
            candidate_ns,
            baseline_ns,
        )?;
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
    println!("  - cases: {}", bundle.rows.len());

    Ok(())
}

fn detect_python_bin() -> PathBuf {
    PathBuf::from(DEFAULT_PYTHON_BIN)
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

fn python_signal_eval(
    python_bin: &Path,
    op: &str,
    payload: serde_json::Value,
    iters: usize,
) -> Result<PythonEval> {
    run_python_eval(
        python_bin,
        PY_SIGNAL_SCRIPT,
        json!({
            "op": op,
            "iters": iters,
            "payload": payload
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

#[allow(clippy::too_many_arguments)]
fn record_case(
    rows: &mut Vec<ContractRow>,
    case_plot_payload: &mut Vec<serde_json::Value>,
    plots_dir: &Path,
    case_id: &str,
    candidate: Vec<f64>,
    baseline: Vec<f64>,
    py: PythonEval,
    candidate_ns: f64,
    baseline_ns: f64,
) -> Result<()> {
    ensure_same_length(case_id, &candidate, &baseline)?;
    ensure_same_length(case_id, &candidate, &py.output)?;

    let overlay = plots_dir.join(format!("{case_id}_overlay.png"));
    let residual = plots_dir.join(format!("{case_id}_residual.png"));

    rows.push(build_row(RowBuildInput {
        case_id,
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

    Ok(())
}

fn usize_vec_to_f64(values: &[usize]) -> Vec<f64> {
    values.iter().map(|&v| v as f64).collect()
}

fn flatten_nested_f64(rows: &[Vec<f64>]) -> Vec<f64> {
    let total = rows.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn flatten_two_real(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

fn flatten_complex_rows<T>(rows: &[Vec<T>], mut to_pair: impl FnMut(&T) -> (f64, f64)) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.iter().map(Vec::len).sum::<usize>() * 2);
    for row in rows {
        for value in row {
            let (re, im) = to_pair(value);
            out.push(re);
            out.push(im);
        }
    }
    out
}

fn flatten_freqs_complex<T>(
    freqs: &[f64],
    spectrum: &[T],
    mut to_pair: impl FnMut(&T) -> (f64, f64),
) -> Vec<f64> {
    let mut out = Vec::with_capacity(freqs.len() + spectrum.len() * 2);
    out.extend_from_slice(freqs);
    for value in spectrum {
        let (re, im) = to_pair(value);
        out.push(re);
        out.push(im);
    }
    out
}

fn flatten_stft_result(result: &StftResult) -> Vec<f64> {
    let mut out = Vec::with_capacity(
        result.frequencies.len()
            + result.times.len()
            + result.zxx.iter().map(Vec::len).sum::<usize>() * 2,
    );
    out.extend_from_slice(&result.frequencies);
    out.extend_from_slice(&result.times);
    out.extend(flatten_complex_rows(&result.zxx, |z| (z.re, z.im)));
    out
}

fn flatten_spectrogram_result(result: &SpectrogramResult) -> Vec<f64> {
    let mut out = Vec::with_capacity(
        result.frequencies.len()
            + result.times.len()
            + result.sxx.iter().map(Vec::len).sum::<usize>(),
    );
    out.extend_from_slice(&result.frequencies);
    out.extend_from_slice(&result.times);
    out.extend(flatten_nested_f64(&result.sxx));
    out
}

fn flatten_sos_coeffs(sos: &[Sos<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(sos.len() * 6);
    for section in sos {
        out.extend_from_slice(&section.b);
        out.extend_from_slice(&section.a);
    }
    out
}

fn flatten_sos_state(sos: &[Sos<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(sos.len() * 2);
    for section in sos {
        out.push(section.zi0);
        out.push(section.zi1);
    }
    out
}

fn flatten_matrix_row_major(matrix: &sci_rs::na::DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            out.push(matrix[(row, col)]);
        }
    }
    out
}

fn flatten_digital_filter_ba(filter: DigitalFilter<f64>) -> Result<Vec<f64>> {
    match filter {
        DigitalFilter::Ba(ba) => {
            let mut out = Vec::with_capacity(ba.b.len() + ba.a.len());
            out.extend(ba.b);
            out.extend(ba.a);
            Ok(out)
        }
        _ => bail!("expected BA filter output"),
    }
}

fn reference_sos() -> Vec<Sos<f64>> {
    let coeffs: [f64; 24] = [
        2.677_576_738_259_783_5e-5,
        5.355_153_476_519_567e-5,
        2.677_576_738_259_783_5e-5,
        1.0,
        -1.799_120_215_461_773_4,
        0.816_257_861_481_900_5,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.877_476_989_441_982_5,
        0.909_430_241_306_808_6,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.923_795_989_286_610_3,
        0.926_379_467_161_616_1,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.978_497_311_228_862,
        0.979_989_488_697_337_8,
    ];
    Sos::from_scipy_dyn(4, coeffs.to_vec())
}

fn ensure_same_length(case_id: &str, a: &[f64], b: &[f64]) -> Result<()> {
    if a.len() != b.len() {
        bail!(
            "case {case_id} has mismatched output lengths: left={}, right={}",
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
