use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::{SystemTime, UNIX_EPOCH};

/// Errors raised by plot utilities.
#[derive(Debug)]
pub enum PlotError {
    /// Underlying process or filesystem I/O failure.
    Io(std::io::Error),
    /// Python subprocess stdin was unavailable.
    StdinUnavailable,
    /// Python subprocess exited unsuccessfully.
    PythonExitFailure(ExitStatus),
}

impl core::fmt::Display for PlotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlotError::Io(err) => write!(f, "plot I/O failure: {err}"),
            PlotError::StdinUnavailable => {
                write!(f, "failed to open stdin for python plotting process")
            }
            PlotError::PythonExitFailure(status) => {
                write!(f, "python plotting script failed with status: {status}")
            }
        }
    }
}

impl std::error::Error for PlotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlotError::Io(err) => Some(err),
            PlotError::StdinUnavailable | PlotError::PythonExitFailure(_) => None,
        }
    }
}

impl From<std::io::Error> for PlotError {
    fn from(value: std::io::Error) -> Self {
        PlotError::Io(value)
    }
}

/// Debug utility function that will run a python script to plot the data.
///
/// This function generates a Python script to create plots of the input data and their autocorrelations.
/// It then executes the script using the system's Python interpreter with a non-interactive backend.
///
/// Note: This function is non-blocking and writes a PNG image into `target/contracts/plots`.
pub fn python_plot(xs: Vec<&[f32]>) {
    let _ = python_plot_to_path(xs, None::<&Path>);
}

/// Generate a non-interactive Python plot and save it to disk.
///
/// Returns the output path when plotting succeeds.
pub fn python_plot_to_path<P: AsRef<Path>>(
    xs: Vec<&[f32]>,
    output_path: Option<P>,
) -> Result<PathBuf, PlotError> {
    let output_path = match output_path {
        Some(path) => path.as_ref().to_path_buf(),
        None => {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            PathBuf::from(format!("target/contracts/plots/python_plot_{ts}.png"))
        }
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let output_path_literal = output_path.to_string_lossy().replace('\\', "\\\\");
    let script = format!(
        r#"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate

xs = {:?}
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(len(xs), 2)
for i, x in enumerate(xs):
    ax = plt.subplot(gs[i, 0])
    ax.plot(x, label = f"C{{i}}")
    ax.legend()
    ax.set_xlabel("Samples")
    ax = plt.subplot(gs[i, 1])
    autocorr = correlate(x, x, mode='full')
    normcorr = autocorr / autocorr.max()
    offsets = range(-len(x) + 1, len(x))
    ax.plot(offsets, normcorr, label = f"Autocorrelation of C{{i}}")
    ax.legend()
    ax.set_xlabel("Lag")
fig.tight_layout()
fig.savefig(r"{}", dpi=150)
plt.close(fig)
"#,
        xs, output_path_literal
    );
    // Run the script with python
    let script = script.as_bytes();
    let mut python = std::process::Command::new("python")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null()) // noisy
        .stderr(std::process::Stdio::null()) // noisy
        .spawn()?;

    if let Some(mut stdin) = python.stdin.take() {
        stdin.write_all(script)?;
    } else {
        return Err(PlotError::StdinUnavailable);
    }

    // Wait for completion and return a deterministic error if plotting fails.
    let status = python.wait()?;
    if !status.success() {
        return Err(PlotError::PythonExitFailure(status));
    }
    Ok(output_path)
}
