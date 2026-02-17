//! Spectral analysis helpers analogous to `scipy.signal` spectral APIs.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Read1D, Write1D};
use crate::signal::filter::design::Sos;
use crate::signal::traits::{
    Coherence1D, Csd1D, Freqz1D, Istft1D, Periodogram1D, SosFreqz1D, Spectrogram1D, Stft1D,
    WelchPsd1D,
};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// STFT result bundle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StftResult {
    /// One-sided frequency bins.
    pub frequencies: Vec<f64>,
    /// Segment times in seconds.
    pub times: Vec<f64>,
    /// Complex STFT matrix in frequency-major layout `[freq][time]`.
    pub zxx: Vec<Vec<Complex<f64>>>,
}

/// Spectrogram result bundle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpectrogramResult {
    /// One-sided frequency bins.
    pub frequencies: Vec<f64>,
    /// Segment times in seconds.
    pub times: Vec<f64>,
    /// Power spectrum in frequency-major layout `[freq][time]`.
    pub sxx: Vec<Vec<f64>>,
}

fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nm1 = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / nm1).cos())
        .collect()
}

fn rfft_real(x: &[f64], nfft: usize) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut buf = vec![Complex::new(0.0, 0.0); nfft];
    for (dst, src) in buf.iter_mut().zip(x.iter().copied()) {
        *dst = Complex::new(src, 0.0);
    }
    fft.process(&mut buf);
    buf
}

fn irfft_real(spec: &[Complex<f64>], nfft: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(nfft);
    let mut buf = vec![Complex::new(0.0, 0.0); nfft];

    let n_freq = nfft / 2 + 1;
    let copied = n_freq.min(spec.len());
    buf[..copied].copy_from_slice(&spec[..copied]);

    if nfft > 1 {
        let max_k = if nfft.is_multiple_of(2) {
            nfft / 2 - 1
        } else {
            nfft / 2
        };
        for k in 1..=max_k {
            buf[nfft - k] = buf[k].conj();
        }
    }

    ifft.process(&mut buf);
    let scale = 1.0 / nfft as f64;
    buf.into_iter().map(|c| c.re * scale).collect()
}

fn onesided_freqs(nfft: usize, fs: f64) -> Vec<f64> {
    let n_freq = nfft / 2 + 1;
    (0..n_freq).map(|k| k as f64 * fs / nfft as f64).collect()
}

fn onesided_psd(spec: &[Complex<f64>], fs: f64, norm: f64) -> Vec<f64> {
    let nfft = spec.len();
    let n_freq = nfft / 2 + 1;
    let mut pxx = vec![0.0; n_freq];
    for k in 0..n_freq {
        let mut v = spec[k].norm_sqr() / (fs * norm);
        if k != 0 && !(nfft.is_multiple_of(2) && k == nfft / 2) {
            v *= 2.0;
        }
        pxx[k] = v;
    }
    pxx
}

fn segment_starts(len: usize, nperseg: usize, noverlap: usize) -> Vec<usize> {
    if nperseg == 0 || noverlap >= nperseg {
        return Vec::new();
    }
    if len <= nperseg {
        return vec![0];
    }
    let hop = nperseg - noverlap;
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + nperseg <= len {
        starts.push(start);
        start += hop;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts
}

fn periodogram_impl(x: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let nfft = x.len();
    let spec = rfft_real(x, nfft);
    let freqs = onesided_freqs(nfft, fs);
    let pxx = onesided_psd(&spec, fs, nfft as f64);
    (freqs, pxx)
}

fn welch_impl(x: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() || nperseg == 0 {
        return (Vec::new(), Vec::new());
    }
    let nperseg = nperseg.min(x.len());
    let noverlap = nperseg / 2;
    let starts = segment_starts(x.len(), nperseg, noverlap);
    let window = hann_window(nperseg);
    let win_norm = window.iter().map(|v| v * v).sum::<f64>();
    let n_freq = nperseg / 2 + 1;
    let mut accum = vec![0.0; n_freq];

    for &start in &starts {
        let mut segment = vec![0.0; nperseg];
        let avail = (x.len() - start).min(nperseg);
        for i in 0..avail {
            segment[i] = x[start + i] * window[i];
        }
        let spec = rfft_real(&segment, nperseg);
        let psd = onesided_psd(&spec, fs, win_norm);
        for (a, p) in accum.iter_mut().zip(psd.iter()) {
            *a += *p;
        }
    }

    let inv = 1.0 / starts.len() as f64;
    for v in &mut accum {
        *v *= inv;
    }
    (onesided_freqs(nperseg, fs), accum)
}

fn csd_impl(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    if x.is_empty() || y.is_empty() || nperseg == 0 {
        return (Vec::new(), Vec::new());
    }
    let len = x.len().min(y.len());
    let nperseg = nperseg.min(len);
    let noverlap = nperseg / 2;
    let starts = segment_starts(len, nperseg, noverlap);
    let window = hann_window(nperseg);
    let win_norm = window.iter().map(|v| v * v).sum::<f64>();
    let n_freq = nperseg / 2 + 1;
    let mut accum = vec![Complex::new(0.0, 0.0); n_freq];

    for &start in &starts {
        let mut sx = vec![0.0; nperseg];
        let mut sy = vec![0.0; nperseg];
        let avail = (len - start).min(nperseg);
        for i in 0..avail {
            sx[i] = x[start + i] * window[i];
            sy[i] = y[start + i] * window[i];
        }
        let xfft = rfft_real(&sx, nperseg);
        let yfft = rfft_real(&sy, nperseg);
        for k in 0..n_freq {
            let mut v = xfft[k] * yfft[k].conj() / (fs * win_norm);
            if k != 0 && !(nperseg.is_multiple_of(2) && k == nperseg / 2) {
                v *= 2.0;
            }
            accum[k] += v;
        }
    }

    let inv = 1.0 / starts.len() as f64;
    for v in &mut accum {
        *v *= inv;
    }
    (onesided_freqs(nperseg, fs), accum)
}

fn coherence_impl(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let (fxy, pxy) = csd_impl(x, y, fs, nperseg);
    let (_, pxx) = welch_impl(x, fs, nperseg);
    let (_, pyy) = welch_impl(y, fs, nperseg);
    if fxy.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let eps = 1e-30;
    let coh = pxy
        .iter()
        .zip(pxx.iter().zip(pyy.iter()))
        .map(|(pxy, (pxx, pyy))| pxy.norm_sqr() / (pxx * pyy + eps))
        .collect();
    (fxy, coh)
}

fn stft_impl(x: &[f64], fs: f64, nperseg: usize, noverlap: usize) -> StftResult {
    if x.is_empty() || nperseg == 0 || noverlap >= nperseg {
        return StftResult::default();
    }

    let starts = segment_starts(x.len(), nperseg, noverlap);
    let window = hann_window(nperseg);
    let n_freq = nperseg / 2 + 1;
    let mut zxx = vec![vec![Complex::new(0.0, 0.0); starts.len()]; n_freq];
    let mut times = Vec::with_capacity(starts.len());

    for (frame, &start) in starts.iter().enumerate() {
        let mut segment = vec![0.0; nperseg];
        let avail = (x.len() - start).min(nperseg);
        for i in 0..avail {
            segment[i] = x[start + i] * window[i];
        }
        let spec = rfft_real(&segment, nperseg);
        for k in 0..n_freq {
            zxx[k][frame] = spec[k];
        }
        let center = start as f64 + (nperseg as f64 / 2.0);
        times.push(center / fs);
    }

    StftResult {
        frequencies: onesided_freqs(nperseg, fs),
        times,
        zxx,
    }
}

fn validate_istft_input(
    zxx: &[Vec<Complex<f64>>],
    nperseg: usize,
    noverlap: usize,
) -> Result<usize, ExecInvariantViolation> {
    if zxx.is_empty() {
        return Err(ExecInvariantViolation::InvalidState {
            reason: "istft input spectrum must be non-empty",
        });
    }
    if nperseg == 0 || noverlap >= nperseg {
        return Err(ExecInvariantViolation::InvalidState {
            reason: "invalid istft segment/overlap configuration",
        });
    }

    let n_freq = nperseg / 2 + 1;
    if zxx.len() != n_freq {
        return Err(ExecInvariantViolation::InvalidState {
            reason: "istft zxx frequency axis does not match nperseg",
        });
    }
    let n_frames = zxx[0].len();
    if n_frames == 0 {
        return Err(ExecInvariantViolation::InvalidState {
            reason: "istft zxx time axis must be non-empty",
        });
    }
    for row in zxx {
        if row.len() != n_frames {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "istft zxx rows must have equal frame counts",
            });
        }
    }

    Ok(n_frames)
}

fn istft_impl(
    zxx: &[Vec<Complex<f64>>],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_frames = match validate_istft_input(zxx, nperseg, noverlap) {
        Ok(n_frames) => n_frames,
        Err(_) => return (Vec::new(), Vec::new()),
    };

    let n_freq = nperseg / 2 + 1;
    let hop = nperseg - noverlap;
    let out_len = nperseg + hop * (n_frames - 1);
    let mut y = vec![0.0; out_len];
    let mut wsum = vec![0.0; out_len];
    let window = hann_window(nperseg);

    for (frame, _) in zxx[0].iter().enumerate().take(n_frames) {
        let mut onesided = vec![Complex::new(0.0, 0.0); n_freq];
        for k in 0..n_freq {
            onesided[k] = zxx[k][frame];
        }
        let frame_td = irfft_real(&onesided, nperseg);
        let offset = frame * hop;
        for i in 0..nperseg {
            let idx = offset + i;
            if idx < out_len {
                y[idx] += frame_td[i] * window[i];
                wsum[idx] += window[i] * window[i];
            }
        }
    }

    for (yi, wi) in y.iter_mut().zip(wsum.iter()) {
        if *wi > 1e-15 {
            *yi /= *wi;
        }
    }

    let t = (0..out_len).map(|i| i as f64 / fs).collect();
    (t, y)
}

fn spectrogram_impl(x: &[f64], fs: f64, nperseg: usize, noverlap: usize) -> SpectrogramResult {
    let st = stft_impl(x, fs, nperseg, noverlap);
    if st.zxx.is_empty() {
        return SpectrogramResult::default();
    }
    let sxx = st
        .zxx
        .iter()
        .map(|row| row.iter().map(|z| z.norm_sqr()).collect::<Vec<_>>())
        .collect();
    SpectrogramResult {
        frequencies: st.frequencies,
        times: st.times,
        sxx,
    }
}

fn freqz_impl(b: &[f64], a: &[f64], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    if b.is_empty() || a.is_empty() || wor_n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut w = Vec::with_capacity(wor_n);
    let mut h = Vec::with_capacity(wor_n);
    for i in 0..wor_n {
        let omega = core::f64::consts::PI * (i as f64) / (wor_n as f64);
        let z = Complex::from_polar(1.0, -omega);

        let mut num = Complex::new(0.0, 0.0);
        let mut zpow = Complex::new(1.0, 0.0);
        for &coeff in b {
            num += zpow * coeff;
            zpow *= z;
        }

        let mut den = Complex::new(0.0, 0.0);
        zpow = Complex::new(1.0, 0.0);
        for &coeff in a {
            den += zpow * coeff;
            zpow *= z;
        }

        w.push(omega);
        h.push(num / den);
    }
    (w, h)
}

fn sosfreqz_impl(sos: &[Sos<f64>], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    if sos.is_empty() || wor_n == 0 {
        return (Vec::new(), Vec::new());
    }
    let (w, mut h) = freqz_impl(&sos[0].b, &sos[0].a, wor_n);
    for sec in &sos[1..] {
        let (_, hs) = freqz_impl(&sec.b, &sec.a, wor_n);
        for (hi, hsi) in h.iter_mut().zip(hs.iter()) {
            *hi *= *hsi;
        }
    }
    (w, h)
}

/// Constructor config for [`PeriodogramKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodogramConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
}

/// Trait-first periodogram kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodogramKernel {
    fs: f64,
}

impl KernelLifecycle for PeriodogramKernel {
    type Config = PeriodogramConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        Ok(Self { fs: config.fs })
    }
}

impl Periodogram1D for PeriodogramKernel {
    fn run_into<I, OF, OP>(
        &self,
        input: &I,
        freqs: &mut OF,
        pxx: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "periodogram input must be non-empty",
            });
        }

        let expected = input.len() / 2 + 1;
        let freqs_out = freqs
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if freqs_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "freqs",
                expected,
                got: freqs_out.len(),
            });
        }
        let pxx_out = pxx
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if pxx_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "pxx",
                expected,
                got: pxx_out.len(),
            });
        }

        let (f, p) = periodogram_impl(input, self.fs);
        freqs_out.copy_from_slice(&f);
        pxx_out.copy_from_slice(&p);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "periodogram input must be non-empty",
            });
        }
        Ok(periodogram_impl(input, self.fs))
    }
}

/// Constructor config for [`WelchKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
}

/// Trait-first Welch PSD kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchKernel {
    fs: f64,
    nperseg: usize,
}

impl WelchKernel {
    fn expected_len(&self, input_len: usize) -> usize {
        self.nperseg.min(input_len) / 2 + 1
    }
}

impl KernelLifecycle for WelchKernel {
    type Config = WelchConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
        })
    }
}

impl WelchPsd1D for WelchKernel {
    fn run_into<I, OF, OP>(
        &self,
        input: &I,
        freqs: &mut OF,
        pxx: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "welch input must be non-empty",
            });
        }

        let expected = self.expected_len(input.len());
        let f_out = freqs
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if f_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "freqs",
                expected,
                got: f_out.len(),
            });
        }
        let p_out = pxx
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if p_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "pxx",
                expected,
                got: p_out.len(),
            });
        }

        let (f, p) = welch_impl(input, self.fs, self.nperseg);
        f_out.copy_from_slice(&f);
        p_out.copy_from_slice(&p);
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "welch input must be non-empty",
            });
        }
        Ok(welch_impl(input, self.fs, self.nperseg))
    }
}

/// Constructor config for [`CsdKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsdConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
}

/// Trait-first CSD kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsdKernel {
    fs: f64,
    nperseg: usize,
}

impl CsdKernel {
    fn expected_len(&self, len_x: usize, len_y: usize) -> usize {
        self.nperseg.min(len_x.min(len_y)) / 2 + 1
    }
}

impl KernelLifecycle for CsdKernel {
    type Config = CsdConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
        })
    }
}

impl Csd1D for CsdKernel {
    fn run_into<I1, I2, OF, OP>(
        &self,
        x: &I1,
        y: &I2,
        freqs: &mut OF,
        pxy: &mut OP,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OP: Write1D<Complex<f64>> + ?Sized,
    {
        let x = x.read_slice().map_err(ExecInvariantViolation::from)?;
        let y = y.read_slice().map_err(ExecInvariantViolation::from)?;
        if x.is_empty() || y.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "csd inputs must be non-empty",
            });
        }

        let expected = self.expected_len(x.len(), y.len());
        let f_out = freqs
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if f_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "freqs",
                expected,
                got: f_out.len(),
            });
        }
        let p_out = pxy
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if p_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "pxy",
                expected,
                got: p_out.len(),
            });
        }

        let (f, p) = csd_impl(x, y, self.fs, self.nperseg);
        f_out.copy_from_slice(&f);
        p_out.copy_from_slice(&p);
        Ok(())
    }

    fn run_alloc<I1, I2>(
        &self,
        x: &I1,
        y: &I2,
    ) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
    {
        let x = x.read_slice().map_err(ExecInvariantViolation::from)?;
        let y = y.read_slice().map_err(ExecInvariantViolation::from)?;
        if x.is_empty() || y.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "csd inputs must be non-empty",
            });
        }
        Ok(csd_impl(x, y, self.fs, self.nperseg))
    }
}

/// Constructor config for [`CoherenceKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoherenceConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
}

/// Trait-first coherence kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoherenceKernel {
    fs: f64,
    nperseg: usize,
}

impl CoherenceKernel {
    fn expected_len(&self, len_x: usize, len_y: usize) -> usize {
        self.nperseg.min(len_x.min(len_y)) / 2 + 1
    }
}

impl KernelLifecycle for CoherenceKernel {
    type Config = CoherenceConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
        })
    }
}

impl Coherence1D for CoherenceKernel {
    fn run_into<I1, I2, OF, OC>(
        &self,
        x: &I1,
        y: &I2,
        freqs: &mut OF,
        coherence: &mut OC,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OF: Write1D<f64> + ?Sized,
        OC: Write1D<f64> + ?Sized,
    {
        let x = x.read_slice().map_err(ExecInvariantViolation::from)?;
        let y = y.read_slice().map_err(ExecInvariantViolation::from)?;
        if x.is_empty() || y.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "coherence inputs must be non-empty",
            });
        }

        let expected = self.expected_len(x.len(), y.len());
        let f_out = freqs
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if f_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "freqs",
                expected,
                got: f_out.len(),
            });
        }
        let c_out = coherence
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if c_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "coherence",
                expected,
                got: c_out.len(),
            });
        }

        let (f, c) = coherence_impl(x, y, self.fs, self.nperseg);
        f_out.copy_from_slice(&f);
        c_out.copy_from_slice(&c);
        Ok(())
    }

    fn run_alloc<I1, I2>(
        &self,
        x: &I1,
        y: &I2,
    ) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
    {
        let x = x.read_slice().map_err(ExecInvariantViolation::from)?;
        let y = y.read_slice().map_err(ExecInvariantViolation::from)?;
        if x.is_empty() || y.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "coherence inputs must be non-empty",
            });
        }
        Ok(coherence_impl(x, y, self.fs, self.nperseg))
    }
}

/// Constructor config for [`StftKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StftConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
    /// Overlap length.
    pub noverlap: usize,
}

/// Trait-first STFT kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StftKernel {
    fs: f64,
    nperseg: usize,
    noverlap: usize,
}

impl KernelLifecycle for StftKernel {
    type Config = StftConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        if config.noverlap >= config.nperseg {
            return Err(ConfigError::InvalidArgument {
                arg: "noverlap",
                reason: "noverlap must be < nperseg",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
            noverlap: config.noverlap,
        })
    }
}

impl Stft1D for StftKernel {
    type Output = StftResult;

    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        *out = self.run_alloc(input)?;
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "stft input must be non-empty",
            });
        }
        Ok(stft_impl(input, self.fs, self.nperseg, self.noverlap))
    }
}

/// Constructor config for [`IstftKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IstftConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
    /// Overlap length.
    pub noverlap: usize,
}

/// Trait-first inverse STFT kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IstftKernel {
    fs: f64,
    nperseg: usize,
    noverlap: usize,
}

impl IstftKernel {
    fn expected_len(&self, n_frames: usize) -> usize {
        let hop = self.nperseg - self.noverlap;
        self.nperseg + hop * (n_frames - 1)
    }
}

impl KernelLifecycle for IstftKernel {
    type Config = IstftConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        if config.noverlap >= config.nperseg {
            return Err(ConfigError::InvalidArgument {
                arg: "noverlap",
                reason: "noverlap must be < nperseg",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
            noverlap: config.noverlap,
        })
    }
}

impl Istft1D for IstftKernel {
    fn run_into<O1, O2>(
        &self,
        zxx: &[Vec<Complex<f64>>],
        t: &mut O1,
        y: &mut O2,
    ) -> Result<(), ExecInvariantViolation>
    where
        O1: Write1D<f64> + ?Sized,
        O2: Write1D<f64> + ?Sized,
    {
        let n_frames = validate_istft_input(zxx, self.nperseg, self.noverlap)?;
        let expected = self.expected_len(n_frames);

        let t_out = t.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if t_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "t",
                expected,
                got: t_out.len(),
            });
        }

        let y_out = y.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if y_out.len() != expected {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "y",
                expected,
                got: y_out.len(),
            });
        }

        let (tt, yy) = istft_impl(zxx, self.fs, self.nperseg, self.noverlap);
        t_out.copy_from_slice(&tt);
        y_out.copy_from_slice(&yy);
        Ok(())
    }

    fn run_alloc(
        &self,
        zxx: &[Vec<Complex<f64>>],
    ) -> Result<(Vec<f64>, Vec<f64>), ExecInvariantViolation> {
        validate_istft_input(zxx, self.nperseg, self.noverlap)?;
        Ok(istft_impl(zxx, self.fs, self.nperseg, self.noverlap))
    }
}

/// Constructor config for [`SpectrogramKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrogramConfig {
    /// Sampling frequency in Hz.
    pub fs: f64,
    /// Segment length.
    pub nperseg: usize,
    /// Overlap length.
    pub noverlap: usize,
}

/// Trait-first spectrogram kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrogramKernel {
    fs: f64,
    nperseg: usize,
    noverlap: usize,
}

impl KernelLifecycle for SpectrogramKernel {
    type Config = SpectrogramConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if !config.fs.is_finite() || config.fs <= 0.0 {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be finite and > 0",
            });
        }
        if config.nperseg == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "nperseg",
                reason: "nperseg must be > 0",
            });
        }
        if config.noverlap >= config.nperseg {
            return Err(ConfigError::InvalidArgument {
                arg: "noverlap",
                reason: "noverlap must be < nperseg",
            });
        }
        Ok(Self {
            fs: config.fs,
            nperseg: config.nperseg,
            noverlap: config.noverlap,
        })
    }
}

impl Spectrogram1D for SpectrogramKernel {
    type Output = SpectrogramResult;

    fn run_into<I>(&self, input: &I, out: &mut Self::Output) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        *out = self.run_alloc(input)?;
        Ok(())
    }

    fn run_alloc<I>(&self, input: &I) -> Result<Self::Output, ExecInvariantViolation>
    where
        I: Read1D<f64> + ?Sized,
    {
        let input = input.read_slice().map_err(ExecInvariantViolation::from)?;
        if input.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "spectrogram input must be non-empty",
            });
        }
        Ok(spectrogram_impl(
            input,
            self.fs,
            self.nperseg,
            self.noverlap,
        ))
    }
}

/// Constructor config for [`FreqzKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreqzConfig {
    /// Number of frequency bins on `[0, pi]`.
    pub wor_n: usize,
}

/// Trait-first frequency response kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreqzKernel {
    wor_n: usize,
}

impl KernelLifecycle for FreqzKernel {
    type Config = FreqzConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.wor_n == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "wor_n",
                reason: "wor_n must be > 0",
            });
        }
        Ok(Self {
            wor_n: config.wor_n,
        })
    }
}

impl Freqz1D for FreqzKernel {
    fn run_into<I1, I2, OW, OH>(
        &self,
        b: &I1,
        a: &I2,
        w: &mut OW,
        h: &mut OH,
    ) -> Result<(), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
        OW: Write1D<f64> + ?Sized,
        OH: Write1D<Complex<f64>> + ?Sized,
    {
        let b = b.read_slice().map_err(ExecInvariantViolation::from)?;
        let a = a.read_slice().map_err(ExecInvariantViolation::from)?;
        if b.is_empty() || a.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "freqz numerator and denominator must be non-empty",
            });
        }

        let w_out = w.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if w_out.len() != self.wor_n {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "w",
                expected: self.wor_n,
                got: w_out.len(),
            });
        }
        let h_out = h.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if h_out.len() != self.wor_n {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "h",
                expected: self.wor_n,
                got: h_out.len(),
            });
        }

        let (ww, hh) = freqz_impl(b, a, self.wor_n);
        w_out.copy_from_slice(&ww);
        h_out.copy_from_slice(&hh);
        Ok(())
    }

    fn run_alloc<I1, I2>(
        &self,
        b: &I1,
        a: &I2,
    ) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I1: Read1D<f64> + ?Sized,
        I2: Read1D<f64> + ?Sized,
    {
        let b = b.read_slice().map_err(ExecInvariantViolation::from)?;
        let a = a.read_slice().map_err(ExecInvariantViolation::from)?;
        if b.is_empty() || a.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "freqz numerator and denominator must be non-empty",
            });
        }
        Ok(freqz_impl(b, a, self.wor_n))
    }
}

/// Constructor config for [`SosFreqzKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SosFreqzConfig {
    /// Number of frequency bins on `[0, pi]`.
    pub wor_n: usize,
}

/// Trait-first SOS frequency response kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SosFreqzKernel {
    wor_n: usize,
}

impl KernelLifecycle for SosFreqzKernel {
    type Config = SosFreqzConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.wor_n == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "wor_n",
                reason: "wor_n must be > 0",
            });
        }
        Ok(Self {
            wor_n: config.wor_n,
        })
    }
}

impl SosFreqz1D for SosFreqzKernel {
    fn run_into<I, OW, OH>(
        &self,
        sos: &I,
        w: &mut OW,
        h: &mut OH,
    ) -> Result<(), ExecInvariantViolation>
    where
        I: Read1D<Sos<f64>> + ?Sized,
        OW: Write1D<f64> + ?Sized,
        OH: Write1D<Complex<f64>> + ?Sized,
    {
        let sos = sos.read_slice().map_err(ExecInvariantViolation::from)?;
        if sos.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "sosfreqz requires at least one section",
            });
        }

        let w_out = w.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if w_out.len() != self.wor_n {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "w",
                expected: self.wor_n,
                got: w_out.len(),
            });
        }
        let h_out = h.write_slice_mut().map_err(ExecInvariantViolation::from)?;
        if h_out.len() != self.wor_n {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "h",
                expected: self.wor_n,
                got: h_out.len(),
            });
        }

        let (ww, hh) = sosfreqz_impl(sos, self.wor_n);
        w_out.copy_from_slice(&ww);
        h_out.copy_from_slice(&hh);
        Ok(())
    }

    fn run_alloc<I>(&self, sos: &I) -> Result<(Vec<f64>, Vec<Complex<f64>>), ExecInvariantViolation>
    where
        I: Read1D<Sos<f64>> + ?Sized,
    {
        let sos = sos.read_slice().map_err(ExecInvariantViolation::from)?;
        if sos.is_empty() {
            return Err(ExecInvariantViolation::InvalidState {
                reason: "sosfreqz requires at least one section",
            });
        }
        Ok(sosfreqz_impl(sos, self.wor_n))
    }
}

/// Estimate one-sided power spectral density using a periodogram.
pub fn periodogram(x: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let kernel = match PeriodogramKernel::try_new(PeriodogramConfig { fs }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(x).unwrap_or_default()
}

/// Estimate PSD with Welch's method using Hann windows and 50% overlap.
pub fn welch(x: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let kernel = match WelchKernel::try_new(WelchConfig { fs, nperseg }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(x).unwrap_or_default()
}

/// Estimate cross power spectral density with Welch averaging.
pub fn csd(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    let kernel = match CsdKernel::try_new(CsdConfig { fs, nperseg }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(x, y).unwrap_or_default()
}

/// Estimate magnitude-squared coherence from Welch/CSD estimates.
pub fn coherence(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let kernel = match CoherenceKernel::try_new(CoherenceConfig { fs, nperseg }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(x, y).unwrap_or_default()
}

/// Short-time Fourier transform (one-sided).
pub fn stft(x: &[f64], fs: f64, nperseg: usize, noverlap: usize) -> StftResult {
    let kernel = match StftKernel::try_new(StftConfig {
        fs,
        nperseg,
        noverlap,
    }) {
        Ok(kernel) => kernel,
        Err(_) => return StftResult::default(),
    };
    kernel.run_alloc(x).unwrap_or_default()
}

/// Inverse STFT using overlap-add with Hann synthesis.
pub fn istft(
    zxx: &[Vec<Complex<f64>>],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    let kernel = match IstftKernel::try_new(IstftConfig {
        fs,
        nperseg,
        noverlap,
    }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(zxx).unwrap_or_default()
}

/// Spectrogram from STFT magnitude squared.
pub fn spectrogram(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let kernel = match SpectrogramKernel::try_new(SpectrogramConfig {
        fs,
        nperseg,
        noverlap,
    }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new(), Vec::new()),
    };
    let result = kernel.run_alloc(x).unwrap_or_default();
    (result.frequencies, result.times, result.sxx)
}

/// Frequency response of digital filter coefficients.
pub fn freqz(b: &[f64], a: &[f64], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    let kernel = match FreqzKernel::try_new(FreqzConfig { wor_n }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(b, a).unwrap_or_default()
}

/// Frequency response of SOS cascades.
pub fn sosfreqz(sos: &[Sos<f64>], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    let kernel = match SosFreqzKernel::try_new(SosFreqzConfig { wor_n }) {
        Ok(kernel) => kernel,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    kernel.run_alloc(sos).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn test_signal() -> Vec<f64> {
        let fs = 100.0;
        (0..512)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * core::f64::consts::PI * 5.0 * t).sin()
                    + 0.2 * (2.0 * core::f64::consts::PI * 12.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn periodogram_returns_nonnegative_density() {
        let x = test_signal();
        let (f, pxx) = periodogram(&x, 100.0);
        assert!(!f.is_empty());
        assert_eq!(f.len(), pxx.len());
        assert!(pxx.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn periodogram_kernel_contracts_validate_config_and_output_shape() {
        assert!(PeriodogramKernel::try_new(PeriodogramConfig { fs: 0.0 }).is_err());

        let kernel =
            PeriodogramKernel::try_new(PeriodogramConfig { fs: 100.0 }).expect("valid config");
        let x = test_signal();
        let expected = x.len() / 2 + 1;
        let mut f = vec![0.0; expected - 1];
        let mut p = vec![0.0; expected];
        let err = kernel
            .run_into(&x, &mut f, &mut p)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn welch_detects_main_frequency_bin() {
        let x = test_signal();
        let (f, pxx) = welch(&x, 100.0, 128);
        let (idx, _) = pxx
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .expect("max bin");
        assert_abs_diff_eq!(f[idx], 5.0, epsilon = 1.0);
    }

    #[test]
    fn welch_kernel_contracts_validate_config() {
        assert!(WelchKernel::try_new(WelchConfig {
            fs: 100.0,
            nperseg: 0,
        })
        .is_err());
    }

    #[test]
    fn csd_and_coherence_shapes_match() {
        let x = test_signal();
        let (fxy, pxy) = csd(&x, &x, 100.0, 128);
        let (fc, coh) = coherence(&x, &x, 100.0, 128);
        assert_eq!(fxy.len(), pxy.len());
        assert_eq!(fc.len(), coh.len());
        assert_eq!(fxy.len(), fc.len());
        assert!(coh.iter().all(|c| *c >= 0.0));
    }

    #[test]
    fn csd_kernel_contracts_validate_output_shape() {
        let kernel = CsdKernel::try_new(CsdConfig {
            fs: 100.0,
            nperseg: 128,
        })
        .expect("valid config");
        let x = test_signal();
        let y = test_signal();
        let expected = 65;
        let mut f = vec![0.0; expected];
        let mut p = vec![Complex::new(0.0, 0.0); expected - 1];
        let err = kernel
            .run_into(&x, &y, &mut f, &mut p)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn coherence_kernel_contracts_validate_output_shape() {
        let kernel = CoherenceKernel::try_new(CoherenceConfig {
            fs: 100.0,
            nperseg: 128,
        })
        .expect("valid config");
        let x = test_signal();
        let y = test_signal();
        let expected = 65;
        let mut f = vec![0.0; expected - 1];
        let mut c = vec![0.0; expected];
        let err = kernel
            .run_into(&x, &y, &mut f, &mut c)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn stft_istft_round_trip_is_stable() {
        let x = test_signal();
        let st = stft(&x, 100.0, 128, 64);
        assert!(!st.zxx.is_empty());
        let (_t, y) = istft(&st.zxx, 100.0, 128, 64);
        assert!(!y.is_empty());
        let n = x.len().min(y.len());
        let mse = x
            .iter()
            .zip(y.iter())
            .take(n)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        assert!(mse < 1e-2);
    }

    #[test]
    fn stft_kernel_contracts_validate_config() {
        assert!(StftKernel::try_new(StftConfig {
            fs: 100.0,
            nperseg: 128,
            noverlap: 128,
        })
        .is_err());
    }

    #[test]
    fn istft_kernel_contracts_validate_output_shape() {
        let kernel = IstftKernel::try_new(IstftConfig {
            fs: 100.0,
            nperseg: 128,
            noverlap: 64,
        })
        .expect("valid config");
        let x = test_signal();
        let st = stft(&x, 100.0, 128, 64);
        let mut t = vec![0.0; 10];
        let mut y = vec![0.0; 10];
        let err = kernel
            .run_into(&st.zxx, &mut t, &mut y)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn spectrogram_matches_stft_shape() {
        let x = test_signal();
        let (f, t, sxx) = spectrogram(&x, 100.0, 128, 64);
        assert!(!sxx.is_empty());
        assert_eq!(sxx.len(), f.len());
        assert_eq!(sxx[0].len(), t.len());
    }

    #[test]
    fn spectrogram_kernel_contracts_validate_config() {
        assert!(SpectrogramKernel::try_new(SpectrogramConfig {
            fs: 100.0,
            nperseg: 128,
            noverlap: 128,
        })
        .is_err());
    }

    #[test]
    fn freqz_dc_gain_matches_simple_moving_average() {
        let b = [0.5f64, 0.5];
        let a = [1.0f64];
        let (_w, h) = freqz(&b, &a, 32);
        assert_abs_diff_eq!(h[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(h[0].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn freqz_kernel_contracts_validate_config_and_output_shape() {
        assert!(FreqzKernel::try_new(FreqzConfig { wor_n: 0 }).is_err());

        let kernel = FreqzKernel::try_new(FreqzConfig { wor_n: 32 }).expect("valid config");
        let b = [0.5f64, 0.5];
        let a = [1.0f64];
        let mut w = vec![0.0; 31];
        let mut h = vec![Complex::new(0.0, 0.0); 32];
        let err = kernel
            .run_into(&b, &a, &mut w, &mut h)
            .expect_err("mismatched output length should error");
        assert!(matches!(err, ExecInvariantViolation::LengthMismatch { .. }));
    }

    #[test]
    fn sosfreqz_matches_single_section_freqz() {
        let sec = Sos::new([0.5f64, 0.5, 0.0], [1.0, 0.0, 0.0]);
        let (_w1, h1) = freqz(&sec.b, &sec.a, 16);
        let (_w2, h2) = sosfreqz(&[sec], 16);
        for (a, b) in h1.iter().zip(h2.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn sosfreqz_kernel_contracts_validate_config_and_input() {
        assert!(SosFreqzKernel::try_new(SosFreqzConfig { wor_n: 0 }).is_err());

        let kernel = SosFreqzKernel::try_new(SosFreqzConfig { wor_n: 16 }).expect("valid config");
        let sos: Vec<Sos<f64>> = Vec::new();
        let err = kernel
            .run_alloc(&sos)
            .expect_err("empty sos should be rejected");
        assert!(matches!(err, ExecInvariantViolation::InvalidState { .. }));
    }
}
