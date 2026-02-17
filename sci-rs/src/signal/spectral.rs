//! Spectral analysis helpers analogous to `scipy.signal` spectral APIs.

use crate::signal::filter::design::Sos;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// STFT result bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct StftResult {
    /// One-sided frequency bins.
    pub frequencies: Vec<f64>,
    /// Segment times in seconds.
    pub times: Vec<f64>,
    /// Complex STFT matrix in frequency-major layout `[freq][time]`.
    pub zxx: Vec<Vec<Complex<f64>>>,
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

/// Estimate one-sided power spectral density using a periodogram.
pub fn periodogram(x: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let nfft = x.len();
    let spec = rfft_real(x, nfft);
    let freqs = onesided_freqs(nfft, fs);
    let pxx = onesided_psd(&spec, fs, nfft as f64);
    (freqs, pxx)
}

/// Estimate PSD with Welch's method using Hann windows and 50% overlap.
pub fn welch(x: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
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

/// Estimate cross power spectral density with Welch averaging.
pub fn csd(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
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

/// Estimate magnitude-squared coherence from Welch/CSD estimates.
pub fn coherence(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let (fxy, pxy) = csd(x, y, fs, nperseg);
    let (_, pxx) = welch(x, fs, nperseg);
    let (_, pyy) = welch(y, fs, nperseg);
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

/// Short-time Fourier transform (one-sided).
pub fn stft(x: &[f64], fs: f64, nperseg: usize, noverlap: usize) -> StftResult {
    if x.is_empty() || nperseg == 0 || noverlap >= nperseg {
        return StftResult {
            frequencies: Vec::new(),
            times: Vec::new(),
            zxx: Vec::new(),
        };
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

/// Inverse STFT using overlap-add with Hann synthesis.
pub fn istft(
    zxx: &[Vec<Complex<f64>>],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    if zxx.is_empty() || nperseg == 0 || noverlap >= nperseg {
        return (Vec::new(), Vec::new());
    }
    let n_freq = nperseg / 2 + 1;
    if zxx.len() != n_freq {
        return (Vec::new(), Vec::new());
    }
    let n_frames = zxx[0].len();
    if n_frames == 0 {
        return (Vec::new(), Vec::new());
    }
    for row in zxx {
        if row.len() != n_frames {
            return (Vec::new(), Vec::new());
        }
    }

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

/// Spectrogram from STFT magnitude squared.
pub fn spectrogram(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let st = stft(x, fs, nperseg, noverlap);
    if st.zxx.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let sxx = st
        .zxx
        .iter()
        .map(|row| row.iter().map(|z| z.norm_sqr()).collect::<Vec<_>>())
        .collect();
    (st.frequencies, st.times, sxx)
}

/// Frequency response of digital filter coefficients.
pub fn freqz(b: &[f64], a: &[f64], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
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

/// Frequency response of SOS cascades.
pub fn sosfreqz(sos: &[Sos<f64>], wor_n: usize) -> (Vec<f64>, Vec<Complex<f64>>) {
    if sos.is_empty() || wor_n == 0 {
        return (Vec::new(), Vec::new());
    }
    let (w, mut h) = freqz(&sos[0].b, &sos[0].a, wor_n);
    for sec in &sos[1..] {
        let (_, hs) = freqz(&sec.b, &sec.a, wor_n);
        for (hi, hsi) in h.iter_mut().zip(hs.iter()) {
            *hi *= *hsi;
        }
    }
    (w, h)
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
    fn spectrogram_matches_stft_shape() {
        let x = test_signal();
        let (f, t, sxx) = spectrogram(&x, 100.0, 128, 64);
        assert!(!sxx.is_empty());
        assert_eq!(sxx.len(), f.len());
        assert_eq!(sxx[0].len(), t.len());
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
    fn sosfreqz_matches_single_section_freqz() {
        let sec = Sos::new([0.5f64, 0.5, 0.0], [1.0, 0.0, 0.0]);
        let (_w1, h1) = freqz(&sec.b, &sec.a, 16);
        let (_w2, h2) = sosfreqz(&[sec], 16);
        for (a, b) in h1.iter().zip(h2.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
        }
    }
}
