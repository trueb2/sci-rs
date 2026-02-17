# SciPy Checklist

SciPy-wide feature completion backlog for `sci-rs`, starting with `scipy.signal`.
This checklist is the canonical source of truth for assignment, verification, and merge readiness.

## Mission and Scope

1. Complete useful computational SciPy APIs in `sci-rs` with trait-first kernels.
2. Start with `scipy.signal` coverage, then continue through remaining SciPy domains.
3. Keep Python/SciPy as the local correctness and behavior reference oracle.
4. Defer only non-computational pythonisms by default (`BadCoefficients`, `test`).

Generated from local SciPy `1.14.1` by `scripts/generate_scipy_checklist_inventory.py`.

## Counters

- total callables tracked: `162`
- required-now count: `160`
- deferred-pythonism count: `2`
- merged count: `0`
- active count: `0`

## Status Legend

- `wishlist`: not yet claimed
- `assigned`: claimed and branch opened
- `impl`: implementation in progress
- `parity`: implementation done; parity evidence in progress
- `perf`: parity passed; performance evidence in progress
- `ready`: all checks passed; merge-ready
- `merged`: merged into active integration branch
- `blocked`: waiting on dependency/decision
- `deferred_pythonism`: intentionally deferred non-computational item

## Signal-First Priority Queue (Next 20)

| symbol | domain | status | scipy_source_anchor | notes |
| --- | --- | --- | --- | --- |
| find_peaks | peak | wishlist | _peak_finding.py:L729 |  |
| peak_prominences | peak | wishlist | _peak_finding.py:L323 |  |
| peak_widths | peak | wishlist | _peak_finding.py:L467 |  |
| argrelextrema | peak | wishlist | _peak_finding.py:L198 |  |
| argrelmax | peak | wishlist | _peak_finding.py:L141 |  |
| argrelmin | peak | wishlist | _peak_finding.py:L83 |  |
| find_peaks_cwt | peak | wishlist | _peak_finding.py:L1201 |  |
| cwt | wavelets | wishlist | _wavelets.py:L459 |  |
| upfirdn | multirate | wishlist | _upfirdn.py:L107 |  |
| resample_poly | multirate | wishlist | _signaltools.py:L3224 |  |
| decimate | multirate | wishlist | _signaltools.py:L4497 |  |
| periodogram | spectral | wishlist | _spectral_py.py:L156 |  |
| welch | spectral | wishlist | _spectral_py.py:L300 |  |
| csd | spectral | wishlist | _spectral_py.py:L470 |  |
| coherence | spectral | wishlist | _spectral_py.py:L1551 |  |
| stft | spectral | wishlist | _spectral_py.py:L1058 |  |
| istft | spectral | wishlist | _spectral_py.py:L1249 |  |
| spectrogram | spectral | wishlist | _spectral_py.py:L626 |  |
| freqz | filter_design | wishlist | _filter_design.py:L274 |  |
| sosfreqz | filter_design | wishlist | _filter_design.py:L739 |  |

## Full Callable Matrix

<!-- CHECKLIST_MATRIX_BEGIN -->
| symbol | scipy_namespace | kind | required_classification | domain | status | owner | branch | scipy_docs_url | scipy_source_anchor | rust_trait_config_kernel | rust_paths | contract_case_ids | benchmark_ids | parity_threshold_profile | perf_gate_result | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BadCoefficients | scipy.signal | class | deferred_pythonism | filter_design | deferred_pythonism |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.BadCoefficients.html | _filter_design.py:L33 |  |  |  |  |  |  |  |
| CZT | scipy.signal | class | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html | _czt.py:L115 |  |  |  |  |  |  |  |
| ShortTimeFFT | scipy.signal | class | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html | _short_time_fft.py:L78 |  |  |  |  |  |  |  |
| StateSpace | scipy.signal | class | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html | _ltisys.py:L1221 |  |  |  |  |  |  |  |
| TransferFunction | scipy.signal | class | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html | _ltisys.py:L490 |  |  |  |  |  |  |  |
| ZerosPolesGain | scipy.signal | class | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZerosPolesGain.html | _ltisys.py:L879 |  |  |  |  |  |  |  |
| ZoomFFT | scipy.signal | class | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZoomFFT.html | _czt.py:L275 |  |  |  |  |  |  |  |
| abcd_normalize | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.abcd_normalize.html | _lti_conversion.py:L149 |  |  |  |  |  |  |  |
| argrelextrema | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html | _peak_finding.py:L198 |  |  |  |  |  |  |  |
| argrelmax | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html | _peak_finding.py:L141 |  |  |  |  |  |  |  |
| argrelmin | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html | _peak_finding.py:L83 |  |  |  |  |  |  |  |
| band_stop_obj | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.band_stop_obj.html | _filter_design.py:L3760 |  |  |  |  |  |  |  |
| bessel | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html | _filter_design.py:L3589 |  |  |  |  |  |  |  |
| besselap | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.besselap.html | _filter_design.py:L4830 |  |  |  |  |  |  |  |
| bilinear | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html | _filter_design.py:L2159 |  |  |  |  |  |  |  |
| bilinear_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear_zpk.html | _filter_design.py:L2681 |  |  |  |  |  |  |  |
| bode | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bode.html | _ltisys.py:L2144 |  |  |  |  |  |  |  |
| buttap | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttap.html | _filter_design.py:L4322 |  |  |  |  |  |  |  |
| butter | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html | _filter_design.py:L3109 |  |  |  |  |  |  |  |
| buttord | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html | _filter_design.py:L3886 |  |  |  |  |  |  |  |
| cascade | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cascade.html | _wavelets.py:L119 |  |  |  |  |  |  |  |
| cheb1ap | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ap.html | _filter_design.py:L4342 |  |  |  |  |  |  |  |
| cheb1ord | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ord.html | _filter_design.py:L4011 |  |  |  |  |  |  |  |
| cheb2ap | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ap.html | _filter_design.py:L4380 |  |  |  |  |  |  |  |
| cheb2ord | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ord.html | _filter_design.py:L4104 |  |  |  |  |  |  |  |
| cheby1 | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html | _filter_design.py:L3234 |  |  |  |  |  |  |  |
| cheby2 | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html | _filter_design.py:L3352 |  |  |  |  |  |  |  |
| check_COLA | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html | _spectral_py.py:L809 |  |  |  |  |  |  |  |
| check_NOLA | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html | _spectral_py.py:L931 |  |  |  |  |  |  |  |
| chirp | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html | _waveforms.py:L264 |  |  |  |  |  |  |  |
| choose_conv_method | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.choose_conv_method.html | _signaltools.py:L1165 |  |  |  |  |  |  |  |
| cmplx_sort | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cmplx_sort.html | _signaltools.py:L2462 |  |  |  |  |  |  |  |
| coherence | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html | _spectral_py.py:L1551 |  |  |  |  |  |  |  |
| cont2discrete | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html | _lti_conversion.py:L335 |  |  |  |  |  |  |  |
| convolve | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html | _signaltools.py:L1304 |  |  |  |  |  |  |  |
| convolve2d | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html | _signaltools.py:L1654 |  |  |  |  |  |  |  |
| correlate | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html | _signaltools.py:L96 |  |  |  |  |  |  |  |
| correlate2d | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html | _signaltools.py:L1744 |  |  |  |  |  |  |  |
| correlation_lags | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html | _signaltools.py:L296 |  |  |  |  |  |  |  |
| csd | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html | _spectral_py.py:L470 |  |  |  |  |  |  |  |
| cspline1d | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d.html | _bsplines.py:L271 |  |  |  |  |  |  |  |
| cspline1d_eval | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d_eval.html | _bsplines.py:L372 |  |  |  |  |  |  |  |
| cspline2d | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline2d.html | builtin |  |  |  |  |  |  |  |
| cwt | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html | _wavelets.py:L459 |  |  |  |  |  |  |  |
| czt | scipy.signal | function | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt.html | _czt.py:L394 |  |  |  |  |  |  |  |
| czt_points | scipy.signal | function | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt_points.html | _czt.py:L53 |  |  |  |  |  |  |  |
| daub | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.daub.html | _wavelets.py:L16 |  |  |  |  |  |  |  |
| dbode | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dbode.html | _ltisys.py:L3423 |  |  |  |  |  |  |  |
| decimate | scipy.signal | function | required_now | multirate | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html | _signaltools.py:L4497 |  |  |  |  |  |  |  |
| deconvolve | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html | _signaltools.py:L2227 |  |  |  |  |  |  |  |
| detrend | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html | _signaltools.py:L3510 |  |  |  |  |  |  |  |
| dfreqresp | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dfreqresp.html | _ltisys.py:L3323 |  |  |  |  |  |  |  |
| dimpulse | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dimpulse.html | _ltisys.py:L3150 |  |  |  |  |  |  |  |
| dlsim | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlsim.html | _ltisys.py:L3034 |  |  |  |  |  |  |  |
| dlti | scipy.signal | class | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.html | _ltisys.py:L299 |  |  |  |  |  |  |  |
| dstep | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dstep.html | _ltisys.py:L3237 |  |  |  |  |  |  |  |
| ellip | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html | _filter_design.py:L3464 |  |  |  |  |  |  |  |
| ellipap | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipap.html | _filter_design.py:L4550 |  |  |  |  |  |  |  |
| ellipord | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html | _filter_design.py:L4229 |  |  |  |  |  |  |  |
| fftconvolve | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html | _signaltools.py:L562 |  |  |  |  |  |  |  |
| filtfilt | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html | _signaltools.py:L4028 |  |  |  |  |  |  |  |
| find_peaks | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html | _peak_finding.py:L729 |  |  |  |  |  |  |  |
| find_peaks_cwt | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html | _peak_finding.py:L1201 |  |  |  |  |  |  |  |
| findfreqs | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.findfreqs.html | _filter_design.py:L58 |  |  |  |  |  |  |  |
| firls | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html | _fir_filter_design.py:L837 |  |  |  |  |  |  |  |
| firwin | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html | _fir_filter_design.py:L251 |  |  |  |  |  |  |  |
| firwin2 | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html | _fir_filter_design.py:L469 |  |  |  |  |  |  |  |
| freqresp | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqresp.html | _ltisys.py:L2208 |  |  |  |  |  |  |  |
| freqs | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs.html | _filter_design.py:L117 |  |  |  |  |  |  |  |
| freqs_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs_zpk.html | _filter_design.py:L194 |  |  |  |  |  |  |  |
| freqz | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html | _filter_design.py:L274 |  |  |  |  |  |  |  |
| freqz_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_zpk.html | _filter_design.py:L491 |  |  |  |  |  |  |  |
| gammatone | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gammatone.html | _filter_design.py:L5394 |  |  |  |  |  |  |  |
| gauss_spline | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gauss_spline.html | _bsplines.py:L74 |  |  |  |  |  |  |  |
| gausspulse | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gausspulse.html | _waveforms.py:L163 |  |  |  |  |  |  |  |
| get_window | scipy.signal | function | required_now | helpers | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html | _windows.py:L2261 |  |  |  |  |  |  |  |
| group_delay | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html | _filter_design.py:L600 |  |  |  |  |  |  |  |
| hilbert | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html | _signaltools.py:L2287 |  |  |  |  |  |  |  |
| hilbert2 | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert2.html | _signaltools.py:L2397 |  |  |  |  |  |  |  |
| iircomb | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iircomb.html | _filter_design.py:L5174 |  |  |  |  |  |  |  |
| iirdesign | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html | _filter_design.py:L2254 |  |  |  |  |  |  |  |
| iirfilter | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html | _filter_design.py:L2428 |  |  |  |  |  |  |  |
| iirnotch | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html | _filter_design.py:L4938 |  |  |  |  |  |  |  |
| iirpeak | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html | _filter_design.py:L5019 |  |  |  |  |  |  |  |
| impulse | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html | _ltisys.py:L2007 |  |  |  |  |  |  |  |
| invres | scipy.signal | function | required_now | residue | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invres.html | _signaltools.py:L2584 |  |  |  |  |  |  |  |
| invresz | scipy.signal | function | required_now | residue | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invresz.html | _signaltools.py:L2962 |  |  |  |  |  |  |  |
| istft | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html | _spectral_py.py:L1249 |  |  |  |  |  |  |  |
| kaiser_atten | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_atten.html | _fir_filter_design.py:L74 |  |  |  |  |  |  |  |
| kaiser_beta | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiser_beta.html | _fir_filter_design.py:L36 |  |  |  |  |  |  |  |
| kaiserord | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.kaiserord.html | _fir_filter_design.py:L117 |  |  |  |  |  |  |  |
| lfilter | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html | _signaltools.py:L1954 |  |  |  |  |  |  |  |
| lfilter_zi | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html | _signaltools.py:L3637 |  |  |  |  |  |  |  |
| lfiltic | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfiltic.html | _signaltools.py:L2149 |  |  |  |  |  |  |  |
| lombscargle | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html | _spectral_py.py:L16 |  |  |  |  |  |  |  |
| lp2bp | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bp.html | _filter_design.py:L1981 |  |  |  |  |  |  |  |
| lp2bp_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bp_zpk.html | _filter_design.py:L2917 |  |  |  |  |  |  |  |
| lp2bs | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bs.html | _filter_design.py:L2070 |  |  |  |  |  |  |  |
| lp2bs_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2bs_zpk.html | _filter_design.py:L3013 |  |  |  |  |  |  |  |
| lp2hp | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2hp.html | _filter_design.py:L1898 |  |  |  |  |  |  |  |
| lp2hp_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2hp_zpk.html | _filter_design.py:L2837 |  |  |  |  |  |  |  |
| lp2lp | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2lp.html | _filter_design.py:L1826 |  |  |  |  |  |  |  |
| lp2lp_zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lp2lp_zpk.html | _filter_design.py:L2765 |  |  |  |  |  |  |  |
| lsim | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html | _ltisys.py:L1761 |  |  |  |  |  |  |  |
| lti | scipy.signal | class | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html | _ltisys.py:L132 |  |  |  |  |  |  |  |
| max_len_seq | scipy.signal | function | required_now | helpers | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.max_len_seq.html | _max_len_seq.py:L22 |  |  |  |  |  |  |  |
| medfilt | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html | _signaltools.py:L1511 |  |  |  |  |  |  |  |
| medfilt2d | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt2d.html | _signaltools.py:L1846 |  |  |  |  |  |  |  |
| minimum_phase | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html | _fir_filter_design.py:L1075 |  |  |  |  |  |  |  |
| morlet | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet.html | _wavelets.py:L232 |  |  |  |  |  |  |  |
| morlet2 | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet2.html | _wavelets.py:L373 |  |  |  |  |  |  |  |
| normalize | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.normalize.html | _filter_design.py:L1726 |  |  |  |  |  |  |  |
| oaconvolve | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html | _signaltools.py:L791 |  |  |  |  |  |  |  |
| order_filter | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.order_filter.html | _signaltools.py:L1442 |  |  |  |  |  |  |  |
| peak_prominences | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html | _peak_finding.py:L323 |  |  |  |  |  |  |  |
| peak_widths | scipy.signal | function | required_now | peak | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html | _peak_finding.py:L467 |  |  |  |  |  |  |  |
| periodogram | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html | _spectral_py.py:L156 |  |  |  |  |  |  |  |
| place_poles | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html | _ltisys.py:L2683 |  |  |  |  |  |  |  |
| qmf | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qmf.html | _wavelets.py:L92 |  |  |  |  |  |  |  |
| qspline1d | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d.html | _bsplines.py:L321 |  |  |  |  |  |  |  |
| qspline1d_eval | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d_eval.html | _bsplines.py:L446 |  |  |  |  |  |  |  |
| qspline2d | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline2d.html | builtin |  |  |  |  |  |  |  |
| remez | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html | _fir_filter_design.py:L666 |  |  |  |  |  |  |  |
| resample | scipy.signal | function | required_now | multirate | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html | _signaltools.py:L3036 |  |  |  |  |  |  |  |
| resample_poly | scipy.signal | function | required_now | multirate | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html | _signaltools.py:L3224 |  |  |  |  |  |  |  |
| residue | scipy.signal | function | required_now | residue | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residue.html | _signaltools.py:L2711 |  |  |  |  |  |  |  |
| residuez | scipy.signal | function | required_now | residue | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residuez.html | _signaltools.py:L2826 |  |  |  |  |  |  |  |
| ricker | scipy.signal | function | required_now | wavelets | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ricker.html | _wavelets.py:L316 |  |  |  |  |  |  |  |
| savgol_coeffs | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html | _savitzky_golay.py:L8 |  |  |  |  |  |  |  |
| savgol_filter | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html | _savitzky_golay.py:L230 |  |  |  |  |  |  |  |
| sawtooth | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html | _waveforms.py:L16 |  |  |  |  |  |  |  |
| sepfir2d | scipy.signal | function | required_now | convolution | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sepfir2d.html | builtin |  |  |  |  |  |  |  |
| sos2tf | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2tf.html | _filter_design.py:L1253 |  |  |  |  |  |  |  |
| sos2zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2zpk.html | _filter_design.py:L1301 |  |  |  |  |  |  |  |
| sosfilt | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html | _signaltools.py:L4272 |  |  |  |  |  |  |  |
| sosfilt_zi | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html | _signaltools.py:L3773 |  |  |  |  |  |  |  |
| sosfiltfilt | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html | _signaltools.py:L4384 |  |  |  |  |  |  |  |
| sosfreqz | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html | _filter_design.py:L739 |  |  |  |  |  |  |  |
| spectrogram | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html | _spectral_py.py:L626 |  |  |  |  |  |  |  |
| spline_filter | scipy.signal | function | required_now | splines | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spline_filter.html | _bsplines.py:L15 |  |  |  |  |  |  |  |
| square | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.square.html | _waveforms.py:L88 |  |  |  |  |  |  |  |
| ss2tf | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2tf.html | _lti_conversion.py:L196 |  |  |  |  |  |  |  |
| ss2zpk | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2zpk.html | _lti_conversion.py:L305 |  |  |  |  |  |  |  |
| step | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html | _ltisys.py:L2077 |  |  |  |  |  |  |  |
| stft | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html | _spectral_py.py:L1058 |  |  |  |  |  |  |  |
| sweep_poly | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sweep_poly.html | _waveforms.py:L475 |  |  |  |  |  |  |  |
| symiirorder1 | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder1.html | builtin |  |  |  |  |  |  |  |
| symiirorder2 | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder2.html | builtin |  |  |  |  |  |  |  |
| test | scipy.signal | function | deferred_pythonism | helpers | deferred_pythonism |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.test.html | builtin |  |  |  |  |  |  |  |
| tf2sos | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2sos.html | _filter_design.py:L1196 |  |  |  |  |  |  |  |
| tf2ss | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html | _lti_conversion.py:L18 |  |  |  |  |  |  |  |
| tf2zpk | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2zpk.html | _filter_design.py:L1037 |  |  |  |  |  |  |  |
| unique_roots | scipy.signal | function | required_now | residue | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unique_roots.html | _signaltools.py:L2497 |  |  |  |  |  |  |  |
| unit_impulse | scipy.signal | function | required_now | waveforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unit_impulse.html | _waveforms.py:L586 |  |  |  |  |  |  |  |
| upfirdn | scipy.signal | function | required_now | multirate | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html | _upfirdn.py:L107 |  |  |  |  |  |  |  |
| vectorstrength | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.vectorstrength.html | _signaltools.py:L3432 |  |  |  |  |  |  |  |
| welch | scipy.signal | function | required_now | spectral | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html | _spectral_py.py:L300 |  |  |  |  |  |  |  |
| wiener | scipy.signal | function | required_now | filtering | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html | _signaltools.py:L1579 |  |  |  |  |  |  |  |
| zoom_fft | scipy.signal | function | required_now | transforms | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zoom_fft.html | _czt.py:L508 |  |  |  |  |  |  |  |
| zpk2sos | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html | _filter_design.py:L1363 |  |  |  |  |  |  |  |
| zpk2ss | scipy.signal | function | required_now | lti | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2ss.html | _lti_conversion.py:L285 |  |  |  |  |  |  |  |
| zpk2tf | scipy.signal | function | required_now | filter_design | wishlist |  |  | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2tf.html | _filter_design.py:L1122 |  |  |  |  |  |  |  |
<!-- CHECKLIST_MATRIX_END -->
