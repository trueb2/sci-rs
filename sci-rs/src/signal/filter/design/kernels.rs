//! Trait-first kernels for filter design APIs.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Write1D};
use crate::signal::traits::{
    ComplexPairSplit, FirWinDesign, IirDesign, RelativeDegreeDesign, ZpkToSosDesign, ZpkToTfDesign,
    ZpkTransform,
};
use crate::signal::windows::{GetWindow, WindowBuilderOwned};
use crate::special::Bessel;
use core::iter::Sum;
use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{real::Real, Float, MulAdd, Pow};

use alloc::vec::Vec;

use super::{
    bilinear_zpk_dyn, butter_checked, cplxreal_checked, firwin_dyn, iirfilter_checked,
    lp2bp_zpk_dyn, lp2bs_zpk_dyn, lp2hp_zpk_dyn, lp2lp_zpk_dyn, relative_degree_checked,
    zpk2sos_dyn, zpk2tf_dyn, BaFormatFilter, DigitalFilter, FilterBandType, FilterOutputType,
    FilterType, SosFormatFilter, ZpkFormatFilter, ZpkPairing,
};

/// Constructor config for [`FirWinKernel`].
#[derive(Debug, Clone, PartialEq)]
pub struct FirWinConfig<F>
where
    F: Real,
{
    /// Number of filter taps.
    pub numtaps: usize,
    /// Cutoff frequencies.
    pub cutoff: Vec<F>,
    /// Optional transition width.
    pub width: Option<F>,
    /// Optional window builder (generated with `sym=true` for firwin semantics).
    pub window: Option<WindowBuilderOwned<F>>,
    /// Filter pass-band type.
    pub pass_zero: FilterBandType,
    /// Optional normalization flag.
    pub scale: Option<bool>,
    /// Optional sample rate.
    pub fs: Option<F>,
}

/// Trait-first FIR design kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct FirWinKernel<F>
where
    F: Real,
{
    numtaps: usize,
    cutoff: Vec<F>,
    width: Option<F>,
    window: Option<WindowBuilderOwned<F>>,
    pass_zero: FilterBandType,
    scale: Option<bool>,
    fs: Option<F>,
}

impl<F> KernelLifecycle for FirWinKernel<F>
where
    F: Real + Copy + PartialOrd,
{
    type Config = FirWinConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.numtaps == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "numtaps",
                reason: "numtaps must be greater than zero",
            });
        }
        if config.cutoff.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "cutoff" });
        }
        if config.window.is_some() && config.width.is_some() {
            return Err(ConfigError::InvalidArgument {
                arg: "window/width",
                reason: "window and width cannot both be set",
            });
        }
        if config.cutoff.iter().any(|x| *x <= F::zero()) {
            return Err(ConfigError::InvalidArgument {
                arg: "cutoff",
                reason: "cutoff frequencies must be positive",
            });
        }
        if config.cutoff.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ConfigError::InvalidArgument {
                arg: "cutoff",
                reason: "cutoff frequencies must be strictly increasing",
            });
        }

        Ok(Self {
            numtaps: config.numtaps,
            cutoff: config.cutoff,
            width: config.width,
            window: config.window,
            pass_zero: config.pass_zero,
            scale: config.scale,
            fs: config.fs,
        })
    }
}

impl<F> FirWinDesign<F> for FirWinKernel<F>
where
    F: Real
        + PartialOrd
        + Float
        + RealField
        + MulAdd<Output = F>
        + Pow<F, Output = F>
        + Bessel
        + Copy,
{
    fn run_into<O>(&self, out: &mut O) -> Result<(), ExecInvariantViolation>
    where
        O: Write1D<F> + ?Sized,
    {
        let coeffs = self.run_alloc()?;
        let out_slice = out
            .write_slice_mut()
            .map_err(ExecInvariantViolation::from)?;
        if out_slice.len() != coeffs.len() {
            return Err(ExecInvariantViolation::LengthMismatch {
                arg: "out",
                expected: coeffs.len(),
                got: out_slice.len(),
            });
        }
        out_slice.copy_from_slice(&coeffs);
        Ok(())
    }

    fn run_alloc(&self) -> Result<Vec<F>, ExecInvariantViolation> {
        let window = self
            .window
            .as_ref()
            .map(|builder| builder.build_window(self.numtaps, Some(false)));
        firwin_dyn(
            self.numtaps,
            self.cutoff.as_slice(),
            self.width,
            window.as_ref(),
            &self.pass_zero,
            self.scale,
            self.fs,
        )
        .map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "firwin kernel execution failed",
        })
    }
}

/// Constructor config for [`IirFilterKernel`].
#[derive(Debug, Clone, PartialEq)]
pub struct IirFilterConfig<F>
where
    F: RealField + Copy,
{
    /// Filter order.
    pub order: usize,
    /// Critical frequencies.
    pub wn: Vec<F>,
    /// Optional pass-band ripple parameter.
    pub rp: Option<F>,
    /// Optional stop-band attenuation parameter.
    pub rs: Option<F>,
    /// Optional band type.
    pub btype: Option<FilterBandType>,
    /// Optional IIR prototype.
    pub ftype: Option<FilterType>,
    /// Optional analog mode.
    pub analog: Option<bool>,
    /// Optional output format.
    pub output: Option<FilterOutputType>,
    /// Optional sample rate.
    pub fs: Option<F>,
}

/// Trait-first IIR design kernel (general `iirfilter`).
#[derive(Debug, Clone, PartialEq)]
pub struct IirFilterKernel<F>
where
    F: RealField + Copy,
{
    order: usize,
    wn: Vec<F>,
    rp: Option<F>,
    rs: Option<F>,
    btype: Option<FilterBandType>,
    ftype: Option<FilterType>,
    analog: Option<bool>,
    output: Option<FilterOutputType>,
    fs: Option<F>,
}

impl<F> KernelLifecycle for IirFilterKernel<F>
where
    F: RealField + Float + Sum + Copy,
{
    type Config = IirFilterConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.order == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be greater than zero",
            });
        }
        if config.wn.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "wn" });
        }
        if config.wn.len() > 2 {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "wn length must be 1 or 2",
            });
        }
        if config.wn.iter().any(|w| *w <= F::zero()) {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "critical frequencies must be greater than 0",
            });
        }
        if config.wn.len() == 2 && config.wn[0] >= config.wn[1] {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "wn[0] must be less than wn[1]",
            });
        }

        let analog = config.analog.unwrap_or(false);
        if analog && config.fs.is_some() {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs cannot be provided for analog filters",
            });
        }
        if !analog {
            if let Some(fs) = config.fs {
                let nyq = fs / F::from(2.0).unwrap();
                if config.wn.iter().any(|w| *w >= nyq) {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "digital wn must satisfy 0 < wn < fs/2",
                    });
                }
            } else if config.wn.iter().any(|w| *w >= F::one()) {
                return Err(ConfigError::InvalidArgument {
                    arg: "wn",
                    reason: "normalized digital wn must satisfy 0 < wn < 1",
                });
            }
        }

        if let Some(rp) = config.rp {
            if rp <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "rp",
                    reason: "rp must be positive",
                });
            }
        }
        if let Some(rs) = config.rs {
            if rs <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "rs",
                    reason: "rs must be positive",
                });
            }
        }

        let band = config.btype.unwrap_or(FilterBandType::Bandpass);
        match band {
            FilterBandType::Lowpass | FilterBandType::Highpass => {
                if config.wn.len() != 1 {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "lowpass/highpass designs require one critical frequency",
                    });
                }
            }
            FilterBandType::Bandpass | FilterBandType::Bandstop => {
                if config.wn.len() != 2 {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "bandpass/bandstop designs require two critical frequencies",
                    });
                }
            }
        }

        let ftype = config.ftype.unwrap_or(FilterType::Butterworth);
        match ftype {
            FilterType::ChebyshevI => {
                if config.rp.is_none() {
                    return Err(ConfigError::InvalidArgument {
                        arg: "rp",
                        reason: "rp is required for Chebyshev I designs",
                    });
                }
            }
            FilterType::ChebyshevII => {
                if config.rs.is_none() {
                    return Err(ConfigError::InvalidArgument {
                        arg: "rs",
                        reason: "rs is required for Chebyshev II designs",
                    });
                }
            }
            FilterType::CauerElliptic | FilterType::BesselThomson(_) => {
                return Err(ConfigError::InvalidArgument {
                    arg: "ftype",
                    reason: "selected filter type is not implemented",
                });
            }
            FilterType::Butterworth => {}
        }

        Ok(Self {
            order: config.order,
            wn: config.wn,
            rp: config.rp,
            rs: config.rs,
            btype: config.btype,
            ftype: config.ftype,
            analog: config.analog,
            output: config.output,
            fs: config.fs,
        })
    }
}

impl<F> IirDesign<F> for IirFilterKernel<F>
where
    F: RealField + Float + Sum + Copy,
{
    type Output = DigitalFilter<F>;

    fn run_alloc(&self) -> Result<Self::Output, ExecInvariantViolation> {
        // SciPy-equivalent Chebyshev II does not consume `rp`; keep a benign value
        // for compatibility with legacy argument shape while using checked execution.
        let rp = if matches!(self.ftype, Some(FilterType::ChebyshevII)) {
            self.rp.or(Some(F::one()))
        } else {
            self.rp
        };

        iirfilter_checked(
            self.order,
            self.wn.clone(),
            rp,
            self.rs,
            self.btype,
            self.ftype,
            self.analog,
            self.output,
            self.fs,
        )
        .map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "iirfilter kernel execution failed",
        })
    }
}

/// Constructor config for [`ButterKernel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ButterConfig<F>
where
    F: RealField + Copy,
{
    /// Filter order.
    pub order: usize,
    /// Critical frequencies.
    pub wn: Vec<F>,
    /// Optional band type.
    pub btype: Option<FilterBandType>,
    /// Optional analog mode.
    pub analog: Option<bool>,
    /// Optional output format.
    pub output: Option<FilterOutputType>,
    /// Optional sample rate.
    pub fs: Option<F>,
}

/// Trait-first Butterworth design kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ButterKernel<F>
where
    F: RealField + Copy,
{
    order: usize,
    wn: Vec<F>,
    btype: Option<FilterBandType>,
    analog: Option<bool>,
    output: Option<FilterOutputType>,
    fs: Option<F>,
}

impl<F> KernelLifecycle for ButterKernel<F>
where
    F: RealField + Float + Sum + Copy,
{
    type Config = ButterConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.order == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be greater than zero",
            });
        }
        if config.wn.is_empty() {
            return Err(ConfigError::EmptyInput { arg: "wn" });
        }
        if config.wn.len() > 2 {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "wn length must be 1 or 2",
            });
        }
        if config.wn.iter().any(|w| *w <= F::zero()) {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "critical frequencies must be greater than 0",
            });
        }
        if config.wn.len() == 2 && config.wn[0] >= config.wn[1] {
            return Err(ConfigError::InvalidArgument {
                arg: "wn",
                reason: "wn[0] must be less than wn[1]",
            });
        }

        let analog = config.analog.unwrap_or(false);
        if analog && config.fs.is_some() {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs cannot be provided for analog filters",
            });
        }
        if !analog {
            if let Some(fs) = config.fs {
                let nyq = fs / F::from(2.0).unwrap();
                if config.wn.iter().any(|w| *w >= nyq) {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "digital wn must satisfy 0 < wn < fs/2",
                    });
                }
            } else if config.wn.iter().any(|w| *w >= F::one()) {
                return Err(ConfigError::InvalidArgument {
                    arg: "wn",
                    reason: "normalized digital wn must satisfy 0 < wn < 1",
                });
            }
        }

        let band = config.btype.unwrap_or(FilterBandType::Lowpass);
        match band {
            FilterBandType::Lowpass | FilterBandType::Highpass => {
                if config.wn.len() != 1 {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "lowpass/highpass designs require one critical frequency",
                    });
                }
            }
            FilterBandType::Bandpass | FilterBandType::Bandstop => {
                if config.wn.len() != 2 {
                    return Err(ConfigError::InvalidArgument {
                        arg: "wn",
                        reason: "bandpass/bandstop designs require two critical frequencies",
                    });
                }
            }
        }

        Ok(Self {
            order: config.order,
            wn: config.wn,
            btype: config.btype,
            analog: config.analog,
            output: config.output,
            fs: config.fs,
        })
    }
}

impl<F> IirDesign<F> for ButterKernel<F>
where
    F: RealField + Float + Sum + Copy,
{
    type Output = DigitalFilter<F>;

    fn run_alloc(&self) -> Result<Self::Output, ExecInvariantViolation> {
        butter_checked(
            self.order,
            self.wn.clone(),
            self.btype,
            self.analog,
            self.output,
            self.fs,
        )
        .map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "butter kernel execution failed",
        })
    }
}

fn validate_relative_degree<F>(
    zeros: &[Complex<F>],
    poles: &[Complex<F>],
) -> Result<(), ExecInvariantViolation>
where
    F: RealField + Float + Copy,
{
    if poles.len() < zeros.len() {
        return Err(ConfigError::InvalidArgument {
            arg: "zpk",
            reason: "improper transfer function; poles must be >= zeros",
        }
        .into());
    }
    Ok(())
}

fn validate_conjugate_pairs<F>(
    roots: &[Complex<F>],
    tol: F,
    arg: &'static str,
) -> Result<(), ExecInvariantViolation>
where
    F: RealField + Float + Copy,
{
    for (i, zi) in roots.iter().enumerate() {
        if Float::abs(zi.im) <= tol * zi.abs() {
            continue;
        }
        let has_conjugate = roots
            .iter()
            .enumerate()
            .any(|(j, zj)| i != j && (*zj - zi.conj()).abs() <= tol * zi.abs());
        if !has_conjugate {
            return Err(ConfigError::InvalidArgument {
                arg,
                reason: "complex roots must have matching conjugate pairs",
            }
            .into());
        }
    }
    Ok(())
}

/// Constructor config for [`RelativeDegreeKernel`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelativeDegreeConfig;

/// Trait-first relative-degree helper kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelativeDegreeKernel;

impl KernelLifecycle for RelativeDegreeKernel {
    type Config = RelativeDegreeConfig;

    fn try_new(_: Self::Config) -> Result<Self, ConfigError> {
        Ok(Self)
    }
}

impl<F> RelativeDegreeDesign<F> for RelativeDegreeKernel
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zeros: &[Complex<F>],
        poles: &[Complex<F>],
    ) -> Result<usize, ExecInvariantViolation> {
        validate_relative_degree(zeros, poles)?;
        relative_degree_checked(zeros, poles).map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "relative-degree computation failed",
        })
    }
}

/// Constructor config for [`CplxRealKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CplxRealConfig<F>
where
    F: RealField + Copy,
{
    /// Relative tolerance used for matching roots.
    pub tol: Option<F>,
}

/// Trait-first complex root splitting kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CplxRealKernel<F>
where
    F: RealField + Copy,
{
    tol: Option<F>,
}

impl<F> KernelLifecycle for CplxRealKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = CplxRealConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(tol) = config.tol {
            if tol <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "tol",
                    reason: "tol must be greater than zero",
                });
            }
        }
        Ok(Self { tol: config.tol })
    }
}

impl<F> ComplexPairSplit<F> for CplxRealKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        roots: Vec<Complex<F>>,
    ) -> Result<(Vec<Complex<F>>, Vec<Complex<F>>), ExecInvariantViolation> {
        if roots.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let tol = self
            .tol
            .unwrap_or_else(|| F::epsilon() * F::from(100.0).unwrap());
        validate_conjugate_pairs(&roots, tol, "roots")?;
        cplxreal_checked(roots, self.tol).map_err(|_| ExecInvariantViolation::InvalidState {
            reason: "complex-pair split failed",
        })
    }
}

/// Constructor config for [`BilinearZpkKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilinearZpkConfig<F>
where
    F: RealField + Copy,
{
    /// Sample rate in Hz.
    pub fs: F,
}

/// Trait-first bilinear ZPK transform kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilinearZpkKernel<F>
where
    F: RealField + Copy,
{
    fs: F,
}

impl<F> KernelLifecycle for BilinearZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = BilinearZpkConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.fs <= F::zero() {
            return Err(ConfigError::InvalidArgument {
                arg: "fs",
                reason: "fs must be greater than zero",
            });
        }
        Ok(Self { fs: config.fs })
    }
}

impl<F> ZpkTransform<F> for BilinearZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<ZpkFormatFilter<F>, ExecInvariantViolation> {
        validate_relative_degree(&zpk.z, &zpk.p)?;
        Ok(bilinear_zpk_dyn(zpk, self.fs))
    }
}

/// Constructor config for [`Lp2LpZpkKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2LpZpkConfig<F>
where
    F: RealField + Copy,
{
    /// Optional output cutoff frequency.
    pub wo: Option<F>,
}

/// Trait-first lowpass-to-lowpass ZPK transform kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2LpZpkKernel<F>
where
    F: RealField + Copy,
{
    wo: Option<F>,
}

impl<F> KernelLifecycle for Lp2LpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = Lp2LpZpkConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(wo) = config.wo {
            if wo <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "wo",
                    reason: "wo must be greater than zero",
                });
            }
        }
        Ok(Self { wo: config.wo })
    }
}

impl<F> ZpkTransform<F> for Lp2LpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<ZpkFormatFilter<F>, ExecInvariantViolation> {
        validate_relative_degree(&zpk.z, &zpk.p)?;
        Ok(lp2lp_zpk_dyn(zpk, self.wo))
    }
}

/// Constructor config for [`Lp2HpZpkKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2HpZpkConfig<F>
where
    F: RealField + Copy,
{
    /// Optional output cutoff frequency.
    pub wo: Option<F>,
}

/// Trait-first lowpass-to-highpass ZPK transform kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2HpZpkKernel<F>
where
    F: RealField + Copy,
{
    wo: Option<F>,
}

impl<F> KernelLifecycle for Lp2HpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = Lp2HpZpkConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(wo) = config.wo {
            if wo <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "wo",
                    reason: "wo must be greater than zero",
                });
            }
        }
        Ok(Self { wo: config.wo })
    }
}

impl<F> ZpkTransform<F> for Lp2HpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<ZpkFormatFilter<F>, ExecInvariantViolation> {
        validate_relative_degree(&zpk.z, &zpk.p)?;
        Ok(lp2hp_zpk_dyn(zpk, self.wo))
    }
}

/// Constructor config for [`Lp2BpZpkKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2BpZpkConfig<F>
where
    F: RealField + Copy,
{
    /// Optional passband center frequency.
    pub wo: Option<F>,
    /// Optional passband width.
    pub bw: Option<F>,
}

/// Trait-first lowpass-to-bandpass ZPK transform kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2BpZpkKernel<F>
where
    F: RealField + Copy,
{
    wo: Option<F>,
    bw: Option<F>,
}

impl<F> KernelLifecycle for Lp2BpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = Lp2BpZpkConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(wo) = config.wo {
            if wo <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "wo",
                    reason: "wo must be greater than zero",
                });
            }
        }
        if let Some(bw) = config.bw {
            if bw <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "bw",
                    reason: "bw must be greater than zero",
                });
            }
        }
        Ok(Self {
            wo: config.wo,
            bw: config.bw,
        })
    }
}

impl<F> ZpkTransform<F> for Lp2BpZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<ZpkFormatFilter<F>, ExecInvariantViolation> {
        validate_relative_degree(&zpk.z, &zpk.p)?;
        Ok(lp2bp_zpk_dyn(zpk, self.wo, self.bw))
    }
}

/// Constructor config for [`Lp2BsZpkKernel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2BsZpkConfig<F>
where
    F: RealField + Copy,
{
    /// Optional stopband center frequency.
    pub wo: Option<F>,
    /// Optional stopband width.
    pub bw: Option<F>,
}

/// Trait-first lowpass-to-bandstop ZPK transform kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lp2BsZpkKernel<F>
where
    F: RealField + Copy,
{
    wo: Option<F>,
    bw: Option<F>,
}

impl<F> KernelLifecycle for Lp2BsZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    type Config = Lp2BsZpkConfig<F>;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if let Some(wo) = config.wo {
            if wo <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "wo",
                    reason: "wo must be greater than zero",
                });
            }
        }
        if let Some(bw) = config.bw {
            if bw <= F::zero() {
                return Err(ConfigError::InvalidArgument {
                    arg: "bw",
                    reason: "bw must be greater than zero",
                });
            }
        }
        Ok(Self {
            wo: config.wo,
            bw: config.bw,
        })
    }
}

impl<F> ZpkTransform<F> for Lp2BsZpkKernel<F>
where
    F: RealField + Float + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<ZpkFormatFilter<F>, ExecInvariantViolation> {
        validate_relative_degree(&zpk.z, &zpk.p)?;
        Ok(lp2bs_zpk_dyn(zpk, self.wo, self.bw))
    }
}

/// Constructor config for [`ZpkToTfKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZpkToTfConfig {
    /// Expected filter order.
    pub order: usize,
}

/// Trait-first ZPK-to-TF conversion kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZpkToTfKernel {
    order: usize,
}

impl KernelLifecycle for ZpkToTfKernel {
    type Config = ZpkToTfConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.order == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be greater than zero",
            });
        }
        Ok(Self {
            order: config.order,
        })
    }
}

impl<F> ZpkToTfDesign<F> for ZpkToTfKernel
where
    F: Float + RealField + Copy,
{
    fn run_alloc(
        &self,
        zeros: &[Complex<F>],
        poles: &[Complex<F>],
        gain: F,
    ) -> Result<BaFormatFilter<F>, ExecInvariantViolation> {
        let required_order = zeros.len().max(poles.len());
        if self.order < required_order {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be at least max(len(zeros), len(poles))",
            }
            .into());
        }
        Ok(zpk2tf_dyn(
            self.order,
            &zeros.to_vec(),
            &poles.to_vec(),
            gain,
        ))
    }
}

/// Constructor config for [`ZpkToSosKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZpkToSosConfig {
    /// Expected filter order.
    pub order: usize,
    /// Pairing strategy for roots.
    pub pairing: Option<ZpkPairing>,
    /// If true, use analog pairing semantics.
    pub analog: Option<bool>,
}

/// Trait-first ZPK-to-SOS conversion kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZpkToSosKernel {
    order: usize,
    pairing: Option<ZpkPairing>,
    analog: Option<bool>,
}

impl KernelLifecycle for ZpkToSosKernel {
    type Config = ZpkToSosConfig;

    fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
        if config.order == 0 {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be greater than zero",
            });
        }

        if config.analog.unwrap_or(false)
            && config
                .pairing
                .is_some_and(|p| !matches!(p, ZpkPairing::Minimal))
        {
            return Err(ConfigError::InvalidArgument {
                arg: "pairing",
                reason: "analog zpk2sos requires minimal pairing",
            });
        }

        Ok(Self {
            order: config.order,
            pairing: config.pairing,
            analog: config.analog,
        })
    }
}

impl<F> ZpkToSosDesign<F> for ZpkToSosKernel
where
    F: RealField + Float + Sum + Copy,
{
    fn run_alloc(
        &self,
        zpk: ZpkFormatFilter<F>,
    ) -> Result<SosFormatFilter<F>, ExecInvariantViolation> {
        let required_order = zpk.z.len().max(zpk.p.len());
        if self.order < required_order {
            return Err(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be at least max(len(zeros), len(poles))",
            }
            .into());
        }

        if self.analog.unwrap_or(false)
            && self
                .pairing
                .is_some_and(|p| !matches!(p, ZpkPairing::Minimal))
        {
            return Err(ConfigError::InvalidArgument {
                arg: "pairing",
                reason: "analog zpk2sos requires minimal pairing",
            }
            .into());
        }

        if self.analog.unwrap_or(false) && zpk.p.len() < zpk.z.len() {
            return Err(ConfigError::InvalidArgument {
                arg: "zpk",
                reason: "analog zpk2sos requires len(poles) >= len(zeros)",
            }
            .into());
        }

        let tol = F::epsilon() * F::from(100.0).unwrap();
        validate_conjugate_pairs(&zpk.z, tol, "zpk.z")?;
        validate_conjugate_pairs(&zpk.p, tol, "zpk.p")?;

        zpk2sos_dyn(self.order, zpk, self.pairing, self.analog).map_err(|_| {
            ConfigError::InvalidArgument {
                arg: "zpk",
                reason: "zpk2sos conversion failed",
            }
            .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BilinearZpkConfig, BilinearZpkKernel, ButterConfig, ButterKernel, CplxRealConfig,
        CplxRealKernel, FirWinConfig, FirWinKernel, IirFilterConfig, IirFilterKernel,
        RelativeDegreeConfig, RelativeDegreeKernel, ZpkToSosConfig, ZpkToSosKernel, ZpkToTfConfig,
        ZpkToTfKernel,
    };
    use crate::kernel::{ConfigError, KernelLifecycle};
    use crate::signal::filter::design::{
        bilinear_zpk_dyn, firwin_dyn, DigitalFilter, FilterBandType, FilterOutputType, FilterType,
        ZpkFormatFilter, ZpkPairing,
    };
    use crate::signal::traits::{
        ComplexPairSplit, FirWinDesign, IirDesign, RelativeDegreeDesign, ZpkToSosDesign,
        ZpkToTfDesign, ZpkTransform,
    };
    use crate::signal::windows::WindowBuilderOwned;
    use approx::assert_abs_diff_eq;
    use nalgebra::Complex;

    #[test]
    fn firwin_kernel_matches_reference_and_supports_run_into() {
        let kernel = FirWinKernel::try_new(FirWinConfig {
            numtaps: 9,
            cutoff: vec![0.2f64],
            width: None,
            window: Some(WindowBuilderOwned::Hamming),
            pass_zero: FilterBandType::Lowpass,
            scale: Some(true),
            fs: None,
        })
        .expect("firwin kernel should initialize");

        let expected = firwin_dyn::<f64, f64>(
            9,
            &[0.2f64],
            None,
            None::<&crate::signal::windows::Hamming>,
            &FilterBandType::Lowpass,
            Some(true),
            None,
        )
        .expect("reference firwin should succeed");

        let mut out = vec![0.0f64; 9];
        kernel
            .run_into(out.as_mut_slice())
            .expect("firwin run_into should succeed");
        out.iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 1e-8));
    }

    #[test]
    fn firwin_kernel_constructor_checks_conflicts() {
        let err = FirWinKernel::try_new(FirWinConfig {
            numtaps: 9,
            cutoff: vec![0.2f64],
            width: Some(0.1),
            window: Some(WindowBuilderOwned::Hamming),
            pass_zero: FilterBandType::Lowpass,
            scale: None,
            fs: None,
        })
        .expect_err("window+width conflict should fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "window/width",
                reason: "window and width cannot both be set",
            }
        );
    }

    #[test]
    fn iirfilter_kernel_designs_sos() {
        let kernel = IirFilterKernel::try_new(IirFilterConfig {
            order: 4,
            wn: vec![10.0f64, 50.0],
            rp: None,
            rs: None,
            btype: Some(FilterBandType::Bandpass),
            ftype: Some(FilterType::Butterworth),
            analog: Some(false),
            output: Some(FilterOutputType::Sos),
            fs: Some(1666.0),
        })
        .expect("iirfilter kernel should initialize");

        match kernel.run_alloc().expect("iirfilter run should succeed") {
            DigitalFilter::Sos(sos) => assert_eq!(sos.sos.len(), 4),
            _ => panic!("expected SOS output"),
        }
    }

    #[test]
    fn iirfilter_kernel_handles_chebyshev2_without_rp() {
        let kernel = IirFilterKernel::try_new(IirFilterConfig {
            order: 4,
            wn: vec![0.25f64],
            rp: None,
            rs: Some(40.0),
            btype: Some(FilterBandType::Lowpass),
            ftype: Some(FilterType::ChebyshevII),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: None,
        })
        .expect("chebyshev II kernel should initialize");

        match kernel
            .run_alloc()
            .expect("chebyshev II design should succeed")
        {
            DigitalFilter::Ba(ba) => {
                assert!(!ba.a.is_empty());
                assert!(!ba.b.is_empty());
            }
            _ => panic!("expected BA output"),
        }
    }

    #[test]
    fn butter_kernel_designs_ba() {
        let kernel = ButterKernel::try_new(ButterConfig {
            order: 3,
            wn: vec![0.2f64],
            btype: Some(FilterBandType::Lowpass),
            analog: Some(false),
            output: Some(FilterOutputType::Ba),
            fs: None,
        })
        .expect("butter kernel should initialize");

        match kernel.run_alloc().expect("butter design should succeed") {
            DigitalFilter::Ba(ba) => {
                assert!(!ba.a.is_empty());
                assert_eq!(ba.a.len(), ba.b.len());
            }
            _ => panic!("expected BA output"),
        }
    }

    #[test]
    fn relative_degree_kernel_rejects_improper_transfer_functions() {
        let kernel = RelativeDegreeKernel::try_new(RelativeDegreeConfig)
            .expect("relative degree kernel should initialize");
        let err = kernel
            .run_alloc(
                &[Complex::new(1.0f64, 0.0), Complex::new(2.0, 0.0)],
                &[Complex::new(1.0f64, 0.0)],
            )
            .expect_err("improper transfer function should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::Config(ConfigError::InvalidArgument {
                arg: "zpk",
                reason: "improper transfer function; poles must be >= zeros",
            })
        ));
    }

    #[test]
    fn bilinear_zpk_kernel_matches_reference() {
        let kernel = BilinearZpkKernel::try_new(BilinearZpkConfig { fs: 2.0f64 })
            .expect("bilinear kernel should initialize");
        let zpk = ZpkFormatFilter::new(
            vec![Complex::new(0.0f64, 0.0)],
            vec![Complex::new(-0.25f64, 0.3)],
            1.0f64,
        );
        let expected = bilinear_zpk_dyn(
            ZpkFormatFilter::new(
                vec![Complex::new(0.0f64, 0.0)],
                vec![Complex::new(-0.25f64, 0.3)],
                1.0f64,
            ),
            2.0,
        );
        let actual = kernel
            .run_alloc(zpk)
            .expect("bilinear kernel should run without panic");
        assert_eq!(actual.z.len(), expected.z.len());
        assert_eq!(actual.p.len(), expected.p.len());
        actual.z.iter().zip(expected.z.iter()).for_each(|(a, e)| {
            assert_abs_diff_eq!(a.re, e.re, epsilon = 1e-12);
            assert_abs_diff_eq!(a.im, e.im, epsilon = 1e-12);
        });
        actual.p.iter().zip(expected.p.iter()).for_each(|(a, e)| {
            assert_abs_diff_eq!(a.re, e.re, epsilon = 1e-12);
            assert_abs_diff_eq!(a.im, e.im, epsilon = 1e-12);
        });
        assert_abs_diff_eq!(actual.k, expected.k, epsilon = 1e-12);
    }

    #[test]
    fn cplxreal_kernel_rejects_unmatched_conjugates() {
        let kernel = CplxRealKernel::try_new(CplxRealConfig { tol: None })
            .expect("cplxreal kernel should initialize");
        let err = kernel
            .run_alloc(vec![Complex::new(0.1f64, 0.5), Complex::new(0.2f64, 0.0)])
            .expect_err("missing conjugate should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::Config(ConfigError::InvalidArgument {
                arg: "roots",
                reason: "complex roots must have matching conjugate pairs",
            })
        ));
    }

    #[test]
    fn zpk_to_tf_kernel_validates_order_against_input() {
        let kernel =
            ZpkToTfKernel::try_new(ZpkToTfConfig { order: 1 }).expect("zpk2tf should initialize");
        let err = kernel
            .run_alloc(
                &[Complex::new(0.0f64, 0.0), Complex::new(0.1f64, 0.0)],
                &[Complex::new(0.2f64, 0.0), Complex::new(0.3f64, 0.0)],
                1.0,
            )
            .expect_err("order too small should fail");
        assert!(matches!(
            err,
            crate::kernel::ExecInvariantViolation::Config(ConfigError::InvalidArgument {
                arg: "order",
                reason: "order must be at least max(len(zeros), len(poles))",
            })
        ));
    }

    #[test]
    fn zpk_to_sos_kernel_rejects_analog_non_minimal_pairing() {
        let err = ZpkToSosKernel::try_new(ZpkToSosConfig {
            order: 2,
            pairing: Some(ZpkPairing::Nearest),
            analog: Some(true),
        })
        .expect_err("analog + nearest pairing must fail");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "pairing",
                reason: "analog zpk2sos requires minimal pairing",
            }
        );
    }

    #[test]
    fn zpk_to_sos_kernel_emits_sections() {
        let kernel = ZpkToSosKernel::try_new(ZpkToSosConfig {
            order: 2,
            pairing: None,
            analog: Some(false),
        })
        .expect("zpk2sos kernel should initialize");
        let zpk = ZpkFormatFilter::new(
            vec![Complex::new(-1.0f64, 0.0), Complex::new(-1.0, 0.0)],
            vec![Complex::new(0.5f64, 0.0), Complex::new(0.25, 0.0)],
            1.0f64,
        );
        let sos = kernel.run_alloc(zpk).expect("zpk2sos should run");
        assert_eq!(sos.sos.len(), 1);
    }
}
