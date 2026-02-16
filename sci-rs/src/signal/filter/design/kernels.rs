//! Trait-first kernels for filter design APIs.

use crate::kernel::{ConfigError, ExecInvariantViolation, KernelLifecycle, Write1D};
use crate::signal::traits::{FirWinDesign, IirDesign};
use crate::signal::windows::{GetWindow, WindowBuilderOwned};
use crate::special::Bessel;
use core::iter::Sum;
use nalgebra::RealField;
use num_traits::{real::Real, Float, MulAdd, Pow};

use alloc::vec::Vec;

use super::{
    butter_dyn, firwin_dyn, iirfilter_dyn, DigitalFilter, FilterBandType, FilterOutputType,
    FilterType,
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
        // SciPy-equivalent Chebyshev II does not consume `rp`; populate a benign value
        // so legacy `iirfilter_dyn` validation does not panic in the transition period.
        let rp = if matches!(self.ftype, Some(FilterType::ChebyshevII)) {
            self.rp.or(Some(F::one()))
        } else {
            self.rp
        };

        Ok(iirfilter_dyn(
            self.order,
            self.wn.clone(),
            rp,
            self.rs,
            self.btype,
            self.ftype,
            self.analog,
            self.output,
            self.fs,
        ))
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
        Ok(butter_dyn(
            self.order,
            self.wn.clone(),
            self.btype,
            self.analog,
            self.output,
            self.fs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ButterConfig, ButterKernel, FirWinConfig, FirWinKernel, IirFilterConfig, IirFilterKernel,
    };
    use crate::kernel::{ConfigError, KernelLifecycle};
    use crate::signal::filter::design::{
        firwin_dyn, DigitalFilter, FilterBandType, FilterOutputType, FilterType,
    };
    use crate::signal::traits::{FirWinDesign, IirDesign};
    use crate::signal::windows::WindowBuilderOwned;
    use approx::assert_abs_diff_eq;

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
}
