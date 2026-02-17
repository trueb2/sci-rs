use core::fmt;

/// Validation errors raised at kernel construction or adapter binding time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// A required input or configuration field is empty.
    EmptyInput {
        /// Name of the argument that is empty.
        arg: &'static str,
    },
    /// A configuration argument value is invalid.
    InvalidArgument {
        /// Name of the argument.
        arg: &'static str,
        /// Human readable reason.
        reason: &'static str,
    },
    /// A contiguous 1D slice view could not be obtained.
    NonContiguous {
        /// Name of the argument that is non-contiguous.
        arg: &'static str,
    },
    /// Output/input lengths did not match required shape.
    LengthMismatch {
        /// Name of the argument.
        arg: &'static str,
        /// Required length.
        expected: usize,
        /// Received length.
        got: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::EmptyInput { arg } => write!(f, "Input `{arg}` was empty."),
            ConfigError::InvalidArgument { arg, reason } => {
                write!(f, "Invalid argument `{arg}`: {reason}")
            }
            ConfigError::NonContiguous { arg } => {
                write!(f, "Argument `{arg}` is not contiguous in memory.")
            }
            ConfigError::LengthMismatch { arg, expected, got } => {
                write!(
                    f,
                    "Length mismatch on `{arg}`. Expected {expected}, got {got}."
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

/// Runtime execution invariant violations for checked kernel entrypoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecInvariantViolation {
    /// An execution precondition was violated.
    InvalidState {
        /// Human readable reason.
        reason: &'static str,
    },
    /// Output length mismatched the expected runtime shape.
    LengthMismatch {
        /// Name of the argument.
        arg: &'static str,
        /// Required length.
        expected: usize,
        /// Received length.
        got: usize,
    },
    /// Adapter binding/configuration failure.
    Config(ConfigError),
}

impl From<ConfigError> for ExecInvariantViolation {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl fmt::Display for ExecInvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecInvariantViolation::InvalidState { reason } => {
                write!(f, "Execution invariant violation: {reason}")
            }
            ExecInvariantViolation::LengthMismatch { arg, expected, got } => {
                write!(
                    f,
                    "Execution length mismatch on `{arg}`. Expected {expected}, got {got}."
                )
            }
            ExecInvariantViolation::Config(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExecInvariantViolation {}
