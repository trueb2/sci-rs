use core::{error, fmt};

/// Errors raised whilst running sci-rs.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Argument parsed into function were invalid.
    #[cfg(feature = "alloc")]
    InvalidArg {
        /// The invalid arg
        arg: alloc::string::String,
        /// Explaining why arg is invalid.
        reason: alloc::string::String,
    },
    /// Argument parsed into function were invalid.
    #[cfg(not(feature = "alloc"))]
    InvalidArg,
    /// Two or more optional arguments passed into functions conflict.
    #[cfg(feature = "alloc")]
    ConflictArg {
        /// Explaining what arg is invalid.
        reason: alloc::string::String,
    },
    /// Two or more optional arguments passed into functions conflict.
    #[cfg(not(feature = "alloc"))]
    ConflictArg,
    /// Execution was attempted with a violated kernel invariant.
    #[cfg(feature = "alloc")]
    ExecInvariantViolation {
        /// Why execution could not proceed.
        reason: alloc::string::String,
    },
    /// Execution was attempted with a violated kernel invariant.
    #[cfg(not(feature = "alloc"))]
    ExecInvariantViolation,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "alloc")]
            Error::InvalidArg { arg, reason } => {
                write!(f, "Invalid argument `{arg}`: {reason}")
            }
            #[cfg(not(feature = "alloc"))]
            Error::InvalidArg => write!(f, "Invalid argument."),
            #[cfg(feature = "alloc")]
            Error::ConflictArg { reason } => write!(f, "Conflicting arguments: {reason}"),
            #[cfg(not(feature = "alloc"))]
            Error::ConflictArg => write!(f, "Conflicting arguments."),
            #[cfg(feature = "alloc")]
            Error::ExecInvariantViolation { reason } => {
                write!(f, "Execution invariant violation: {reason}")
            }
            #[cfg(not(feature = "alloc"))]
            Error::ExecInvariantViolation => write!(f, "Execution invariant violation."),
        }
    }
}

impl error::Error for Error {}
