//! Shared trait-first kernel substrate.
//!
//! This module defines reusable interfaces for constructor validation and
//! 1D buffer/stream adapters used by trait-first signal/stat kernels.

mod errors;
mod io;
mod lifecycle;

pub use errors::*;
pub use io::*;
pub use lifecycle::*;
