use super::ConfigError;

/// Constructor validation lifecycle shared by kernel structs.
pub trait KernelLifecycle: Sized {
    /// Kernel config type.
    type Config;

    /// Construct a validated kernel from config.
    fn try_new(config: Self::Config) -> Result<Self, ConfigError>;
}
