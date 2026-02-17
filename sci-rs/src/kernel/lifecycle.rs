use super::ConfigError;

/// Constructor validation lifecycle shared by kernel structs.
pub trait KernelLifecycle: Sized {
    /// Kernel config type.
    type Config;

    /// Construct a validated kernel from config.
    fn try_new(config: Self::Config) -> Result<Self, ConfigError>;
}

#[cfg(test)]
mod tests {
    use super::{ConfigError, KernelLifecycle};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DummyConfig {
        gain: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DummyKernel {
        gain: usize,
    }

    impl KernelLifecycle for DummyKernel {
        type Config = DummyConfig;

        fn try_new(config: Self::Config) -> Result<Self, ConfigError> {
            if config.gain == 0 {
                return Err(ConfigError::InvalidArgument {
                    arg: "gain",
                    reason: "gain must be greater than zero",
                });
            }
            Ok(Self { gain: config.gain })
        }
    }

    #[test]
    fn lifecycle_constructor_accepts_valid_config() {
        let kernel = DummyKernel::try_new(DummyConfig { gain: 4 }).expect("valid config");
        assert_eq!(kernel.gain, 4);
    }

    #[test]
    fn lifecycle_constructor_rejects_invalid_config() {
        let err = DummyKernel::try_new(DummyConfig { gain: 0 }).expect_err("invalid config");
        assert_eq!(
            err,
            ConfigError::InvalidArgument {
                arg: "gain",
                reason: "gain must be greater than zero",
            }
        );
    }
}
