use clap::Args;
use link::{Backend, OpenConfig};
use scan::finite::{Config as RuntimeConfig, Error as RuntimeError};
use serde::{Deserialize, Serialize};
use std::{net::Ipv6Addr, time::Duration};

#[derive(Args, Clone, Serialize, Deserialize)]
pub struct RuntimeArgs {
    #[arg(short = 'r', long, default_value = "10000")]
    pub pps: usize,
    #[arg(short = 't', long, default_value = "8")]
    pub timeout: usize,
    /// Network worker count
    #[arg(long, default_value_t = 1)]
    pub network_workers: usize,
    #[serde(default)]
    #[arg(long = "backend", default_value_t = Backend::AfPacket)]
    pub backend: Backend,
    #[arg(short, long)]
    pub interface: Option<String>,
    #[arg(short, long)]
    pub src_ip: Option<String>,
}

impl RuntimeArgs {
    pub(crate) fn open_config(&self) -> Result<OpenConfig, RuntimeError> {
        let source_addr = self
            .src_ip
            .as_deref()
            .map(|addr| {
                addr.parse::<Ipv6Addr>().map_err(|err| {
                    RuntimeError::InvalidConfiguration(format!(
                        "invalid IPv6 source address '{addr}': {err}"
                    ))
                })
            })
            .transpose()?;

        Ok(OpenConfig::new()
            .backend(self.backend)
            .maybe_interface(self.interface.clone())
            .maybe_source_addr(source_addr))
    }

    pub(crate) fn apply(&self, config: RuntimeConfig) -> RuntimeConfig {
        config
            .pps(self.pps)
            .network_workers(self.network_workers)
            .receive_window(Duration::from_secs(self.timeout as u64))
            .stats_interval(None)
    }
}
