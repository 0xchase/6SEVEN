use serde::{Deserialize, Serialize};

use crate::probe::dns::DnsConfig;
use crate::probe::icmp::IcmpConfig;
use crate::probe::ntp::NtpConfig;
use crate::probe::tcp::TcpConfig;
use crate::probe::udp::UdpConfig;
use crate::{CompiledProbe, PreparedProbe};

#[derive(thiserror::Error, Debug, Serialize, Deserialize)]
pub enum ProbeError {
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("network error: {0}")]
    NetworkError(String),
    #[error("internal error: {0}")]
    InternalError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbeConfig {
    Icmp(IcmpConfig),
    Tcp(TcpConfig),
    Udp(UdpConfig),
    Dns(DnsConfig),
    Ntp(NtpConfig),
}

impl ProbeConfig {
    pub(crate) fn compile(self) -> Result<CompiledProbe, ProbeError> {
        match self {
            Self::Icmp(cfg) => Ok(CompiledProbe::Icmp(cfg.into())),
            Self::Tcp(cfg) => Ok(CompiledProbe::Tcp(cfg.into())),
            Self::Udp(cfg) => Ok(CompiledProbe::Udp(cfg.into())),
            Self::Dns(cfg) => Ok(CompiledProbe::Dns(cfg.into())),
            Self::Ntp(cfg) => Ok(CompiledProbe::Ntp(cfg.into())),
        }
    }

    pub fn prepare_for(self, source: std::net::Ipv6Addr) -> Result<PreparedProbe, ProbeError> {
        self.compile().map(|probe| probe.bind(source))
    }
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self::Icmp(IcmpConfig)
    }
}
