use std::net::Ipv6Addr;
use std::str::FromStr;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU32, Ordering},
};

use pnet::packet::Packet;
use pnet::packet::ethernet::{EtherTypes, EthernetPacket, MutableEthernetPacket};
use pnet::packet::ip::IpNextHeaderProtocol;
use pnet::packet::ipv6::{Ipv6Packet, MutableIpv6Packet};
use pnet::util::MacAddr;
use serde::{Deserialize, Serialize};

mod afpacket;
mod afxdp;

pub const ETH_HEADER_LEN: usize = 14;
pub const IPV6_HEADER_LEN: usize = 40;
pub const DEFAULT_TX_FRAME_LEN: usize = 1514;
pub const DEFAULT_RX_FRAME_LEN: usize = 4096;
const IPV6_HOP_LIMIT: u8 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutboundIpv6Packet<'a> {
    pub destination: Ipv6Addr,
    pub next_header: IpNextHeaderProtocol,
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InboundIpv6Packet<'a> {
    pub source: Ipv6Addr,
    pub next_header: IpNextHeaderProtocol,
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecode<'a> {
    Packet(InboundIpv6Packet<'a>),
    NotMine,
    Malformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Icmpv6ErrorTransport<'a> {
    pub target: Ipv6Addr,
    pub protocol: IpNextHeaderProtocol,
    pub payload: &'a [u8],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    AfPacket,
    AfXdp,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AfPacket => "af-packet",
            Self::AfXdp => "af-xdp",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "af-packet" | "af_packet" | "afpacket" | "packet" => Ok(Self::AfPacket),
            "af-xdp" | "af_xdp" | "afxdp" | "xdp" => Ok(Self::AfXdp),
            other => Err(format!(
                "invalid backend '{other}', expected af-packet or af-xdp"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    tx_frame_len: usize,
    rx_frame_len: usize,
}

impl FrameConfig {
    pub fn new(tx_frame_len: usize, rx_frame_len: usize) -> Self {
        Self {
            tx_frame_len,
            rx_frame_len,
        }
    }

    pub fn tx_frame_len(self) -> usize {
        self.tx_frame_len
    }

    pub fn rx_frame_len(self) -> usize {
        self.rx_frame_len
    }
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            tx_frame_len: DEFAULT_TX_FRAME_LEN,
            rx_frame_len: DEFAULT_RX_FRAME_LEN,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenConfig {
    interface: Option<String>,
    backend: Backend,
    source_addr: Option<Ipv6Addr>,
}

impl OpenConfig {
    pub fn new() -> Self {
        Self {
            interface: None,
            backend: Backend::default(),
            source_addr: None,
        }
    }

    pub fn interface(mut self, name: impl Into<String>) -> Self {
        self.interface = Some(name.into());
        self
    }

    pub fn maybe_interface(mut self, name: Option<String>) -> Self {
        self.interface = name;
        self
    }

    pub fn source_addr(mut self, addr: Ipv6Addr) -> Self {
        self.source_addr = Some(addr);
        self
    }

    pub fn maybe_source_addr(mut self, addr: Option<Ipv6Addr>) -> Self {
        self.source_addr = addr;
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn interface_name(&self) -> Option<&str> {
        self.interface.as_deref()
    }

    pub fn source_addr_value(&self) -> Option<Ipv6Addr> {
        self.source_addr
    }

    pub fn backend_value(&self) -> Backend {
        self.backend
    }
}

impl Default for OpenConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Interface {
    name: String,
    index: u32,
    source_mac: MacAddr,
    source_addr: Ipv6Addr,
    gateway_mac: MacAddr,
    mtu: Option<usize>,
    fanout_group: u16,
    loopback: bool,
    backend: Backend,
    xdp_queue_count: Option<u32>,
    next_xdp_queue: Arc<AtomicU32>,
}

impl Interface {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn index(&self) -> u32 {
        self.index
    }

    pub fn source_addr(&self) -> Ipv6Addr {
        self.source_addr
    }

    pub(crate) fn gateway_mac(&self) -> MacAddr {
        self.gateway_mac
    }

    pub fn mtu(&self) -> Option<usize> {
        self.mtu
    }

    pub fn sender_shards(&self, requested: usize) -> usize {
        let requested = requested.max(1);
        match self.backend {
            Backend::AfPacket => requested,
            Backend::AfXdp => self
                .xdp_queue_count
                .map(|count| requested.min((count.max(1)) as usize))
                .unwrap_or(1),
        }
    }

    pub fn tx(&self, frames: FrameConfig) -> Result<Tx, Error> {
        match self.backend {
            Backend::AfPacket => afpacket::Tx::open(self, frames).map(Tx::afpacket),
            Backend::AfXdp => afxdp::Tx::open(self, frames).map(Tx::afxdp),
        }
    }

    pub fn rx(&self, frames: FrameConfig) -> Result<Rx, Error> {
        afpacket::Rx::open(self, frames).map(Rx::afpacket)
    }

    pub fn is_loopback(&self) -> bool {
        self.loopback
    }

    pub(crate) fn fanout_group(&self) -> u16 {
        self.fanout_group
    }

    pub(crate) fn claim_xdp_queue(&self) -> Result<u32, Error> {
        let queue = self.next_xdp_queue.fetch_add(1, Ordering::Relaxed);
        if let Some(count) = self.xdp_queue_count
            && queue >= count
        {
            return Err(Error::InvalidConfiguration(format!(
                "af_xdp requested queue {queue}, but interface {} reports {count} active queues; reduce shards or increase NIC queue count",
                self.name
            )));
        }
        Ok(queue)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("invalid interface: {0}")]
    InvalidInterface(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("backend not available: {0}")]
    BackendUnavailable(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct Tx(TxInner);

enum TxInner {
    AfPacket(afpacket::Tx),
    AfXdp(Box<afxdp::Tx>),
}

impl Tx {
    fn afpacket(tx: afpacket::Tx) -> Self {
        Self(TxInner::AfPacket(tx))
    }

    fn afxdp(tx: afxdp::Tx) -> Self {
        Self(TxInner::AfXdp(Box::new(tx)))
    }

    pub fn try_send(&mut self, frame: &[u8]) -> Result<bool, Error> {
        match &mut self.0 {
            TxInner::AfPacket(tx) => tx.try_send(frame),
            TxInner::AfXdp(tx) => tx.try_send(frame),
        }
    }

    pub async fn send(&mut self, frame: &[u8]) -> Result<(), Error> {
        let mut retried_after_flush = false;
        loop {
            if self.try_send(frame)? {
                return Ok(());
            }

            self.flush().await?;
            if retried_after_flush {
                self.wait_writable().await?;
            }
            retried_after_flush = true;
        }
    }

    pub async fn flush(&mut self) -> Result<(), Error> {
        match &mut self.0 {
            TxInner::AfPacket(tx) => tx.flush().await,
            TxInner::AfXdp(tx) => tx.flush().await,
        }
    }

    async fn wait_writable(&mut self) -> Result<(), Error> {
        match &mut self.0 {
            TxInner::AfPacket(tx) => tx.wait_writable().await,
            TxInner::AfXdp(tx) => tx.wait_writable().await,
        }
    }
}

pub struct Rx(RxInner);

enum RxInner {
    AfPacket(afpacket::Rx),
}

impl Rx {
    fn afpacket(rx: afpacket::Rx) -> Self {
        Self(RxInner::AfPacket(rx))
    }

    pub fn try_recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, Error> {
        match &mut self.0 {
            RxInner::AfPacket(rx) => rx.try_recv(buf),
        }
    }

    pub async fn recv(&mut self, buf: &mut [u8]) -> Result<usize, Error> {
        loop {
            if let Some(n) = self.try_recv(buf)? {
                return Ok(n);
            }
            self.wait_readable().await?;
        }
    }

    async fn wait_readable(&mut self) -> Result<(), Error> {
        match &mut self.0 {
            RxInner::AfPacket(rx) => rx.wait_readable().await,
        }
    }
}

pub fn encode_ipv6_frame(
    interface: &Interface,
    buf: &mut [u8],
    packet: OutboundIpv6Packet<'_>,
) -> Option<usize> {
    let payload_len = u16::try_from(packet.payload.len()).ok()?;
    let frame_len = ETH_HEADER_LEN
        .checked_add(IPV6_HEADER_LEN)?
        .checked_add(packet.payload.len())?;
    let frame = buf.get_mut(..frame_len)?;

    let (ethernet, ipv6_and_payload) = frame.split_at_mut(ETH_HEADER_LEN);
    let mut ethernet = MutableEthernetPacket::new(ethernet)?;
    ethernet.set_destination(interface.gateway_mac);
    ethernet.set_source(interface.source_mac);
    ethernet.set_ethertype(EtherTypes::Ipv6);

    let (ipv6_header, payload) = ipv6_and_payload.split_at_mut(IPV6_HEADER_LEN);
    let mut ipv6 = MutableIpv6Packet::new(ipv6_header)?;
    ipv6.set_version(6);
    ipv6.set_payload_length(payload_len);
    ipv6.set_next_header(packet.next_header);
    ipv6.set_hop_limit(IPV6_HOP_LIMIT);
    ipv6.set_source(interface.source_addr);
    ipv6.set_destination(packet.destination);
    payload.copy_from_slice(packet.payload);

    Some(frame_len)
}

pub fn decode_ipv6_frame<'a>(interface: &Interface, frame: &'a [u8]) -> FrameDecode<'a> {
    let Some(eth) = EthernetPacket::new(frame) else {
        return FrameDecode::Malformed;
    };
    if eth.get_destination() != interface.source_mac || eth.get_ethertype() != EtherTypes::Ipv6 {
        return FrameDecode::NotMine;
    }

    let Some(ipv6) = Ipv6Packet::new(eth.payload()) else {
        return FrameDecode::Malformed;
    };
    if ipv6.get_destination() != interface.source_addr {
        return FrameDecode::NotMine;
    }

    let payload_start = ETH_HEADER_LEN + IPV6_HEADER_LEN;
    let payload_len = usize::from(ipv6.get_payload_length());
    let Some(payload_end) = payload_start.checked_add(payload_len) else {
        return FrameDecode::Malformed;
    };
    let Some(ipv6_payload) = frame.get(payload_start..payload_end) else {
        return FrameDecode::Malformed;
    };
    let (next_header, payload) = walk_ipv6_headers(ipv6.get_next_header().0, ipv6_payload);
    FrameDecode::Packet(InboundIpv6Packet {
        source: ipv6.get_source(),
        next_header,
        payload,
    })
}

pub fn icmpv6_error_transport(
    payload: &[u8],
    our_ipv6: Ipv6Addr,
) -> Option<Icmpv6ErrorTransport<'_>> {
    if payload.len() < 8 + IPV6_HEADER_LEN {
        return None;
    }
    let icmp_type = payload[0];
    if !(1..=4).contains(&icmp_type) {
        return None;
    }
    let inner = &payload[8..];
    if (inner[0] >> 4) != 6 {
        return None;
    }
    let inner_src = Ipv6Addr::from(<[u8; 16]>::try_from(&inner[8..24]).ok()?);
    if inner_src != our_ipv6 {
        return None;
    }
    let target = Ipv6Addr::from(<[u8; 16]>::try_from(&inner[24..40]).ok()?);
    let next_header = inner[6];
    let (protocol, transport) = walk_ipv6_headers(next_header, &inner[IPV6_HEADER_LEN..]);
    Some(Icmpv6ErrorTransport {
        target,
        protocol,
        payload: transport,
    })
}

fn walk_ipv6_headers(mut next_header: u8, payload: &[u8]) -> (IpNextHeaderProtocol, &[u8]) {
    let mut offset = 0;
    loop {
        match next_header {
            0 | 43 | 60 | 51 => {
                if offset + 2 > payload.len() {
                    break;
                }
                let nh = payload[offset];
                let hdr_len = (payload[offset + 1] as usize + 1) * 8;
                if offset + hdr_len > payload.len() {
                    break;
                }
                next_header = nh;
                offset += hdr_len;
            }
            44 => {
                if offset + 8 > payload.len() {
                    break;
                }
                next_header = payload[offset];
                offset += 8;
            }
            _ => break,
        }
    }
    (IpNextHeaderProtocol::new(next_header), &payload[offset..])
}

pub fn open_interface(cfg: &OpenConfig) -> Result<Interface, Error> {
    let interface = match cfg.interface_name() {
        Some(name) => find_interface(name)?,
        None => default_interface().ok_or_else(|| {
            Error::InvalidConfiguration(
                "no interface specified and default IPv6 gateway not found".to_string(),
            )
        })?,
    };

    let source_addr = match cfg.source_addr_value() {
        Some(ip) => {
            let on_interface = interface.ipv6.iter().any(|net| net.addr == ip);
            if !on_interface {
                return Err(Error::InvalidConfiguration(format!(
                    "--src-ip {ip} is not assigned to interface {}",
                    interface.name
                )));
            }
            ip
        }
        None => interface_source_ipv6(&interface)?,
    };

    let is_loopback = interface.is_loopback();
    let source_mac = interface
        .mac_addr
        .map(default_mac_to_pnet)
        .or_else(|| is_loopback.then(MacAddr::zero))
        .ok_or_else(|| Error::InvalidConfiguration("interface missing MAC".to_string()))?;

    let gateway_mac = interface
        .gateway
        .as_ref()
        .map(|gateway| default_mac_to_pnet(gateway.mac_addr))
        .or_else(|| is_loopback.then_some(source_mac))
        .ok_or_else(|| Error::InvalidConfiguration("missing gateway MAC".to_string()))?;

    let backend = cfg.backend_value();
    let xdp_queue_count = match backend {
        Backend::AfPacket => None,
        Backend::AfXdp => afxdp::queue_count(interface.index),
    };
    let mtu = interface_mtu(&interface.name);

    Ok(Interface {
        name: interface.name,
        index: interface.index,
        source_mac,
        source_addr,
        gateway_mac,
        mtu,
        fanout_group: process_fanout_group(),
        loopback: is_loopback,
        backend,
        xdp_queue_count,
        next_xdp_queue: Arc::new(AtomicU32::new(0)),
    })
}

fn find_interface(name: &str) -> Result<default_net::Interface, Error> {
    default_net::get_interfaces()
        .into_iter()
        .find(|iface| iface.name == name)
        .ok_or_else(|| Error::InvalidInterface(format!("interface not found: {name}")))
}

fn default_interface() -> Option<default_net::Interface> {
    default_net::get_default_interface().ok()
}

fn interface_source_ipv6(interface: &default_net::Interface) -> Result<Ipv6Addr, Error> {
    interface
        .ipv6
        .iter()
        .find_map(|net| {
            if net.addr.is_unicast_link_local() {
                None
            } else {
                Some(net.addr)
            }
        })
        .ok_or_else(|| {
            Error::InvalidConfiguration("interface missing non-link-local IPv6".to_string())
        })
}

fn default_mac_to_pnet(mac: default_net::mac::MacAddr) -> MacAddr {
    MacAddr::from(mac.octets())
}

fn interface_mtu(name: &str) -> Option<usize> {
    std::fs::read_to_string(format!("/sys/class/net/{name}/mtu"))
        .ok()
        .and_then(|mtu| mtu.trim().parse().ok())
}

fn process_fanout_group() -> u16 {
    static GROUP: OnceLock<u16> = OnceLock::new();
    *GROUP.get_or_init(|| (std::process::id() as u16) ^ 0x6d5a)
}
