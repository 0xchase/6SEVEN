use std::mem::{size_of, zeroed};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{Ordering, fence};

use pnet::util::MacAddr;
use tokio::io::unix::AsyncFd;

use crate::{Error, FrameConfig, Interface};

const ETH_P_IPV6: u16 = 0x86DD;
const PACKET_IGNORE_OUTGOING: libc::c_int = 23;
const PACKET_QDISC_BYPASS: libc::c_int = 20;
const PACKET_FANOUT: libc::c_int = 18;
const PACKET_FANOUT_LB: libc::c_uint = 1;
const PACKET_FANOUT_FLAG_ROLLOVER: libc::c_uint = 0x1000;
const PACKET_LOSS: libc::c_int = 14;

const TX_BLOCK_SIZE: u32 = 1 << 20;
const TX_BLOCKS: u32 = 32;
const TX_MIN_FRAME_SIZE: usize = 2048;
const TX_KICK_BATCH: usize = 512;

const RX_BLOCK_SIZE: u32 = 1 << 20;
const RX_BLOCKS: u32 = 16;
const RX_MIN_FRAME_SIZE: usize = 4096;

const TX_DATA_OFFSET: usize = libc::TPACKET2_HDRLEN - size_of::<libc::sockaddr_ll>();
const RX_DATA_RESERVE: usize = libc::TPACKET2_HDRLEN;

pub struct Tx {
    fd: AsyncFd<OwnedFd>,
    ring: PacketRing,
    destination: [u8; 6],
    current: usize,
    pending: usize,
}

impl Tx {
    pub(crate) fn open(interface: &Interface, frames: FrameConfig) -> Result<Self, Error> {
        let fd = open_socket()?;
        set_nonblocking(&fd)?;
        set_packet_int(&fd, PACKET_QDISC_BYPASS, 1)?;
        set_packet_int(&fd, PACKET_LOSS, 1)?;
        set_packet_version(&fd)?;
        let req = tx_ring_request(frames.tx_frame_len())?;
        set_packet_ring(&fd, libc::PACKET_TX_RING, &req)?;
        bind_interface(&fd, interface.index() as i32)?;
        let ring = PacketRing::map(&fd, req)?;
        let fd = AsyncFd::new(fd)?;

        Ok(Self {
            fd,
            ring,
            destination: mac_octets(interface.gateway_mac()),
            current: 0,
            pending: 0,
        })
    }

    pub fn try_send(&mut self, frame: &[u8]) -> Result<bool, Error> {
        if frame.len() < 14 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ethernet frame too short",
            )));
        }
        if frame.len() > self.ring.frame_size - TX_DATA_OFFSET {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ethernet frame too large for transmit ring",
            )));
        }
        debug_assert_eq!(&frame[..6], self.destination.as_slice());

        let index = self.current;
        let hdr = self.ring.frame(index).cast::<libc::tpacket2_hdr>();
        match read_status(hdr) {
            libc::TP_STATUS_AVAILABLE => {}
            libc::TP_STATUS_WRONG_FORMAT => {
                write_status(hdr, libc::TP_STATUS_AVAILABLE);
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "kernel rejected transmit ring frame",
                )));
            }
            _ => return Ok(false),
        }

        let frame_base = self.ring.frame(index);

        unsafe {
            (*hdr.as_ptr()).tp_len = frame.len() as u32;
            (*hdr.as_ptr()).tp_snaplen = frame.len() as u32;
            (*hdr.as_ptr()).tp_mac = TX_DATA_OFFSET as u16;
            (*hdr.as_ptr()).tp_net = (TX_DATA_OFFSET + 14) as u16;
            std::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                frame_base.as_ptr().add(TX_DATA_OFFSET),
                frame.len(),
            );
        }

        fence(Ordering::Release);
        write_status(hdr, libc::TP_STATUS_SEND_REQUEST);

        self.current = self.ring.next(index);
        self.pending += 1;
        if self.pending >= TX_KICK_BATCH {
            match self.flush_now() {
                Ok(()) => {}
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(err) => return Err(err),
            }
        }
        Ok(true)
    }

    pub async fn flush(&mut self) -> Result<(), Error> {
        loop {
            match self.flush_now() {
                Ok(()) => return Ok(()),
                Err(Error::Io(err)) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    self.wait_writable().await?;
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn flush_now(&mut self) -> Result<(), Error> {
        if self.pending == 0 {
            return Ok(());
        }

        loop {
            let rc = unsafe { libc::send(self.fd.get_ref().as_raw_fd(), std::ptr::null(), 0, 0) };
            if rc >= 0 {
                self.pending = 0;
                return Ok(());
            }

            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                return Err(Error::Io(err));
            }
        }
    }

    pub async fn wait_writable(&mut self) -> Result<(), Error> {
        let mut guard = self.fd.writable_mut().await?;
        guard.clear_ready();
        Ok(())
    }
}

impl Drop for Tx {
    fn drop(&mut self) {
        let _ = self.flush_now();
    }
}

pub struct Rx {
    fd: AsyncFd<OwnedFd>,
    ring: PacketRing,
    current: usize,
}

impl Rx {
    pub(crate) fn open(interface: &Interface, frames: FrameConfig) -> Result<Self, Error> {
        let fd = open_socket()?;
        set_nonblocking(&fd)?;
        set_packet_version(&fd)?;
        let req = rx_ring_request(frames.rx_frame_len())?;
        set_packet_ring(&fd, libc::PACKET_RX_RING, &req)?;
        bind_interface(&fd, interface.index() as i32)?;

        set_packet_int(&fd, PACKET_IGNORE_OUTGOING, 1)?;
        set_packet_int(&fd, PACKET_QDISC_BYPASS, 1)?;
        set_packet_fanout(&fd, interface.fanout_group())?;
        let ring = PacketRing::map(&fd, req)?;
        let fd = AsyncFd::new(fd)?;

        Ok(Self {
            fd,
            ring,
            current: 0,
        })
    }

    pub fn try_recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, Error> {
        let index = self.current;
        let frame_base = self.ring.frame(index);
        let hdr = frame_base.cast::<libc::tpacket2_hdr>();
        let status = read_status(hdr);

        if status & libc::TP_STATUS_USER == 0 {
            return Ok(None);
        }

        fence(Ordering::Acquire);
        let copied = unsafe {
            let hdr_ref = &*hdr.as_ptr();
            let start = hdr_ref.tp_mac as usize;
            let available = hdr_ref.tp_snaplen as usize;
            if start
                .checked_add(available)
                .is_none_or(|end| end > self.ring.frame_size)
            {
                self.release_rx_frame(hdr);
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "receive ring frame has invalid packet bounds",
                )));
            }

            let copied = available.min(buf.len());
            std::ptr::copy_nonoverlapping(frame_base.as_ptr().add(start), buf.as_mut_ptr(), copied);
            copied
        };

        self.release_rx_frame(hdr);
        self.current = self.ring.next(index);
        Ok(Some(copied))
    }

    pub async fn wait_readable(&mut self) -> Result<(), Error> {
        let mut guard = self.fd.readable_mut().await?;
        guard.clear_ready();
        Ok(())
    }

    fn release_rx_frame(&self, hdr: NonNull<libc::tpacket2_hdr>) {
        fence(Ordering::Release);
        write_status(hdr, libc::TP_STATUS_KERNEL);
    }
}

struct PacketRing {
    map: NonNull<u8>,
    len: usize,
    frame_size: usize,
    frame_count: usize,
}

unsafe impl Send for PacketRing {}

impl PacketRing {
    fn map(fd: &OwnedFd, req: libc::tpacket_req) -> Result<Self, Error> {
        let len = (req.tp_block_size as usize)
            .checked_mul(req.tp_block_nr as usize)
            .ok_or_else(|| {
                Error::InvalidConfiguration("packet ring size overflows usize".to_string())
            })?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }

        Ok(Self {
            map: NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
                Error::Io(std::io::Error::other("packet ring mmap returned null"))
            })?,
            len,
            frame_size: req.tp_frame_size as usize,
            frame_count: req.tp_frame_nr as usize,
        })
    }

    fn frame(&self, index: usize) -> NonNull<u8> {
        debug_assert!(index < self.frame_count);
        unsafe { NonNull::new_unchecked(self.map.as_ptr().add(index * self.frame_size)) }
    }

    fn next(&self, index: usize) -> usize {
        if index + 1 == self.frame_count {
            0
        } else {
            index + 1
        }
    }
}

impl Drop for PacketRing {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map.as_ptr().cast::<libc::c_void>(), self.len);
        }
    }
}

fn open_socket() -> Result<OwnedFd, Error> {
    let fd = unsafe {
        libc::socket(
            libc::AF_PACKET,
            libc::SOCK_RAW,
            i32::from(ETH_P_IPV6.to_be()) as libc::c_int,
        )
    };
    if fd < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn set_nonblocking(fd: &OwnedFd) -> Result<(), Error> {
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    let rc = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if rc < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

fn bind_interface(fd: &OwnedFd, ifindex: i32) -> Result<(), Error> {
    let mut addr = protocol_addr(ifindex);
    let rc = unsafe {
        libc::bind(
            fd.as_raw_fd(),
            (&mut addr as *mut libc::sockaddr_ll).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_ll>() as libc::socklen_t,
        )
    };
    if rc != 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

fn protocol_addr(ifindex: i32) -> libc::sockaddr_ll {
    let mut addr: libc::sockaddr_ll = unsafe { zeroed() };
    addr.sll_family = libc::AF_PACKET as u16;
    addr.sll_protocol = ETH_P_IPV6.to_be();
    addr.sll_ifindex = ifindex;
    addr
}

fn tx_ring_request(frame_len: usize) -> Result<libc::tpacket_req, Error> {
    let frame_size = ring_frame_size(frame_len, TX_DATA_OFFSET, TX_MIN_FRAME_SIZE, TX_BLOCK_SIZE)?;
    Ok(ring_request(TX_BLOCK_SIZE, TX_BLOCKS, frame_size))
}

fn rx_ring_request(frame_len: usize) -> Result<libc::tpacket_req, Error> {
    let frame_size = ring_frame_size(frame_len, RX_DATA_RESERVE, RX_MIN_FRAME_SIZE, RX_BLOCK_SIZE)?;
    Ok(ring_request(RX_BLOCK_SIZE, RX_BLOCKS, frame_size))
}

fn ring_frame_size(
    frame_len: usize,
    reserve: usize,
    minimum: usize,
    block_size: u32,
) -> Result<u32, Error> {
    let required = frame_len
        .checked_add(reserve)
        .ok_or_else(|| Error::InvalidConfiguration("packet ring frame size overflow".to_string()))?
        .max(minimum);
    let frame_size = required.checked_next_power_of_two().ok_or_else(|| {
        Error::InvalidConfiguration("packet ring frame size overflow".to_string())
    })?;
    if frame_size > block_size as usize {
        return Err(Error::InvalidConfiguration(format!(
            "packet ring frame size {frame_size} exceeds block size {block_size}"
        )));
    }
    Ok(frame_size as u32)
}

fn ring_request(block_size: u32, block_count: u32, frame_size: u32) -> libc::tpacket_req {
    libc::tpacket_req {
        tp_block_size: block_size,
        tp_block_nr: block_count,
        tp_frame_size: frame_size,
        tp_frame_nr: (block_size / frame_size) * block_count,
    }
}

fn set_packet_version(fd: &OwnedFd) -> Result<(), Error> {
    let version = libc::tpacket_versions::TPACKET_V2 as libc::c_int;
    set_packet_int(fd, libc::PACKET_VERSION, version)
}

fn set_packet_ring(fd: &OwnedFd, opt: libc::c_int, req: &libc::tpacket_req) -> Result<(), Error> {
    set_sockopt(
        fd,
        libc::SOL_PACKET,
        opt,
        (req as *const libc::tpacket_req).cast::<libc::c_void>(),
        size_of::<libc::tpacket_req>() as libc::socklen_t,
    )
}

fn set_packet_fanout(fd: &OwnedFd, group: u16) -> Result<(), Error> {
    let mode = PACKET_FANOUT_LB | PACKET_FANOUT_FLAG_ROLLOVER;
    let arg: libc::c_uint = (group as libc::c_uint) | (mode << 16);
    set_sockopt(
        fd,
        libc::SOL_PACKET,
        PACKET_FANOUT,
        (&arg as *const libc::c_uint).cast::<libc::c_void>(),
        size_of::<libc::c_uint>() as libc::socklen_t,
    )
}

fn set_packet_int(fd: &OwnedFd, opt: libc::c_int, value: libc::c_int) -> Result<(), Error> {
    set_sockopt(
        fd,
        libc::SOL_PACKET,
        opt,
        (&value as *const libc::c_int).cast::<libc::c_void>(),
        size_of::<libc::c_int>() as libc::socklen_t,
    )
}

fn set_sockopt(
    fd: &OwnedFd,
    level: libc::c_int,
    opt: libc::c_int,
    value: *const libc::c_void,
    value_len: libc::socklen_t,
) -> Result<(), Error> {
    let rc = unsafe { libc::setsockopt(fd.as_raw_fd(), level, opt, value, value_len) };
    if rc != 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

fn read_status(hdr: NonNull<libc::tpacket2_hdr>) -> u32 {
    unsafe { std::ptr::read_volatile(&(*hdr.as_ptr()).tp_status) }
}

fn write_status(hdr: NonNull<libc::tpacket2_hdr>, status: u32) {
    unsafe {
        std::ptr::write_volatile(&mut (*hdr.as_ptr()).tp_status, status);
    }
}

fn mac_octets(mac: MacAddr) -> [u8; 6] {
    [mac.0, mac.1, mac.2, mac.3, mac.4, mac.5]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_ring_uses_minimum_frame_for_small_probes() {
        let req = tx_ring_request(128).unwrap();
        assert_eq!(req.tp_frame_size, TX_MIN_FRAME_SIZE as u32);
    }

    #[test]
    fn tx_ring_grows_to_fit_large_frames() {
        let req = tx_ring_request(4096).unwrap();
        assert!(req.tp_frame_size as usize >= 4096 + TX_DATA_OFFSET);
        assert!(req.tp_frame_size.is_power_of_two());
    }

    #[test]
    fn rx_ring_grows_to_fit_large_frames() {
        let req = rx_ring_request(8192).unwrap();
        assert!(req.tp_frame_size as usize >= 8192 + RX_DATA_RESERVE);
        assert!(req.tp_frame_size.is_power_of_two());
    }
}
