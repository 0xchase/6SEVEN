use std::os::fd::AsRawFd;

use tokio::io::unix::AsyncFd;
use xdp::nic::NicIndex;
use xdp::slab::{HeapSlab, Slab};
use xdp::socket::{XdpSocket, XdpSocketBuilder};
use xdp::umem::{FrameSize, UmemCfgBuilder};
use xdp::{CompletionRing, RingConfigBuilder, Umem, WakableFillRing, WakableTxRing};

use crate::{Error, FrameConfig, Interface};

const FRAME_COUNT: u32 = 16 * 1024;
const RING_COUNT: u32 = 2048;
const TX_BATCH: usize = 256;
const TX_WAKE_BATCH: usize = 1024;
const XDP_PACKET_HEADROOM: usize = xdp::libc::xdp::XDP_PACKET_HEADROOM as usize;
const XDP_2K_CAPACITY: usize = 2048 - XDP_PACKET_HEADROOM;
const XDP_4K_CAPACITY: usize = 4096 - XDP_PACKET_HEADROOM;

pub(crate) fn queue_count(interface_index: u32) -> Option<u32> {
    let queues = NicIndex::new(interface_index).queue_count().ok()?;
    let count = queues.rx_current.min(queues.tx_current);
    (count > 0).then_some(count)
}

pub(crate) struct Tx {
    endpoint: TxEndpoint,
    pending: HeapSlab,
}

unsafe impl Send for Tx {}

impl Tx {
    pub(crate) fn open(interface: &Interface, frames: FrameConfig) -> Result<Self, Error> {
        Ok(Self {
            endpoint: TxEndpoint::open(interface, frames)?,
            pending: HeapSlab::with_capacity(TX_BATCH),
        })
    }

    pub(crate) fn try_send(&mut self, frame: &[u8]) -> Result<bool, Error> {
        if frame.is_empty() {
            return Ok(true);
        }
        if self.pending.len() >= TX_BATCH {
            return Ok(false);
        }

        let Some(mut packet) = self.endpoint.try_alloc_packet() else {
            return Ok(false);
        };
        if let Err(err) = packet.append(frame) {
            self.endpoint.umem.free_packet(packet);
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("af_xdp packet append failed: {err}"),
            )));
        }

        if self.pending.push_front(packet).is_some() {
            return Err(Error::Io(std::io::Error::other(
                "af_xdp transmit batch unexpectedly full",
            )));
        }
        if self.pending.len() >= TX_BATCH {
            let _ = self.endpoint.try_flush(&mut self.pending)?;
        }
        Ok(true)
    }

    pub(crate) async fn flush(&mut self) -> Result<(), Error> {
        while !self.pending.is_empty() {
            let queued = self.endpoint.try_flush(&mut self.pending)?;
            if queued == 0 && !self.pending.is_empty() && !self.endpoint.kick()? {
                self.endpoint.wait_writable().await?;
            }
        }
        self.endpoint.force_wake().await?;
        Ok(())
    }

    pub(crate) async fn wait_writable(&mut self) -> Result<(), Error> {
        self.endpoint.wait_writable().await
    }
}

impl Drop for Tx {
    fn drop(&mut self) {
        let _ = self.endpoint.try_flush(&mut self.pending);
        let _ = self.endpoint.kick();
    }
}

struct TxEndpoint {
    umem: Umem,
    socket: AsyncFd<XdpSocket>,
    _fill_ring: WakableFillRing,
    completion_ring: CompletionRing,
    tx_ring: WakableTxRing,
    queued_since_wake: usize,
}

impl TxEndpoint {
    fn open(interface: &Interface, frames: FrameConfig) -> Result<Self, Error> {
        let queue_id = interface.claim_xdp_queue()?;
        let frame_size = frame_size_for(frames.tx_frame_len())?;
        let umem_cfg = UmemCfgBuilder {
            frame_size,
            frame_count: FRAME_COUNT,
            ..Default::default()
        }
        .build()
        .map_err(|e| Error::BackendUnavailable(format!("af_xdp umem config: {e}")))?;
        let umem = Umem::map(umem_cfg)
            .map_err(|e| Error::BackendUnavailable(format!("af_xdp umem map: {e}")))?;

        let ring_cfg = RingConfigBuilder {
            rx_count: 0,
            tx_count: RING_COUNT,
            fill_count: RING_COUNT,
            completion_count: RING_COUNT,
        }
        .build()
        .map_err(|e| Error::BackendUnavailable(format!("af_xdp ring config: {e}")))?;

        let mut socket_builder = XdpSocketBuilder::new()
            .map_err(|e| Error::BackendUnavailable(format!("af_xdp socket create: {e}")))?;
        let (rings, bind_flags) = socket_builder
            .build_wakable_rings(&umem, ring_cfg)
            .map_err(|e| Error::BackendUnavailable(format!("af_xdp ring map: {e}")))?;
        let socket = socket_builder
            .bind(NicIndex::new(interface.index()), queue_id, bind_flags)
            .map_err(|e| Error::BackendUnavailable(format!("af_xdp bind queue {queue_id}: {e}")))?;
        set_nonblocking(socket.as_raw_fd())?;
        let tx_ring = rings
            .tx_ring
            .ok_or_else(|| Error::BackendUnavailable("af_xdp missing tx ring".to_string()))?;
        let socket = AsyncFd::new(socket)?;

        Ok(Self {
            umem,
            socket,
            _fill_ring: rings.fill_ring,
            completion_ring: rings.completion_ring,
            tx_ring,
            queued_since_wake: 0,
        })
    }

    fn try_alloc_packet(&mut self) -> Option<xdp::Packet> {
        if let Some(packet) = unsafe { self.umem.alloc() } {
            return Some(packet);
        }

        self.reap_completions();
        unsafe { self.umem.alloc() }
    }

    fn reap_completions(&mut self) {
        while self.completion_ring.dequeue(&mut self.umem, TX_BATCH) != 0 {}
    }

    fn try_flush(&mut self, pending: &mut HeapSlab) -> Result<usize, Error> {
        self.reap_completions();
        let queued = match unsafe { self.tx_ring.send(pending, false) } {
            Ok(queued) => queued,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                self.reap_completions();
                0
            }
            Err(err) => return Err(Error::Io(err)),
        };

        self.queued_since_wake += queued;
        if self.queued_since_wake >= TX_WAKE_BATCH || (queued == 0 && !pending.is_empty()) {
            let _ = self.kick()?;
        }

        Ok(queued)
    }

    fn kick(&mut self) -> Result<bool, Error> {
        if self.queued_since_wake == 0 {
            return Ok(false);
        }

        loop {
            let rc = unsafe {
                libc::sendto(
                    self.socket.get_ref().as_raw_fd(),
                    std::ptr::null(),
                    0,
                    libc::MSG_DONTWAIT,
                    std::ptr::null(),
                    0,
                )
            };
            if rc >= 0 {
                self.queued_since_wake = 0;
                return Ok(true);
            }

            let err = std::io::Error::last_os_error();
            match err.kind() {
                std::io::ErrorKind::Interrupted => {}
                std::io::ErrorKind::WouldBlock => return Ok(false),
                _ => return Err(Error::Io(err)),
            }
        }
    }

    async fn force_wake(&mut self) -> Result<(), Error> {
        while self.queued_since_wake != 0 {
            if self.kick()? {
                return Ok(());
            }
            self.wait_writable().await?;
        }
        Ok(())
    }

    async fn wait_writable(&mut self) -> Result<(), Error> {
        let mut guard = self.socket.writable_mut().await?;
        guard.clear_ready();
        Ok(())
    }
}

fn set_nonblocking(fd: std::os::fd::RawFd) -> Result<(), Error> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    let rc = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if rc < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

fn frame_size_for(tx_frame_len: usize) -> Result<FrameSize, Error> {
    if tx_frame_len <= XDP_2K_CAPACITY {
        Ok(FrameSize::TwoK)
    } else if tx_frame_len <= XDP_4K_CAPACITY {
        Ok(FrameSize::FourK)
    } else {
        Err(Error::BackendUnavailable(format!(
            "af_xdp transmit frames are limited to {XDP_4K_CAPACITY} bytes with the current UMEM frame sizes; probe needs {tx_frame_len} bytes"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdp_uses_two_k_chunks_when_frame_fits() {
        assert!(matches!(
            frame_size_for(XDP_2K_CAPACITY),
            Ok(FrameSize::TwoK)
        ));
    }

    #[test]
    fn xdp_uses_four_k_chunks_for_larger_frames() {
        assert!(matches!(
            frame_size_for(XDP_2K_CAPACITY + 1),
            Ok(FrameSize::FourK)
        ));
    }

    #[test]
    fn xdp_rejects_frames_that_exceed_supported_chunk_capacity() {
        assert!(frame_size_for(XDP_4K_CAPACITY + 1).is_err());
    }
}
