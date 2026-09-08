use std::net::Ipv6Addr;

pub struct RepeatedTargets {
    target: Ipv6Addr,
    remaining: usize,
}
impl RepeatedTargets {
    pub fn new(target: Ipv6Addr, count: usize) -> Self {
        Self {
            target,
            remaining: count,
        }
    }
}
impl Iterator for RepeatedTargets {
    type Item = Result<Ipv6Addr, sixseven_core::TgaError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(Ok(self.target))
    }
}
