use std::net::Ipv6Addr;

pub(crate) const TARGET_BATCH_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbePurpose {
    Discovery,
    Alias,
}

pub(crate) struct TargetBatch {
    pub purpose: ProbePurpose,
    pub token: u8,
    pub id: u64,
    targets: Box<[Ipv6Addr]>,
    len: usize,
}

impl TargetBatch {
    pub(crate) fn new() -> Self {
        Self {
            purpose: ProbePurpose::Discovery,
            token: 0,
            id: 0,
            targets: Box::new([Ipv6Addr::UNSPECIFIED; TARGET_BATCH_SIZE]),
            len: 0,
        }
    }

    pub(crate) fn alias(targets: [Ipv6Addr; 3], token: u8, id: u64) -> Self {
        Self {
            purpose: ProbePurpose::Alias,
            token,
            id,
            targets: Box::new(targets),
            len: 3,
        }
    }

    pub(crate) fn push(&mut self, target: Ipv6Addr) -> bool {
        if self.is_full() {
            return false;
        }

        self.targets[self.len] = target;
        self.len += 1;
        true
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_full(&self) -> bool {
        self.len == self.targets.len()
    }

    pub(crate) fn as_slice(&self) -> &[Ipv6Addr] {
        &self.targets[..self.len]
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = Ipv6Addr> + '_ {
        self.targets[..self.len].iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_insert_order() {
        let first = "2001:db8::1".parse().unwrap();
        let second = "2001:db8::2".parse().unwrap();
        let mut batch = TargetBatch::new();

        assert!(batch.push(first));
        assert!(batch.push(second));

        let targets: Vec<_> = batch.iter().collect();
        assert_eq!(targets, vec![first, second]);
    }
}
