use super::{Config, OnlineConfig};
use crate::finite::Error;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sixseven_core::{Ipv6Prefix, PrefixSet};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::Ipv6Addr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    Pending(Ipv6Prefix),
    Aliased,
    NotDetected,
}

#[derive(Debug, Clone)]
pub(crate) struct Test {
    pub prefix: Ipv6Prefix,
    pub targets: [Ipv6Addr; 3],
    // The probe protocol has three correlation bits. A token identifies a
    // prefix length, preventing late replies from satisfying a child test.
    pub token: u8,
}
struct Pending {
    test: Test,
    replies: u8,
}

pub(crate) struct Detector {
    pub aliases: PrefixSet,
    online: Option<OnlineConfig>,
    complete: HashSet<Ipv6Prefix>,
    pending: HashMap<Ipv6Prefix, Pending>,
    queue: VecDeque<Ipv6Prefix>,
    samples: HashMap<(Ipv6Addr, u8), (Ipv6Prefix, u8)>,
    discovered: Vec<Ipv6Prefix>,
}
impl Detector {
    pub fn new(mut config: Config) -> Result<Self, Error> {
        if let Some(online) = &mut config.online {
            online.validate()?;
        }
        Ok(Self {
            aliases: config.aliases,
            online: config.online,
            complete: HashSet::new(),
            pending: HashMap::new(),
            queue: VecDeque::new(),
            samples: HashMap::new(),
            discovered: Vec::new(),
        })
    }
    pub fn consider(&mut self, address: Ipv6Addr) -> Status {
        if self.aliases.contains(address) {
            return Status::Aliased;
        }
        let Some(config) = &self.online else {
            return Status::NotDetected;
        };
        for (token, &length) in config.prefix_lengths.iter().enumerate() {
            let prefix = Ipv6Prefix::new(address, length).unwrap().trunc();
            if self.complete.contains(&prefix) {
                continue;
            }
            if !self.pending.contains_key(&prefix) {
                let mut seed = prefix.network().octets();
                for (n, byte) in config.seed.to_be_bytes().iter().enumerate() {
                    seed[n] ^= byte;
                }
                seed[15] ^= length;
                let mut full_seed = [0; 32];
                full_seed[..16].copy_from_slice(&seed);
                full_seed[16..].copy_from_slice(&prefix.network().octets());
                let mut rng = StdRng::from_seed(full_seed);
                let mask = u128::MAX.checked_shr(u32::from(length)).unwrap_or(0);
                let mut targets = [Ipv6Addr::UNSPECIFIED; 3];
                for n in 0..3 {
                    loop {
                        let target = Ipv6Addr::from(
                            u128::from(prefix.network()) | (rng.r#gen::<u128>() & mask),
                        );
                        if !targets[..n].contains(&target) {
                            targets[n] = target;
                            break;
                        }
                    }
                }
                let test = Test {
                    prefix,
                    targets,
                    token: token as u8,
                };
                self.pending.insert(prefix, Pending { test, replies: 0 });
                self.queue.push_back(prefix);
            }
            return Status::Pending(prefix);
        }
        Status::NotDetected
    }
    pub fn next_test(&mut self) -> Option<Test> {
        while let Some(prefix) = self.queue.pop_front() {
            if self.aliases.covers(prefix) {
                self.pending.remove(&prefix);
                continue;
            }
            let test = self.pending.get(&prefix)?.test.clone();
            for (n, &target) in test.targets.iter().enumerate() {
                self.samples.insert((target, test.token), (prefix, n as u8));
            }
            return Some(test);
        }
        None
    }
    pub fn reply(&mut self, address: Ipv6Addr, token: u8) {
        if let Some(&(prefix, slot)) = self.samples.get(&(address, token)) {
            if let Some(pending) = self.pending.get_mut(&prefix) {
                pending.replies |= 1 << slot;
                if pending.replies.count_ones() >= 2 && self.aliases.insert(prefix) {
                    self.discovered.push(prefix);
                }
            }
        }
    }
    pub fn finish(&mut self, prefix: Ipv6Prefix) {
        if let Some(pending) = self.pending.remove(&prefix) {
            for target in pending.test.targets {
                self.samples.remove(&(target, pending.test.token));
            }
            self.complete.insert(prefix);
        }
    }
    pub fn take_aliases(&mut self) -> Vec<Ipv6Prefix> {
        std::mem::take(&mut self.discovered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn detector() -> Detector {
        Detector::new(Config {
            online: Some(OnlineConfig {
                prefix_lengths: vec![112, 120],
                seed: 0,
            }),
            ..Config::default()
        })
        .unwrap()
    }
    #[test]
    fn duplicate_replies_and_cached_parent_tests() {
        let mut d = detector();
        let ip = "2001:db8::42".parse().unwrap();
        assert!(matches!(d.consider(ip), Status::Pending(_)));
        let t = d.next_test().unwrap();
        assert!(matches!(
            d.consider("2001:db8::43".parse().unwrap()),
            Status::Pending(_)
        ));
        assert!(d.next_test().is_none());
        d.reply(t.targets[0], t.token);
        d.reply(t.targets[0], t.token);
        assert!(d.aliases.is_empty());
        d.finish(t.prefix);
        assert!(matches!(d.consider(ip), Status::Pending(_)));
        let child = d.next_test().unwrap();
        assert_eq!(child.prefix.prefix_len(), 120);
        d.reply(child.targets[0], child.token);
        d.reply(child.targets[1], child.token);
        assert_eq!(d.consider(ip), Status::Aliased);
        assert_eq!(d.take_aliases(), vec![child.prefix]);
    }
    #[test]
    fn deterministic_samples_and_negative_completion() {
        let ip = "2001:db8::123".parse().unwrap();
        let mut a = detector();
        let mut b = detector();
        for _ in 0..2 {
            assert!(matches!(a.consider(ip), Status::Pending(_)));
            b.consider(ip);
            let t = a.next_test().unwrap();
            let copy = b.next_test().unwrap();
            assert_eq!(t.targets, copy.targets);
            assert!(t.targets.iter().all(|ip| t.prefix.contains(ip)));
            assert_ne!(t.targets[0], t.targets[1]);
            assert_ne!(t.targets[1], t.targets[2]);
            // A discovery reply token must not count for a different prefix test.
            a.reply(t.targets[0], t.token + 1);
            a.finish(t.prefix);
            b.finish(copy.prefix);
        }
        assert_eq!(a.consider(ip), Status::NotDetected);
        assert!(a.aliases.is_empty());
    }
}
