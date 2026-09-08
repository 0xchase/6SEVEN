use super::target_batch::TARGET_BATCH_SIZE;
use std::time::{Duration, Instant};

pub(crate) struct Rate {
    pps: usize,
    next: Instant,
}
impl Rate {
    pub(crate) fn new(pps: usize) -> Self {
        Self {
            pps,
            next: Instant::now(),
        }
    }
    pub(crate) fn burst_size(&self) -> usize {
        if self.pps == 0 {
            TARGET_BATCH_SIZE
        } else {
            self.pps.div_ceil(1000).clamp(1, TARGET_BATCH_SIZE)
        }
    }
    pub(crate) fn reserve(&mut self, count: usize, now: Instant) -> Instant {
        if self.pps == 0 {
            return now;
        }
        let credit_floor = now.checked_sub(Duration::from_millis(2)).unwrap_or(now);
        let deadline = self.next.max(credit_floor);
        self.next = deadline + Duration::from_secs_f64(count as f64 / self.pps as f64);
        deadline
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservations_charge_the_whole_burst() {
        let mut rate = Rate::new(1_000_000);
        let now = rate.next;
        assert_eq!(rate.reserve(1000, now), now);
        assert_eq!(rate.reserve(1000, now), now + Duration::from_millis(1));
    }
    #[test]
    fn idle_time_does_not_accumulate_unbounded_credit() {
        let mut rate = Rate::new(10_000_000);
        let now = rate.next + Duration::from_secs(60);
        assert_eq!(rate.reserve(1024, now), now - Duration::from_millis(2));
    }
    #[test]
    fn low_rates_limit_burst_size() {
        assert_eq!(Rate::new(10).burst_size(), 1);
        assert_eq!(Rate::new(10_000_000).burst_size(), TARGET_BATCH_SIZE);
        assert_eq!(Rate::new(0).burst_size(), TARGET_BATCH_SIZE);
    }
}
