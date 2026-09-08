use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

pub(crate) async fn run<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, tokio::task::JoinError> {
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    let slots = SLOTS.get_or_init(|| {
        Arc::new(Semaphore::new(
            std::thread::available_parallelism().map_or(1, usize::from),
        ))
    });
    run_with_slots(slots.clone(), work).await
}

async fn run_with_slots<T: Send + 'static>(
    slots: Arc<Semaphore>,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, tokio::task::JoinError> {
    let permit = slots
        .acquire_owned()
        .await
        .expect("blocking admission remains open");
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_callers_do_not_release_running_work_capacity() {
        let slots = Arc::new(Semaphore::new(1));
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let task = tokio::spawn(run_with_slots(slots.clone(), move || {
            started.send(()).unwrap();
            wait.recv().unwrap();
        }));
        ready.await.unwrap();
        task.abort();
        let _ = task.await;
        assert_eq!(slots.available_permits(), 0);
        release.send(()).unwrap();
        let permit = tokio::time::timeout(std::time::Duration::from_secs(1), slots.acquire())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
        assert_eq!(slots.available_permits(), 1);
    }
}
