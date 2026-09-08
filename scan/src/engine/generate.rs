use super::target_batch::TargetBatch;
use crate::finite::{Counters, Error};
use tokio_util::sync::CancellationToken;

pub(crate) async fn stream_targets(
    source: sixseven_core::source::TargetSource,
    target_count: Option<usize>,
    target_tx: flume::Sender<TargetBatch>,
    cancel: CancellationToken,
    stats: Counters,
) -> Result<(), Error> {
    let guard = cancel.clone().drop_guard();
    let (done, result) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("scan-source".into())
        .spawn(move || {
            let _ = done.send(produce_targets(
                source,
                target_count,
                target_tx,
                cancel,
                stats,
            ));
        })
        .map_err(|error| Error::InternalError(format!("start target source: {error}")))?;
    let result = result
        .await
        .map_err(|error| Error::InternalError(format!("target source failed: {error}")))?;
    guard.disarm();
    result
}

fn produce_targets(
    mut source: sixseven_core::source::TargetSource,
    target_count: Option<usize>,
    target_tx: flume::Sender<TargetBatch>,
    cancel: CancellationToken,
    stats: Counters,
) -> Result<(), Error> {
    let required_count = target_count;
    let target_count = target_count.unwrap_or(usize::MAX);
    let mut generated = 0usize;
    let mut sink_closed = false;

    while generated < target_count && !cancel.is_cancelled() {
        let mut batch = TargetBatch::new();
        while generated.saturating_add(batch.len()) < target_count
            && !batch.is_full()
            && !cancel.is_cancelled()
        {
            let Some(target) = source.next() else {
                break;
            };
            let target = target.map_err(|err| Error::InvalidConfiguration(err.to_string()))?;
            if !batch.push(target) {
                break;
            }
        }

        if batch.is_empty() {
            break;
        }

        let batch_len = batch.len();
        loop {
            if cancel.is_cancelled() {
                return Ok(());
            }
            match target_tx.send_timeout(batch, std::time::Duration::from_millis(50)) {
                Ok(()) => break,
                Err(flume::SendTimeoutError::Timeout(returned)) => batch = returned,
                Err(flume::SendTimeoutError::Disconnected(_)) => {
                    sink_closed = true;
                    break;
                }
            }
        }
        if sink_closed {
            break;
        }
        generated = generated.saturating_add(batch_len);
        stats.add_generated(batch_len as u64);
    }

    if required_count.is_some()
        && !cancel.is_cancelled()
        && !sink_closed
        && generated < target_count
    {
        return Err(Error::InvalidConfiguration(format!(
            "target stream exhausted after generating {generated} of planned {target_count} targets"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn source_errors_are_reported() {
        let source = Box::new(std::iter::once(Err(sixseven_core::TgaError::Generation(
            "broken source".into(),
        ))));
        let (tx, _rx) = flume::bounded(1);
        let result = stream_targets(
            source,
            None,
            tx,
            CancellationToken::new(),
            Counters::default(),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("broken source"));
    }

    #[tokio::test]
    async fn cancellation_stops_a_backpressured_source() {
        let source =
            sixseven_core::source::addresses(std::iter::repeat(std::net::Ipv6Addr::LOCALHOST));
        let (tx, rx) = flume::bounded(1);
        let cancel = CancellationToken::new();
        let task = tokio::spawn(stream_targets(
            source,
            None,
            tx,
            cancel.clone(),
            Counters::default(),
        ));
        rx.recv_async().await.unwrap();
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
