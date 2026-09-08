use super::{MAX_MESSAGE_BYTES, Transport, plugin_error};
use serde_json::Value;
use sixseven_core::TgaError;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::Duration,
};

pub(super) struct Process {
    child: Child,
    requests: mpsc::SyncSender<Vec<u8>>,
    responses: mpsc::Receiver<Result<Value, String>>,
    timeout: Duration,
}

impl Process {
    pub(super) fn open(
        interpreter: &Path,
        script: &Path,
        timeout: Duration,
    ) -> Result<Self, TgaError> {
        let mut child = Command::new(interpreter)
            .arg("-u")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(plugin_error)?;
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| plugin_error("missing plugin stdin"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| plugin_error("missing plugin stdout"))?;
        let (tx, responses) = mpsc::sync_channel(1);
        let (requests, incoming) = mpsc::sync_channel::<Vec<u8>>(1);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            while let Ok(request) = incoming.recv() {
                let mut bytes = Vec::new();
                let result = input
                    .write_all(&request)
                    .and_then(|()| input.flush())
                    .map_err(|e| e.to_string())
                    .and_then(|()| read_message(&mut reader, &mut bytes))
                    .and_then(|()| serde_json::from_slice(&bytes).map_err(|e| e.to_string()));
                let failed = result.is_err();
                if tx.send(result).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            requests,
            responses,
            timeout,
        })
    }
}

fn read_message(reader: &mut impl BufRead, out: &mut Vec<u8>) -> Result<(), String> {
    loop {
        let bytes = reader.fill_buf().map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            return Err("plugin stdout closed".into());
        }
        let newline = bytes.iter().position(|byte| *byte == b'\n');
        let length = newline.map_or(bytes.len(), |i| i + 1);
        if out.len() + length > MAX_MESSAGE_BYTES {
            return Err("plugin message exceeds limit".into());
        }
        out.extend_from_slice(&bytes[..length]);
        reader.consume(length);
        if newline.is_some() {
            return Ok(());
        }
    }
}

impl Transport for Process {
    fn exchange(&mut self, request: &Value) -> Result<Value, TgaError> {
        let mut bytes = serde_json::to_vec(request).map_err(plugin_error)?;
        if bytes.len() >= MAX_MESSAGE_BYTES {
            return Err(plugin_error("request exceeds limit"));
        }
        bytes.push(b'\n');
        self.requests.try_send(bytes).map_err(plugin_error)?;
        match self.responses.recv_timeout(self.timeout) {
            Ok(result) => result.map_err(plugin_error),
            Err(error) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                Err(plugin_error(error))
            }
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process(script: &str) -> (tempfile::NamedTempFile, Process) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(script.as_bytes()).unwrap();
        let process = Process::open(
            Path::new("python3"),
            file.path(),
            Duration::from_millis(300),
        )
        .unwrap();
        (file, process)
    }

    #[test]
    fn timeout_covers_blocked_input() {
        let (_file, mut process) = process("import time\ntime.sleep(30)\n");
        let started = std::time::Instant::now();
        assert!(
            process
                .exchange(&Value::String("x".repeat(1024 * 1024)))
                .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(process.child.try_wait().unwrap().is_some());
    }

    #[test]
    fn timeout_covers_missing_response() {
        let (_file, mut process) =
            process("import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n");
        assert!(process.exchange(&Value::Null).is_err());
        assert!(process.child.try_wait().unwrap().is_some());
    }

    #[test]
    fn malformed_response_is_an_error() {
        let (_file, mut process) =
            process("import sys\nsys.stdin.readline()\nprint('broken', flush=True)\n");
        assert!(process.exchange(&Value::Null).is_err());
    }
}
