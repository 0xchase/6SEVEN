use super::dealias::AliasArgs;
use super::runtime_args::RuntimeArgs;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(test)]
use probe::ReplyKind;
use probe::config::ProbeConfig;
use probe::probe::dns::{DnsConfig, DnsQueryType};
use probe::probe::icmp::IcmpConfig;
use probe::probe::ntp::NtpConfig;
use probe::probe::tcp::TcpConfig;
use probe::probe::udp::UdpConfig;
use scan::finite::{
    self, Config as RuntimeConfig, Error as RuntimeError, Event as RuntimeEvent,
    Outcome as RunOutcome,
};
use scan::reply_record as scan_record_from_result;
use serde::{Deserialize, Serialize};
use sixseven_formats::csv::scan::ScanRecord;
#[cfg(test)]
use std::net::Ipv6Addr;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_UDP_PAYLOAD_BASE64: &str = "R0VUIC8gSFRUUC8xLjENCkhvc3Q6IHd3dw0KDQo=";

#[derive(Parser, Serialize, Deserialize)]
pub struct ScanCommand {
    /// Path to a fixed target list or target CSV
    #[arg(value_name = "TARGETS")]
    pub targets: Option<PathBuf>,
    #[arg(long, conflicts_with = "targets", requires = "count")]
    pub model: Option<PathBuf>,
    #[arg(long, requires = "model")]
    pub count: Option<usize>,
    #[arg(long, requires = "model")]
    pub output_model: Option<PathBuf>,
    #[arg(long, requires = "model")]
    pub feedback_file: Option<PathBuf>,
    #[arg(long, default_value = "256")]
    pub batch_size: std::num::NonZeroUsize,
    /// Maximum outstanding targets during adaptive scanning.
    #[arg(long, default_value = "262144")]
    pub max_in_flight: std::num::NonZeroUsize,
    #[arg(long, default_value_t = 100_000_000)]
    pub max_attempts: u64,
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    #[serde(default)]
    pub aliases: AliasArgs,
    /// Detect aliased prefixes using additional ICMPv6 probes
    #[arg(long)]
    #[serde(default)]
    pub dealias: bool,
    #[command(subcommand)]
    pub mode: Option<ScanModeArgs>,
    #[arg(short = 'o', long = "output-file", value_name = "PATH")]
    pub output_file: Option<PathBuf>,
    #[arg(long = "columns", value_name = "COLUMNS", value_delimiter = ',')]
    pub columns: Option<Vec<String>>,
}

#[derive(Clone, Subcommand, Serialize, Deserialize)]
pub enum ScanModeArgs {
    Icmp(IcmpArgs),
    Tcp(TcpArgs),
    Udp(UdpArgs),
    Dns(DnsArgs),
    Ntp(NtpArgs),
}

impl Default for ScanModeArgs {
    fn default() -> Self {
        Self::Icmp(IcmpArgs)
    }
}

impl ScanModeArgs {
    fn probe_config(&self) -> Result<ProbeConfig, String> {
        match self {
            Self::Icmp(cfg) => Ok(ProbeConfig::Icmp(cfg.config())),
            Self::Tcp(cfg) => Ok(ProbeConfig::Tcp(cfg.config())),
            Self::Udp(cfg) => cfg.config().map(ProbeConfig::Udp),
            Self::Dns(cfg) => Ok(ProbeConfig::Dns(cfg.config())),
            Self::Ntp(cfg) => Ok(ProbeConfig::Ntp(cfg.config())),
        }
    }
}

#[derive(Clone, Args, Default, Serialize, Deserialize)]
pub struct IcmpArgs;

impl IcmpArgs {
    fn config(&self) -> IcmpConfig {
        IcmpConfig
    }
}

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct TcpArgs {
    #[arg(long, default_value = "80")]
    pub port: u16,
}

impl TcpArgs {
    fn config(&self) -> TcpConfig {
        TcpConfig { port: self.port }
    }
}

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct UdpArgs {
    #[arg(long, default_value = "53")]
    pub port: u16,
    #[arg(long, default_value = DEFAULT_UDP_PAYLOAD_BASE64)]
    pub payload: String,
}

impl UdpArgs {
    fn config(&self) -> Result<UdpConfig, String> {
        let payload = BASE64
            .decode(&self.payload)
            .map_err(|e| format!("invalid UDP payload base64: {e}"))?;
        Ok(UdpConfig {
            port: self.port,
            payload,
        })
    }
}

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct DnsArgs {
    #[arg(long, default_value = "53")]
    pub port: u16,
    #[arg(long, default_value = "www.google.com")]
    pub domain: String,
    #[arg(long, default_value = "a")]
    pub query_type: DnsQueryTypeArg,
}

impl DnsArgs {
    fn config(&self) -> DnsConfig {
        DnsConfig {
            port: self.port,
            domain: self.domain.clone(),
            query_type: self.query_type.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DnsQueryTypeArg {
    A,
    Aaaa,
    Mx,
    Txt,
    Ns,
    Soa,
}

impl From<DnsQueryTypeArg> for DnsQueryType {
    fn from(value: DnsQueryTypeArg) -> Self {
        match value {
            DnsQueryTypeArg::A => DnsQueryType::A,
            DnsQueryTypeArg::Aaaa => DnsQueryType::Aaaa,
            DnsQueryTypeArg::Mx => DnsQueryType::Mx,
            DnsQueryTypeArg::Txt => DnsQueryType::Txt,
            DnsQueryTypeArg::Ns => DnsQueryType::Ns,
            DnsQueryTypeArg::Soa => DnsQueryType::Soa,
        }
    }
}

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct NtpArgs {
    #[arg(long, default_value = "123")]
    pub port: u16,
}

impl NtpArgs {
    fn config(&self) -> NtpConfig {
        NtpConfig { port: self.port }
    }
}

impl ScanCommand {
    async fn run_adaptive(
        &self,
        path: &std::path::Path,
        registry: &sixseven_core::Registry,
        dealias: scan::dealias::Config,
    ) -> Result<(), RuntimeError> {
        if self.feedback_file.as_ref().is_some_and(|path| {
            path.extension()
                .is_none_or(|extension| extension != "jsonl")
        }) {
            return Err(RuntimeError::InvalidConfiguration(
                "feedback journal must use a .jsonl extension".into(),
            ));
        }
        let model_path = path.to_path_buf();
        let loader = registry.clone();
        let (mut artifact, model) = tokio::task::spawn_blocking(move || {
            let artifact = sixseven_formats::model::load(&model_path)
                .map_err(|error| RuntimeError::InvalidConfiguration(error.to_string()))?;
            let model = loader
                .open(&artifact)
                .map_err(|error| RuntimeError::InvalidConfiguration(error.to_string()))?;
            Ok::<_, RuntimeError>((artifact, model))
        })
        .await
        .map_err(|error| RuntimeError::InternalError(error.to_string()))??;
        let interface =
            link::open_interface(&self.runtime.open_config()?).map_err(map_link_error)?;
        let probe = self
            .mode
            .clone()
            .unwrap_or_default()
            .probe_config()
            .map_err(RuntimeError::InvalidConfiguration)?
            .prepare_for(interface.source_addr())
            .map_err(|e| RuntimeError::InvalidConfiguration(e.to_string()))?;
        let columns = self.columns.clone().unwrap_or_else(default_scan_columns);
        let (output, writer) = start_writer(
            self.output_file.clone(),
            columns,
            self.feedback_file.clone(),
        )?;
        let config = scan::adaptive::Config {
            interface,
            probe,
            pps: self.runtime.pps,
            network_workers: self.runtime.network_workers,
            receive_window: Duration::from_secs(self.runtime.timeout as u64),
            batch_size: self.batch_size,
            max_in_flight: self.max_in_flight,
            dealias,
            generation: sixseven_core::GenerationOptions {
                count: self.count.unwrap_or(0),
                unique: true,
                exclude: Vec::new(),
                max_attempts: self.max_attempts,
            },
        };
        let scanned = scan::adaptive::run(model, config, output).await;
        let written = writer
            .await
            .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
        written.map_err(RuntimeError::InternalError)?;
        let result = scanned.map_err(|e| RuntimeError::InternalError(e.to_string()))?;
        self.aliases
            .write_report(&result.dealias)
            .map_err(RuntimeError::InternalError)?;
        if let Some(path) = self.output_model.clone() {
            let registry = registry.clone();
            tokio::task::spawn_blocking(move || {
                registry
                    .save_model(&mut artifact, result.model.as_ref())
                    .map_err(|error| RuntimeError::InternalError(error.to_string()))?;
                sixseven_formats::model::save(&path, &artifact)
                    .map_err(|error| RuntimeError::InternalError(error.to_string()))
            })
            .await
            .map_err(|error| RuntimeError::InternalError(error.to_string()))??;
        }
        Ok(())
    }

    pub async fn execute(&self, registry: &sixseven_core::Registry) -> Result<(), RuntimeError> {
        if self.dealias && self.model.is_some() && self.max_in_flight.get() < 3 {
            return Err(RuntimeError::InvalidConfiguration(
                "active dealiasing requires --max-in-flight of at least 3".into(),
            ));
        }
        let dealias = self
            .aliases
            .config(self.dealias)
            .map_err(RuntimeError::InvalidConfiguration)?;
        self.aliases
            .validate_output_paths(&[
                self.targets.as_deref(),
                self.model.as_deref(),
                self.output_file.as_deref(),
                self.output_model.as_deref(),
                self.feedback_file.as_deref(),
            ])
            .map_err(RuntimeError::InvalidConfiguration)?;
        if let Some(path) = &self.model {
            self.run_adaptive(path, registry, dealias).await
        } else {
            self.run_standard(dealias).await
        }
    }

    async fn run_standard(&self, dealias: scan::dealias::Config) -> Result<(), RuntimeError> {
        let path = self.targets.as_ref().ok_or_else(|| {
            RuntimeError::InvalidConfiguration("provide targets or --model".into())
        })?;
        let targets = sixseven_formats::targets::read(path, None)
            .map_err(|e| RuntimeError::InvalidConfiguration(e.to_string()))?;
        let interface =
            link::open_interface(&self.runtime.open_config()?).map_err(map_link_error)?;
        let probe = self
            .mode
            .clone()
            .unwrap_or_default()
            .probe_config()
            .map_err(RuntimeError::InvalidConfiguration)?
            .prepare_for(interface.source_addr())
            .map_err(|err| RuntimeError::InvalidConfiguration(err.to_string()))?;
        let columns = self
            .columns
            .clone()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(default_scan_columns);
        let (output, writer) = start_writer(self.output_file.clone(), columns, None)?;
        let config = self.runtime.apply(
            RuntimeConfig::from_source(
                Box::new(targets.map(|item| {
                    item.map_err(|e| sixseven_core::TgaError::Generation(e.to_string()))
                })),
                interface,
                probe,
            )
            .dealias(dealias),
        );
        let mut run = finite::run(config).await?;
        let streamed = async {
            let mut records = Vec::with_capacity(1024);
            let mut tick = tokio::time::interval(Duration::from_millis(100));
            loop {
                let timed_flush = tokio::select! {
                    event = run.next() => match event {
                        Some(RuntimeEvent::Reply(reply)) => { records.push(scan_record_from_result(&reply)); false },
                        Some(RuntimeEvent::Stats(_)) => continue,
                        None => break,
                    },
                    _ = tick.tick(), if !records.is_empty() => true,
                };
                if records.len() >= 1024 || timed_flush {
                    output.send(scan::adaptive::Event::Records(std::mem::take(&mut records)))
                        .await.map_err(|error| RuntimeError::InternalError(error.to_string()))?;
                }
            }
            if !records.is_empty() {
                output.send(scan::adaptive::Event::Records(records)).await
                    .map_err(|error| RuntimeError::InternalError(error.to_string()))?;
            }
            Ok::<_, RuntimeError>(())
        }.await;
        if streamed.is_err() {
            run.cancel();
        }
        drop(output);
        writer
            .await
            .map_err(|error| RuntimeError::InternalError(error.to_string()))?
            .map_err(RuntimeError::InternalError)?;
        streamed?;
        match run.join().await? {
            RunOutcome::Completed(summary) => {
                self.aliases
                    .write_report(&summary.dealias)
                    .map_err(RuntimeError::InternalError)?;
            }
            RunOutcome::Cancelled(_) => {
                return Err(RuntimeError::InternalError("scan cancelled".into()));
            }
            RunOutcome::Failed(error) => return Err(error),
        }

        Ok(())
    }
}

type WriterResult = tokio::sync::oneshot::Receiver<Result<(), String>>;

fn start_writer(
    path: Option<PathBuf>,
    columns: Vec<String>,
    journal: Option<PathBuf>,
) -> Result<
    (
        tokio::sync::mpsc::Sender<scan::adaptive::Event>,
        WriterResult,
    ),
    RuntimeError,
> {
    let (events, mut input) = tokio::sync::mpsc::channel(64);
    let (done, result) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("scan-output".into())
        .spawn(move || {
            let result = (|| {
                let mut output = ScanOutput::new(path.as_ref(), &columns)?;
                let mut journal = journal
                    .map(std::fs::File::create)
                    .transpose()
                    .map_err(|error| error.to_string())?
                    .map(std::io::BufWriter::new);
                while let Some(event) = input.blocking_recv() {
                    match event {
                        scan::adaptive::Event::Records(records) => {
                            for record in records {
                                output.write_scan_record(&columns, &record)?;
                            }
                        }
                        scan::adaptive::Event::Feedback(feedback) => {
                            if let Some(journal) = &mut journal {
                                sixseven_formats::feedback::write_batch(journal, &feedback)
                                    .map_err(|error| error.to_string())?;
                            }
                        }
                    }
                }
                if let Some(journal) = &mut journal {
                    std::io::Write::flush(journal).map_err(|error| error.to_string())?;
                }
                output.finish()
            })();
            let _ = done.send(result);
        })
        .map_err(|error| RuntimeError::InternalError(error.to_string()))?;
    Ok((events, result))
}

fn default_scan_columns() -> Vec<String> {
    ScanRecord::COLUMNS
        .iter()
        .map(|column| (*column).to_string())
        .collect()
}

fn map_link_error(err: link::Error) -> RuntimeError {
    match err {
        link::Error::InvalidInterface(msg) | link::Error::InvalidConfiguration(msg) => {
            RuntimeError::InvalidConfiguration(msg)
        }
        link::Error::BackendUnavailable(msg) => RuntimeError::NetworkError(msg),
        link::Error::Io(err) => RuntimeError::NetworkError(err.to_string()),
    }
}

struct ScanOutput {
    writer: csv::Writer<Box<dyn std::io::Write + Send>>,
}

impl ScanOutput {
    fn new(path: Option<&PathBuf>, columns: &[String]) -> Result<Self, String> {
        let sink: Box<dyn std::io::Write + Send> = match path {
            Some(path) => Box::new(
                std::fs::File::create(path)
                    .map_err(|e| format!("failed to create output file {}: {e}", path.display()))?,
            ),
            None => Box::new(std::io::stdout()),
        };
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(sink);
        writer
            .write_record(columns)
            .map_err(|e| format!("failed to write csv headers: {e}"))?;
        Ok(Self { writer })
    }

    fn write_scan_record(&mut self, columns: &[String], record: &ScanRecord) -> Result<(), String> {
        self.write_record(record.csv_row(columns))
    }

    fn write_record(&mut self, row: Vec<String>) -> Result<(), String> {
        self.writer
            .write_record(&row)
            .map_err(|e| format!("failed to write csv row: {e}"))?;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|e| format!("failed to flush csv writer: {e}"))?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writer_reports_file_creation_failures() {
        let directory = tempfile::tempdir().unwrap();
        let (output, result) = start_writer(
            Some(directory.path().to_path_buf()),
            default_scan_columns(),
            None,
        )
        .unwrap();
        drop(output);
        assert!(result.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn writer_flushes_records_and_feedback_before_completion() {
        let directory = tempfile::tempdir().unwrap();
        let csv = directory.path().join("results.csv");
        let journal = directory.path().join("feedback.jsonl");
        let (output, result) = start_writer(
            Some(csv.clone()),
            default_scan_columns(),
            Some(journal.clone()),
        )
        .unwrap();
        let reply = probe::Reply {
            target: Ipv6Addr::LOCALHOST,
            responder: Ipv6Addr::LOCALHOST,
            token: 0,
            kind: ReplyKind::Direct,
        };
        output
            .send(scan::adaptive::Event::Records(vec![
                scan_record_from_result(&reply),
            ]))
            .await
            .unwrap();
        output
            .send(scan::adaptive::Event::Feedback(vec![
                sixseven_core::Feedback::Active(Ipv6Addr::LOCALHOST),
            ]))
            .await
            .unwrap();
        drop(output);
        result.await.unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(csv).unwrap().lines().count(), 2);
        let feedback: Vec<sixseven_core::Feedback> =
            serde_json::from_str(&std::fs::read_to_string(journal).unwrap()).unwrap();
        assert_eq!(
            feedback,
            vec![sixseven_core::Feedback::Active(Ipv6Addr::LOCALHOST)]
        );
    }
}
