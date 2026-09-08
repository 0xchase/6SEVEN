use std::hint::black_box;
use std::net::Ipv6Addr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use link::{Backend, OpenConfig};
use probe::config::ProbeConfig;
use probe::probe::icmp::IcmpConfig;
use repeated_targets::RepeatedTargets;
use scan::finite::{
    self, Config as RuntimeConfig, Error as RuntimeError, Event as RuntimeEvent,
    Outcome as RunOutcome, Stats as ScanStats, Summary as RunSummary,
};
use std::io::BufRead;
use tga::{AlgorithmConfig, GenerationOptions, ModelArtifact, Observation};

mod repeated_targets;

const DEFAULT_PROBES: usize = 10_000_000;
const DEFAULT_SOURCE_WORKERS: usize = 8;
const DEFAULT_GENERATION_COUNT: usize = 5_000_000;
const DEFAULT_MIN_BENCH_SECONDS: f64 = 0.5;
const DEFAULT_MAX_PASSES: usize = 256;

#[derive(Parser)]
#[command(name = "6seven-profile")]
#[command(about = "Profiling tools for 6SEVEN packet I/O and target generation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the shared scan runtime against repeated targets.
    Loopback(LoopbackArgs),
    /// Benchmark TGA training and in-process generation.
    Tga(TgaArgs),
    /// Benchmark ICMP encoding in memory without opening sockets.
    Encode {
        #[arg(long, default_value = "10000000")]
        count: std::num::NonZeroUsize,
    },
}

#[derive(Args)]
struct LoopbackArgs {
    /// Loopback interface to use.
    #[arg(long, default_value = "lo")]
    interface: String,
    /// IPv6 source address assigned to the loopback interface.
    #[arg(long, default_value = "::1")]
    source: Ipv6Addr,
    /// IPv6 target address. Defaults to localhost.
    #[arg(long, default_value = "::1")]
    target: Ipv6Addr,
    /// Number of probes to issue.
    #[arg(short = 'n', long, default_value_t = DEFAULT_PROBES)]
    probes: usize,
    /// Scan rate limit. Use 0 for unpaced sending.
    #[arg(long, default_value_t = 0)]
    pps: usize,
    /// Target-source shards to use.
    #[arg(
        long = "network-workers",
        alias = "target-workers",
        alias = "shards",
        default_value_t = DEFAULT_SOURCE_WORKERS
    )]
    network_workers: usize,
    /// Packet transmit backend to benchmark.
    #[arg(long, default_value_t = Backend::AfPacket)]
    backend: Backend,
    /// Seconds to keep the receiver alive after sending finishes.
    #[arg(long, default_value_t = 2)]
    timeout: usize,
    /// Allow running on a non-loopback interface.
    #[arg(long)]
    allow_non_loopback: bool,
    /// Disable the live progress bar.
    #[arg(long)]
    no_progress: bool,
    /// Progress refresh interval in milliseconds.
    #[arg(long, default_value_t = 500)]
    progress_interval_ms: u64,
}

#[derive(Args)]
struct TgaArgs {
    /// Path to file containing seed addresses (one per line).
    #[arg(short, long)]
    seeds: PathBuf,
    /// Number of targets to request for the first generation timing pass.
    #[arg(short = 'n', long, default_value_t = DEFAULT_GENERATION_COUNT)]
    count: usize,
    /// Minimum wall-clock duration to accept for a generation timing pass.
    #[arg(long, default_value_t = DEFAULT_MIN_BENCH_SECONDS)]
    min_bench_seconds: f64,
    /// Maximum number of repeated generation passes to aggregate.
    #[arg(long, default_value_t = DEFAULT_MAX_PASSES)]
    max_passes: usize,
    /// Ensure generated addresses are unique.
    #[arg(short = 'u', long)]
    unique: bool,
    /// TGA algorithm to use for training.
    #[command(subcommand)]
    algorithm: AlgorithmConfig,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Loopback(args) => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(run_loopback(args)),
        Command::Tga(args) => run_tga(args),
        Command::Encode { count } => benchmark_encoding(count.get()),
    }
}

fn benchmark_encoding(count: usize) -> Result<()> {
    let source: Ipv6Addr = "2001:db8::1".parse()?;
    let probe = ProbeConfig::default().prepare_for(source)?;
    let mut buffer = vec![0; probe.max_packet_len()];
    let started = Instant::now();
    for index in 0..count {
        let mut target = source.octets();
        target[8..].copy_from_slice(&(index as u64).to_be_bytes());
        let packet = probe
            .encode(
                &mut buffer,
                probe::Request {
                    target: target.into(),
                    token: 0,
                },
            )
            .ok_or_else(|| anyhow!("probe encoding failed"))?;
        black_box(&buffer[..packet.payload_len]);
    }
    let elapsed = started.elapsed();
    println!(
        "ICMP encodes: {count}, elapsed: {elapsed:?}, encodes/s: {:.0}",
        count as f64 / elapsed.as_secs_f64()
    );
    println!("In-memory encoding only. No sockets opened or probes transmitted.");
    Ok(())
}

async fn run_loopback(args: LoopbackArgs) -> Result<()> {
    let interface = open_loopback_interface(&args)?;
    let probe = ProbeConfig::Icmp(IcmpConfig)
        .prepare_for(interface.source_addr())
        .map_err(|err| anyhow!(err.to_string()))?;
    let config = scan_config(&args, interface, probe);
    let mut run = finite::run(config).await.map_err(map_scan_error)?;
    let started = Instant::now();
    let mut generation_done_at = None;
    let mut progress = ProgressReporter::new(&args);

    loop {
        match run.next().await {
            Some(RuntimeEvent::Reply(_)) => {}
            Some(RuntimeEvent::Stats(stats)) => {
                progress.update(stats, started, &mut generation_done_at, args.probes);
            }
            None => break,
        }
    }
    let outcome = run.join().await.map_err(map_scan_error)?;
    progress.finish();
    let summary = match outcome {
        RunOutcome::Completed(summary) => summary,
        RunOutcome::Cancelled(_) => return Err(anyhow!("scan cancelled")),
        RunOutcome::Failed(error) => return Err(map_scan_error(error)),
    };
    let elapsed = started.elapsed();
    let generation_elapsed = generation_done_at
        .map(|done| done.duration_since(started))
        .unwrap_or(elapsed);
    print_report(&args, summary, elapsed, generation_elapsed);
    Ok(())
}

fn run_tga(args: TgaArgs) -> Result<()> {
    let observations = load_seed_observations(&args.seeds)?;
    let seed_count = observations.len();

    let registry = tga::builtin_registry();
    let started = Instant::now();
    let mut artifact = registry.train(&args.algorithm.spec()?, &observations)?;
    artifact
        .metadata
        .insert("seed_count".into(), seed_count.to_string());
    artifact
        .metadata
        .insert("seed_path".into(), args.seeds.display().to_string());
    let train_elapsed = started.elapsed();

    let generation = benchmark_generation(
        &artifact,
        args.count,
        args.unique,
        args.min_bench_seconds,
        args.max_passes,
    )?;

    print_tga_report(&args, &artifact, train_elapsed, generation);
    Ok(())
}

fn scan_config(
    args: &LoopbackArgs,
    interface: link::Interface,
    probe: probe::PreparedProbe,
) -> RuntimeConfig {
    RuntimeConfig::new(
        Box::new(RepeatedTargets::new(args.target, args.probes)),
        args.probes,
        interface,
        probe,
    )
    .pps(args.pps)
    .network_workers(args.network_workers)
    .receive_window(Duration::from_secs(args.timeout as u64))
    .stats_interval(Some(Duration::from_millis(
        args.progress_interval_ms.max(50),
    )))
}

fn open_loopback_interface(args: &LoopbackArgs) -> Result<link::Interface> {
    let link_cfg = OpenConfig::new()
        .backend(args.backend)
        .interface(args.interface.clone())
        .source_addr(args.source);
    let iface = link::open_interface(&link_cfg).with_context(|| {
        format!(
            "open interface {}; raw packet access may require root or CAP_NET_RAW",
            args.interface
        )
    })?;
    if !args.allow_non_loopback && !iface.is_loopback() {
        bail!(
            "resolved interface '{}' is not loopback; use --allow-non-loopback to override",
            iface.name()
        );
    }
    Ok(iface)
}

fn map_scan_error(err: RuntimeError) -> anyhow::Error {
    anyhow!(err.to_string())
}

struct ProgressReporter {
    bar: Option<ProgressBar>,
    last: ScanStats,
    last_at: Instant,
}

impl ProgressReporter {
    fn new(args: &LoopbackArgs) -> Self {
        let bar = if args.no_progress {
            None
        } else {
            let bar = ProgressBar::new(args.probes as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("=>-"),
            );
            Some(bar)
        };

        Self {
            bar,
            last: ScanStats::default(),
            last_at: Instant::now(),
        }
    }

    fn update(
        &mut self,
        current: ScanStats,
        started: Instant,
        generation_done_at: &mut Option<Instant>,
        total: usize,
    ) {
        let now = Instant::now();
        let total = total as u64;
        if generation_done_at.is_none() && current.generated >= total {
            *generation_done_at = Some(now);
        }

        let Some(bar) = &self.bar else {
            self.last = current;
            self.last_at = now;
            return;
        };

        let elapsed = now.duration_since(self.last_at);
        bar.set_position(current.sent.min(total));
        bar.set_message(progress_message(self.last, current, elapsed, started, now));
        self.last = current;
        self.last_at = now;
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

fn progress_message(
    last: ScanStats,
    current: ScanStats,
    elapsed: Duration,
    started: Instant,
    now: Instant,
) -> String {
    format!(
        "gen {:>8.0}pps tx {:>8.0}pps rx {:>8.0}pps total-tx {} errors {} avg-tx {:>8.0}pps",
        rate(current.generated.saturating_sub(last.generated), elapsed),
        rate(current.sent.saturating_sub(last.sent), elapsed),
        rate(current.replies.saturating_sub(last.replies), elapsed),
        current.sent,
        current.errors,
        rate(current.sent, now.duration_since(started)),
    )
}

fn print_report(
    args: &LoopbackArgs,
    summary: RunSummary,
    elapsed: Duration,
    generation_elapsed: Duration,
) {
    println!("loopback scan profile");
    println!("  runtime path:      scan::finite::run");
    println!("  target:            {}", args.target);
    println!("  probes requested:  {}", args.probes);
    println!("  targets generated: {}", summary.stats.generated);
    println!("  frames sent:       {}", summary.stats.sent);
    println!("  decoded replies:   {}", summary.stats.replies);
    println!("  errors:            {}", summary.stats.errors);
    println!("  source workers:    {}", summary.plan.network_workers);
    println!("  tx workers:        {}", summary.plan.sender_workers);
    println!("  rx workers:        {}", summary.plan.receiver_workers);
    println!("  backend:           {}", args.backend);
    println!("  pps limit:         {}", args.pps);
    println!(
        "  target gen elapsed:{:>7.3}s",
        generation_elapsed.as_secs_f64()
    );
    println!(
        "  target gen pps:    {:.0}",
        rate(summary.stats.generated, generation_elapsed)
    );
    println!("  total elapsed:     {:.3}s", elapsed.as_secs_f64());
    println!(
        "  send pps:          {:.0}",
        rate(summary.stats.sent, elapsed)
    );
    println!(
        "  reply pps:         {:.0}",
        rate(summary.stats.replies, elapsed)
    );
    println!(
        "  decoded reply pct: {:>7.2}%",
        percentage(summary.stats.replies, summary.stats.sent)
    );
}

struct GenerationBenchmark {
    requested_per_pass: usize,
    passes: usize,
    elapsed: Duration,
    stats: tga::GenerationStats,
}

fn benchmark_generation(
    artifact: &ModelArtifact,
    requested: usize,
    unique: bool,
    min_bench_seconds: f64,
    max_passes: usize,
) -> Result<GenerationBenchmark> {
    let max_passes = max_passes.max(1);
    let mut passes = 0usize;
    let mut elapsed = Duration::ZERO;
    let mut total = tga::GenerationStats::default();

    while passes < max_passes {
        let started = Instant::now();
        let mut model = tga::builtin_registry().open(artifact)?;
        let pass = sixseven_core::generation::generate(
            model.as_mut(),
            GenerationOptions {
                max_attempts: 100_000_000,
                count: requested,
                unique,
                exclude: Vec::new(),
            },
            |addr| {
                black_box(addr);
                Ok(())
            },
        )?;
        elapsed += started.elapsed();
        passes += 1;
        total.attempts = total.attempts.saturating_add(pass.attempts);
        total.written = total.written.saturating_add(pass.written);
        total.unique = pass
            .unique
            .map(|value| total.unique.unwrap_or(0).saturating_add(value));
        total.duplicates = total.duplicates.saturating_add(pass.duplicates);
        total.excluded = total.excluded.saturating_add(pass.excluded);

        if elapsed.as_secs_f64() >= min_bench_seconds || pass.written == 0 {
            break;
        }
    }

    Ok(GenerationBenchmark {
        requested_per_pass: requested,
        passes,
        elapsed,
        stats: total,
    })
}

fn load_seed_observations(path: &std::path::Path) -> Result<Vec<Observation>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open seeds file {}", path.display()))?;
    let lines = std::io::BufReader::new(file).lines();
    let mut observations = Vec::new();

    for line in lines {
        let line = line.with_context(|| format!("read seeds file {}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let address = trimmed
            .parse::<Ipv6Addr>()
            .with_context(|| format!("parse IPv6 seed '{trimmed}' in {}", path.display()))?;
        observations.push(Observation {
            address: address.octets(),
            active: true,
        });
    }

    if observations.is_empty() {
        bail!("no valid seeds loaded from {}", path.display());
    }

    Ok(observations)
}

fn print_tga_report(
    args: &TgaArgs,
    artifact: &ModelArtifact,
    train_elapsed: Duration,
    generation: GenerationBenchmark,
) {
    println!("tga profile");
    println!("  algorithm:         {}", artifact.algorithm);
    println!("  seeds:             {}", args.seeds.display());
    println!(
        "  seeds loaded:      {}",
        artifact
            .metadata
            .get("seed_count")
            .map(String::as_str)
            .unwrap_or("?")
    );
    println!("  unique:            {}", args.unique);
    println!("  train elapsed:     {:.6}s", train_elapsed.as_secs_f64());
    println!("  benchmark passes:  {}", generation.passes);
    println!("  requested/pass:    {}", generation.requested_per_pass);
    println!("  written targets:   {}", generation.stats.written);
    println!("  attempts:          {}", generation.stats.attempts);
    println!(
        "  generation elapsed:{:>10.6}s",
        generation.elapsed.as_secs_f64()
    );
    println!(
        "  generation pps:    {:.0}",
        rate(generation.stats.written, generation.elapsed)
    );
    println!(
        "RESULT algorithm={} train_s={:.6} requested_per_pass={} passes={} written={} attempts={} gen_s={:.6} rate={:.6}",
        artifact.algorithm,
        train_elapsed.as_secs_f64(),
        generation.requested_per_pass,
        generation.passes,
        generation.stats.written,
        generation.stats.attempts,
        generation.elapsed.as_secs_f64(),
        rate(generation.stats.written, generation.elapsed),
    );
}

fn rate(count: u64, elapsed: Duration) -> f64 {
    count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
}

fn percentage(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64 / total as f64) * 100.0
    }
}
