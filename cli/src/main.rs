use clap::Parser;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::ExitCode;
use tracing_appender::non_blocking;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use tracing::info;

mod commands;
mod data;
mod frontends;
mod loaders;
mod sink;

use frontends::cli::Cli;

fn setup_logging(
    log_path: Option<&PathBuf>,
) -> Result<Option<tracing_appender::non_blocking::WorkerGuard>, String> {
    let make_stderr_layer = || {
        fmt::layer()
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_span_events(fmt::format::FmtSpan::NONE)
            .with_timer(fmt::time::LocalTime::new(
                time::macros::format_description!("[hour]:[minute]:[second]"),
            ))
    };
    let filter_layer = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Add IndicatifLayer for progress bars
    let indicatif_layer = IndicatifLayer::new();

    let registry = tracing_subscriber::registry()
        .with(filter_layer)
        .with(indicatif_layer);

    if let Some(path) = log_path {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("failed to create log file {}: {e}", path.display()))?;
        let (file_writer, guard) = non_blocking(file);

        let file_layer = fmt::layer()
            .with_target(false)
            .with_ansi(false)
            .with_span_events(fmt::format::FmtSpan::NONE)
            .with_timer(fmt::time::LocalTime::new(
                time::macros::format_description!("[hour]:[minute]:[second]"),
            ))
            .with_writer(file_writer);

        registry.with(make_stderr_layer()).with(file_layer).init();

        info!("Logging to file: {:?}", path);
        Ok(Some(guard))
    } else {
        registry.with(make_stderr_layer()).init();
        Ok(None)
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let _file_guard = match setup_logging(cli.log.as_ref()) {
        Ok(guard) => guard,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let mut registry = tga::builtin_registry();
    if let Some(path) = &cli.plugins
        && let Err(error) = sixseven_plugins::load_config(&mut registry, path)
    {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    match cli.command.run(&registry) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            tracing::error!("{message}");
            ExitCode::FAILURE
        }
    }
}
