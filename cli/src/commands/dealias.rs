use super::runtime_args::RuntimeArgs;
use clap::{Args, Parser};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Args, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AliasArgs {
    /// Known aliased IPv6 prefixes, one CIDR per line
    #[arg(long)]
    pub aliased_prefixes: Option<PathBuf>,
    /// Write combined known and detected alias prefixes
    #[arg(long)]
    pub output_aliases: Option<PathBuf>,
    /// Prefix lengths to test (default: 48,64,96,112,116,120; at most eight)
    #[arg(long, value_delimiter = ',')]
    pub alias_prefix_lengths: Option<Vec<u8>>,
    /// Reproducible seed for alias probe addresses
    #[arg(long)]
    pub alias_seed: Option<u64>,
}
impl AliasArgs {
    pub(crate) fn config(&self, online: bool) -> Result<scan::dealias::Config, String> {
        if !online && (self.alias_prefix_lengths.is_some() || self.alias_seed.is_some()) {
            return Err(
                "alias prefix lengths and seed require active detection (--online or --dealias)"
                    .into(),
            );
        }
        let aliases = self
            .aliased_prefixes
            .as_ref()
            .map(sixseven_formats::prefixes::read)
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        let online = if online {
            let mut config = scan::dealias::OnlineConfig::default();
            if let Some(lengths) = &self.alias_prefix_lengths {
                config.prefix_lengths = lengths.clone();
            }
            if let Some(seed) = self.alias_seed {
                config.seed = seed;
            }
            config.validate().map_err(|e| e.to_string())?;
            Some(config)
        } else {
            None
        };
        Ok(scan::dealias::Config { aliases, online })
    }
    pub(crate) fn write_report(&self, report: &scan::dealias::Report) -> Result<(), String> {
        if let Some(path) = &self.output_aliases {
            sixseven_formats::prefixes::write(path, &report.aliases).map_err(|e| e.to_string())?;
        }
        tracing::info!(
            alias_probes = report.probes_sent,
            alias_prefixes = report.aliases.prefixes().count(),
            "Dealiasing complete"
        );
        Ok(())
    }
    pub(crate) fn validate_output_paths(&self, paths: &[Option<&Path>]) -> Result<(), String> {
        if let Some(output) = &self.output_aliases {
            for input in paths
                .iter()
                .copied()
                .flatten()
                .chain(self.aliased_prefixes.as_deref())
            {
                distinct(input, output)?;
            }
        }
        Ok(())
    }
}

#[derive(Parser, Serialize, Deserialize)]
pub struct DealiasCommand {
    /// Address list or scan CSV to filter
    pub input: PathBuf,
    /// Send ICMPv6 probes to discover additional aliases
    #[arg(long)]
    pub online: bool,
    #[command(flatten)]
    pub aliases: AliasArgs,
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    /// Filtered output (stdout if omitted), preserving the input format
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}
impl DealiasCommand {
    pub async fn execute(&self) -> Result<(), String> {
        if !self.online && self.aliases.aliased_prefixes.is_none() {
            return Err(
                "provide --aliased-prefixes for offline filtering or --online for active detection"
                    .into(),
            );
        }
        let config = self.aliases.config(self.online)?;
        self.aliases
            .validate_output_paths(&[Some(&self.input), self.output.as_deref()])?;
        if let Some(output) = &self.output {
            distinct(&self.input, output)?;
            if let Some(aliases) = &self.aliases.aliased_prefixes {
                distinct(aliases, output)?;
            }
        }
        let report = if self.online {
            let candidates =
                sixseven_formats::targets::candidates(&self.input).map_err(|e| e.to_string())?;
            let interface =
                link::open_interface(&self.runtime.open_config().map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            scan::dealias::run(
                Box::new(candidates.map(|item| {
                    item.map_err(|e| sixseven_core::TgaError::Generation(e.to_string()))
                })),
                interface,
                config,
                self.runtime.pps,
                self.runtime.network_workers,
                Duration::from_secs(self.runtime.timeout as u64),
            )
            .await
            .map_err(|e| e.to_string())?
        } else {
            scan::dealias::Report {
                aliases: config.aliases,
                probes_sent: 0,
            }
        };
        let input = self.input.clone();
        let output = self.output.clone();
        let prefixes = report.aliases.clone();
        let stats = tokio::task::spawn_blocking(move || {
            // A temporary file prevents malformed input from leaving a partial output.
            if let Some(path) = output {
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let mut temporary =
                    tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
                let stats =
                    sixseven_formats::targets::filter(&input, temporary.as_file_mut(), &prefixes)
                        .map_err(|e| e.to_string())?;
                temporary.persist(&path).map_err(|e| e.to_string())?;
                Ok::<_, String>(stats)
            } else {
                sixseven_formats::targets::filter(&input, std::io::stdout(), &prefixes)
                    .map_err(|e| e.to_string())
            }
        })
        .await
        .map_err(|e| e.to_string())??;
        tracing::info!(
            retained = stats.retained,
            removed = stats.removed,
            "Filtered addresses"
        );
        self.aliases.write_report(&report)
    }
}

fn distinct(input: &Path, output: &Path) -> Result<(), String> {
    fn resolved(path: &Path) -> Result<PathBuf, String> {
        if path.exists() {
            path.canonicalize().map_err(|e| e.to_string())
        } else {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            Ok(parent
                .canonicalize()
                .map_err(|e| e.to_string())?
                .join(path.file_name().ok_or("missing output file name")?))
        }
    }
    let mut same = resolved(input)? == resolved(output)?;
    #[cfg(unix)]
    if let (Ok(a), Ok(b)) = (std::fs::metadata(input), std::fs::metadata(output)) {
        use std::os::unix::fs::MetadataExt;
        same |= a.dev() == b.dev() && a.ino() == b.ino();
    }
    if same {
        Err(format!(
            "input and output paths must differ: {}",
            output.display()
        ))
    } else {
        Ok(())
    }
}
