use clap::Parser;
use serde::{Deserialize, Serialize};
use sixseven_formats::csv::targets::TargetRecord;
use std::io::{BufWriter, Write as _};
use std::net::Ipv6Addr;
use std::path::PathBuf;
use std::str::FromStr;
use tga::{BitRange, GenerationInput, GenerationOptions};
use tracing::info;

use crate::commands::Command;

#[derive(Parser, Serialize, Deserialize)]
pub struct GenerateCommand {
    /// Load one or more trained models from file as PATH or PATH,RANGE
    #[arg(short, long)]
    pub model: Vec<ModelInputArg>,

    /// Add one or more deterministic pseudorandom components as [RANGE]
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub random: Vec<RandomInputArg>,

    /// Add one or more fixed IPv6 components as ADDR or ADDR,RANGE
    #[arg(long)]
    pub fixed: Vec<FixedInputArg>,

    /// Number of addresses to generate
    #[arg(short = 'n', long, default_value = "1000")]
    pub count: usize,
    #[arg(long, default_value_t = 100_000_000)]
    pub max_attempts: u64,

    /// Ensure generated addresses are unique
    #[arg(short = 'u', long)]
    pub unique: bool,

    /// Exclude a list of addresses from generation
    #[arg(short = 'e', long)]
    pub exclude: Option<PathBuf>,

    /// Output file to save the generated addresses (prints to stdout if not provided)
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
}

impl Command for GenerateCommand {
    fn run(&self, registry: &sixseven_core::Registry) -> Result<(), String> {
        info!(
            "Generating targets from {} model input(s), {} random component(s), and {} fixed component(s)",
            self.model.len(),
            self.random.len(),
            self.fixed.len(),
        );

        let exclude = load_exclude_list(self.exclude.as_ref())?;
        let options = GenerationOptions {
            max_attempts: self.max_attempts,
            count: self.count,
            unique: self.unique,
            exclude,
        };

        let mut writer = open_output_writer(self.output.as_ref())?;
        writer
            .write_all(format!("{}\n", TargetRecord::COLUMNS.join(",")).as_bytes())
            .map_err(|e| format!("failed to write output header: {e}"))?;

        let inputs = build_generation_inputs(&self.model, &self.random, &self.fixed)?;
        let mut model =
            sixseven_core::generation::compose(registry, inputs).map_err(|e| e.to_string())?;
        sixseven_core::generation::generate(model.as_mut(), options, |address| {
            writeln!(writer, "{address}").map_err(|e| tga::TgaError::Generation(e.to_string()))
        })
        .map_err(|e| e.to_string())?;
        writer
            .flush()
            .map_err(|e| format!("failed to flush output writer: {e}"))?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitRangeArg {
    start: u8,
    end: u8,
}

impl BitRangeArg {
    fn into_tga(self) -> Result<BitRange, String> {
        BitRange::new(self.start, self.end).map_err(|e| e.to_string())
    }
}

impl FromStr for BitRangeArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_bit_range(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInputArg {
    path: PathBuf,
    bits: Option<BitRangeArg>,
}

impl FromStr for ModelInputArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (path, bits) = split_optional_range(value)?;
        if path.trim().is_empty() {
            return Err("model path cannot be empty".into());
        }
        Ok(Self {
            path: PathBuf::from(path),
            bits,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomInputArg {
    bits: Option<BitRangeArg>,
}

impl FromStr for RandomInputArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() {
            return Ok(Self { bits: None });
        }
        Ok(Self {
            bits: Some(parse_bit_range(value)?),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedInputArg {
    address: Ipv6Addr,
    bits: Option<BitRangeArg>,
}

impl FromStr for FixedInputArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (address, bits) = split_optional_range(value)?;
        let address = address
            .trim()
            .parse::<Ipv6Addr>()
            .map_err(|e| format!("invalid fixed IPv6 address '{address}': {e}"))?;
        Ok(Self { address, bits })
    }
}

fn open_output_writer(path: Option<&PathBuf>) -> Result<Box<dyn std::io::Write>, String> {
    match path {
        Some(path) => {
            let file = std::fs::File::create(path)
                .map_err(|e| format!("failed to create output file {}: {e}", path.display()))?;
            Ok(Box::new(BufWriter::new(file)))
        }
        None => Ok(Box::new(BufWriter::new(std::io::stdout()))),
    }
}

fn load_exclude_list(path: Option<&PathBuf>) -> Result<Vec<Ipv6Addr>, String> {
    match path {
        Some(path) => sixseven_formats::targets::read(path, None)
            .and_then(|reader| reader.collect())
            .map_err(|e| e.to_string()),
        None => Ok(Vec::new()),
    }
}

fn parse_bit_range(value: &str) -> Result<BitRangeArg, String> {
    if let Some((start, end)) = value.split_once("..=") {
        let start = parse_bound(start, "start")?;
        let end = parse_bound(end, "end")?;
        return BitRange::new(start, end.saturating_add(1))
            .map(|range| BitRangeArg {
                start: range.start(),
                end: range.end(),
            })
            .map_err(|e| e.to_string());
    }

    if let Some((start, end)) = value.split_once("..") {
        let start = parse_bound(start, "start")?;
        let end = parse_bound(end, "end")?;
        return BitRange::new(start, end)
            .map(|range| BitRangeArg {
                start: range.start(),
                end: range.end(),
            })
            .map_err(|e| e.to_string());
    }

    if let Some((start, end)) = value.split_once('-') {
        let start = parse_bound(start, "start")?;
        let end = parse_bound(end, "end")?;
        return BitRange::new(start, end.saturating_add(1))
            .map(|range| BitRangeArg {
                start: range.start(),
                end: range.end(),
            })
            .map_err(|e| e.to_string());
    }

    Err(format!(
        "invalid bit range '{value}'; expected START..END, START..=END, or START-END"
    ))
}

fn parse_bound(value: &str, label: &str) -> Result<u8, String> {
    value
        .trim()
        .parse::<u8>()
        .map_err(|e| format!("invalid {label} bit '{value}': {e}"))
}

fn split_optional_range(value: &str) -> Result<(&str, Option<BitRangeArg>), String> {
    if let Some((head, tail)) = value.rsplit_once(',')
        && let Ok(bits) = parse_bit_range(tail.trim())
    {
        return Ok((head.trim(), Some(bits)));
    }
    Ok((value.trim(), None))
}

fn build_generation_inputs(
    models: &[ModelInputArg],
    random: &[RandomInputArg],
    fixed: &[FixedInputArg],
) -> Result<Vec<GenerationInput>, String> {
    if models.is_empty() && random.is_empty() && fixed.is_empty() {
        return Err("at least one --model, --random, or --fixed input is required".into());
    }

    let component_count = models.len() + random.len() + fixed.len();
    let require_ranges = component_count > 1;
    let mut inputs = Vec::with_capacity(component_count.max(1));
    for model_arg in models {
        let model = sixseven_formats::model::load(&model_arg.path).map_err(|e| e.to_string())?;
        let bits = resolve_bits(model_arg.bits, require_ranges)?;
        inputs.push(GenerationInput::Model { model, bits });
    }

    for random_arg in random {
        inputs.push(GenerationInput::Random {
            bits: resolve_bits(random_arg.bits, require_ranges)?,
        });
    }

    for fixed_arg in fixed {
        inputs.push(GenerationInput::Fixed {
            address: fixed_arg.address.octets(),
            bits: resolve_bits(fixed_arg.bits, require_ranges)?,
        });
    }

    Ok(inputs)
}

fn resolve_bits(
    bits: Option<BitRangeArg>,
    require_ranges: bool,
) -> Result<Option<BitRange>, String> {
    match (require_ranges, bits) {
        (true, Some(bits)) => Ok(Some(bits.into_tga()?)),
        (true, None) => Err(
            "every generation component must include a range when composing multiple inputs".into(),
        ),
        (false, Some(bits)) => Ok(Some(bits.into_tga()?)),
        (false, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_generation_count_matches_six_gcvae_reference_script() {
        let parsed = GenerateCommand::try_parse_from(["bin"]).unwrap();
        assert_eq!(parsed.count, 1000);
    }
}
