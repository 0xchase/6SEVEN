use clap::Parser;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use sixseven_formats::csv::scan::ScanRecord;
use sixseven_formats::csv::targets::TargetRecord;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::net::Ipv6Addr;
use std::path::{Path, PathBuf};

pub use super::analysis_args::AnalyzeCommand;
use analyze::analysis::categories;

use crate::commands::Command;
use crate::commands::common::AddressPredicate;
use crate::data::{DataRow, DataStreamInfo, DataStreamResult, stream_from_iter};
use crate::sink::print_datastream_result;

const RATE_RENDER_SCALE: u32 = 2;
const RATE_WIDTH_PX: u32 = 700;
const RATE_HEIGHT_PX: u32 = 420;
const RATE_TICK_PT: i32 = 6;
const RATE_SUCCESS_COLOR: RGBColor = RGBColor(31, 119, 180);
const RATE_ERROR_COLOR: RGBColor = RGBColor(214, 39, 40);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseKind {
    Error,
    Success,
}

impl ResponseKind {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Success, _) | (_, Self::Success) => Self::Success,
            _ => Self::Error,
        }
    }
}

#[derive(Debug, Clone)]
struct RatePlotJob {
    label: String,
    results_path: PathBuf,
    targets_path: PathBuf,
}

#[derive(Parser, Serialize, Deserialize)]
pub struct AnalyzeCommandArgs {
    /// Path to file containing data to analyze
    #[arg(value_name = "FILE")]
    pub file: Option<PathBuf>,
    /// Column name to select from input data
    #[arg(short = 'f', long, value_name = "FIELD")]
    pub field: Option<String>,
    /// Include addresses matching these predicates (can be specified multiple times)
    #[arg(long, value_enum)]
    pub include: Vec<AddressPredicate>,
    /// Exclude addresses matching these predicates (can be specified multiple times)
    #[arg(long, value_enum)]
    pub exclude: Vec<AddressPredicate>,
    /// Remove duplicate addresses from input dataset before analysis
    #[arg(short = 'u', long)]
    pub unique: bool,
    /// Analysis subcommand to run
    #[command(subcommand)]
    pub analysis: AnalyzeCommand,
}

impl Command for AnalyzeCommandArgs {
    fn run(&self, _registry: &sixseven_core::Registry) -> Result<(), String> {
        let result = match &self.analysis {
            AnalyzeCommand::Categories {
                scan_results,
                output_uncategorized,
            } => self.analyze_categories(scan_results, output_uncategorized.as_ref())?,
            AnalyzeCommand::Entropy {
                start_bit,
                end_bit,
                scan_results,
                output_heatmap,
            } if *output_heatmap => {
                if start_bit >= end_bit {
                    return Err("start_bit must be less than end_bit".to_string());
                }
                if start_bit % 4 != 0 || end_bit % 4 != 0 {
                    return Err(
                        "heatmap output requires start_bit and end_bit to align to nibble boundaries"
                            .to_string(),
                    );
                }
                if scan_results.is_empty() {
                    return Err(
                        "--scan-results must be provided at least once when using --output-heatmap"
                            .to_string(),
                    );
                }
                self.analyze_entropy_scan_results_heatmap(scan_results, *start_bit, *end_bit)?
            }
            AnalyzeCommand::Rate {
                results,
                targets,
                results_dir,
                targets_dir,
                output_dir,
                sample_every,
            } => self.analyze_rate(
                results,
                targets,
                results_dir.as_ref(),
                targets_dir.as_ref(),
                output_dir,
                *sample_every,
            )?,
            other => {
                let file = self
                    .file
                    .as_ref()
                    .ok_or_else(|| "FILE argument is required for this subcommand".to_string())?;
                let addresses = self.load_and_filter_addresses(file)?;
                match other {
                    AnalyzeCommand::Dispersion => self.analyze_dispersion(&addresses)?,
                    AnalyzeCommand::Entropy {
                        start_bit,
                        end_bit,
                        scan_results: _,
                        output_heatmap: _,
                    } => {
                        if start_bit >= end_bit {
                            return Err("start_bit must be less than end_bit".to_string());
                        }
                        self.analyze_entropy(&addresses, *start_bit, *end_bit)?
                    }
                    AnalyzeCommand::Bias { start_bit, end_bit } => {
                        if start_bit >= end_bit {
                            return Err("start_bit must be less than end_bit".to_string());
                        }
                        self.analyze_bias(&addresses, *start_bit, *end_bit)?
                    }
                    AnalyzeCommand::Prefix { prefix_length } => {
                        self.analyze_prefix(&addresses, *prefix_length)?
                    }
                    AnalyzeCommand::Subnets {
                        max_subnets,
                        prefix_length,
                    } => self.analyze_subnets(&addresses, *max_subnets, *prefix_length)?,
                    AnalyzeCommand::Counts => self.analyze_counts(&addresses)?,
                    AnalyzeCommand::Categories { .. } => unreachable!(),
                    AnalyzeCommand::Rate { .. } => unreachable!(),
                }
            }
        };

        // Print the result instead of returning it
        print_datastream_result(result, "-")?;
        Ok(())
    }
}

impl AnalyzeCommandArgs {
    fn load_and_filter_addresses(&self, file: &PathBuf) -> Result<Vec<Ipv6Addr>, String> {
        let addresses = sixseven_formats::targets::read(file, self.field.as_deref())
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        // Apply predicate filters
        let filtered_addresses = self.apply_predicates(addresses)?;

        // Apply unique filter if requested
        let final_addresses = if self.unique {
            let mut unique_set = std::collections::HashSet::new();
            filtered_addresses
                .into_iter()
                .filter(|addr| unique_set.insert(*addr))
                .collect()
        } else {
            filtered_addresses
        };

        Ok(final_addresses)
    }

    fn apply_predicates(&self, addresses: Vec<Ipv6Addr>) -> Result<Vec<Ipv6Addr>, String> {
        Ok(addresses
            .into_iter()
            .filter(|address| {
                (self.include.is_empty()
                    || self
                        .include
                        .iter()
                        .any(|predicate| predicate.test(*address)))
                    && !self
                        .exclude
                        .iter()
                        .any(|predicate| predicate.test(*address))
            })
            .collect())
    }

    fn analyze_dispersion(&self, addresses: &[Ipv6Addr]) -> Result<DataStreamResult, String> {
        let analyze::metrics::Dispersion {
            min_distance: min_distance_out,
            max_distance,
            avg_distance,
            total_pairs,
        } = analyze::metrics::dispersion(addresses);

        let row = DataRow::new()
            .with_column("metric", "dispersion")
            .with_column("min_distance", min_distance_out.to_string())
            .with_column("max_distance", max_distance.to_string())
            .with_column("avg_distance", format!("{avg_distance:.6}"))
            .with_column("total_pairs", total_pairs.to_string())
            .with_column("addresses_analyzed", addresses.len().to_string());

        Ok(DataStreamResult::single_row(row))
    }

    fn analyze_entropy(
        &self,
        addresses: &[Ipv6Addr],
        start_bit: u8,
        end_bit: u8,
    ) -> Result<DataStreamResult, String> {
        let range = sixseven_core::BitRange::new(start_bit, end_bit).map_err(|e| e.to_string())?;
        let metrics = analyze::metrics::bit_statistics(addresses, range);
        let total_bits = metrics.total_bits();
        let one_bits = metrics.one_bits();
        let zero_bits = total_bits - one_bits;
        let entropy_value = metrics.entropy();

        let row = DataRow::new()
            .with_column("metric", "entropy")
            .with_column("start_bit", start_bit.to_string())
            .with_column("end_bit", end_bit.to_string())
            .with_column("entropy_value", format!("{entropy_value:.6}"))
            .with_column("total_bits", total_bits.to_string())
            .with_column("zero_bits", zero_bits.to_string())
            .with_column("one_bits", one_bits.to_string())
            .with_column("addresses_analyzed", addresses.len().to_string());

        Ok(DataStreamResult::single_row(row))
    }

    fn analyze_entropy_scan_results_heatmap(
        &self,
        scan_results: &[PathBuf],
        start_bit: u8,
        end_bit: u8,
    ) -> Result<DataStreamResult, String> {
        let start_nibble = start_bit / 4;
        let end_nibble = end_bit / 4;
        let mut rows = Vec::new();

        for (scan_index, path) in scan_results.iter().enumerate() {
            let addresses = self.load_and_filter_scan_result_addresses(path)?;
            let values: Vec<u128> = addresses
                .iter()
                .map(|addr| u128::from_be_bytes(addr.octets()))
                .collect();
            let entropy = analyze::analysis::entropy::compute_nibble_entropy(
                &values,
                start_nibble,
                end_nibble,
            );
            let scan_name = path
                .file_stem()
                .or_else(|| path.file_name())
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());

            for (offset, entropy_bits) in entropy.into_iter().enumerate() {
                rows.push(
                    DataRow::new()
                        .with_column("scan_index", scan_index.to_string())
                        .with_column("nibble_index", (start_nibble as usize + offset).to_string())
                        .with_column("scan_result", scan_name.clone())
                        .with_column("entropy_bits", format!("{entropy_bits:.6}")),
                );
            }
        }

        let info = DataStreamInfo::new(vec![
            "scan_index".into(),
            "nibble_index".into(),
            "scan_result".into(),
            "entropy_bits".into(),
        ])
        .with_total_rows(rows.len())
        .with_description(format!(
            "Per-nibble entropy across {} scan result files",
            scan_results.len()
        ));
        let stream = stream_from_iter(rows);
        Ok(DataStreamResult::new(info, stream))
    }

    fn analyze_bias(
        &self,
        addresses: &[Ipv6Addr],
        start_bit: u8,
        end_bit: u8,
    ) -> Result<DataStreamResult, String> {
        let range = sixseven_core::BitRange::new(start_bit, end_bit).map_err(|e| e.to_string())?;
        let metrics = analyze::metrics::bit_statistics(addresses, range);
        let bit_width = metrics.one_counts.len();
        let one_counts = metrics.one_counts;

        let addresses_analyzed = addresses.len();
        let total_bits = addresses_analyzed.saturating_mul(bit_width);
        let one_bits: usize = one_counts.iter().sum();
        let zero_bits = total_bits.saturating_sub(one_bits);

        let one_ratio = if total_bits > 0 {
            one_bits as f64 / total_bits as f64
        } else {
            0.0
        };
        let zero_ratio = if total_bits > 0 {
            zero_bits as f64 / total_bits as f64
        } else {
            0.0
        };

        let global_bias = (one_ratio - 0.5).abs();
        let normalized_global_bias = (global_bias * 2.0).clamp(0.0, 1.0);

        let mut mean_bit_bias = 0.0f64;
        let mut max_bit_bias = 0.0f64;
        let mut most_biased_bit: Option<u8> = None;

        if addresses_analyzed > 0 && bit_width > 0 {
            let denom = addresses_analyzed as f64;
            for (i, ones) in one_counts.iter().enumerate() {
                let bit_one_ratio = *ones as f64 / denom;
                let bit_bias = (bit_one_ratio - 0.5).abs();
                mean_bit_bias += bit_bias;
                if bit_bias > max_bit_bias {
                    max_bit_bias = bit_bias;
                    most_biased_bit = Some(start_bit + i as u8);
                }
            }
            mean_bit_bias /= bit_width as f64;
        }

        let row = DataRow::new()
            .with_column("metric", "bias")
            .with_column("start_bit", start_bit.to_string())
            .with_column("end_bit", end_bit.to_string())
            .with_column("addresses_analyzed", addresses_analyzed.to_string())
            .with_column("bits_per_address", bit_width.to_string())
            .with_column("total_bits", total_bits.to_string())
            .with_column("zero_bits", zero_bits.to_string())
            .with_column("one_bits", one_bits.to_string())
            .with_column("zero_ratio", format!("{zero_ratio:.6}"))
            .with_column("one_ratio", format!("{one_ratio:.6}"))
            .with_column("global_bias", format!("{global_bias:.6}"))
            .with_column(
                "normalized_global_bias",
                format!("{normalized_global_bias:.6}"),
            )
            .with_column("mean_bit_bias", format!("{mean_bit_bias:.6}"))
            .with_column("max_bit_bias", format!("{max_bit_bias:.6}"))
            .with_column(
                "most_biased_bit",
                most_biased_bit.map(|b| b.to_string()).unwrap_or_default(),
            );

        Ok(DataStreamResult::single_row(row))
    }

    fn analyze_subnets(
        &self,
        addresses: &[Ipv6Addr],
        max_subnets: usize,
        prefix_length: u8,
    ) -> Result<DataStreamResult, String> {
        let mut subnet_list: Vec<_> = analyze::metrics::subnets(addresses, prefix_length)?
            .into_iter()
            .collect();
        subnet_list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        subnet_list.truncate(max_subnets);

        let data_rows: Vec<DataRow> = subnet_list
            .into_iter()
            .map(|(subnet, count)| {
                DataRow::new()
                    .with_column("subnet", format!("{subnet}/{prefix_length}"))
                    .with_column("count", count.to_string())
                    .with_column("prefix_length", prefix_length.to_string())
            })
            .collect();

        let headers = vec![
            "subnet".to_string(),
            "count".to_string(),
            "prefix_length".to_string(),
        ];
        let info = DataStreamInfo::new(headers)
            .with_total_rows(data_rows.len())
            .with_description(format!(
                "Top {} subnets with /{} prefix",
                data_rows.len(),
                prefix_length
            ));

        let stream = stream_from_iter(data_rows);
        Ok(DataStreamResult::new(info, stream))
    }

    fn analyze_prefix(
        &self,
        addresses: &[Ipv6Addr],
        prefix_length: u8,
    ) -> Result<DataStreamResult, String> {
        let unique_count = analyze::metrics::subnets(addresses, prefix_length)?.len();
        let addresses_count = addresses.len();
        let avg_per_prefix = if unique_count > 0 {
            addresses_count as f64 / unique_count as f64
        } else {
            0.0
        };

        let row = DataRow::new()
            .with_column("metric", "prefix")
            .with_column("prefix_length", prefix_length.to_string())
            .with_column("unique_prefixes", unique_count.to_string())
            .with_column("addresses_analyzed", addresses_count.to_string())
            .with_column("avg_addresses_per_prefix", format!("{avg_per_prefix:.6}"));

        Ok(DataStreamResult::single_row(row))
    }

    fn load_and_filter_scan_result_addresses(
        &self,
        path: &PathBuf,
    ) -> Result<Vec<Ipv6Addr>, String> {
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let headers = reader
            .headers()
            .map_err(|e| format!("Failed to read {} header: {}", path.display(), e))?
            .clone();
        let schema = ScanRecord::schema(&headers)
            .map_err(|e| format!("Invalid scan result schema in {}: {e}", path.display()))?;

        let mut addresses = Vec::new();
        for record in reader.records() {
            let record = record.map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let parsed = ScanRecord::from_csv_record(&schema, &record)
                .map_err(|e| format!("Failed to parse {} row: {e}", path.display()))?;
            addresses.push(parsed.target);
        }

        let filtered_addresses = self.apply_predicates(addresses)?;
        let final_addresses = if self.unique {
            let mut unique_set = std::collections::HashSet::new();
            filtered_addresses
                .into_iter()
                .filter(|addr| unique_set.insert(*addr))
                .collect()
        } else {
            filtered_addresses
        };

        Ok(final_addresses)
    }

    fn analyze_rate(
        &self,
        results: &[PathBuf],
        targets: &[PathBuf],
        results_dir: Option<&PathBuf>,
        targets_dir: Option<&PathBuf>,
        output_dir: &PathBuf,
        sample_every: usize,
    ) -> Result<DataStreamResult, String> {
        if sample_every == 0 {
            return Err("--sample-every must be greater than zero".to_string());
        }

        let mut jobs = build_rate_jobs(results, targets, results_dir, targets_dir, output_dir)?;
        jobs.retain(|job| job.label.ends_with("-1M"));
        if jobs.is_empty() {
            return Err("No 1M results/targets pairs were found for rate analysis".to_string());
        }

        fs::create_dir_all(output_dir).map_err(|e| {
            format!(
                "Failed to create output directory {}: {}",
                output_dir.display(),
                e
            )
        })?;
        let output_path = output_dir.join("rate.png");

        let mut series = Vec::with_capacity(jobs.len());
        let mut rows = Vec::with_capacity(jobs.len());
        for job in jobs {
            let plot = collect_rate_plot(&job, sample_every)?;
            let stats = plot.stats;
            rows.push(
                DataRow::new()
                    .with_column("label", job.label.clone())
                    .with_column("results", job.results_path.display().to_string())
                    .with_column("targets", job.targets_path.display().to_string())
                    .with_column("output", output_path.display().to_string())
                    .with_column("targets_processed", stats.targets_processed.to_string())
                    .with_column("successes", stats.successes.to_string())
                    .with_column("errors", stats.errors.to_string())
                    .with_column("sample_points", stats.sample_points.to_string()),
            );
            series.push(plot);
        }

        render_combined_rate_plot(&series, &output_path)?;

        let info = DataStreamInfo::new(vec![
            "label".into(),
            "results".into(),
            "targets".into(),
            "output".into(),
            "targets_processed".into(),
            "successes".into(),
            "errors".into(),
            "sample_points".into(),
        ])
        .with_total_rows(rows.len())
        .with_description("Cumulative response-rate plots written to PNG".to_string());
        let stream = stream_from_iter(rows);
        Ok(DataStreamResult::new(info, stream))
    }

    fn analyze_categories(
        &self,
        scan_results: &[PathBuf],
        output_uncategorized: Option<&PathBuf>,
    ) -> Result<DataStreamResult, String> {
        let category_names = categories::all_category_names();

        struct FileStats {
            name: String,
            counts: HashMap<String, usize>,
            total: usize,
        }

        let mut all_stats = Vec::new();
        let mut all_uncategorized: Vec<Ipv6Addr> = Vec::new();

        for path in scan_results {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let mut counts: HashMap<String, usize> = HashMap::new();
            for name in &category_names {
                counts.insert(name.to_string(), 0);
            }

            let mut lines = content.lines();
            let header = lines.next().unwrap_or("");
            let columns: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
            let saddr_idx = columns.iter().position(|&c| c == "saddr").unwrap_or(0);
            let success_idx = columns.iter().position(|&c| c == "success");
            let alias_idx = columns.iter().position(|&c| c == "alias");

            for line in lines {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let fields: Vec<&str> = line.split(',').collect();
                // Skip error responses if a success column exists
                if let Some(si) = success_idx
                    && fields.get(si).map(|v| v.trim()) != Some("1")
                {
                    continue;
                }
                // Count aliased addresses separately
                if let Some(ai) = alias_idx
                    && fields.get(ai).map(|v| v.trim()) == Some("aliased")
                {
                    *counts.entry("aliased".to_string()).or_insert(0) += 1;
                    continue;
                }
                let field = fields.get(saddr_idx).copied().unwrap_or("");
                if let Ok(addr) = field.parse::<Ipv6Addr>() {
                    let cat = categories::categorize(addr);
                    *counts.entry(cat.to_string()).or_insert(0) += 1;
                    if cat == "uncategorized" && output_uncategorized.is_some() {
                        all_uncategorized.push(addr);
                    }
                }
            }

            let total: usize = counts.values().sum();
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());

            all_stats.push(FileStats {
                name: file_name,
                counts,
                total,
            });
        }

        // Write uncategorized addresses if requested
        if let Some(out_path) = output_uncategorized {
            let content = all_uncategorized
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(out_path, content)
                .map_err(|e| format!("Failed to write uncategorized: {}", e))?;
            eprintln!(
                "Wrote {} uncategorized addresses to {}",
                all_uncategorized.len(),
                out_path.display()
            );
        }

        // Build rows: one per category
        let mut headers = vec!["category".to_string()];
        for stat in &all_stats {
            headers.push(stat.name.clone());
        }

        // Check if any file had aliased addresses
        let has_aliased = all_stats.iter().any(|s| s.counts.contains_key("aliased"));

        let mut all_names: Vec<&str> = Vec::new();
        if has_aliased {
            all_names.push("aliased");
        }
        all_names.extend(category_names.iter());

        let mut data_rows = Vec::new();
        for cat_name in &all_names {
            let mut row = DataRow::new().with_column("category", *cat_name);
            for stat in &all_stats {
                let count = stat.counts.get(*cat_name).copied().unwrap_or(0);
                let pct = if stat.total > 0 {
                    (count as f64 / stat.total as f64) * 100.0
                } else {
                    0.0
                };
                row = row.with_column(&stat.name, format!("{} ({:.2}%)", count, pct));
            }
            data_rows.push(row);
        }

        let info = DataStreamInfo::new(headers).with_total_rows(data_rows.len());
        let stream = stream_from_iter(data_rows);
        Ok(DataStreamResult::new(info, stream))
    }

    fn analyze_counts(&self, addresses: &[Ipv6Addr]) -> Result<DataStreamResult, String> {
        // Analyze predicate counts
        let all_predicates = analyze::analysis::predicates::get_all_predicates();

        let mut data_rows = Vec::new();

        for (predicate_name, predicate_fn) in all_predicates {
            let count = addresses.iter().filter(|addr| predicate_fn(**addr)).count();

            let row = DataRow::new()
                .with_column("predicate", predicate_name)
                .with_column("count", count.to_string())
                .with_column(
                    "percentage",
                    format!("{:.2}%", (count as f64 / addresses.len() as f64) * 100.0),
                );

            data_rows.push(row);
        }

        let headers = vec![
            "predicate".to_string(),
            "count".to_string(),
            "percentage".to_string(),
        ];
        let info = DataStreamInfo::new(headers)
            .with_total_rows(data_rows.len())
            .with_description(format!(
                "Predicate analysis for {} addresses",
                addresses.len()
            ));

        let stream = stream_from_iter(data_rows);
        Ok(DataStreamResult::new(info, stream))
    }
}

#[derive(Debug, Clone, Copy)]
struct RatePlotStats {
    targets_processed: usize,
    successes: usize,
    errors: usize,
    sample_points: usize,
}

fn build_rate_jobs(
    results: &[PathBuf],
    targets: &[PathBuf],
    results_dir: Option<&PathBuf>,
    targets_dir: Option<&PathBuf>,
    _output_dir: &Path,
) -> Result<Vec<RatePlotJob>, String> {
    let using_explicit = !results.is_empty() || !targets.is_empty();
    let using_dirs = results_dir.is_some() || targets_dir.is_some();
    if using_explicit && using_dirs {
        return Err(
            "Use either explicit --results/--targets pairs or --results-dir/--targets-dir, not both"
                .to_string(),
        );
    }

    if using_explicit {
        if results.is_empty() || targets.is_empty() {
            return Err(
                "Explicit rate analysis requires both --results and --targets to be provided"
                    .to_string(),
            );
        }
        if results.len() != targets.len() {
            return Err(format!(
                "Expected the same number of --results and --targets files, got {} and {}",
                results.len(),
                targets.len()
            ));
        }

        return results
            .iter()
            .zip(targets.iter())
            .map(|(results_path, targets_path)| {
                let label = pair_label_from_paths(results_path, targets_path);
                Ok(RatePlotJob {
                    label,
                    results_path: results_path.clone(),
                    targets_path: targets_path.clone(),
                })
            })
            .collect();
    }

    let results_root = results_dir.ok_or_else(|| {
        "--results-dir is required when explicit --results are not provided".to_string()
    })?;
    let targets_root = targets_dir.ok_or_else(|| {
        "--targets-dir is required when explicit --targets are not provided".to_string()
    })?;

    let mut result_files = Vec::new();
    collect_csv_files(results_root, &mut result_files)?;

    let mut targets_by_key = BTreeMap::new();
    let mut target_files = Vec::new();
    collect_csv_files(targets_root, &mut target_files)?;
    for target_path in target_files {
        let key = relative_pair_key(targets_root, &target_path, true)?;
        if targets_by_key
            .insert(key.clone(), target_path.clone())
            .is_some()
        {
            return Err(format!(
                "Multiple target files mapped to the same rate-analysis key '{}'",
                key
            ));
        }
    }

    let mut jobs = Vec::new();
    for results_path in result_files {
        let key = relative_pair_key(results_root, &results_path, false)?;
        let Some(targets_path) = targets_by_key.get(&key) else {
            continue;
        };
        jobs.push(RatePlotJob {
            label: key,
            results_path,
            targets_path: targets_path.clone(),
        });
    }

    jobs.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(jobs)
}

fn collect_csv_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(root)
        .map_err(|e| format!("Failed to read directory {}: {}", root.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_csv_files(&path, out)?;
            continue;
        }
        let is_csv = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("csv"))
            .unwrap_or(false);
        if is_csv {
            out.push(path);
        }
    }
    out.sort();
    Ok(())
}

fn relative_pair_key(
    root: &Path,
    path: &Path,
    strip_target_version: bool,
) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("{} is not under {}", path.display(), root.display()))?;
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let stem = relative
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("Invalid UTF-8 file name for {}", path.display()))?;
    let stem = if strip_target_version {
        strip_target_version_suffix(stem)
    } else {
        stem.to_string()
    };

    let mut key = parent.join(stem).to_string_lossy().replace('\\', "/");
    while key.starts_with("./") {
        key = key[2..].to_string();
    }
    Ok(key)
}

fn strip_target_version_suffix(stem: &str) -> String {
    for marker in ["-v", "_v"] {
        if let Some((prefix, suffix)) = stem.rsplit_once(marker)
            && !prefix.is_empty()
            && suffix.chars().all(|ch| ch.is_ascii_digit())
        {
            return prefix.to_string();
        }
    }
    stem.to_string()
}

fn pair_label_from_paths(results_path: &Path, targets_path: &Path) -> String {
    let results_stem = results_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("results");
    let targets_stem = targets_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(strip_target_version_suffix)
        .unwrap_or_else(|| "targets".to_string());
    if results_stem == targets_stem {
        results_stem.to_string()
    } else {
        format!("{results_stem}__{targets_stem}")
    }
}

#[derive(Debug, Clone)]
struct RatePlotData {
    label: String,
    success_points: Vec<(f64, f64)>,
    error_points: Vec<(f64, f64)>,
    x_max: f64,
    y_max: f64,
    stats: RatePlotStats,
}

fn collect_rate_plot(job: &RatePlotJob, sample_every: usize) -> Result<RatePlotData, String> {
    let responses = load_rate_responses(&job.results_path)?;
    let mut target_reader = csv::Reader::from_path(&job.targets_path)
        .map_err(|e| format!("Failed to open {}: {}", job.targets_path.display(), e))?;
    let target_headers = target_reader
        .headers()
        .map_err(|e| {
            format!(
                "Failed to read {} header: {}",
                job.targets_path.display(),
                e
            )
        })?
        .clone();
    let schema = TargetRecord::schema(&target_headers).map_err(|e| {
        format!(
            "Invalid target schema in {}: {e}",
            job.targets_path.display()
        )
    })?;

    let mut success_total = 0usize;
    let mut error_total = 0usize;
    let mut target_total = 0usize;
    let mut success_points = Vec::new();
    let mut error_points = Vec::new();

    for record in target_reader.records() {
        let record =
            record.map_err(|e| format!("Failed to read {}: {}", job.targets_path.display(), e))?;
        let address = TargetRecord::from_csv_record(&schema, &record)
            .map_err(|e| format!("Failed to parse {} row: {e}", job.targets_path.display()))?
            .address;

        target_total += 1;
        match responses.get(&address).copied() {
            Some(ResponseKind::Success) => success_total += 1,
            Some(ResponseKind::Error) => error_total += 1,
            None => {}
        }

        if target_total.is_multiple_of(sample_every) {
            push_rate_point(
                &mut success_points,
                &mut error_points,
                target_total,
                success_total,
                error_total,
            );
        }
    }

    if target_total == 0 {
        return Err(format!(
            "No targets found in {}",
            job.targets_path.display()
        ));
    }

    if success_points
        .last()
        .map(|(index, _)| *index != target_total as f64)
        .unwrap_or(true)
    {
        push_rate_point(
            &mut success_points,
            &mut error_points,
            target_total,
            success_total,
            error_total,
        );
    }

    let max_rate = success_points
        .iter()
        .chain(error_points.iter())
        .map(|(_, value)| *value)
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let sample_points = success_points.len();

    Ok(RatePlotData {
        label: job.label.clone(),
        success_points,
        error_points,
        x_max: target_total as f64,
        y_max: max_rate,
        stats: RatePlotStats {
            targets_processed: target_total,
            successes: success_total,
            errors: error_total,
            sample_points,
        },
    })
}

fn render_combined_rate_plot(series: &[RatePlotData], output_path: &Path) -> Result<(), String> {
    if series.is_empty() {
        return Err("No rate series available to plot".to_string());
    }

    let width = RATE_WIDTH_PX * RATE_RENDER_SCALE;
    let height = RATE_HEIGHT_PX * RATE_RENDER_SCALE;
    let global_x_max = series
        .iter()
        .map(|plot| plot.x_max)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let global_y_max = series
        .iter()
        .map(|plot| plot.y_max)
        .fold(0.0f64, f64::max)
        .max(1e-9)
        * 1.05;

    let backend = BitMapBackend::new(output_path, (width, height));
    let root = backend.into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| format!("Failed to initialize {}: {:?}", output_path.display(), e))?;

    let mut chart = ChartBuilder::on(&root)
        .margin((8 * RATE_RENDER_SCALE) as i32)
        .x_label_area_size(28 * RATE_RENDER_SCALE)
        .y_label_area_size(34 * RATE_RENDER_SCALE)
        .build_cartesian_2d(0f64..global_x_max, 0f64..global_y_max)
        .map_err(|e| format!("Failed to build {}: {:?}", output_path.display(), e))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .light_line_style(TRANSPARENT)
        .x_labels(5)
        .y_labels(5)
        .x_label_style(("serif", RATE_TICK_PT * RATE_RENDER_SCALE as i32))
        .y_label_style(("serif", RATE_TICK_PT * RATE_RENDER_SCALE as i32))
        .x_label_formatter(&|value| format_target_position(*value))
        .y_label_formatter(&|value| format_rate(*value))
        .draw()
        .map_err(|e| format!("Failed to draw axes for {}: {:?}", output_path.display(), e))?;

    for (index, plot) in series.iter().enumerate() {
        let success_style = rate_line_style(index, series.len(), true);
        let error_style = rate_line_style(index, series.len(), false);
        chart
            .draw_series(std::iter::once(PathElement::new(
                plot.success_points.clone(),
                success_style,
            )))
            .map_err(|e| {
                format!(
                    "Failed to draw success series '{}' in {}: {:?}",
                    plot.label,
                    output_path.display(),
                    e
                )
            })?;
        chart
            .draw_series(std::iter::once(PathElement::new(
                plot.error_points.clone(),
                error_style,
            )))
            .map_err(|e| {
                format!(
                    "Failed to draw error series '{}' in {}: {:?}",
                    plot.label,
                    output_path.display(),
                    e
                )
            })?;
    }

    root.present()
        .map_err(|e| format!("Failed to write {}: {:?}", output_path.display(), e))?;
    Ok(())
}

fn load_rate_responses(path: &Path) -> Result<HashMap<Ipv6Addr, ResponseKind>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("Failed to read {} header: {}", path.display(), e))?
        .clone();
    let schema = ScanRecord::schema(&headers)
        .map_err(|e| format!("Invalid scan result schema in {}: {e}", path.display()))?;

    let mut responses = HashMap::new();
    for record in reader.records() {
        let record = record.map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let parsed = ScanRecord::from_csv_record(&schema, &record)
            .map_err(|e| format!("Failed to parse {} row: {e}", path.display()))?;
        let kind = if parsed.success {
            ResponseKind::Success
        } else {
            ResponseKind::Error
        };
        responses
            .entry(parsed.target)
            .and_modify(|existing: &mut ResponseKind| *existing = (*existing).merge(kind))
            .or_insert(kind);
    }

    Ok(responses)
}

fn push_rate_point(
    success_points: &mut Vec<(f64, f64)>,
    error_points: &mut Vec<(f64, f64)>,
    target_total: usize,
    success_total: usize,
    error_total: usize,
) {
    let index = target_total as f64;
    let denom = target_total as f64;
    success_points.push((index, success_total as f64 / denom));
    error_points.push((index, error_total as f64 / denom));
}

fn rate_line_style(index: usize, total: usize, success: bool) -> ShapeStyle {
    let total = total.max(1);
    let shade = if total == 1 {
        1.0
    } else {
        0.35 + 0.65 * (index as f64 / (total - 1) as f64)
    };
    let color = if success {
        RATE_SUCCESS_COLOR.mix(shade)
    } else {
        RATE_ERROR_COLOR.mix(shade)
    };
    color.stroke_width(RATE_RENDER_SCALE.max(1))
}

fn format_target_position(value: f64) -> String {
    let value = value.max(0.0).round() as u64;
    match value {
        0..=999 => value.to_string(),
        1_000..=999_999 => format!("{:.1}K", value as f64 / 1_000.0),
        1_000_000..=999_999_999 => format!("{:.1}M", value as f64 / 1_000_000.0),
        _ => format!("{:.1}B", value as f64 / 1_000_000_000.0),
    }
}

fn format_rate(value: f64) -> String {
    if value >= 0.1 {
        format!("{value:.2}")
    } else if value >= 0.01 {
        format!("{value:.3}")
    } else if value >= 0.001 {
        format!("{value:.4}")
    } else if value > 0.0 {
        format!("{value:.1e}")
    } else {
        "0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_target_versions() {
        assert_eq!(strip_target_version_suffix("ntp-10K-v0"), "ntp-10K");
        assert_eq!(strip_target_version_suffix("hitlist_1M_v12"), "hitlist_1M");
        assert_eq!(strip_target_version_suffix("plain-name"), "plain-name");
    }

    #[test]
    fn success_overrides_error() {
        assert_eq!(
            ResponseKind::Error.merge(ResponseKind::Success),
            ResponseKind::Success
        );
        assert_eq!(
            ResponseKind::Success.merge(ResponseKind::Error),
            ResponseKind::Success
        );
    }
}
