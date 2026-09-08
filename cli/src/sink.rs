//! Output formatting and streaming for command results

use crate::data::DataStreamResult;
use crate::loaders::csv::write_csv_stream;
use comfy_table::{ContentArrangement, Table, modifiers, presets};
use std::path::PathBuf;

/// Print streaming data results to console or file
pub fn print_datastream_result(result: DataStreamResult, output_file: &str) -> Result<(), String> {
    if output_file == "-" {
        // Print to console as table
        print_to_console(result)
    } else {
        // Write to file
        let path = PathBuf::from(output_file);
        if is_csv_file(&path) {
            write_csv_stream(result, &path)
        } else {
            write_plain_text(result, &path)
        }
    }
}

/// Print streaming results to console as a formatted table
fn print_to_console(mut result: DataStreamResult) -> Result<(), String> {
    let headers = result.info.headers.clone();

    // Create table with headers
    let mut table = Table::new();
    table
        .load_preset(presets::UTF8_FULL_CONDENSED)
        .apply_modifier(modifiers::UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(&headers);

    // Add rows from stream
    let mut row_count = 0;
    for result_item in result.stream.by_ref() {
        match result_item {
            Ok(row) => {
                let row_values: Vec<String> = headers
                    .iter()
                    .map(|h| row.get(h).cloned().unwrap_or_default())
                    .collect();

                table.add_row(row_values);
                row_count += 1;

                // Print partial results for large datasets (every 1000 rows)
                if row_count % 1000 == 0 {
                    println!("Processed {} rows...", row_count);
                }
            }
            Err(e) => return Err(format!("Stream error: {}", e)),
        }
    }

    // Print the final table
    println!("{}", table);

    if let Some(description) = result.info.description {
        println!("\n{}", description);
    }

    println!("Total rows: {}", row_count);

    Ok(())
}

/// Write streaming results to a plain text file
fn write_plain_text(mut result: DataStreamResult, path: &PathBuf) -> Result<(), String> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(path).map_err(|e| format!("Failed to create output file: {}", e))?;

    let mut writer = BufWriter::new(file);
    let headers = result.info.headers.clone();

    // Write headers
    let header_line = headers.join("\t");
    writer
        .write_all(format!("{}\n", header_line).as_bytes())
        .map_err(|e| format!("Failed to write headers: {}", e))?;

    // Write data rows
    for result_item in result.stream.by_ref() {
        match result_item {
            Ok(row) => {
                let row_values: Vec<String> = headers
                    .iter()
                    .map(|h| row.get(h).cloned().unwrap_or_default())
                    .collect();

                let row_line = row_values.join("\t");
                writer
                    .write_all(format!("{}\n", row_line).as_bytes())
                    .map_err(|e| format!("Failed to write row: {}", e))?;
            }
            Err(e) => return Err(format!("Stream error: {}", e)),
        }
    }

    writer
        .flush()
        .map_err(|e| format!("Failed to flush output: {}", e))?;

    Ok(())
}

/// Check if file path indicates CSV format
fn is_csv_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase() == "csv")
        .unwrap_or(false)
}
