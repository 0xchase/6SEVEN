//! CSV writer utilities used by CLI output sinks.

use crate::data::{DataRow, DataStreamResult};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub struct CsvWriter {
    writer: BufWriter<File>,
    headers_written: bool,
}

impl CsvWriter {
    pub fn from_file(file_path: &PathBuf) -> Result<Self, String> {
        let file =
            File::create(file_path).map_err(|e| format!("Failed to create CSV file: {}", e))?;

        Ok(Self {
            writer: BufWriter::new(file),
            headers_written: false,
        })
    }

    pub fn write_headers(&mut self, headers: &[String]) -> Result<(), String> {
        if !self.headers_written {
            let header_line = crate::data::csv_line(headers.iter().map(String::as_str));
            self.writer
                .write_all(format!("{}\n", header_line).as_bytes())
                .map_err(|e| format!("Failed to write CSV headers: {}", e))?;
            self.headers_written = true;
        }
        Ok(())
    }

    pub fn write_row(&mut self, row: &DataRow, headers: &[String]) -> Result<(), String> {
        let csv_line = row.to_csv_line(headers);
        self.writer
            .write_all(format!("{}\n", csv_line).as_bytes())
            .map_err(|e| format!("Failed to write CSV row: {}", e))
    }

    pub fn write_stream<S>(&mut self, stream: S, headers: &[String]) -> Result<(), String>
    where
        S: Iterator<Item = Result<DataRow, String>>,
    {
        self.write_headers(headers)?;

        for result in stream {
            match result {
                Ok(row) => self.write_row(&row, headers)?,
                Err(e) => return Err(format!("Stream error: {}", e)),
            }
        }

        Ok(())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|e| format!("Failed to flush CSV output: {}", e))
    }
}

pub fn write_csv_stream(result: DataStreamResult, file_path: &PathBuf) -> Result<(), String> {
    let mut writer = CsvWriter::from_file(file_path)?;
    let headers = result.info.headers.clone();
    let stream = result.stream;

    writer.write_stream(stream, &headers)?;
    writer.finish()
}
