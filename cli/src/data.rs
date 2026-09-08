//! Tabular command results.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single row of data with named columns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRow {
    pub columns: HashMap<String, String>,
}

impl DataRow {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    pub fn with_column(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.columns.insert(name.into(), value.into());
        self
    }

    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.columns.insert(name.into(), value.into());
    }

    pub fn get(&self, name: &str) -> Option<&String> {
        self.columns.get(name)
    }

    /// Convert to CSV line with specified column order
    pub fn to_csv_line(&self, headers: &[String]) -> String {
        csv_line(
            headers
                .iter()
                .map(|h| self.columns.get(h).map(String::as_str).unwrap_or("")),
        )
    }

    /// Convert to CSV line with automatic column order
    pub fn to_csv_line_auto(&self) -> (Vec<String>, String) {
        let mut headers: Vec<String> = self.columns.keys().cloned().collect();
        headers.sort(); // Consistent ordering
        let line = self.to_csv_line(&headers);
        (headers, line)
    }
}

pub fn csv_line<'a>(fields: impl IntoIterator<Item = &'a str>) -> String {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    writer.write_record(fields).expect("writing CSV to memory");
    let bytes = writer.into_inner().expect("flushing CSV to memory");
    let mut line = String::from_utf8(bytes).expect("CSV preserves UTF-8");
    line.pop();
    line
}

impl Default for DataRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a data stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStreamInfo {
    pub headers: Vec<String>,
    pub total_rows: Option<usize>,
    pub description: Option<String>,
}

impl DataStreamInfo {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            total_rows: None,
            description: None,
        }
    }

    pub fn with_total_rows(mut self, total: usize) -> Self {
        self.total_rows = Some(total);
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Result type for command execution with streaming data
pub struct DataStreamResult {
    pub info: DataStreamInfo,
    pub stream: Box<dyn Iterator<Item = Result<DataRow, String>> + Send>,
}

impl std::fmt::Debug for DataStreamResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataStreamResult")
            .field("info", &self.info)
            .field("stream", &"<Stream>")
            .finish()
    }
}

impl DataStreamResult {
    pub fn new<S>(info: DataStreamInfo, stream: S) -> Self
    where
        S: Iterator<Item = Result<DataRow, String>> + Send + 'static,
    {
        Self {
            info,
            stream: Box::new(stream),
        }
    }

    /// Create a simple result with a single row
    pub fn single_row(row: DataRow) -> Self {
        let (headers, _) = row.to_csv_line_auto();
        let info = DataStreamInfo::new(headers).with_total_rows(1);
        let stream = std::iter::once(Ok(row));
        Self::new(info, stream)
    }
}

/// Helper trait for converting various data types to DataRow
pub trait IntoDataRow {
    fn into_data_row(self) -> DataRow;
}

impl IntoDataRow for DataRow {
    fn into_data_row(self) -> DataRow {
        self
    }
}

impl IntoDataRow for HashMap<String, String> {
    fn into_data_row(self) -> DataRow {
        DataRow { columns: self }
    }
}

impl<const N: usize> IntoDataRow for [(&str, &str); N] {
    fn into_data_row(self) -> DataRow {
        let mut row = DataRow::new();
        for (key, value) in self {
            row.insert(key, value);
        }
        row
    }
}

/// Helper for creating streams from iterators
pub fn stream_from_iter<I, T>(iter: I) -> impl Iterator<Item = Result<DataRow, String>>
where
    I: IntoIterator<Item = T>,
    T: IntoDataRow,
{
    iter.into_iter().map(|item| Ok(item.into_data_row()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_row_creation() {
        let row = DataRow::new()
            .with_column("name", "Alice")
            .with_column("age", "25");

        assert_eq!(row.get("name"), Some(&"Alice".to_string()));
        assert_eq!(row.get("age"), Some(&"25".to_string()));
        assert_eq!(row.get("missing"), None);
    }

    #[test]
    fn test_csv_conversion() {
        let row = DataRow::new()
            .with_column("name", "Alice")
            .with_column("age", "25");

        let headers = vec!["name".to_string(), "age".to_string()];
        let csv_line = row.to_csv_line(&headers);
        assert_eq!(csv_line, "Alice,25");

        let (auto_headers, auto_line) = row.to_csv_line_auto();
        assert!(auto_headers.contains(&"name".to_string()));
        assert!(auto_headers.contains(&"age".to_string()));
        assert!(auto_line.contains("Alice"));
        assert!(auto_line.contains("25"));
    }

    #[test]
    fn test_stream_result() {
        let rows = vec![
            DataRow::new().with_column("name", "Alice"),
            DataRow::new().with_column("name", "Bob"),
        ];

        let info = DataStreamInfo::new(vec!["name".to_string()]);
        let stream = stream_from_iter(rows);
        let result = DataStreamResult::new(info, stream);

        assert_eq!(result.info.headers, vec!["name".to_string()]);
    }
}
