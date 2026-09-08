use crate::Error;
use std::{
    fs::File,
    io::{BufRead, BufReader, Lines, Seek},
    net::Ipv6Addr,
    path::Path,
};

pub enum AddressReader {
    Plain {
        lines: Lines<BufReader<File>>,
        line: usize,
    },
    Csv {
        reader: csv::Reader<File>,
        column: usize,
    },
}

pub fn read(path: impl AsRef<Path>, field: Option<&str>) -> Result<AddressReader, Error> {
    let mut file = File::open(path)?;
    let mut first = String::new();
    {
        let mut reader = BufReader::new(&mut file);
        loop {
            first.clear();
            if reader.read_line(&mut first)? == 0 {
                break;
            }
            if !first.trim().is_empty() && !first.trim().starts_with('#') {
                break;
            }
        }
    }
    file.rewind()?;
    let csv = field.is_some() || first.contains(',') || first.trim().eq_ignore_ascii_case("saddr");
    if csv {
        let mut reader = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .trim(csv::Trim::All)
            .from_reader(file);
        let name = field.unwrap_or("saddr");
        let column = reader
            .headers()?
            .iter()
            .position(|header| header.eq_ignore_ascii_case(name))
            .ok_or_else(|| Error::Invalid(format!("missing address column '{name}'")))?;
        Ok(AddressReader::Csv { reader, column })
    } else {
        Ok(AddressReader::Plain {
            lines: BufReader::new(file).lines(),
            line: 0,
        })
    }
}

impl Iterator for AddressReader {
    type Item = Result<Ipv6Addr, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Plain { lines, line } => loop {
                let value = match lines.next()? {
                    Ok(value) => value,
                    Err(error) => return Some(Err(error.into())),
                };
                *line += 1;
                let value = value.trim();
                if value.is_empty() || value.starts_with('#') {
                    continue;
                }
                return Some(value.parse().map_err(|error| {
                    Error::Invalid(format!("invalid address on line {line}: {error}"))
                }));
            },
            Self::Csv { reader, column } => {
                let mut record = csv::StringRecord::new();
                match reader.read_record(&mut record) {
                    Ok(false) => None,
                    Err(error) => Some(Err(error.into())),
                    Ok(true) => Some(
                        record
                            .get(*column)
                            .ok_or_else(|| Error::Invalid("missing address field".into()))
                            .and_then(|value| {
                                value.parse().map_err(|error| {
                                    Error::Invalid(format!("invalid CSV address: {error}"))
                                })
                            }),
                    ),
                }
            }
        }
    }
}

pub async fn load_addresses(
    path: impl AsRef<Path>,
    budget: Option<usize>,
) -> Result<Vec<Ipv6Addr>, Error> {
    let path = path.as_ref().to_path_buf();
    tokio::task::spawn_blocking(move || {
        read(path, None)?
            .take(budget.unwrap_or(usize::MAX))
            .collect()
    })
    .await
    .map_err(|error| Error::Invalid(error.to_string()))?
}

/// Addresses eligible for active dealiasing. Scan errors never supply candidates.
pub fn candidates(
    path: impl AsRef<Path>,
) -> Result<Box<dyn Iterator<Item = Result<Ipv6Addr, Error>> + Send>, Error> {
    let input = read(path, None)?;
    match input {
        AddressReader::Plain { .. } => Ok(Box::new(input)),
        AddressReader::Csv { mut reader, column } => {
            let headers = reader.headers()?;
            let scan = headers.iter().any(|h| {
                matches!(
                    h.to_ascii_lowercase().as_str(),
                    "class" | "classification" | "success"
                )
            });
            let schema = if scan {
                Some(crate::csv::scan::ScanRecord::schema(headers).map_err(Error::Invalid)?)
            } else {
                None
            };
            Ok(Box::new(reader.into_records().filter_map(move |row| {
                let row = match row {
                    Ok(row) => row,
                    Err(error) => return Some(Err(error.into())),
                };
                if let Some(schema) = &schema {
                    match crate::csv::scan::ScanRecord::from_csv_record(schema, &row) {
                        Ok(record) if record.success => Some(Ok(record.target)),
                        Ok(_) => None,
                        Err(error) => Some(Err(Error::Invalid(error))),
                    }
                } else {
                    Some(
                        row.get(column)
                            .unwrap_or_default()
                            .trim()
                            .parse()
                            .map_err(|e| Error::Invalid(format!("invalid address: {e}"))),
                    )
                }
            })))
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct FilterStats {
    pub retained: u64,
    pub removed: u64,
}

/// Filter by target address while retaining CSV columns/order or plain-list lines.
pub fn filter(
    path: impl AsRef<Path>,
    output: impl std::io::Write,
    prefixes: &sixseven_core::PrefixSet,
) -> Result<FilterStats, Error> {
    let path = path.as_ref();
    let input = read(path, None)?;
    let mut stats = FilterStats::default();
    match input {
        AddressReader::Csv { column, .. } => {
            let mut reader = csv::ReaderBuilder::new()
                .comment(Some(b'#'))
                .from_path(path)?;
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(output);
            writer.write_record(reader.headers()?)?;
            for row in reader.records() {
                let row = row?;
                let address = row
                    .get(column)
                    .unwrap_or_default()
                    .trim()
                    .parse()
                    .map_err(|e| Error::Invalid(format!("invalid address: {e}")))?;
                if prefixes.contains(address) {
                    stats.removed += 1;
                } else {
                    stats.retained += 1;
                    writer.write_record(&row)?;
                }
            }
            writer.flush()?;
        }
        AddressReader::Plain { .. } => {
            let mut reader = BufReader::new(File::open(path)?);
            let mut writer = std::io::BufWriter::new(output);
            let mut line = String::new();
            while reader.read_line(&mut line)? != 0 {
                let value = line.trim();
                let keep = if value.is_empty() || value.starts_with('#') {
                    true
                } else {
                    let address = value
                        .parse()
                        .map_err(|e| Error::Invalid(format!("invalid address: {e}")))?;
                    if prefixes.contains(address) {
                        stats.removed += 1;
                        false
                    } else {
                        stats.retained += 1;
                        true
                    }
                };
                if keep {
                    std::io::Write::write_all(&mut writer, line.as_bytes())?;
                }
                line.clear();
            }
            std::io::Write::flush(&mut writer)?;
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn preserves_unknown_csv_fields_and_tests_only_successes() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "saddr,class,success,note\n::1,reply,1,\"hello, world\"\n::2,timeout,0, keep spaces \n::3,reply,1,last\n").unwrap();
        let values: Vec<_> = candidates(file.path())
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            values,
            vec!["::1".parse::<Ipv6Addr>().unwrap(), "::3".parse().unwrap()]
        );
        let prefixes = ["::1/128".parse().unwrap()].into_iter().collect();
        let mut output = Vec::new();
        let stats = filter(file.path(), &mut output, &prefixes).unwrap();
        assert_eq!(
            stats,
            FilterStats {
                retained: 2,
                removed: 1
            }
        );
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "saddr,class,success,note\n::2,timeout,0, keep spaces \n::3,reply,1,last\n"
        );
    }
    #[test]
    fn plain_lists_keep_comments_and_order() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "# hosts\n::1\n\n::2\n::3").unwrap();
        let prefixes = ["::2/127".parse().unwrap()].into_iter().collect();
        let mut output = Vec::new();
        filter(file.path(), &mut output, &prefixes).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "# hosts\n::1\n\n");
    }
}
