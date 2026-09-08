use crate::csv::scan::{OutcomeClass, ScanRecord};
use sixseven_core::{Feedback, Observation};
use std::io::{BufRead, Write};
use std::path::Path;

pub fn load_batches(path: impl AsRef<Path>) -> Result<Vec<Vec<Feedback>>, crate::Error> {
    let path = path.as_ref();
    if path
        .extension()
        .is_some_and(|extension| extension == "jsonl")
    {
        std::io::BufReader::new(std::fs::File::open(path)?)
            .lines()
            .filter_map(|line| match line {
                Ok(line) if line.trim().is_empty() => None,
                other => Some(other),
            })
            .map(|line| Ok(serde_json::from_str(&line?)?))
            .collect()
    } else {
        Ok(vec![load(path)?])
    }
}

pub fn write_batch(writer: &mut impl Write, feedback: &[Feedback]) -> Result<(), crate::Error> {
    serde_json::to_writer(&mut *writer, feedback)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> Result<Vec<Feedback>, crate::Error> {
    let mut reader = csv::Reader::from_path(path)?;
    let schema = ScanRecord::schema(reader.headers()?).map_err(crate::Error::Invalid)?;
    let mut outcomes = std::collections::BTreeMap::new();
    for row in reader.records() {
        let record = ScanRecord::from_csv_record(&schema, &row?).map_err(crate::Error::Invalid)?;
        let rank = match record.class {
            OutcomeClass::Reply => 2,
            OutcomeClass::Timeout | OutcomeClass::Unreachable => 1,
            OutcomeClass::Error => 0,
        };
        outcomes
            .entry(record.target)
            .and_modify(|previous: &mut u8| *previous = (*previous).max(rank))
            .or_insert(rank);
    }
    let mut feedback: Vec<_> = outcomes
        .into_iter()
        .map(|(target, rank)| match rank {
            2 => Feedback::Active(target),
            1 => Feedback::Inactive(target),
            _ => Feedback::Skipped(target),
        })
        .collect();
    feedback.push(Feedback::BatchComplete);
    Ok(feedback)
}

pub fn observations(path: impl AsRef<Path>) -> Result<Vec<Observation>, crate::Error> {
    Ok(load(path)?
        .into_iter()
        .filter_map(|item| match item {
            Feedback::Active(ip) => Some(Observation {
                address: ip.octets(),
                active: true,
            }),
            Feedback::Inactive(ip) => Some(Observation {
                address: ip.octets(),
                active: false,
            }),
            _ => None,
        })
        .collect())
}
