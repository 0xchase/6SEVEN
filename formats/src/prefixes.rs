use crate::Error;
use sixseven_core::{Ipv6Prefix, PrefixSet};
use std::{
    io::{BufRead, Write},
    path::Path,
};

pub fn read(path: impl AsRef<Path>) -> Result<PrefixSet, Error> {
    let path = path.as_ref();
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut prefixes = PrefixSet::default();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let value = line.split('#').next().unwrap_or_default().trim();
        if value.is_empty() {
            continue;
        }
        let prefix: Ipv6Prefix = value.parse().map_err(|error| {
            Error::Invalid(format!("{}:{}: {error}", path.display(), index + 1))
        })?;
        prefixes.insert(prefix);
    }
    Ok(prefixes)
}

pub fn write(path: impl AsRef<Path>, prefixes: &PrefixSet) -> Result<(), Error> {
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    for prefix in prefixes.prefixes() {
        writeln!(writer, "{prefix}")?;
    }
    writer.flush()?;
    Ok(())
}
