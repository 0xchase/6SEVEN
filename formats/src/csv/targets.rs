use std::net::Ipv6Addr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetRecord {
    pub address: Ipv6Addr,
}

#[derive(Debug, Clone, Copy)]
pub struct TargetRecordSchema {
    saddr_idx: usize,
}

impl TargetRecord {
    pub const COLUMNS: [&'static str; 1] = ["saddr"];

    pub fn schema(headers: &csv::StringRecord) -> Result<TargetRecordSchema, String> {
        let saddr_idx = headers
            .iter()
            .position(|header| header.eq_ignore_ascii_case("saddr"))
            .ok_or_else(|| "target record header missing 'saddr'".to_string())?;
        Ok(TargetRecordSchema { saddr_idx })
    }

    pub fn from_csv_record(
        schema: &TargetRecordSchema,
        record: &csv::StringRecord,
    ) -> Result<Self, String> {
        let address = record
            .get(schema.saddr_idx)
            .ok_or_else(|| "target record row missing saddr".to_string())?
            .trim()
            .parse::<Ipv6Addr>()
            .map_err(|e| format!("invalid IPv6 saddr: {e}"))?;
        Ok(Self { address })
    }

    pub fn csv_row(&self, columns: &[String]) -> Vec<String> {
        columns.iter().map(|column| self.value(column)).collect()
    }

    pub fn value(&self, column: &str) -> String {
        match column.trim().to_ascii_lowercase().as_str() {
            "saddr" => self.address.to_string(),
            _ => String::new(),
        }
    }
}
