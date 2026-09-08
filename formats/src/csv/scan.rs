use std::net::Ipv6Addr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeClass {
    Reply,
    Timeout,
    Unreachable,
    Error,
}

impl OutcomeClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reply => "reply",
            Self::Timeout => "timeout",
            Self::Unreachable => "unreachable",
            Self::Error => "error",
        }
    }

    fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "reply" => Some(Self::Reply),
            "timeout" => Some(Self::Timeout),
            "unreachable" => Some(Self::Unreachable),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanRecord {
    pub target: Ipv6Addr,
    pub responder: Option<Ipv6Addr>,
    pub class: OutcomeClass,
    pub success: bool,
    pub rtt_ms: Option<u64>,
    pub icmp_type: Option<u8>,
    pub icmp_code: Option<u8>,
    pub router: Option<Ipv6Addr>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScanRecordSchema {
    saddr_idx: usize,
    responder_idx: Option<usize>,
    class_idx: Option<usize>,
    success_idx: Option<usize>,
    rtt_ms_idx: Option<usize>,
    icmp_type_idx: Option<usize>,
    icmp_code_idx: Option<usize>,
    router_idx: Option<usize>,
}

impl ScanRecord {
    pub const COLUMNS: [&'static str; 8] = [
        "saddr",
        "responder",
        "class",
        "success",
        "rtt_ms",
        "icmp_type",
        "icmp_code",
        "router",
    ];

    pub fn schema(headers: &csv::StringRecord) -> Result<ScanRecordSchema, String> {
        let saddr_idx = headers
            .iter()
            .position(|header| header.eq_ignore_ascii_case("saddr"))
            .ok_or_else(|| "scan record header missing 'saddr'".to_string())?;
        let class_idx = headers.iter().position(|header| {
            header.eq_ignore_ascii_case("class") || header.eq_ignore_ascii_case("classification")
        });
        let success_idx = headers
            .iter()
            .position(|header| header.eq_ignore_ascii_case("success"));
        if class_idx.is_none() && success_idx.is_none() {
            return Err("scan record header missing 'class' or 'success'".to_string());
        }

        Ok(ScanRecordSchema {
            saddr_idx,
            responder_idx: headers.iter().position(|header| {
                header.eq_ignore_ascii_case("responder")
                    || header.eq_ignore_ascii_case("reply_addr")
            }),
            class_idx,
            success_idx,
            rtt_ms_idx: headers
                .iter()
                .position(|header| header.eq_ignore_ascii_case("rtt_ms")),
            icmp_type_idx: headers
                .iter()
                .position(|header| header.eq_ignore_ascii_case("icmp_type")),
            icmp_code_idx: headers
                .iter()
                .position(|header| header.eq_ignore_ascii_case("icmp_code")),
            router_idx: headers.iter().position(|header| {
                header.eq_ignore_ascii_case("router") || header.eq_ignore_ascii_case("outersaddr")
            }),
        })
    }

    pub fn from_csv_record(
        schema: &ScanRecordSchema,
        record: &csv::StringRecord,
    ) -> Result<Self, String> {
        let target = record
            .get(schema.saddr_idx)
            .ok_or_else(|| "scan record row missing saddr".to_string())?
            .trim()
            .parse::<Ipv6Addr>()
            .map_err(|e| format!("invalid IPv6 saddr: {e}"))?;

        let success = schema
            .success_idx
            .and_then(|idx| record.get(idx))
            .filter(|value| !value.trim().is_empty())
            .map(|value| {
                parse_bool_like(value).ok_or_else(|| format!("invalid success value: {value}"))
            })
            .transpose()?;
        let class = schema
            .class_idx
            .and_then(|idx| record.get(idx))
            .filter(|value| !value.trim().is_empty())
            .map(|value| {
                OutcomeClass::parse(value).ok_or_else(|| format!("invalid outcome class: {value}"))
            })
            .transpose()?
            .or_else(|| {
                success.map(|ok| {
                    if ok {
                        OutcomeClass::Reply
                    } else {
                        OutcomeClass::Error
                    }
                })
            })
            .ok_or_else(|| "scan record row missing class/success".to_string())?;
        let success = success.unwrap_or(matches!(class, OutcomeClass::Reply));
        if success != matches!(class, OutcomeClass::Reply) {
            return Err("scan outcome class conflicts with success".into());
        }

        Ok(Self {
            target,
            responder: optional_ipv6(schema.responder_idx.and_then(|idx| record.get(idx)))?,
            class,
            success,
            rtt_ms: optional_u64(schema.rtt_ms_idx.and_then(|idx| record.get(idx)))?,
            icmp_type: optional_u8(schema.icmp_type_idx.and_then(|idx| record.get(idx)))?,
            icmp_code: optional_u8(schema.icmp_code_idx.and_then(|idx| record.get(idx)))?,
            router: optional_ipv6(schema.router_idx.and_then(|idx| record.get(idx)))?,
        })
    }

    pub fn csv_row(&self, columns: &[String]) -> Vec<String> {
        columns.iter().map(|column| self.value(column)).collect()
    }

    pub fn value(&self, column: &str) -> String {
        let key = column.to_ascii_lowercase();
        match key.as_str() {
            "saddr" => self.target.to_string(),
            "responder" | "reply_addr" => {
                self.responder.map(|ip| ip.to_string()).unwrap_or_default()
            }
            "success" => {
                if self.success {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            "classification" | "class" => self.class.as_str().to_string(),
            "rtt_ms" => self.rtt_ms.map(|v| v.to_string()).unwrap_or_default(),
            "icmp_type" => self.icmp_type.map(|v| v.to_string()).unwrap_or_default(),
            "icmp_code" => self.icmp_code.map(|v| v.to_string()).unwrap_or_default(),
            "router" | "outersaddr" => self.router.map(|ip| ip.to_string()).unwrap_or_default(),
            _ => String::new(),
        }
    }
}

fn parse_bool_like(input: &str) -> Option<bool> {
    match input.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "reply" | "active" => Some(true),
        "0" | "false" | "no" | "n" | "inactive" | "error" | "timeout" | "unreachable" => {
            Some(false)
        }
        _ => None,
    }
}

fn optional_ipv6(value: Option<&str>) -> Result<Option<Ipv6Addr>, String> {
    match value.map(str::trim) {
        Some("") | None => Ok(None),
        Some(value) => value
            .parse::<Ipv6Addr>()
            .map(Some)
            .map_err(|e| format!("invalid IPv6 address '{value}': {e}")),
    }
}

fn optional_u64(value: Option<&str>) -> Result<Option<u64>, String> {
    match value.map(str::trim) {
        Some("") | None => Ok(None),
        Some(value) => value
            .parse::<u64>()
            .map(Some)
            .map_err(|e| format!("invalid integer '{value}': {e}")),
    }
}

fn optional_u8(value: Option<&str>) -> Result<Option<u8>, String> {
    match value.map(str::trim) {
        Some("") | None => Ok(None),
        Some(value) => value
            .parse::<u8>()
            .map(Some)
            .map_err(|e| format!("invalid integer '{value}': {e}")),
    }
}
