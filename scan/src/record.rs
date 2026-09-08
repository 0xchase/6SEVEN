use probe::ReplyKind;
use sixseven_formats::csv::scan::{OutcomeClass, ScanRecord};

pub fn reply_record(reply: &probe::Reply) -> ScanRecord {
    let (class, success, icmp_type, icmp_code, router) = match reply.kind {
        ReplyKind::Direct => (OutcomeClass::Reply, true, None, None, None),
        ReplyKind::IcmpError {
            icmp_type,
            icmp_code,
        } => (
            outcome_class_for_icmp(icmp_type),
            false,
            Some(icmp_type),
            Some(icmp_code),
            Some(reply.responder),
        ),
    };

    ScanRecord {
        target: reply.target,
        responder: Some(reply.responder),
        class,
        success,
        rtt_ms: None,
        icmp_type,
        icmp_code,
        router,
    }
}

fn outcome_class_for_icmp(icmp_type: u8) -> OutcomeClass {
    match icmp_type {
        1 | 2 => OutcomeClass::Unreachable,
        3 => OutcomeClass::Timeout,
        _ => OutcomeClass::Error,
    }
}
