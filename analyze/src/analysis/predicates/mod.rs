pub mod documentation;
pub mod eui64;
pub mod multicast;
pub mod protocols;
pub mod reserved;
pub mod special;
pub mod special_purpose;
pub mod transition;

use crate::contracts::Predicate;
use std::net::Ipv6Addr;

pub type PredicateFn = fn(Ipv6Addr) -> bool;
pub type NamedPredicate = (&'static str, PredicateFn);

macro_rules! predicates {
    ($($variant:ident: ($name:literal, $predicate:path);)*) => {
        #[derive(Debug, Clone, Copy, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
        pub enum AddressPredicate { $(#[value(name = $name)] #[serde(rename = $name)] $variant,)* }
        impl AddressPredicate {
            pub fn test(self, address: Ipv6Addr) -> bool { match self { $(Self::$variant => $predicate.predicate(address),)* } }
            pub fn to_filter_name(self) -> String { match self { $(Self::$variant => $name.into(),)* } }
        }
        pub fn get_all_predicates() -> Vec<NamedPredicate> { vec![$(($name, |addr| $predicate.predicate(addr)),)*] }
    };
}
predicates! {
    Loopback: ("loopback", reserved::LoopbackPredicate);
    Unspecified: ("unspecified", reserved::UnspecifiedPredicate);
    LinkLocal: ("link_local", reserved::LinkLocalPredicate);
    UniqueLocal: ("unique_local", reserved::UniqueLocalPredicate);
    Multicast: ("multicast", multicast::IsMulticastPredicate);
    SolicitedNode: ("solicited_node", multicast::SolicitedNodeMulticastPredicate);
    Ipv4Mapped: ("ipv4_mapped", transition::Ipv4MappedPredicate);
    Ipv4ToIpv6: ("ipv4_to_ipv6", transition::Ipv4ToIpv6Predicate);
    ExtendedIpv4: ("extended_ipv4", transition::ExtendedIpv4Ipv6Predicate);
    Ipv6ToIpv4: ("ipv6_to_ipv4", transition::Ipv6ToIpv4Predicate);
    Documentation: ("documentation", documentation::DocumentationPredicate);
    Documentation2: ("documentation_2", documentation::Documentation2Predicate);
    Benchmarking: ("benchmarking", documentation::BenchmarkingPredicate);
    Teredo: ("teredo", protocols::TeredoPredicate);
    IetfProtocol: ("ietf_protocol", protocols::IetfProtocolPredicate);
    PortControl: ("port_control", protocols::PortControlProtocolPredicate);
    Turn: ("turn", protocols::TurnPredicate);
    DnsSd: ("dns_sd", protocols::DnsSdPredicate);
    Amt: ("amt", protocols::AmtPredicate);
    SegmentRouting: ("segment_routing", protocols::SegmentRoutingPredicate);
    DiscardOnly: ("discard_only", special_purpose::DiscardOnlyPredicate);
    DummyPrefix: ("dummy_prefix", special_purpose::DummyPrefixPredicate);
    As112V6: ("as112_v6", special_purpose::As112V6Predicate);
    DirectAs112: ("direct_as112", special_purpose::DirectAs112Predicate);
    DeprecatedOrchid: ("deprecated_orchid", special_purpose::DeprecatedOrchidPredicate);
    OrchidV2: ("orchid_v2", special_purpose::OrchidV2Predicate);
    DroneRemoteId: ("drone_remote_id", special_purpose::DroneRemoteIdPredicate);
    Eui64: ("eui64", eui64::Eui64Analysis);
    LowByteHost: ("low_byte_host", eui64::IsLowByteHostPredicate);
}
