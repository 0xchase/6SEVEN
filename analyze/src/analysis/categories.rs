//! Address categories that explain how a TGA might have found the target.

use std::net::Ipv6Addr;

pub type CategoryPredicate = fn(Ipv6Addr) -> bool;
pub type NamedCategory = (&'static str, CategoryPredicate);

/// IID is ::small (segments 4-6 are zero, segment 7 is a small value).
pub fn is_low_byte_host(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] == 0 && s[5] == 0 && s[6] == 0 && s[7] < 0xffff
}

/// IID has the form ::XXXX:YYYY (segments 4,5 are zero, but not low_byte_host).
pub fn is_sparse_iid(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] == 0 && s[5] == 0 && !(s[6] == 0 && s[7] < 0xffff)
}

/// Matches IPv4 transition IIDs whose first segment is 1 or 2.
pub fn is_ipv4_embedded(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] >= 1 && s[4] <= 2 && !(s[5] == 0 && s[6] == 0 && s[7] < 0xffff)
}

/// The first IID segment uses at most 12 bits.
pub fn is_small_iid_prefix(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] >= 3 && s[4] <= 0x0fff
}

/// IID has at most 2 non-zero segments (e.g. 8000::, feed::XXX).
pub fn is_iid_mostly_zero(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    let nonzero = (s[4] != 0) as u8 + (s[5] != 0) as u8 + (s[6] != 0) as u8 + (s[7] != 0) as u8;
    nonzero <= 2
}

/// IID segments 4-6 are all high-fill (>= 0xff00).
pub fn is_high_fill_iid(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] >= 0xff00 && s[5] >= 0xff00 && s[6] >= 0xff00
}

/// IID has a zero first segment but nonzero second (0:X:Y:Z pattern).
pub fn is_inner_sparse_iid(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[4] == 0 && s[5] > 0
}

/// Jio (Reliance Jio) network: 2401:4900::/32.
pub fn is_jio(addr: Ipv6Addr) -> bool {
    let s = addr.segments();
    s[0] == 0x2401 && s[1] == 0x4900
}

/// All defined categories in priority order.
pub fn get_categories() -> Vec<NamedCategory> {
    vec![
        ("low_byte_host", is_low_byte_host as fn(Ipv6Addr) -> bool),
        ("sparse_iid", is_sparse_iid),
        ("ipv4_embedded", is_ipv4_embedded),
        ("small_iid_prefix", is_small_iid_prefix),
        ("iid_mostly_zero", is_iid_mostly_zero),
        ("high_fill_iid", is_high_fill_iid),
        ("inner_sparse_iid", is_inner_sparse_iid),
        ("jio", is_jio),
    ]
}

/// All category names including the uncategorized fallback.
pub fn all_category_names() -> Vec<&'static str> {
    vec![
        "low_byte_host",
        "sparse_iid",
        "ipv4_embedded",
        "small_iid_prefix",
        "iid_mostly_zero",
        "high_fill_iid",
        "inner_sparse_iid",
        "jio",
        "uncategorized",
    ]
}

/// Classify a single address, returning the category name.
pub fn categorize(addr: Ipv6Addr) -> &'static str {
    for (name, pred) in get_categories() {
        if pred(addr) {
            return name;
        }
    }
    "uncategorized"
}
