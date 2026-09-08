pub const MAX_INDEX_DOMAIN: usize = usize::MAX;

pub(crate) fn cap_index_size(size: u128) -> usize {
    size.min(MAX_INDEX_DOMAIN as u128) as usize
}
