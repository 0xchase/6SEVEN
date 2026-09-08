use std::net::Ipv6Addr;

pub type TargetSource = Box<dyn Iterator<Item = Result<Ipv6Addr, crate::TgaError>> + Send>;

pub fn addresses<I>(values: I) -> TargetSource
where
    I: IntoIterator<Item = Ipv6Addr>,
    I::IntoIter: Send + 'static,
{
    Box::new(values.into_iter().map(Ok))
}
