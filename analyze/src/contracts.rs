pub trait PluginInfo {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
}

pub trait Predicate: PluginInfo + Send + Sync {
    type In;

    fn predicate(&self, x: Self::In) -> bool;
}
