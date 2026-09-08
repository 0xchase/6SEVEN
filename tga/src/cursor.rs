use crate::{Address, Generated, GenerationState, TgaError};

pub(crate) struct GenerationCursor<I>(pub(crate) Option<I>);
impl<I> Default for GenerationCursor<I> {
    fn default() -> Self {
        Self(None)
    }
}
impl<I: Clone> Clone for GenerationCursor<I> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<I> std::fmt::Debug for GenerationCursor<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationCursor")
            .field("initialized", &self.0.is_some())
            .finish()
    }
}

pub(crate) trait Candidate {
    fn into_result(self) -> Result<Address, TgaError>;
}
impl Candidate for Address {
    fn into_result(self) -> Result<Address, TgaError> {
        Ok(self)
    }
}
impl Candidate for Result<Address, TgaError> {
    fn into_result(self) -> Self {
        self
    }
}

pub(crate) fn fill<I: Iterator>(
    mut iter: I,
    output: &mut [Address],
    terminal: GenerationState,
) -> Result<Generated, TgaError>
where
    I::Item: Candidate,
{
    sixseven_core::generation::validate_output(output)?;
    let mut written = 0;
    for slot in output.iter_mut() {
        let Some(address) = iter.next() else {
            break;
        };
        *slot = address.into_result()?;
        written += 1;
    }
    let state = if written < output.len() {
        terminal
    } else {
        GenerationState::Ready
    };
    Ok(Generated { written, state })
}
