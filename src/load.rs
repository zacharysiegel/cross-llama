use std::num::NonZeroU32;

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlidingWindowCache {
    /// Lets a shared prompt prefix be reused across requests on models with sliding-window attention.
    Full,
    WindowOnly,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadOptions {
    pub context_token_count: u32,
    pub sliding_window_cache: SlidingWindowCache,
}

impl LoadOptions {
    pub(crate) fn nonzero_context_token_count(&self) -> Result<NonZeroU32, Error> {
        NonZeroU32::new(self.context_token_count).ok_or_else(|| Error::new("the context must hold at least one token"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoadProgress {
    Downloading {
        downloaded_byte_count: u64,
        total_byte_count: Option<u64>,
    },
    Loading {
        loaded_proportion: f32,
    },
}
