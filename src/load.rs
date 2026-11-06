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
