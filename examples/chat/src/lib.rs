//! The chat demo's core, which every client shares.

pub mod chat;
mod error;
// reads the local filesystem, which a browser page cannot
#[cfg(not(target_arch = "wasm32"))]
pub mod model_file;

pub use error::*;
