//! Runs a GGUF language model with one API on every target.

mod error;
mod generation;
mod generation_checks;
// Only the browser backend reads the vocabulary from the model file; host tests cover it.
#[cfg(any(target_arch = "wasm32", test))]
mod gguf_vocabulary;
// llama-cpp-sys-2 compiles llama.cpp with CMake and a native C++ compiler, which cannot target wasm32.
#[cfg(not(target_arch = "wasm32"))]
mod llama_cpp;
mod load;
// wllama is JavaScript and needs a browser, which other targets lack.
#[cfg(target_arch = "wasm32")]
mod wllama;

pub use error::*;
pub use generation::*;
// llama-cpp-sys-2 compiles llama.cpp with CMake and a native C++ compiler, which cannot target wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub use llama_cpp::*;
pub use load::*;
// wllama is JavaScript and needs a browser, which other targets lack.
#[cfg(target_arch = "wasm32")]
pub use wllama::*;
