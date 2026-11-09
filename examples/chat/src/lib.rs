//! The chat demo's clients and the web server that serves one of them.

// cross-llama runs the model, which the web server does not
#[cfg(any(feature = "hydrate", feature = "terminal"))]
pub mod chat;
#[cfg(any(feature = "hydrate", feature = "ssr"))] // the page, which the terminal client lacks
pub mod chat_page;
#[cfg(feature = "hydrate")] // runs the model in the browser, which the server build cannot
mod browser_chat;
mod error;
pub mod model_file;
#[cfg(feature = "ssr")] // binds a socket and serves files, which the wasm32 build cannot
pub mod site_server;
#[cfg(any(feature = "hydrate", feature = "ssr"))] // the web client's URLs, which the terminal client lacks
pub mod site_paths;

pub use error::*;

#[cfg(feature = "hydrate")] // the wasm32 build's entry point, which cargo-leptos's page script calls
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(chat_page::ChatPage);
}
