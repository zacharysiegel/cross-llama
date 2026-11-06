use js_sys::{Function, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::AbortSignal;

use crate::Error;

#[wasm_bindgen(module = "/src/wllama/wllama_bridge.js")]
extern "C" {
    /// A wllama instance, which runs at most one model.
    pub type Wllama;

    /// A model's files in wllama's cache.
    pub type WllamaModelFiles;

    #[wasm_bindgen(js_name = createWllama)]
    pub fn create_wllama(wllama_directory_url: &str) -> Promise;

    /// Calls `on_progress` with the bytes downloaded and the total, which is zero when the server does not give it.
    #[wasm_bindgen(js_name = downloadModel)]
    pub fn download_model(wllama: &Wllama, model_url: &str, abort_signal: &AbortSignal, on_progress: &Function) -> Promise;

    /// Resolves to a `Uint8Array`, shorter than `byte_count` when the file is.
    #[wasm_bindgen(js_name = readModelPrefix)]
    pub fn read_model_prefix(model_files: &WllamaModelFiles, byte_count: u32) -> Promise;

    /// Resolves to the loaded context's details as JSON.
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(wllama: &Wllama, model_files: &WllamaModelFiles, load_parameters_json: &str) -> Promise;

    /// Calls `on_chunk` with each streamed chunk as JSON.
    #[wasm_bindgen(js_name = generate)]
    pub fn generate(wllama: &Wllama, request_json: &str, abort_signal: &AbortSignal, on_chunk: &Function) -> Promise;

    #[wasm_bindgen(js_name = exit)]
    pub fn exit(wllama: &Wllama) -> Promise;
}

pub async fn settle(promise: Promise) -> Result<JsValue, Error> {
    let settled: Result<JsValue, JsValue> = JsFuture::from(promise).await;
    settled.map_err(|javascript_error: JsValue| create_error(&javascript_error))
}

pub fn create_error(javascript_error: &JsValue) -> Error {
    let message: String = match javascript_error.dyn_ref::<js_sys::Error>() {
        Some(error_object) => String::from(error_object.message()),
        None => javascript_error.as_string().unwrap_or_else(|| format!("{javascript_error:?}")),
    };

    Error::new(&format!("wllama failed: {message}"))
}
