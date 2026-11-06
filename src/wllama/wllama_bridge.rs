use js_sys::{Array, Function, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::AbortSignal;

use crate::Error;

#[wasm_bindgen(module = "/src/wllama/wllama_bridge.js")]
extern "C" {
    /// A wllama instance, which runs at most one model.
    pub type Wllama;

    /// Keeps model files in wllama's cache in the browser's storage.
    pub type ModelManager;

    /// A model's files in wllama's cache.
    pub type WllamaModelFiles;

    #[wasm_bindgen(method, getter, js_name = modelManager)]
    pub fn model_manager(this: &Wllama) -> ModelManager;

    #[wasm_bindgen(js_name = createWllama)]
    pub fn create_wllama(wllama_directory_url: &str) -> Promise;

    #[wasm_bindgen(js_name = createModelManager)]
    pub fn create_model_manager(wllama_directory_url: &str) -> Promise;

    /// Calls `on_progress` with the bytes downloaded and the total, which is zero when the server does not give it.
    #[wasm_bindgen(js_name = downloadModel)]
    pub fn download_model(model_manager: &ModelManager, model_url: &str, abort_signal: &AbortSignal, on_progress: &Function) -> Promise;

    /// Resolves to a boolean.
    #[wasm_bindgen(js_name = isModelDownloaded)]
    pub fn is_model_downloaded(model_manager: &ModelManager, model_url: &str) -> Promise;

    /// Resolves once the lock is held; aborting `abort_signal` releases it once every promise in `held_operations` has
    /// settled.
    #[wasm_bindgen(js_name = acquireModelLock)]
    pub fn acquire_model_lock(model_url: &str, abort_signal: &AbortSignal, held_operations: &Array) -> Promise;

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

    /// Resolves to the capability as JSON.
    #[wasm_bindgen(js_name = probeBrowserCapability)]
    pub fn probe_browser_capability() -> Promise;
}

pub async fn settle(promise: Promise) -> Result<JsValue, Error> {
    let settled: Result<JsValue, JsValue> = JsFuture::from(promise).await;
    settled.map_err(|javascript_error: JsValue| create_error(&javascript_error))
}

pub async fn settle_wllama_call(promise: Promise) -> Result<JsValue, Error> {
    let settled: Result<JsValue, JsValue> = JsFuture::from(promise).await;
    settled.map_err(|javascript_error: JsValue| create_wllama_error(&javascript_error))
}

pub fn create_error(javascript_error: &JsValue) -> Error {
    Error::new(&read_error_message(javascript_error))
}

pub fn create_wllama_error(javascript_error: &JsValue) -> Error {
    Error::new(&format!("wllama failed: {}", read_error_message(javascript_error)))
}

fn read_error_message(javascript_error: &JsValue) -> String {
    match javascript_error.dyn_ref::<js_sys::Error>() {
        Some(error_object) => String::from(error_object.message()),
        None => javascript_error.as_string().unwrap_or_else(|| format!("{javascript_error:?}")),
    }
}
