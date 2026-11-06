use serde::Deserialize;
use wasm_bindgen::JsValue;

use crate::Error;
use crate::wllama::wllama_bridge;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BrowserCapability {
    /// Absent when the browser has no WebGPU adapter, or only a fallback adapter, which runs on the CPU.
    pub webgpu_adapter: Option<WebGpuAdapter>,
    /// wllama's multithreaded build needs it.
    pub cross_origin_isolated: bool,
    pub wllama_build: WllamaBuildKind,
    /// Absent when the browser does not estimate its storage.
    pub storage_quota_byte_count: Option<u64>,
    /// Absent when the browser does not estimate its storage.
    pub storage_usage_byte_count: Option<u64>,
    /// The browser keeps persisted storage under storage pressure instead of evicting it.
    pub storage_persisted: bool,
    /// `navigator.deviceMemory`, which browsers round and cap; absent when the browser does not report it.
    pub device_memory_gigabytes: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct WebGpuAdapter {
    /// The adapter's `maxBufferSize` limit.
    pub maximum_buffer_byte_count: u64,
}

/// The wllama build the browser gets: the compatibility build when it lacks JSPI or WebAssembly 64-bit memory, except
/// on Firefox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WllamaBuildKind {
    Default,
    Compatibility,
}

pub async fn probe_browser_capability() -> Result<BrowserCapability, Error> {
    let capability_json: JsValue = wllama_bridge::settle(wllama_bridge::probe_browser_capability()).await?;
    let capability_text: String = capability_json
        .as_string()
        .ok_or_else(|| Error::new("the browser capability probe returned no description"))?;

    let capability: BrowserCapability = serde_json::from_str(&capability_text)?;
    Ok(capability)
}
