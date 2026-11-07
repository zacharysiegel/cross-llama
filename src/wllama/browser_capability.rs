use serde::Deserialize;
use wasm_bindgen::JsValue;

use crate::Error;
use crate::wllama::wllama_bridge;

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebGpuAdapter {
    /// The adapter's `maxBufferSize` limit.
    pub maximum_buffer_byte_count: u64,
}

/// The wllama build the browser gets: the compatibility build when it lacks JSPI or WebAssembly 64-bit memory, except
/// on Firefox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WllamaBuildKind {
    Default,
    Compatibility,
}

/// The bridge's format, which stays out of the public API.
#[derive(Deserialize)]
struct BrowserCapabilityJson {
    webgpu_adapter: Option<WebGpuAdapterJson>,
    cross_origin_isolated: bool,
    wllama_build: WllamaBuildKindJson,
    storage_quota_byte_count: Option<u64>,
    storage_usage_byte_count: Option<u64>,
    storage_persisted: bool,
    device_memory_gigabytes: Option<f64>,
}

impl From<BrowserCapabilityJson> for BrowserCapability {
    fn from(capability_json: BrowserCapabilityJson) -> Self {
        BrowserCapability {
            webgpu_adapter: capability_json.webgpu_adapter.map(WebGpuAdapter::from),
            cross_origin_isolated: capability_json.cross_origin_isolated,
            wllama_build: WllamaBuildKind::from(capability_json.wllama_build),
            storage_quota_byte_count: capability_json.storage_quota_byte_count,
            storage_usage_byte_count: capability_json.storage_usage_byte_count,
            storage_persisted: capability_json.storage_persisted,
            device_memory_gigabytes: capability_json.device_memory_gigabytes,
        }
    }
}

#[derive(Deserialize)]
struct WebGpuAdapterJson {
    maximum_buffer_byte_count: u64,
}

impl From<WebGpuAdapterJson> for WebGpuAdapter {
    fn from(adapter_json: WebGpuAdapterJson) -> Self {
        WebGpuAdapter {
            maximum_buffer_byte_count: adapter_json.maximum_buffer_byte_count,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WllamaBuildKindJson {
    Default,
    Compatibility,
}

impl From<WllamaBuildKindJson> for WllamaBuildKind {
    fn from(build_json: WllamaBuildKindJson) -> Self {
        match build_json {
            WllamaBuildKindJson::Default => WllamaBuildKind::Default,
            WllamaBuildKindJson::Compatibility => WllamaBuildKind::Compatibility,
        }
    }
}

pub async fn probe_browser_capability() -> Result<BrowserCapability, Error> {
    let capability_json: JsValue = wllama_bridge::settle(wllama_bridge::probe_browser_capability()).await?;
    let capability_text: String = capability_json
        .as_string()
        .ok_or_else(|| Error::new("the browser capability probe returned no description"))?;

    let bridge_capability: BrowserCapabilityJson = serde_json::from_str(&capability_text)?;
    Ok(BrowserCapability::from(bridge_capability))
}
