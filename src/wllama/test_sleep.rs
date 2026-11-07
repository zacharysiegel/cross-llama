use js_sys::Promise;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = "export function sleep(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }")]
extern "C" {
    pub fn sleep(milliseconds: u32) -> Promise;
}
