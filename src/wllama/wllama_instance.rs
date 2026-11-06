use wasm_bindgen::JsValue;

use crate::Error;
use crate::wllama::wllama_bridge;
use crate::wllama::wllama_bridge::Wllama;

/// Exits when dropped, which frees its model.
pub struct WllamaInstance {
    wllama: Wllama,
}

impl WllamaInstance {
    pub fn new(wllama: Wllama) -> Self {
        WllamaInstance {
            wllama,
        }
    }

    pub fn wllama(&self) -> &Wllama {
        &self.wllama
    }

    /// wllama's exit does nothing once it has run, so the exit on drop that follows is harmless.
    pub async fn exit(&self) -> Result<(), Error> {
        let _exited: JsValue = wllama_bridge::settle(wllama_bridge::exit(&self.wllama)).await?;
        Ok(())
    }
}

impl Drop for WllamaInstance {
    fn drop(&mut self) {
        // A destructor cannot wait for the exit or report its failure.
        let _ = wllama_bridge::exit(&self.wllama);
    }
}
