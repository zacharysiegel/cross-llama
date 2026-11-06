use web_sys::AbortController;

use crate::Error;
use crate::wllama::wllama_bridge;

/// A Web Lock on a model's URL, shared by every page of the origin, so only one writes or loads the model's cached files
/// at a time.
pub struct ModelLock {
    model_url: String,
    // Aborting abandons a pending request or releases a held lock.
    abort_controller: AbortController,
}

impl ModelLock {
    /// Waits for every other holder to release it.
    pub async fn acquire(model_url: &str) -> Result<Self, Error> {
        let abort_controller: AbortController = AbortController::new().map_err(|javascript_error| wllama_bridge::create_error(&javascript_error))?;
        // Dropped while waiting, this abandons the request.
        let model_lock: ModelLock = ModelLock {
            model_url: model_url.to_string(),
            abort_controller,
        };

        wllama_bridge::settle(wllama_bridge::acquire_model_lock(model_url, &model_lock.abort_controller.signal())).await?;
        Ok(model_lock)
    }

    pub fn model_url(&self) -> &str {
        &self.model_url
    }

    pub fn release(self) {
        self.abort_controller.abort();
    }
}

impl Drop for ModelLock {
    fn drop(&mut self) {
        self.abort_controller.abort();
    }
}
