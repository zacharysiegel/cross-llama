use js_sys::{Array, Promise};
use web_sys::AbortController;

use crate::Error;
use crate::wllama::wllama_bridge;

/// A Web Lock on a model's URL, shared by every page of the origin, so only one writes or loads the model's cached files
/// at a time.
pub struct ModelLock {
    model_url: String,
    // Aborting abandons a pending request or releases a held lock.
    abort_controller: AbortController,
    // Promises the release waits for.
    held_operations: Array,
}

impl ModelLock {
    /// Waits for every other holder to release it.
    pub async fn acquire(model_url: &str) -> Result<Self, Error> {
        let abort_controller: AbortController = AbortController::new().map_err(|javascript_error| wllama_bridge::create_error(&javascript_error))?;
        let model_lock: ModelLock = ModelLock {
            model_url: model_url.to_string(),
            abort_controller,
            held_operations: Array::new(),
        };

        let lock_request: Promise = wllama_bridge::acquire_model_lock(model_url, &model_lock.abort_controller.signal(), &model_lock.held_operations);
        wllama_bridge::settle(lock_request).await?;

        Ok(model_lock)
    }

    pub fn model_url(&self) -> &str {
        &self.model_url
    }

    /// Keeps the lock held, after a release or a drop, until `operation` settles.
    pub fn hold_until_settled(&self, operation: &Promise) {
        self.held_operations.push(operation);
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

#[cfg(test)]
mod tests {
    use futures::future;
    use futures::future::Either;
    use wasm_bindgen_futures::JsFuture;
    use wasm_bindgen_test::wasm_bindgen_test;

    use crate::Error;
    use crate::wllama::model_lock::ModelLock;
    use crate::wllama::test_sleep;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    const HELD_OPERATION_MILLISECONDS: u32 = 1000;
    const LOCK_WAIT_MILLISECONDS: u32 = 500;

    #[wasm_bindgen_test]
    async fn release_keeps_lock_until_held_operation_settles() {
        let model_url: String = format!("/held-operation-{}.gguf", js_sys::Date::now());
        let model_lock: ModelLock = ModelLock::acquire(&model_url).await.unwrap();

        model_lock.hold_until_settled(&test_sleep::sleep(HELD_OPERATION_MILLISECONDS));
        model_lock.release();
        let next_acquire = Box::pin(ModelLock::acquire(&model_url));
        let first_finished = future::select(next_acquire, JsFuture::from(test_sleep::sleep(LOCK_WAIT_MILLISECONDS))).await;
        let Either::Right((_, waiting_acquire)) = first_finished
        else {
            panic!("the lock was acquired while the previous holder's operation was still running");
        };
        let next_model_lock: Result<ModelLock, Error> = waiting_acquire.await;

        assert!(next_model_lock.is_ok());
    }
}
