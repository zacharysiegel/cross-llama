use futures::channel::mpsc::UnboundedSender;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

use crate::wllama::abortable_operation::{AbortableOperation, OperationEvent};
use crate::wllama::model_lock::ModelLock;
use crate::wllama::wllama::ModelSource;
use crate::wllama::wllama_bridge;
use crate::wllama::wllama_bridge::{ModelManager, WllamaModelFiles};
use crate::{Error, LoadProgress};

#[derive(Clone, Debug, PartialEq)]
pub struct DownloadProgress {
    pub downloaded_byte_count: u64,
    /// Absent when the server does not send the file's length.
    pub total_byte_count: Option<u64>,
}

impl From<DownloadProgress> for LoadProgress {
    fn from(download_progress: DownloadProgress) -> Self {
        LoadProgress::Downloading {
            downloaded_byte_count: download_progress.downloaded_byte_count,
            total_byte_count: download_progress.total_byte_count,
        }
    }
}

/// Downloads into wllama's cache in the browser's storage, where `Model::load` finds it, without loading it into memory.
pub async fn download_model(source: &ModelSource, mut on_progress: impl FnMut(DownloadProgress)) -> Result<(), Error> {
    let model_manager: ModelManager = create_model_manager(&source.wllama_directory_url).await?;
    let model_lock: ModelLock = ModelLock::acquire(&source.model_url).await?;

    let _model_files: WllamaModelFiles = download_model_files(&model_manager, &model_lock, &mut on_progress).await?;
    model_lock.release();

    Ok(())
}

/// Reads only the browser's storage.
pub async fn is_model_downloaded(source: &ModelSource) -> Result<bool, Error> {
    let model_manager: ModelManager = create_model_manager(&source.wllama_directory_url).await?;
    let downloaded: JsValue = wllama_bridge::settle(wllama_bridge::is_model_downloaded(&model_manager, &source.model_url)).await?;

    downloaded
        .as_bool()
        .ok_or_else(|| Error::new("wllama did not say whether the model is downloaded"))
}

/// Finds the model's files in the cache, or downloads them there.
pub(crate) async fn download_model_files(
    model_manager: &ModelManager,
    model_lock: &ModelLock,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<WllamaModelFiles, Error> {
    let mut download: AbortableOperation<DownloadProgress, dyn FnMut(f64, f64)> = AbortableOperation::start(
        |progress_sender: UnboundedSender<DownloadProgress>| {
            Closure::new(move |downloaded_byte_count: f64, total_byte_count: f64| {
                let progress: DownloadProgress = create_download_progress(downloaded_byte_count, total_byte_count);
                // Fails only once the download has stopped waiting.
                let _ = progress_sender.unbounded_send(progress);
            })
        },
        |abort_signal, on_progress| wllama_bridge::download_model(model_manager, model_lock.model_url(), abort_signal, on_progress),
    )?;

    loop {
        let event: OperationEvent<DownloadProgress> = download.next().await;

        match event {
            OperationEvent::Event(progress) => on_progress(progress),
            OperationEvent::Settled(settlement) => return settlement.map(|model_files: JsValue| model_files.unchecked_into::<WllamaModelFiles>()),
        }
    }
}

async fn create_model_manager(wllama_directory_url: &str) -> Result<ModelManager, Error> {
    let model_manager: JsValue = wllama_bridge::settle(wllama_bridge::create_model_manager(wllama_directory_url)).await?;
    Ok(model_manager.unchecked_into::<ModelManager>())
}

fn create_download_progress(downloaded_byte_count: f64, total_byte_count: f64) -> DownloadProgress {
    // JavaScript numbers hold byte counts exactly up to 2^53, which no model file reaches.
    DownloadProgress {
        downloaded_byte_count: downloaded_byte_count as u64,
        total_byte_count: (total_byte_count > 0.0).then_some(total_byte_count as u64),
    }
}

#[cfg(test)]
mod tests {
    use futures::future;
    use futures::future::Either;
    use js_sys::Promise;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use wasm_bindgen_test::wasm_bindgen_test;

    use crate::Error;
    use crate::wllama::model_cache;
    use crate::wllama::model_cache::DownloadProgress;
    use crate::wllama::model_lock::ModelLock;
    use crate::wllama::wllama::ModelSource;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    // wasm-bindgen-test's runner serves the crate root, and sends small files with their length, which wllama's cache checks.
    const SMALL_MODEL_URL: &str = "/tests/data/empty.gguf";
    const WLLAMA_DIRECTORY_URL: &str = "/build/wllama/";
    const LOCK_WAIT_MILLISECONDS: u32 = 500;

    #[wasm_bindgen(inline_js = "export function sleep(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }")]
    extern "C" {
        fn sleep(milliseconds: u32) -> Promise;
    }

    #[wasm_bindgen_test]
    async fn download_model_waits_while_model_lock_is_held_elsewhere() {
        // The query keeps this test's cached copy apart from other tests'; the runner ignores it.
        let source: ModelSource = ModelSource {
            model_url: format!("{SMALL_MODEL_URL}?test=model-lock-{}", js_sys::Date::now()),
            wllama_directory_url: WLLAMA_DIRECTORY_URL.to_string(),
        };
        let model_lock: ModelLock = ModelLock::acquire(&source.model_url).await.unwrap();
        let download = Box::pin(model_cache::download_model(&source, |_: DownloadProgress| {}));

        let first_finished = future::select(download, JsFuture::from(sleep(LOCK_WAIT_MILLISECONDS))).await;
        let Either::Right((_, waiting_download)) = first_finished
        else {
            panic!("the download finished while another holder had the model's lock");
        };
        model_lock.release();
        let downloaded: Result<(), Error> = waiting_download.await;

        assert!(downloaded.is_ok());
    }
}
