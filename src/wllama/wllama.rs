use std::cell::{Cell, RefCell};
use std::num::NonZeroU32;

use futures::channel::mpsc::UnboundedSender;
use futures::lock::{Mutex, MutexGuard};
use js_sys::{Promise, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::AbortController;

use crate::chat_template::{ChatTemplate, SequenceTokens};
use crate::gguf_vocabulary::{GgufVocabulary, VocabularyRead};
use crate::wllama::abortable_operation::{AbortableOperation, OperationEvent};
use crate::wllama::model_cache;
use crate::wllama::model_cache::DownloadProgress;
use crate::wllama::model_lock::ModelLock;
use crate::wllama::streamed_reply::StreamedReply;
use crate::wllama::wllama_bridge;
use crate::wllama::wllama_bridge::{ModelManager, Wllama, WllamaModelFiles};
use crate::wllama::wllama_instance::WllamaInstance;
use crate::{
    ChatMessage,
    Error,
    Generation,
    GenerationError,
    GenerationProgress,
    GenerationRequest,
    LoadOptions,
    LoadProgress,
    SlidingWindowCache,
    generation_checks,
    gguf_vocabulary,
};

// llama.cpp offloads at most every layer, however many more are asked for.
const EVERY_LAYER_ON_GPU: i32 = i32::MAX;
// One sequence, so every request can reuse the prefix the previous one left in the cache.
const PARALLEL_SEQUENCE_COUNT: u32 = 1;
// The first prompt progress report, which the context check needs, arrives only after the first batch is read.
const PROMPT_BATCH_TOKEN_COUNT: u32 = 512;
const INITIAL_VOCABULARY_PREFIX_BYTE_COUNT: u32 = 8 * 1024 * 1024;
const VOCABULARY_PREFIX_GROWTH_FACTOR: u32 = 4;

minimer::impl_from_error!(Error, serde_json::Error);

#[derive(Clone, Debug, PartialEq)]
pub struct ModelSource {
    pub model_url: String,
    /// The directory `scripts/fetch-wllama.sh` writes, served by the app.
    pub wllama_directory_url: String,
}

#[derive(Deserialize)]
struct LoadedContext {
    #[serde(rename = "n_ctx")]
    context_token_count: usize,
    #[serde(rename = "list_tokens_eog")]
    end_of_generation_token_ids: Vec<u32>,
    /// Negative when the vocabulary has none.
    #[serde(rename = "token_bos")]
    beginning_of_sequence_token_id: i64,
    /// Negative when the vocabulary has none.
    #[serde(rename = "token_eos")]
    end_of_sequence_token_id: i64,
    #[serde(rename = "add_bos_token")]
    tokenizer_adds_beginning_of_sequence: bool,
    #[serde(rename = "add_eos_token")]
    tokenizer_adds_end_of_sequence: bool,
    /// Absent when the model file has no chat template.
    #[serde(rename = "chat_template")]
    chat_template_source: Option<String>,
}

/// A model loaded into its own wllama instance, which holds the model's cache between requests.
pub struct Model {
    wllama_instance: WllamaInstance,
    context_token_count: usize,
    control_token_texts: Vec<String>,
    chat_template: ChatTemplate,
    generation_turn: Mutex<()>,
    cancellation_epoch: Cell<u64>,
    running_generation_abort_controller: RefCell<Option<AbortController>>,
}

impl Model {
    pub async fn load(source: ModelSource, load_options: &LoadOptions, mut on_progress: impl FnMut(LoadProgress)) -> Result<Model, Error> {
        let context_token_count: NonZeroU32 = load_options.nonzero_context_token_count()?;

        let wllama: JsValue = wllama_bridge::settle_wllama_call(wllama_bridge::create_wllama(&source.wllama_directory_url)).await?;
        // Exits on failure, or when this future is dropped.
        let wllama_instance: WllamaInstance = WllamaInstance::new(wllama.unchecked_into::<Wllama>());

        let model_lock: ModelLock = ModelLock::acquire(&source.model_url).await?;
        let model_manager: ModelManager = wllama_instance.wllama().model_manager();
        let model_files: WllamaModelFiles =
            model_cache::download_model_files(&source.wllama_directory_url, &model_manager, &model_lock, &mut |progress: DownloadProgress| {
                on_progress(LoadProgress::from(progress));
            })
            .await?;
        let vocabulary: GgufVocabulary = read_vocabulary(&model_files, &model_lock).await?;

        on_progress(LoadProgress::Loading {
            loaded_proportion: 0.0,
        });
        let loaded_context: LoadedContext = load_model(&wllama_instance, &model_files, &model_lock, context_token_count, load_options.sliding_window_cache).await?;
        model_lock.release();
        on_progress(LoadProgress::Loading {
            loaded_proportion: 1.0,
        });

        let chat_template: ChatTemplate = create_chat_template(&loaded_context, &vocabulary)?;

        Ok(Model {
            wllama_instance,
            context_token_count: loaded_context.context_token_count,
            control_token_texts: vocabulary.control_token_texts(&loaded_context.end_of_generation_token_ids),
            chat_template,
            generation_turn: Mutex::new(()),
            cancellation_epoch: Cell::new(0),
            running_generation_abort_controller: RefCell::new(None),
        })
    }

    /// Waits for any generation already running on this model to finish first.
    pub async fn generate(&self, request: &GenerationRequest, mut on_progress: impl FnMut(GenerationProgress)) -> Result<Generation, GenerationError> {
        let submitted_cancellation_epoch: u64 = self.cancellation_epoch.get();
        let _generation_turn: MutexGuard<()> = self.generation_turn.lock().await;

        self.check_cancellation(submitted_cancellation_epoch)?;
        check_no_control_token_text(&request.messages, &self.control_token_texts)?;

        let prompt: String = self.chat_template.render_prompt(&request.messages)?;
        let mut completion: AbortableOperation<String, dyn FnMut(String)> = self.start_completion(request, &prompt)?;
        let _running_generation: RunningGeneration = RunningGeneration::register(&self.running_generation_abort_controller, completion.abort_controller());
        let mut streamed_reply: StreamedReply = StreamedReply::new();

        loop {
            let event: OperationEvent<String> = completion.next().await;

            match event {
                OperationEvent::Event(chunk_json) => {
                    let progress_reports: Vec<GenerationProgress> = streamed_reply.apply_chunk(&chunk_json, request.maximum_token_count)?;
                    self.report_progress(progress_reports, request.maximum_token_count, submitted_cancellation_epoch, &mut on_progress)?;
                }
                OperationEvent::Settled(Ok(_)) => break,
                OperationEvent::Settled(Err(javascript_error)) => {
                    self.check_cancellation(submitted_cancellation_epoch)?;
                    return Err(GenerationError::Failed(wllama_bridge::create_wllama_error(&javascript_error)));
                }
            }
        }

        let generation: Generation = streamed_reply.into_generation(request)?;
        Ok(generation)
    }

    /// Stops the generation in progress, and any waiting for it, with `GenerationError::Cancelled`.
    pub fn cancel(&self) {
        self.cancellation_epoch.set(self.cancellation_epoch.get() + 1);

        if let Some(abort_controller) = self.running_generation_abort_controller.borrow().as_ref() {
            abort_controller.abort();
        }
    }

    /// Resolves once wllama has freed the model.
    pub async fn unload(self) -> Result<(), Error> {
        self.cancel();
        self.wllama_instance.exit().await
    }

    fn start_completion(&self, request: &GenerationRequest, prompt: &str) -> Result<AbortableOperation<String, dyn FnMut(String)>, Error> {
        let request_json: String = create_request_json(request, prompt);

        AbortableOperation::start(
            |chunk_sender: UnboundedSender<String>| {
                Closure::new(move |chunk_json: String| {
                    // Fails only once the generation has stopped waiting.
                    let _ = chunk_sender.unbounded_send(chunk_json);
                })
            },
            |abort_signal, on_chunk| wllama_bridge::generate(self.wllama_instance.wllama(), &request_json, abort_signal, on_chunk),
        )
    }

    /// Checks for cancellation after each report, since the caller may cancel from its progress callback.
    fn report_progress(
        &self,
        progress_reports: Vec<GenerationProgress>,
        maximum_token_count: usize,
        submitted_cancellation_epoch: u64,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<(), GenerationError> {
        for progress in progress_reports {
            if let GenerationProgress::ReadingPrompt { prompt_token_count, .. } = progress {
                generation_checks::check_context_capacity(prompt_token_count, maximum_token_count, self.context_token_count)?;
            }

            on_progress(progress);
            self.check_cancellation(submitted_cancellation_epoch)?;
        }

        Ok(())
    }

    fn check_cancellation(&self, submitted_cancellation_epoch: u64) -> Result<(), GenerationError> {
        if self.cancellation_epoch.get() != submitted_cancellation_epoch {
            return Err(GenerationError::Cancelled);
        }

        Ok(())
    }
}

/// Makes the running generation's abort controller reachable from `Model::cancel` while it runs.
struct RunningGeneration<'model> {
    running_generation_abort_controller: &'model RefCell<Option<AbortController>>,
}

impl<'model> RunningGeneration<'model> {
    fn register(running_generation_abort_controller: &'model RefCell<Option<AbortController>>, abort_controller: &AbortController) -> Self {
        running_generation_abort_controller.replace(Some(abort_controller.clone()));

        RunningGeneration {
            running_generation_abort_controller,
        }
    }
}

impl Drop for RunningGeneration<'_> {
    fn drop(&mut self) {
        self.running_generation_abort_controller.replace(None);
    }
}

async fn read_vocabulary(model_files: &WllamaModelFiles, model_lock: &ModelLock) -> Result<GgufVocabulary, Error> {
    let mut prefix_byte_count: u32 = INITIAL_VOCABULARY_PREFIX_BYTE_COUNT;

    loop {
        let prefix_read: Promise = wllama_bridge::read_model_prefix(model_files, prefix_byte_count);
        model_lock.hold_until_settled(&prefix_read);
        let prefix_array: JsValue = wllama_bridge::settle_wllama_call(prefix_read).await?;
        let prefix: Vec<u8> = Uint8Array::new(&prefix_array).to_vec();
        let vocabulary_read: VocabularyRead = gguf_vocabulary::read_vocabulary(&prefix)?;
        let whole_file_read: bool = prefix.len() < usize::try_from(prefix_byte_count)?;

        match vocabulary_read {
            VocabularyRead::Complete(vocabulary) => return Ok(vocabulary),
            VocabularyRead::NeedsLongerPrefix if whole_file_read => return Err(Error::new("the model file ends inside its metadata")),
            VocabularyRead::NeedsLongerPrefix => {
                prefix_byte_count = prefix_byte_count
                    .checked_mul(VOCABULARY_PREFIX_GROWTH_FACTOR)
                    .ok_or_else(|| Error::new("the model file's metadata is too large to read"))?;
            }
        }
    }
}

async fn load_model(
    wllama_instance: &WllamaInstance,
    model_files: &WllamaModelFiles,
    model_lock: &ModelLock,
    context_token_count: NonZeroU32,
    sliding_window_cache: SlidingWindowCache,
) -> Result<LoadedContext, Error> {
    let load_parameters: serde_json::Value = serde_json::json!({
        "n_ctx": context_token_count.get(),
        "n_gpu_layers": EVERY_LAYER_ON_GPU,
        "n_parallel": PARALLEL_SEQUENCE_COUNT,
        "swa_full": sliding_window_cache == SlidingWindowCache::Full,
        "n_batch": PROMPT_BATCH_TOKEN_COUNT,
    });
    let loading: Promise = wllama_bridge::load_model(wllama_instance.wllama(), model_files, &load_parameters.to_string());
    model_lock.hold_until_settled(&loading);
    let loaded_context_json: JsValue = wllama_bridge::settle_wllama_call(loading).await?;
    let loaded_context_text: String = loaded_context_json
        .as_string()
        .ok_or_else(|| Error::new("wllama did not describe the loaded context"))?;

    let loaded_context: LoadedContext = serde_json::from_str(&loaded_context_text)?;
    Ok(loaded_context)
}

/// Completes the prompt this crate templated, so both backends read the same prompt under the same schema grammar.
fn create_request_json(request: &GenerationRequest, prompt: &str) -> String {
    // Turns off every sampler but temperature, which the server and the model's metadata would otherwise enable.
    let mut request_json: serde_json::Value = serde_json::json!({
        "prompt": prompt,
        "temperature": request.temperature,
        "seed": request.seed,
        "max_tokens": request.maximum_token_count,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "xtc_probability": 0.0,
        "repeat_penalty": 1.0,
        "mirostat": 0,
    });

    if let Some(json_schema) = &request.json_schema {
        request_json["json_schema"] = json_schema.clone();
    }

    request_json.to_string()
}

fn create_chat_template(loaded_context: &LoadedContext, vocabulary: &GgufVocabulary) -> Result<ChatTemplate, Error> {
    let sequence_tokens: SequenceTokens = SequenceTokens {
        beginning_text: get_token_text(vocabulary, loaded_context.beginning_of_sequence_token_id),
        end_text: get_token_text(vocabulary, loaded_context.end_of_sequence_token_id),
        tokenizer_adds_beginning: loaded_context.tokenizer_adds_beginning_of_sequence,
        tokenizer_adds_end: loaded_context.tokenizer_adds_end_of_sequence,
    };

    ChatTemplate::new(loaded_context.chat_template_source.clone(), sequence_tokens)
}

fn get_token_text(vocabulary: &GgufVocabulary, token_id: i64) -> String {
    let token_text: Option<&str> = u32::try_from(token_id).ok().and_then(|token_id: u32| vocabulary.token_text(token_id));
    token_text.unwrap_or_default().to_string()
}

/// The llama.cpp server tokenizes the templated prompt with its control tokens parsed, so control-token text in a message would forge turns.
fn check_no_control_token_text(messages: &[ChatMessage], control_token_texts: &[String]) -> Result<(), Error> {
    for message in messages {
        let contains_control_token_text: bool = control_token_texts
            .iter()
            .any(|control_token_text| message.text.contains(control_token_text.as_str()));

        if contains_control_token_text {
            return Err(Error::new("a message's text contains one of the model's control tokens"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use js_sys::Promise;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    use crate::wllama::model_cache;
    use crate::wllama::model_cache::DownloadProgress;
    use crate::wllama::model_lock::ModelLock;
    use crate::wllama::wllama::{Model, ModelSource};
    use crate::wllama::wllama_bridge;
    use crate::wllama::wllama_bridge::{ModelManager, Wllama, WllamaModelFiles};
    use crate::wllama::wllama_instance::WllamaInstance;
    use crate::{Error, LoadOptions, LoadProgress, SlidingWindowCache};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    // wasm-bindgen-test's runner serves the crate root alongside its own files.
    const WLLAMA_DIRECTORY_URL: &str = "/build/wllama/";
    const TEST_MODEL_URL: &str = "/build/models/gemma-3-270m-it-Q8_0.gguf";
    const CONTEXT_TOKEN_COUNT: u32 = 4096;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(method, js_name = isModelLoaded)]
        fn is_model_loaded(this: &Wllama) -> bool;
    }

    fn create_load_options() -> LoadOptions {
        LoadOptions {
            context_token_count: CONTEXT_TOKEN_COUNT,
            sliding_window_cache: SlidingWindowCache::Full,
        }
    }

    #[wasm_bindgen_test]
    async fn unload_frees_the_model_in_wllama() {
        let source: ModelSource = ModelSource {
            model_url: TEST_MODEL_URL.to_string(),
            wllama_directory_url: WLLAMA_DIRECTORY_URL.to_string(),
        };
        let model: Model = Model::load(source, &create_load_options(), |_: LoadProgress| {}).await.unwrap();
        let wllama_value: &JsValue = model.wllama_instance.wllama();
        let wllama: Wllama = wllama_value.clone().unchecked_into::<Wllama>();

        let unloaded: Result<(), Error> = model.unload().await;

        assert!(unloaded.is_ok());
        assert!(!wllama.is_model_loaded());
    }

    #[wasm_bindgen_test]
    async fn exit_frees_the_model_whose_load_was_pending() {
        let created_wllama: JsValue = wllama_bridge::settle_wllama_call(wllama_bridge::create_wllama(WLLAMA_DIRECTORY_URL)).await.unwrap();
        let wllama_instance: WllamaInstance = WllamaInstance::new(created_wllama.unchecked_into::<Wllama>());
        let model_lock: ModelLock = ModelLock::acquire(TEST_MODEL_URL).await.unwrap();
        let model_manager: ModelManager = wllama_instance.wllama().model_manager();
        let model_files: WllamaModelFiles = model_cache::download_model_files(WLLAMA_DIRECTORY_URL, &model_manager, &model_lock, &mut |_: DownloadProgress| {})
            .await
            .unwrap();
        let load_parameters: serde_json::Value = serde_json::json!({ "n_ctx": CONTEXT_TOKEN_COUNT });

        let loading: Promise = wllama_bridge::load_model(wllama_instance.wllama(), &model_files, &load_parameters.to_string());
        let exited: Result<(), Error> = wllama_instance.exit().await;
        let _: Result<JsValue, Error> = wllama_bridge::settle_wllama_call(loading).await;

        assert!(exited.is_ok());
        assert!(!wllama_instance.wllama().is_model_loaded());
    }
}
