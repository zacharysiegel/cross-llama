use std::cell::{Cell, RefCell};
use std::num::NonZeroU32;
use std::time::Duration;

use futures::channel::mpsc::UnboundedSender;
use futures::lock::{Mutex, MutexGuard};
use js_sys::{Promise, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::AbortController;

use crate::chat_template::{ChatTemplate, SequenceTokens};
use crate::generation_checks::ReplyEnding;
use crate::gguf_vocabulary::{GgufVocabulary, VocabularyRead};
use crate::thinking::{ModelThinking, ReasoningPlan};
use crate::wllama::abortable_operation::{AbortableOperation, OperationEvent};
use crate::wllama::model_cache;
use crate::wllama::model_cache::DownloadProgress;
use crate::wllama::model_lock::ModelLock;
use crate::wllama::streamed_completion::{Completion, CompletionProgress, StreamedCompletion};
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
    Reasoning,
    SlidingWindowCache,
    ThinkingCapability,
    generation_checks,
    gguf_vocabulary,
    thinking,
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
    model_thinking: ModelThinking,
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
        let model_thinking: ModelThinking = ModelThinking::new(&chat_template, |text: &str| vocabulary.contains_token_text(text));

        Ok(Model {
            wllama_instance,
            context_token_count: loaded_context.context_token_count,
            control_token_texts: vocabulary.control_token_texts(&loaded_context.end_of_generation_token_ids),
            chat_template,
            model_thinking,
            generation_turn: Mutex::new(()),
            cancellation_epoch: Cell::new(0),
            running_generation_abort_controller: RefCell::new(None),
        })
    }

    pub fn thinking_capability(&self) -> ThinkingCapability {
        self.model_thinking.capability()
    }

    /// Waits for any generation already running on this model to finish first.
    pub async fn generate(&self, request: &GenerationRequest, mut on_progress: impl FnMut(GenerationProgress)) -> Result<Generation, GenerationError> {
        let submitted_cancellation_epoch: u64 = self.cancellation_epoch.get();
        let _generation_turn: MutexGuard<()> = self.generation_turn.lock().await;

        self.check_cancellation(submitted_cancellation_epoch)?;

        let prompt: String = self.render_prompt(request)?;
        let reasoning_plan: Option<ReasoningPlan> = self.model_thinking.plan_reasoning(request, &prompt);
        let generation_turn: GenerationTurn = GenerationTurn {
            request,
            submitted_cancellation_epoch,
        };

        match reasoning_plan {
            Some(reasoning_plan) => self.generate_reasoning_then_answer(&generation_turn, &reasoning_plan, &mut on_progress).await,
            None => self.generate_answer_alone(&generation_turn, &prompt, &mut on_progress).await,
        }
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

    fn render_prompt(&self, request: &GenerationRequest) -> Result<String, Error> {
        check_no_control_token_text(&request.messages, &self.control_token_texts)?;
        self.model_thinking.check_request(request)?;

        self.chat_template.render_prompt(&request.messages, request.thinking, request.thinking_effort)
    }

    async fn generate_answer_alone(
        &self,
        generation_turn: &GenerationTurn<'_>,
        prompt: &str,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<Generation, GenerationError> {
        let request: &GenerationRequest = generation_turn.request;
        let completion_request: CompletionRequest = CompletionRequest {
            prompt,
            token_limit: request.maximum_token_count,
            json_schema: request.json_schema.as_ref(),
            stop_text: None,
        };
        let answer_phase: CompletionPhase = CompletionPhase {
            reply_part: ReplyPartKind::Answer {
                reasoning_token_count: 0,
            },
            prompt_progress_report: PromptProgressReportKind::Reported,
        };

        let answer: Completion = self.run_completion(generation_turn, &completion_request, &answer_phase, on_progress).await?;

        generation_checks::check_reply_completes_json_value(request, &answer.text, answer.ending)?;

        Ok(Generation {
            text: answer.text,
            reasoning: None,
            prompt_token_count: answer.prompt_token_count,
            reused_token_count: answer.reused_token_count,
            generated_token_count: answer.generated_token_count,
            prompt_reading_duration: answer.prompt_reading_duration,
            generating_duration: answer.generating_duration,
        })
    }

    /// Reads the prompt once: the reasoning continues it, and the answer continues the reasoning.
    async fn generate_reasoning_then_answer(
        &self,
        generation_turn: &GenerationTurn<'_>,
        reasoning_plan: &ReasoningPlan,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<Generation, GenerationError> {
        let request: &GenerationRequest = generation_turn.request;
        let reasoning_completion: Option<Completion> = self.run_reasoning(generation_turn, reasoning_plan, on_progress).await?;

        let reasoning_reply_text: &str = reasoning_completion.as_ref().map_or("", |completion: &Completion| completion.text.as_str());
        let reasoning: Reasoning = thinking::create_reasoning(
            reasoning_reply_text,
            reasoning_completion.as_ref().map_or(0, |completion: &Completion| completion.generated_token_count),
        );
        let answer_token_limit: usize = reasoning_plan.answer_token_limit(reasoning.token_count);
        let answer_prompt: String = reasoning_plan.create_answer_prompt(reasoning_reply_text);
        let answer_request: CompletionRequest = CompletionRequest {
            prompt: &answer_prompt,
            token_limit: answer_token_limit,
            json_schema: request.json_schema.as_ref(),
            stop_text: None,
        };
        let answer_reply_part: ReplyPartKind = ReplyPartKind::Answer {
            reasoning_token_count: reasoning.token_count,
        };

        let (prompt_reading, generating_duration, answer): (PromptReading, Duration, Completion) = match reasoning_completion {
            None => {
                let answer_phase: CompletionPhase = CompletionPhase {
                    reply_part: answer_reply_part,
                    prompt_progress_report: PromptProgressReportKind::Reported,
                };
                let answer: Completion = self.run_completion(generation_turn, &answer_request, &answer_phase, on_progress).await?;
                (PromptReading::from(&answer), answer.generating_duration, answer)
            }
            Some(reasoning_completion) if answer_token_limit == 0 => (
                PromptReading::from(&reasoning_completion),
                reasoning_completion.generating_duration,
                create_empty_answer_completion(),
            ),
            Some(reasoning_completion) => {
                let answer_phase: CompletionPhase = CompletionPhase {
                    reply_part: answer_reply_part,
                    prompt_progress_report: PromptProgressReportKind::Unreported,
                };
                let answer: Completion = self.run_completion(generation_turn, &answer_request, &answer_phase, on_progress).await?;
                let generating_duration: Duration = reasoning_completion.generating_duration + answer.prompt_reading_duration + answer.generating_duration;
                (PromptReading::from(&reasoning_completion), generating_duration, answer)
            }
        };

        generation_checks::check_reply_completes_json_value(request, &answer.text, answer.ending)?;

        Ok(Generation {
            text: answer.text,
            generated_token_count: reasoning.token_count + answer.generated_token_count,
            reasoning: Some(reasoning),
            prompt_token_count: prompt_reading.prompt_token_count,
            reused_token_count: prompt_reading.reused_token_count,
            prompt_reading_duration: prompt_reading.prompt_reading_duration,
            generating_duration,
        })
    }

    /// Ends at the model's end delimiter or end of turn, or at the token limit; `None` when the budget allows no
    /// reasoning, which leaves the prompt unread.
    async fn run_reasoning(
        &self,
        generation_turn: &GenerationTurn<'_>,
        reasoning_plan: &ReasoningPlan,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<Option<Completion>, GenerationError> {
        if reasoning_plan.reasoning_token_limit == 0 {
            return Ok(None);
        }

        let reasoning_request: CompletionRequest = CompletionRequest {
            prompt: &reasoning_plan.reasoning_prompt,
            token_limit: reasoning_plan.reasoning_token_limit,
            json_schema: None,
            stop_text: Some(reasoning_plan.end_delimiter()),
        };
        let reasoning_phase: CompletionPhase = CompletionPhase {
            reply_part: ReplyPartKind::Reasoning,
            prompt_progress_report: PromptProgressReportKind::Reported,
        };

        let reasoning_completion: Completion = self.run_completion(generation_turn, &reasoning_request, &reasoning_phase, on_progress).await?;
        Ok(Some(reasoning_completion))
    }

    async fn run_completion(
        &self,
        generation_turn: &GenerationTurn<'_>,
        completion_request: &CompletionRequest<'_>,
        completion_phase: &CompletionPhase,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<Completion, GenerationError> {
        let mut completion: AbortableOperation<String, dyn FnMut(String)> = self.start_completion(generation_turn.request, completion_request)?;
        let _running_generation: RunningGeneration = RunningGeneration::register(&self.running_generation_abort_controller, completion.abort_controller());
        let mut streamed_completion: StreamedCompletion = StreamedCompletion::new();

        loop {
            let event: OperationEvent<String> = completion.next().await;

            match event {
                OperationEvent::Event(chunk_json) => {
                    let progress_reports: Vec<CompletionProgress> = streamed_completion.apply_chunk(&chunk_json)?;
                    self.report_progress(generation_turn, progress_reports, completion_request.token_limit, completion_phase, on_progress)?;
                }
                OperationEvent::Settled(Ok(_)) => break,
                OperationEvent::Settled(Err(javascript_error)) => {
                    self.check_cancellation(generation_turn.submitted_cancellation_epoch)?;
                    return Err(GenerationError::Failed(wllama_bridge::create_wllama_error(&javascript_error)));
                }
            }
        }

        let finished_completion: Completion = streamed_completion.into_completion()?;
        Ok(finished_completion)
    }

    fn start_completion(&self, request: &GenerationRequest, completion_request: &CompletionRequest) -> Result<AbortableOperation<String, dyn FnMut(String)>, Error> {
        let request_json: String = create_request_json(request, completion_request);

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
        generation_turn: &GenerationTurn<'_>,
        progress_reports: Vec<CompletionProgress>,
        reply_token_limit: usize,
        completion_phase: &CompletionPhase,
        on_progress: &mut impl FnMut(GenerationProgress),
    ) -> Result<(), GenerationError> {
        for progress in progress_reports {
            if let CompletionProgress::ReadingPrompt { prompt_token_count, .. } = progress {
                generation_checks::check_context_capacity(prompt_token_count, reply_token_limit, self.context_token_count)?;
            }

            let generation_progress: Option<GenerationProgress> = completion_phase.create_generation_progress(progress, generation_turn.request.maximum_token_count);

            if let Some(generation_progress) = generation_progress {
                on_progress(generation_progress);
            }

            self.check_cancellation(generation_turn.submitted_cancellation_epoch)?;
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

struct GenerationTurn<'request> {
    request: &'request GenerationRequest,
    submitted_cancellation_epoch: u64,
}

struct CompletionRequest<'request> {
    prompt: &'request str,
    token_limit: usize,
    json_schema: Option<&'request serde_json::Value>,
    /// Ends the completion, without appearing in its text.
    stop_text: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplyPartKind {
    Reasoning,
    Answer {
        reasoning_token_count: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptProgressReportKind {
    Reported,
    /// The answer's prompt after the reasoning, which repeats the prompt already reported.
    Unreported,
}

struct CompletionPhase {
    reply_part: ReplyPartKind,
    prompt_progress_report: PromptProgressReportKind,
}

impl CompletionPhase {
    /// `None` for a report the phase does not pass on.
    fn create_generation_progress(&self, progress: CompletionProgress, maximum_token_count: usize) -> Option<GenerationProgress> {
        match (progress, self.reply_part, self.prompt_progress_report) {
            (CompletionProgress::ReadingPrompt { .. }, _, PromptProgressReportKind::Unreported) => None,
            (
                CompletionProgress::ReadingPrompt {
                    read_token_count,
                    prompt_token_count,
                    reused_token_count,
                },
                _,
                PromptProgressReportKind::Reported,
            ) => Some(GenerationProgress::ReadingPrompt {
                read_token_count,
                prompt_token_count,
                reused_token_count,
            }),
            (CompletionProgress::Generating { generated_token_count, text }, ReplyPartKind::Reasoning, _) => Some(GenerationProgress::GeneratingReasoning {
                generated_token_count,
                maximum_token_count,
                text,
            }),
            (CompletionProgress::Generating { generated_token_count, text }, ReplyPartKind::Answer { reasoning_token_count }, _) => {
                Some(GenerationProgress::GeneratingAnswer {
                    generated_token_count: reasoning_token_count + generated_token_count,
                    maximum_token_count,
                    text,
                })
            }
        }
    }
}

struct PromptReading {
    prompt_token_count: usize,
    reused_token_count: usize,
    prompt_reading_duration: Duration,
}

impl From<&Completion> for PromptReading {
    fn from(completion: &Completion) -> Self {
        PromptReading {
            prompt_token_count: completion.prompt_token_count,
            reused_token_count: completion.reused_token_count,
            prompt_reading_duration: completion.prompt_reading_duration,
        }
    }
}

/// The answer the reasoning left no tokens for.
fn create_empty_answer_completion() -> Completion {
    Completion {
        text: String::new(),
        generated_token_count: 0,
        ending: ReplyEnding::MaximumTokenCount,
        prompt_token_count: 0,
        reused_token_count: 0,
        prompt_reading_duration: Duration::ZERO,
        generating_duration: Duration::ZERO,
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
fn create_request_json(request: &GenerationRequest, completion_request: &CompletionRequest) -> String {
    // Turns off every sampler but temperature, which the server and the model's metadata would otherwise enable.
    let mut request_json: serde_json::Value = serde_json::json!({
        "prompt": completion_request.prompt,
        "temperature": request.temperature,
        "seed": request.seed,
        "max_tokens": completion_request.token_limit,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "xtc_probability": 0.0,
        "repeat_penalty": 1.0,
        "mirostat": 0,
    });

    if let Some(json_schema) = completion_request.json_schema {
        request_json["json_schema"] = json_schema.clone();
    }

    // The server streams a control token's text only when it is preserved, and matches stop texts against that text.
    if let Some(stop_text) = completion_request.stop_text {
        request_json["stop"] = serde_json::json!([stop_text]);
        request_json["preserved_tokens"] = serde_json::json!([stop_text]);
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
    use crate::thinking::ModelThinking;
    use crate::wllama::wllama_instance::WllamaInstance;
    use crate::{Error, Generation, GenerationProgress, LoadOptions, LoadProgress, SlidingWindowCache, test_model};

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

    fn create_test_model_source() -> ModelSource {
        ModelSource {
            model_url: TEST_MODEL_URL.to_string(),
            wllama_directory_url: WLLAMA_DIRECTORY_URL.to_string(),
        }
    }

    /// Returns the generation and the progress it reported.
    async fn generate_with_stand_in_reasoning_delimiters() -> (Generation, Vec<GenerationProgress>) {
        let mut model: Model = Model::load(create_test_model_source(), &create_load_options(), |_: LoadProgress| {}).await.unwrap();
        model.model_thinking = ModelThinking::with_reasoning_delimiters(test_model::STAND_IN_REASONING_DELIMITERS);
        let mut progress_reports: Vec<GenerationProgress> = Vec::new();

        let generation: Generation = model
            .generate(&test_model::create_budgeted_thinking_request(), |progress: GenerationProgress| progress_reports.push(progress))
            .await
            .unwrap();

        (generation, progress_reports)
    }

    #[wasm_bindgen_test]
    async fn generate_with_thinking_budget_ends_reasoning_at_budget_and_answers_with_matching_json() {
        let (generation, _): (Generation, Vec<GenerationProgress>) = generate_with_stand_in_reasoning_delimiters().await;

        test_model::assert_reasoning_ends_at_budget_and_answer_matches_schema(&generation);
    }

    #[wasm_bindgen_test]
    async fn generate_with_thinking_reports_reasoning_progress_before_answer_progress() {
        let (generation, progress_reports): (Generation, Vec<GenerationProgress>) = generate_with_stand_in_reasoning_delimiters().await;

        test_model::assert_reasoning_progress_precedes_answer_progress(&progress_reports, &generation);
    }

    #[wasm_bindgen_test]
    async fn unload_frees_the_model_in_wllama() {
        let model: Model = Model::load(create_test_model_source(), &create_load_options(), |_: LoadProgress| {}).await.unwrap();
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
