use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvError};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use futures::channel::mpsc::{TrySendError, UnboundedSender};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;
use llama_cpp_2::{
    ApplyChatTemplateError,
    ChatTemplateError,
    LlamaContextLoadError,
    LlamaModelLoadError,
    LogOptions,
    NewLlamaChatMessageError,
};

use crate::generation_checks;
use crate::generation_checks::ReplyEnding;
use crate::llama_cpp::prompt_cache::PromptCache;
use crate::llama_cpp::reply_text::ReplyText;
use crate::llama_cpp::schema_constraint::SchemaConstraints;
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
};

// llama.cpp allows one initialization per process.
static LLAMA_BACKEND: LazyLock<Result<LlamaBackend, String>> = LazyLock::new(initialize_llama_backend);

const PROMPT_BATCH_TOKEN_COUNT: usize = 512;
// llama.cpp reads a negative output index as counting from the end.
const LAST_OUTPUT_INDEX: i32 = -1;

minimer::impl_from_error!(Error, LlamaModelLoadError);
minimer::impl_from_error!(Error, LlamaContextLoadError);
minimer::impl_from_error!(Error, ChatTemplateError);
minimer::impl_from_error!(Error, ApplyChatTemplateError);
minimer::impl_from_error!(Error, NewLlamaChatMessageError);

pub enum WorkerEvent<Progress, Outcome> {
    Progress(Progress),
    Finished(Outcome),
}

pub type LoadEvent = WorkerEvent<LoadProgress, Result<(), Error>>;
pub type LoadEventSender = UnboundedSender<LoadEvent>;
pub type GenerationEvent = WorkerEvent<GenerationProgress, Result<Generation, GenerationError>>;
pub type GenerationEventSender = UnboundedSender<GenerationEvent>;

pub enum WorkerCommand {
    Generate(GenerationJob),
    Shutdown,
}

pub struct GenerationJob {
    pub request: GenerationRequest,
    /// The job counts as cancelled once the epoch moves past this value.
    pub submitted_cancellation_epoch: u64,
    pub event_sender: GenerationEventSender,
}

pub struct WorkerStart {
    pub model_path: PathBuf,
    pub load_options: LoadOptions,
    pub cancellation_epoch: Arc<AtomicU64>,
    pub command_receiver: Receiver<WorkerCommand>,
    pub load_event_sender: LoadEventSender,
}

pub fn run_worker(worker_start: WorkerStart) {
    let WorkerStart {
        model_path,
        load_options,
        cancellation_epoch,
        command_receiver,
        load_event_sender,
    } = worker_start;

    let model_loaded: Result<LlamaModel, Error> = load_model(&model_path, &load_event_sender);
    let model: LlamaModel = match model_loaded {
        Ok(model) => model,
        Err(load_error) => {
            report_load_outcome(&load_event_sender, Err(load_error));
            return;
        }
    };

    let session_prepared: Result<GenerationSession, Error> = GenerationSession::prepare(&model, &load_options, cancellation_epoch);
    let mut session: GenerationSession = match session_prepared {
        Ok(session) => session,
        Err(preparation_error) => {
            report_load_outcome(&load_event_sender, Err(preparation_error));
            return;
        }
    };

    report_load_outcome(&load_event_sender, Ok(()));
    session.serve_commands(&command_receiver);
}

fn initialize_llama_backend() -> Result<LlamaBackend, String> {
    llama_cpp_2::send_logs_to_tracing(LogOptions::default());
    LlamaBackend::init().map_err(|initialization_error| initialization_error.to_string())
}

fn get_llama_backend() -> Result<&'static LlamaBackend, Error> {
    let initialization: &'static Result<LlamaBackend, String> = &LLAMA_BACKEND;

    initialization
        .as_ref()
        .map_err(|message| Error::new(&format!("could not initialize llama.cpp: {message}")))
}

fn report_load_outcome(load_event_sender: &LoadEventSender, load_outcome: Result<(), Error>) {
    // Fails only once the caller has dropped the load future.
    let _ = load_event_sender.unbounded_send(WorkerEvent::Finished(load_outcome));
}

fn load_model(model_path: &Path, load_event_sender: &LoadEventSender) -> Result<LlamaModel, Error> {
    let model_file_exists: bool = model_path.is_file();

    if !model_file_exists {
        return Err(Error::new(&format!("there is no model file at {}", model_path.display())));
    }

    let llama_backend: &LlamaBackend = get_llama_backend()?;
    let progress_sender: LoadEventSender = load_event_sender.clone();
    let model_parameters: LlamaModelParams = LlamaModelParams::default()
        .with_n_gpu_layers(u32::MAX)
        .with_progress_callback(move |loaded_proportion: f32| {
            let progress: LoadEvent = WorkerEvent::Progress(LoadProgress::Loading {
                loaded_proportion,
            });
            let reported: Result<(), TrySendError<LoadEvent>> = progress_sender.unbounded_send(progress);

            // Returning false aborts the load.
            reported.is_ok()
        });

    let model: LlamaModel = LlamaModel::load_from_file(llama_backend, model_path, &model_parameters)?;
    Ok(model)
}

struct GenerationSession<'model> {
    model: &'model LlamaModel,
    context: LlamaContext<'model>,
    chat_template: LlamaChatTemplate,
    batch: LlamaBatch<'static>,
    prompt_cache: PromptCache,
    schema_constraints: SchemaConstraints,
    cancellation_epoch: Arc<AtomicU64>,
}

impl<'model> GenerationSession<'model> {
    fn prepare(model: &'model LlamaModel, load_options: &LoadOptions, cancellation_epoch: Arc<AtomicU64>) -> Result<Self, Error> {
        let llama_backend: &LlamaBackend = get_llama_backend()?;
        let context_token_count: NonZeroU32 = load_options.nonzero_context_token_count()?;
        let keep_whole_sliding_window_cache: bool = load_options.sliding_window_cache == SlidingWindowCache::Full;

        let context_parameters: LlamaContextParams = LlamaContextParams::default()
            .with_n_ctx(Some(context_token_count))
            .with_n_batch(u32::try_from(PROMPT_BATCH_TOKEN_COUNT)?)
            .with_swa_full(keep_whole_sliding_window_cache);
        let context: LlamaContext<'model> = model.new_context(llama_backend, context_parameters)?;
        let chat_template: LlamaChatTemplate = model.chat_template(None)?;

        Ok(GenerationSession {
            model,
            context,
            chat_template,
            batch: LlamaBatch::new(PROMPT_BATCH_TOKEN_COUNT, 1),
            prompt_cache: PromptCache::new(),
            schema_constraints: SchemaConstraints::new(),
            cancellation_epoch,
        })
    }

    fn serve_commands(&mut self, command_receiver: &Receiver<WorkerCommand>) {
        loop {
            let received: Result<WorkerCommand, RecvError> = command_receiver.recv();
            let Ok(command) = received
            else {
                return;
            };

            match command {
                WorkerCommand::Generate(generation_job) => self.run_generation_job(generation_job),
                WorkerCommand::Shutdown => return,
            }
        }
    }

    fn run_generation_job(&mut self, generation_job: GenerationJob) {
        let monitor: GenerationMonitor = GenerationMonitor {
            event_sender: &generation_job.event_sender,
            cancellation_epoch: Arc::clone(&self.cancellation_epoch),
            submitted_cancellation_epoch: generation_job.submitted_cancellation_epoch,
        };
        let generation_outcome: Result<Generation, GenerationError> = self.generate(&generation_job.request, &monitor);

        // Fails only once the caller has dropped the generation future.
        let _ = generation_job.event_sender.unbounded_send(WorkerEvent::Finished(generation_outcome));
    }

    fn generate(&mut self, request: &GenerationRequest, monitor: &GenerationMonitor) -> Result<Generation, GenerationError> {
        monitor.check_cancellation()?;

        let prompt_tokens: Vec<LlamaToken> = self.tokenize_prompt(&request.messages)?;

        self.check_context_capacity(prompt_tokens.len(), request.maximum_token_count)?;

        let mut sampler: LlamaSampler = self.create_sampler(request)?;

        let reused_token_count: usize = self.prompt_cache.keep_reusable_prefix(&mut self.context, &prompt_tokens)?;

        let prompt_reading_started_at: Instant = Instant::now();
        self.read_prompt(&prompt_tokens, reused_token_count, monitor)?;
        let prompt_reading_duration: Duration = prompt_reading_started_at.elapsed();

        let generating_started_at: Instant = Instant::now();
        let reply: Reply = self.generate_reply(&mut sampler, request.maximum_token_count, monitor)?;
        let generating_duration: Duration = generating_started_at.elapsed();

        generation_checks::check_reply_completes_json_value(request, &reply.text, reply.ending)?;

        Ok(Generation {
            text: reply.text,
            prompt_token_count: prompt_tokens.len(),
            reused_token_count,
            generated_token_count: reply.generated_token_count,
            prompt_reading_duration,
            generating_duration,
        })
    }

    fn tokenize_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<LlamaToken>, Error> {
        self.check_no_control_token_text(messages)?;

        let chat_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(create_llama_chat_message)
            .collect::<Result<Vec<LlamaChatMessage>, Error>>()?;
        let prompt: String = self.model.apply_chat_template(&self.chat_template, &chat_messages, true)?;
        let prompt_tokens: Vec<LlamaToken> = self.model.vocab().tokenize(prompt.as_bytes(), true, true);

        if prompt_tokens.is_empty() {
            return Err(Error::new("the prompt has no tokens"));
        }

        Ok(prompt_tokens)
    }

    /// The templated prompt is tokenized with its control tokens parsed, so control-token text in a message would forge turns.
    fn check_no_control_token_text(&self, messages: &[ChatMessage]) -> Result<(), Error> {
        let vocabulary: LlamaVocab = self.model.vocab();

        for message in messages {
            let literal_tokens: Vec<LlamaToken> = vocabulary.tokenize(message.text.as_bytes(), false, false);
            let control_parsed_tokens: Vec<LlamaToken> = vocabulary.tokenize(message.text.as_bytes(), false, true);

            if literal_tokens != control_parsed_tokens {
                return Err(Error::new("a message's text contains one of the model's control tokens"));
            }
        }

        Ok(())
    }

    fn check_context_capacity(&self, prompt_token_count: usize, maximum_token_count: usize) -> Result<(), Error> {
        let context_token_count: usize = usize::try_from(self.context.n_ctx())?;
        generation_checks::check_context_capacity(prompt_token_count, maximum_token_count, context_token_count)
    }

    fn create_sampler(&mut self, request: &GenerationRequest) -> Result<LlamaSampler, Error> {
        let mut samplers: Vec<LlamaSampler> = Vec::new();

        if let Some(json_schema) = &request.json_schema {
            let schema_sampler: LlamaSampler = self.schema_constraints.create_sampler(self.model, json_schema)?;
            samplers.push(schema_sampler);
        }

        samplers.extend(create_token_choice_samplers(request.temperature, request.seed));
        Ok(LlamaSampler::chain_simple(samplers))
    }

    fn read_prompt(&mut self, prompt_tokens: &[LlamaToken], reused_token_count: usize, monitor: &GenerationMonitor) -> Result<(), GenerationError> {
        let unread_tokens: &[LlamaToken] = &prompt_tokens[reused_token_count..];

        for batch_tokens in unread_tokens.chunks(PROMPT_BATCH_TOKEN_COUNT) {
            self.prompt_cache.decode(&mut self.context, &mut self.batch, batch_tokens)?;

            let progress: GenerationProgress = GenerationProgress::ReadingPrompt {
                read_token_count: self.prompt_cache.cached_token_count(),
                prompt_token_count: prompt_tokens.len(),
                reused_token_count,
            };
            monitor.report(progress)?;
        }

        self.context.synchronize();
        Ok(())
    }

    fn generate_reply(&mut self, sampler: &mut LlamaSampler, maximum_token_count: usize, monitor: &GenerationMonitor) -> Result<Reply, GenerationError> {
        let vocabulary: LlamaVocab = self.model.vocab();
        let mut reply_text: ReplyText = ReplyText::new();
        let mut generated_token_count: usize = 0;

        // Samples once past the maximum to learn whether the reply ended exactly there.
        let ending: ReplyEnding = loop {
            let token: LlamaToken = sampler.sample(&self.context, LAST_OUTPUT_INDEX);

            if vocabulary.is_eog(token) {
                break ReplyEnding::EndOfGeneration;
            }

            if generated_token_count == maximum_token_count {
                break ReplyEnding::MaximumTokenCount;
            }

            reply_text.push_piece(&vocabulary.token_to_piece(token, false, None));
            self.prompt_cache.decode(&mut self.context, &mut self.batch, &[token])?;
            generated_token_count += 1;

            let progress: GenerationProgress = GenerationProgress::Generating {
                generated_token_count,
                maximum_token_count,
                text: reply_text.text_so_far(),
            };
            monitor.report(progress)?;
        };

        Ok(Reply {
            text: reply_text.into_text(),
            generated_token_count,
            ending,
        })
    }
}

struct Reply {
    text: String,
    generated_token_count: usize,
    ending: ReplyEnding,
}

struct GenerationMonitor<'job> {
    event_sender: &'job GenerationEventSender,
    cancellation_epoch: Arc<AtomicU64>,
    submitted_cancellation_epoch: u64,
}

impl GenerationMonitor<'_> {
    /// A caller that dropped the generation future counts as having cancelled it.
    fn check_cancellation(&self) -> Result<(), GenerationError> {
        let current_cancellation_epoch: u64 = self.cancellation_epoch.load(Ordering::SeqCst);
        let caller_stopped_listening: bool = self.event_sender.is_closed();

        if current_cancellation_epoch != self.submitted_cancellation_epoch || caller_stopped_listening {
            return Err(GenerationError::Cancelled);
        }

        Ok(())
    }

    fn report(&self, progress: GenerationProgress) -> Result<(), GenerationError> {
        // Fails only once the caller has stopped listening, which the cancellation check reports.
        let _ = self.event_sender.unbounded_send(WorkerEvent::Progress(progress));
        self.check_cancellation()
    }
}

fn create_llama_chat_message(message: &ChatMessage) -> Result<LlamaChatMessage, Error> {
    let llama_chat_message: LlamaChatMessage = LlamaChatMessage::new(message.role.chat_template_name().to_string(), message.text.clone())?;
    Ok(llama_chat_message)
}

fn create_token_choice_samplers(temperature: f32, seed: u32) -> Vec<LlamaSampler> {
    if temperature <= 0.0 {
        return vec![LlamaSampler::greedy()];
    }

    vec![
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(seed),
    ]
}
