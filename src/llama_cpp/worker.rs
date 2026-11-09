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
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;
use llama_cpp_2::{LlamaContextLoadError, LlamaModelLoadError, LogOptions};

use crate::generation_checks;
use crate::generation_checks::ReplyEnding;
use crate::chat_template::ChatTemplate;
use crate::llama_cpp::model_chat_template;
use crate::llama_cpp::prompt_cache::PromptCache;
use crate::llama_cpp::reply_text::ReplyText;
use crate::llama_cpp::schema_constraint::SchemaConstraints;
use crate::thinking;
use crate::thinking::{ModelThinking, PromptProgressReportKind, ReasoningPlan};
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
};

// llama.cpp allows one initialization per process.
static LLAMA_BACKEND: LazyLock<Result<LlamaBackend, String>> = LazyLock::new(initialize_llama_backend);

const PROMPT_BATCH_TOKEN_COUNT: usize = 512;
// llama.cpp reads a negative output index as counting from the end.
const LAST_OUTPUT_INDEX: i32 = -1;

minimer::impl_from_error!(Error, LlamaModelLoadError);
minimer::impl_from_error!(Error, LlamaContextLoadError);

pub enum WorkerEvent<Progress, Outcome> {
    Progress(Progress),
    Finished(Outcome),
}

pub type LoadEvent = WorkerEvent<LoadProgress, Result<ThinkingCapability, Error>>;
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

    report_load_outcome(&load_event_sender, Ok(session.model_thinking.capability()));
    session.serve_commands(&command_receiver);
}

fn initialize_llama_backend() -> Result<LlamaBackend, String> {
    llama_cpp_2::send_logs_to_tracing(LogOptions::default());
    LlamaBackend::init().map_err(|initialization_error| initialization_error.to_string())
}

pub fn get_llama_backend() -> Result<&'static LlamaBackend, Error> {
    let initialization: &'static Result<LlamaBackend, String> = &LLAMA_BACKEND;

    initialization
        .as_ref()
        .map_err(|message| Error::new(&format!("could not initialize llama.cpp: {message}")))
}

fn report_load_outcome(load_event_sender: &LoadEventSender, load_outcome: Result<ThinkingCapability, Error>) {
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

/// Tokenized with control tokens parsed, as a templated prompt is.
fn tokenize_single_token(vocabulary: &LlamaVocab, text: &str) -> Option<LlamaToken> {
    let tokens: Vec<LlamaToken> = vocabulary.tokenize(text.as_bytes(), false, true);

    match tokens.as_slice() {
        [token] => Some(*token),
        _ => None,
    }
}

struct PromptReading {
    prompt_token_count: usize,
    reused_token_count: usize,
    prompt_reading_duration: Duration,
    finished_at: Instant,
}

struct GenerationSession<'model> {
    model: &'model LlamaModel,
    context: LlamaContext<'model>,
    chat_template: ChatTemplate,
    model_thinking: ModelThinking,
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
        let chat_template: ChatTemplate = model_chat_template::read_chat_template(model)?;
        let vocabulary: LlamaVocab = model.vocab();
        let model_thinking: ModelThinking = ModelThinking::new(&chat_template, |text: &str| tokenize_single_token(&vocabulary, text).is_some());

        Ok(GenerationSession {
            model,
            context,
            chat_template,
            model_thinking,
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

        let prompt: String = self.render_prompt(request)?;
        let reasoning_plan: Option<ReasoningPlan> = self.model_thinking.plan_reasoning(request, &prompt);

        match reasoning_plan {
            Some(reasoning_plan) => self.generate_reasoning_then_answer(request, &reasoning_plan, monitor),
            None => self.generate_answer_alone(request, &prompt, monitor),
        }
    }

    fn render_prompt(&self, request: &GenerationRequest) -> Result<String, Error> {
        self.check_no_control_token_text(&request.messages)?;
        self.model_thinking.check_request(request)?;

        self.chat_template.render_prompt(&request.messages, request.thinking, request.thinking_effort)
    }

    fn generate_answer_alone(&mut self, request: &GenerationRequest, prompt: &str, monitor: &GenerationMonitor) -> Result<Generation, GenerationError> {
        let prompt_tokens: Vec<LlamaToken> = self.tokenize_prompt(prompt, request.maximum_token_count)?;
        let mut answer_sampler: LlamaSampler = self.create_sampler(request, request.json_schema.as_ref())?;

        let prompt_reading: PromptReading = self.read_prompt(&prompt_tokens, PromptProgressReportKind::Reported, monitor)?;
        let answer: Reply = self.generate_answer(&mut answer_sampler, request, request.maximum_token_count, 0, monitor)?;
        let generating_duration: Duration = prompt_reading.finished_at.elapsed();

        generation_checks::check_reply_completes_json_value(request, &answer.text, answer.ending)?;

        Ok(Generation {
            text: answer.text,
            reasoning: None,
            prompt_token_count: prompt_reading.prompt_token_count,
            reused_token_count: prompt_reading.reused_token_count,
            generated_token_count: answer.generated_token_count,
            prompt_reading_duration: prompt_reading.prompt_reading_duration,
            generating_duration,
        })
    }

    /// Reads the prompt once: the reasoning continues it, and the answer continues the reasoning.
    fn generate_reasoning_then_answer(
        &mut self,
        request: &GenerationRequest,
        reasoning_plan: &ReasoningPlan,
        monitor: &GenerationMonitor,
    ) -> Result<Generation, GenerationError> {
        let mut answer_sampler: LlamaSampler = self.create_sampler(request, request.json_schema.as_ref())?;
        let reasoning_run: Option<ReasoningRun> = self.run_reasoning(request, reasoning_plan, monitor)?;

        let reasoning_reply_text: &str = reasoning_run.as_ref().map_or("", |run: &ReasoningRun| run.reply.text.as_str());
        let reasoning: Reasoning = thinking::create_reasoning(
            reasoning_reply_text,
            reasoning_run.as_ref().map_or(0, |run: &ReasoningRun| run.reply.generated_token_count),
        );
        let answer_token_limit: usize = reasoning_plan.answer_token_limit(reasoning.token_count);
        let answer_prompt: String = reasoning_plan.create_answer_prompt(reasoning_reply_text);

        let (prompt_reading, answer): (PromptReading, Reply) = match reasoning_run {
            None => {
                let answer_prompt_tokens: Vec<LlamaToken> = self.tokenize_prompt(&answer_prompt, answer_token_limit)?;
                let prompt_reading: PromptReading = self.read_prompt(&answer_prompt_tokens, PromptProgressReportKind::Reported, monitor)?;
                let answer: Reply = self.generate_answer(&mut answer_sampler, request, answer_token_limit, reasoning.token_count, monitor)?;
                (prompt_reading, answer)
            }
            Some(reasoning_run) if answer_token_limit == 0 => (reasoning_run.prompt_reading, Reply::create_empty_at_maximum()),
            Some(reasoning_run) => {
                let answer_prompt_tokens: Vec<LlamaToken> = self.tokenize_prompt(&answer_prompt, answer_token_limit)?;
                self.read_prompt(&answer_prompt_tokens, PromptProgressReportKind::Unreported, monitor)?;
                let answer: Reply = self.generate_answer(&mut answer_sampler, request, answer_token_limit, reasoning.token_count, monitor)?;
                (reasoning_run.prompt_reading, answer)
            }
        };
        let generating_duration: Duration = prompt_reading.finished_at.elapsed();

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
    fn run_reasoning(
        &mut self,
        request: &GenerationRequest,
        reasoning_plan: &ReasoningPlan,
        monitor: &GenerationMonitor,
    ) -> Result<Option<ReasoningRun>, GenerationError> {
        if reasoning_plan.reasoning_token_limit == 0 {
            return Ok(None);
        }

        let vocabulary: LlamaVocab = self.model.vocab();
        let reasoning_end_token: LlamaToken = tokenize_single_token(&vocabulary, reasoning_plan.end_delimiter())
            .ok_or_else(|| Error::new("the reasoning's end delimiter is not one token"))?;
        let prompt_tokens: Vec<LlamaToken> = self.tokenize_prompt(&reasoning_plan.reasoning_prompt, request.maximum_token_count)?;
        let mut reasoning_sampler: LlamaSampler = self.create_sampler(request, None)?;

        let prompt_reading: PromptReading = self.read_prompt(&prompt_tokens, PromptProgressReportKind::Reported, monitor)?;
        let reply: Reply = self.generate_reply(
            &mut reasoning_sampler,
            reasoning_plan.reasoning_token_limit,
            |token: LlamaToken| token == reasoning_end_token || vocabulary.is_eog(token),
            |generated_token_count: usize, text: String| GenerationProgress::GeneratingReasoning {
                generated_token_count,
                maximum_token_count: request.maximum_token_count,
                text,
            },
            monitor,
        )?;

        Ok(Some(ReasoningRun {
            reply,
            prompt_reading,
        }))
    }

    fn generate_answer(
        &mut self,
        answer_sampler: &mut LlamaSampler,
        request: &GenerationRequest,
        answer_token_limit: usize,
        reasoning_token_count: usize,
        monitor: &GenerationMonitor,
    ) -> Result<Reply, GenerationError> {
        let vocabulary: LlamaVocab = self.model.vocab();

        self.generate_reply(
            answer_sampler,
            answer_token_limit,
            |token: LlamaToken| vocabulary.is_eog(token),
            |generated_token_count: usize, text: String| GenerationProgress::GeneratingAnswer {
                generated_token_count: reasoning_token_count + generated_token_count,
                maximum_token_count: request.maximum_token_count,
                text,
            },
            monitor,
        )
    }

    fn tokenize_prompt(&self, prompt: &str, reply_token_limit: usize) -> Result<Vec<LlamaToken>, Error> {
        let prompt_tokens: Vec<LlamaToken> = self.model.vocab().tokenize(prompt.as_bytes(), true, true);

        if prompt_tokens.is_empty() {
            return Err(Error::new("the prompt has no tokens"));
        }

        self.check_context_capacity(prompt_tokens.len(), reply_token_limit)?;

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

    fn create_sampler(&mut self, request: &GenerationRequest, json_schema: Option<&serde_json::Value>) -> Result<LlamaSampler, Error> {
        let mut samplers: Vec<LlamaSampler> = Vec::new();

        if let Some(json_schema) = json_schema {
            let schema_sampler: LlamaSampler = self.schema_constraints.create_sampler(self.model, json_schema)?;
            samplers.push(schema_sampler);
        }

        samplers.extend(create_token_choice_samplers(request.temperature, request.seed));
        Ok(LlamaSampler::chain_simple(samplers))
    }

    fn read_prompt(
        &mut self,
        prompt_tokens: &[LlamaToken],
        report_kind: PromptProgressReportKind,
        monitor: &GenerationMonitor,
    ) -> Result<PromptReading, GenerationError> {
        let reused_token_count: usize = self.prompt_cache.keep_reusable_prefix(&mut self.context, prompt_tokens)?;
        let unread_tokens: &[LlamaToken] = &prompt_tokens[reused_token_count..];
        let reading_started_at: Instant = Instant::now();

        for batch_tokens in unread_tokens.chunks(PROMPT_BATCH_TOKEN_COUNT) {
            self.prompt_cache.decode(&mut self.context, &mut self.batch, batch_tokens)?;

            match report_kind {
                PromptProgressReportKind::Reported => monitor.report(GenerationProgress::ReadingPrompt {
                    read_token_count: self.prompt_cache.cached_token_count(),
                    prompt_token_count: prompt_tokens.len(),
                    reused_token_count,
                })?,
                PromptProgressReportKind::Unreported => monitor.check_cancellation()?,
            }
        }

        self.context.synchronize();

        Ok(PromptReading {
            prompt_token_count: prompt_tokens.len(),
            reused_token_count,
            prompt_reading_duration: reading_started_at.elapsed(),
            finished_at: Instant::now(),
        })
    }

    fn generate_reply(
        &mut self,
        sampler: &mut LlamaSampler,
        token_limit: usize,
        ends_reply: impl Fn(LlamaToken) -> bool,
        create_progress: impl Fn(usize, String) -> GenerationProgress,
        monitor: &GenerationMonitor,
    ) -> Result<Reply, GenerationError> {
        let vocabulary: LlamaVocab = self.model.vocab();
        let mut reply_text: ReplyText = ReplyText::new();
        let mut generated_token_count: usize = 0;

        // Samples once past the limit to learn whether the reply ended exactly there.
        let ending: ReplyEnding = loop {
            let token: LlamaToken = sampler.sample(&self.context, LAST_OUTPUT_INDEX);

            if ends_reply(token) {
                break ReplyEnding::EndOfGeneration;
            }

            if generated_token_count == token_limit {
                break ReplyEnding::MaximumTokenCount;
            }

            reply_text.push_piece(&vocabulary.token_to_piece(token, false, None));
            self.prompt_cache.decode(&mut self.context, &mut self.batch, &[token])?;
            generated_token_count += 1;

            monitor.report(create_progress(generated_token_count, reply_text.complete_text()))?;
        };

        Ok(Reply {
            text: reply_text.complete_text(),
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

impl Reply {
    fn create_empty_at_maximum() -> Self {
        Reply {
            text: String::new(),
            generated_token_count: 0,
            ending: ReplyEnding::MaximumTokenCount,
        }
    }
}

struct ReasoningRun {
    reply: Reply,
    prompt_reading: PromptReading,
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

fn create_token_choice_samplers(temperature: f32, seed: u32) -> Vec<LlamaSampler> {
    if temperature <= 0.0 {
        return vec![LlamaSampler::greedy()];
    }

    vec![
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(seed),
    ]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use futures::channel::mpsc::UnboundedReceiver;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    use crate::llama_cpp::worker;
    use crate::llama_cpp::worker::{GenerationEvent, GenerationEventSender, GenerationMonitor, GenerationSession, WorkerEvent};
    use crate::thinking::ModelThinking;
    use crate::{Generation, GenerationError, GenerationProgress, GenerationRequest, LoadOptions, SlidingWindowCache, test_model};

    const CONTEXT_TOKEN_COUNT: u32 = 4096;

    fn load_test_model() -> LlamaModel {
        let model_parameters: LlamaModelParams = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        LlamaModel::load_from_file(worker::get_llama_backend().unwrap(), test_model::get_test_model_path(), &model_parameters).unwrap()
    }

    /// Returns the generation's outcome and the progress it reported.
    fn generate_with_stand_in_reasoning_delimiters(request: &GenerationRequest) -> (Result<Generation, GenerationError>, Vec<GenerationProgress>) {
        let model: LlamaModel = load_test_model();
        let load_options: LoadOptions = LoadOptions {
            context_token_count: CONTEXT_TOKEN_COUNT,
            sliding_window_cache: SlidingWindowCache::Full,
        };
        let cancellation_epoch: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let mut session: GenerationSession = GenerationSession::prepare(&model, &load_options, Arc::clone(&cancellation_epoch)).unwrap();
        session.model_thinking = ModelThinking::with_reasoning_delimiters(test_model::STAND_IN_REASONING_DELIMITERS);
        let (event_sender, mut event_receiver): (GenerationEventSender, UnboundedReceiver<GenerationEvent>) = futures::channel::mpsc::unbounded();
        let monitor: GenerationMonitor = GenerationMonitor {
            event_sender: &event_sender,
            cancellation_epoch,
            submitted_cancellation_epoch: 0,
        };

        let generation_outcome: Result<Generation, GenerationError> = session.generate(request, &monitor);
        let progress_reports: Vec<GenerationProgress> = std::iter::from_fn(|| event_receiver.try_recv().ok())
            .map(|event: GenerationEvent| match event {
                WorkerEvent::Progress(progress) => progress,
                WorkerEvent::Finished(_) => panic!("the session reports its outcome by returning it"),
            })
            .collect();

        (generation_outcome, progress_reports)
    }

    #[test]
    fn generate_with_thinking_budget_ends_reasoning_at_budget_and_answers_with_matching_json() {
        let (generation_outcome, _): (Result<Generation, GenerationError>, Vec<GenerationProgress>) =
            generate_with_stand_in_reasoning_delimiters(&test_model::create_budgeted_thinking_request());

        test_model::assert_reasoning_ends_at_budget_and_answer_matches_schema(&generation_outcome.unwrap());
    }

    #[test]
    fn generate_with_thinking_reports_reasoning_progress_before_answer_progress() {
        let (generation_outcome, progress_reports): (Result<Generation, GenerationError>, Vec<GenerationProgress>) =
            generate_with_stand_in_reasoning_delimiters(&test_model::create_budgeted_thinking_request());

        test_model::assert_reasoning_progress_precedes_answer_progress(&progress_reports, &generation_outcome.unwrap());
    }

    #[test]
    fn generate_with_thinking_budget_rejects_maximum_exceeding_context_before_reporting_progress() {
        let request: GenerationRequest = test_model::create_budgeted_thinking_request_with_context_filling_maximum(CONTEXT_TOKEN_COUNT);

        let (generation_outcome, progress_reports): (Result<Generation, GenerationError>, Vec<GenerationProgress>) = generate_with_stand_in_reasoning_delimiters(&request);

        assert!(generation_outcome.is_err());
        assert!(progress_reports.is_empty());
    }
}
