use std::time::Duration;

use cross_llama::{
    ChatMessage,
    Generation,
    GenerationError,
    GenerationProgress,
    GenerationRequest,
    LoadOptions,
    LoadProgress,
    Model,
    ModelSource,
    Reasoning,
    Role,
    SlidingWindowCache,
    ThinkingCapability,
    ThinkingKind,
};

const CONTEXT_TOKEN_COUNT: u32 = 4096;
const MAXIMUM_ANSWER_TOKEN_COUNT: usize = 512;
// A reasoning model without a budget can repeat itself until it reaches the maximum.
const THINKING_BUDGET_TOKEN_COUNT: usize = 512;
// A model this small repeats itself at temperature zero.
const TEMPERATURE: f32 = 0.7;
const SEED: u32 = 0;
const BYTES_PER_MEGABYTE: f64 = 1_000_000.0;
const PERCENT_PER_PROPORTION: f32 = 100.0;
// minimer starts every error message with this.
const ERROR_MESSAGE_PREFIX: &str = "Error: ";

#[derive(Clone, Debug, PartialEq)]
pub enum ReplyProgress {
    ReadingPrompt {
        read_token_count: usize,
        prompt_token_count: usize,
    },
    Reasoning {
        generated_token_count: usize,
        maximum_token_count: usize,
        /// The reasoning text since the previous report.
        new_text: String,
    },
    Generating {
        generated_token_count: usize,
        maximum_token_count: usize,
        /// The reply text since the previous report.
        new_text: String,
    },
}

/// How much of the reasoning and of the answer the earlier reports carried.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StreamedTextLengths {
    reasoning: usize,
    answer: usize,
}

pub async fn load_model(source: ModelSource, on_progress: impl FnMut(LoadProgress)) -> Result<Model, cross_llama::Error> {
    let load_options: LoadOptions = LoadOptions {
        context_token_count: CONTEXT_TOKEN_COUNT,
        sliding_window_cache: SlidingWindowCache::WindowOnly,
    };

    Model::load(source, &load_options, on_progress).await
}

/// Whether `reply` accepts `ThinkingKind::Enabled` for the model.
pub fn offers_thinking(model: &Model) -> bool {
    let thinking_capability: ThinkingCapability = model.thinking_capability();

    thinking_capability.separates_reasoning
}

/// Sends the prompt alone, without earlier prompts or replies.
pub async fn reply(
    model: &Model,
    prompt: &str,
    thinking: ThinkingKind,
    mut on_progress: impl FnMut(ReplyProgress),
) -> Result<Generation, GenerationError> {
    let request: GenerationRequest = create_request(prompt, thinking);
    let mut streamed_text_lengths: StreamedTextLengths = StreamedTextLengths::default();

    let generation: Generation = model
        .generate(&request, |progress: GenerationProgress| {
            on_progress(create_reply_progress(progress, &mut streamed_text_lengths));
        })
        .await?;

    // The final text can end with a character whose bytes were incomplete in the last report.
    let unstreamed_text: &str = generation.text.get(streamed_text_lengths.answer..).unwrap_or_default();

    if !unstreamed_text.is_empty() {
        on_progress(ReplyProgress::Generating {
            generated_token_count: generation.generated_token_count,
            maximum_token_count: request.maximum_token_count,
            new_text: unstreamed_text.to_string(),
        });
    }

    Ok(generation)
}

pub fn describe_load_progress(progress: &LoadProgress) -> String {
    match progress {
        LoadProgress::Downloading {
            downloaded_byte_count,
            total_byte_count: Some(total_byte_count),
        } => format!(
            "Downloading the model: {:.0} of {:.0} MB",
            megabytes(*downloaded_byte_count),
            megabytes(*total_byte_count),
        ),
        LoadProgress::Downloading {
            downloaded_byte_count,
            total_byte_count: None,
        } => format!("Downloading the model: {:.0} MB", megabytes(*downloaded_byte_count)),
        // wllama reports loading only at 0 and 1.
        LoadProgress::Loading { loaded_proportion } if *loaded_proportion <= 0.0 => "Loading the model".to_string(),
        LoadProgress::Loading { loaded_proportion } => format!("Loading the model: {:.0}%", loaded_proportion * PERCENT_PER_PROPORTION),
    }
}

pub fn describe_reply_progress(progress: &ReplyProgress) -> String {
    match progress {
        ReplyProgress::ReadingPrompt {
            read_token_count,
            prompt_token_count,
        } => format!("Reading the prompt: {read_token_count} of {prompt_token_count} tokens"),
        ReplyProgress::Reasoning {
            generated_token_count,
            maximum_token_count,
            ..
        } => format!("Reasoning: {generated_token_count} of at most {maximum_token_count} tokens"),
        ReplyProgress::Generating {
            generated_token_count,
            maximum_token_count,
            ..
        } => format!("Generating: {generated_token_count} of at most {maximum_token_count} tokens"),
    }
}

/// The failure's message, without the backtrace its `Display` includes.
pub fn describe_reply_failure(error: &GenerationError) -> String {
    match error {
        GenerationError::Cancelled => error.to_string(),
        GenerationError::Failed(failure) => failure.message.strip_prefix(ERROR_MESSAGE_PREFIX).unwrap_or(&failure.message).to_string(),
    }
}

pub fn describe_generation_speed(generation: &Generation) -> String {
    let generating_seconds: f64 = generation.generating_duration.as_secs_f64();
    let token_description: String = describe_generated_tokens(generation.generated_token_count, generation.reasoning.as_ref());

    format!(
        "{token_description} in {:.2} s, {:.1} tokens per second",
        generating_seconds,
        tokens_per_second(generation.generated_token_count, generation.generating_duration),
    )
}

fn describe_generated_tokens(generated_token_count: usize, reasoning: Option<&Reasoning>) -> String {
    match reasoning {
        Some(reasoning) => format!("{generated_token_count} tokens ({} of them reasoning)", reasoning.token_count),
        None => format!("{generated_token_count} tokens"),
    }
}

fn create_request(prompt: &str, thinking: ThinkingKind) -> GenerationRequest {
    let (maximum_token_count, thinking_budget_token_count): (usize, Option<usize>) = match thinking {
        ThinkingKind::Disabled => (MAXIMUM_ANSWER_TOKEN_COUNT, None),
        ThinkingKind::Enabled => (MAXIMUM_ANSWER_TOKEN_COUNT + THINKING_BUDGET_TOKEN_COUNT, Some(THINKING_BUDGET_TOKEN_COUNT)),
    };

    GenerationRequest {
        messages: vec![ChatMessage {
            role: Role::User,
            text: prompt.to_string(),
        }],
        json_schema: None,
        temperature: TEMPERATURE,
        seed: SEED,
        maximum_token_count,
        thinking,
        thinking_budget_token_count,
        thinking_effort: None,
    }
}

fn create_reply_progress(progress: GenerationProgress, streamed_text_lengths: &mut StreamedTextLengths) -> ReplyProgress {
    match progress {
        GenerationProgress::ReadingPrompt {
            read_token_count,
            prompt_token_count,
            ..
        } => ReplyProgress::ReadingPrompt {
            read_token_count,
            prompt_token_count,
        },
        GenerationProgress::GeneratingReasoning {
            generated_token_count,
            maximum_token_count,
            text,
        } => ReplyProgress::Reasoning {
            generated_token_count,
            maximum_token_count,
            new_text: take_new_text(&text, &mut streamed_text_lengths.reasoning),
        },
        GenerationProgress::GeneratingAnswer {
            generated_token_count,
            maximum_token_count,
            text,
        } => ReplyProgress::Generating {
            generated_token_count,
            maximum_token_count,
            new_text: take_new_text(&text, &mut streamed_text_lengths.answer),
        },
    }
}

/// `text` is everything generated so far, of which earlier reports carried the first `streamed_length` bytes.
fn take_new_text(text: &str, streamed_length: &mut usize) -> String {
    let new_text: String = text.get(*streamed_length..).unwrap_or_default().to_string();
    *streamed_length = (*streamed_length).max(text.len());

    new_text
}

fn tokens_per_second(token_count: usize, duration: Duration) -> f64 {
    let seconds: f64 = duration.as_secs_f64();

    if seconds <= 0.0 {
        return 0.0;
    }

    token_count as f64 / seconds
}

fn megabytes(byte_count: u64) -> f64 {
    byte_count as f64 / BYTES_PER_MEGABYTE
}

#[cfg(test)]
mod tests {
    use cross_llama::{GenerationError, GenerationProgress, GenerationRequest, LoadProgress, ThinkingKind};

    use super::{ReplyProgress, StreamedTextLengths};

    #[test]
    fn create_reply_progress_reports_only_text_since_previous_report() {
        let mut streamed_text_lengths: StreamedTextLengths = StreamedTextLengths::default();
        let _first: ReplyProgress = super::create_reply_progress(answer_progress("Olá"), &mut streamed_text_lengths);

        let second: ReplyProgress = super::create_reply_progress(answer_progress("Olá, mundo"), &mut streamed_text_lengths);

        assert_eq!(
            second,
            ReplyProgress::Generating {
                generated_token_count: 1,
                maximum_token_count: 8,
                new_text: ", mundo".to_string(),
            },
        );
    }

    #[test]
    fn create_reply_progress_streams_the_answer_apart_from_the_reasoning() {
        let mut streamed_text_lengths: StreamedTextLengths = StreamedTextLengths::default();
        let reasoning_progress: GenerationProgress = GenerationProgress::GeneratingReasoning {
            generated_token_count: 1,
            maximum_token_count: 8,
            text: "Pensando".to_string(),
        };
        let _reasoning: ReplyProgress = super::create_reply_progress(reasoning_progress, &mut streamed_text_lengths);

        let answer: ReplyProgress = super::create_reply_progress(answer_progress("Olá"), &mut streamed_text_lengths);

        assert_eq!(
            answer,
            ReplyProgress::Generating {
                generated_token_count: 1,
                maximum_token_count: 8,
                new_text: "Olá".to_string(),
            },
        );
    }

    #[test]
    fn create_request_gives_enabled_thinking_a_budget_beside_the_answer_maximum() {
        let request: GenerationRequest = super::create_request("Olá", ThinkingKind::Enabled);

        assert_eq!(request.thinking_budget_token_count, Some(super::THINKING_BUDGET_TOKEN_COUNT));
        assert_eq!(request.maximum_token_count, super::MAXIMUM_ANSWER_TOKEN_COUNT + super::THINKING_BUDGET_TOKEN_COUNT);
    }

    #[test]
    fn describe_reply_failure_gives_only_the_message() {
        let error: GenerationError = GenerationError::Failed(cross_llama::Error::new("the prompt is too long"));

        assert_eq!(super::describe_reply_failure(&error), "the prompt is too long");
    }

    #[test]
    fn describe_load_progress_omits_percentage_before_loading_starts() {
        let description: String = super::describe_load_progress(&LoadProgress::Loading { loaded_proportion: 0.0 });

        assert_eq!(description, "Loading the model");
    }

    fn answer_progress(text: &str) -> GenerationProgress {
        GenerationProgress::GeneratingAnswer {
            generated_token_count: 1,
            maximum_token_count: 8,
            text: text.to_string(),
        }
    }
}
