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
    Role,
    SlidingWindowCache,
};

pub const MODEL_FILE_NAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

const CONTEXT_TOKEN_COUNT: u32 = 4096;
const MAXIMUM_REPLY_TOKEN_COUNT: usize = 512;
// A model this small repeats itself at temperature zero.
const TEMPERATURE: f32 = 0.7;
const SEED: u32 = 0;
const BYTES_PER_MEGABYTE: f64 = 1_000_000.0;
const PERCENT_PER_PROPORTION: f32 = 100.0;

#[derive(Clone, Debug, PartialEq)]
pub enum ReplyProgress {
    ReadingPrompt {
        read_token_count: usize,
        prompt_token_count: usize,
    },
    Generating {
        generated_token_count: usize,
        maximum_token_count: usize,
        /// The reply text since the previous report.
        new_text: String,
    },
}

pub async fn load_model(source: ModelSource, on_progress: impl FnMut(LoadProgress)) -> Result<Model, cross_llama::Error> {
    let load_options: LoadOptions = LoadOptions {
        context_token_count: CONTEXT_TOKEN_COUNT,
        sliding_window_cache: SlidingWindowCache::WindowOnly,
    };

    Model::load(source, &load_options, on_progress).await
}

/// Sends the prompt alone, without earlier prompts or replies.
pub async fn reply(model: &Model, prompt: &str, mut on_progress: impl FnMut(ReplyProgress)) -> Result<Generation, GenerationError> {
    let request: GenerationRequest = create_request(prompt);
    let mut streamed_text_length: usize = 0;

    let generation: Generation = model
        .generate(&request, |progress: GenerationProgress| {
            on_progress(create_reply_progress(progress, &mut streamed_text_length));
        })
        .await?;

    // The final text can end with a character whose bytes were incomplete in the last report.
    let unstreamed_text: &str = generation.text.get(streamed_text_length..).unwrap_or_default();

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
        ReplyProgress::Generating {
            generated_token_count,
            maximum_token_count,
            ..
        } => format!("Generating: {generated_token_count} of at most {maximum_token_count} tokens"),
    }
}

pub fn describe_generation_speed(generation: &Generation) -> String {
    let generating_seconds: f64 = generation.generating_duration.as_secs_f64();

    format!(
        "{} tokens in {:.2} s, {:.1} tokens per second",
        generation.generated_token_count,
        generating_seconds,
        tokens_per_second(generation.generated_token_count, generation.generating_duration),
    )
}

fn create_request(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        messages: vec![ChatMessage {
            role: Role::User,
            text: prompt.to_string(),
        }],
        json_schema: None,
        temperature: TEMPERATURE,
        seed: SEED,
        maximum_token_count: MAXIMUM_REPLY_TOKEN_COUNT,
    }
}

fn create_reply_progress(progress: GenerationProgress, streamed_text_length: &mut usize) -> ReplyProgress {
    match progress {
        GenerationProgress::ReadingPrompt {
            read_token_count,
            prompt_token_count,
            ..
        } => ReplyProgress::ReadingPrompt {
            read_token_count,
            prompt_token_count,
        },
        GenerationProgress::Generating {
            generated_token_count,
            maximum_token_count,
            text,
        } => {
            let new_text: String = text.get(*streamed_text_length..).unwrap_or_default().to_string();
            *streamed_text_length = (*streamed_text_length).max(text.len());

            ReplyProgress::Generating {
                generated_token_count,
                maximum_token_count,
                new_text,
            }
        }
    }
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
    use cross_llama::{GenerationProgress, LoadProgress};

    use super::ReplyProgress;

    #[test]
    fn create_reply_progress_reports_only_text_since_previous_report() {
        let mut streamed_text_length: usize = 0;
        let _first: ReplyProgress = super::create_reply_progress(generating_progress("Olá"), &mut streamed_text_length);

        let second: ReplyProgress = super::create_reply_progress(generating_progress("Olá, mundo"), &mut streamed_text_length);

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
    fn describe_load_progress_omits_percentage_before_loading_starts() {
        let description: String = super::describe_load_progress(&LoadProgress::Loading { loaded_proportion: 0.0 });

        assert_eq!(description, "Loading the model");
    }

    fn generating_progress(text: &str) -> GenerationProgress {
        GenerationProgress::Generating {
            generated_token_count: 1,
            maximum_token_count: 8,
            text: text.to_string(),
        }
    }
}
