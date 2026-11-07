use std::time::Duration;

use serde::Deserialize;

use crate::generation_checks;
use crate::generation_checks::ReplyEnding;
use crate::{Error, Generation, GenerationProgress, GenerationRequest};

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

/// One streamed chunk of a llama.cpp server completion, as wllama passes it on.
#[derive(Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    timings: Option<ChunkTimings>,
    prompt_progress: Option<PromptProgress>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    text: Option<String>,
    finish_reason: Option<FinishReason>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FinishReason {
    /// The model ended its reply; the server counts the ending token among the generated ones.
    Stop,
    Length,
}

#[derive(Deserialize)]
struct PromptProgress {
    #[serde(rename = "total")]
    prompt_token_count: usize,
    #[serde(rename = "cache")]
    reused_token_count: usize,
    /// Includes the reused tokens.
    #[serde(rename = "processed")]
    read_token_count: usize,
}

#[derive(Clone, Copy, Deserialize)]
struct ChunkTimings {
    #[serde(rename = "cache_n")]
    reused_token_count: usize,
    /// Excludes the reused tokens.
    #[serde(rename = "prompt_n")]
    read_token_count: usize,
    #[serde(rename = "prompt_ms")]
    prompt_reading_milliseconds: f64,
    #[serde(rename = "predicted_n")]
    generated_token_count: usize,
    #[serde(rename = "predicted_ms")]
    generating_milliseconds: f64,
}

/// The reply as its chunks arrive.
pub struct StreamedReply {
    text: String,
    latest_timings: Option<ChunkTimings>,
    finish_reason: Option<FinishReason>,
}

impl StreamedReply {
    pub fn new() -> Self {
        StreamedReply {
            text: String::new(),
            latest_timings: None,
            finish_reason: None,
        }
    }

    /// Returns the progress the chunk reports, if any.
    pub fn apply_chunk(&mut self, chunk_json: &str, maximum_token_count: usize) -> Result<Vec<GenerationProgress>, Error> {
        let chunk: CompletionChunk = serde_json::from_str(chunk_json)?;
        let mut progress_reports: Vec<GenerationProgress> = Vec::new();

        if let Some(timings) = chunk.timings {
            self.latest_timings = Some(timings);
        }

        if let Some(prompt_progress) = chunk.prompt_progress {
            progress_reports.push(GenerationProgress::ReadingPrompt {
                read_token_count: prompt_progress.read_token_count,
                prompt_token_count: prompt_progress.prompt_token_count,
                reused_token_count: prompt_progress.reused_token_count,
            });
        }

        for choice in chunk.choices {
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }

            let text: String = choice.text.unwrap_or_default();

            if text.is_empty() {
                continue;
            }

            self.text.push_str(&text);
            progress_reports.push(GenerationProgress::Generating {
                generated_token_count: self.latest_timings.map_or(0, |timings| timings.generated_token_count),
                maximum_token_count,
                text: self.text.clone(),
            });
        }

        Ok(progress_reports)
    }

    pub fn into_generation(self, request: &GenerationRequest) -> Result<Generation, Error> {
        let (Some(timings), Some(finish_reason)) = (self.latest_timings, self.finish_reason)
        else {
            return Err(Error::new("wllama finished the generation without reporting how the reply ended"));
        };

        let (ending, generated_token_count): (ReplyEnding, usize) = match finish_reason {
            FinishReason::Stop => (ReplyEnding::EndOfGeneration, timings.generated_token_count.saturating_sub(1)),
            FinishReason::Length => (ReplyEnding::MaximumTokenCount, timings.generated_token_count),
        };

        generation_checks::check_reply_completes_json_value(request, &self.text, ending)?;

        Ok(Generation {
            text: self.text,
            prompt_token_count: timings.reused_token_count + timings.read_token_count,
            reused_token_count: timings.reused_token_count,
            generated_token_count,
            prompt_reading_duration: create_duration(timings.prompt_reading_milliseconds),
            generating_duration: create_duration(timings.generating_milliseconds),
        })
    }
}

fn create_duration(milliseconds: f64) -> Duration {
    Duration::from_secs_f64(milliseconds.max(0.0) / MILLISECONDS_PER_SECOND)
}
