use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn chat_template_name(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
}

/// Rendered into the chat template as its `enable_thinking` variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingKind {
    Disabled,
    /// Needs `ThinkingCapability::separates_reasoning`.
    Enabled,
}

/// Rendered into the chat template as its `reasoning_effort` variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingEffortKind {
    Low,
    Medium,
    High,
}

impl ThinkingEffortKind {
    pub fn chat_template_value(self) -> &'static str {
        match self {
            ThinkingEffortKind::Low => "low",
            ThinkingEffortKind::Medium => "medium",
            ThinkingEffortKind::High => "high",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// Formatted by the model's own chat template; a message whose text contains one of the model's control tokens is rejected.
    pub messages: Vec<ChatMessage>,
    /// Constrains the answer only, never the reasoning.
    pub json_schema: Option<serde_json::Value>,
    /// Zero or below always picks the most likely token, which makes `seed` irrelevant.
    pub temperature: f32,
    pub seed: u32,
    /// The most tokens the model generates, reasoning and answer together; the delimiters around the reasoning are
    /// not counted.
    pub maximum_token_count: usize,
    pub thinking: ThinkingKind,
    /// The most reasoning tokens before the reasoning is closed and the answer begins; `None` leaves only
    /// `maximum_token_count`. Needs `ThinkingKind::Enabled`.
    pub thinking_budget_token_count: Option<usize>,
    /// Needs `ThinkingCapability::reads_thinking_effort`; `None` leaves the template's default.
    pub thinking_effort: Option<ThinkingEffortKind>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    /// The answer, without the reasoning.
    pub text: String,
    /// `None` when thinking was disabled.
    pub reasoning: Option<Reasoning>,
    pub prompt_token_count: usize,
    /// The part of `prompt_token_count` taken from the cache instead of read again.
    pub reused_token_count: usize,
    /// Counted as `GenerationRequest::maximum_token_count` counts them.
    pub generated_token_count: usize,
    pub prompt_reading_duration: Duration,
    /// Includes the reasoning.
    pub generating_duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reasoning {
    /// Without its delimiters or the whitespace around it.
    pub text: String,
    pub token_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenerationProgress {
    ReadingPrompt {
        /// Includes the reused tokens.
        read_token_count: usize,
        prompt_token_count: usize,
        reused_token_count: usize,
    },
    GeneratingReasoning {
        generated_token_count: usize,
        maximum_token_count: usize,
        /// The reasoning so far, as the model wrote it.
        text: String,
    },
    GeneratingAnswer {
        /// Includes the reasoning's tokens.
        generated_token_count: usize,
        maximum_token_count: usize,
        /// The answer so far.
        text: String,
    },
}

/// Which of `GenerationRequest`'s thinking controls a loaded model supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkingCapability {
    /// The crate knows the model's reasoning delimiters, which `ThinkingKind::Enabled` and a thinking budget need.
    pub separates_reasoning: bool,
    /// The chat template reads `enable_thinking`, so `ThinkingKind` turns the model's thinking on and off.
    pub switches_thinking: bool,
    /// The chat template reads `reasoning_effort`, which `thinking_effort` needs.
    pub reads_thinking_effort: bool,
}
