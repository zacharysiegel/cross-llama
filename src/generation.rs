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

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// Formatted by the model's own chat template.
    pub messages: Vec<ChatMessage>,
    pub json_schema: Option<serde_json::Value>,
    /// Zero or below always picks the most likely token, which makes `seed` irrelevant.
    pub temperature: f32,
    pub seed: u32,
    pub maximum_token_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub text: String,
    pub prompt_token_count: usize,
    /// The part of `prompt_token_count` taken from the cache instead of read again.
    pub reused_token_count: usize,
    pub generated_token_count: usize,
    pub prompt_reading_duration: Duration,
    pub generating_duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenerationProgress {
    ReadingPrompt {
        /// Includes the reused tokens.
        read_token_count: usize,
        prompt_token_count: usize,
        reused_token_count: usize,
    },
    Generating {
        generated_token_count: usize,
        maximum_token_count: usize,
        text: String,
    },
}
