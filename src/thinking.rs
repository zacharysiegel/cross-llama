//! Separates a reasoning model's reasoning from its answer, which it generates after the reasoning's end delimiter.

use crate::chat_template::{ChatTemplate, ENABLE_THINKING_VARIABLE_NAME, THINKING_EFFORT_VARIABLE_NAME};
use crate::{Error, GenerationRequest, Reasoning, ThinkingCapability, ThinkingKind};

/* The thinking start and end tags of llama.cpp's chat parsers (common/chat.cpp and common/parsers/ in llama.cpp as
   llama-cpp-sys-2 0.1.161 bundles it), for the forms whose answer follows the reasoning. The answer prefix is what
   llama.cpp writes between the end delimiter and the answer when it continues a reply. */
const KNOWN_REASONING_DELIMITERS: [ReasoningDelimiters; 4] = [
    // Gemma 4
    ReasoningDelimiters {
        start: "<|channel>thought",
        end: "<channel|>",
        answer_prefix: "",
    },
    // gpt-oss, whose answer is a message of its own on the final channel
    ReasoningDelimiters {
        start: "<|channel|>analysis<|message|>",
        end: "<|end|>",
        answer_prefix: "<|start|>assistant<|channel|>final<|message|>",
    },
    // Ministral 3 and Magistral
    ReasoningDelimiters {
        start: "[THINK]",
        end: "[/THINK]",
        answer_prefix: "",
    },
    // Qwen3, DeepSeek-R1, Kimi K2, LFM2, MiniCPM5 and Ling 3
    ReasoningDelimiters {
        start: "<think>",
        end: "</think>",
        answer_prefix: "",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReasoningDelimiters {
    pub start: &'static str,
    /// One token of the model's vocabulary, so the backends can stop at it and count it.
    pub end: &'static str,
    pub answer_prefix: &'static str,
}

/// What a loaded model's chat template and vocabulary allow of `GenerationRequest`'s thinking controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelThinking {
    reasoning_delimiters: Option<ReasoningDelimiters>,
    switches_thinking: bool,
    reads_thinking_effort: bool,
}

impl ModelThinking {
    pub fn new(chat_template: &ChatTemplate, is_single_token: impl Fn(&str) -> bool) -> Self {
        ModelThinking {
            reasoning_delimiters: find_reasoning_delimiters(chat_template.source(), is_single_token),
            switches_thinking: chat_template.reads_variable(ENABLE_THINKING_VARIABLE_NAME),
            reads_thinking_effort: chat_template.reads_variable(THINKING_EFFORT_VARIABLE_NAME),
        }
    }

    /// Separates reasoning with `reasoning_delimiters` and reads no thinking variable.
    #[cfg(test)] // stands in delimiters for a test model that has none
    pub fn with_reasoning_delimiters(reasoning_delimiters: ReasoningDelimiters) -> Self {
        ModelThinking {
            reasoning_delimiters: Some(reasoning_delimiters),
            switches_thinking: false,
            reads_thinking_effort: false,
        }
    }

    pub fn capability(&self) -> ThinkingCapability {
        ThinkingCapability {
            separates_reasoning: self.reasoning_delimiters.is_some(),
            switches_thinking: self.switches_thinking,
            reads_thinking_effort: self.reads_thinking_effort,
        }
    }

    /// Fails for a thinking control the model does not support or the request's other controls rule out.
    pub fn check_request(&self, request: &GenerationRequest) -> Result<(), Error> {
        if request.thinking == ThinkingKind::Enabled && self.reasoning_delimiters.is_none() {
            return Err(Error::new("thinking cannot be enabled: cross-llama does not know this model's reasoning delimiters"));
        }

        if request.thinking == ThinkingKind::Disabled && request.thinking_budget_token_count.is_some() {
            return Err(Error::new("a thinking budget needs thinking enabled"));
        }

        if request.thinking_effort.is_some() && !self.reads_thinking_effort {
            return Err(Error::new(&format!(
                "a thinking effort needs a chat template that reads `{THINKING_EFFORT_VARIABLE_NAME}`, which this model's does not",
            )));
        }

        Ok(())
    }

    /// `None` when the request has thinking disabled.
    pub fn plan_reasoning(&self, request: &GenerationRequest, prompt: &str) -> Option<ReasoningPlan> {
        let (ThinkingKind::Enabled, Some(reasoning_delimiters)) = (request.thinking, self.reasoning_delimiters)
        else {
            return None;
        };

        let maximum_token_count: usize = request.maximum_token_count;

        Some(ReasoningPlan {
            reasoning_prompt: create_reasoning_prompt(prompt, &reasoning_delimiters),
            reasoning_token_limit: request.thinking_budget_token_count.unwrap_or(maximum_token_count).min(maximum_token_count),
            maximum_token_count,
            reasoning_delimiters,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningPlan {
    /// Ends inside the reasoning, after its start delimiter.
    pub reasoning_prompt: String,
    pub reasoning_token_limit: usize,
    maximum_token_count: usize,
    reasoning_delimiters: ReasoningDelimiters,
}

impl ReasoningPlan {
    pub fn end_delimiter(&self) -> &'static str {
        self.reasoning_delimiters.end
    }

    /// Ends where the answer begins; `reasoning_reply_text` is the reasoning as generated, without its end delimiter.
    pub fn create_answer_prompt(&self, reasoning_reply_text: &str) -> String {
        format!(
            "{}{reasoning_reply_text}{}{}",
            self.reasoning_prompt, self.reasoning_delimiters.end, self.reasoning_delimiters.answer_prefix,
        )
    }

    /// Zero when the reasoning has used the whole maximum.
    pub fn answer_token_limit(&self, reasoning_token_count: usize) -> usize {
        self.maximum_token_count.saturating_sub(reasoning_token_count)
    }
}

pub fn create_reasoning(reasoning_reply_text: &str, token_count: usize) -> Reasoning {
    Reasoning {
        text: reasoning_reply_text.trim().to_string(),
        token_count,
    }
}

/// The first known delimiters whose start the template writes and whose end is one token of the model's vocabulary.
fn find_reasoning_delimiters(template_source: &str, is_single_token: impl Fn(&str) -> bool) -> Option<ReasoningDelimiters> {
    KNOWN_REASONING_DELIMITERS
        .into_iter()
        .find(|reasoning_delimiters: &ReasoningDelimiters| template_source.contains(reasoning_delimiters.start) && is_single_token(reasoning_delimiters.end))
}

/// Some templates already open the reasoning at the end of the prompt.
fn create_reasoning_prompt(prompt: &str, reasoning_delimiters: &ReasoningDelimiters) -> String {
    let reasoning_already_open: bool = prompt.trim_end().ends_with(reasoning_delimiters.start);

    match reasoning_already_open {
        true => prompt.to_string(),
        false => format!("{prompt}{}", reasoning_delimiters.start),
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_template::{ChatTemplate, SequenceTokens};
    use crate::thinking::{ModelThinking, ReasoningPlan};
    use crate::{ChatMessage, Error, GenerationRequest, Role, ThinkingCapability, ThinkingEffortKind, ThinkingKind};

    const GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE: &str = concat!(
        "{%- set enable_thinking = enable_thinking | default(false) -%}",
        "{{- '<|turn>model\\n' -}}",
        "{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}",
    );
    const OPENING_THINK_TEMPLATE_SOURCE: &str = "<|im_start|>assistant\n<think>\n";
    const GPT_OSS_GENERATION_PROMPT_TEMPLATE_SOURCE: &str = concat!(
        "{{- 'Reasoning: ' + reasoning_effort | default('medium') -}}",
        "{%- if message.thinking -%}{{- '<|channel|>analysis<|message|>' + message.thinking + '<|end|>' -}}{%- endif -%}",
        "{{- '<|start|>assistant' -}}",
    );

    fn create_model_thinking(template_source: &str, single_token_texts: &[&str]) -> ModelThinking {
        let sequence_tokens: SequenceTokens = SequenceTokens {
            beginning_text: String::new(),
            end_text: String::new(),
            tokenizer_adds_beginning: false,
            tokenizer_adds_end: false,
        };
        let chat_template: ChatTemplate = ChatTemplate::new(Some(template_source.to_string()), sequence_tokens).unwrap();

        ModelThinking::new(&chat_template, |text: &str| single_token_texts.contains(&text))
    }

    fn create_request(thinking: ThinkingKind, thinking_budget_token_count: Option<usize>, thinking_effort: Option<ThinkingEffortKind>) -> GenerationRequest {
        GenerationRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                text: "Olá".to_string(),
            }],
            json_schema: None,
            temperature: 0.0,
            seed: 0,
            maximum_token_count: 100,
            thinking,
            thinking_budget_token_count,
            thinking_effort,
        }
    }

    fn plan_gemma_4_reasoning(thinking_budget_token_count: Option<usize>) -> ReasoningPlan {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<channel|>"]);
        let request: GenerationRequest = create_request(ThinkingKind::Enabled, thinking_budget_token_count, None);

        model_thinking.plan_reasoning(&request, "<|turn>model\n").unwrap()
    }

    #[test]
    fn capability_reports_gemma_4_thinking_controls() {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<channel|>"]);

        let expected_capability: ThinkingCapability = ThinkingCapability {
            separates_reasoning: true,
            switches_thinking: true,
            reads_thinking_effort: false,
        };
        assert_eq!(model_thinking.capability(), expected_capability);
    }

    #[test]
    fn capability_reports_thinking_effort_template_reads() {
        let model_thinking: ModelThinking = create_model_thinking(GPT_OSS_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<|end|>"]);

        let expected_capability: ThinkingCapability = ThinkingCapability {
            separates_reasoning: true,
            switches_thinking: false,
            reads_thinking_effort: true,
        };
        assert_eq!(model_thinking.capability(), expected_capability);
    }

    #[test]
    fn capability_does_not_separate_reasoning_whose_end_delimiter_is_not_one_token() {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &[]);

        assert!(!model_thinking.capability().separates_reasoning);
    }

    #[test]
    fn check_request_rejects_enabled_thinking_without_known_delimiters() {
        let model_thinking: ModelThinking = create_model_thinking("{{ enable_thinking }}", &[]);

        let checked: Result<(), Error> = model_thinking.check_request(&create_request(ThinkingKind::Enabled, None, None));

        assert!(checked.is_err());
    }

    #[test]
    fn check_request_rejects_budget_with_thinking_disabled() {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<channel|>"]);

        let checked: Result<(), Error> = model_thinking.check_request(&create_request(ThinkingKind::Disabled, Some(16), None));

        assert!(checked.is_err());
    }

    #[test]
    fn check_request_rejects_effort_template_does_not_read() {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<channel|>"]);

        let checked: Result<(), Error> = model_thinking.check_request(&create_request(ThinkingKind::Enabled, None, Some(ThinkingEffortKind::Low)));

        assert!(checked.is_err());
    }

    #[test]
    fn plan_reasoning_opens_reasoning_after_prompt() {
        let reasoning_plan: ReasoningPlan = plan_gemma_4_reasoning(None);

        assert_eq!(reasoning_plan.reasoning_prompt, "<|turn>model\n<|channel>thought");
    }

    #[test]
    fn plan_reasoning_keeps_reasoning_template_already_opened() {
        let model_thinking: ModelThinking = create_model_thinking(OPENING_THINK_TEMPLATE_SOURCE, &["</think>"]);
        let request: GenerationRequest = create_request(ThinkingKind::Enabled, None, None);

        let reasoning_plan: ReasoningPlan = model_thinking.plan_reasoning(&request, OPENING_THINK_TEMPLATE_SOURCE).unwrap();

        assert_eq!(reasoning_plan.reasoning_prompt, OPENING_THINK_TEMPLATE_SOURCE);
    }

    #[test]
    fn plan_reasoning_limits_reasoning_to_budget_within_maximum() {
        let budgeted_plan: ReasoningPlan = plan_gemma_4_reasoning(Some(16));
        let over_maximum_plan: ReasoningPlan = plan_gemma_4_reasoning(Some(1000));
        let unlimited_plan: ReasoningPlan = plan_gemma_4_reasoning(None);

        assert_eq!(budgeted_plan.reasoning_token_limit, 16);
        assert_eq!(over_maximum_plan.reasoning_token_limit, 100);
        assert_eq!(unlimited_plan.reasoning_token_limit, 100);
    }

    #[test]
    fn create_answer_prompt_closes_reasoning_and_writes_answer_prefix() {
        let model_thinking: ModelThinking = create_model_thinking(GPT_OSS_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<|end|>"]);
        let request: GenerationRequest = create_request(ThinkingKind::Enabled, None, None);
        let reasoning_plan: ReasoningPlan = model_thinking.plan_reasoning(&request, "<|start|>assistant").unwrap();

        let answer_prompt: String = reasoning_plan.create_answer_prompt("Think.");

        assert_eq!(
            answer_prompt,
            "<|start|>assistant<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>",
        );
    }

    #[test]
    fn plan_reasoning_is_absent_with_thinking_disabled() {
        let model_thinking: ModelThinking = create_model_thinking(GEMMA_4_GENERATION_PROMPT_TEMPLATE_SOURCE, &["<channel|>"]);

        let reasoning_plan: Option<ReasoningPlan> = model_thinking.plan_reasoning(&create_request(ThinkingKind::Disabled, None, None), "<|turn>model\n");

        assert!(reasoning_plan.is_none());
    }
}
