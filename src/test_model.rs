use std::path::PathBuf;

use crate::thinking::ReasoningDelimiters;
use crate::{ChatMessage, Generation, GenerationProgress, GenerationRequest, Reasoning, Role, ThinkingKind};

// `scripts/setup.sh` fetches it.
const TEST_MODEL_RELATIVE_PATH: &str = "build/models/gemma-3-270m-it-Q8_0.gguf";
/* The test model has no reasoning delimiters. These stand in for them: the reasoning is the reply as the model begins it,
   and its end is a token the model never writes, so the budget ends it. */
pub const STAND_IN_REASONING_DELIMITERS: ReasoningDelimiters = ReasoningDelimiters {
    start: "",
    end: "<unused0>",
    answer_prefix: "",
};
pub const THINKING_BUDGET_TOKEN_COUNT: usize = 8;
const CAPITAL_ANSWER_MAXIMUM_CHARACTER_COUNT: usize = 40;

pub fn get_test_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_MODEL_RELATIVE_PATH)
}

pub fn create_budgeted_thinking_request() -> GenerationRequest {
    GenerationRequest {
        messages: vec![ChatMessage {
            role: Role::User,
            text: "What is the capital of Portugal? Reply in JSON.".to_string(),
        }],
        json_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string", "maxLength": CAPITAL_ANSWER_MAXIMUM_CHARACTER_COUNT },
                "confident": { "type": "boolean" },
            },
            "required": ["answer", "confident"],
            "additionalProperties": false,
        })),
        temperature: 0.0,
        seed: 0,
        maximum_token_count: 100,
        thinking: ThinkingKind::Enabled,
        thinking_budget_token_count: Some(THINKING_BUDGET_TOKEN_COUNT),
        thinking_effort: None,
    }
}

pub fn assert_reasoning_ends_at_budget_and_answer_matches_schema(generation: &Generation) {
    let reasoning: &Reasoning = generation.reasoning.as_ref().unwrap();
    let answer: serde_json::Value = serde_json::from_str(&generation.text).unwrap();
    let answer_object: &serde_json::Map<String, serde_json::Value> = answer.as_object().unwrap();

    assert_eq!(reasoning.token_count, THINKING_BUDGET_TOKEN_COUNT);
    assert!(!reasoning.text.is_empty());
    assert!(generation.generated_token_count > reasoning.token_count);
    assert_eq!(answer_object.len(), 2);
    assert!(answer_object["confident"].is_boolean());
}

pub fn assert_reasoning_progress_precedes_answer_progress(progress_reports: &[GenerationProgress], generation: &Generation) {
    let first_reasoning_index: usize = progress_reports
        .iter()
        .position(|progress| matches!(progress, GenerationProgress::GeneratingReasoning { .. }))
        .unwrap();
    let first_answer_index: usize = progress_reports
        .iter()
        .position(|progress| matches!(progress, GenerationProgress::GeneratingAnswer { .. }))
        .unwrap();
    let reasoning_token_counts: Vec<usize> = progress_reports[first_reasoning_index..first_answer_index]
        .iter()
        .map(|progress| match progress {
            GenerationProgress::GeneratingReasoning { generated_token_count, .. } => *generated_token_count,
            GenerationProgress::ReadingPrompt { .. } | GenerationProgress::GeneratingAnswer { .. } => panic!(),
        })
        .collect();
    let answer_token_counts: Vec<usize> = progress_reports[first_answer_index..]
        .iter()
        .map(|progress| match progress {
            GenerationProgress::GeneratingAnswer { generated_token_count, .. } => *generated_token_count,
            GenerationProgress::ReadingPrompt { .. } | GenerationProgress::GeneratingReasoning { .. } => panic!(),
        })
        .collect();

    // The browser backend's counts can skip a token whose text the server holds back.
    assert!(progress_reports[..first_reasoning_index].iter().all(|progress| matches!(progress, GenerationProgress::ReadingPrompt { .. })));
    assert!(reasoning_token_counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(reasoning_token_counts.last().is_some_and(|last_count| *last_count <= THINKING_BUDGET_TOKEN_COUNT));
    assert!(answer_token_counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(answer_token_counts.first().is_some_and(|first_count| *first_count > THINKING_BUDGET_TOKEN_COUNT));
    assert!(answer_token_counts.last().is_some_and(|last_count| *last_count <= generation.generated_token_count));
}
