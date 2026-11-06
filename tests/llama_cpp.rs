// llama-cpp-sys-2 compiles llama.cpp with CMake and a native C++ compiler, which cannot target wasm32.
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use cross_llama::{
    ChatMessage,
    Error,
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
use futures::executor;

const TEST_MODEL_RELATIVE_PATH: &str = "build/models/gemma-3-270m-it-Q8_0.gguf";
const CONTEXT_TOKEN_COUNT: u32 = 4096;
const LONG_INSTRUCTION_SENTENCE: &str = "You are a patient tutor who grades short answers written by students of Portuguese. ";
const LONG_INSTRUCTION_REPETITION_COUNT: usize = 80;

fn create_load_options(sliding_window_cache: SlidingWindowCache) -> LoadOptions {
    LoadOptions {
        context_token_count: CONTEXT_TOKEN_COUNT,
        sliding_window_cache,
    }
}

fn create_model_source() -> ModelSource {
    ModelSource {
        path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_MODEL_RELATIVE_PATH),
    }
}

fn load_test_model(sliding_window_cache: SlidingWindowCache) -> Model {
    executor::block_on(Model::load(create_model_source(), &create_load_options(sliding_window_cache), |_: LoadProgress| {})).unwrap()
}

fn create_message(role: Role, text: &str) -> ChatMessage {
    ChatMessage {
        role,
        text: text.to_string(),
    }
}

/// Long enough to take more than one prompt batch.
fn create_long_instruction() -> String {
    LONG_INSTRUCTION_SENTENCE.repeat(LONG_INSTRUCTION_REPETITION_COUNT)
}

fn create_request(messages: Vec<ChatMessage>, json_schema: Option<serde_json::Value>, maximum_token_count: usize) -> GenerationRequest {
    GenerationRequest {
        messages,
        json_schema,
        temperature: 0.0,
        seed: 0,
        maximum_token_count,
    }
}

fn generate_ignoring_progress(model: &Model, request: &GenerationRequest) -> Generation {
    executor::block_on(model.generate(request, |_: GenerationProgress| {})).unwrap()
}

fn generate_after_previous_request_and_on_fresh_model(
    sliding_window_cache: SlidingWindowCache,
    previous_request: &GenerationRequest,
    request: &GenerationRequest,
) -> (Generation, Generation) {
    let model: Model = load_test_model(sliding_window_cache);
    let fresh_model: Model = load_test_model(sliding_window_cache);

    generate_ignoring_progress(&model, previous_request);
    let generation: Generation = generate_ignoring_progress(&model, request);
    let fresh_generation: Generation = generate_ignoring_progress(&fresh_model, request);

    (generation, fresh_generation)
}

#[test]
fn load_reports_increasing_progress_up_to_whole() {
    let mut loaded_proportions: Vec<f32> = Vec::new();
    let load_outcome: Result<Model, Error> = executor::block_on(Model::load(create_model_source(), &create_load_options(SlidingWindowCache::Full), |progress: LoadProgress| {
        if let LoadProgress::Loading { loaded_proportion } = progress {
            loaded_proportions.push(loaded_proportion);
        }
    }));

    assert!(load_outcome.is_ok());
    assert!(loaded_proportions.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(loaded_proportions.last().copied(), Some(1.0));
}

#[test]
fn generate_returns_plain_reply_within_maximum() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let maximum_token_count: usize = 16;
    let request: GenerationRequest = create_request(vec![create_message(Role::User, "Say hello to me.")], None, maximum_token_count);

    let generation: Generation = generate_ignoring_progress(&model, &request);

    assert!(!generation.text.trim().is_empty());
    assert!((1..=maximum_token_count).contains(&generation.generated_token_count));
}

#[test]
fn generate_with_json_schema_returns_matching_json() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let json_schema: serde_json::Value = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string", "maxLength": 40 },
            "confident": { "type": "boolean" },
        },
        "required": ["answer", "confident"],
        "additionalProperties": false,
    });
    let request: GenerationRequest = create_request(
        vec![create_message(Role::User, "What is the capital of Portugal? Reply in JSON.")],
        Some(json_schema),
        100,
    );

    let generation: Generation = generate_ignoring_progress(&model, &request);
    let reply: serde_json::Value = serde_json::from_str(&generation.text).unwrap();
    let reply_object: &serde_json::Map<String, serde_json::Value> = reply.as_object().unwrap();

    assert_eq!(reply_object.len(), 2);
    assert!(reply_object["answer"].as_str().is_some_and(|answer| answer.chars().count() <= 40));
    assert!(reply_object["confident"].is_boolean());
}

#[test]
fn generate_reuses_prompt_prefix_shared_with_previous_request() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let long_instruction: String = create_long_instruction();
    let warm_up_request: GenerationRequest = create_request(vec![create_message(Role::User, "Count to three.")], None, 4);
    let first_request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &long_instruction),
            create_message(Role::User, "The student wrote: Eu sou cansado."),
        ],
        None,
        4,
    );
    let second_request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &long_instruction),
            create_message(Role::User, "The student wrote: Ela tem fome."),
        ],
        None,
        4,
    );

    generate_ignoring_progress(&model, &warm_up_request);
    let first_generation: Generation = generate_ignoring_progress(&model, &first_request);
    let second_generation: Generation = generate_ignoring_progress(&model, &second_request);

    assert!(second_generation.reused_token_count > first_generation.reused_token_count);
    assert!(second_generation.reused_token_count < second_generation.prompt_token_count);
    assert!(second_generation.prompt_reading_duration < first_generation.prompt_reading_duration);
}

#[test]
fn generate_with_reused_prefix_replies_as_fresh_model_does() {
    let long_instruction: String = create_long_instruction();
    let previous_request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &long_instruction),
            create_message(Role::User, "The student wrote: Eu sou cansado."),
        ],
        None,
        16,
    );
    let request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &long_instruction),
            create_message(Role::User, "The student wrote: Ela tem fome."),
        ],
        None,
        16,
    );

    let (generation, fresh_generation): (Generation, Generation) =
        generate_after_previous_request_and_on_fresh_model(SlidingWindowCache::Full, &previous_request, &request);

    assert!(generation.reused_token_count > 0);
    assert_eq!(generation.text, fresh_generation.text);
}

#[test]
fn generate_after_window_only_cache_dropped_early_positions_replies_as_fresh_model_does() {
    let previous_request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &create_long_instruction()),
            create_message(Role::User, "Write one sentence about the sea."),
        ],
        None,
        4,
    );
    let request: GenerationRequest = create_request(
        vec![create_message(Role::User, "Name three colours of the rainbow, one per line.")],
        None,
        16,
    );

    let (generation, fresh_generation): (Generation, Generation) =
        generate_after_previous_request_and_on_fresh_model(SlidingWindowCache::WindowOnly, &previous_request, &request);

    assert_eq!(generation.reused_token_count, 0);
    assert_eq!(generation.text, fresh_generation.text);
}

#[test]
fn generate_reports_prompt_progress_then_token_progress_in_order() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = create_request(
        vec![
            create_message(Role::System, &create_long_instruction()),
            create_message(Role::User, "Write one sentence about the sea."),
        ],
        None,
        24,
    );
    let mut progress_reports: Vec<GenerationProgress> = Vec::new();

    let generation: Generation =
        executor::block_on(model.generate(&request, |progress: GenerationProgress| progress_reports.push(progress))).unwrap();

    let first_generating_index: usize = progress_reports
        .iter()
        .position(|progress| matches!(progress, GenerationProgress::Generating { .. }))
        .unwrap();
    let (reading_reports, generating_reports): (&[GenerationProgress], &[GenerationProgress]) = progress_reports.split_at(first_generating_index);
    let read_token_counts: Vec<usize> = reading_reports
        .iter()
        .map(|progress| match progress {
            GenerationProgress::ReadingPrompt { read_token_count, .. } => *read_token_count,
            GenerationProgress::Generating { .. } => panic!(),
        })
        .collect();
    let generated_token_counts: Vec<usize> = generating_reports
        .iter()
        .map(|progress| match progress {
            GenerationProgress::Generating { generated_token_count, .. } => *generated_token_count,
            GenerationProgress::ReadingPrompt { .. } => panic!(),
        })
        .collect();
    let expected_generated_token_counts: Vec<usize> = (1..=generation.generated_token_count).collect();
    let last_reported_text: Option<&String> = generating_reports.last().map(|progress| match progress {
        GenerationProgress::Generating { text, .. } => text,
        GenerationProgress::ReadingPrompt { .. } => panic!(),
    });

    assert!(read_token_counts.len() > 1);
    assert!(read_token_counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(read_token_counts.last().copied(), Some(generation.prompt_token_count));
    assert_eq!(generated_token_counts, expected_generated_token_counts);
    assert_eq!(last_reported_text, Some(&generation.text));
}

#[test]
fn cancel_stops_generation_in_progress() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let maximum_token_count: usize = 200;
    let request: GenerationRequest = create_request(
        vec![create_message(Role::User, "Tell me a long story about a lighthouse keeper.")],
        None,
        maximum_token_count,
    );
    let mut generated_token_count_at_last_report: usize = 0;

    let generation_outcome: Result<Generation, GenerationError> = executor::block_on(model.generate(&request, |progress: GenerationProgress| {
        if let GenerationProgress::Generating { generated_token_count, .. } = progress {
            generated_token_count_at_last_report = generated_token_count;
            model.cancel();
        }
    }));

    assert!(matches!(generation_outcome, Err(GenerationError::Cancelled)));
    assert!(generated_token_count_at_last_report < maximum_token_count);
}

#[test]
fn generate_after_cancel_completes() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = create_request(vec![create_message(Role::User, "Say hello to me.")], None, 8);

    model.cancel();
    let generation_outcome: Result<Generation, GenerationError> = executor::block_on(model.generate(&request, |_: GenerationProgress| {}));

    assert!(generation_outcome.is_ok());
}
