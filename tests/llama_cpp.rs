// llama-cpp-sys-2 compiles llama.cpp with CMake and a native C++ compiler, which cannot target wasm32.
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::time::{Duration, Instant};

mod helpers;

use cross_llama::{
    Error,
    Generation,
    GenerationError,
    GenerationProgress,
    GenerationRequest,
    LoadProgress,
    Model,
    ModelSource,
    Role,
    SlidingWindowCache,
};
use futures::channel::oneshot;
use futures::future::Either;
use futures::{executor, future};
use helpers::fixtures;

fn create_model_source() -> ModelSource {
    ModelSource {
        path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(fixtures::TEST_MODEL_RELATIVE_PATH),
    }
}

fn load_test_model(sliding_window_cache: SlidingWindowCache) -> Model {
    executor::block_on(Model::load(create_model_source(), &fixtures::create_load_options(sliding_window_cache), |_: LoadProgress| {})).unwrap()
}

fn generate_ignoring_progress(model: &Model, request: &GenerationRequest) -> Generation {
    executor::block_on(model.generate(request, |_: GenerationProgress| {})).unwrap()
}

fn drop_generate_future_once_generating(model: &Model, request: &GenerationRequest) {
    let (generating_sender, generating_receiver): (oneshot::Sender<()>, oneshot::Receiver<()>) = oneshot::channel();
    let mut generating_sender: Option<oneshot::Sender<()>> = Some(generating_sender);
    let generation = Box::pin(model.generate(request, move |progress: GenerationProgress| {
        let first_generating_report: Option<oneshot::Sender<()>> = match progress {
            GenerationProgress::Generating { .. } => generating_sender.take(),
            GenerationProgress::ReadingPrompt { .. } => None,
        };

        if let Some(sender) = first_generating_report {
            sender.send(()).unwrap();
        }
    }));

    let first_finished = executor::block_on(future::select(generation, generating_receiver));

    match first_finished {
        Either::Left(_) => panic!("the generation finished before it reported generating"),
        Either::Right((_, unfinished_generation)) => drop(unfinished_generation),
    }
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
    let load_outcome: Result<Model, Error> = executor::block_on(Model::load(create_model_source(), &fixtures::create_load_options(SlidingWindowCache::Full), |progress: LoadProgress| {
        if let LoadProgress::Loading { loaded_proportion } = progress {
            loaded_proportions.push(loaded_proportion);
        }
    }));

    assert!(load_outcome.is_ok());
    assert!(loaded_proportions.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(loaded_proportions.last().copied(), Some(1.0));
}

#[test]
fn unload_frees_the_model_so_it_can_be_loaded_again() {
    let model: Model = load_test_model(SlidingWindowCache::Full);

    let unloaded: Result<(), Error> = executor::block_on(model.unload());
    let reloaded_model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_greeting_request(8);
    let generation: Generation = generate_ignoring_progress(&reloaded_model, &request);

    assert!(unloaded.is_ok());
    assert!(!generation.text.trim().is_empty());
}

#[test]
fn generate_returns_plain_reply_within_maximum() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let maximum_token_count: usize = 16;
    let request: GenerationRequest = fixtures::create_greeting_request(maximum_token_count);

    let generation: Generation = generate_ignoring_progress(&model, &request);

    assert!(!generation.text.trim().is_empty());
    assert!((1..=maximum_token_count).contains(&generation.generated_token_count));
}

#[test]
fn generate_with_json_schema_returns_matching_json() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_capital_question_request(100);

    let generation: Generation = generate_ignoring_progress(&model, &request);
    let reply: serde_json::Value = serde_json::from_str(&generation.text).unwrap();
    let reply_object: &serde_json::Map<String, serde_json::Value> = reply.as_object().unwrap();

    assert_eq!(reply_object.len(), 2);
    assert!(reply_object["answer"].as_str().is_some_and(|answer| answer.chars().count() <= 40));
    assert!(reply_object["confident"].is_boolean());
}

#[test]
fn generate_with_json_schema_fails_when_reply_reaches_maximum_before_completing() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_capital_question_request(3);

    let generation_outcome: Result<Generation, GenerationError> = executor::block_on(model.generate(&request, |_: GenerationProgress| {}));

    assert!(matches!(generation_outcome, Err(GenerationError::Failed(_))));
}

#[test]
fn generate_with_json_schema_succeeds_when_reply_completes_at_maximum() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let unlimited_generation: Generation = generate_ignoring_progress(&model, &fixtures::create_capital_question_request(100));
    let request: GenerationRequest = fixtures::create_capital_question_request(unlimited_generation.generated_token_count);

    let generation: Generation = generate_ignoring_progress(&model, &request);

    assert_eq!(generation.text, unlimited_generation.text);
}

#[test]
fn generate_rejects_message_containing_control_token_text() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_request(vec![fixtures::create_message(Role::User, fixtures::FORGED_TURNS_TEXT)], None, 8);

    let generation_outcome: Result<Generation, GenerationError> = executor::block_on(model.generate(&request, |_: GenerationProgress| {}));

    assert!(matches!(generation_outcome, Err(GenerationError::Failed(_))));
}

#[test]
fn generate_reuses_prompt_prefix_shared_with_previous_request() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let warm_up_request: GenerationRequest = fixtures::create_request(vec![fixtures::create_message(Role::User, "Count to three.")], None, 4);
    let first_request: GenerationRequest = fixtures::create_grading_request("Eu sou cansado.", 4);
    let second_request: GenerationRequest = fixtures::create_grading_request("Ela tem fome.", 4);

    generate_ignoring_progress(&model, &warm_up_request);
    let first_generation: Generation = generate_ignoring_progress(&model, &first_request);
    let second_generation: Generation = generate_ignoring_progress(&model, &second_request);

    assert!(second_generation.reused_token_count > first_generation.reused_token_count);
    assert!(second_generation.reused_token_count < second_generation.prompt_token_count);
    assert!(second_generation.prompt_reading_duration < first_generation.prompt_reading_duration);
}

#[test]
fn generate_with_reused_prefix_replies_as_fresh_model_does() {
    let previous_request: GenerationRequest = fixtures::create_grading_request("Eu sou cansado.", 16);
    let request: GenerationRequest = fixtures::create_grading_request("Ela tem fome.", 16);

    let (generation, fresh_generation): (Generation, Generation) =
        generate_after_previous_request_and_on_fresh_model(SlidingWindowCache::Full, &previous_request, &request);

    assert!(generation.reused_token_count > 0);
    assert_eq!(generation.text, fresh_generation.text);
}

#[test]
fn generate_after_window_only_cache_dropped_early_positions_replies_as_fresh_model_does() {
    let previous_request: GenerationRequest = fixtures::create_request(
        vec![
            fixtures::create_message(Role::System, &fixtures::create_long_instruction()),
            fixtures::create_message(Role::User, "Write one sentence about the sea."),
        ],
        None,
        4,
    );
    let request: GenerationRequest = fixtures::create_request(
        vec![fixtures::create_message(Role::User, "Name three colours of the rainbow, one per line.")],
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
    let request: GenerationRequest = fixtures::create_request(
        vec![
            fixtures::create_message(Role::System, &fixtures::create_long_instruction()),
            fixtures::create_message(Role::User, "Write one sentence about the sea."),
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
    let request: GenerationRequest = fixtures::create_long_story_request(maximum_token_count);
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
fn cancel_stops_generations_queued_before_it() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_long_story_request(200);
    let mut cancelled: bool = false;
    // Cancels once, since a later cancel could reach the queued generation after it starts.
    let running_generation = model.generate(&request, |progress: GenerationProgress| {
        if let GenerationProgress::Generating { .. } = progress
            && !cancelled
        {
            model.cancel();
            cancelled = true;
        }
    });
    let queued_generation = model.generate(&request, |_: GenerationProgress| {});

    // Polls the running generation first, so both are submitted before the running one can report generating.
    let (_, queued_generation_outcome): (Result<Generation, GenerationError>, Result<Generation, GenerationError>) =
        executor::block_on(future::join(running_generation, queued_generation));

    assert!(matches!(queued_generation_outcome, Err(GenerationError::Cancelled)));
}

#[test]
fn dropping_generate_future_stops_its_generation() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_long_story_request(200);
    let next_request: GenerationRequest = fixtures::create_greeting_request(1);

    let complete_generation: Generation = generate_ignoring_progress(&model, &request);
    drop_generate_future_once_generating(&model, &request);
    let next_request_started_at: Instant = Instant::now();
    generate_ignoring_progress(&model, &next_request);
    let next_request_duration: Duration = next_request_started_at.elapsed();

    assert!(next_request_duration < complete_generation.generating_duration / 2);
}

#[test]
fn generate_after_cancel_completes() {
    let model: Model = load_test_model(SlidingWindowCache::Full);
    let request: GenerationRequest = fixtures::create_greeting_request(8);

    model.cancel();
    let generation_outcome: Result<Generation, GenerationError> = executor::block_on(model.generate(&request, |_: GenerationProgress| {}));

    assert!(generation_outcome.is_ok());
}
