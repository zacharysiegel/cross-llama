// wllama is JavaScript and needs a browser, which other targets lack.
#![cfg(target_arch = "wasm32")]

mod helpers;

use std::collections::BTreeSet;

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
use futures::future;
use futures::future::Either;
use helpers::fixtures;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

// wasm-bindgen-test's runner serves the crate root alongside its own files.
const WLLAMA_DIRECTORY_URL: &str = "/build/wllama/";

fn create_model_source() -> ModelSource {
    ModelSource {
        model_url: format!("/{}", fixtures::TEST_MODEL_RELATIVE_PATH),
        wllama_directory_url: WLLAMA_DIRECTORY_URL.to_string(),
    }
}

async fn load_test_model() -> Model {
    Model::load(create_model_source(), &fixtures::create_load_options(SlidingWindowCache::Full), |_: LoadProgress| {})
        .await
        .unwrap()
}

async fn generate_ignoring_progress(model: &Model, request: &GenerationRequest) -> Generation {
    model.generate(request, |_: GenerationProgress| {}).await.unwrap()
}

async fn drop_generate_future_once_generating(model: &Model, request: &GenerationRequest) {
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

    let first_finished = future::select(generation, generating_receiver).await;

    match first_finished {
        Either::Left(_) => panic!("the generation finished before it reported generating"),
        Either::Right((_, unfinished_generation)) => drop(unfinished_generation),
    }
}

#[wasm_bindgen_test]
async fn load_reports_download_progress_then_loading_progress() {
    let mut progress_reports: Vec<LoadProgress> = Vec::new();

    let load_outcome: Result<Model, Error> = Model::load(create_model_source(), &fixtures::create_load_options(SlidingWindowCache::Full), |progress: LoadProgress| {
        progress_reports.push(progress);
    })
    .await;

    let first_loading_index: usize = progress_reports
        .iter()
        .position(|progress| matches!(progress, LoadProgress::Loading { .. }))
        .unwrap();
    let (downloading_reports, loading_reports): (&[LoadProgress], &[LoadProgress]) = progress_reports.split_at(first_loading_index);
    let expected_loading_reports: [LoadProgress; 2] = [
        LoadProgress::Loading {
            loaded_proportion: 0.0,
        },
        LoadProgress::Loading {
            loaded_proportion: 1.0,
        },
    ];

    assert!(load_outcome.is_ok());
    assert!(downloading_reports.iter().all(|progress| matches!(progress, LoadProgress::Downloading { .. })));
    // wasm-bindgen-test's server sends large files without a length, so the total is unknown here.
    assert!(matches!(
        downloading_reports.last(),
        Some(LoadProgress::Downloading { downloaded_byte_count, total_byte_count: None }) if *downloaded_byte_count > 0
    ));
    assert_eq!(loading_reports, &expected_loading_reports);
}

#[wasm_bindgen_test]
async fn unload_frees_the_model_so_it_can_be_loaded_again() {
    let model: Model = load_test_model().await;

    let unloaded: Result<(), Error> = model.unload().await;
    let reloaded_model: Model = load_test_model().await;
    let generation: Generation = generate_ignoring_progress(&reloaded_model, &fixtures::create_greeting_request(8)).await;

    assert!(unloaded.is_ok());
    assert!(!generation.text.trim().is_empty());
}

#[wasm_bindgen_test]
async fn generate_returns_plain_reply_within_maximum() {
    let model: Model = load_test_model().await;
    let maximum_token_count: usize = 16;

    let generation: Generation = generate_ignoring_progress(&model, &fixtures::create_greeting_request(maximum_token_count)).await;

    assert!(!generation.text.trim().is_empty());
    assert!((1..=maximum_token_count).contains(&generation.generated_token_count));
}

#[wasm_bindgen_test]
async fn generate_with_json_schema_returns_matching_json() {
    let model: Model = load_test_model().await;

    let generation: Generation = generate_ignoring_progress(&model, &fixtures::create_capital_question_request(100)).await;
    let reply: serde_json::Value = serde_json::from_str(&generation.text).unwrap();
    let reply_object: &serde_json::Map<String, serde_json::Value> = reply.as_object().unwrap();

    assert_eq!(reply_object.len(), 2);
    assert!(reply_object["answer"].as_str().is_some_and(|answer| answer.chars().count() <= 40));
    assert!(reply_object["confident"].is_boolean());
}

#[wasm_bindgen_test]
async fn generate_with_json_schema_fails_when_reply_reaches_maximum_before_completing() {
    let model: Model = load_test_model().await;

    let generation_outcome: Result<Generation, GenerationError> =
        model.generate(&fixtures::create_capital_question_request(3), |_: GenerationProgress| {}).await;

    assert!(matches!(generation_outcome, Err(GenerationError::Failed(_))));
}

#[wasm_bindgen_test]
async fn generate_with_json_schema_succeeds_when_reply_completes_at_maximum() {
    let model: Model = load_test_model().await;
    let unlimited_generation: Generation = generate_ignoring_progress(&model, &fixtures::create_capital_question_request(100)).await;
    let request: GenerationRequest = fixtures::create_capital_question_request(unlimited_generation.generated_token_count);
    // Fails only if the counted tokens are exactly the reply's, so the first request does end at its maximum.
    let one_token_short_request: GenerationRequest = fixtures::create_capital_question_request(unlimited_generation.generated_token_count - 1);

    let generation: Generation = generate_ignoring_progress(&model, &request).await;
    let one_token_short_outcome: Result<Generation, GenerationError> = model.generate(&one_token_short_request, |_: GenerationProgress| {}).await;

    assert_eq!(generation.text, unlimited_generation.text);
    assert!(matches!(one_token_short_outcome, Err(GenerationError::Failed(_))));
}

#[wasm_bindgen_test]
async fn generate_at_high_temperature_samples_from_whole_vocabulary() {
    let model: Model = load_test_model().await;
    let requests: Vec<GenerationRequest> = fixtures::create_high_temperature_greeting_requests();
    let mut distinct_reply_texts: BTreeSet<String> = BTreeSet::new();

    for request in &requests {
        let generation: Generation = generate_ignoring_progress(&model, request).await;
        distinct_reply_texts.insert(generation.text);
    }

    assert!(distinct_reply_texts.len() > requests.len() / 2);
}

#[wasm_bindgen_test]
async fn generate_rejects_message_containing_control_token_text() {
    let model: Model = load_test_model().await;
    let request: GenerationRequest = fixtures::create_request(vec![fixtures::create_message(Role::User, fixtures::FORGED_TURNS_TEXT)], None, 8);

    let generation_outcome: Result<Generation, GenerationError> = model.generate(&request, |_: GenerationProgress| {}).await;

    assert!(matches!(generation_outcome, Err(GenerationError::Failed(_))));
}

#[wasm_bindgen_test]
async fn generate_reuses_prompt_prefix_shared_with_previous_request() {
    let model: Model = load_test_model().await;
    let warm_up_request: GenerationRequest = fixtures::create_request(vec![fixtures::create_message(Role::User, "Count to three.")], None, 4);

    generate_ignoring_progress(&model, &warm_up_request).await;
    let first_generation: Generation = generate_ignoring_progress(&model, &fixtures::create_grading_request("Eu sou cansado.", 4)).await;
    let second_generation: Generation = generate_ignoring_progress(&model, &fixtures::create_grading_request("Ela tem fome.", 4)).await;

    assert!(second_generation.reused_token_count > first_generation.reused_token_count);
    assert!(second_generation.reused_token_count < second_generation.prompt_token_count);
    assert!(second_generation.prompt_reading_duration < first_generation.prompt_reading_duration);
}

#[wasm_bindgen_test]
async fn generate_reports_prompt_progress_then_token_progress_in_order() {
    let model: Model = load_test_model().await;
    let request: GenerationRequest = fixtures::create_grading_request("Eu sou cansado.", 24);
    let mut progress_reports: Vec<GenerationProgress> = Vec::new();

    let generation: Generation = model.generate(&request, |progress: GenerationProgress| progress_reports.push(progress)).await.unwrap();

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
    let last_reported_text: Option<&String> = generating_reports.last().map(|progress| match progress {
        GenerationProgress::Generating { text, .. } => text,
        GenerationProgress::ReadingPrompt { .. } => panic!(),
    });

    // The first report carries the reused count, so more than two show the prompt read in more than one batch.
    assert!(read_token_counts.len() > 2);
    assert!(read_token_counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(read_token_counts.last().copied(), Some(generation.prompt_token_count));
    assert!(generated_token_counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(generated_token_counts.last().is_some_and(|last_count| *last_count <= generation.generated_token_count));
    assert_eq!(last_reported_text, Some(&generation.text));
}

#[wasm_bindgen_test]
async fn cancel_stops_generation_in_progress() {
    let model: Model = load_test_model().await;
    let maximum_token_count: usize = 200;
    let mut generated_token_count_at_last_report: usize = 0;

    let generation_outcome: Result<Generation, GenerationError> = model
        .generate(&fixtures::create_long_story_request(maximum_token_count), |progress: GenerationProgress| {
            if let GenerationProgress::Generating { generated_token_count, .. } = progress {
                generated_token_count_at_last_report = generated_token_count;
                model.cancel();
            }
        })
        .await;

    assert!(matches!(generation_outcome, Err(GenerationError::Cancelled)));
    assert!(generated_token_count_at_last_report < maximum_token_count);
}

#[wasm_bindgen_test]
async fn cancel_stops_generations_queued_before_it() {
    let model: Model = load_test_model().await;
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
        future::join(running_generation, queued_generation).await;

    assert!(matches!(queued_generation_outcome, Err(GenerationError::Cancelled)));
}

#[wasm_bindgen_test]
async fn dropping_generate_future_stops_its_generation() {
    let model: Model = load_test_model().await;
    let request: GenerationRequest = fixtures::create_long_story_request(200);

    let complete_generation: Generation = generate_ignoring_progress(&model, &request).await;
    drop_generate_future_once_generating(&model, &request).await;
    let next_request_started_at_milliseconds: f64 = js_sys::Date::now();
    generate_ignoring_progress(&model, &fixtures::create_greeting_request(1)).await;
    let next_request_milliseconds: f64 = js_sys::Date::now() - next_request_started_at_milliseconds;

    assert!(next_request_milliseconds < complete_generation.generating_duration.as_secs_f64() * 1000.0 / 2.0);
}

#[wasm_bindgen_test]
async fn generate_after_cancel_completes() {
    let model: Model = load_test_model().await;

    model.cancel();
    let generation_outcome: Result<Generation, GenerationError> = model.generate(&fixtures::create_greeting_request(8), |_: GenerationProgress| {}).await;

    assert!(generation_outcome.is_ok());
}
