use std::rc::Rc;

use cross_llama::{Generation, GenerationError, LoadProgress, Model, ModelSource, ThinkingKind};
use leptos::prelude::*;

use crate::chat::ReplyProgress;
use crate::chat_page::{ModelState, ProgressMeasure, ReplySignals, ReplyState, StatusProgress};
use crate::{chat, model_file, site_paths};

/// `None` until the model has loaded.
pub type BrowserModel = StoredValue<Option<Rc<Model>>, LocalStorage>;

pub fn start_loading_model(model_state: RwSignal<ModelState>) -> BrowserModel {
    let browser_model: BrowserModel = StoredValue::new_local(None);
    let source: ModelSource = ModelSource {
        model_url: format!("{}{}", site_paths::MODELS_URL_PREFIX, model_file::MODEL_FILE_NAME),
        wllama_directory_url: site_paths::WLLAMA_URL_PREFIX.to_string(),
    };

    leptos::task::spawn_local(async move {
        let load_outcome: Result<Model, cross_llama::Error> = chat::load_model(source, |progress: LoadProgress| {
            model_state.set(ModelState::Loading(create_load_status(&progress)));
        })
        .await;

        match load_outcome {
            Ok(model) => {
                let offers_thinking: bool = chat::offers_thinking(&model);

                browser_model.set_value(Some(Rc::new(model)));
                model_state.set(ModelState::Ready { offers_thinking });
            }
            Err(error) => model_state.set(ModelState::Failed(chat::describe_error(&error))),
        }
    });

    browser_model
}

/// Thinks first when `thinking_chosen` and the model offers thinking.
pub fn start_reply(prompt: String, thinking_chosen: bool, browser_model: BrowserModel, model_state: ModelState, reply_signals: ReplySignals) {
    let model: Option<Rc<Model>> = browser_model.get_value();

    let Some(model) = model
    else {
        return;
    };

    let thinking: ThinkingKind = choose_thinking(thinking_chosen, &model_state);
    let initial_progress: ReplyProgress = ReplyProgress::ReadingPrompt {
        read_token_count: 0,
        prompt_token_count: 0,
    };
    reply_signals.reasoning_text.set(String::new());
    reply_signals.reply_text.set(String::new());
    reply_signals.reply_state.set(create_replying_state(&initial_progress));

    leptos::task::spawn_local(async move {
        let reply_outcome: Result<Generation, GenerationError> = chat::reply(&model, &prompt, thinking, |progress: ReplyProgress| {
            show_reply_progress(&progress, reply_signals);
        })
        .await;

        let finished_state: ReplyState = match reply_outcome {
            Ok(generation) => ReplyState::Finished {
                speed_description: chat::describe_generation_speed(&generation),
            },
            Err(error) => ReplyState::Failed(chat::describe_reply_failure(&error)),
        };

        reply_signals.reply_state.set(finished_state);
    });
}

fn choose_thinking(thinking_chosen: bool, model_state: &ModelState) -> ThinkingKind {
    if thinking_chosen && model_state.offers_thinking() {
        return ThinkingKind::Enabled;
    }

    ThinkingKind::Disabled
}

fn show_reply_progress(progress: &ReplyProgress, reply_signals: ReplySignals) {
    match progress {
        ReplyProgress::ReadingPrompt { .. } => {}
        ReplyProgress::Reasoning { new_text, .. } => reply_signals.reasoning_text.update(|text: &mut String| text.push_str(new_text)),
        ReplyProgress::Generating { new_text, .. } => reply_signals.reply_text.update(|text: &mut String| text.push_str(new_text)),
    }

    reply_signals.reply_state.set(create_replying_state(progress));
}

fn create_load_status(progress: &LoadProgress) -> StatusProgress {
    StatusProgress {
        description: chat::describe_load_progress(progress),
        measure: get_load_progress_measure(progress),
    }
}

/// Indeterminate while loading, since wllama reports no progress then.
fn get_load_progress_measure(progress: &LoadProgress) -> ProgressMeasure {
    match progress {
        LoadProgress::Downloading {
            downloaded_byte_count,
            total_byte_count: Some(total_byte_count),
        } => ProgressMeasure::Determinate {
            completed: *downloaded_byte_count as f64,
            total: *total_byte_count as f64,
        },
        LoadProgress::Downloading { total_byte_count: None, .. } | LoadProgress::Loading { .. } => ProgressMeasure::Indeterminate,
    }
}

fn create_replying_state(progress: &ReplyProgress) -> ReplyState {
    ReplyState::Replying {
        description: chat::describe_reply_progress(progress),
    }
}
