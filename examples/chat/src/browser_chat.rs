use std::rc::Rc;

use cross_llama::{Generation, GenerationError, LoadProgress, Model, ModelSource};
use leptos::prelude::*;

use crate::chat::ReplyProgress;
use crate::chat_page::{ModelState, ProgressMeasure, ReplyState, StatusProgress};
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
                browser_model.set_value(Some(Rc::new(model)));
                model_state.set(ModelState::Ready);
            }
            Err(error) => model_state.set(ModelState::Failed(error.to_string())),
        }
    });

    browser_model
}

pub fn start_reply(prompt: String, browser_model: BrowserModel, reply_text: RwSignal<String>, reply_state: RwSignal<ReplyState>) {
    let model: Option<Rc<Model>> = browser_model.get_value();

    let Some(model) = model
    else {
        return;
    };

    let initial_progress: ReplyProgress = ReplyProgress::ReadingPrompt {
        read_token_count: 0,
        prompt_token_count: 0,
    };
    reply_text.set(String::new());
    reply_state.set(create_replying_state(&initial_progress));

    leptos::task::spawn_local(async move {
        let reply_outcome: Result<Generation, GenerationError> = chat::reply(&model, &prompt, |progress: ReplyProgress| {
            if let ReplyProgress::Generating { new_text, .. } = &progress {
                reply_text.update(|text: &mut String| text.push_str(new_text));
            }

            reply_state.set(create_replying_state(&progress));
        })
        .await;

        let finished_state: ReplyState = match reply_outcome {
            Ok(generation) => ReplyState::Finished {
                speed_description: chat::describe_generation_speed(&generation),
            },
            Err(error) => ReplyState::Failed(chat::describe_reply_failure(&error)),
        };

        reply_state.set(finished_state);
    });
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
