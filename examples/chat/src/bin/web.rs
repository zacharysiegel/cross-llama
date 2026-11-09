// The web client runs in a browser page; other targets run the terminal client instead.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("The web client runs in a browser; ./scripts/serve-web.sh builds and serves it");
}

// runs in a browser page, which other targets lack
#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(web::ChatPage);
}

// runs in a browser page, which other targets lack
#[cfg(target_arch = "wasm32")]
mod web {
    use std::rc::Rc;

    use cross_llama::{Generation, GenerationError, LoadProgress, Model, ModelSource};
    use leptos::ev::SubmitEvent;
    use leptos::prelude::*;

    use cross_llama_chat::chat;
    use cross_llama_chat::chat::ReplyProgress;

    const MODELS_DIRECTORY_URL: &str = "/models/";
    const WLLAMA_DIRECTORY_URL: &str = "/wllama/";

    #[derive(Clone, Debug, PartialEq)]
    enum ModelState {
        Loading(LoadProgress),
        Ready,
        Failed(String),
    }

    #[derive(Clone, Debug, PartialEq)]
    enum ReplyState {
        Idle,
        Replying(ReplyProgress),
        Finished { speed_description: String },
        Failed(String),
    }

    impl ReplyState {
        fn is_replying(&self) -> bool {
            matches!(self, ReplyState::Replying(_))
        }
    }

    #[component]
    pub fn ChatPage() -> impl IntoView {
        let model_state: RwSignal<ModelState> = RwSignal::new(ModelState::Loading(LoadProgress::Downloading {
            downloaded_byte_count: 0,
            total_byte_count: None,
        }));
        let loaded_model: StoredValue<Option<Rc<Model>>, LocalStorage> = StoredValue::new_local(None);
        let prompt_text: RwSignal<String> = RwSignal::new(String::new());
        let asked_prompt: RwSignal<String> = RwSignal::new(String::new());
        let reply_text: RwSignal<String> = RwSignal::new(String::new());
        let reply_state: RwSignal<ReplyState> = RwSignal::new(ReplyState::Idle);

        start_loading_model(model_state, loaded_model);

        let can_send = move || model_state.get() == ModelState::Ready && !reply_state.get().is_replying();
        let on_submit = move |event: SubmitEvent| {
            event.prevent_default();

            let prompt: String = prompt_text.get_untracked();

            if prompt.trim().is_empty() || !can_send() {
                return;
            }

            prompt_text.set(String::new());
            asked_prompt.set(prompt.clone());
            start_reply(prompt, loaded_model, reply_text, reply_state);
        };

        view! {
            <main>
                <h1>"cross-llama chat"</h1>
                <p class="explanation">"A language model running in this page through cross-llama's wllama backend. Each prompt is answered on its own, without the earlier ones."</p>
                <ModelStatus model_state=model_state />
                <form on:submit=on_submit>
                    <input type="text" placeholder="Type a prompt" bind:value=prompt_text disabled=move || model_state.get() != ModelState::Ready />
                    <button type="submit" disabled=move || !can_send()>
                        {move || if reply_state.get().is_replying() { "Replying…" } else { "Send" }}
                    </button>
                </form>
                <Show when=move || !asked_prompt.get().is_empty()>
                    <section class="exchange">
                        <p class="prompt">{move || asked_prompt.get()}</p>
                        <p class="reply">{move || reply_text.get()}</p>
                        <ReplyStatus reply_state=reply_state />
                    </section>
                </Show>
            </main>
        }
    }

    #[component]
    fn ModelStatus(model_state: RwSignal<ModelState>) -> impl IntoView {
        move || match model_state.get() {
            ModelState::Loading(progress) => view! {
                <div class="status">
                    <p>{chat::describe_load_progress(&progress)}</p>
                    {load_progress_bar(&progress)}
                </div>
            }
            .into_any(),
            ModelState::Ready => view! { <p class="status">"The model is loaded."</p> }.into_any(),
            ModelState::Failed(message) => view! { <p class="status error">{format!("The model could not be loaded: {message}")}</p> }.into_any(),
        }
    }

    #[component]
    fn ReplyStatus(reply_state: RwSignal<ReplyState>) -> impl IntoView {
        move || match reply_state.get() {
            ReplyState::Idle => view! { <p class="status"></p> }.into_any(),
            ReplyState::Replying(progress) => view! {
                <div class="status">
                    <p>{chat::describe_reply_progress(&progress)}</p>
                    <progress></progress>
                </div>
            }
            .into_any(),
            ReplyState::Finished { speed_description } => view! { <p class="status">{speed_description}</p> }.into_any(),
            ReplyState::Failed(message) => view! { <p class="status error">{format!("The reply failed: {message}")}</p> }.into_any(),
        }
    }

    /// Indeterminate while loading, since wllama reports no progress then.
    fn load_progress_bar(progress: &LoadProgress) -> AnyView {
        match progress {
            LoadProgress::Downloading {
                downloaded_byte_count,
                total_byte_count: Some(total_byte_count),
            } => view! { <progress max=*total_byte_count as f64 value=*downloaded_byte_count as f64></progress> }.into_any(),
            LoadProgress::Downloading { total_byte_count: None, .. } | LoadProgress::Loading { .. } => view! { <progress></progress> }.into_any(),
        }
    }

    fn start_loading_model(model_state: RwSignal<ModelState>, loaded_model: StoredValue<Option<Rc<Model>>, LocalStorage>) {
        let source: ModelSource = ModelSource {
            model_url: format!("{MODELS_DIRECTORY_URL}{}", chat::MODEL_FILE_NAME),
            wllama_directory_url: WLLAMA_DIRECTORY_URL.to_string(),
        };

        leptos::task::spawn_local(async move {
            let load_outcome: Result<Model, cross_llama::Error> =
                chat::load_model(source, |progress: LoadProgress| model_state.set(ModelState::Loading(progress))).await;

            match load_outcome {
                Ok(model) => {
                    loaded_model.set_value(Some(Rc::new(model)));
                    model_state.set(ModelState::Ready);
                }
                Err(error) => model_state.set(ModelState::Failed(error.to_string())),
            }
        });
    }

    fn start_reply(
        prompt: String,
        loaded_model: StoredValue<Option<Rc<Model>>, LocalStorage>,
        reply_text: RwSignal<String>,
        reply_state: RwSignal<ReplyState>,
    ) {
        let model: Option<Rc<Model>> = loaded_model.get_value();

        let Some(model) = model
        else {
            return;
        };

        reply_text.set(String::new());
        reply_state.set(ReplyState::Replying(ReplyProgress::ReadingPrompt {
            read_token_count: 0,
            prompt_token_count: 0,
        }));

        leptos::task::spawn_local(async move {
            let reply_outcome: Result<Generation, GenerationError> = chat::reply(&model, &prompt, |progress: ReplyProgress| {
                if let ReplyProgress::Generating { new_text, .. } = &progress {
                    reply_text.update(|text: &mut String| text.push_str(new_text));
                }

                reply_state.set(ReplyState::Replying(progress));
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
}
