use leptos::ev::SubmitEvent;
use leptos::hydration::{AutoReload, HydrationScripts};
use leptos::prelude::*;

// runs the model in the browser, which the server build does not
#[cfg(feature = "hydrate")]
use crate::browser_chat;

// The client's first load report replaces this; the server renders it.
const INITIAL_MODEL_DESCRIPTION: &str = "Loading the model";

#[derive(Clone, Debug, PartialEq)]
pub enum ProgressMeasure {
    Indeterminate,
    Determinate { completed: f64, total: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct StatusProgress {
    pub description: String,
    pub measure: ProgressMeasure,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelState {
    Loading(StatusProgress),
    Ready { offers_thinking: bool },
    Failed(String),
}

impl ModelState {
    fn is_ready(&self) -> bool {
        matches!(self, ModelState::Ready { .. })
    }

    pub fn offers_thinking(&self) -> bool {
        matches!(self, ModelState::Ready { offers_thinking: true })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReplyState {
    Idle,
    Replying { description: String },
    Finished { speed_description: String },
    Failed(String),
}

impl ReplyState {
    fn is_replying(&self) -> bool {
        matches!(self, ReplyState::Replying { .. })
    }
}

/// What a reply updates as it streams.
#[derive(Clone, Copy, Debug)]
pub struct ReplySignals {
    pub reasoning_text: RwSignal<String>,
    pub reply_text: RwSignal<String>,
    pub reply_state: RwSignal<ReplyState>,
}

pub fn shell(options: LeptosOptions) -> impl IntoView {
    let stylesheet_url: String = format!("/{}/{}.css", options.site_pkg_dir, options.output_name);

    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>"cross-llama chat"</title>
                <link rel="icon" href="data:," />
                <link rel="stylesheet" href=stylesheet_url />
                <AutoReload options=options.clone() />
                <HydrationScripts options />
            </head>
            <body>
                <ChatPage />
            </body>
        </html>
    }
}

#[component]
pub fn ChatPage() -> impl IntoView {
    let model_state: RwSignal<ModelState> = RwSignal::new(ModelState::Loading(StatusProgress {
        description: INITIAL_MODEL_DESCRIPTION.to_string(),
        measure: ProgressMeasure::Indeterminate,
    }));
    let prompt_text: RwSignal<String> = RwSignal::new(String::new());
    let thinking_chosen: RwSignal<bool> = RwSignal::new(false);
    let asked_prompt: RwSignal<String> = RwSignal::new(String::new());
    let reply_signals: ReplySignals = ReplySignals {
        reasoning_text: RwSignal::new(String::new()),
        reply_text: RwSignal::new(String::new()),
        reply_state: RwSignal::new(ReplyState::Idle),
    };
    let reply_state: RwSignal<ReplyState> = reply_signals.reply_state;

    // the server renders the page without a model
    #[cfg(feature = "hydrate")]
    let browser_model: browser_chat::BrowserModel = browser_chat::start_loading_model(model_state);

    let can_send = move || model_state.get().is_ready() && !reply_state.get().is_replying();
    let on_submit = move |event: SubmitEvent| {
        event.prevent_default();

        let prompt: String = prompt_text.get_untracked();

        if prompt.trim().is_empty() || !can_send() {
            return;
        }

        prompt_text.set(String::new());
        asked_prompt.set(prompt.clone());

        // the server never receives the form's events
        #[cfg(feature = "hydrate")]
        browser_chat::start_reply(prompt, thinking_chosen.get_untracked(), browser_model, model_state.get_untracked(), reply_signals);
    };

    view! {
        <main>
            <h1>"cross-llama chat"</h1>
            <p class="explanation">"A language model running in this page through cross-llama's wllama backend. Each prompt is answered on its own, without the earlier ones."</p>
            <ModelStatus model_state=model_state />
            <form on:submit=on_submit>
                <input type="text" placeholder="Type a prompt" bind:value=prompt_text disabled=move || !model_state.get().is_ready() />
                <button type="submit" disabled=move || !can_send()>
                    {move || if reply_state.get().is_replying() { "Replying…" } else { "Send" }}
                </button>
            </form>
            <Show when=move || model_state.get().offers_thinking()>
                <label class="thinking">
                    <input type="checkbox" bind:checked=thinking_chosen />
                    "Think before answering"
                </label>
            </Show>
            <Show when=move || !asked_prompt.get().is_empty()>
                <section class="exchange">
                    <p class="prompt">{move || asked_prompt.get()}</p>
                    <Show when=move || !reply_signals.reasoning_text.with(|text: &String| text.trim().is_empty())>
                        <details class="reasoning">
                            <summary>"Reasoning"</summary>
                            // The model can surround its reasoning with whitespace.
                            <p>{move || reply_signals.reasoning_text.with(|text: &String| text.trim().to_string())}</p>
                        </details>
                    </Show>
                    <p class="reply">{move || reply_signals.reply_text.get()}</p>
                    <ReplyStatus reply_state=reply_state />
                </section>
            </Show>
        </main>
    }
}

#[component]
fn ModelStatus(model_state: RwSignal<ModelState>) -> impl IntoView {
    move || match model_state.get() {
        ModelState::Loading(status_progress) => view! {
            <div class="status">
                <p>{status_progress.description}</p>
                {progress_bar(&status_progress.measure)}
            </div>
        }
        .into_any(),
        ModelState::Ready { .. } => view! { <p class="status">"The model is loaded."</p> }.into_any(),
        ModelState::Failed(message) => view! { <p class="status error">{format!("The model could not be loaded: {message}")}</p> }.into_any(),
    }
}

#[component]
fn ReplyStatus(reply_state: RwSignal<ReplyState>) -> impl IntoView {
    move || match reply_state.get() {
        ReplyState::Idle => view! { <p class="status"></p> }.into_any(),
        ReplyState::Replying { description } => view! {
            <div class="status">
                <p>{description}</p>
                <progress></progress>
            </div>
        }
        .into_any(),
        ReplyState::Finished { speed_description } => view! { <p class="status">{speed_description}</p> }.into_any(),
        ReplyState::Failed(message) => view! { <p class="status error">{format!("The reply failed: {message}")}</p> }.into_any(),
    }
}

fn progress_bar(measure: &ProgressMeasure) -> AnyView {
    match measure {
        ProgressMeasure::Indeterminate => view! { <progress></progress> }.into_any(),
        ProgressMeasure::Determinate { completed, total } => view! { <progress max=*total value=*completed></progress> }.into_any(),
    }
}
