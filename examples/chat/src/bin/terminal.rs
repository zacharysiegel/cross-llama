fn main() -> std::process::ExitCode {
    let outcome: Result<(), cross_llama_chat::AppError> = futures::executor::block_on(terminal::run());

    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            // The message alone, without the backtrace that `Display` adds.
            eprintln!("{}", error.message);
            std::process::ExitCode::FAILURE
        }
    }
}

mod terminal {
    use std::cell::Cell;
    use std::ffi::OsString;
    use std::io::{BufRead, IsTerminal, Write};
    use std::path::PathBuf;

    use cross_llama::{Generation, GenerationError, LoadProgress, Model, ModelSource, ThinkingKind};
    use cross_llama_chat::chat::ReplyProgress;
    use cross_llama_chat::{AppError, chat, model_file};

    const CLEAR_LINE: &str = "\r\x1b[2K";
    const THINKING_FLAG: &str = "--thinking";
    const REASONING_LABEL: &str = "[Reasoning] ";

    /// `[--thinking] [<model path>]`
    struct TerminalArguments {
        model_path: PathBuf,
        thinking: ThinkingKind,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum StreamedPartKind {
        Nothing,
        Reasoning,
        Answer,
    }

    /// A line on standard error that shows progress while standard error is a terminal.
    struct StatusLine {
        enabled: bool,
        shown: Cell<bool>,
    }

    impl StatusLine {
        fn for_standard_error() -> StatusLine {
            StatusLine {
                enabled: std::io::stderr().is_terminal(),
                shown: Cell::new(false),
            }
        }

        fn show(&self, text: &str) {
            if !self.enabled {
                return;
            }

            eprint!("{CLEAR_LINE}{text}");
            self.shown.set(true);
        }

        fn clear(&self) {
            if !self.shown.get() {
                return;
            }

            eprint!("{CLEAR_LINE}");
            self.shown.set(false);
        }
    }

    pub async fn run() -> Result<(), AppError> {
        let terminal_arguments: TerminalArguments = read_arguments();
        model_file::check_model_file(&terminal_arguments.model_path)?;

        let status_line: StatusLine = StatusLine::for_standard_error();
        let model_source: ModelSource = ModelSource {
            path: terminal_arguments.model_path,
        };
        let model: Model = chat::load_model(model_source, |progress: LoadProgress| {
            status_line.show(&chat::describe_load_progress(&progress));
        })
        .await
        .map_err(create_model_failure)?;
        status_line.clear();

        if terminal_arguments.thinking == ThinkingKind::Enabled && !chat::offers_thinking(&model) {
            return Err(AppError::new(&format!(
                "{THINKING_FLAG} needs a model whose reasoning cross-llama can separate from its answer, which this model is not",
            )));
        }

        let prompts_from_terminal: bool = std::io::stdin().is_terminal();

        if prompts_from_terminal {
            eprintln!("Type a prompt and press Enter; Ctrl-D quits. Each prompt is answered on its own, without the earlier ones.");
        }

        loop {
            if prompts_from_terminal {
                eprint!("> ");
            }

            let prompt: Option<String> = read_prompt()?;

            let Some(prompt) = prompt
            else {
                break;
            };

            if prompt.trim().is_empty() {
                continue;
            }

            let reply_outcome: Result<(), GenerationError> = reply_to_prompt(&model, &prompt, terminal_arguments.thinking, &status_line).await;

            if let Err(error) = reply_outcome {
                status_line.clear();
                eprintln!("The reply failed: {}", chat::describe_reply_failure(&error));
            }
        }

        model.unload().await.map_err(create_model_failure)?;

        Ok(())
    }

    /// Carries the model error's message alone, without the backtrace its `Display` includes.
    fn create_model_failure(error: cross_llama::Error) -> AppError {
        AppError::new(&chat::describe_error(&error))
    }

    fn read_arguments() -> TerminalArguments {
        let mut model_path_argument: Option<OsString> = None;
        let mut thinking: ThinkingKind = ThinkingKind::Disabled;

        for argument in std::env::args_os().skip(1) {
            if argument == THINKING_FLAG {
                thinking = ThinkingKind::Enabled;
                continue;
            }

            model_path_argument = Some(argument);
        }

        TerminalArguments {
            model_path: model_path_argument.map(PathBuf::from).unwrap_or_else(model_file::get_default_model_path),
            thinking,
        }
    }

    /// Returns `None` at the end of the input.
    fn read_prompt() -> Result<Option<String>, AppError> {
        let mut line: String = String::new();
        let read_byte_count: usize = std::io::stdin().lock().read_line(&mut line)?;

        if read_byte_count == 0 {
            return Ok(None);
        }

        Ok(Some(line.trim_end_matches(['\r', '\n']).to_string()))
    }

    /// Streams the reasoning to standard error and the answer to standard output.
    async fn reply_to_prompt(model: &Model, prompt: &str, thinking: ThinkingKind, status_line: &StatusLine) -> Result<(), GenerationError> {
        let mut streamed_part: StreamedPartKind = StreamedPartKind::Nothing;

        let generation: Generation = chat::reply(model, prompt, thinking, |progress: ReplyProgress| match &progress {
            ReplyProgress::ReadingPrompt { .. } => status_line.show(&chat::describe_reply_progress(&progress)),
            ReplyProgress::Reasoning { new_text, .. } => {
                status_line.clear();
                streamed_part = stream_reasoning(new_text, streamed_part);
            }
            ReplyProgress::Generating { new_text, .. } => {
                status_line.clear();
                streamed_part = stream_answer(new_text, streamed_part);
            }
        })
        .await?;

        status_line.clear();
        println!();
        eprintln!("[{}]", chat::describe_generation_speed(&generation));

        Ok(())
    }

    fn stream_reasoning(new_text: &str, streamed_part: StreamedPartKind) -> StreamedPartKind {
        if streamed_part == StreamedPartKind::Reasoning {
            eprint!("{new_text}");
            return StreamedPartKind::Reasoning;
        }

        // The model can begin its reasoning with whitespace.
        let reasoning_start: &str = new_text.trim_start();

        if reasoning_start.is_empty() {
            return streamed_part;
        }

        eprint!("{REASONING_LABEL}{reasoning_start}");

        StreamedPartKind::Reasoning
    }

    fn stream_answer(new_text: &str, streamed_part: StreamedPartKind) -> StreamedPartKind {
        if streamed_part == StreamedPartKind::Reasoning {
            eprintln!("\n");
        }

        print_and_flush(new_text);

        StreamedPartKind::Answer
    }

    fn print_and_flush(text: &str) {
        print!("{text}");
        // A failed flush leaves the text for the next one.
        let _ = std::io::stdout().flush();
    }
}
