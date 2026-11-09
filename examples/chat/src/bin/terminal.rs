fn main() -> std::process::ExitCode {
    let outcome: Result<(), cross_llama_chat::AppError> = futures::executor::block_on(terminal::run());

    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

mod terminal {
    use std::cell::Cell;
    use std::io::{BufRead, IsTerminal, Write};
    use std::path::PathBuf;

    use cross_llama::{Generation, GenerationError, LoadProgress, Model, ModelSource};
    use cross_llama_chat::chat::ReplyProgress;
    use cross_llama_chat::{AppError, chat, model_file};

    const CLEAR_LINE: &str = "\r\x1b[2K";

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
        let model_path: PathBuf = read_model_path();
        model_file::check_model_file(&model_path)?;

        let status_line: StatusLine = StatusLine::for_standard_error();
        let model: Model = chat::load_model(ModelSource { path: model_path }, |progress: LoadProgress| {
            status_line.show(&chat::describe_load_progress(&progress));
        })
        .await?;
        status_line.clear();

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

            let reply_outcome: Result<(), GenerationError> = reply_to_prompt(&model, &prompt, &status_line).await;

            if let Err(error) = reply_outcome {
                status_line.clear();
                eprintln!("The reply failed: {}", chat::describe_reply_failure(&error));
            }
        }

        model.unload().await?;

        Ok(())
    }

    fn read_model_path() -> PathBuf {
        let model_path_argument: Option<std::ffi::OsString> = std::env::args_os().nth(1);

        model_path_argument.map(PathBuf::from).unwrap_or_else(model_file::get_default_model_path)
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

    async fn reply_to_prompt(model: &Model, prompt: &str, status_line: &StatusLine) -> Result<(), GenerationError> {
        let generation: Generation = chat::reply(model, prompt, |progress: ReplyProgress| match &progress {
            ReplyProgress::ReadingPrompt { .. } | ReplyProgress::Reasoning { .. } => status_line.show(&chat::describe_reply_progress(&progress)),
            ReplyProgress::Generating { new_text, .. } => {
                status_line.clear();
                print_and_flush(new_text);
            }
        })
        .await?;

        status_line.clear();
        println!();
        eprintln!("[{}]", chat::describe_generation_speed(&generation));

        Ok(())
    }

    fn print_and_flush(text: &str) {
        print!("{text}");
        // A failed flush leaves the text for the next one.
        let _ = std::io::stdout().flush();
    }
}
