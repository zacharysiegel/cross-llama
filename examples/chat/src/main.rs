#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let outcome: Result<(), cross_llama_chat::AppError> = cross_llama_chat::site_server::serve().await;

    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            // The message alone, without the backtrace that `Display` adds.
            eprintln!("{}", error.message);
            std::process::ExitCode::FAILURE
        }
    }
}
