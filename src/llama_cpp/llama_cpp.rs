use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SendError, Sender};
use std::thread::JoinHandle;

use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;

use crate::llama_cpp::worker;
use crate::llama_cpp::worker::{
    GenerationEvent,
    GenerationEventSender,
    GenerationJob,
    LoadEvent,
    LoadEventSender,
    WorkerCommand,
    WorkerEvent,
    WorkerStart,
};
use crate::{Error, Generation, GenerationError, GenerationProgress, GenerationRequest, LoadOptions, LoadProgress};

const WORKER_THREAD_NAME: &str = "cross-llama";

#[derive(Clone, Debug, PartialEq)]
pub struct ModelSource {
    pub path: PathBuf,
}

/// A loaded model and the thread that runs it, which holds the model's cache between requests.
pub struct Model {
    command_sender: Sender<WorkerCommand>,
    cancellation_epoch: Arc<AtomicU64>,
    worker_thread: Option<JoinHandle<()>>,
}

impl Model {
    pub async fn load(source: ModelSource, load_options: &LoadOptions, on_progress: impl FnMut(LoadProgress)) -> Result<Model, Error> {
        let (command_sender, command_receiver): (Sender<WorkerCommand>, Receiver<WorkerCommand>) = std::sync::mpsc::channel();
        let (load_event_sender, load_event_receiver): (LoadEventSender, UnboundedReceiver<LoadEvent>) = futures::channel::mpsc::unbounded();
        let cancellation_epoch: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

        let worker_start: WorkerStart = WorkerStart {
            model_path: source.path,
            load_options: load_options.clone(),
            cancellation_epoch: Arc::clone(&cancellation_epoch),
            command_receiver,
            load_event_sender,
        };
        let worker_thread: JoinHandle<()> = std::thread::Builder::new()
            .name(WORKER_THREAD_NAME.to_string())
            .spawn(move || worker::run_worker(worker_start))?;

        // On failure, or when this future is dropped, the thread stops by itself.
        let load_outcome: Option<Result<(), Error>> = receive_until_finished(load_event_receiver, on_progress).await;

        match load_outcome {
            Some(Ok(())) => Ok(Model {
                command_sender,
                cancellation_epoch,
                worker_thread: Some(worker_thread),
            }),
            Some(Err(load_error)) => Err(load_error),
            None => Err(Error::new("the model's thread stopped while loading")),
        }
    }

    /// Waits for any generation already running on this model to finish first.
    pub async fn generate(&self, request: &GenerationRequest, on_progress: impl FnMut(GenerationProgress)) -> Result<Generation, GenerationError> {
        let (event_sender, event_receiver): (GenerationEventSender, UnboundedReceiver<GenerationEvent>) = futures::channel::mpsc::unbounded();
        let generation_job: GenerationJob = GenerationJob {
            request: request.clone(),
            submitted_cancellation_epoch: self.cancellation_epoch.load(Ordering::SeqCst),
            event_sender,
        };
        let submitted: Result<(), SendError<WorkerCommand>> = self.command_sender.send(WorkerCommand::Generate(generation_job));

        if submitted.is_err() {
            return Err(GenerationError::Failed(Error::new("the model's thread has stopped")));
        }

        let generation_outcome: Option<Result<Generation, GenerationError>> = receive_until_finished(event_receiver, on_progress).await;
        generation_outcome.unwrap_or_else(|| Err(GenerationError::Failed(Error::new("the model's thread stopped during the generation"))))
    }

    /// Stops the generation in progress, and any waiting for it, with `GenerationError::Cancelled`.
    pub fn cancel(&self) {
        self.cancellation_epoch.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.cancel();
        // Fails only when the thread has already stopped.
        let _ = self.command_sender.send(WorkerCommand::Shutdown);
        let worker_thread: Option<JoinHandle<()>> = self.worker_thread.take();

        if let Some(worker_thread) = worker_thread {
            // A panic on the thread has nowhere to propagate to from a destructor.
            let _ = worker_thread.join();
        }
    }
}

async fn receive_until_finished<Progress, Outcome>(
    mut event_receiver: UnboundedReceiver<WorkerEvent<Progress, Outcome>>,
    mut on_progress: impl FnMut(Progress),
) -> Option<Outcome> {
    loop {
        let event: Option<WorkerEvent<Progress, Outcome>> = event_receiver.next().await;

        match event {
            Some(WorkerEvent::Progress(progress)) => on_progress(progress),
            Some(WorkerEvent::Finished(outcome)) => return Some(outcome),
            None => return None,
        }
    }
}
