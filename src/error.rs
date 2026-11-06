use std::fmt::{Display, Formatter};

// The static variant is `Send`, so errors can cross from the thread that runs the model.
minimer::define_app_error_static!(pub Error);

#[derive(Debug)]
pub enum GenerationError {
    Cancelled,
    Failed(Error),
}

impl Display for GenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationError::Cancelled => write!(formatter, "the generation was cancelled"),
            GenerationError::Failed(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for GenerationError {}

impl From<Error> for GenerationError {
    fn from(error: Error) -> Self {
        GenerationError::Failed(error)
    }
}
