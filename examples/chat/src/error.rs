minimer::define_app_error!(pub AppError);

// cross-llama runs the model, which the web server does not
#[cfg(any(feature = "hydrate", feature = "terminal"))]
mod model_errors {
    use cross_llama::GenerationError;

    use super::AppError;

    minimer::impl_from_error!(AppError, cross_llama::Error);
    minimer::impl_from_error!(AppError, GenerationError);
}
