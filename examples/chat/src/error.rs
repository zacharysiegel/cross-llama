use cross_llama::GenerationError;

minimer::define_app_error!(pub AppError);
minimer::impl_from_error!(AppError, cross_llama::Error);
minimer::impl_from_error!(AppError, GenerationError);
