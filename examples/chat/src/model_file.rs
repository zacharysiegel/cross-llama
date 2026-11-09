pub const MODEL_FILE_NAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

// reads the local filesystem, which a browser page cannot
#[cfg(not(target_arch = "wasm32"))]
pub use filesystem::*;

// reads the local filesystem, which a browser page cannot
#[cfg(not(target_arch = "wasm32"))]
mod filesystem {
    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    use crate::AppError;

    const GGUF_MAGIC: [u8; 4] = *b"GGUF";

    pub fn get_models_directory() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
    }

    pub fn get_default_model_path() -> PathBuf {
        get_models_directory().join(super::MODEL_FILE_NAME)
    }

    /// Rejects a file other than a GGUF model, such as the Git LFS pointer a clone without git-lfs leaves in its place.
    pub fn check_model_file(path: &Path) -> Result<(), AppError> {
        let opened: Result<File, std::io::Error> = File::open(path);

        let mut file: File = match opened {
            Ok(file) => file,
            Err(error) => return Err(AppError::from_error(&format!("cannot open the model at {}", path.display()), Box::new(error))),
        };

        let mut magic: [u8; 4] = [0; 4];
        file.read_exact(&mut magic)?;

        if magic != GGUF_MAGIC {
            return Err(AppError::new(&format!(
                "{} is not a GGUF file; if it is a Git LFS pointer, run `git lfs pull`",
                path.display(),
            )));
        }

        Ok(())
    }
}
