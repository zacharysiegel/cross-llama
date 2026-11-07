use std::path::PathBuf;

// `scripts/setup.sh` fetches it.
const TEST_MODEL_RELATIVE_PATH: &str = "build/models/gemma-3-270m-it-Q8_0.gguf";

pub fn get_test_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_MODEL_RELATIVE_PATH)
}
