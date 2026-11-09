# cross-llama

Runs a GGUF language model from Rust behind one async API: through llama.cpp on targets other than `wasm32`, and through wllama (llama.cpp compiled to WebAssembly) in the browser. Both format prompts with the model file's own Jinja chat template, and return a reasoning model's reasoning apart from its answer.

[examples/chat/](./examples/chat/) is a chat demo with a web client and a terminal client for macOS and Linux.

## Example

```rust
use cross_llama::{ChatMessage, Generation, GenerationRequest, LoadOptions, Model, ModelSource, Role, SlidingWindowCache, ThinkingKind};

let load_options: LoadOptions = LoadOptions {
    context_token_count: 4096,
    sliding_window_cache: SlidingWindowCache::Full,
};
let model: Model = Model::load(ModelSource { path: "model.gguf".into() }, &load_options, |_progress| {}).await?;

let request: GenerationRequest = GenerationRequest {
    messages: vec![ChatMessage { role: Role::User, text: "Is \"Eu sou cansado\" correct?".to_string() }],
    json_schema: Some(serde_json::json!({
        "type": "object",
        "properties": { "correct": { "type": "boolean" } },
        "required": ["correct"],
    })),
    temperature: 0.0,
    seed: 0,
    maximum_token_count: 64,
    thinking: ThinkingKind::Disabled,
    thinking_budget_token_count: None,
    thinking_effort: None,
};
let generation: Generation = model.generate(&request, |_progress| {}).await?;
println!("{}", generation.text);
```

In the browser, `ModelSource` holds two URLs instead of a path:

```rust
let source: ModelSource = ModelSource {
    model_url: "/models/model.gguf".to_string(),
    wllama_directory_url: "/wllama/".to_string(),
};
```

In the browser only:

- `download_model(&source, on_progress)` puts the model in wllama's cache without loading it.
- `is_model_downloaded(&source)` checks that cache without a network request.
- `probe_browser_capability()` reports WebGPU, cross-origin isolation, the wllama build the browser gets, storage and device memory.

`Model::cancel` stops the generation in progress, which then returns `GenerationError::Cancelled`. The crate does not depend on an async runtime; any executor can drive the futures.

## Thinking

- `thinking: ThinkingKind::Enabled` generates the reasoning first, unconstrained, then the answer, under the JSON schema when one is set; `generation.reasoning` holds the reasoning, `generation.text` the answer.
- `thinking_budget_token_count` caps the reasoning's tokens; `maximum_token_count` caps reasoning and answer together.
- `thinking_effort` sets the template's `reasoning_effort`.
- `model.thinking_capability()` tells which of these the loaded model supports; a request using one it does not fails before generating.

## Serving the browser backend's files

- wllama's files: `./scripts/fetch-wllama.sh <directory>` writes them; serve that directory at `wllama_directory_url`.
- The model file, at `model_url`, with a `Content-Length` header; without one wllama cannot check its cached copy and downloads the model again on every load.
- Every page that loads a model needs the cross-origin isolation headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, which wllama's multithreaded build needs.

## Setup

```sh
./scripts/setup.sh
```

The script checks for the programs the build needs, installs the Rust toolchain, fetches wllama's browser files and the integration tests' model into `build/`, and installs the browser tests' wasm-bindgen test runner and ChromeDriver into `build/tools/`.

## Tests

```sh
cargo test
./scripts/test-browser.sh
./scripts/test-browser.sh --safari
```

`cargo test` runs the llama.cpp backend's tests in `tests/llama_cpp.rs`, which load the model that `scripts/setup.sh` fetches into `build/models/`. `scripts/test-browser.sh` runs the browser backend's tests in `tests/wllama.rs` and `src/wllama/` against the same model, in headless Chrome or, with `--safari`, in Safari; Safari needs its WebDriver enabled once with `safaridriver --enable`.
