# cross-llama

Runs a GGUF language model from Rust behind one async API, through llama.cpp on targets other than `wasm32`. The design, including the planned browser backend, is in [docs/design.md](./docs/design.md).

## Example

```rust
use cross_llama::{ChatMessage, Generation, GenerationRequest, LoadOptions, Model, ModelSource, Role, SlidingWindowCache};

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
};
let generation: Generation = model.generate(&request, |_progress| {}).await?;
println!("{}", generation.text);
```

`Model::cancel` stops the generation in progress, which then returns `GenerationError::Cancelled`. The crate does not depend on an async runtime; any executor can drive the futures.

## Setup

```sh
./scripts/setup.sh
```

The script checks for the programs the build needs, installs the Rust toolchain, and fetches wllama's browser files and the integration tests' model into `build/`.

## Tests

```sh
cargo test
```

The integration tests in `tests/` load the model that `scripts/setup.sh` fetches into `build/models/`.
