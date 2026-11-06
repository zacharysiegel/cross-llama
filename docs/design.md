# cross-llama design

Runs a GGUF language model from Rust with one API on every target: through llama.cpp on targets other than `wasm32`, and through wllama (llama.cpp compiled to WebAssembly) in the browser. Its first consumer is the language app's grader, which sends a short conversation and a JSON schema and expects a schema-valid reply in a few seconds.

## Public API

```rust
let model: Model = Model::load(source, &load_options, |progress: LoadProgress| { ... }).await?;
let generation: Generation = model.generate(&request).await?;
model.cancel();
```

- `ModelSource`: a file path on targets other than `wasm32`; a URL in the browser.
- `LoadOptions`:
  - context length in tokens
  - whether to keep the whole sliding-window cache, which lets a shared prompt prefix be reused across requests (llama.cpp's `swa_full`)
- `LoadProgress`: bytes downloaded and total, then a loading stage.
- `GenerationRequest`:
  - messages, each a role and text; the model's own chat template formats them
  - an optional JSON schema the reply must match
  - temperature and seed
  - maximum tokens
- `Generation`:
  - the reply text
  - prompt tokens read, tokens reused from the cache, tokens generated
  - time spent reading the prompt and generating
- `cancel` stops the generation in progress, which then returns a cancellation error.
- One generation runs at a time per `Model`; a second call waits for the first.
- Errors use `minimer`.

Tokens, batches, samplers and the cache stay inside the crate. A caller that later needs one of them gets it added deliberately.

## Backends

Each backend is a module selected by `#[cfg]`, re-exported as the same public types.

### `llama_cpp`, for targets other than `wasm32`

- Wraps `llama-cpp-2` (and `llama-cpp-sys-2`, whose build script compiles llama.cpp with CMake and a C++ compiler).
- Features:
  - `llguidance` for schema-constrained output; the schema is compiled once per distinct schema and reused, because compiling took about 0.35 s.
  - `metal` on macOS and iOS; other GPU backends (CUDA, Vulkan) are opt-in crate features passed through.
- Generation runs on a blocking thread behind the `async` API, holding the context between requests.
- Prefix reuse: the backend remembers the tokens in the cache, keeps the longest common prefix with the new prompt, clears the cache from the first differing token (`clear_kv_cache_seq`) and decodes only the rest.
- Measured on an M4 Max with gemma3 12B Q4_K_M: 0.73 s median per graded answer with prefix reuse, 2.6 s without.

### `wllama`, for `wasm32` in the browser

- Calls wllama's JavaScript API through `wasm-bindgen` imports (`#[wasm_bindgen] extern "C"` declarations) of a small JavaScript module shipped in this crate.
- Loading: `loadModelFromUrl`, which caches the model in the browser's storage, with progress; `n_gpu_layers` set to keep every layer on the GPU through WebGPU; `swa_full` from `LoadOptions`.
- Generation: `createChatCompletion` with `response_format` set to the JSON schema; wllama reuses the cached prompt prefix itself.
- Builds:
  - wllama uses its default build when the browser has WebAssembly 64-bit memory and JSPI, and switches to its compatibility build by itself otherwise.
  - The app serves both builds' files itself; the crate tells wllama where the compatibility build is (`setCompat`) instead of letting it fetch from a CDN.
  - `scripts/check-browser-features.sh` reports which browsers' current releases get which build.
- Measured on an M4 Max with gemma3 12B Q4_K_M:
  - Chrome, default build: 124 tokens/s reading the prompt, 27 generating; p95 3.0 s per graded answer with prefix reuse.
  - Safari, compatibility build: 26 tokens/s reading, 16 generating.
- The 7.3 GB model loads into GPU memory, not WebAssembly memory, so the compatibility build's 4 GB WebAssembly limit does not stop it.

## Assets the app serves in the browser

- wllama's JavaScript and WebAssembly files, default and compatibility builds, at paths the app passes in `LoadOptions`.
  - `scripts/fetch-wllama.sh <directory>` writes them, from the pinned npm release, after checking each package against its published integrity hash; the app runs it as part of its build and serves the directory.
  - They are not committed to this repository, and the crate's build script does not download them: a build script writes only to Cargo's output directory, which the app does not serve, and downloading there would break offline builds.
- The model file, at the URL in `ModelSource`.
- Cross-origin isolation headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`), which wllama's multithreaded build needs.

## Testing

- Shared request and response types: host unit tests.
- `llama_cpp`: integration tests that load a small GGUF (gemma-3-270m), generate with and without a schema, and check prefix reuse and cancellation.
- `wllama`: `wasm-bindgen-test` in a headless Chrome, with the same tests against the same small model, since this backend's code only runs in a browser.

## Out of scope for the first version

- Downloading or managing model files on targets other than `wasm32`; the app supplies the path.
- Streaming tokens as they are generated.
- Images, audio, embeddings and tool calls.

## Platforms

- The browser backend serves the language app's grading, which runs only on the web.
- The llama.cpp backend covers desktop operating systems and the crate's own fast tests. llama.cpp also builds for iOS and Android, but no phone-sized model passed the grading bar, so neither is a target for now.
