# cross-llama design

Runs a GGUF language model from Rust with one API on every target: through llama.cpp on targets other than `wasm32`, and through wllama (llama.cpp compiled to WebAssembly) in the browser. Its first consumer is the language app's grader, which sends a short conversation and a JSON schema and expects a schema-valid reply in a few seconds.

## Public API

```rust
let model: Model = Model::load(source, &load_options, |progress: LoadProgress| { ... }).await?;
let generation: Generation = model.generate(&request, |progress: GenerationProgress| { ... }).await?;
model.cancel();
model.unload().await?;
```

- `ModelSource`: a file path on targets other than `wasm32`; in the browser, the model's URL and the URL of the directory holding wllama's files.
  - The wllama directory was planned for `LoadOptions`; it moved to `ModelSource`, the one type that already differs by target, so `LoadOptions` stays the same everywhere.
- `LoadOptions`:
  - context length in tokens
  - whether to keep the whole sliding-window cache, which lets a shared prompt prefix be reused across requests (llama.cpp's `swa_full`)
- `LoadProgress`: `Downloading` with bytes downloaded and total (the browser backend only), then `Loading` with the proportion loaded.
  - wllama reports no progress while loading, so the browser backend reports `Loading` at 0 and at 1 only.
- `GenerationRequest`:
  - messages, each a role and text; the model's own chat template formats them; text containing the model's control tokens is rejected, so a message cannot end its turn and forge another
  - an optional JSON schema the reply must match
  - temperature and seed
  - maximum tokens
- `Generation`:
  - the reply text
  - prompt tokens read, tokens reused from the cache, tokens generated
  - time spent reading the prompt and generating
- `GenerationProgress`, reported as the generation advances, for progress indicators:
  - `ReadingPrompt`: tokens read so far, tokens in the prompt, and how many of those came from the cache
  - `Generating`: tokens generated so far, the request's maximum, and the reply text so far
- `generate` returns `GenerationError::Cancelled` or `GenerationError::Failed(Error)`, so a caller can tell a cancellation from a failure.
  - A reply that reaches the maximum tokens unfinished is a failure when a schema is set, because its JSON is incomplete; without a schema it returns the text so far.
    - Both backends share the check: a schema reply at the maximum fails unless it parses as JSON, and a top-level number also needs whitespace after it, since a cut-off number still parses. The llama.cpp server inside wllama stops at the maximum without sampling the end token, and its JSON grammar allows trailing whitespace, so a complete reply can still reach the maximum.
  - A request whose prompt plus maximum reply does not fit in the context fails before reading the prompt.
    - wllama does not expose its tokenizer, so the browser backend checks on wllama's first prompt progress report, which wllama delivers only after reading the prompt's first batch, so such a request fails after one batch of 512 tokens (`n_batch`; llama.cpp's default of 2048 reads most prompts whole first).
- `cancel` stops the generation in progress and any queued before the call; dropping a `generate` future also stops its generation.
- One generation runs at a time per `Model`; a second call waits for the first.
- `unload` frees the model and resolves once its memory is freed; dropping a `Model` frees it too, but blocks the dropping thread until the model's worker thread has finished its current batch and freed the model.
- Errors use `minimer`'s static error type, which keeps the source error as text, so errors can cross the worker thread and be sent between async tasks.

Tokens, batches, samplers and the cache stay inside the crate. A caller that later needs one of them gets it added deliberately.

## Backends

Each backend is a module selected by `#[cfg]`, re-exported as the same public types.

### `llama_cpp`, for targets other than `wasm32`

- Wraps `llama-cpp-2` (and `llama-cpp-sys-2`, whose build script compiles llama.cpp with CMake and a C++ compiler).
- Features:
  - `llguidance` for schema-constrained output; the schema is compiled once per distinct schema and reused, because compiling took about 0.35 s.
  - `metal` on macOS and iOS; other GPU backends (CUDA, Vulkan) are opt-in crate features passed through.
- Generation runs on a blocking thread behind the `async` API, holding the context between requests.
- Progress: the prompt is decoded in batches, reporting after each batch, then after each generated token.
- Prefix reuse: the backend remembers the tokens in the cache, keeps the longest common prefix with the new prompt, clears the cache from the first differing token (`clear_kv_cache_seq`) and decodes only the rest.
  - With the window-only cache, a prefix is reused only while the cache still holds position 0, so once a prompt plus reply has filled the window nothing is reused; keeping the whole cache avoids this.
- Schemas are compiled without llguidance's flexible whitespace unless the schema sets its own llguidance options, because small models otherwise pad the reply with blank lines until the maximum.
- llama.cpp's logs go through `tracing`.
- Measured on an M4 Max with gemma3 12B Q4_K_M: 0.73 s median per graded answer with prefix reuse, 2.6 s without.

### `wllama`, for `wasm32` in the browser

- Calls wllama's JavaScript API through `wasm-bindgen` imports (`#[wasm_bindgen] extern "C"` declarations) of a small JavaScript module shipped in this crate (`src/wllama/wllama_bridge.js`).
  - The bridge imports wllama from the directory in `ModelSource` at run time and passes requests and streamed chunks across as JSON strings.
  - Each `Model` has its own wllama instance; `unload` and dropping the `Model` call wllama's `exit`, which ends its worker.
    - wllama's `exit` frees nothing while `loadModel` has yet to create its worker, so the bridge's `exit` first waits for a pending load to settle; dropping the `Model::load` future during the load then frees the model once it has loaded.
- Loading: wllama's model manager downloads the model into the browser's storage, or finds it there, reporting `Downloading`; then `loadModel` with `n_gpu_layers` set to keep every layer on the GPU through WebGPU, `swa_full` from `LoadOptions`, one sequence (`n_parallel`), so each request can reuse the prefix the previous one left in the cache, and 512-token batches (`n_batch`).
  - The design planned `loadModelFromUrl`, which does both steps at once; splitting them separates `Downloading` from `Loading` and lets the backend read the vocabulary between them.
- Control tokens: wllama does not expose its tokenizer, so the backend reads the token texts and types from the GGUF file's metadata in wllama's cache and rejects a message containing the text of a control, unknown or end-of-generation token. This matches the tokens llama.cpp parses as special in a templated prompt.
- Generation: `createChatCompletion` with `response_format` set to the JSON schema; wllama reuses the cached prompt prefix itself.
- Progress: `stream` with `onData`, `return_progress` and `timings_per_token`; each chunk carries either `prompt_progress` (`total`, `cache`, `processed`) or the next piece of the reply, with the timings so far.
  - `ReadingPrompt` comes after each batch wllama reads, starting with one whose read count is the reused count.
  - `Generating` comes with each piece of reply text; a token with no text of its own, such as whitespace the server holds back, reports nothing, so the counts can skip.
  - `Generation`'s counts and durations come from the final timings; the server counts the end token as generated, which the backend subtracts.
- Cancellation: each generation has an `AbortController`, which `cancel` and dropping the `generate` future abort; generations queued before a `cancel` see the cancellation when their turn comes.
- `Model` holds JavaScript handles, so on `wasm32` it is neither `Send` nor `Sync`.
- Builds:
  - wllama uses its default build when the browser has WebAssembly 64-bit memory and JSPI, and switches to its compatibility build by itself otherwise.
  - The app serves both builds' files itself; the bridge tells wllama where the compatibility build is (`setCompat`) instead of letting it fetch from a CDN.
  - `scripts/check-browser-features.sh` reports which browsers' current releases get which build.
- Measured on an M4 Max with gemma3 12B Q4_K_M:
  - Chrome, default build: 124 tokens/s reading the prompt, 27 generating; p95 3.0 s per graded answer with prefix reuse.
  - Safari, compatibility build: 26 tokens/s reading, 16 generating.
- The 7.3 GB model loads into GPU memory, not WebAssembly memory, so the compatibility build's 4 GB WebAssembly limit does not stop it.

## Assets the app serves in the browser

- wllama's JavaScript and WebAssembly files, default and compatibility builds, in the directory whose URL the app passes in `ModelSource`.
  - `scripts/fetch-wllama.sh <directory>` writes them, from the pinned npm release, after checking each package against its published integrity hash; the app runs it as part of its build and serves the directory.
  - They are not committed to this repository, and the crate's build script does not download them: a build script writes only to Cargo's output directory, which the app does not serve, and downloading there would break offline builds.
- The model file, at the URL in `ModelSource`, with `Content-Length`; without it wllama cannot check its cached copy and downloads the model again on every load.
- Cross-origin isolation headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`), which wllama's multithreaded build needs.

## Testing

- Shared request and response types: host unit tests.
- `llama_cpp`: integration tests that load a small GGUF (gemma-3-270m), generate with and without a schema, and check prefix reuse and cancellation.
- `wllama`: `wasm-bindgen-test` in a real browser, with the same tests against the same small model, since this backend's code only runs in a browser.
  - `scripts/test-browser.sh` runs them in headless Chrome, or in Safari with `--safari`. wasm-bindgen-test's runner serves the repository root, so the tests load `build/wllama/` and `build/models/` from it, with the cross-origin isolation headers it sends by default.
  - Safari runs each test in a page of its own, because loading a model into a page that had already loaded others hung inside wllama's compatibility build (wllama 3.8.1, Safari 26.6.2; not traced further). One reload, as the unload test does, works.
  - The runner sends large files without `Content-Length`, so in these tests wllama reports no download total and downloads the model again for every load.
- Shared request and fixture helpers for both backends' tests live in `tests/helpers/`.

## Out of scope for the first version

- Downloading or managing model files on targets other than `wasm32`; the app supplies the path.
- Images, audio, embeddings and tool calls.

## Platforms

- The browser backend serves the language app's grading, which runs only on the web.
- The llama.cpp backend covers desktop operating systems and the crate's own fast tests. llama.cpp also builds for iOS and Android, but no phone-sized model passed the grading bar, so neither is a target for now.
