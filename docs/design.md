# cross-llama design

Runs a GGUF language model from Rust with one API on every target: through llama.cpp on targets other than `wasm32`, and through wllama (llama.cpp compiled to WebAssembly) in the browser. Its first consumer is the language app's grader, which sends a short conversation and a JSON schema and expects a schema-valid reply in a few seconds.

## Public API

```rust
let model: Model = Model::load(source, &load_options, |progress: LoadProgress| { ... }).await?;
let capability: ThinkingCapability = model.thinking_capability();
let generation: Generation = model.generate(&request, |progress: GenerationProgress| { ... }).await?;
model.cancel();
model.unload().await?;

// wasm32 only
download_model(&source, |progress: DownloadProgress| { ... }).await?;
let downloaded: bool = is_model_downloaded(&source).await?;
let capability: BrowserCapability = probe_browser_capability().await?;
```

- `ModelSource`: a file path on targets other than `wasm32`; in the browser, the model's URL and the URL of the directory holding wllama's files.
  - The wllama directory was planned for `LoadOptions`; it moved to `ModelSource`, the one type that already differs by target, so `LoadOptions` stays the same everywhere.
- `LoadOptions`:
  - context length in tokens
  - whether to keep the whole sliding-window cache, which lets a shared prompt prefix be reused across requests (llama.cpp's `swa_full`)
- `LoadProgress`: `Downloading` with bytes downloaded and total (the browser backend only), then `Loading` with the proportion loaded.
  - wllama reports no progress while loading, so the browser backend reports `Loading` at 0 and at 1 only.
- `GenerationRequest`:
  - messages, each a role and text, formatted as described under "Chat template"; text containing the model's control tokens is rejected, so a message cannot end its turn and forge another
  - an optional JSON schema the answer must match; it never constrains the reasoning
  - temperature and seed
  - maximum tokens: the most the model generates, reasoning and answer together; the delimiters around the reasoning are not counted
  - thinking (`ThinkingKind`), a thinking budget and a thinking effort (`ThinkingEffortKind`), described under "Thinking"
- `Generation`:
  - the answer text
  - the reasoning, its text and token count, absent with thinking disabled
  - prompt tokens read, tokens reused from the cache, tokens generated
  - time spent reading the prompt and generating, the reasoning included
- `GenerationProgress`, reported as the generation advances, for progress indicators:
  - `ReadingPrompt`: tokens read so far, tokens in the prompt, and how many of those came from the cache
  - `GeneratingReasoning`: reasoning tokens so far, the request's maximum, and the reasoning so far
  - `GeneratingAnswer`: tokens generated so far, the reasoning's included, the request's maximum, and the answer so far
- `thinking_capability` returns a `ThinkingCapability`: whether the crate separates the model's reasoning, and whether its template reads `enable_thinking` and `reasoning_effort`.
- `generate` returns `GenerationError::Cancelled` or `GenerationError::Failed(Error)`, so a caller can tell a cancellation from a failure.
  - A reply that reaches the maximum tokens unfinished is a failure when a schema is set, because its JSON is incomplete; without a schema it returns the text so far.
    - Both backends share the check: a schema reply at the maximum fails unless it parses as JSON, and a top-level number also needs whitespace after it, since a cut-off number still parses. The llama.cpp server inside wllama stops at the maximum without sampling the end token, and its JSON grammar allows trailing whitespace, so a complete reply can still reach the maximum.
  - A request whose prompt plus maximum reply does not fit in the context fails before reading the prompt.
    - wllama does not expose its tokenizer, so the browser backend checks on wllama's first prompt progress report, which wllama delivers only after reading the prompt's first batch, so such a request fails after one batch of 512 tokens (`n_batch`; llama.cpp's default of 2048 reads most prompts whole first).
- `cancel` stops the generation in progress and any queued before the call; dropping a `generate` future also stops its generation.
- One generation runs at a time per `Model`; a second call waits for the first.
- `unload` frees the model and resolves once its memory is freed; dropping a `Model` frees it too, but blocks the dropping thread until the model's worker thread has finished its current batch and freed the model.
- Errors use `minimer`'s static error type, which keeps the source error as text, so errors can cross the worker thread and be sent between async tasks.
- In the browser only:
  - `download_model` puts the model in wllama's cache in the browser's storage, reporting `DownloadProgress`, without loading it or starting a wllama worker; `Model::load` then finds it there.
  - `is_model_downloaded` reads only that cache.
  - `probe_browser_capability` returns a `BrowserCapability`:
    - the WebGPU adapter's `maxBufferSize`, absent with no adapter or only a fallback adapter
    - cross-origin isolation
    - which wllama build the browser gets (`WllamaBuildKind`), by wllama's rule under "Builds"
    - storage quota, usage and whether it is persisted
    - `navigator.deviceMemory`, where the browser reports it

Tokens, batches, samplers and the cache stay inside the crate. A caller that later needs one of them gets it added deliberately.

## Chat template

- Both backends render the GGUF file's Jinja template (`tokenizer.chat_template`) with the same code, minijinja, so the same request gives the same prompt; a file without a template gets ChatML, llama.cpp's own fallback.
- Rendering follows Hugging Face's transformers and llama.cpp's server: `trim_blocks` and `lstrip_blocks`, Python's string and dictionary methods (minijinja-contrib's `pycompat`), `raise_exception`, and the variables `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, `enable_thinking` and, when the request sets an effort, `reasoning_effort`.
- With thinking disabled, `enable_thinking` is false, so a reasoning model's prompt closes its thinking block itself (gemma4) and the reply holds only the answer.
- A sequence token the template writes and tokenizing also adds is removed from the rendered prompt, as llama.cpp's server does.
- llama.cpp's built-in templates detect known templates by their text and missed gemma4's.
- When a template rejects or drops a system message (gemma-2, Mistral v0.x), checked once when the template is read, a leading system message's text joins the next message after a newline, as llama.cpp's server does.

## Thinking

- `ThinkingKind::Enabled` renders `enable_thinking` true; `ThinkingEffortKind` renders `reasoning_effort` as `low`, `medium` or `high`.
- `ThinkingCapability`:
  - `switches_thinking` and `reads_thinking_effort`: the template reads the variable, by minijinja's static analysis joined with the lookups of one probe render. The analysis misses a variable the template sets from itself (gemma4's `set enable_thinking = enable_thinking | default(false)`); the probe misses branches it does not take.
  - `separates_reasoning`: the crate knows delimiters whose start the template writes and whose end is one token of the vocabulary.
- A request fails before generating when it enables thinking without known delimiters, sets a budget with thinking disabled, or sets an effort the template does not read.
- Known delimiters, the thinking tags of llama.cpp's chat parsers (`common/chat.cpp` and `common/parsers/` in llama-cpp-sys-2 0.1.161), with the text llama.cpp writes between the end delimiter and the answer when it continues a reply:
  - Gemma 4: `<|channel>thought` to `<channel|>`
  - gpt-oss: `<|channel|>analysis<|message|>` to `<|end|>`, then `<|start|>assistant<|channel|>final<|message|>`
  - Ministral 3 and Magistral: `[THINK]` to `[/THINK]`
  - Qwen3, DeepSeek-R1, Kimi K2, LFM2, MiniCPM5 and Ling 3: `<think>` to `</think>`
  - Forms whose answer opens with more than fixed text (Cohere's, Kimi K3's) are left out.
- Generation, the same on both backends, in two completions of one sequence:
  - Reasoning: the prompt plus the start delimiter, unless the template already ends with it, generated without the schema until the end delimiter, the end of the turn or the reasoning limit, the budget or the maximum, whichever is smaller.
  - Answer: the reasoning's prompt, the reasoning as generated, the end delimiter and the answer's opening text, generated under the schema with the rest of the maximum. Both backends tokenize this text again and reuse the cached prefix, so they read the same tokens, and only the delimiter and any text that tokenizes differently are read again.
  - The llama.cpp backend stops at the end delimiter's token. wllama's server stops at its text as a `stop` string, which it can match only because the request marks the token `preserved_tokens`; it otherwise streams a control token as no text.
  - A budget of zero skips the reasoning completion; a reasoning that uses the whole maximum skips the answer completion, which then ends at the maximum.
  - The reasoning's prompt is checked against the context with the whole maximum, before any progress, and the answer's prompt with its own token limit; the delimiters need room too, so a request that leaves none fails after the reasoning.
  - `ReadingPrompt` comes from the first completion only; `Generation`'s prompt counts and reading time are that completion's.
  - The reasoning is returned without the whitespace around it.
- Measured on an M4 Max with gemma4 12B QAT, twelve grader cases at temperature 0, with thinking a maximum of 4096 tokens:
  - Thinking disabled: every reply equals Ollama's on both backends.
  - No budget: reasoning of 123 to 760 tokens, and two that repeated a sentence until the maximum and failed; the other ten answers match the schema. Median 13.8 s with llama.cpp, 28.6 s in Chrome.
  - Budget of 128: every answer matches the schema. Median 4.3 s with llama.cpp, 8.7 s in Chrome.
  - Both backends give the same reasoning token counts and answers in every case.
  - The reasoning text can still differ in wording between backends from the same prompt tokens; one case's diverged about halfway through.
- llama.cpp's reasoning budget sampler and lazy grammar were not used: llama-cpp-2 exposes no budget sampler and a lazy grammar only in GBNF, not the llguidance schema grammar the llama.cpp backend uses; the server's grammar for a plain completion constrains from the first token, and its chat completion grammar for gemma4 wraps the answer in a code fence (see `wllama`). Two plain completions behave the same on both backends.

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
  - The bridge reads the model's own parts from the cache in place of the model manager's `getModelOrDownload`, whose listing of every cached model throws while any split model lacks a part's metadata, as a download stopped partway leaves it (wllama 3.8.1).
    - A copy with a part missing, without metadata or of the wrong size is not downloaded; downloading it calls `downloadModel`, which keeps each part whose size matches its metadata, or, without metadata, the server's `Content-Length`, and fetches the others again. The progress counts the parts kept with metadata as downloaded, and ends at the whole model's size.
  - A Web Lock (`navigator.locks`) named for the model's absolute URL covers downloading, reading and loading the cached files, in `Model::load` and `download_model` alike, so two pages, or a prefetch and a load, never write the same file at once. Dropping the future releases it once wllama's download, read or load under it has settled, since an aborted download still writes until then.
  - The chat template, the sequence tokens and whether tokenizing adds them come from wllama after the load.
  - wllama's `loadModel` resolves even when llama.cpp fails to load the model, with `success` false in the loaded context and no reason (wllama 3.8.1), so the bridge rejects then and `Model::load` fails, releasing the lock and the wllama instance.
- Control tokens: wllama does not expose its tokenizer, so the backend reads the token texts and types from the GGUF file's metadata in wllama's cache and rejects a message containing the text of a control, unknown or end-of-generation token. This matches the tokens llama.cpp parses as special in a templated prompt.
- Generation: `createCompletion` of the prompt the crate rendered, with the JSON schema as `json_schema`, which gets the llama.cpp server's schema-only grammar; wllama reuses the cached prompt prefix itself.
  - The planned `createChatCompletion` with Jinja on constrains a schema reply to JSON or JSON inside a code fence, which the server counts but leaves out of the text; gemma4 took the fence on every grader reply, five tokens more, and two of twelve replies differed from the llama.cpp backend's.
  - The request turns off the samplers the llama.cpp server enables by default or takes from the model's metadata (`top_k`, `top_p`, `min_p` and others), so both backends sample from the whole distribution after temperature.
- Progress: `stream` with `onData`, `return_progress` and `timings_per_token`; each chunk carries either `prompt_progress` (`total`, `cache`, `processed`) or the next piece of the reply, with the timings so far.
  - `ReadingPrompt` comes after each batch wllama reads, starting with one whose read count is the reused count.
  - `GeneratingReasoning` and `GeneratingAnswer` come with each piece of reply text; a token with no text of its own, such as whitespace the server holds back, reports nothing, so the counts can skip.
  - `Generation`'s counts and durations come from the final timings; the server counts the end token, or the token of a stop string, as generated, which the backend subtracts.
- Cancellation: each generation has an `AbortController`, which `cancel` and dropping the `generate` future abort; generations queued before a `cancel` see the cancellation when their turn comes.
- `Model` holds JavaScript handles, so on `wasm32` it is neither `Send` nor `Sync`.
- Builds:
  - wllama 3.8.1 uses its default build when the browser has WebAssembly 64-bit memory and JSPI, and switches to its compatibility build by itself otherwise, except on Firefox, which `setCompat`'s default mode keeps on the default build, without WebGPU when it lacks JSPI.
  - The app serves both builds' files itself; the bridge tells wllama where the compatibility build is (`setCompat`) instead of letting it fetch from a CDN.
  - `scripts/check-browser-features.sh` reports which browsers' current releases get which build.
- Measured on an M4 Max with gemma3 12B Q4_K_M:
  - Chrome, default build: 124 tokens/s reading the prompt, 27 generating; p95 3.0 s per graded answer with prefix reuse.
  - Safari, compatibility build: 26 tokens/s reading, 16 generating.
- Measured on an M4 Max with gemma4 12B QAT, Chrome, default build, twelve grader cases: 20 tokens/s generating, median 3.3 s per graded answer with prefix reuse; replies and token counts identical to the llama.cpp backend's.
- gemma3 12B Q4_K_M (7.3 GB) loads into GPU memory, not WebAssembly memory, so the compatibility build's 4 GB WebAssembly limit does not stop it.

## Assets the app serves in the browser

- wllama's JavaScript and WebAssembly files, default and compatibility builds, in the directory whose URL the app passes in `ModelSource`.
  - `scripts/fetch-wllama.sh <directory>` writes them, from the pinned npm release, after checking each package against its published integrity hash, and records the version in a `VERSION` file there; it does nothing when that file already names the pinned version. The app runs it as part of its build and serves the directory.
  - They are not committed to this repository, and the crate's build script does not download them: a build script writes only to Cargo's output directory, which the app does not serve, and downloading there would break offline builds.
- The model file, at the URL in `ModelSource`, with `Content-Length`; without it wllama cannot check its cached copy and downloads the model again on every load.
- Cross-origin isolation headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`), which wllama's multithreaded build needs.

## Testing

- Shared request and response types: host unit tests.
- Chat template rendering: host unit tests, including that gemma-3-270m's template renders as llama.cpp's built-in template does.
- `llama_cpp`: integration tests that load a small GGUF (gemma-3-270m), generate with and without a schema, and check prefix reuse and cancellation.
- `wllama`: `wasm-bindgen-test` in a real browser, with the same tests against the same small model, since this backend's code only runs in a browser.
  - Unit tests under `src/wllama/` check that `exit` frees the model, through wllama's own `isModelLoaded`, and that `download_model` waits for a held model lock, and a released lock for the operations it holds, none of which the public API can observe.
  - The cache tests download `tests/data/empty.gguf`, 24 bytes, each under its own query string, which gives it its own cache entry.
    - The split model tests serve two copies of it named as parts (`empty-00001-of-00002.gguf`) and cache the first part whole and the second part's first bytes, or all of them, without metadata, writing through wllama's internal storage backend (`cacheManager.sb`, wllama 3.8.1).
  - The failed load test loads `tests/data/vocabulary-only.gguf`, which has the token metadata the vocabulary read needs but no architecture, so llama.cpp rejects it.
  - `scripts/test-browser.sh` runs them in headless Chrome, or in Safari with `--safari`. wasm-bindgen-test's runner serves the repository root, so the tests load `build/wllama/` and `build/models/` from it, with the cross-origin isolation headers it sends by default.
  - Safari runs each test in a page of its own, because loading a model into a page that had already loaded others hung inside wllama's compatibility build (wllama 3.8.1, Safari 26.6.2; not traced further). One reload, as the unload test does, works.
  - The runner sends files over 32 KB without `Content-Length`, so in these tests wllama reports no download total for the model and downloads it again for every load.
  - The script sets `NO_PROXY` for the loopback addresses, since an HTTP proxy in the environment otherwise receives the runner's requests to the WebDriver.
- Shared request and fixture helpers for both backends' tests live in `tests/helpers/`.
- Thinking: host unit tests for delimiter detection, the capability, request checks and the reasoning's prompts and limits.
  - The test model has no reasoning delimiters, so both backends' public tests check that it reports no thinking control and rejects enabled thinking; in-crate tests on each backend substitute delimiters whose end the model never writes (`<unused0>`) to run both completions, with a budget ending the reasoning.

## Out of scope

- Downloading or managing model files on targets other than `wasm32`; the app supplies the path.
- Images, audio, embeddings and tool calls.

## Platforms

- The browser backend serves the language app's grading, which runs only on the web.
- The llama.cpp backend covers desktop operating systems and the crate's own fast tests. llama.cpp also builds for iOS and Android, but no phone-sized model passed the grading bar, so neither is a target for now.
