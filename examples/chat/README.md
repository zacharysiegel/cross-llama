# cross-llama chat demo

A single-prompt chat on one shared core (`src/chat.rs`) that runs a local language model through cross-llama: in the browser through the wllama backend, and in a terminal on macOS (Metal) and Linux (CPU) through the llama.cpp backend. Each prompt is answered on its own, without earlier prompts or replies; the reply streams as it is generated and ends with its speed in tokens per second.

The model, Qwen2.5 0.5B Instruct at Q4_K_M, is committed in [models/](./models/) with Git LFS under the Apache 2.0 license; its source and checksum are in [models/README.md](./models/README.md). Run `git lfs pull` if the file is a pointer.

Both clients offer thinking only for a model whose reasoning cross-llama separates from its answer, which Qwen2.5 is not. A reply that thinks shows its reasoning apart from its answer and has a reasoning budget of 512 tokens.

## Terminal client

```sh
./scripts/run-terminal.sh [--thinking] [<model path>]
```

Reads one prompt per line from standard input until it ends, so prompts can also be piped in. The answer goes to standard output, the reasoning and the speed to standard error. `<model path>` is another GGUF model. Building needs CMake and a C++ compiler, and on Linux also libclang.

## Web client

```sh
./scripts/serve-web.sh [<port>]
```

Fetches wllama's files into `build/wllama/`, then builds and serves the Leptos app with cargo-leptos at http://127.0.0.1:4417/ or the given port. The server renders the page, sends the cross-origin isolation headers wllama needs, and serves wllama's files and the model with `Content-Length` and `Range` support; the client runs the model in the browser. The page downloads the model into the browser's storage on its first visit and loads it from there afterwards, and shows a "Think before answering" checkbox when the model offers thinking.

Needs cargo-leptos and the wasm-bindgen command line at the version `Cargo.lock` locks; the script checks both and prints the install command for a missing one.
