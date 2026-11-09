# cross-llama chat demo

A single-prompt chat on one shared core (`src/chat.rs`) that runs a local language model through cross-llama: in the browser through the wllama backend, and in a terminal on macOS (Metal) and Linux (CPU) through the llama.cpp backend. Each prompt is answered on its own, without earlier prompts or replies; the reply streams as it is generated and ends with its speed in tokens per second.

The model, Qwen2.5 0.5B Instruct at Q4_K_M, is committed in [models/](./models/) with Git LFS under the Apache 2.0 license; its source and checksum are in [models/README.md](./models/README.md). Run `git lfs pull` if the file is a pointer.

## Terminal client

```sh
./scripts/run-terminal.sh
```

Reads one prompt per line from standard input until it ends, so prompts can also be piped in. An optional argument is another GGUF model's path. Building needs CMake and a C++ compiler, and on Linux also libclang.

## Web client

```sh
./scripts/serve-web.sh
```

Builds the Leptos client with [Trunk](https://trunkrs.dev) (`cargo install trunk --locked`), fetches wllama's files into `build/wllama/`, and serves the site, wllama's files and the model at http://127.0.0.1:4417/ with the cross-origin isolation headers wllama needs. An optional argument is another port. The page downloads the model into the browser's storage on its first visit and loads it from there afterwards.
