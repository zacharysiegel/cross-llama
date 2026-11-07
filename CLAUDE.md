# CLAUDE.md

The `cross-llama` crate: one Rust API for running a GGUF language model through llama.cpp on targets other than `wasm32` and through wllama in the browser. `docs/design.md` is the design; `./scripts/setup.sh` prepares a machine.

## Workflow

- Changes go through a draft pull request to `main`.
- After a pull request that changes the crate is merged, publish it to crates.io from `main` with `cargo publish`, bumping the version in the pull request.
- Before merging, run `cargo test`, `cargo check --target wasm32-unknown-unknown` and `./scripts/test-browser.sh`.
