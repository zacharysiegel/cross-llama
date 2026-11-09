#!/usr/bin/env bash
# Fetches wllama's files into build/wllama/ when they are missing or from another version, builds the web client and
# its server with cargo-leptos, and serves the page, wllama's files and the model.
#
# Usage:
#   ./scripts/serve-web.sh [<port>]

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${package_dir}"

readonly WLLAMA_DIRECTORY="build/wllama"
readonly FETCH_WLLAMA_SCRIPT="../../scripts/fetch-wllama.sh"

function fail {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

function check_cargo_leptos {
    local cargo_leptos_path
    cargo_leptos_path="$(command -v cargo-leptos || true)"

    if test -z "${cargo_leptos_path}"; then
        fail "cargo-leptos is required; install it with: cargo install --locked cargo-leptos"
    fi
}

# cargo-leptos runs the wasm-bindgen on the PATH, which must be the version Cargo.lock locks.
function check_wasm_bindgen {
    local locked_version
    locked_version="$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | tail -1 | cut -d'"' -f2)"
    local installed_version
    installed_version="$(wasm-bindgen --version 2>/dev/null | cut -d' ' -f2 || true)"

    if test "${installed_version}" != "${locked_version}"; then
        fail "wasm-bindgen ${locked_version} is required (found '${installed_version}'); install it with: cargo install --locked --version ${locked_version} wasm-bindgen-cli"
    fi
}

check_cargo_leptos
check_wasm_bindgen

port="${1:-}"

if test -n "${port}"; then
    export LEPTOS_SITE_ADDR="127.0.0.1:${port}"
fi

"${FETCH_WLLAMA_SCRIPT}" "${WLLAMA_DIRECTORY}"
exec cargo leptos serve --release
