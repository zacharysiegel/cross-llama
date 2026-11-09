#!/usr/bin/env bash
# Fetches wllama's files into build/wllama/ when they are missing or from another version, builds the web client and
# its server with cargo-leptos through the xtask, and serves the page, wllama's files and the model.
#
# Usage:
#   ./scripts/serve-web.sh [<port>]

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${package_dir}"

readonly WLLAMA_DIRECTORY="build/wllama"
readonly FETCH_WLLAMA_SCRIPT="../../scripts/fetch-wllama.sh"

port="${1:-}"

if test -n "${port}"; then
    export LEPTOS_SITE_ADDR="127.0.0.1:${port}"
fi

"${FETCH_WLLAMA_SCRIPT}" "${WLLAMA_DIRECTORY}"
exec cargo xtask leptos serve --release
