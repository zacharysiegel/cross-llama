#!/usr/bin/env bash
# Builds the web client with Trunk, fetches wllama's files into build/wllama/ when they are missing or from another
# version, and serves the site, wllama's files and the model.
#
# Usage:
#   ./scripts/serve-web.sh [<port>]

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${package_dir}"

readonly WLLAMA_DIRECTORY="build/wllama"
readonly FETCH_WLLAMA_SCRIPT="../../scripts/fetch-wllama.sh"

function check_trunk_installed {
    local trunk_path
    trunk_path=$(command -v trunk || true)

    if test -z "${trunk_path}"; then
        echo "The \`trunk\` program is required"
        echo "  install: cargo install trunk --locked"
        exit 1
    fi
}

check_trunk_installed
"${FETCH_WLLAMA_SCRIPT}" "${WLLAMA_DIRECTORY}"
trunk build --release
exec cargo run --quiet --release --bin server -- "$@"
