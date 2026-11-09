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

function fetch_wllama_unless_current {
    local wanted_version
    wanted_version=$(grep -E '^readonly WLLAMA_VERSION=' "${FETCH_WLLAMA_SCRIPT}" | cut -d'"' -f2)
    local fetched_version
    fetched_version=$(cat "${WLLAMA_DIRECTORY}/VERSION" 2>/dev/null || true)

    if test "${fetched_version}" = "${wanted_version}"; then
        return 0
    fi

    rm -rf "${WLLAMA_DIRECTORY}"
    "${FETCH_WLLAMA_SCRIPT}" "${WLLAMA_DIRECTORY}"
    echo "${wanted_version}" > "${WLLAMA_DIRECTORY}/VERSION"
}

check_trunk_installed
fetch_wllama_unless_current
trunk build --release
exec cargo run --quiet --release --bin server -- "$@"
