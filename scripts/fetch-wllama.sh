#!/usr/bin/env bash

# Downloads the pinned wllama release from the npm registry, checks each package against its published integrity
# hash, and copies the files the browser loads into a directory the app serves:
#   <destination>/index.js            wllama's JavaScript module
#   <destination>/wllama.wasm         the default build
#   <destination>/compat/wllama.js    the compatibility build's worker
#   <destination>/compat/wllama.wasm  the compatibility build
#   <destination>/VERSION             the wllama version written
# Does nothing when <destination>/VERSION already names the pinned version.
#
# Usage:
#   ./scripts/fetch-wllama.sh <destination directory>

set -euo pipefail

readonly WLLAMA_VERSION="3.8.1"
readonly WLLAMA_TARBALL_URL="https://registry.npmjs.org/@wllama/wllama/-/wllama-${WLLAMA_VERSION}.tgz"
readonly WLLAMA_INTEGRITY="sha512-Pl7BEJhPU8OHnV/yhn3DFZYhLLOdSHh+4TND79WCftHnlUAdQPAsyc/CfIOY+4mVxvf5VW3Xw3Uf7BBfotS1gw=="
readonly WLLAMA_COMPAT_TARBALL_URL="https://registry.npmjs.org/@wllama/wllama-compat/-/wllama-compat-${WLLAMA_VERSION}.tgz"
readonly WLLAMA_COMPAT_INTEGRITY="sha512-IvmjrJ+JzDbt3w1FQq1yAmnFEiLGCceowK2Q80zW+k9ChFkPBM1ByX/9xm2xVxyd8MHC50VpuDo6SDE83k9kIQ=="

function fail {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

destination="${1:-}"

if test -z "${destination}"; then
    printf 'usage: %s <destination directory>\n' "$0" >&2
    exit 64
fi

readonly VERSION_PATH="${destination}/VERSION"
fetched_version="$(cat "${VERSION_PATH}" 2>/dev/null || true)"

if test "${fetched_version}" = "${WLLAMA_VERSION}"; then
    exit 0
fi

working_directory="$(mktemp -d)"
trap 'rm -rf "${working_directory}"' EXIT

# npm's integrity strings are "sha512-" followed by the base64 of the tarball's SHA-512 digest.
function download_verified_tarball {
    local url="$1"
    local expected_integrity="$2"
    local tarball_path="$3"

    local download_status=0
    curl --silent --show-error --fail --location --max-time 300 --output "${tarball_path}" "${url}" || download_status=$?

    if test "${download_status}" -ne 0; then
        fail "could not download ${url} (curl exit ${download_status})"
    fi

    local actual_integrity
    actual_integrity="sha512-$(openssl dgst -sha512 -binary "${tarball_path}" | openssl base64 -A)"

    if test "${actual_integrity}" != "${expected_integrity}"; then
        fail "integrity mismatch for ${url}: expected ${expected_integrity}, got ${actual_integrity}"
    fi
}

function extract_tarball {
    local tarball_path="$1"
    local extraction_directory="$2"

    mkdir -p "${extraction_directory}"
    tar -xzf "${tarball_path}" -C "${extraction_directory}"
}

download_verified_tarball "${WLLAMA_TARBALL_URL}" "${WLLAMA_INTEGRITY}" "${working_directory}/wllama.tgz"
download_verified_tarball "${WLLAMA_COMPAT_TARBALL_URL}" "${WLLAMA_COMPAT_INTEGRITY}" "${working_directory}/wllama-compat.tgz"

extract_tarball "${working_directory}/wllama.tgz" "${working_directory}/wllama"
extract_tarball "${working_directory}/wllama-compat.tgz" "${working_directory}/wllama-compat"

mkdir -p "${destination}/compat"
cp "${working_directory}/wllama/package/esm/index.js" "${destination}/index.js"
cp "${working_directory}/wllama/package/esm/wasm/wllama.wasm" "${destination}/wllama.wasm"
cp "${working_directory}/wllama-compat/package/wasm/wllama.js" "${destination}/compat/wllama.js"
cp "${working_directory}/wllama-compat/package/wasm/wllama.wasm" "${destination}/compat/wllama.wasm"
printf '%s\n' "${WLLAMA_VERSION}" > "${VERSION_PATH}"

printf 'wllama %s written to %s\n' "${WLLAMA_VERSION}" "${destination}"
