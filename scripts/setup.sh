#!/bin/zsh
# One-shot bootstrap for a fresh dev machine. Idempotent; safe to re-run.
#   - checks for the programs the build and scripts need
#   - installs the Rust toolchain rust-toolchain.toml names, with its targets
#   - fetches wllama's browser files into build/wllama/ when they are missing or from another version
#   - fetches the integration tests' model into build/models/ when it is missing or fails its checksum

set -euo pipefail

# The repository root is the parent of this script's directory, whatever the caller's working directory.
repo_dir="${0:A:h:h}"
cd "${repo_dir}"

readonly WLLAMA_DIRECTORY="build/wllama"
readonly TEST_MODEL_DIRECTORY="build/models"
readonly TEST_MODEL_FILE_NAME="gemma-3-270m-it-Q8_0.gguf"
readonly TEST_MODEL_URL="https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/e7647be17ae1108f2f605ed061ca0608b171afff/${TEST_MODEL_FILE_NAME}"
readonly TEST_MODEL_SHA256="0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3"

function check_prerequisites {
    function required_program {
        local program_name="$1"
        local install_hint="$2"
        local program_path
        program_path=$(command -v "${program_name}" || true)

        if test -z "${program_path}"; then
            echo "The \`${program_name}\` program is required"
            echo "  install: ${install_hint}"
            exit 1
        fi
    }

    required_program "rustup"  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    # llama-cpp-sys-2's build script compiles llama.cpp with these.
    required_program "cmake"   "brew install cmake"
    required_program "c++"     "xcode-select --install"
    # scripts/fetch-wllama.sh, scripts/check-browser-features.sh and the test model's download
    required_program "curl"    "brew install curl"
    required_program "openssl" "brew install openssl"
    required_program "tar"     "brew install gnu-tar"
    required_program "jq"      "brew install jq"
}
check_prerequisites

echo "Installing the Rust toolchain from rust-toolchain.toml"
rustup toolchain install

function fetch_wllama_unless_current {
    local wanted_version
    wanted_version=$(grep -E '^readonly WLLAMA_VERSION=' scripts/fetch-wllama.sh | cut -d'"' -f2)
    local fetched_version
    fetched_version=$(cat "${WLLAMA_DIRECTORY}/VERSION" 2>/dev/null || true)

    if test "${fetched_version}" = "${wanted_version}"; then
        return 0
    fi

    echo "Fetching wllama ${wanted_version} into ${WLLAMA_DIRECTORY}"
    rm -rf "${WLLAMA_DIRECTORY}"
    ./scripts/fetch-wllama.sh "${WLLAMA_DIRECTORY}"
    echo "${wanted_version}" > "${WLLAMA_DIRECTORY}/VERSION"
}
fetch_wllama_unless_current

function sha256_of_file {
    local file_path="$1"

    openssl dgst -sha256 -r "${file_path}" | cut -d' ' -f1
}

function fetch_test_model_unless_verified {
    local model_path="${TEST_MODEL_DIRECTORY}/${TEST_MODEL_FILE_NAME}"
    local existing_sha256
    existing_sha256=$(sha256_of_file "${model_path}" 2>/dev/null || true)

    if test "${existing_sha256}" = "${TEST_MODEL_SHA256}"; then
        return 0
    fi

    echo "Fetching the test model into ${TEST_MODEL_DIRECTORY}"
    mkdir -p "${TEST_MODEL_DIRECTORY}"
    local partial_path="${model_path}.partial"
    local download_status=0
    curl --silent --show-error --fail --location --max-time 1800 --output "${partial_path}" "${TEST_MODEL_URL}" || download_status=$?

    if test "${download_status}" -ne 0; then
        rm -f "${partial_path}"
        echo "Could not download ${TEST_MODEL_URL} (curl exit ${download_status})"
        exit 1
    fi

    local downloaded_sha256
    downloaded_sha256=$(sha256_of_file "${partial_path}")

    if test "${downloaded_sha256}" != "${TEST_MODEL_SHA256}"; then
        rm -f "${partial_path}"
        echo "Checksum mismatch for ${TEST_MODEL_URL}: expected SHA-256 ${TEST_MODEL_SHA256}, got ${downloaded_sha256}"
        exit 1
    fi

    mv "${partial_path}" "${model_path}"
}
fetch_test_model_unless_verified

echo "Setup complete."
