#!/bin/zsh
# One-shot bootstrap for a fresh dev machine. Idempotent; safe to re-run.
#   - checks for the programs the build and scripts need
#   - installs the Rust toolchain rust-toolchain.toml names, with its targets
#   - fetches wllama's browser files into build/wllama/ when they are missing or from another version
#   - fetches the integration tests' model into build/models/ when it is missing or fails its checksum
#   - installs, into build/tools/, the browser tests' wasm-bindgen test runner at the wasm-bindgen version Cargo.lock
#     pins and a ChromeDriver matching the installed Google Chrome

set -euo pipefail

# The repository root is the parent of this script's directory, whatever the caller's working directory.
repo_dir="${0:A:h:h}"
cd "${repo_dir}"

source ./scripts/browser-tools.zsh

readonly WLLAMA_DIRECTORY="build/wllama"
readonly TEST_MODEL_DIRECTORY="build/models"
readonly TEST_MODEL_FILE_NAME="gemma-3-270m-it-Q8_0.gguf"
readonly TEST_MODEL_URL="https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/e7647be17ae1108f2f605ed061ca0608b171afff/${TEST_MODEL_FILE_NAME}"
readonly TEST_MODEL_SHA256="0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3"
readonly CHROME_FOR_TESTING_DOWNLOADS_URL="https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build-with-downloads.json"

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
    # the browser tests' ChromeDriver download
    required_program "unzip"   "brew install unzip"
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

function install_wasm_bindgen_test_runner_unless_present {
    local wanted_version
    wanted_version=$(print_wasm_bindgen_version)
    local install_root
    install_root=$(print_wasm_bindgen_install_root)

    if test -x "${install_root}/bin/wasm-bindgen-test-runner"; then
        return 0
    fi

    echo "Installing wasm-bindgen-cli ${wanted_version} into ${install_root}"
    cargo install --locked --quiet --root "${install_root}" --version "${wanted_version}" wasm-bindgen-cli
}
install_wasm_bindgen_test_runner_unless_present

function install_chromedriver_unless_present {
    local chrome_build
    chrome_build=$(print_chrome_build)

    if test -z "${chrome_build}"; then
        echo "Google Chrome is required by the browser tests"
        echo "  install: brew install --cask google-chrome"
        exit 1
    fi

    local install_directory
    install_directory=$(print_chromedriver_directory)

    if test -x "${install_directory}/chromedriver"; then
        return 0
    fi

    local platform
    platform=$(chrome_for_testing_platform)
    local download_url
    download_url=$(curl --silent --show-error --fail --location --max-time 120 "${CHROME_FOR_TESTING_DOWNLOADS_URL}" | jq --raw-output --arg build "${chrome_build}" --arg platform "${platform}" '.builds[$build].downloads.chromedriver[]? | select(.platform == $platform) | .url')

    if test -z "${download_url}"; then
        echo "Chrome for Testing has no ChromeDriver for Chrome ${chrome_build} on ${platform}"
        exit 1
    fi

    echo "Installing ChromeDriver for Chrome ${chrome_build} into ${install_directory}"
    local working_directory
    working_directory=$(mktemp -d)
    curl --silent --show-error --fail --location --max-time 300 --output "${working_directory}/chromedriver.zip" "${download_url}"
    unzip -q "${working_directory}/chromedriver.zip" -d "${working_directory}"
    mkdir -p "${install_directory}"
    mv "${working_directory}/chromedriver-${platform}/chromedriver" "${install_directory}/chromedriver"
    rm -rf "${working_directory}"
}

function chrome_for_testing_platform {
    local operating_system
    operating_system=$(uname -s)
    local architecture
    architecture=$(uname -m)

    case "${operating_system}-${architecture}" in
        Darwin-arm64) echo "mac-arm64" ;;
        Darwin-x86_64) echo "mac-x64" ;;
        *) echo "linux64" ;;
    esac
}
install_chromedriver_unless_present

echo "Setup complete."
