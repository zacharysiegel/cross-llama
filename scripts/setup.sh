#!/bin/zsh
# One-shot bootstrap for a fresh dev machine. Idempotent; safe to re-run.
#   - checks for the programs the build and scripts need
#   - installs the Rust toolchain rust-toolchain.toml names, with its targets
#   - fetches wllama's browser files into build/wllama/ when they are missing or from another version

set -euo pipefail

# The repository root is the parent of this script's directory, whatever the caller's working directory.
repo_dir="${0:A:h:h}"
cd "${repo_dir}"

readonly WLLAMA_DIRECTORY="build/wllama"

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
    # scripts/fetch-wllama.sh and scripts/check-browser-features.sh
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

echo "Setup complete."
