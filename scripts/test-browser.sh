#!/bin/zsh
# Runs the browser backend's tests (tests/wllama.rs) in a real browser, through wasm-bindgen-test's runner, which serves
# the repository root, including build/wllama/ and build/models/, with the cross-origin isolation headers wllama's
# multithreaded build needs.
#
# Usage:
#   ./scripts/test-browser.sh            in headless Google Chrome
#   ./scripts/test-browser.sh --safari   in Safari, which needs `safaridriver --enable` run once
# Further arguments go to the test binary, for example a test name to filter by.
#
# Safari runs each test in a page of its own: there, wllama's compatibility build can hang while loading a model into a
# page that has already loaded one.

set -euo pipefail

repo_dir="${0:A:h:h}"
cd "${repo_dir}"

source ./scripts/browser-tools.zsh

# Covers loading the model in the slower compatibility build.
readonly TEST_TIMEOUT_SECONDS=900
readonly SAFARIDRIVER_PROGRAM="/usr/bin/safaridriver"

browser="chrome"

if test "${1:-}" = "--safari"; then
    browser="safari"
    shift
fi

test_runner="$(print_wasm_bindgen_install_root)/bin/wasm-bindgen-test-runner"

if test ! -x "${test_runner}"; then
    echo "There is no wasm-bindgen test runner at ${test_runner}; run ./scripts/setup.sh"
    exit 1
fi

export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER="${repo_dir}/${test_runner}"
export WASM_BINDGEN_TEST_TIMEOUT="${TEST_TIMEOUT_SECONDS}"

case "${browser}" in
    chrome)
        chromedriver_program="$(print_chromedriver_directory)/chromedriver"

        if test ! -x "${chromedriver_program}"; then
            echo "There is no ChromeDriver for the installed Chrome at ${chromedriver_program}; run ./scripts/setup.sh"
            exit 1
        fi

        export CHROMEDRIVER="${repo_dir}/${chromedriver_program}"
        export WASM_BINDGEN_TEST_WEBDRIVER_JSON="${repo_dir}/tests/chrome-webdriver.json"
        ;;
    safari)
        export SAFARIDRIVER="${SAFARIDRIVER_PROGRAM}"
        ;;
esac

function run_browser_tests {
    cargo test --target wasm32-unknown-unknown --test wllama -- "$@"
}

function run_each_browser_test_in_own_page {
    local test_listing
    test_listing=$(run_browser_tests --list "$@" 2>/dev/null)
    local test_names=("${(@f)$(echo "${test_listing}" | sed -n 's/: test$//p')}")
    local failed_test_names=()

    for test_name in "${test_names[@]}"; do
        local test_status=0
        run_browser_tests --exact "${test_name}" || test_status=$?

        if test "${test_status}" -ne 0; then
            failed_test_names+=("${test_name}")
        fi
    done

    echo "${#test_names[@]} tests, each in its own page; ${#failed_test_names[@]} failed"

    if test "${#failed_test_names[@]}" -ne 0; then
        printf '  %s\n' "${failed_test_names[@]}"
        exit 1
    fi
}

case "${browser}" in
    chrome) run_browser_tests "$@" ;;
    safari) run_each_browser_test_in_own_page "$@" ;;
esac
