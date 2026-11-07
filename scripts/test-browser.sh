#!/bin/zsh
# Runs the browser backend's tests (tests/wllama.rs and the unit tests under src/wllama/) in a real browser, through
# wasm-bindgen-test's runner, which serves
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
# Keeps the runner's requests to the WebDriver on this machine away from an HTTP proxy set in the environment.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

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

readonly CARGO_TEST_TARGET_SELECTIONS=(--lib --test=wllama)

function run_browser_tests {
    cargo test --target wasm32-unknown-unknown "${CARGO_TEST_TARGET_SELECTIONS[@]}" -- "$@"
}

function run_target_browser_tests {
    local cargo_test_target_selection="$1"
    shift
    cargo test --target wasm32-unknown-unknown "${cargo_test_target_selection}" -- "$@"
}

# Drops the test-name filter, which the listing has already applied.
function print_options_without_filter {
    while test "$#" -gt 0; do
        case "$1" in
            --skip|--format)
                printf '%s\n' "$1" "$2"
                shift 2
                ;;
            -*)
                printf '%s\n' "$1"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
}

function run_each_browser_test_in_own_page {
    local test_options=("${(@f)$(print_options_without_filter "$@")}")
    test_options=("${(@)test_options:#}")
    local test_count=0
    local failed_test_names=()
    local test_listing

    for cargo_test_target_selection in "${CARGO_TEST_TARGET_SELECTIONS[@]}"; do
        test_listing=$(run_target_browser_tests "${cargo_test_target_selection}" --list "$@")
        local test_names=("${(@f)$(echo "${test_listing}" | sed -n 's/: test$//p')}")
        test_names=("${(@)test_names:#}")

        for test_name in "${test_names[@]}"; do
            local test_status=0
            run_target_browser_tests "${cargo_test_target_selection}" --exact "${test_name}" "${test_options[@]}" || test_status=$?
            test_count=$((test_count + 1))

            if test "${test_status}" -ne 0; then
                failed_test_names+=("${test_name}")
            fi
        done
    done

    if test "${test_count}" -eq 0; then
        echo "No tests match the arguments: $*"
        exit 1
    fi

    echo "${test_count} tests, each in its own page; ${#failed_test_names[@]} failed"

    if test "${#failed_test_names[@]}" -ne 0; then
        printf '  %s\n' "${failed_test_names[@]}"
        exit 1
    fi
}

case "${browser}" in
    chrome) run_browser_tests "$@" ;;
    safari) run_each_browser_test_in_own_page "$@" ;;
esac
