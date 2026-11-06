# Locates the browser tests' tools, for scripts/setup.sh and scripts/test-browser.sh to source from the repository root.

readonly BROWSER_TOOLS_DIRECTORY="build/tools"
readonly MAC_CHROME_PROGRAM="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The test runner must be the same version as the wasm-bindgen crate.
function print_wasm_bindgen_version {
    awk '/^name = "wasm-bindgen"$/ { getline; split($0, parts, "\""); print parts[2] }' Cargo.lock
}

function print_wasm_bindgen_install_root {
    echo "${BROWSER_TOOLS_DIRECTORY}/wasm-bindgen-$(print_wasm_bindgen_version)"
}

# Prints nothing when Google Chrome is not installed.
function print_chrome_build {
    local chrome_program="${MAC_CHROME_PROGRAM}"

    if test ! -x "${chrome_program}"; then
        chrome_program=$(command -v google-chrome || true)
    fi

    if test -z "${chrome_program}"; then
        return 0
    fi

    local chrome_version
    chrome_version=$("${chrome_program}" --version)
    # "Google Chrome 156.0.8078.81" has the build "156.0.8078", which a ChromeDriver must match.
    echo "${chrome_version}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
}

function print_chromedriver_directory {
    echo "${BROWSER_TOOLS_DIRECTORY}/chromedriver-$(print_chrome_build)"
}
