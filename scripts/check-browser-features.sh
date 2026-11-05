#!/usr/bin/env bash

# Reports, for each major browser's current release, whether it supports the two WebAssembly features wllama's
# default build needs: 64-bit memory (memory64) and JavaScript Promise Integration (jspi). A browser that lacks
# either runs wllama's compatibility build. The source is MDN's browser compatibility data.
#
# Usage:
#   ./scripts/check-browser-features.sh

set -euo pipefail

readonly COMPATIBILITY_DATA_URL="https://unpkg.com/@mdn/browser-compat-data/data.json"
readonly COMPATIBILITY_DATA_PATH="/tmp/cross-llama-browser-compat-data.json"

function fail {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

installed_jq="$(command -v jq || true)"

if test -z "${installed_jq}"; then
    fail "jq is not installed"
fi

download_status=0
curl --silent --show-error --fail --location --max-time 120 --output "${COMPATIBILITY_DATA_PATH}" "${COMPATIBILITY_DATA_URL}" || download_status=$?

if test "${download_status}" -ne 0; then
    fail "could not download ${COMPATIBILITY_DATA_URL} (curl exit ${download_status})"
fi

feature_report="$(jq --raw-output '
    def version_parts: split(".") | map(tonumber? // 0);

    def first_statement: if type == "array" then .[0] else . end;

    def supported_in($current_release):
        (.version_added) as $added
        | if $added == true then "yes"
          elif ($added | type) != "string" then "no"
          elif ($added | test("^[0-9.]+$") | not) then "no (" + $added + ")"
          elif .flags != null then "behind a flag"
          elif ($added | version_parts) <= ($current_release | version_parts) then "yes (since " + $added + ")"
          else "no (from " + $added + ")"
          end;

    . as $data
    | ["chrome", "edge", "firefox", "safari", "chrome_android", "firefox_android", "safari_ios"][] as $browser
    | ($data.browsers[$browser].releases | to_entries | map(select(.value.status == "current")) | .[0].key) as $current_release
    | ($data.webassembly.memory64.__compat.support[$browser] | first_statement | supported_in($current_release)) as $memory64
    | ($data.webassembly.jspi.__compat.support[$browser] | first_statement | supported_in($current_release)) as $jspi
    | [
        $browser,
        $current_release,
        $memory64,
        $jspi,
        (if ($memory64 | startswith("yes")) and ($jspi | startswith("yes")) then "default" else "compatibility" end)
      ]
    | @tsv
' "${COMPATIBILITY_DATA_PATH}")"

printf 'BROWSER\tCURRENT RELEASE\tMEMORY64\tJSPI\tWLLAMA BUILD\n%s\n' "${feature_report}" | column -t -s $'\t'

compatibility_data_date="$(jq --raw-output '.__meta.timestamp' "${COMPATIBILITY_DATA_PATH}")"

printf '\nMDN browser compatibility data as of %s\n' "${compatibility_data_date}"
