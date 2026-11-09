#!/usr/bin/env bash
# Runs the terminal client, which reads prompts from standard input and streams each reply to standard output.
#
# Usage:
#   ./scripts/run-terminal.sh [<model path>]

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${package_dir}"

exec cargo run --quiet --release --features terminal --bin terminal -- "$@"
