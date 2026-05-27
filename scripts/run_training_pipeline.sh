#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# This script is intentionally host-side. It orchestrates the existing host
# wrapper scripts, and those wrappers handle Docker/container path translation.
PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m soridormi_runtime.training_pipeline "$@"
