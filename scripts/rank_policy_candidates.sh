#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Host-side and container-safe: this module only needs the standard library.
PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m soridormi_runtime.policy_candidate_leaderboard "$@"
