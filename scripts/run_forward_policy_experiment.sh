#!/usr/bin/env bash
set -euo pipefail

# M4.2 profile-based entrypoint for the ONNX forward policy experiment.
# Start the MuJoCo server first, then run this script in a second terminal.
cd "$(dirname "$0")/.."
exec ./scripts/run_policy_experiment.sh "${SORIDORMI_POLICY_PROFILE:-open_duck_forward}" "$@"
