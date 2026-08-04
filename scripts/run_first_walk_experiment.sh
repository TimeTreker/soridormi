#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible first-walk integration entrypoint. ONNX forward-policy compatibility routes it to the ONNX forward
# policy experiment, not to any open-loop gait controller.
cd "$(dirname "$0")/.."
exec ./scripts/run_forward_policy_experiment.sh
