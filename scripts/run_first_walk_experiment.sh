#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible M4.0 entrypoint. M4.1 routes it to the ONNX forward
# policy experiment, not to any open-loop gait controller.
cd "$(dirname "$0")/.."
exec ./scripts/run_forward_policy_experiment.sh
