#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

POLICY_PATH="${SORIDORMI_POLICY_PATH:-/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx}"

# This script intentionally runs in the runtime-dev container, not the simulator.
# It only inspects ONNX Runtime providers and performs one dummy inference.
docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_POLICY_PATH="${POLICY_PATH}" \
  -e SORIDORMI_USE_CUDA_PROVIDER="${SORIDORMI_USE_CUDA_PROVIDER:-1}" \
  runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.inspect_onnx_policy "$SORIDORMI_POLICY_PATH"
  '
