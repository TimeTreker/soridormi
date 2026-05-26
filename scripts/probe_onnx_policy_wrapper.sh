#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

POLICY_PATH="${SORIDORMI_POLICY_PATH:-/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx}"

docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_POLICY_PATH="${POLICY_PATH}" \
  runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.probe_onnx_policy_wrapper
  '
