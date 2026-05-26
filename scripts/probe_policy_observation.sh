#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

POLICY_PATH="${SORIDORMI_POLICY_PATH:-/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx}"
ROBOT_CONFIG="${SORIDORMI_ROBOT_CONFIG:-/app/configs/robots/open_duck_mini_v2.yaml}"

docker compose -f compose.sim.yaml run --rm       -e SORIDORMI_POLICY_PATH="${POLICY_PATH}"       -e SORIDORMI_ROBOT_CONFIG="${ROBOT_CONFIG}"       -e SORIDORMI_USE_CUDA_PROVIDER="${SORIDORMI_USE_CUDA_PROVIDER:-1}"       runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.probe_onnx_observation           --policy "$SORIDORMI_POLICY_PATH"           --config "$SORIDORMI_ROBOT_CONFIG"
  '
