#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

POLICY_PATH_VALUE="${SORIDORMI_POLICY_PATH:-/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx}"
LOG_VALUE="${SORIDORMI_RUNTIME_LOG:-1}"
LOG_FORMAT_VALUE="${SORIDORMI_RUNTIME_LOG_FORMAT:-mcap}"
CONTROL_HZ_VALUE="${CONTROL_HZ:-50}"

# Use explicit -e forwarding so policy path overrides work even if compose.sim.yaml
# does not list every experimental policy runtime foundation variable yet.
docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_RUNTIME_MODE=onnx_policy \
  -e SORIDORMI_POLICY_PATH="${POLICY_PATH_VALUE}" \
  -e SORIDORMI_RUNTIME_LOG="${LOG_VALUE}" \
  -e SORIDORMI_RUNTIME_LOG_FORMAT="${LOG_FORMAT_VALUE}" \
  -e CONTROL_HZ="${CONTROL_HZ_VALUE}" \
  runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.main
  '
