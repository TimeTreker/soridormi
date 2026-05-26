#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SORIDORMI_RUNTIME_MODE="${SORIDORMI_RUNTIME_MODE:-stand}"
export SORIDORMI_STAND_RAMP_SECONDS="${SORIDORMI_STAND_RAMP_SECONDS:-5.0}"

echo "Starting Soridormi runtime in standing mode:"
echo "  SORIDORMI_RUNTIME_MODE=${SORIDORMI_RUNTIME_MODE}"
echo "  SORIDORMI_STAND_RAMP_SECONDS=${SORIDORMI_STAND_RAMP_SECONDS}"

exec ./scripts/run_runtime_loop.sh
