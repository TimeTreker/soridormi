#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# This still uses Soridormi's official-baseline wrapper so we avoid a heavy JAX
# dependency, but it starts with zero command and keeps the official keyboard
# callback. In the MuJoCo viewer: arrow up = forward, arrow down = backward,
# left/right arrows = lateral, a/e = yaw.
SORIDORMI_OFFICIAL_COMMAND_X="${SORIDORMI_OFFICIAL_COMMAND_X:-0.0}" \
SORIDORMI_OFFICIAL_COMMAND_Y="${SORIDORMI_OFFICIAL_COMMAND_Y:-0.0}" \
SORIDORMI_OFFICIAL_COMMAND_YAW="${SORIDORMI_OFFICIAL_COMMAND_YAW:-0.0}" \
SORIDORMI_OFFICIAL_VIEWER="${SORIDORMI_OFFICIAL_VIEWER:-1}" \
SORIDORMI_OFFICIAL_MAX_SECONDS="${SORIDORMI_OFFICIAL_MAX_SECONDS:-120}" \
./scripts/run_official_forward_baseline.sh
