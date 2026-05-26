#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
