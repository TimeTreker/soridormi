#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SORIDORMI_SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"
export SORIDORMI_MUJOCO_VIEWER="${SORIDORMI_MUJOCO_VIEWER:-1}"
export SORIDORMI_MUJOCO_FIXED_BASE="${SORIDORMI_MUJOCO_FIXED_BASE:-1}"

echo "Starting MuJoCo sim server for fixed-base standing debug:"
echo "  SORIDORMI_SIM_BACKEND=${SORIDORMI_SIM_BACKEND}"
echo "  SORIDORMI_MUJOCO_VIEWER=${SORIDORMI_MUJOCO_VIEWER}"
echo "  SORIDORMI_MUJOCO_FIXED_BASE=${SORIDORMI_MUJOCO_FIXED_BASE}"

exec ./scripts/run_sim_server.sh
