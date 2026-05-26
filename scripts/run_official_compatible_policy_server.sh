#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${1:-${SORIDORMI_POLICY_PROFILE:-open_duck_forward}}"

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

# Resolve simulator-side settings from the policy profile on the host when possible.
# This avoids starting MuJoCo without the official home keyframe while the runtime
# profile expects official Open Duck compatibility.
if command -v python3 >/dev/null 2>&1; then
  eval "$(PYTHONPATH=src python3 -m soridormi_runtime.policy_profiles "${PROFILE}" --shell | grep -E 'SORIDORMI_(SIM_BACKEND|MUJOCO_USE_HOME_KEYFRAME|MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE|AUTO_RESET|MUJOCO_VIEWER)=')" || true
fi

export SORIDORMI_SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"
export SORIDORMI_MUJOCO_USE_HOME_KEYFRAME="${SORIDORMI_MUJOCO_USE_HOME_KEYFRAME:-1}"
export SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE="${SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE:-1}"
export SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE="${SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE:-1}"
export SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE="${SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE:-1}"
export SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE="${SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE:-1}"
export SORIDORMI_MUJOCO_VIEWER="${SORIDORMI_MUJOCO_VIEWER:-1}"
export SORIDORMI_AUTO_RESET="${SORIDORMI_AUTO_RESET:-1}"

echo "Soridormi official-compatible policy server"
echo "  profile=${PROFILE}"
echo "  backend=${SORIDORMI_SIM_BACKEND}"
echo "  home_keyframe=${SORIDORMI_MUJOCO_USE_HOME_KEYFRAME}"
echo "  home_overrides_reset_pose=${SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE}"
echo "  official_reset_sequence=${SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE}"
echo "  official_sensor_mode=${SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE}"
echo "  official_contact_mode=${SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE}"
echo "  auto_reset=${SORIDORMI_AUTO_RESET}"
echo "  viewer=${SORIDORMI_MUJOCO_VIEWER}"

./scripts/run_sim_server.sh
