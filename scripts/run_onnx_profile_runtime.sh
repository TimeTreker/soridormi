#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${1:-${SORIDORMI_POLICY_PROFILE:-crawl_safe}}"

case "${PROFILE}" in
  idle_debug)
    export SORIDORMI_COMMAND_X="0.0"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.0"
    export SORIDORMI_PHASE_FREQUENCY="0.0"
    export SORIDORMI_ACTION_SCALE="0.05"
    export SORIDORMI_MAX_MOTOR_VELOCITY="2.0"
    ;;
  crawl_very_safe)
    export SORIDORMI_COMMAND_X="0.005"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.0"
    export SORIDORMI_PHASE_FREQUENCY="0.8"
    export SORIDORMI_ACTION_SCALE="0.05"
    export SORIDORMI_MAX_MOTOR_VELOCITY="2.0"
    ;;
  crawl_safe)
    export SORIDORMI_COMMAND_X="0.01"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.0"
    export SORIDORMI_PHASE_FREQUENCY="1.0"
    export SORIDORMI_ACTION_SCALE="0.10"
    export SORIDORMI_MAX_MOTOR_VELOCITY="3.0"
    ;;
  walk_cautious)
    export SORIDORMI_COMMAND_X="0.02"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.0"
    export SORIDORMI_PHASE_FREQUENCY="1.0"
    export SORIDORMI_ACTION_SCALE="0.12"
    export SORIDORMI_MAX_MOTOR_VELOCITY="4.0"
    ;;
  walk_default_soft)
    export SORIDORMI_COMMAND_X="0.03"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.0"
    export SORIDORMI_PHASE_FREQUENCY="1.0"
    export SORIDORMI_ACTION_SCALE="0.20"
    export SORIDORMI_MAX_MOTOR_VELOCITY="5.24"
    ;;
  turn_cautious)
    export SORIDORMI_COMMAND_X="0.0"
    export SORIDORMI_COMMAND_Y="0.0"
    export SORIDORMI_COMMAND_YAW="0.05"
    export SORIDORMI_PHASE_FREQUENCY="1.0"
    export SORIDORMI_ACTION_SCALE="0.10"
    export SORIDORMI_MAX_MOTOR_VELOCITY="3.0"
    ;;
  *)
    echo "Unknown policy profile: ${PROFILE}" >&2
    echo "Available: idle_debug crawl_very_safe crawl_safe walk_cautious walk_default_soft turn_cautious" >&2
    exit 2
    ;;
esac

export SORIDORMI_POLICY_PROFILE="${PROFILE}"
export SORIDORMI_RUNTIME_MODE="onnx_policy"
export SORIDORMI_RUNTIME_LOG="1"
export SORIDORMI_RUNTIME_LOG_FORMAT="${SORIDORMI_RUNTIME_LOG_FORMAT:-mcap}"
export SORIDORMI_RUNTIME_LOG_EVERY_N="${SORIDORMI_RUNTIME_LOG_EVERY_N:-1}"
export SORIDORMI_RUNTIME_LOG_PREFIX="runtime_${PROFILE}"

echo "Soridormi ONNX policy profile: ${PROFILE}"
echo "  command: x=${SORIDORMI_COMMAND_X} y=${SORIDORMI_COMMAND_Y} yaw=${SORIDORMI_COMMAND_YAW}"
echo "  phase_frequency=${SORIDORMI_PHASE_FREQUENCY}"
echo "  action_scale=${SORIDORMI_ACTION_SCALE}"
echo "  max_motor_velocity=${SORIDORMI_MAX_MOTOR_VELOCITY}"
echo "  log_prefix=${SORIDORMI_RUNTIME_LOG_PREFIX}"

./scripts/run_runtime_loop.sh
