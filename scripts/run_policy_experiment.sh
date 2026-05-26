#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${1:-${SORIDORMI_POLICY_PROFILE:-open_duck_forward}}"
shift || true

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

echo "Soridormi policy experiment"
echo "  profile=${PROFILE}"

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  PROFILE="$1"
  shift || true
  echo "Resolving policy profile: ${PROFILE}"
  eval "$(python -m soridormi_runtime.policy_profiles "${PROFILE}" --shell)"
  echo "Policy: ${SORIDORMI_POLICY_PROFILE}"
  echo "Model:  ${SORIDORMI_POLICY_PATH}"
  echo "Command: x=${SORIDORMI_COMMAND_X} y=${SORIDORMI_COMMAND_Y} yaw=${SORIDORMI_COMMAND_YAW}"
  echo "Phase: mode=${SORIDORMI_PHASE_MODE} period_steps=${SORIDORMI_PHASE_PERIOD_STEPS} frequency=${SORIDORMI_PHASE_FREQUENCY}"
  echo "Phase reference: ${SORIDORMI_PHASE_REFERENCE_DATA:-} require=${SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA:-0}"
  echo "Action: scale=${SORIDORMI_ACTION_SCALE} max_motor_velocity=${SORIDORMI_MAX_MOTOR_VELOCITY}"
  echo "Sync step: ${SORIDORMI_SIM_SYNC_STEP:-0}"
  echo "Postprocess: enabled=${SORIDORMI_ACTION_POSTPROCESS} leg_gain=${SORIDORMI_LEG_ACTION_GAIN} head_gain=${SORIDORMI_HEAD_ACTION_GAIN} clip_abs=${SORIDORMI_ACTION_CLIP_ABS}"
  if [ "${SORIDORMI_SKIP_POLICY_CHECK:-0}" != "1" ]; then
    python -m soridormi_runtime.check_policy_model --profile "${PROFILE}"
  fi
  python -m soridormi_runtime.main "$@"
' _ "${PROFILE}" "$@"
