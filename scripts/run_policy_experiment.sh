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
  # User-requested smoke logging overrides must survive policy profile resolution.
  # Policy profiles are allowed to define default logging, but parity/debug tools
  # often need JSONL traces even when the profile defaults to MCAP.
  LOG_FORMAT_OVERRIDE="${SORIDORMI_RUNTIME_LOG_FORMAT_OVERRIDE:-}"
  LOG_PREFIX_OVERRIDE="${SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE:-}"
  LOG_DIR_OVERRIDE="${SORIDORMI_RUNTIME_LOG_DIR_OVERRIDE:-}"
  LOG_EVERY_N_OVERRIDE="${SORIDORMI_RUNTIME_LOG_EVERY_N_OVERRIDE:-}"

  echo "Resolving policy profile: ${PROFILE}"
  eval "$(python -m soridormi_runtime.policy_profiles "${PROFILE}" --shell)"

  if [ -n "${LOG_FORMAT_OVERRIDE}" ]; then
    export SORIDORMI_RUNTIME_LOG_FORMAT="${LOG_FORMAT_OVERRIDE}"
  fi
  if [ -n "${LOG_PREFIX_OVERRIDE}" ]; then
    export SORIDORMI_RUNTIME_LOG_PREFIX="${LOG_PREFIX_OVERRIDE}"
  fi
  if [ -n "${LOG_DIR_OVERRIDE}" ]; then
    export SORIDORMI_RUNTIME_LOG_DIR="${LOG_DIR_OVERRIDE}"
  fi
  if [ -n "${LOG_EVERY_N_OVERRIDE}" ]; then
    export SORIDORMI_RUNTIME_LOG_EVERY_N="${LOG_EVERY_N_OVERRIDE}"
  fi

  echo "Policy: ${SORIDORMI_POLICY_PROFILE}"
  echo "Model:  ${SORIDORMI_POLICY_PATH}"
  echo "Command: x=${SORIDORMI_COMMAND_X} y=${SORIDORMI_COMMAND_Y} yaw=${SORIDORMI_COMMAND_YAW}"
  echo "Phase: mode=${SORIDORMI_PHASE_MODE} period_steps=${SORIDORMI_PHASE_PERIOD_STEPS} frequency=${SORIDORMI_PHASE_FREQUENCY}"
  echo "Phase reference: ${SORIDORMI_PHASE_REFERENCE_DATA:-} require=${SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA:-0}"
  echo "Action: scale=${SORIDORMI_ACTION_SCALE} max_motor_velocity=${SORIDORMI_MAX_MOTOR_VELOCITY}"
  echo "Sync step: ${SORIDORMI_SIM_SYNC_STEP:-0}"
  echo "Sync pre-roll steps: ${SORIDORMI_SIM_PREROLL_STEPS:-0}"
  echo "Runtime log: enabled=${SORIDORMI_RUNTIME_LOG:-0} format=${SORIDORMI_RUNTIME_LOG_FORMAT:-mcap} dir=${SORIDORMI_RUNTIME_LOG_DIR:-/data/logs} prefix=${SORIDORMI_RUNTIME_LOG_PREFIX:-runtime} every_n=${SORIDORMI_RUNTIME_LOG_EVERY_N:-1}"
  echo "Postprocess: enabled=${SORIDORMI_ACTION_POSTPROCESS} leg_gain=${SORIDORMI_LEG_ACTION_GAIN} head_gain=${SORIDORMI_HEAD_ACTION_GAIN} clip_abs=${SORIDORMI_ACTION_CLIP_ABS}"
  if [ "${SORIDORMI_SKIP_POLICY_CHECK:-0}" != "1" ]; then
    python -m soridormi_runtime.check_policy_model --profile "${PROFILE}"
  fi
  python -m soridormi_runtime.main "$@"
' _ "${PROFILE}" "$@"
