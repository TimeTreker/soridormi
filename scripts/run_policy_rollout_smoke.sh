#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/run_policy_rollout_smoke.sh PROFILE [options]

Run a bounded Soridormi policy rollout against an already-running simulator.
This is the M6.10 smoke harness for promoted/replacement policy profiles.

Options:
  --steps N                    Stop runtime after N control steps (default: 200).
  --seconds S                  Stop runtime after S seconds instead of/in addition to steps.
  --require-provider NAME      Require an active ONNX Runtime provider during preflight. May repeat.
  --skip-model-check           Skip check_policy_model.sh preflight.
  -h, --help                   Show this help.

The simulator server must already be running, for example:
  SORIDORMI_SIM_BACKEND=mujoco ./scripts/run_sim_server.sh
EOF
}

profile="${1:-}"
if [ -z "${profile}" ] || [ "${profile}" = "-h" ] || [ "${profile}" = "--help" ]; then
  usage
  exit 0
fi
shift || true

steps="200"
seconds=""
skip_model_check="0"
require_provider_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --steps)
      steps="${2:?--steps requires a value}"
      shift 2
      ;;
    --seconds)
      seconds="${2:?--seconds requires a value}"
      shift 2
      ;;
    --require-provider)
      require_provider_args+=(--require-provider "${2:?--require-provider requires a value}")
      shift 2
      ;;
    --skip-model-check)
      skip_model_check="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

echo "Soridormi bounded policy rollout smoke"
echo "======================================="
echo "Profile: ${profile}"
echo "Steps: ${steps:-disabled}"
echo "Seconds: ${seconds:-disabled}"

if [ "${skip_model_check}" != "1" ]; then
  ./scripts/check_policy_model.sh --profile "${profile}" "${require_provider_args[@]}"
fi

export SORIDORMI_MAX_STEPS="${steps:-0}"
export SORIDORMI_MAX_SECONDS="${seconds:-0}"
export SORIDORMI_RUNTIME_LOG="${SORIDORMI_RUNTIME_LOG:-1}"
export SORIDORMI_RUNTIME_LOG_FORMAT="${SORIDORMI_RUNTIME_LOG_FORMAT:-mcap}"

./scripts/run_policy_experiment.sh "${profile}"
