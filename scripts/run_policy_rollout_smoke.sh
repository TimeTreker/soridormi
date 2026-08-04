#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/run_policy_rollout_smoke.sh PROFILE [options]

Run a bounded Soridormi policy rollout against an already-running simulator.
This is the bounded rollout smoke harness for promoted/replacement policy profiles.

Options:
  --steps N                    Stop runtime after N control steps (default: 200).
  --seconds S                  Stop runtime after S seconds instead of/in addition to steps.
  --log-format FORMAT          Runtime log format: mcap or jsonl. Overrides the policy profile.
  --log-prefix PREFIX          Runtime log filename prefix. Overrides the policy profile.
  --log-dir DIR                Runtime log directory inside the container (default: /data/logs).
  --log-every-n N              Log every N runtime steps. Overrides the policy profile.
  --no-log                     Disable runtime logging for this smoke rollout.
  --require-provider NAME      Require an active ONNX Runtime provider during preflight. May repeat.
  --skip-model-check           Skip check_policy_model.sh preflight.
  -h, --help                   Show this help.

The simulator server must already be running with matching MuJoCo compatibility
flags, for example:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
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
runtime_log="1"
log_format_override="${SORIDORMI_RUNTIME_LOG_FORMAT+x}"
log_format="${SORIDORMI_RUNTIME_LOG_FORMAT:-}"
log_prefix_override="${SORIDORMI_RUNTIME_LOG_PREFIX+x}"
log_prefix="${SORIDORMI_RUNTIME_LOG_PREFIX:-}"
log_dir_override="${SORIDORMI_RUNTIME_LOG_DIR+x}"
log_dir="${SORIDORMI_RUNTIME_LOG_DIR:-}"
log_every_n_override="${SORIDORMI_RUNTIME_LOG_EVERY_N+x}"
log_every_n="${SORIDORMI_RUNTIME_LOG_EVERY_N:-}"

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
    --log-format)
      log_format="${2:?--log-format requires a value}"
      log_format_override="1"
      shift 2
      ;;
    --log-format=*)
      log_format="${1#*=}"
      log_format_override="1"
      shift
      ;;
    --log-prefix)
      log_prefix="${2:?--log-prefix requires a value}"
      log_prefix_override="1"
      shift 2
      ;;
    --log-prefix=*)
      log_prefix="${1#*=}"
      log_prefix_override="1"
      shift
      ;;
    --log-dir)
      log_dir="${2:?--log-dir requires a value}"
      log_dir_override="1"
      shift 2
      ;;
    --log-dir=*)
      log_dir="${1#*=}"
      log_dir_override="1"
      shift
      ;;
    --log-every-n)
      log_every_n="${2:?--log-every-n requires a value}"
      log_every_n_override="1"
      shift 2
      ;;
    --log-every-n=*)
      log_every_n="${1#*=}"
      log_every_n_override="1"
      shift
      ;;
    --no-log)
      runtime_log="0"
      shift
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
if [ -n "${log_format}" ]; then
  case "${log_format}" in
    mcap|jsonl) ;;
    *)
      echo "Unsupported --log-format: ${log_format}. Use 'mcap' or 'jsonl'." >&2
      exit 2
      ;;
  esac
fi

echo "Profile: ${profile}"
echo "Steps: ${steps:-disabled}"
echo "Seconds: ${seconds:-disabled}"
echo "Runtime log: ${runtime_log}"
if [ -n "${log_format}" ]; then
  echo "Runtime log format override: ${log_format}"
fi
if [ -n "${log_prefix}" ]; then
  echo "Runtime log prefix override: ${log_prefix}"
fi
if [ -n "${log_dir}" ]; then
  echo "Runtime log dir override: ${log_dir}"
fi
if [ -n "${log_every_n}" ]; then
  echo "Runtime log every N override: ${log_every_n}"
fi

if [ "${skip_model_check}" != "1" ]; then
  ./scripts/check_policy_model.sh --profile "${profile}" "${require_provider_args[@]}"
fi

export SORIDORMI_MAX_STEPS="${steps:-0}"
export SORIDORMI_MAX_SECONDS="${seconds:-0}"
export SORIDORMI_RUNTIME_LOG="${runtime_log}"

# Policy profiles can set their own logging defaults. These override variables
# tell run_policy_experiment.sh to re-apply user-requested smoke logging after
# resolving the profile inside the runtime container. This is required for JSONL
# parity traces because open_duck_forward defaults to MCAP logging.
if [ -n "${log_format_override}" ]; then
  export SORIDORMI_RUNTIME_LOG_FORMAT="${log_format}"
  export SORIDORMI_RUNTIME_LOG_FORMAT_OVERRIDE="${log_format}"
fi
if [ -n "${log_prefix_override}" ]; then
  export SORIDORMI_RUNTIME_LOG_PREFIX="${log_prefix}"
  export SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE="${log_prefix}"
fi
if [ -n "${log_dir_override}" ]; then
  export SORIDORMI_RUNTIME_LOG_DIR="${log_dir}"
  export SORIDORMI_RUNTIME_LOG_DIR_OVERRIDE="${log_dir}"
fi
if [ -n "${log_every_n_override}" ]; then
  export SORIDORMI_RUNTIME_LOG_EVERY_N="${log_every_n}"
  export SORIDORMI_RUNTIME_LOG_EVERY_N_OVERRIDE="${log_every_n}"
fi

./scripts/run_policy_experiment.sh "${profile}"
