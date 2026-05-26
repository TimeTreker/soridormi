#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Host-side wrapper for the M4.11 parity checker.
# Usage on host:
#   ./scripts/check_observation_action_parity.sh [official_trace] [soridormi_log]
#
# The checker itself must run inside the runtime container so it can import
# soridormi_runtime and read /data-mounted logs/traces.

OFFICIAL_TRACE_HOST="${1:-${SORIDORMI_OFFICIAL_TRACE:-data/official_baseline/latest_official_baseline.trace.jsonl}}"
SORIDORMI_LOG_HOST="${2:-${SORIDORMI_TRACE_LOG:-}}"
POLICY_PATH="${SORIDORMI_POLICY_PATH:-/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx}"
STEPS="${SORIDORMI_COMPARE_STEPS:-100}"

if [[ -z "${SORIDORMI_LOG_HOST}" ]]; then
  SORIDORMI_LOG_HOST="$(ls -1t data/logs/policy_*.mcap data/logs/runtime_*.mcap data/logs/*.jsonl 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${OFFICIAL_TRACE_HOST}" ]]; then
  OFFICIAL_TRACE_HOST="$(ls -1t data/official_baseline/official_forward_trace_*.trace.jsonl 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${OFFICIAL_TRACE_HOST}" ]]; then
  echo "No official trace found. Run:" >&2
  echo "  SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh" >&2
  exit 2
fi

if [[ -z "${SORIDORMI_LOG_HOST}" ]]; then
  echo "No Soridormi log found. Run:" >&2
  echo "  ./scripts/run_policy_experiment.sh open_duck_forward" >&2
  exit 2
fi

# Convert common host paths into container paths. /data/... paths are already OK.
to_container_path() {
  local path="$1"
  if [[ "${path}" == /data/* ]]; then
    echo "${path}"
  elif [[ "${path}" == data/official_baseline/* ]]; then
    echo "/data/official_baseline/$(basename "${path}")"
  elif [[ "${path}" == data/logs/* ]]; then
    echo "/data/logs/$(basename "${path}")"
  else
    # Best-effort fallback for manually supplied files under data/.
    case "${path}" in
      *official_baseline*) echo "/data/official_baseline/$(basename "${path}")" ;;
      *logs*) echo "/data/logs/$(basename "${path}")" ;;
      *) echo "${path}" ;;
    esac
  fi
}

OFFICIAL_TRACE_CONTAINER="$(to_container_path "${OFFICIAL_TRACE_HOST}")"
SORIDORMI_LOG_CONTAINER="$(to_container_path "${SORIDORMI_LOG_HOST}")"

echo "Official trace:  ${OFFICIAL_TRACE_HOST} -> ${OFFICIAL_TRACE_CONTAINER}"
echo "Soridormi log:   ${SORIDORMI_LOG_HOST} -> ${SORIDORMI_LOG_CONTAINER}"
echo "Policy path:     ${POLICY_PATH}"

if [[ ! -f .env ]]; then
  ./scripts/setup_env.sh
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.compare_observation_action_parity \
    --official "$1" \
    --soridormi "$2" \
    --policy "$3" \
    --steps "$4"
' _ "${OFFICIAL_TRACE_CONTAINER}" "${SORIDORMI_LOG_CONTAINER}" "${POLICY_PATH}" "${STEPS}"
