#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lib/latest_policy_log.sh

OFFICIAL_TRACE="${1:-${SORIDORMI_OFFICIAL_TRACE:-/data/official_baseline/latest_official_baseline.trace.jsonl}}"
SORIDORMI_LOG="${2:-${SORIDORMI_TRACE_LOG:-}}"
STEPS="${SORIDORMI_TRACE_COMPARE_STEPS:-100}"

if [ -z "${SORIDORMI_LOG}" ]; then
  if [ -d data/logs ]; then
    latest_host_log="$(find_latest_policy_log data/logs)"
    if [ -n "${latest_host_log}" ]; then
      SORIDORMI_LOG="/data/logs/$(basename "${latest_host_log}")"
    fi
  fi
fi

if [ -z "${SORIDORMI_LOG}" ]; then
  echo "Could not find a Soridormi log. Pass it as the second argument, e.g.:"
  echo "  ./scripts/compare_official_soridormi_trace.sh /data/official_baseline/latest_official_baseline.trace.jsonl /data/logs/policy_open_duck_forward_XXXX.mcap"
  exit 1
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.compare_official_soridormi_trace \
    --official "$1" \
    --soridormi "$2" \
    --steps "$3"
' _ "${OFFICIAL_TRACE}" "${SORIDORMI_LOG}" "${STEPS}"
