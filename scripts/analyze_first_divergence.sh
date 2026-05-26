#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OFFICIAL_TRACE="${SORIDORMI_OFFICIAL_TRACE:-/data/official_baseline/latest_official_baseline.trace.jsonl}"
SORIDORMI_LOG="${SORIDORMI_TRACE_LOG:-}"
STEPS="${SORIDORMI_TRACE_COMPARE_STEPS:-100}"
THRESHOLD="${SORIDORMI_FIRST_DIVERGENCE_THRESHOLD:-1e-4}"
EXTRA_ANALYZER_ARGS=()

if [[ $# -gt 0 && "${1}" != --* ]]; then
  OFFICIAL_TRACE="$1"
  shift
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
  SORIDORMI_LOG="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --steps" >&2
        exit 2
      fi
      STEPS="$2"
      shift 2
      ;;
    --threshold)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --threshold" >&2
        exit 2
      fi
      THRESHOLD="$2"
      shift 2
      ;;
    *)
      EXTRA_ANALYZER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "${SORIDORMI_LOG}" ]; then
  if [ -d data/logs ]; then
    latest_host_log="$(ls -t data/logs/policy_*.mcap data/logs/runtime_*.mcap data/logs/*.jsonl 2>/dev/null | head -n 1 || true)"
    if [ -n "${latest_host_log}" ]; then
      SORIDORMI_LOG="/data/logs/$(basename "${latest_host_log}")"
    fi
  fi
fi

if [ -z "${SORIDORMI_LOG}" ]; then
  echo "Could not find a Soridormi log." >&2
  echo "Pass it as the second argument, e.g.:" >&2
  echo "  ./scripts/analyze_first_divergence.sh /data/official_baseline/latest_official_baseline.trace.jsonl /data/logs/policy_open_duck_forward_XXXX.mcap" >&2
  exit 1
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.first_divergence_analyzer \
    --official "$1" \
    --soridormi "$2" \
    --steps "$3" \
    --threshold "$4" \
    "${@:5}"
' _ "${OFFICIAL_TRACE}" "${SORIDORMI_LOG}" "${STEPS}" "${THRESHOLD}" "${EXTRA_ANALYZER_ARGS[@]}"
