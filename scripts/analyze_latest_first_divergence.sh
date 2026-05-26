#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

official_host="${SORIDORMI_OFFICIAL_TRACE:-}"
if [[ -z "${official_host}" ]]; then
  if [[ -f data/official_baseline/latest_official_baseline.trace.jsonl ]]; then
    official_host="data/official_baseline/latest_official_baseline.trace.jsonl"
  else
    official_host="$(ls -1t data/official_baseline/official_forward_trace_*.trace.jsonl 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${official_host}" ]]; then
  echo "No official trace found. Run:" >&2
  echo "  SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh" >&2
  exit 2
fi

soridormi_host="${SORIDORMI_TRACE_LOG:-}"
if [[ -z "${soridormi_host}" ]]; then
  soridormi_host="$(ls -1t data/logs/policy_*.mcap data/logs/runtime_*.mcap data/logs/*.jsonl 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${soridormi_host}" ]]; then
  echo "No Soridormi policy log found. Run:" >&2
  echo "  ./scripts/run_policy_experiment.sh open_duck_forward" >&2
  exit 2
fi

to_container_data_path() {
  local path="$1"
  case "${path}" in
    /data/*)
      printf '%s\n' "${path}"
      ;;
    data/*)
      printf '/data/%s\n' "${path#data/}"
      ;;
    "${PWD}"/data/*)
      printf '/data/%s\n' "${path#"${PWD}"/data/}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

official_container="$(to_container_data_path "${official_host}")"
soridormi_container="$(to_container_data_path "${soridormi_host}")"

echo "Official trace: ${official_host}"
echo "Soridormi log: ${soridormi_host}"
echo "Container official trace: ${official_container}"
echo "Container Soridormi log: ${soridormi_container}"

./scripts/analyze_first_divergence.sh "${official_container}" "${soridormi_container}" "$@"
