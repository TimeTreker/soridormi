#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lib/latest_policy_log.sh

# Host-side latest-file wrapper. It discovers the newest official trace and
# policy MCAP on the host, then delegates to check_observation_action_parity.sh,
# which runs the actual Python checker inside the runtime container.

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
  soridormi_host="$(find_latest_policy_log data/logs)"
fi

if [[ -z "${soridormi_host}" ]]; then
  echo "No Soridormi policy log found. Run:" >&2
  echo "  ./scripts/run_policy_experiment.sh open_duck_forward" >&2
  exit 2
fi

echo "Official trace:  ${official_host}"
echo "Soridormi log:   ${soridormi_host}"

./scripts/check_observation_action_parity.sh "${official_host}" "${soridormi_host}"
