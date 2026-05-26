#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OFFICIAL_TRACE="${1:-${SORIDORMI_OFFICIAL_TRACE:-/data/official_baseline/latest_official_baseline.trace.jsonl}}"
REPLAY_TRACE="${2:-${SORIDORMI_REPLAY_TRACE:-/data/official_baseline/latest_official_target_replay.trace.jsonl}}"
STEPS="${SORIDORMI_TRACE_COMPARE_STEPS:-100}"

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
' _ "${OFFICIAL_TRACE}" "${REPLAY_TRACE}" "${STEPS}"
