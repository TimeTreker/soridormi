#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

if [ "$#" -eq 0 ]; then
  docker compose -f compose.sim.yaml run --rm runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    shopt -s nullglob
    logs=(/data/logs/*.mcap /data/logs/*.jsonl)
    if [ "${#logs[@]}" -eq 0 ]; then
      echo "No logs found in /data/logs" >&2
      exit 1
    fi
    python -m soridormi_runtime.compare_policy_logs "${logs[@]}"
  '
else
  docker compose -f compose.sim.yaml run --rm runtime bash -lc '
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.compare_policy_logs "$@"
  ' bash "$@"
fi
