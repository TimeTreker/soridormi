#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

latest="$(ls -t data/logs/runtime_*.mcap data/logs/runtime_*.jsonl 2>/dev/null | head -n 1 || true)"

if [ -z "$latest" ]; then
  echo "No Soridormi runtime logs found in data/logs."
  exit 1
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc "
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.inspect_log /data/logs/$(basename "$latest")
"
