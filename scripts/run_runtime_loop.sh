#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  if [ ! -x /opt/venvs/runtime/bin/python ]; then bootstrap_runtime; fi
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.main
'
