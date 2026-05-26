#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.probe_action_mapper
'
