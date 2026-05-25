#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

xhost +local:docker >/dev/null 2>&1 || true

docker compose -f compose.sim.yaml run --rm sim bash -lc '
  if [ ! -x /opt/venvs/sim/bin/python ]; then bootstrap_simulator; fi
  source /opt/venvs/sim/bin/activate
  python -m soridormi_sim.mujoco_server
'
