#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.policy_tuning_profiles "$@"
' bash "$@"
