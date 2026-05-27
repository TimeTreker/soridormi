#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

output_args=()
has_output_dir=0
for arg in "$@"; do
  if [ "${arg}" = "--output-dir" ]; then
    has_output_dir=1
    break
  fi
done
if [ "${has_output_dir}" = "0" ]; then
  output_args=(--output-dir /data/policy_acceptance)
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.policy_acceptance "$@"
' _ "${output_args[@]}" "$@"
