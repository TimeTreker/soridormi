#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

translated_args=()
soridormi_translate_container_data_args translated_args "$@"

has_directory=0
for arg in "$@"; do
  if [ "${arg}" = "--directory" ]; then
    has_directory=1
    break
  fi
done
if [ "${has_directory}" = "0" ]; then
  translated_args=(--directory /data/policy_packages "${translated_args[@]}")
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.policy_package index "$@"
' _ "${translated_args[@]}"
