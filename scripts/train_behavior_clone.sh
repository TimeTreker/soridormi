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

docker compose -f compose.sim.yaml run --rm \
  --entrypoint bash \
  runtime -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.train_behavior_clone "$@"
  ' _ "${translated_args[@]}"
