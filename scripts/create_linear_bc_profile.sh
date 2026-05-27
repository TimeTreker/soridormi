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

# Profile creation writes YAML under configs/policies by default.
SORIDORMI_CONFIGS_MOUNT_MODE=rw \
  docker compose -f compose.sim.yaml run --rm runtime bash -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.create_linear_bc_profile "$@"
  ' _ "${translated_args[@]}"
