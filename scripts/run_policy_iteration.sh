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

# Policy iteration writes training artifacts under /data and may write promoted
# profile YAML under configs/policies, so use a temporary writable configs mount.
SORIDORMI_CONFIGS_MOUNT_MODE=rw \
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.policy_iteration "$@"
' _ "${translated_args[@]}"
