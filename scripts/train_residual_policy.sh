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

# The residual trainer writes artifacts under /data and can optionally create a
# runtime profile under configs/policies, so allow /app/configs writes only for
# this workflow.
SORIDORMI_CONFIGS_MOUNT_MODE=rw \
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.train_residual_policy "$@"
' _ "${translated_args[@]}"
