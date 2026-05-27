#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

records_args=()
has_records_dir=0
for arg in "$@"; do
  if [ "${arg}" = "--records-dir" ]; then
    has_records_dir=1
    break
  fi
done
if [ "${has_records_dir}" = "0" ]; then
  records_args=(--records-dir /data/policy_promotions)
fi

translated_user_args=()
soridormi_translate_container_data_args translated_user_args "$@"

# Promotion writes a YAML file under configs/policies by default, so temporarily
# remount /app/configs read-write for this explicit promotion command only.
SORIDORMI_CONFIGS_MOUNT_MODE=rw \
  docker compose -f compose.sim.yaml run --rm runtime bash -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.promote_policy_candidate "$@"
  ' _ "${records_args[@]}" "${translated_user_args[@]}"
