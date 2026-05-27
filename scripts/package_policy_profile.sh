#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

output_args=()
has_output_dir=0
for arg in "$@"; do
  if [ "${arg}" = "--output-dir" ]; then
    has_output_dir=1
    break
  fi
done
if [ "${has_output_dir}" = "0" ]; then
  output_args=(--output-dir /data/policy_packages)
fi

translated_user_args=()
soridormi_translate_container_data_args translated_user_args "$@"

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.policy_package package "$@"
' _ "${output_args[@]}" "${translated_user_args[@]}"
