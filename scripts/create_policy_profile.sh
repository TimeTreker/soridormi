#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

# Profile creation writes a YAML file under configs/policies by default.
# The normal development compose mount keeps /app/configs read-only so runtime
# and simulator commands cannot accidentally mutate host configuration files.
# For this one scaffolding command, temporarily remount configs read-write.
SORIDORMI_CONFIGS_MOUNT_MODE=rw \
  docker compose -f compose.sim.yaml run --rm runtime bash -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.create_policy_profile "$@"
  ' _ "$@"
