#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:-/data/open_duck_mini_v2.policy_compat_generated.yaml}

docker compose -f compose.sim.yaml run --rm sim \
  python -m soridormi_sim.export_policy_compat_config \
    --output "${OUTPUT}"

echo "Generated inside container at: ${OUTPUT}"
echo "If OUTPUT is under /data, host path is: ./data/$(basename "${OUTPUT}")"
