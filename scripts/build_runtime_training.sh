#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

export RUNTIME_DEV_EXTRA="${RUNTIME_DEV_EXTRA:-runtime-gpu,training}"
export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "Building Soridormi runtime image with training dependencies"
echo "  RUNTIME_DEV_EXTRA=${RUNTIME_DEV_EXTRA}"
echo "  PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}"

docker compose -f compose.sim.yaml build runtime
