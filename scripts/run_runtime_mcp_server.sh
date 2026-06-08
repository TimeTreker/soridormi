#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

exec docker compose -f compose.sim.yaml --profile mcp-runtime \
  up --build mcp-runtime
