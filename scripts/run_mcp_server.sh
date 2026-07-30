#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${SORIDORMI_SOURCE_REVISION:-}" ]; then
  SORIDORMI_SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
  export SORIDORMI_SOURCE_REVISION
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

exec docker compose -f compose.mcp.yaml up --build mcp
