#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

xhost +local:docker >/dev/null 2>&1 || true

docker compose -f compose.sim.yaml run --rm sim
