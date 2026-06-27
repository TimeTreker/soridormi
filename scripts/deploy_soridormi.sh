#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_IMAGES=1
RUN_VALIDATION=1
START_AFTER=0
PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-open_duck_forward}"
VIEWER=1
FOLLOW_CAMERA=1
KEEP_RUNNING=0

usage() {
  cat <<'USAGE'
Usage: ./scripts/deploy_soridormi.sh [options]

Prepare a fresh Soridormi checkout for local MuJoCo simulator deployment.
This script prepares the body runtime; start Chromie separately after Soridormi
is running.

Options:
  --build              Build simulator/runtime/MCP images (default)
  --skip-build         Prepare env/submodules/validation only; do not build
  --skip-validation    Skip dry validation gates
  --start              Start Soridormi after deployment
  --profile NAME       Policy profile for --start; default: open_duck_forward
  --viewer             With --start, open MuJoCo viewer; default
  --no-viewer          With --start, run headless
  --follow-camera      With --start, follow the robot; default
  --no-follow-camera   With --start, disable follow camera
  --keep-running       With --start, leave services running after exit
  -h, --help           Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build) BUILD_IMAGES=1; shift ;;
    --skip-build) BUILD_IMAGES=0; shift ;;
    --skip-validation) RUN_VALIDATION=0; shift ;;
    --start) START_AFTER=1; shift ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --viewer) VIEWER=1; shift ;;
    --no-viewer) VIEWER=0; shift ;;
    --follow-camera) FOLLOW_CAMERA=1; shift ;;
    --no-follow-camera) FOLLOW_CAMERA=0; shift ;;
    --keep-running) KEEP_RUNNING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[deploy-soridormi][error] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for cmd in docker git python3; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[deploy-soridormi][error] Required command not found: $cmd" >&2
    exit 1
  }
done

docker info >/dev/null 2>&1 || {
  echo "[deploy-soridormi][error] Docker daemon is not reachable." >&2
  exit 1
}

for path in \
  compose.sim.yaml \
  scripts/add_submodules.sh \
  scripts/build_sim.sh \
  scripts/setup_env.sh \
  scripts/start_soridormi_mujoco.sh \
  scripts/validate_pre_wbc_scenario_surface.sh \
  scripts/validate_task_agent_contract.sh; do
  [ -e "$path" ] || {
    echo "[deploy-soridormi][error] Missing repository file: $path" >&2
    exit 1
  }
done

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
else
  echo "[deploy-soridormi] Reusing existing .env."
fi

if [ ! -d workspace/Open_Duck_Mini ] || \
   [ ! -d workspace/Open_Duck_Mini_Runtime ] || \
   [ ! -d workspace/Open_Duck_Playground ]; then
  echo "[deploy-soridormi] Initializing upstream Open Duck workspaces..."
  ./scripts/add_submodules.sh
else
  echo "[deploy-soridormi] Upstream workspaces are present."
fi

COMPOSE_ARGS=(-f compose.sim.yaml --profile mcp-runtime)

if [ "$BUILD_IMAGES" = "1" ]; then
  echo "[deploy-soridormi] Building runtime and simulator images..."
  ./scripts/build_sim.sh
  echo "[deploy-soridormi] Building runtime MCP image..."
  docker compose "${COMPOSE_ARGS[@]}" build mcp-runtime
else
  echo "[deploy-soridormi] Skipping image build."
fi

if [ "$RUN_VALIDATION" = "1" ]; then
  echo "[deploy-soridormi] Running dry/offline validation gates..."
  ./scripts/validate_pre_wbc_scenario_surface.sh
  ./scripts/validate_task_agent_contract.sh
else
  echo "[deploy-soridormi] Skipping validation."
fi

cat <<EOF_DONE

======================================================================
Soridormi deployment preparation complete
======================================================================
Environment: .env
MCP start command:
  ./scripts/start_soridormi_mujoco.sh --profile ${PROFILE} --viewer --follow-camera

Then start Chromie from the Chromie checkout:
  ./scripts/start_chromie.sh --mcp-url http://127.0.0.1:8000/mcp
======================================================================
EOF_DONE

if [ "$START_AFTER" = "1" ]; then
  start_args=(--profile "$PROFILE")
  if [ "$VIEWER" = "1" ]; then start_args+=(--viewer); else start_args+=(--no-viewer); fi
  if [ "$FOLLOW_CAMERA" = "1" ]; then start_args+=(--follow-camera); else start_args+=(--no-follow-camera); fi
  [ "$KEEP_RUNNING" = "1" ] && start_args+=(--keep-running)
  exec ./scripts/start_soridormi_mujoco.sh "${start_args[@]}"
fi
