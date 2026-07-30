#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-open_duck_forward}"
SIM_PORT="${SIM_PORT:-5555}"
MCP_PORT="${SORIDORMI_MCP_PORT:-8000}"
MCP_PATH="${SORIDORMI_MCP_PATH:-/mcp}"
VIEWER=1
FOLLOW_CAMERA=1
SOCIAL_EYE_FRAME=0
BUILD_IMAGES=0
KEEP_RUNNING=0
REUSE_EXISTING_SIM=0
RESTART_MCP=0

usage() {
  cat <<'USAGE'
Usage: ./scripts/start_soridormi_mujoco.sh [options]

Start Soridormi's MuJoCo simulator and runtime-backed MCP server.
The script uses the image names defined by Soridormi's own compose.sim.yaml.
It does not replace them with latest tags and does not pull images.

With the current repository, the expected local images are resolved by Compose,
normally including:
  soridormi-sim:cuda13.1-cudnn
  soridormi-runtime:cuda13.1-cudnn-dev
  soridormi-runtime-mcp:cuda13.1-cudnn-dev

Options:
  --build              Run the repository's image build flow first
  --profile NAME       Policy profile; default: open_duck_forward
  --viewer             Open MuJoCo viewer; default
  --no-viewer          Run headless
  --follow-camera      Follow the robot; default
  --no-follow-camera   Disable follow camera
  --social-eye-frame   Show an RGB debug frame at the generated eye anchor
  --no-social-eye-frame
                       Hide the generated eye debug frame; default
  --keep-running       Leave services running after launcher exits
  --reuse-existing-sim Reuse a simulator already listening on SIM_PORT
  --restart-existing-sim
                       Restart an existing Soridormi sim first; default
  -h, --help           Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build) BUILD_IMAGES=1; shift ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --viewer) VIEWER=1; shift ;;
    --no-viewer) VIEWER=0; shift ;;
    --follow-camera) FOLLOW_CAMERA=1; shift ;;
    --no-follow-camera) FOLLOW_CAMERA=0; shift ;;
    --social-eye-frame) SOCIAL_EYE_FRAME=1; shift ;;
    --no-social-eye-frame) SOCIAL_EYE_FRAME=0; shift ;;
    --keep-running) KEEP_RUNNING=1; shift ;;
    --reuse-existing-sim) REUSE_EXISTING_SIM=1; shift ;;
    --restart-existing-sim) REUSE_EXISTING_SIM=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[soridormi][error] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -f "$SCRIPT_DIR/../compose.sim.yaml" ]; then
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -f "$SCRIPT_DIR/compose.sim.yaml" ]; then
  ROOT_DIR="$SCRIPT_DIR"
else
  echo "[soridormi][error] Put this script in the Soridormi root or scripts/." >&2
  exit 1
fi
cd "$ROOT_DIR"

if [ -z "${SORIDORMI_SOURCE_REVISION:-}" ]; then
  SORIDORMI_SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
  export SORIDORMI_SOURCE_REVISION
fi

for cmd in docker python3; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[soridormi][error] Required command not found: $cmd" >&2
    exit 1
  }
done

docker info >/dev/null 2>&1 || {
  echo "[soridormi][error] Docker daemon is not reachable." >&2
  exit 1
}

if [ "$VIEWER" = "1" ] && [ -z "${DISPLAY:-}" ]; then
  echo "[soridormi][error] DISPLAY is not set. Use --no-viewer." >&2
  exit 1
fi

for path in \
  compose.sim.yaml \
  scripts/setup_env.sh \
  scripts/build_sim.sh \
  scripts/run_sim_server.sh; do
  [ -e "$path" ] || {
    echo "[soridormi][error] Missing repository file: $path" >&2
    exit 1
  }
done

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

REQUIRED_UPSTREAM_PATHS=(
  workspace/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
  workspace/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml
  workspace/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl
)

missing_upstream=0
for path in "${REQUIRED_UPSTREAM_PATHS[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[soridormi][error] Missing required upstream asset: $path" >&2
    missing_upstream=1
  fi
done
if [ "$missing_upstream" = "1" ]; then
  echo "[soridormi][hint] Run ./scripts/add_submodules.sh or ./scripts/deploy_soridormi.sh." >&2
  exit 1
fi

COMPOSE_ARGS=(-f compose.sim.yaml --profile mcp-runtime)

if [ "$BUILD_IMAGES" = "1" ]; then
  echo "[soridormi] Running repository build script for runtime and simulator..."
  ./scripts/build_sim.sh
  echo "[soridormi] Building runtime MCP image with its Compose-defined tag..."
  docker compose "${COMPOSE_ARGS[@]}" build mcp-runtime
fi

mapfile -t REQUIRED_IMAGES < <(
  docker compose "${COMPOSE_ARGS[@]}" config --images | awk 'NF && !seen[$0]++'
)

if [ "${#REQUIRED_IMAGES[@]}" -eq 0 ]; then
  echo "[soridormi][error] Could not resolve images from compose.sim.yaml." >&2
  exit 1
fi

missing=0
for image in "${REQUIRED_IMAGES[@]}"; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "[soridormi][error] Required local image is missing: $image" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "[soridormi][hint] Run this launcher with --build." >&2
  exit 1
fi

echo "[soridormi] Using Compose-defined local images only:"
printf '  %s\n' "${REQUIRED_IMAGES[@]}"

python_tcp_check() {
  python3 - "$1" "$2" <<'PY' >/dev/null 2>&1
import socket, sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1.0):
    pass
PY
}

wait_for_tcp() {
  local host="$1" port="$2" timeout_s="$3" label="$4"
  local deadline=$((SECONDS + timeout_s))
  echo "[soridormi] Waiting for $label at $host:$port..."
  until python_tcp_check "$host" "$port"; do
    if (( SECONDS >= deadline )); then
      echo "[soridormi][error] Timed out waiting for $label." >&2
      return 1
    fi
    sleep 2
  done
  echo "[soridormi] $label is ready."
}

wait_for_tcp_or_process() {
  local host="$1" port="$2" timeout_s="$3" label="$4" pid="$5"
  local deadline=$((SECONDS + timeout_s))
  echo "[soridormi] Waiting for $label at $host:$port..."
  until python_tcp_check "$host" "$port"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[soridormi][error] $label process exited before becoming ready." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[soridormi][error] Timed out waiting for $label." >&2
      return 1
    fi
    sleep 2
  done
  echo "[soridormi] $label is ready."
}

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]
}

stop_existing_sim_containers() {
  local names=()
  local name

  while IFS= read -r name; do
    case "$name" in
      soridormi-sim|soridormi-sim-*) names+=("$name") ;;
    esac
  done < <(docker ps --format '{{.Names}}')

  if [ "${#names[@]}" -eq 0 ]; then
    echo "[soridormi][warn] No Soridormi simulator container name found for active SIM_PORT." >&2
    return 0
  fi

  echo "[soridormi] Stopping existing simulator container(s): ${names[*]}"
  docker stop "${names[@]}" >/dev/null
}

RUN_DIR="${SORIDORMI_CHROMIE_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/soridormi-chromie-${UID:-$(id -u)}}"
mkdir -p "$RUN_DIR"
SIM_LOG="$RUN_DIR/mujoco.log"
MCP_LOG="$RUN_DIR/mcp.log"

OWN_SIM=0
OWN_MCP=0
SIM_PID=""
CLEANED=0

cleanup() {
  local rc=$?
  [ "$CLEANED" = "0" ] || return "$rc"
  CLEANED=1

  if [ "$KEEP_RUNNING" = "1" ]; then
    echo "[soridormi] Leaving Soridormi services running."
    return "$rc"
  fi

  echo
  echo "[soridormi] Stopping services started by this launcher..."
  if [ "$OWN_MCP" = "1" ]; then
    docker compose "${COMPOSE_ARGS[@]}" stop mcp-runtime >/dev/null 2>&1 || true
  fi
  if [ "$OWN_SIM" = "1" ] && [ -n "$SIM_PID" ] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill -TERM -- "-$SIM_PID" 2>/dev/null || kill -TERM "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
  return "$rc"
}
trap cleanup EXIT INT TERM

export SIM_PORT
export SORIDORMI_MCP_PORT="$MCP_PORT"
export SORIDORMI_MCP_PATH="$MCP_PATH"

if python_tcp_check 127.0.0.1 "$SIM_PORT"; then
  if [ "$REUSE_EXISTING_SIM" = "1" ]; then
    echo "[soridormi] Reusing existing simulator at 127.0.0.1:$SIM_PORT."
    echo "[soridormi][warn] Viewer, profile, follow-camera, and eye-frame options are not reapplied to reused simulators." >&2
  else
    echo "[soridormi] Existing simulator detected at 127.0.0.1:$SIM_PORT; restarting so viewer/profile options are applied."
    stop_existing_sim_containers
    RESTART_MCP=1
    if python_tcp_check 127.0.0.1 "$SIM_PORT"; then
      echo "[soridormi][error] SIM_PORT $SIM_PORT is still in use after stopping known Soridormi simulator containers." >&2
      echo "[soridormi][hint] Stop the process using the port, pass --reuse-existing-sim, or choose another SIM_PORT." >&2
      exit 1
    fi
  fi
fi

if ! python_tcp_check 127.0.0.1 "$SIM_PORT"; then
  echo "[soridormi] Starting MuJoCo with the repository launcher..."
  : > "$SIM_LOG"
  sim_args=(--backend mujoco --profile "$PROFILE")
  if [ "$VIEWER" = "1" ]; then sim_args+=(--viewer); else sim_args+=(--no-viewer); fi
  if [ "$FOLLOW_CAMERA" = "1" ]; then sim_args+=(--follow-camera); else sim_args+=(--no-follow-camera); fi
  if [ "$SOCIAL_EYE_FRAME" = "1" ]; then sim_args+=(--social-eye-frame); else sim_args+=(--no-social-eye-frame); fi

  if command -v setsid >/dev/null 2>&1; then
    setsid ./scripts/run_sim_server.sh "${sim_args[@]}" >>"$SIM_LOG" 2>&1 </dev/null &
  else
    ./scripts/run_sim_server.sh "${sim_args[@]}" >>"$SIM_LOG" 2>&1 </dev/null &
  fi
  SIM_PID=$!
  OWN_SIM=1

  if ! wait_for_tcp_or_process 127.0.0.1 "$SIM_PORT" 300 "MuJoCo simulator" "$SIM_PID"; then
    tail -n 160 "$SIM_LOG" >&2 || true
    exit 1
  fi
fi

if { [ "$RESTART_MCP" = "1" ] || [ "$OWN_SIM" = "1" ]; } && container_running soridormi-runtime-mcp; then
  echo "[soridormi] Restarting runtime MCP so it reconnects to this simulator."
  RESTART_MCP=1
  docker compose "${COMPOSE_ARGS[@]}" stop mcp-runtime >/dev/null 2>&1 || true
fi

if [ "$RESTART_MCP" = "0" ] && container_running soridormi-runtime-mcp && python_tcp_check 127.0.0.1 "$MCP_PORT"; then
  echo "[soridormi] Reusing existing runtime MCP server."
else
  echo "[soridormi] Starting runtime MCP without build or pull..."
  : > "$MCP_LOG"
  docker compose "${COMPOSE_ARGS[@]}" \
    up -d --no-build --pull never mcp-runtime >>"$MCP_LOG" 2>&1
  OWN_MCP=1
  if ! wait_for_tcp 127.0.0.1 "$MCP_PORT" 300 "Soridormi MCP"; then
    docker compose "${COMPOSE_ARGS[@]}" logs --tail=160 mcp-runtime >&2 || true
    exit 1
  fi
fi

if [ "$VIEWER" = "1" ]; then
  VIEWER_STATUS="enabled"
else
  VIEWER_STATUS="disabled"
fi

if [ "$FOLLOW_CAMERA" = "1" ]; then
  FOLLOW_CAMERA_STATUS="enabled"
else
  FOLLOW_CAMERA_STATUS="disabled"
fi

cat <<EOF_READY

======================================================================
Soridormi is ready for Chromie
======================================================================
MuJoCo:  127.0.0.1:${SIM_PORT}
MCP:     http://127.0.0.1:${MCP_PORT}${MCP_PATH}
Profile: ${PROFILE}
Viewer:  ${VIEWER_STATUS}
Follow camera: ${FOLLOW_CAMERA_STATUS}
Docker policy: Compose-defined local images only; no image pulling
Logs: ${RUN_DIR}

Now start Chromie in another terminal.
Press Ctrl+C to stop Soridormi services started by this launcher.
======================================================================
EOF_READY

while true; do
  sleep 3
  if ! python_tcp_check 127.0.0.1 "$SIM_PORT"; then
    echo "[soridormi][error] MuJoCo simulator stopped unexpectedly." >&2
    tail -n 120 "$SIM_LOG" >&2 || true
    exit 1
  fi
  if ! python_tcp_check 127.0.0.1 "$MCP_PORT"; then
    echo "[soridormi][error] Runtime MCP server stopped unexpectedly." >&2
    docker compose "${COMPOSE_ARGS[@]}" logs --tail=120 mcp-runtime >&2 || true
    exit 1
  fi
done
