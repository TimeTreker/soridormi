#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_sim_server.sh [options]

Start the Soridormi simulator server. The normal locomotion/sim validation
backend is MuJoCo. The MuJoCo viewer is off by default; enable it explicitly
when you want a visual functional test.

Options:
  --backend NAME        Simulator backend to start. default: mujoco
                        Common values: mujoco, fake
  --viewer              Enable the passive MuJoCo viewer.
  --no-viewer           Disable the passive MuJoCo viewer. default
  -h, --help            Show this help.

Examples:
  ./scripts/run_sim_server.sh --backend mujoco --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --viewer
USAGE
}

SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"
VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend)
      SIM_BACKEND="${2:?--backend requires a value}"
      shift 2
      ;;
    --viewer)
      VIEWER_ENABLED="1"
      shift
      ;;
    --no-viewer)
      VIEWER_ENABLED="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

AUTO_XHOST="${SORIDORMI_XHOST_AUTO:-1}"
XHOST_ADDED=0

is_true() {
  case "${1,,}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

enable_xhost_if_needed() {
  if ! is_true "$AUTO_XHOST"; then
    return 0
  fi

  if ! is_true "$VIEWER_ENABLED"; then
    return 0
  fi

  if [ -z "${DISPLAY:-}" ]; then
    echo "Warning: DISPLAY is not set. MuJoCo viewer may not open."
    return 0
  fi

  if ! command -v xhost >/dev/null 2>&1; then
    echo "Warning: xhost not found. Install it with: sudo apt install x11-xserver-utils"
    return 0
  fi

  echo "Allowing local Docker containers to access X11..."
  xhost +local:docker >/dev/null || true
  XHOST_ADDED=1
}

cleanup_xhost() {
  if [ "$XHOST_ADDED" = "1" ]; then
    echo "Removing local Docker X11 access..."
    xhost -local:docker >/dev/null 2>&1 || true
  fi
}

trap cleanup_xhost EXIT INT TERM

enable_xhost_if_needed

export SORIDORMI_SIM_BACKEND="${SIM_BACKEND}"
export SORIDORMI_MUJOCO_VIEWER="${VIEWER_ENABLED}"

echo "Soridormi simulator server"
echo "=========================="
echo "Backend: ${SORIDORMI_SIM_BACKEND}"
echo "MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"

docker compose -f compose.sim.yaml run --rm sim bash -lc '
  source /opt/venvs/sim/bin/activate
  python -m soridormi_sim.mujoco_server
'
