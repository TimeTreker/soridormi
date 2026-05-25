#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

AUTO_XHOST="${SORIDORMI_XHOST_AUTO:-1}"
VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"
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
  xhost +local:docker >/dev/null
  XHOST_ADDED=1
}

cleanup_xhost() {
  if [ "$XHOST_ADDED" = "1" ]; then
    echo "Removing local Docker X11 access..."
    xhost -local:docker >/dev/null || true
  fi
}

trap cleanup_xhost EXIT INT TERM

enable_xhost_if_needed

docker compose -f compose.sim.yaml run --rm sim bash -lc '
  source /opt/venvs/sim/bin/activate
  python -m soridormi_sim.mujoco_server
'
