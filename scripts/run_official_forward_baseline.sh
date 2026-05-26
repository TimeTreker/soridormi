#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

if [ ! -d workspace/Open_Duck_Playground/playground/open_duck_mini_v2 ]; then
  echo "Open_Duck_Playground submodule is missing. Run:"
  echo "  git submodule update --init --recursive workspace/Open_Duck_Playground workspace/Open_Duck_Mini"
  exit 1
fi

if [ ! -f workspace/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx ]; then
  echo "Policy file missing: workspace/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
  echo "Make sure Open_Duck_Mini submodule/assets are initialized."
  exit 1
fi

AUTO_XHOST="${SORIDORMI_XHOST_AUTO:-1}"
VIEWER_ENABLED="${SORIDORMI_OFFICIAL_VIEWER:-1}"
XHOST_ADDED=0

is_true() {
  case "${1,,}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

enable_xhost_if_needed() {
  if ! is_true "$AUTO_XHOST" || ! is_true "$VIEWER_ENABLED"; then
    return 0
  fi
  if [ -z "${DISPLAY:-}" ]; then
    echo "Warning: DISPLAY is not set. MuJoCo viewer may not open."
    return 0
  fi
  if command -v xhost >/dev/null 2>&1; then
    echo "Allowing local Docker containers to access X11..."
    xhost +local:docker >/dev/null || true
    XHOST_ADDED=1
  fi
}

cleanup_xhost() {
  if [ "$XHOST_ADDED" = "1" ]; then
    echo "Removing local Docker X11 access..."
    xhost -local:docker >/dev/null 2>&1 || true
  fi
}
trap cleanup_xhost EXIT INT TERM

enable_xhost_if_needed

viewer_arg=""
if ! is_true "$VIEWER_ENABLED"; then
  viewer_arg="--no-viewer"
fi

realtime_arg=""
if ! is_true "${SORIDORMI_OFFICIAL_REALTIME:-1}"; then
  realtime_arg="--no-realtime"
fi

mkdir -p data/official_baseline

docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_OFFICIAL_COMMAND_X="${SORIDORMI_OFFICIAL_COMMAND_X:-0.15}" \
  -e SORIDORMI_OFFICIAL_COMMAND_Y="${SORIDORMI_OFFICIAL_COMMAND_Y:-0.0}" \
  -e SORIDORMI_OFFICIAL_COMMAND_YAW="${SORIDORMI_OFFICIAL_COMMAND_YAW:-0.0}" \
  -e SORIDORMI_OFFICIAL_MAX_SECONDS="${SORIDORMI_OFFICIAL_MAX_SECONDS:-20}" \
  -e SORIDORMI_OFFICIAL_VIEWER="${SORIDORMI_OFFICIAL_VIEWER:-1}" \
  -e SORIDORMI_OFFICIAL_REALTIME="${SORIDORMI_OFFICIAL_REALTIME:-1}" \
  -e SORIDORMI_OFFICIAL_FAST_EXIT="${SORIDORMI_OFFICIAL_FAST_EXIT:-1}" \
  sim bash -lc "
    source /opt/venvs/sim/bin/activate
    export PYTHONPATH=/app/src:/workspaces/Open_Duck_Playground:\${PYTHONPATH:-}
    python -m soridormi_sim.official_open_duck_baseline \
      --playground-root /workspaces/Open_Duck_Playground \
      --model-path /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml \
      --reference-data /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl \
      --onnx-model-path /workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
      --output-dir /data/official_baseline \
      --summary-prefix official_forward \
      --command-x \${SORIDORMI_OFFICIAL_COMMAND_X:-0.15} \
      --command-y \${SORIDORMI_OFFICIAL_COMMAND_Y:-0.0} \
      --command-yaw \${SORIDORMI_OFFICIAL_COMMAND_YAW:-0.0} \
      --max-seconds \${SORIDORMI_OFFICIAL_MAX_SECONDS:-20} \
      ${viewer_arg} ${realtime_arg}
  "
