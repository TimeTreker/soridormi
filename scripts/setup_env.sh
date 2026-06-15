#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

detect_gid() {
  local group_name="$1"
  local fallback="$2"

  if getent group "${group_name}" >/dev/null 2>&1; then
    getent group "${group_name}" | cut -d: -f3
  else
    echo "${fallback}"
  fi
}

detect_render_gid() {
  if getent group render >/dev/null 2>&1; then
    getent group render | cut -d: -f3
  elif [ -e /dev/dri/renderD128 ]; then
    stat -c '%g' /dev/dri/renderD128
  elif [ -e /dev/dri/card0 ]; then
    stat -c '%g' /dev/dri/card0
  else
    echo "109"
  fi
}

UID_VALUE="$(id -u)"
GID_VALUE="$(id -g)"
VIDEO_GID_VALUE="$(detect_gid video 44)"
RENDER_GID_VALUE="$(detect_render_gid)"

# Docker image references have one source of truth: this generated .env file.
# Callers may override any value in the environment when running setup_env.sh.
: "${SORIDORMI_RUNTIME_IMAGE:=soridormi-runtime:cuda13.1-cudnn-dev}"
: "${SORIDORMI_RUNTIME_MCP_IMAGE:=soridormi-runtime-mcp:cuda13.1-cudnn-dev}"
: "${SORIDORMI_SIM_IMAGE:=soridormi-sim:cuda13.1-cudnn}"
: "${SORIDORMI_ORIN_RUNTIME_IMAGE:=soridormi-runtime:orin}"
: "${SORIDORMI_THOR_RUNTIME_IMAGE:=soridormi-runtime:thor}"

: "${RUNTIME_DEV_BASE:=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
: "${ORIN_RUNTIME_BASE:=nvcr.io/nvidia/l4t-jetpack:r36.4.0}"
: "${THOR_RUNTIME_BASE:=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04}"
: "${SIM_BASE:=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"

cat > .env <<EOF_ENV
UID=${UID_VALUE}
GID=${GID_VALUE}
CONTAINER_USER=chromie

VIDEO_GID=${VIDEO_GID_VALUE}
RENDER_GID=${RENDER_GID_VALUE}

SORIDORMI_RUNTIME_IMAGE=${SORIDORMI_RUNTIME_IMAGE}
SORIDORMI_RUNTIME_MCP_IMAGE=${SORIDORMI_RUNTIME_MCP_IMAGE}
SORIDORMI_SIM_IMAGE=${SORIDORMI_SIM_IMAGE}
SORIDORMI_ORIN_RUNTIME_IMAGE=${SORIDORMI_ORIN_RUNTIME_IMAGE}
SORIDORMI_THOR_RUNTIME_IMAGE=${SORIDORMI_THOR_RUNTIME_IMAGE}

RUNTIME_DEV_BASE=${RUNTIME_DEV_BASE}
RUNTIME_DEV_EXTRA=runtime-gpu

ORIN_RUNTIME_BASE=${ORIN_RUNTIME_BASE}
ORIN_RUNTIME_EXTRA=runtime

THOR_RUNTIME_BASE=${THOR_RUNTIME_BASE}
THOR_RUNTIME_EXTRA=runtime

SIM_BASE=${SIM_BASE}
SIM_HOST=127.0.0.1
SIM_PORT=5555
CONTROL_HZ=50
SORIDORMI_ONNXRUNTIME_GPU=1
SORIDORMI_USE_CUDA_PROVIDER=1

SORIDORMI_SIM_BACKEND=fake
SORIDORMI_ROBOT_CONFIG=/app/configs/robots/open_duck_mini_v2.yaml
MUJOCO_MODEL_PATH=
EOF_ENV

echo "Generated .env:"
echo "  UID=${UID_VALUE}"
echo "  GID=${GID_VALUE}"
echo "  CONTAINER_USER=chromie"
echo "  VIDEO_GID=${VIDEO_GID_VALUE}"
echo "  RENDER_GID=${RENDER_GID_VALUE}"
echo "  SORIDORMI_RUNTIME_IMAGE=${SORIDORMI_RUNTIME_IMAGE}"
echo "  SORIDORMI_RUNTIME_MCP_IMAGE=${SORIDORMI_RUNTIME_MCP_IMAGE}"
echo "  SORIDORMI_SIM_IMAGE=${SORIDORMI_SIM_IMAGE}"
echo "  SORIDORMI_ORIN_RUNTIME_IMAGE=${SORIDORMI_ORIN_RUNTIME_IMAGE}"
echo "  SORIDORMI_THOR_RUNTIME_IMAGE=${SORIDORMI_THOR_RUNTIME_IMAGE}"
echo "  RUNTIME_DEV_BASE=${RUNTIME_DEV_BASE}"
echo "  ORIN_RUNTIME_BASE=${ORIN_RUNTIME_BASE}"
echo "  THOR_RUNTIME_BASE=${THOR_RUNTIME_BASE}"
echo "  SIM_BASE=${SIM_BASE}"
echo "  SORIDORMI_SIM_BACKEND=fake"
echo "  SORIDORMI_ROBOT_CONFIG=/app/configs/robots/open_duck_mini_v2.yaml"
