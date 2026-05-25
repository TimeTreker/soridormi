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

cat > .env <<EOF_ENV
UID=${UID_VALUE}
GID=${GID_VALUE}
CONTAINER_USER=chromie

VIDEO_GID=${VIDEO_GID_VALUE}
RENDER_GID=${RENDER_GID_VALUE}

RUNTIME_DEV_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
RUNTIME_DEV_EXTRA=runtime-gpu

ORIN_RUNTIME_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
ORIN_RUNTIME_EXTRA=runtime

THOR_RUNTIME_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
THOR_RUNTIME_EXTRA=runtime

SIM_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
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
echo "  RUNTIME_DEV_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04"
echo "  SIM_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04"
echo "  SORIDORMI_SIM_BACKEND=fake"
echo "  SORIDORMI_ROBOT_CONFIG=/app/configs/robots/open_duck_mini_v2.yaml"
