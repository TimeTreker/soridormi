#!/usr/bin/env bash
set -euo pipefail

VENV=/opt/venvs/runtime
APP=/app
RUNTIME_REPO=/workspaces/Open_Duck_Mini_Runtime

if [ -x "${VENV}/bin/python" ]; then
  echo "Runtime environment already exists at ${VENV}."
  echo "This environment is built into the Docker image by default."
  echo "Use:"
  echo "  source ${VENV}/bin/activate"
  echo ""
  echo "To reinstall Soridormi inside this running container, use:"
  echo "  uv pip install --python ${VENV}/bin/python -e ${APP}[runtime-gpu,dev]"
  exit 0
fi

echo "Creating runtime Python environment..."
uv venv --python python3 "${VENV}"
uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel

RUNTIME_EXTRA="runtime"
if [ "${SORIDORMI_ONNXRUNTIME_GPU:-0}" = "1" ]; then
  RUNTIME_EXTRA="runtime-gpu"
fi
uv pip install --python "${VENV}/bin/python" -e "${APP}[${RUNTIME_EXTRA},dev]"

if [ -d "${RUNTIME_REPO}" ]; then
  uv pip install --python "${VENV}/bin/python" -e "${RUNTIME_REPO}"
else
  echo "Warning: ${RUNTIME_REPO} not found. Run scripts/add_submodules.sh on host if needed."
fi

echo ""
echo "Runtime environment ready."
echo "Use:"
echo "  source ${VENV}/bin/activate"
