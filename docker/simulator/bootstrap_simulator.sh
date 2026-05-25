#!/usr/bin/env bash
set -euo pipefail

VENV=/opt/venvs/sim
APP=/app
PLAYGROUND=/workspaces/Open_Duck_Playground

if [ -x "${VENV}/bin/python" ]; then
  echo "Simulator environment already exists at ${VENV}."
  echo "This environment is built into the Docker image by default."
  echo "Use:"
  echo "  source ${VENV}/bin/activate"
  echo ""
  echo "To install the optional editable Open Duck Playground repo into this running container, use:"
  echo "  uv pip install --python ${VENV}/bin/python -e ${PLAYGROUND}"
  exit 0
fi

echo "Creating simulator Python environment..."
uv venv --python python3 "${VENV}"
uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel
uv pip install --python "${VENV}/bin/python" -e "${APP}[sim,dev]"
uv pip install --python "${VENV}/bin/python" tensorboard

if [ -d "${PLAYGROUND}" ]; then
  uv pip install --python "${VENV}/bin/python" -e "${PLAYGROUND}"
else
  echo "Warning: ${PLAYGROUND} not found. Run scripts/add_submodules.sh on host if needed."
fi

echo ""
echo "Simulation environment ready."
echo "Use:"
echo "  source ${VENV}/bin/activate"
