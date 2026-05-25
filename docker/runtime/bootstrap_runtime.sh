#!/usr/bin/env bash
set -euo pipefail

APP=/app
RUNTIME_REPO=/workspaces/Open_Duck_Mini_Runtime

uv venv --python python3 /opt/venvs/runtime
uv pip install --python /opt/venvs/runtime/bin/python --upgrade pip setuptools wheel

# Install this project first so API/runtime imports work immediately.
uv pip install --python /opt/venvs/runtime/bin/python -e "$APP[runtime]"

# Install upstream runtime repo when available.
if [ -d "$RUNTIME_REPO" ]; then
  uv pip install --python /opt/venvs/runtime/bin/python -e "$RUNTIME_REPO"
else
  echo "Warning: $RUNTIME_REPO not found. Run scripts/add_submodules.sh on host if needed."
fi

echo ""
echo "Runtime environment ready."
echo "Use:"
echo "  source /opt/venvs/runtime/bin/activate"
