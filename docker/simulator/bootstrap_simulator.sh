#!/usr/bin/env bash
set -euo pipefail

APP=/app
MINI=/workspaces/Open_Duck_Mini
PLAYGROUND=/workspaces/Open_Duck_Playground

uv venv --python python3 /opt/venvs/sim
uv pip install --python /opt/venvs/sim/bin/python --upgrade pip setuptools wheel

# Install this project first so the simulator API server works immediately.
uv pip install --python /opt/venvs/sim/bin/python -e "$APP[sim]"

# Install Open Duck Playground when available.
if [ -d "$PLAYGROUND" ]; then
  uv pip install --python /opt/venvs/sim/bin/python -e "$PLAYGROUND"
else
  echo "Warning: $PLAYGROUND not found. Run scripts/add_submodules.sh on host if needed."
fi

# Install legacy Open Duck Mini optional dependencies only if needed.
if [ -d "$MINI" ]; then
  echo "Open_Duck_Mini found at $MINI"
else
  echo "Warning: $MINI not found. Run scripts/add_submodules.sh on host if needed."
fi

# TensorBoard is useful for training logs.
uv pip install --python /opt/venvs/sim/bin/python tensorboard

echo ""
echo "Simulation environment ready."
echo "Use:"
echo "  source /opt/venvs/sim/bin/activate"
