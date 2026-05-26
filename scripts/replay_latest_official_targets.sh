#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

TRACE="${1:-${SORIDORMI_OFFICIAL_TRACE:-/data/official_baseline/latest_official_baseline.trace.jsonl}}"
MAX_STEPS="${SORIDORMI_REPLAY_MAX_STEPS:-0}"

mkdir -p data/official_baseline

docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_SIM_BACKEND=mujoco \
  -e SORIDORMI_MUJOCO_VIEWER=0 \
  -e SORIDORMI_AUTO_RESET=0 \
  -e SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 \
  -e SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1 \
  -e SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1 \
  -e SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1 \
  -e SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1 \
  sim bash -lc '
    set -euo pipefail
    source /opt/venvs/sim/bin/activate
    export PYTHONPATH=/app/src:/workspaces/Open_Duck_Playground:${PYTHONPATH:-}
    python -m soridormi_sim.replay_official_trace \
      --trace "$1" \
      --output-dir /data/official_baseline \
      --summary-prefix official_target_replay \
      --max-steps "$2"
  ' _ "${TRACE}" "${MAX_STEPS}"
