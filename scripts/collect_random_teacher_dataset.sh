#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/collect_random_teacher_dataset.sh [options]

Collect behavior-cloning samples by rolling out the teacher policy under random
piecewise velocity commands. Command changes are ramped by default so the
dataset covers continuous speed transitions instead of only abrupt jumps. Start
the MuJoCo sim server separately before using this script.

Common options:
  --profile NAME                 teacher profile (default: open_duck_forward)
  --output PATH                  dataset JSONL output
  --episodes N                   rollout episodes
  --steps-per-episode N          rollout steps per episode
  --vx-range MIN,MAX             random forward velocity range
  --vy-range MIN,MAX             random lateral velocity range
  --yaw-range MIN,MAX            random yaw velocity range
  --command-hold-steps MIN,MAX   random command segment duration in control steps
  --command-ramp-steps N         smooth command changes over N control steps (default: 20)
  --seed N                       deterministic random seed

Negative ranges may be passed either as separate values or with '=':
  --vx-range -0.03,0.15
  --vx-range=-0.03,0.15
  --backend NAME                 expected sim backend; default: mujoco
  --viewer                       viewer mode hint for matching sim server command
  --no-viewer                    headless mode hint; default

Sim server commands for walking teacher parity:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
USAGE
}

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

sim_backend="mujoco"
viewer_enabled="0"
collector_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend)
      sim_backend="${2:?--backend requires a value}"
      shift 2
      ;;
    --backend=*)
      sim_backend="${1#*=}"
      shift
      ;;
    --viewer)
      viewer_enabled="1"
      shift
      ;;
    --no-viewer)
      viewer_enabled="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      collector_args+=("$1")
      shift
      ;;
  esac
done

if [ "${sim_backend}" != "mujoco" ]; then
  echo "Warning: random teacher collection is intended for MuJoCo; requested backend: ${sim_backend}" >&2
fi

cat <<EOF
Soridormi random-command teacher collection
==========================================
Expected sim backend: ${sim_backend}
MuJoCo viewer hint: ${viewer_enabled}

Make sure the simulator is already running in another terminal with the same
teacher profile compatibility flags, for example:
  ./scripts/run_sim_server.sh --backend ${sim_backend} --profile open_duck_forward $(if [ "${viewer_enabled}" = "1" ]; then echo --viewer; else echo --no-viewer; fi)
EOF

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

translated_args=()
soridormi_translate_container_data_args translated_args "${collector_args[@]}"

export SORIDORMI_SIM_BACKEND="${sim_backend}"
export SORIDORMI_MUJOCO_VIEWER="${viewer_enabled}"

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.random_teacher_dataset_collect "$@"
' _ "${translated_args[@]}"
