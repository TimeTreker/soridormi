#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/report_dataset_coverage.sh DATASET_OR_PREPARED_DIR [more inputs...] [options]

Report Soridormi behavior-cloning dataset coverage by scenario_id, skill_id,
vx/vy/yaw command distributions, ramp alpha, terrain type, dataset tags, and
failure/stuck flags. Inputs may be raw JSONL datasets, prepared_manifest.json,
or a prepared dataset directory containing train/val/test JSONL files.

Common examples:
  ./scripts/report_dataset_coverage.sh \
    /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
    --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1 \
    --json

  ./scripts/report_dataset_coverage.sh \
    /data/training_datasets/prepared/flat_walk_varied_speed_v1 \
    --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1_prepared

Recommended MuJoCo-first collection flow before reporting:
  collect_random_teacher_dataset.sh owns its MuJoCo collection lifecycle; do not
  start a separate run_sim_server.sh for the same collection run. Use --viewer
  on the collector command for visual inspection.
  ./scripts/collect_random_teacher_dataset.sh \
    --backend mujoco \
    --viewer \
    --scenario flat_walk_varied_speed_v1 \
    --profile open_duck_forward \
    --episodes 2 \
    --steps-per-episode 300 \
    --command-ramp-steps 20 \
    --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
    --json | python -m json.tool
  ./scripts/report_dataset_coverage.sh /data/training_datasets/flat_walk_varied_speed_v1.jsonl
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

translated_args=()
soridormi_translate_container_data_args translated_args "$@"

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.dataset_coverage_report "$@"
' _ "${translated_args[@]}"
