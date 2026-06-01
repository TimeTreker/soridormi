#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_foot_clearance_eval.sh PROFILE [options]

Run a policy smoke rollout with JSONL logging, then analyze left/right foot-clearance
from state.feet_position_xyz. Start the MuJoCo simulator separately;
for normal flat-ground validation use:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
For visual/rough-ground validation use:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --rough-ground

Options:
  --steps N                       Rollout steps. default: 1000
  --output-dir DIR                Report directory. default: data/foot_clearance/<profile>
  --ground-z Z                    Ground height for clearance. default: 0.0
  --min-swing-clearance M         Low-clearance threshold. default: 0.015
  --target-swing-clearance M      Target median swing clearance. default: 0.025
  --max-low-clearance-ratio R     Warning threshold. default: 0.25
  --log-prefix PREFIX             Runtime log prefix. default: foot_clearance_<profile>
  -h, --help                      Show this help.
USAGE
}

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
esac

PROFILE="$1"
shift
STEPS=1000
OUTPUT_DIR="data/foot_clearance/${PROFILE}"
GROUND_Z=0.0
MIN_SWING=0.015
TARGET_SWING=0.025
MAX_LOW_RATIO=0.25
LOG_PREFIX="foot_clearance_${PROFILE}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --steps)
      STEPS="${2:?--steps requires a value}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --ground-z)
      GROUND_Z="${2:?--ground-z requires a value}"
      shift 2
      ;;
    --min-swing-clearance)
      MIN_SWING="${2:?--min-swing-clearance requires a value}"
      shift 2
      ;;
    --target-swing-clearance)
      TARGET_SWING="${2:?--target-swing-clearance requires a value}"
      shift 2
      ;;
    --max-low-clearance-ratio)
      MAX_LOW_RATIO="${2:?--max-low-clearance-ratio requires a value}"
      shift 2
      ;;
    --log-prefix)
      LOG_PREFIX="${2:?--log-prefix requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}" data/logs

./scripts/run_policy_rollout_smoke.sh "${PROFILE}" \
  --steps "${STEPS}" \
  --log-format jsonl \
  --log-prefix "${LOG_PREFIX}" \
  --log-dir /data/logs

LOG_PATH="$(find data/logs -type f -name "${LOG_PREFIX}*.jsonl" | sort | tail -1)"
if [ -z "${LOG_PATH}" ]; then
  echo "No JSONL log found for prefix ${LOG_PREFIX}" >&2
  exit 1
fi

REPORT_PATH="${OUTPUT_DIR}/foot_clearance_report.md"
JSON_PATH="${OUTPUT_DIR}/foot_clearance_report.json"

PYTHONPATH=src python -m soridormi_runtime.foot_clearance_eval "${LOG_PATH}" \
  --ground-z "${GROUND_Z}" \
  --min-swing-clearance "${MIN_SWING}" \
  --target-swing-clearance "${TARGET_SWING}" \
  --max-low-clearance-ratio "${MAX_LOW_RATIO}" \
  --output "${REPORT_PATH}"

PYTHONPATH=src python -m soridormi_runtime.foot_clearance_eval "${LOG_PATH}" \
  --ground-z "${GROUND_Z}" \
  --min-swing-clearance "${MIN_SWING}" \
  --target-swing-clearance "${TARGET_SWING}" \
  --max-low-clearance-ratio "${MAX_LOW_RATIO}" \
  --json > "${JSON_PATH}"

echo "Foot-clearance JSON: ${JSON_PATH}"
