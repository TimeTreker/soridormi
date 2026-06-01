#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_stride_step_metrics_eval.sh PROFILE [options]

Run a bounded policy smoke rollout with JSONL logging, then analyze stride,
step, progress, clearance, stuck, and fall metrics from the resulting log.
Start the MuJoCo simulator separately; for visual validation use:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

Options:
  --steps N                       Rollout steps. default: 1200
  --output-dir DIR                Report directory. default: data/stride_step/<profile>
  --fallback-control-hz HZ        Used if JSONL lacks robot time. default: 50
  --min-forward-speed MPS         Warning threshold. default: 0.02
  --max-stuck-sample-ratio R      Warning threshold. default: 0.40
  --min-base-z M                  Fall threshold. default: 0.12
  --max-abs-roll-pitch RAD        Fall threshold. default: 0.90
  --min-touchdown-count N         Warning threshold. default: 4
  --min-step-length M             Warning threshold. default: 0.01
  --min-swing-clearance M         Low-clearance threshold. default: 0.015
  --max-low-clearance-ratio R     Warning threshold. default: 0.35
  --contact-threshold C           Foot contact threshold. default: 0.5
  --log-prefix PREFIX             Runtime log prefix. default: stride_step_<profile>
  --skip-model-check              Forwarded to rollout smoke harness.
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
STEPS=1200
OUTPUT_DIR="data/stride_step/${PROFILE}"
FALLBACK_CONTROL_HZ=50
MIN_FORWARD_SPEED=0.02
MAX_STUCK_RATIO=0.40
MIN_BASE_Z=0.12
MAX_ABS_ROLL_PITCH=0.90
MIN_TOUCHDOWN_COUNT=4
MIN_STEP_LENGTH=0.01
MIN_SWING_CLEARANCE=0.015
MAX_LOW_CLEARANCE_RATIO=0.35
CONTACT_THRESHOLD=0.5
LOG_PREFIX="stride_step_${PROFILE}"
SKIP_MODEL_CHECK=0

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
    --fallback-control-hz)
      FALLBACK_CONTROL_HZ="${2:?--fallback-control-hz requires a value}"
      shift 2
      ;;
    --min-forward-speed)
      MIN_FORWARD_SPEED="${2:?--min-forward-speed requires a value}"
      shift 2
      ;;
    --max-stuck-sample-ratio)
      MAX_STUCK_RATIO="${2:?--max-stuck-sample-ratio requires a value}"
      shift 2
      ;;
    --min-base-z)
      MIN_BASE_Z="${2:?--min-base-z requires a value}"
      shift 2
      ;;
    --max-abs-roll-pitch)
      MAX_ABS_ROLL_PITCH="${2:?--max-abs-roll-pitch requires a value}"
      shift 2
      ;;
    --min-touchdown-count)
      MIN_TOUCHDOWN_COUNT="${2:?--min-touchdown-count requires a value}"
      shift 2
      ;;
    --min-step-length)
      MIN_STEP_LENGTH="${2:?--min-step-length requires a value}"
      shift 2
      ;;
    --min-swing-clearance)
      MIN_SWING_CLEARANCE="${2:?--min-swing-clearance requires a value}"
      shift 2
      ;;
    --max-low-clearance-ratio)
      MAX_LOW_CLEARANCE_RATIO="${2:?--max-low-clearance-ratio requires a value}"
      shift 2
      ;;
    --contact-threshold)
      CONTACT_THRESHOLD="${2:?--contact-threshold requires a value}"
      shift 2
      ;;
    --log-prefix)
      LOG_PREFIX="${2:?--log-prefix requires a value}"
      shift 2
      ;;
    --skip-model-check)
      SKIP_MODEL_CHECK=1
      shift
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

rollout_args=(
  "${PROFILE}"
  --steps "${STEPS}"
  --log-format jsonl
  --log-prefix "${LOG_PREFIX}"
  --log-dir /data/logs
)
if [ "${SKIP_MODEL_CHECK}" = "1" ]; then
  rollout_args+=(--skip-model-check)
fi

./scripts/run_policy_rollout_smoke.sh "${rollout_args[@]}"

LOG_PATH="$(find data/logs -type f -name "${LOG_PREFIX}*.jsonl" | sort | tail -1)"
if [ -z "${LOG_PATH}" ]; then
  echo "No JSONL log found for prefix ${LOG_PREFIX}" >&2
  exit 1
fi

REPORT_PATH="${OUTPUT_DIR}/stride_step_metrics_report.md"
JSON_PATH="${OUTPUT_DIR}/stride_step_metrics_report.json"

PYTHONPATH=src python -m soridormi_runtime.stride_step_metrics_eval "${LOG_PATH}" \
  --fallback-control-hz "${FALLBACK_CONTROL_HZ}" \
  --min-forward-speed "${MIN_FORWARD_SPEED}" \
  --max-stuck-sample-ratio "${MAX_STUCK_RATIO}" \
  --min-base-z "${MIN_BASE_Z}" \
  --max-abs-roll-pitch "${MAX_ABS_ROLL_PITCH}" \
  --min-touchdown-count "${MIN_TOUCHDOWN_COUNT}" \
  --min-step-length "${MIN_STEP_LENGTH}" \
  --min-swing-clearance "${MIN_SWING_CLEARANCE}" \
  --max-low-clearance-ratio "${MAX_LOW_CLEARANCE_RATIO}" \
  --contact-threshold "${CONTACT_THRESHOLD}" \
  --output "${REPORT_PATH}" \
  --json-output "${JSON_PATH}"

echo "Stride/step JSON: ${JSON_PATH}"
