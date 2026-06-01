#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_skill_in_sim.sh SKILL [options]

Resolve an available M7 locomotion skill into a high-level velocity command and
execute it through the Soridormi policy runtime against an already-running MuJoCo
simulator. This is sim-only orchestration; it does not talk to hardware.

Options:
  --args JSON              Skill parameter JSON object (default: {}).
  --profile PROFILE        Policy profile to execute with (default: open_duck_forward).
  --steps N                Override rollout control-step count.
  --control-hz HZ          Control frequency used to convert skill duration to steps (default: 50).
  --seconds S              Optional wall-clock cutoff. Not used by default.
  --log-format FORMAT      Runtime log format: mcap or jsonl.
  --log-prefix PREFIX      Runtime log prefix (default: skill_SKILL).
  --log-dir DIR            Runtime log directory inside the container (default: /data/logs).
  --log-every-n N          Runtime log cadence.
  --no-log                 Disable runtime logging.
  --skip-model-check       Skip policy model preflight.
  --dry-run-only           Print the resolved plan and command overrides, then exit.
  -h, --help               Show this help.

Start MuJoCo first with the same profile and an explicit backend/viewer choice:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

By default, skill duration is converted to control steps. This avoids ending a
skill after one simulator step when CUDA/ONNX warm-up consumes the wall-clock
--seconds budget.

Example:
  ./scripts/run_skill_in_sim.sh walk_velocity \
    --args '{"vx_mps":0.12,"duration_s":3.0}' \
    --profile open_duck_forward \
    --log-format jsonl
USAGE
}

skill="${1:-}"
if [ -z "${skill}" ] || [ "${skill}" = "-h" ] || [ "${skill}" = "--help" ]; then
  usage
  exit 0
fi
shift || true

args="{}"
profile="open_duck_forward"
steps=""
steps_override="0"
control_hz="${SORIDORMI_SKILL_CONTROL_HZ:-50}"
seconds=""
seconds_override="0"
log_format=""
log_prefix=""
log_dir="/data/logs"
log_every_n=""
no_log="0"
skip_model_check="0"
dry_run_only="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --args)
      args="${2:?--args requires a JSON object}"
      shift 2
      ;;
    --args=*)
      args="${1#*=}"
      shift
      ;;
    --profile)
      profile="${2:?--profile requires a value}"
      shift 2
      ;;
    --profile=*)
      profile="${1#*=}"
      shift
      ;;
    --steps)
      steps="${2:?--steps requires a value}"
      steps_override="1"
      shift 2
      ;;
    --steps=*)
      steps="${1#*=}"
      steps_override="1"
      shift
      ;;
    --control-hz)
      control_hz="${2:?--control-hz requires a value}"
      shift 2
      ;;
    --control-hz=*)
      control_hz="${1#*=}"
      shift
      ;;
    --seconds)
      seconds="${2:?--seconds requires a value}"
      seconds_override="1"
      shift 2
      ;;
    --seconds=*)
      seconds="${1#*=}"
      seconds_override="1"
      shift
      ;;
    --log-format)
      log_format="${2:?--log-format requires a value}"
      shift 2
      ;;
    --log-format=*)
      log_format="${1#*=}"
      shift
      ;;
    --log-prefix)
      log_prefix="${2:?--log-prefix requires a value}"
      shift 2
      ;;
    --log-prefix=*)
      log_prefix="${1#*=}"
      shift
      ;;
    --log-dir)
      log_dir="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --log-dir=*)
      log_dir="${1#*=}"
      shift
      ;;
    --log-every-n)
      log_every_n="${2:?--log-every-n requires a value}"
      shift 2
      ;;
    --log-every-n=*)
      log_every_n="${1#*=}"
      shift
      ;;
    --no-log)
      no_log="1"
      shift
      ;;
    --skip-model-check)
      skip_model_check="1"
      shift
      ;;
    --dry-run-only)
      dry_run_only="1"
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

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Validate and show the plan before executing anything.
python -m soridormi_runtime.skill_execution "${skill}" --profile "${profile}" --args "${args}"

eval "$(python -m soridormi_runtime.skill_execution "${skill}" --profile "${profile}" --args "${args}" --shell-env)"

if [ -z "${log_prefix}" ]; then
  log_prefix="skill_${skill}"
fi

if [ -z "${steps}" ]; then
  steps="$(python - "${SORIDORMI_SKILL_DURATION_SECONDS}" "${control_hz}" <<'PY'
import math
import sys

duration = float(sys.argv[1])
hz = float(sys.argv[2])
if duration <= 0:
    raise SystemExit("skill duration must be positive")
if hz <= 0:
    raise SystemExit("control frequency must be positive")
print(max(1, int(math.ceil(duration * hz))))
PY
)"
fi

# Validate numeric rollout limits early so mistakes fail before launching Docker.
validate_args=("${steps}" "${control_hz}")
if [ "${seconds_override}" = "1" ]; then
  validate_args+=("${seconds}")
fi
python -c '
import sys

steps = int(sys.argv[1])
control_hz = float(sys.argv[2])
if steps <= 0:
    raise SystemExit("--steps must be positive")
if control_hz <= 0:
    raise SystemExit("--control-hz must be positive")
if len(sys.argv) > 3 and float(sys.argv[3]) <= 0:
    raise SystemExit("--seconds must be positive")
' "${validate_args[@]}"

echo ""
echo "Soridormi skill MuJoCo execution"
echo "=================================="
echo "Skill: ${skill}"
echo "Profile: ${profile}"
echo "Command overrides: x=${SORIDORMI_COMMAND_X_OVERRIDE} y=${SORIDORMI_COMMAND_Y_OVERRIDE} yaw=${SORIDORMI_COMMAND_YAW_OVERRIDE}"
echo "Skill duration seconds: ${SORIDORMI_SKILL_DURATION_SECONDS}"
echo "Control Hz: ${control_hz}"
if [ "${steps_override}" = "1" ]; then
  echo "Rollout steps: ${steps} (user override)"
else
  echo "Rollout steps: ${steps} (derived from skill duration)"
fi
if [ "${seconds_override}" = "1" ]; then
  echo "Wall-clock seconds cutoff: ${seconds} (user override)"
else
  echo "Wall-clock seconds cutoff: disabled"
fi
echo "This assumes MuJoCo is already running with:"
echo "  ./scripts/run_sim_server.sh --backend mujoco --profile ${profile} --viewer --follow-camera"

if [ "${dry_run_only}" = "1" ]; then
  echo "Dry-run only; not launching runtime."
  exit 0
fi

smoke_args=("${profile}" --steps "${steps}" --log-prefix "${log_prefix}" --log-dir "${log_dir}")
if [ "${seconds_override}" = "1" ]; then
  smoke_args+=(--seconds "${seconds}")
fi
if [ -n "${log_format}" ]; then
  smoke_args+=(--log-format "${log_format}")
fi
if [ -n "${log_every_n}" ]; then
  smoke_args+=(--log-every-n "${log_every_n}")
fi
if [ "${no_log}" = "1" ]; then
  smoke_args+=(--no-log)
fi
if [ "${skip_model_check}" = "1" ]; then
  smoke_args+=(--skip-model-check)
fi

export SORIDORMI_COMMAND_X_OVERRIDE
export SORIDORMI_COMMAND_Y_OVERRIDE
export SORIDORMI_COMMAND_YAW_OVERRIDE
export SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE

./scripts/run_policy_rollout_smoke.sh "${smoke_args[@]}"
