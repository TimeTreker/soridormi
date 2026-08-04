#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/collect_random_teacher_dataset.sh [options]

Collect behavior-cloning samples by rolling out the teacher policy under random
piecewise velocity commands. Command changes are ramped by default so the
dataset covers continuous speed transitions instead of only abrupt jumps. The
collector owns its MuJoCo collection lifecycle by default; do not start a second
sim server for the same collection run.

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
  --reset-attempts N             retry transient simulator reset failures (default: 5)
  --reset-retry-sleep S          seconds between reset attempts (default: 0.25)
  --seed N                       deterministic random seed
  --scenario ID                  use scenario curriculum command ranges/context
  --list-scenarios               list configured scenarios and exit
  --allow-planned-scenario       allow planned metadata-only scenario rows
  --json                         print machine-readable JSON to stdout

Negative ranges may be passed either as separate values or with '=':
  --vx-range -0.03,0.15
  --vx-range=-0.03,0.15
  --backend NAME                 expected sim backend; default: mujoco
  --viewer                       request viewer for the collector-owned MuJoCo run
  --no-viewer                    request headless collection; default
  --follow-camera                follow robot in the collector-owned viewer; default
  --no-follow-camera             disable collector-owned viewer follow camera
  --external-sim                 do not start a sim server; connect to an existing one
  --sim-start-timeout S          seconds to wait for collector-owned sim (default: 45)

Collector ownership:
  This wrapper owns the random-teacher collection run by default. Do not start a separate
  ./scripts/run_sim_server.sh for this command; use --viewer here when
  visual inspection is needed. Use --external-sim only for advanced debugging
  when you intentionally want to connect to a server that is already running.
USAGE
}

status() {
  printf '%s\n' "$*" >&2
}

is_true() {
  case "${1,,}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

sim_backend="${SORIDORMI_SIM_BACKEND:-mujoco}"
viewer_enabled="${SORIDORMI_MUJOCO_VIEWER:-0}"
follow_camera="${SORIDORMI_MUJOCO_FOLLOW_CAMERA:-1}"
json_output="0"
list_scenarios="0"
external_sim="${SORIDORMI_RANDOM_TEACHER_EXTERNAL_SIM:-0}"
sim_start_timeout="${SORIDORMI_RANDOM_TEACHER_SIM_START_TIMEOUT:-45}"
profile="${SORIDORMI_POLICY_PROFILE:-open_duck_forward}"
sim_host="${SIM_HOST:-127.0.0.1}"
sim_port="${SIM_PORT:-5555}"
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
    --follow-camera)
      follow_camera="1"
      shift
      ;;
    --no-follow-camera)
      follow_camera="0"
      shift
      ;;
    --external-sim)
      external_sim="1"
      shift
      ;;
    --sim-start-timeout)
      sim_start_timeout="${2:?--sim-start-timeout requires a value}"
      shift 2
      ;;
    --sim-start-timeout=*)
      sim_start_timeout="${1#*=}"
      shift
      ;;
    --profile)
      profile="${2:?--profile requires a value}"
      collector_args+=("$1" "$2")
      shift 2
      ;;
    --profile=*)
      profile="${1#*=}"
      collector_args+=("$1")
      shift
      ;;
    --host)
      sim_host="${2:?--host requires a value}"
      collector_args+=("$1" "$2")
      shift 2
      ;;
    --host=*)
      sim_host="${1#*=}"
      collector_args+=("$1")
      shift
      ;;
    --port)
      sim_port="${2:?--port requires a value}"
      collector_args+=("$1" "$2")
      shift 2
      ;;
    --port=*)
      sim_port="${1#*=}"
      collector_args+=("$1")
      shift
      ;;
    --list-scenarios)
      list_scenarios="1"
      collector_args+=("$1")
      shift
      ;;
    --json)
      json_output="1"
      collector_args+=("$1")
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
  status "Warning: random teacher collection is intended for MuJoCo; requested backend: ${sim_backend}"
fi

# The Python collector owns transient reset retries. Append conservative defaults
# unless the caller provided explicit values.
if [[ " ${collector_args[*]} " != *" --reset-attempts "* && " ${collector_args[*]} " != *" --reset-attempts="* ]]; then
  collector_args+=("--reset-attempts" "5")
fi
if [[ " ${collector_args[*]} " != *" --reset-retry-sleep "* && " ${collector_args[*]} " != *" --reset-retry-sleep="* ]]; then
  collector_args+=("--reset-retry-sleep" "0.25")
fi

status "Soridormi random-command teacher collection"
status "=========================================="
status "Expected sim backend: ${sim_backend}"
status "Teacher/runtime profile: ${profile}"
status "MuJoCo viewer hint: ${viewer_enabled}"
status "MuJoCo follow camera: ${follow_camera}"
status "External sim: ${external_sim}"
status ""
if is_true "${list_scenarios}"; then
  status "List-scenarios mode: no simulator will be started."
elif is_true "${external_sim}"; then
  status "External-sim mode: connect to an already-running simulator."
else
  status "Collector-owned sim lifecycle: do not start a second run_sim_server.sh for this collection."
  status "Use --viewer on this command when visual inspection is needed."
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

translated_args=()
soridormi_translate_container_data_args translated_args "${collector_args[@]}"

export SORIDORMI_SIM_BACKEND="${sim_backend}"
export SORIDORMI_MUJOCO_VIEWER="${viewer_enabled}"
export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${follow_camera}"

owned_sim_pid=""
owned_sim_container=""
owned_sim_log=""
XHOST_ADDED=0

cleanup_xhost() {
  if [ "${XHOST_ADDED}" = "1" ]; then
    status "Removing local Docker X11 access..."
    xhost -local:docker >/dev/null 2>&1 || true
  fi
}

enable_xhost_if_needed() {
  if ! is_true "${viewer_enabled}"; then
    return 0
  fi
  if [ -z "${DISPLAY:-}" ]; then
    status "Warning: DISPLAY is not set. MuJoCo viewer may not open."
    return 0
  fi
  if ! command -v xhost >/dev/null 2>&1; then
    status "Warning: xhost not found. Install it with: sudo apt install x11-xserver-utils"
    return 0
  fi
  status "Allowing local Docker containers to access X11..."
  xhost +local:docker >/dev/null || true
  XHOST_ADDED=1
}

stop_owned_sim() {
  if [ -n "${owned_sim_container}" ]; then
    docker stop "${owned_sim_container}" >/dev/null 2>&1 || true
  fi
  if [ -n "${owned_sim_pid}" ]; then
    wait "${owned_sim_pid}" >/dev/null 2>&1 || true
  fi
  cleanup_xhost
}

cleanup_owned_sim() {
  local rc=$?
  stop_owned_sim
  exit "${rc}"
}

wait_for_sim_port() {
  local timeout_s="$1"
  local wait_host="${sim_host}"
  if [ "${wait_host}" = "0.0.0.0" ]; then
    wait_host="127.0.0.1"
  fi
  local deadline=$((SECONDS + timeout_s))
  while [ "${SECONDS}" -le "${deadline}" ]; do
    if (exec 3<>"/dev/tcp/${wait_host}/${sim_port}") >/dev/null 2>&1; then
      exec 3<&- || true
      exec 3>&- || true
      return 0
    fi
    if [ -n "${owned_sim_pid}" ] && ! kill -0 "${owned_sim_pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 0.5
  done
  return 1
}

start_owned_sim_server() {
  enable_xhost_if_needed
  owned_sim_container="soridormi-random-teacher-sim-$RANDOM-$$"
  owned_sim_log="$(mktemp -t soridormi-random-teacher-sim.XXXXXX.log)"
  status "Starting collector-owned simulator container: ${owned_sim_container}"
  status "Simulator startup log: ${owned_sim_log}"

  docker compose -f compose.sim.yaml run --rm \
    --name "${owned_sim_container}" \
    -e SORIDORMI_SIM_POLICY_PROFILE="${profile}" \
    -e SORIDORMI_SIM_BACKEND_OVERRIDE="${sim_backend}" \
    -e SORIDORMI_MUJOCO_VIEWER_OVERRIDE="${viewer_enabled}" \
    -e SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE="${follow_camera}" \
    sim bash -lc '
      set -euo pipefail
      source /opt/venvs/sim/bin/activate

      if [ -n "${SORIDORMI_SIM_POLICY_PROFILE:-}" ]; then
        echo "Resolving simulator compatibility from policy profile: ${SORIDORMI_SIM_POLICY_PROFILE}"
        eval "$(python -m soridormi_runtime.policy_profiles "${SORIDORMI_SIM_POLICY_PROFILE}" --shell)"
      fi

      export SORIDORMI_SIM_BACKEND="${SORIDORMI_SIM_BACKEND_OVERRIDE:-mujoco}"
      export SORIDORMI_MUJOCO_VIEWER="${SORIDORMI_MUJOCO_VIEWER_OVERRIDE:-0}"
      export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE:-1}"
      echo "Collector-owned sim backend: ${SORIDORMI_SIM_BACKEND}"
      echo "Collector-owned MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"
      echo "Collector-owned MuJoCo follow camera: ${SORIDORMI_MUJOCO_FOLLOW_CAMERA}"
      echo "Collector-owned MuJoCo profile flags: home_keyframe=${SORIDORMI_MUJOCO_USE_HOME_KEYFRAME:-0} official_reset=${SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE:-0} official_sensor=${SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE:-0} official_contact=${SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE:-0}"
      python -m soridormi_sim.mujoco_server
    ' >"${owned_sim_log}" 2>&1 &
  owned_sim_pid=$!

  if ! wait_for_sim_port "${sim_start_timeout}"; then
    status "Collector-owned simulator did not become ready within ${sim_start_timeout}s."
    status "Last simulator log lines:"
    tail -120 "${owned_sim_log}" >&2 || true
    return 1
  fi
  status "Collector-owned simulator is listening on ${sim_host}:${sim_port}."
}

run_collector() {
  docker compose -f compose.sim.yaml run --rm \
    --entrypoint bash \
    runtime -lc '
      set -euo pipefail
      source /opt/venvs/runtime/bin/activate
      python -m soridormi_runtime.random_teacher_dataset_collect "$@"
    ' _ "${translated_args[@]}"
}

emit_startup_failure_json() {
  python -c '
import json
payload = {
    "ok": False,
    "schema_version": "soridormi.random_teacher_wrapper_failure.v2",
    "errors": ["collector-owned simulator failed to start before dataset collection"],
}
print(json.dumps(payload, indent=2, sort_keys=True))
'
}

if ! is_true "${external_sim}" && ! is_true "${list_scenarios}"; then
  trap cleanup_owned_sim EXIT INT TERM
  if ! start_owned_sim_server; then
    if [ "${json_output}" = "1" ]; then
      emit_startup_failure_json
    fi
    exit 1
  fi
fi

if [ "${json_output}" != "1" ]; then
  run_collector
  exit $?
fi

# In JSON mode, stdout is reserved for one machine-readable payload.  Docker
# Compose status messages, container stderr, and CUDA banner noise must not be
# allowed to corrupt the JSON stream.  The CUDA banner is avoided by overriding
# the image entrypoint in run_collector.
tmp_dir="$(mktemp -d)"
stdout_path="${tmp_dir}/collector_stdout.json"
stderr_path="${tmp_dir}/collector_stderr.txt"
cleanup_tmp() {
  rm -rf "${tmp_dir}"
}
cleanup_json_mode() {
  local rc=$?
  cleanup_tmp
  stop_owned_sim
  exit "${rc}"
}
trap cleanup_json_mode EXIT INT TERM

collector_rc=0
if run_collector >"${stdout_path}" 2>"${stderr_path}"; then
  collector_rc=0
else
  collector_rc=$?
fi

if [ -s "${stderr_path}" ]; then
  cat "${stderr_path}" >&2
fi

if python -m json.tool "${stdout_path}" >/dev/null 2>&1; then
  cat "${stdout_path}"
  if [ "${collector_rc}" -eq 0 ]; then
    status "Random teacher dataset collection finished successfully."
  else
    status "Random teacher dataset collection finished with errors; inspect the JSON payload."
  fi
  exit "${collector_rc}"
fi

status "Random teacher dataset collection failed before producing valid JSON."
python -c '
import json
import sys
from pathlib import Path

rc = int(sys.argv[1])
stdout_path = Path(sys.argv[2])
stderr_path = Path(sys.argv[3])
stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
errors = ["collector did not produce valid JSON on stdout"]
if rc:
    errors.append(f"docker collector exited with status {rc}")
if not stdout_text.strip():
    errors.append("collector stdout was empty")
payload = {
    "ok": False,
    "schema_version": "soridormi.random_teacher_wrapper_failure.v1",
    "docker_exit_code": rc,
    "errors": errors,
    "stdout_preview": stdout_text[:2000],
    "stderr_preview": stderr_text[:2000],
}
print(json.dumps(payload, indent=2, sort_keys=True))
' "$collector_rc" "$stdout_path" "$stderr_path"
exit 1
