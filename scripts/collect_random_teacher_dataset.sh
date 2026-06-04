#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/collect_random_teacher_dataset.sh [options]

Collect behavior-cloning samples by rolling out the teacher policy under random
piecewise velocity commands. Command changes are ramped by default so the
dataset covers continuous speed transitions instead of only abrupt jumps. The
collector owns its MuJoCo collection lifecycle; do not start a second sim server
for the same collection run.

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

Collector ownership:
  This wrapper owns the random-teacher collection run. Do not start a separate
  ./scripts/run_sim_server.sh for this command; use --viewer here when visual
  inspection is needed.
USAGE
}

status() {
  printf '%s\n' "$*" >&2
}

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

sim_backend="mujoco"
viewer_enabled="0"
json_output="0"
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

status "Soridormi random-command teacher collection"
status "=========================================="
status "Expected sim backend: ${sim_backend}"
status "MuJoCo viewer hint: ${viewer_enabled}"
status ""
status "Collector-owned sim lifecycle: do not start a second run_sim_server.sh for this collection."
status "Use --viewer on this command when visual inspection is needed."

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

translated_args=()
soridormi_translate_container_data_args translated_args "${collector_args[@]}"

export SORIDORMI_SIM_BACKEND="${sim_backend}"
export SORIDORMI_MUJOCO_VIEWER="${viewer_enabled}"

run_collector() {
  docker compose -f compose.sim.yaml run --rm \
    --entrypoint bash \
    runtime -lc '
      set -euo pipefail
      source /opt/venvs/runtime/bin/activate
      python -m soridormi_runtime.random_teacher_dataset_collect "$@"
    ' _ "${translated_args[@]}"
}

if [ "${json_output}" != "1" ]; then
  run_collector
  exit $?
fi

# In JSON mode, stdout is reserved for one machine-readable payload.  Docker
# Compose status messages, container stderr, and CUDA banner noise must not be
# allowed to corrupt the JSON stream.  The CUDA banner is avoided by overriding
# the image entrypoint above; any remaining noise is captured and forwarded to
# stderr after the command finishes.
tmp_dir="$(mktemp -d)"
stdout_path="${tmp_dir}/collector_stdout.json"
stderr_path="${tmp_dir}/collector_stderr.txt"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

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
    "schema_version": "m9.random_teacher_wrapper_failure.v1",
    "docker_exit_code": rc,
    "errors": errors,
    "stdout_preview": stdout_text[:2000],
    "stderr_preview": stderr_text[:2000],
}
print(json.dumps(payload, indent=2, sort_keys=True))
' "$collector_rc" "$stdout_path" "$stderr_path"
exit 1
