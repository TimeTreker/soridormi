#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/accept_policy_rollout.sh LOG [options]

Evaluate a bounded Soridormi policy rollout log and write acceptance artifacts.

Options:
  --profile NAME              Store profile name in the report.
  --output-dir DIR            Output directory for rollout_acceptance.json/report.md.
  --min-policy-records N      Require at least N policy records.
  --min-robot-duration S      Require at least S robot-time seconds.
  --max-reset-count N         Allow at most N robot-time resets (default: 0).
  --max-action-abs X          Require policy action abs_max <= X (default: 5.0).
  --disable-action-bound      Do not apply the action abs_max bound.
  --max-joint-abs X           Require joint position abs_max <= X when joint logs exist.
  --min-forward-x M           Require base forward displacement >= M meters.
  --max-lateral-abs M         Require abs(base lateral displacement) <= M meters.
  --json                      Print result JSON.
  -h, --help                  Show this help.

Host data/... paths are translated to /data/... inside the runtime container.
EOF
}

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

source ./scripts/lib/container_paths.sh
translated_args=()
soridormi_translate_container_data_args translated_args "$@"

# Output artifacts are usually written under /data. Allow the report command to
# run in the same dependency-complete runtime container used by other policy tools.
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
set -euo pipefail
source /opt/venvs/runtime/bin/activate
python -m soridormi_runtime.rollout_acceptance "$@"
' bash "${translated_args[@]}"
