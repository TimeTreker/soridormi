#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/report_scripted_social_readiness.sh [options]

Generate a machine-readable readiness report for promoting scripted social
skills from available_sim_experimental to available_sim. By default this runs
safe dry-run acceptance inside the runtime Docker container. Attach a live
MuJoCo acceptance JSON and pass --require-live before using the report as a
promotion gate.

Options:
  --skill SKILL              Skill id to include; repeatable.
  --live-acceptance-json P   JSON from evaluate_scripted_social_skills.sh --execute --json.
  --require-live             Require live MuJoCo acceptance to pass overall.
  --output-dir DIR           Write JSON and Markdown reports.
  --json                     Print machine-readable JSON.
  --control-hz HZ            Dry-run control frequency (default: 50).
  --transition-fraction N    Dry-run transition fraction (default: 0.40).
  --max-head-velocity-radps N
                             Dry-run head speed limit in rad/s (default: 0.35; 0 disables).
  --no-auto-stretch-duration Do not extend gestures to satisfy the speed limit.
  -h, --help                 Show this help.

Recommended promotion workflow:
  ./scripts/evaluate_scripted_social_skills.sh --execute --backend mujoco --require-observed --json \
    > artifacts/scripted_social/live_acceptance.json

  ./scripts/report_scripted_social_readiness.sh \
    --live-acceptance-json artifacts/scripted_social/live_acceptance.json \
    --require-live \
    --output-dir artifacts/scripted_social/readiness \
    --json
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

# The runtime service mounts src/configs/tests/scripts/data, but host shell
# redirection usually writes promotion artifacts under ./artifacts.  Mount the
# full repository at /host_repo for this read/write reporting tool, then rewrite
# repo-relative path arguments so they are visible inside the container.
container_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --live-acceptance-json|--output-dir|--manifest)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      path="$2"
      shift 2
      case "${path}" in
        /app/*|/data/*|/host_repo/*)
          container_path="${path}"
          ;;
        /*)
          case "${path}" in
            "$(pwd)"/*)
              container_path="/host_repo/${path#"$(pwd)"/}"
              ;;
            *)
              echo "error: absolute path is outside this repo and is not mounted in the runtime container: ${path}" >&2
              echo "       use a repo-relative path such as artifacts/scripted_social/live_acceptance.json or a /data path" >&2
              exit 2
              ;;
          esac
          ;;
        *)
          container_path="/host_repo/${path#./}"
          ;;
      esac
      container_args+=("${opt}" "${container_path}")
      ;;
    *)
      container_args+=("$1")
      shift
      ;;
  esac
done

# Override the CUDA image entrypoint so --json stdout is not prefixed by the
# NVIDIA container banner when callers pipe to python -m json.tool.
docker compose -f compose.sim.yaml run --rm   --entrypoint bash   -v "$(pwd):/host_repo"   runtime -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.scripted_social_readiness "$@"
  ' _ "${container_args[@]}"
