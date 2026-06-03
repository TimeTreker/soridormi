#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/gate_dataset_scenario_coverage.sh DATASET_OR_PREPARED_DIR [more inputs...] [options]

Validate Soridormi BC/teacher dataset coverage against the scenario curriculum.
This is stricter than the descriptive coverage report: it fails when required
scenarios are missing, command ranges are too narrow, or structured context and
fall/stuck metadata are absent.

Options:
  --require-scenario SCENARIO       Required scenario id; repeat or comma-separate.
  --require-ready-locomotion        Require all registry-ready locomotion scenarios.
  --scenario-manifest PATH          Scenario manifest path.
  --output-dir DIR                  Write JSON/Markdown gate artifacts.
  --min-samples-per-scenario N      Minimum valid samples per required/present scenario (default: 1).
  --command-source SOURCE           applied_command, desired_command, or policy_command.
  --required-command FIELD          vx_mps, vy_mps, yaw_radps; repeat or comma-separate.
  --min-command-range-fraction N    Required fraction of manifest command span (default: 0.20).
  --max-failure-ratio N             Maximum failure/stuck/fall/terminated ratio (default: 0.50).
  --allow-any-failure-ratio         Do not gate on failure ratio.
  --no-require-ramp-alpha           Do not require command_ramp_alpha on every sample.
  --no-require-task-context         Do not require task_context on every sample.
  --no-require-environment-context  Do not require environment_context on every sample.
  --no-require-failure-flags        Do not require fall/stuck/termination metadata on every sample.
  --json                           Print machine-readable JSON to stdout.
  -h, --help                       Show this help.

Examples:
  ./scripts/gate_dataset_scenario_coverage.sh \
    /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
    --require-scenario flat_walk_varied_speed_v1 \
    --min-samples-per-scenario 300 \
    --output-dir artifacts/dataset_coverage/pre_bc \
    --json | python -m json.tool

  ./scripts/gate_dataset_scenario_coverage.sh \
    /data/training_datasets/prepared/pre_bc/prepared_manifest.json \
    --require-ready-locomotion \
    --min-samples-per-scenario 1000 \
    --min-command-range-fraction 0.35

Recommended MuJoCo-first collection flow:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
  ./scripts/collect_random_teacher_dataset.sh --backend mujoco --scenario flat_walk_varied_speed_v1 --profile open_duck_forward --episodes 4 --steps-per-episode 500 --command-ramp-steps 30 --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl
  ./scripts/gate_dataset_scenario_coverage.sh /data/training_datasets/flat_walk_varied_speed_v1.jsonl --require-scenario flat_walk_varied_speed_v1
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

to_container_path() {
  local path="$1"
  local repo_root
  repo_root="$(pwd)"
  case "${path}" in
    /app/*|/data|/data/*|/host_repo/*)
      printf '%s\n' "${path}"
      ;;
    data)
      printf '/data\n'
      ;;
    data/*)
      printf '/data/%s\n' "${path#data/}"
      ;;
    "${repo_root}")
      printf '/host_repo\n'
      ;;
    "${repo_root}"/*)
      printf '/host_repo/%s\n' "${path#"${repo_root}"/}"
      ;;
    /*)
      echo "error: absolute path is outside this repo and is not mounted in the runtime container: ${path}" >&2
      echo "       use a repo-relative path, data/... path, or /data path" >&2
      exit 2
      ;;
    *)
      printf '/host_repo/%s\n' "${path#./}"
      ;;
  esac
}

container_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --scenario-manifest|--output-dir)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --scenario-manifest=*|--output-dir=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --require-scenario|--min-samples-per-scenario|--command-source|--required-command|--min-command-range-fraction|--max-failure-ratio|--observation-size|--action-size)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a value" >&2
        exit 2
      fi
      container_args+=("${opt}" "$2")
      shift 2
      ;;
    --require-scenario=*|--min-samples-per-scenario=*|--command-source=*|--required-command=*|--min-command-range-fraction=*|--max-failure-ratio=*|--observation-size=*|--action-size=*)
      container_args+=("$1")
      shift
      ;;
    --require-ready-locomotion|--allow-any-failure-ratio|--no-require-ramp-alpha|--no-require-task-context|--no-require-environment-context|--no-require-failure-flags|--json)
      container_args+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      container_args+=("$(to_container_path "$1")")
      shift
      ;;
  esac
done

# Override the CUDA image entrypoint so --json stdout remains parseable.
docker compose -f compose.sim.yaml run --rm \
  --entrypoint bash \
  -v "$(pwd):/host_repo" \
  runtime -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.dataset_scenario_gate "$@"
  ' _ "${container_args[@]}"
