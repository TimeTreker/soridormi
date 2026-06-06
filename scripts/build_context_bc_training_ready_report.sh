#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/build_context_bc_training_ready_report.sh PREPARED_MANIFEST [options]

Bundle the scenario coverage gate, prepared dataset gate, contract, hashes, and
recommended context BC train commands into a training-ready report.

Options:
  --scenario-gate PATH      dataset_scenario_gate_summary.json path (required)
  --prepared-gate PATH      prepared_context_gate_report.json path (required)
  --contract PATH           BC contract JSON path
  --profile-name NAME       Candidate profile name for the neural command
  --linear-output-dir DIR   Linear BC output directory
  --neural-output-dir DIR   Neural BC output directory
  --input-mode MODE         Policy input mode (default: context_stage1_command)
  --output PATH             Optional Markdown report path
  --output-dir DIR          Write training_ready_manifest.json/md into DIR
  --json                    Print machine-readable JSON to stdout
  -h, --help                Show this help

Example:
  ./scripts/build_context_bc_training_ready_report.sh \
    /data/training_datasets/context_bc/prepared/context_stage1_three_scenario_10ep/prepared_manifest.json \
    --scenario-gate artifacts/dataset_coverage/context_stage1_three_scenario_10ep/dataset_scenario_gate_summary.json \
    --prepared-gate artifacts/training/context_bc/prepared_gate/context_stage1_three_scenario_10ep/prepared_context_gate_report.json \
    --profile-name context_stage1_candidate \
    --output-dir artifacts/training/context_bc/training_ready/context_stage1_three_scenario_10ep \
    --json | python -m json.tool
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
    --scenario-gate|--prepared-gate|--contract|--linear-output-dir|--neural-output-dir|--output|--output-dir)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --scenario-gate=*|--prepared-gate=*|--contract=*|--linear-output-dir=*|--neural-output-dir=*|--output=*|--output-dir=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --profile-name|--input-mode)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a value" >&2
        exit 2
      fi
      container_args+=("${opt}" "$2")
      shift 2
      ;;
    --profile-name=*|--input-mode=*)
      container_args+=("$1")
      shift
      ;;
    --json)
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
    cd /host_repo
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.context_bc_training_ready "$@"
  ' _ "${container_args[@]}"
