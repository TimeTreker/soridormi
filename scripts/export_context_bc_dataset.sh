#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/export_context_bc_dataset.sh INPUT [INPUT ...] [options]

Convert legacy/scenario-aware Soridormi teacher JSONL rows into the context-conditioned
BC contract rows:

  robot_state + desired_command + task_context + environment_context + short_history -> action_14d

Inputs may be JSONL files, directories containing JSONL files, or prepared_manifest.json files.

Options:
  --output PATH             Output context BC JSONL path (default: /data/training_datasets/context_bc/context_bc_dataset.jsonl)
  --manifest PATH           Optional output manifest path
  --contract PATH           BC contract JSON path
  --scenario-manifest PATH  Scenario manifest JSON path
  --no-short-history        Do not add bounded short_history fields
  --strict-context          Fail rows whose scenario/skill cannot be resolved from the scenario manifest
  --max-samples N           Read at most N source samples
  --report PATH             Optional Markdown report path
  --json                    Print machine-readable JSON to stdout
  -h, --help                Show this help

Examples:
  ./scripts/export_context_bc_dataset.sh \
    /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
    --output /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
    --report artifacts/training/context_bc/flat_walk_varied_speed_v1.md \
    --json | python -m json.tool

  ./scripts/validate_bc_training_contract.sh \
    --sample-jsonl /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
    --json | python -m json.tool
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
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
    --output|--manifest|--contract|--scenario-manifest|--report)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --output=*|--manifest=*|--contract=*|--scenario-manifest=*|--report=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --no-short-history|--strict-context|--json)
      container_args+=("$1")
      shift
      ;;
    --max-samples)
      if [ "$#" -lt 2 ]; then
        echo "error: --max-samples requires an integer argument" >&2
        exit 2
      fi
      container_args+=("$1" "$2")
      shift 2
      ;;
    --max-samples=*)
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

if [ "${#container_args[@]}" -eq 0 ]; then
  echo "error: at least one input path is required" >&2
  usage >&2
  exit 2
fi

# Override the CUDA image entrypoint so --json stdout remains parseable.
docker compose -f compose.sim.yaml run --rm \
  --entrypoint bash \
  -v "$(pwd):/host_repo" \
  runtime -lc '
    set -euo pipefail
    cd /host_repo
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.context_bc_dataset_export "$@"
  ' _ "${container_args[@]}"
