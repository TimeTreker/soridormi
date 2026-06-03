#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_bc_training_contract.sh [options]

Validate the versioned Soridormi context-conditioned BC training contract and,
optionally, validate a JSONL dataset against that contract.

Options:
  --contract PATH       Contract JSON path (default: configs/training/open_duck_mini_v2_context_bc_contract_v1.json)
  --sample-jsonl PATH   Optional dataset JSONL to validate against the contract
  --allow-legacy        Accept soridormi.policy_supervision.v1 samples for Stage 1 only
  --output PATH         Optional Markdown report path
  --json                Print machine-readable JSON to stdout
  -h, --help            Show this help

Examples:
  ./scripts/validate_bc_training_contract.sh --json | python -m json.tool

  ./scripts/validate_bc_training_contract.sh \
    --sample-jsonl /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
    --allow-legacy \
    --output artifacts/training_contract/contract_report.md
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
    --contract|--sample-jsonl|--output)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --contract=*|--sample-jsonl=*|--output=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --allow-legacy|--json)
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
      echo "Unexpected positional argument: $1" >&2
      usage >&2
      exit 2
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
    python -m soridormi_runtime.bc_training_contract "$@"
  ' _ "${container_args[@]}"
