#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/prepare_context_bc_dataset.sh INPUT [INPUT ...] [options]

Prepare context-conditioned BC JSONL rows into train/val/test splits while
preserving rollout/scenario grouping to avoid adjacent-timestep leakage.

Options:
  --output-dir DIR             Output prepared dataset directory (default: /data/training_datasets/context_bc/prepared)
  --contract PATH              BC contract JSON path
  --train-ratio N              Train split ratio (default: 0.8)
  --val-ratio N                Validation split ratio (default: 0.1)
  --test-ratio N               Test split ratio (default: 0.1)
  --seed N                     Deterministic hash split seed (default: 0)
  --no-shuffle                 Keep first-seen group order instead of hash shuffling
  --split-group-field FIELD    Leakage boundary; default: rollout_id
  --no-stratify-by-scenario    Do not split groups separately inside each scenario
  --skip-invalid               Skip invalid rows without failing the result
  --max-samples N              Read at most N source samples
  --report PATH                Optional Markdown report path
  --json                       Print machine-readable JSON to stdout
  -h, --help                   Show this help

Examples:
  ./scripts/prepare_context_bc_dataset.sh \
    /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
    --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
    --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
    --json | python -m json.tool

  ./scripts/validate_bc_training_contract.sh \
    --sample-jsonl /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/train.jsonl \
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
    --output-dir|--contract|--report)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --output-dir=*|--contract=*|--report=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --train-ratio|--val-ratio|--test-ratio|--seed|--split-group-field|--max-samples)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a value" >&2
        exit 2
      fi
      container_args+=("${opt}" "$2")
      shift 2
      ;;
    --train-ratio=*|--val-ratio=*|--test-ratio=*|--seed=*|--split-group-field=*|--max-samples=*)
      container_args+=("$1")
      shift
      ;;
    --no-shuffle|--no-stratify-by-scenario|--skip-invalid|--json)
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
    python -m soridormi_runtime.context_bc_dataset_prepare "$@"
  ' _ "${container_args[@]}"
