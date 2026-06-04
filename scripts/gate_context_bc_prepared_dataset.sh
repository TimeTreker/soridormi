#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/gate_context_bc_prepared_dataset.sh PREPARED_MANIFEST_OR_DIR [options]

Validate a prepared context BC train/val/test dataset before BC training. The
prepared manifest and split JSONL files must be non-empty, contract-valid, and
must not leak rollout groups across splits.

Options:
  --contract PATH                           BC contract JSON path
  --scenario-manifest PATH                  Scenario manifest path
  --require-scenario SCENARIO               Required scenario id; repeat or comma-separate
  --require-ready-locomotion                Require all registry-ready locomotion scenarios
  --min-samples-per-required-scenario N     Minimum samples for every required scenario (default: 1)
  --min-train-samples N                     Minimum train samples (default: 1)
  --min-val-samples N                       Minimum validation samples (default: 1)
  --min-test-samples N                      Minimum test samples (default: 1)
  --allow-empty-val                         Do not require validation samples
  --allow-empty-test                        Do not require test samples
  --allow-manifest-failed                   Do not fail solely because prepared manifest ok=false
  --allow-group-leakage                     Do not fail when rollout groups appear in multiple splits
  --output PATH                             Optional Markdown report path
  --output-dir DIR                          Write prepared_context_gate_report.json/md into DIR
  --json                                    Print machine-readable JSON to stdout
  -h, --help                                Show this help

Examples:
  ./scripts/gate_context_bc_prepared_dataset.sh \
    /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
    --require-scenario flat_walk_varied_speed_v1 \
    --min-train-samples 1 --min-val-samples 1 --min-test-samples 1 \
    --output-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1 \
    --json | python -m json.tool

  ./scripts/gate_context_bc_prepared_dataset.sh \
    /data/training_datasets/context_bc/prepared/pre_bc/prepared_manifest.json \
    --require-ready-locomotion \
    --min-samples-per-required-scenario 1000
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
    --contract|--scenario-manifest|--output|--output-dir)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a path argument" >&2
        exit 2
      fi
      container_args+=("${opt}" "$(to_container_path "$2")")
      shift 2
      ;;
    --contract=*|--scenario-manifest=*|--output=*|--output-dir=*)
      opt="${1%%=*}"
      value="${1#*=}"
      container_args+=("${opt}=$(to_container_path "${value}")")
      shift
      ;;
    --require-scenario|--min-samples-per-required-scenario|--min-train-samples|--min-val-samples|--min-test-samples)
      opt="$1"
      if [ "$#" -lt 2 ]; then
        echo "error: ${opt} requires a value" >&2
        exit 2
      fi
      container_args+=("${opt}" "$2")
      shift 2
      ;;
    --require-scenario=*|--min-samples-per-required-scenario=*|--min-train-samples=*|--min-val-samples=*|--min-test-samples=*)
      container_args+=("$1")
      shift
      ;;
    --require-ready-locomotion|--allow-empty-val|--allow-empty-test|--allow-manifest-failed|--allow-group-leakage|--json)
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
    python -m soridormi_runtime.context_bc_prepared_gate "$@"
  ' _ "${container_args[@]}"
