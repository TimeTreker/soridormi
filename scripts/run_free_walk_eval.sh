#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_free_walk_eval.sh CANDIDATE_PROFILE [options]

Run the M6A command-conditioned free-walk evaluation suite against a candidate
policy profile. This wraps the existing teacher-vs-candidate command-grid
comparison flow with conservative free-walk defaults.

Options:
  --suite PATH                  Free-walk suite YAML
                                default: configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
  --output-dir PATH             Output dir
                                default: data/free_walk_evals/<candidate>
  --require-provider NAME       Require ONNX provider during profile checks. May repeat.
  --force                       Overwrite generated profiles/artifacts.
  --dry-run                     Validate/generate planned rollouts only; does not call MuJoCo rollouts.

  Additional comparison thresholds are forwarded to run_command_grid_comparison.sh:
  --min-candidate-policy-records N
  --min-candidate-duration SEC
  --max-candidate-resets N
  --min-forward-ratio VALUE
  --disable-forward-ratio
  --min-speed-ratio VALUE
  --max-lateral-abs VALUE
  --max-lateral-ratio VALUE
  --disable-lateral-ratio
  --max-action-abs VALUE
  --disable-action-bound

Start the MuJoCo sim server in another terminal before running without --dry-run.
Use the teacher/candidate profile's simulator compatibility settings for walking parity, e.g.:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
USAGE
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi

candidate_profile="$1"
shift

suite="configs/teacher_suites/open_duck_free_walk_eval_v1.yaml"
output_dir="data/free_walk_evals/${candidate_profile}"
forwarded_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --suite)
      suite="${2:?--suite requires a value}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --require-provider|--min-candidate-policy-records|--min-candidate-duration|--max-candidate-resets|--min-forward-ratio|--min-speed-ratio|--max-lateral-abs|--max-lateral-ratio|--max-action-abs)
      forwarded_args+=("$1" "${2:?$1 requires a value}")
      shift 2
      ;;
    --force|--dry-run|--disable-forward-ratio|--disable-lateral-ratio|--disable-action-bound)
      forwarded_args+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PYTHONPATH=src python -m soridormi_runtime.free_walk_eval --suite "${suite}"

./scripts/run_command_grid_comparison.sh \
  "${candidate_profile}" \
  --suite "${suite}" \
  --output-dir "${output_dir}" \
  "${forwarded_args[@]}"
