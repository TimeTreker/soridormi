#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/run_residual_finetune_comparison.sh RESIDUAL_PROFILE [options]

Run a bounded teacher rollout, run a bounded residual/fine-tuned rollout, then
compare the two logs. The simulator server must already be running.

Options:
  --teacher-profile PROFILE    Teacher profile to compare against (default: open_duck_forward).
  --steps N                    Steps for each rollout (default: 1000).
  --require-provider NAME      Required ONNX Runtime provider. May repeat.
  --skip-model-check           Skip model preflight checks.
  --compare-arg ARG            Extra argument forwarded to compare_policy_rollouts.sh. May repeat.
  -h, --help                   Show this help.
EOF
}

residual_profile="${1:-}"
if [ -z "${residual_profile}" ] || [ "${residual_profile}" = "-h" ] || [ "${residual_profile}" = "--help" ]; then
  usage
  exit 0
fi
shift || true

teacher_profile="open_duck_forward"
steps="1000"
skip_model_check="0"
require_provider_args=()
compare_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --teacher-profile)
      teacher_profile="${2:?--teacher-profile requires a value}"
      shift 2
      ;;
    --steps)
      steps="${2:?--steps requires a value}"
      shift 2
      ;;
    --require-provider)
      require_provider_args+=(--require-provider "${2:?--require-provider requires a value}")
      shift 2
      ;;
    --skip-model-check)
      skip_model_check="1"
      shift
      ;;
    --compare-arg)
      compare_args+=("${2:?--compare-arg requires a value}")
      shift 2
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

if [ "${skip_model_check}" != "1" ]; then
  ./scripts/check_policy_model.sh --profile "${teacher_profile}" "${require_provider_args[@]}"
  ./scripts/check_policy_model.sh --profile "${residual_profile}" "${require_provider_args[@]}"
fi

echo "Running teacher rollout: ${teacher_profile}"
./scripts/run_policy_rollout_smoke.sh "${teacher_profile}" --steps "${steps}" --skip-model-check
teacher_log="$(ls -t "data/logs/policy_${teacher_profile}_"*.mcap 2>/dev/null | head -1 || true)"
if [ -z "${teacher_log}" ]; then
  echo "Could not find teacher rollout log for ${teacher_profile}" >&2
  exit 1
fi

echo "Running residual rollout: ${residual_profile}"
./scripts/run_policy_rollout_smoke.sh "${residual_profile}" --steps "${steps}" --skip-model-check
candidate_log="$(ls -t "data/logs/policy_${residual_profile}_"*.mcap 2>/dev/null | head -1 || true)"
if [ -z "${candidate_log}" ]; then
  echo "Could not find residual rollout log for ${residual_profile}" >&2
  exit 1
fi

./scripts/compare_policy_rollouts.sh "${teacher_log}" "${candidate_log}" "${compare_args[@]}"
