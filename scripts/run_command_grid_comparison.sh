#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/run_command_grid_comparison.sh CANDIDATE_PROFILE [options]

Run a teacher-vs-candidate rollout comparison over every scenario in a teacher
suite. This is the multi-command evaluation step for policies trained from the
teacher suite: stop, forward speeds, turning, curves, lateral commands, and
head/neck gesture cues.

Options:
  --suite PATH                       Teacher suite YAML
                                     default: configs/teacher_suites/open_duck_teacher_v1.yaml
  --output-dir PATH                  Output dir
                                     default: data/command_grids/<candidate>
  --require-provider NAME            Require ONNX provider during profile checks. May repeat.
  --force                            Overwrite generated suite/candidate profiles.
  --dry-run                          Generate profiles and print planned rollouts only.

  --min-candidate-policy-records N   Per-scenario rollout threshold. default: 1
  --min-candidate-duration SEC       Per-scenario rollout threshold. default: 0
  --max-candidate-resets N           Per-scenario rollout threshold. default: 0
  --min-forward-ratio VALUE          Per-scenario comparison threshold. default: 0.35
  --disable-forward-ratio            Disable forward-ratio threshold for all scenarios.
  --min-speed-ratio VALUE            Optional speed-ratio threshold.
  --max-lateral-abs VALUE            Optional absolute lateral threshold.
  --max-lateral-ratio VALUE          Lateral-ratio threshold. default: 3.0
  --disable-lateral-ratio            Disable lateral-ratio threshold.
  --max-action-abs VALUE             Action magnitude threshold. default: 5.0
  --disable-action-bound             Disable action magnitude threshold.
  -h, --help                         Show this help.

Start the MuJoCo sim server in another terminal first.
EOF
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi

candidate_profile="$1"
shift

suite="configs/teacher_suites/open_duck_teacher_v1.yaml"
output_dir="data/command_grids/${candidate_profile}"
force="0"
dry_run="0"

require_provider_args=()
compare_args=(
  --min-candidate-policy-records 1
  --min-candidate-duration 0
  --max-candidate-resets 0
  --min-forward-ratio 0.35
  --max-lateral-ratio 3.0
  --max-action-abs 5.0
)

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
    --require-provider)
      require_provider_args+=(--require-provider "${2:?--require-provider requires a value}")
      shift 2
      ;;
    --force)
      force="1"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    --min-candidate-policy-records|--min-candidate-duration|--max-candidate-resets|--min-forward-ratio|--min-speed-ratio|--max-lateral-abs|--max-lateral-ratio|--max-action-abs)
      compare_args+=("$1" "${2:?$1 requires a value}")
      shift 2
      ;;
    --disable-forward-ratio|--disable-lateral-ratio|--disable-action-bound)
      compare_args+=("$1")
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

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

SORIDORMI_REPO_ROOT="$(pwd)"
source scripts/lib/container_paths.sh

mkdir -p "${output_dir}"

teacher_dir="${output_dir%/}/teacher_suite"
candidate_dir="${output_dir%/}/candidate_grid"
comparisons_dir="${output_dir%/}/comparisons"

teacher_args=(--suite "${suite}" --output-dir "${teacher_dir}")
if [ "${force}" = "1" ]; then
  teacher_args+=(--force)
fi

./scripts/generate_teacher_suite.sh "${teacher_args[@]}"

teacher_manifest="${teacher_dir}/teacher_suite_manifest.json"
teacher_manifest_container="$(soridormi_to_container_data_path "${teacher_manifest}")"
candidate_dir_container="$(soridormi_to_container_data_path "${candidate_dir}")"

candidate_args=(
  "${candidate_profile}"
  --teacher-manifest "${teacher_manifest_container}"
  --output-dir "${candidate_dir_container}"
)

if [ "${force}" = "1" ]; then
  candidate_args+=(--force)
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.command_grid generate "$@"
' _ "${candidate_args[@]}"

candidate_manifest="${candidate_dir}/command_grid_manifest.json"
if [ ! -f "${candidate_manifest}" ]; then
  echo "Candidate command-grid manifest not found: ${candidate_manifest}" >&2
  exit 1
fi

echo
printf 'Command grid manifest: %s\n' "${candidate_manifest}"

mapfile -t rollout_lines < <(python - "${candidate_manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for scenario in payload.get("scenarios", []):
    seconds = scenario.get("seconds")
    print("\t".join([
        scenario["name"],
        scenario["teacher_profile_path"],
        scenario["candidate_profile_path"],
        str(scenario.get("steps") or 0),
        "" if seconds is None else str(seconds),
    ]))
PY
)

run_rollout_and_extract_log() {
  local tmp
  tmp="$(mktemp)"

  set +e
  ./scripts/run_policy_rollout_smoke.sh "$@" 2>&1 | tee "${tmp}" >&2
  local rollout_status=${PIPESTATUS[0]}
  set -e

  if [ "${rollout_status}" -ne 0 ]; then
    echo "Rollout failed: ./scripts/run_policy_rollout_smoke.sh $*" >&2
    echo "Recent rollout output:" >&2
    tail -n 80 "${tmp}" >&2 || true
    rm -f "${tmp}"
    return "${rollout_status}"
  fi

  local log_path
  log_path="$(
    python - "${tmp}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
text = text.replace("\r", "\n")

# Runtime output can print either:
#   Runtime log: /data/logs/foo.mcap
# or:
#   Runtime log:
#   /data/logs/foo.mcap
#
# So do not depend on the "Runtime log:" line format. Extract any MCAP log path.
matches = re.findall(r"(?:/data|data)/logs/[^\s`'\"]+\.mcap", text)
if matches:
    print(matches[-1])
PY
  )"

  if [ -z "${log_path}" ]; then
    echo "Could not find runtime log path in rollout output" >&2
    echo "Recent rollout output:" >&2
    tail -n 80 "${tmp}" >&2 || true
    rm -f "${tmp}"
    return 1
  fi

  rm -f "${tmp}"
  soridormi_to_container_data_path "${log_path}"
}

for line in "${rollout_lines[@]}"; do
  IFS=$'\t' read -r name teacher_profile candidate_profile_path steps seconds <<<"${line}"

  echo
  echo "Command-grid scenario: ${name}"
  echo "  teacher=${teacher_profile}"
  echo "  candidate=${candidate_profile_path}"
  echo "  steps=${steps} seconds=${seconds:-disabled}"

  if [ "${dry_run}" = "1" ]; then
    continue
  fi

  rollout_args=(--steps "${steps}" "${require_provider_args[@]}")
  if [ -n "${seconds}" ]; then
    rollout_args+=(--seconds "${seconds}")
  fi

  teacher_log="$(run_rollout_and_extract_log "${teacher_profile}" "${rollout_args[@]}")"
  candidate_log="$(run_rollout_and_extract_log "${candidate_profile_path}" "${rollout_args[@]}")"

  comparison_out="${comparisons_dir}/${name}"

  ./scripts/compare_policy_rollouts.sh \
    "${teacher_log}" \
    "${candidate_log}" \
    --output-dir "${comparison_out}" \
    "${compare_args[@]}"
done

if [ "${dry_run}" = "1" ]; then
  echo
  echo "Dry run complete; no rollouts or comparisons were executed."
  exit 0
fi

comparisons_dir_container="$(soridormi_to_container_data_path "${comparisons_dir}")"
output_dir_container="$(soridormi_to_container_data_path "${output_dir}")"

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.command_grid summarize "$@"
' _ "${comparisons_dir_container}" --output-dir "${output_dir_container}"
