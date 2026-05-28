#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/run_teacher_suite.sh [options]

Generate and run a command-conditioned teacher suite against an already-running
MuJoCo simulator. This creates one temporary profile per scenario and then uses
run_policy_rollout_smoke.sh for each profile.

Options:
  --suite PATH                 Teacher suite YAML (default: configs/teacher_suites/open_duck_teacher_v1.yaml)
  --output-dir PATH            Output dir (default: data/teacher_suites/open_duck_teacher_v1)
  --require-provider NAME      Require ONNX provider during profile checks. May repeat.
  --force                      Overwrite generated profiles.
  --dry-run                    Generate profiles and print planned rollouts only.
  -h, --help                   Show this help.

Start the sim server in another terminal first.
EOF
}

suite="configs/teacher_suites/open_duck_teacher_v1.yaml"
output_dir="data/teacher_suites/open_duck_teacher_v1"
force="0"
dry_run="0"
require_provider_args=()

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

generate_args=(--suite "${suite}" --output-dir "${output_dir}")
if [ "${force}" = "1" ]; then
  generate_args+=(--force)
fi

./scripts/generate_teacher_suite.sh "${generate_args[@]}"

manifest="${output_dir%/}/teacher_suite_manifest.json"
if [ ! -f "${manifest}" ]; then
  echo "Teacher suite manifest not found: ${manifest}" >&2
  exit 1
fi

echo "Teacher suite manifest: ${manifest}"

mapfile -t rollout_lines < <(python - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
payload = json.loads(manifest.read_text())
for scenario in payload.get("scenarios", []):
    profile = scenario["profile_path"]
    steps = str(scenario.get("steps") or 0)
    seconds = scenario.get("seconds")
    name = scenario.get("name", "scenario")
    if seconds is None:
        seconds_text = ""
    else:
        seconds_text = str(seconds)
    print("\t".join([name, profile, steps, seconds_text]))
PY
)

for line in "${rollout_lines[@]}"; do
  IFS=$'\t' read -r name profile steps seconds <<<"${line}"
  echo
  echo "Teacher scenario: ${name}"
  echo "  profile=${profile}"
  echo "  steps=${steps} seconds=${seconds:-disabled}"
  if [ "${dry_run}" = "1" ]; then
    continue
  fi
  args=("${profile}" --steps "${steps}" "${require_provider_args[@]}")
  if [ -n "${seconds}" ]; then
    args+=(--seconds "${seconds}")
  fi
  ./scripts/run_policy_rollout_smoke.sh "${args[@]}"
done
