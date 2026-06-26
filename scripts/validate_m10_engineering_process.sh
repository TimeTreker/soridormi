#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

profile_name="clearance_liftscale_stack_s143_step090_offset005"
scenario_eval_root="artifacts/scenario_eval"
suite_dir=""
output_dir="${SORIDORMI_M10_PROCESS_OUTPUT_DIR:-/tmp/soridormi_m10_engineering_process}"
run_pytest="1"

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_m10_engineering_process.sh [options]

Validate the offline M10 engineering process: candidate-history reporting,
clearance readiness, visual-inspection planning, evidence packaging, docs, and
focused tests. This command does not train, launch MuJoCo, or send actuator
commands.

Options:
  --profile-name NAME       Profile/suite to validate
  --scenario-eval-root DIR  Root containing scenario-evaluation suites
  --suite-dir DIR           Explicit suite directory for --profile-name
  --output-dir DIR          Temporary output directory
  --skip-pytest             Skip the focused pytest gate
  -h, --help                Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile-name)
      profile_name="${2:?--profile-name requires a value}"
      shift 2
      ;;
    --scenario-eval-root)
      scenario_eval_root="${2:?--scenario-eval-root requires a value}"
      shift 2
      ;;
    --suite-dir)
      suite_dir="${2:?--suite-dir requires a value}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --skip-pytest)
      run_pytest="0"
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

if [ -z "${suite_dir}" ]; then
  suite_dir="${scenario_eval_root}/${profile_name}"
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

require_path() {
  if [ ! -e "$1" ]; then
    echo "ERROR: required path does not exist: $1" >&2
    exit 1
  fi
}

json_check() {
  python -m json.tool "$1" >/dev/null
}

echo "Soridormi M10 engineering-process validation"
echo "============================================"
echo "Profile:            ${profile_name}"
echo "Scenario eval root: ${scenario_eval_root}"
echo "Suite dir:          ${suite_dir}"
echo "Output dir:         ${output_dir}"
echo

require_path "${suite_dir}"
mkdir -p "${output_dir}"

echo "Checking shell wrappers..."
bash -n scripts/report_clearance_candidate_history.sh
bash -n scripts/analyze_clearance_readiness.sh
bash -n scripts/plan_policy_visual_inspection.sh
bash -n scripts/build_clearance_evidence_package.sh
bash -n scripts/compare_policy_teacher_suite.sh

echo "Compiling M10 process modules..."
python -m compileall -q \
  src/soridormi_runtime/m10_clearance_history.py \
  src/soridormi_runtime/m10_clearance_readiness.py \
  src/soridormi_runtime/m10_visual_inspection.py \
  src/soridormi_runtime/m10_evidence_package.py \
  src/soridormi_runtime/m10_teacher_comparison.py

echo "Regenerating candidate-history report..."
history_stdout="${output_dir}/clearance_candidate_history_stdout.json"
./scripts/report_clearance_candidate_history.sh \
  --scenario-eval-root "${scenario_eval_root}" \
  --profile "${profile_name}" \
  --reference-profile-name "${profile_name}" \
  --reference-suite-dir "${suite_dir}" \
  --output-dir "${output_dir}/history" \
  --json >"${history_stdout}"
json_check "${history_stdout}"

echo "Regenerating clearance readiness report..."
readiness_stdout="${output_dir}/clearance_readiness_stdout.json"
./scripts/analyze_clearance_readiness.sh \
  --profile-name "${profile_name}" \
  --suite-dir "${suite_dir}" \
  --output-dir "${output_dir}/readiness" \
  --json >"${readiness_stdout}"
json_check "${readiness_stdout}"

echo "Planning follow-camera visual inspection without launching MuJoCo..."
visual_stdout="${output_dir}/policy_visual_inspection_stdout.json"
./scripts/plan_policy_visual_inspection.sh \
  --profile-name "${profile_name}" \
  --output-dir "${output_dir}/visual_inspection" \
  --readiness-report "${output_dir}/readiness/clearance_readiness.json" \
  --json >"${visual_stdout}"
json_check "${visual_stdout}"

echo "Packaging evidence manifest without promotion claims..."
evidence_stdout="${output_dir}/clearance_evidence_stdout.json"
./scripts/build_clearance_evidence_package.sh \
  --profile-name "${profile_name}" \
  --output-dir "${output_dir}/evidence" \
  --readiness-report "${output_dir}/readiness/clearance_readiness.json" \
  --visual-plan "${output_dir}/visual_inspection/policy_visual_inspection_plan.json" \
  --no-require-clearance-ready \
  --json >"${evidence_stdout}"
json_check "${evidence_stdout}"

echo "Checking M10 process docs..."
rg -n "report_clearance_candidate_history.sh" \
  docs/README.md \
  docs/SORIDORMI_EXECUTION_ROADMAP.md \
  docs/SORIDORMI_TARGET_AND_ROADMAP.md \
  docs/M6_SIM_TRAINING_LOOP.md >/dev/null
rg -n "validate_m10_engineering_process.sh" \
  docs/README.md \
  docs/SORIDORMI_EXECUTION_ROADMAP.md \
  docs/SORIDORMI_TARGET_AND_ROADMAP.md \
  docs/M6_SIM_TRAINING_LOOP.md \
  docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md >/dev/null
python - <<'PY'
from pathlib import Path

paths = [
    Path("docs/README.md"),
    Path("docs/SORIDORMI_EXECUTION_ROADMAP.md"),
    Path("docs/SORIDORMI_TARGET_AND_ROADMAP.md"),
    Path("docs/M6_SIM_TRAINING_LOOP.md"),
    Path("docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md"),
]
bad = [str(path) for path in paths if path.read_text(encoding="utf-8").count("```") % 2]
assert not bad, bad
PY

if [ "${run_pytest}" = "1" ]; then
  echo "Running focused M10 process tests..."
  pytest -q \
    tests/test_m10_engineering_process.py \
    tests/test_m10_clearance_history.py \
    tests/test_m10_clearance_readiness.py \
    tests/test_m10_clearance_promotion_gate.py \
    tests/test_m10_visual_inspection.py \
    tests/test_m10_evidence_package.py \
    tests/test_m10_teacher_comparison.py
else
  echo "Skipping focused pytest gate (--skip-pytest)."
fi

echo
echo "M10 engineering-process validation: PASS"
