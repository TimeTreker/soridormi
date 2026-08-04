#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

contract="configs/wbc/open_duck_mini_v2_clearance_contract.json"
scenario_manifest="configs/scenarios/open_duck_mini_v2_scenarios.json"
output_dir="${SORIDORMI_PRE_WBC_SCENARIO_SURFACE_OUTPUT_DIR:-/tmp/soridormi_pre_wbc_scenario_surface}"
run_pytest="1"

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_pre_wbc_scenario_surface.sh [options]

Validate the dry/offline pre-WBC scenario surface. This checks that the default
ready locomotion suite, clearance qualification core scenarios, and WBC clearance contract agree
before WBC tuning starts. It does not train, launch MuJoCo, create runtime
profiles, launch hardware, or send actuator commands.

Options:
  --contract PATH            WBC clearance contract JSON
  --scenario-manifest PATH   Scenario manifest path
  --output-dir DIR           Temporary output directory
  --skip-pytest              Skip focused pytest checks
  -h, --help                 Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --contract)
      contract="${2:?--contract requires a value}"
      shift 2
      ;;
    --scenario-manifest)
      scenario_manifest="${2:?--scenario-manifest requires a value}"
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

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Soridormi pre-WBC scenario-surface validation"
echo "============================================="
echo "Contract:          ${contract}"
echo "Scenario manifest: ${scenario_manifest}"
echo "Output dir:        ${output_dir}"
echo

echo "Checking wrappers..."
bash -n scripts/evaluate_scenario_suite.sh
bash -n scripts/plan_wbc_clearance_experiment.sh
bash -n scripts/validate_pre_wbc_scenario_surface.sh

echo "Compiling pre-WBC scenario-surface module..."
python -m compileall -q src/soridormi_runtime/pre_wbc_scenario_surface.py

echo "Planning default ready locomotion suite..."
mkdir -p "${output_dir}"
./scripts/evaluate_scenario_suite.sh \
  --scenario-manifest "${scenario_manifest}" \
  --dry-run-only \
  --output-dir "${output_dir}/suite_dry_run" \
  --json >"${output_dir}/suite_dry_run_stdout.json"
python -m json.tool "${output_dir}/suite_dry_run_stdout.json" >/dev/null

echo "Validating pre-WBC scenario surface..."
python -m soridormi_runtime.pre_wbc_scenario_surface \
  --contract "${contract}" \
  --scenario-manifest "${scenario_manifest}" \
  --output-dir "${output_dir}/surface" \
  --json \
  --strict >"${output_dir}/pre_wbc_surface_stdout.json"
python -m json.tool "${output_dir}/pre_wbc_surface_stdout.json" >/dev/null

echo "Checking docs..."
rg -n "validate_pre_wbc_scenario_surface.sh|pre-WBC|six-scenario" \
  docs/README.md \
  docs/SORIDORMI_SCENARIO_SUITE_EVAL.md \
  docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md \
  docs/SORIDORMI_EXECUTION_ROADMAP.md >/dev/null
python - <<'PY'
from pathlib import Path

paths = [
    Path("docs/README.md"),
    Path("docs/SORIDORMI_SCENARIO_SUITE_EVAL.md"),
    Path("docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md"),
    Path("docs/SORIDORMI_EXECUTION_ROADMAP.md"),
]
bad = [str(path) for path in paths if path.read_text(encoding="utf-8").count("```") % 2]
assert not bad, bad
PY

if [ "${run_pytest}" = "1" ]; then
  echo "Running focused pre-WBC scenario-surface tests..."
  pytest -q \
    tests/test_pre_wbc_scenario_surface.py \
    tests/test_scenario_suite_eval.py \
    tests/test_scenario_suite_wrapper.py \
    tests/test_wbc_clearance_contract.py
else
  echo "Skipping focused pytest gate (--skip-pytest)."
fi

echo
echo "Pre-WBC scenario-surface validation: PASS"
