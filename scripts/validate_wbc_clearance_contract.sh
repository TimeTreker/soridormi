#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

contract="configs/wbc/open_duck_mini_v2_clearance_contract.json"
output_dir="${SORIDORMI_WBC_CLEARANCE_OUTPUT_DIR:-/tmp/soridormi_wbc_clearance_contract}"
run_pytest="1"

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_wbc_clearance_contract.sh [options]

Validate the first sim-only WBC clearance-control contract and planning harness.
This command is dry/offline: it does not train, launch MuJoCo, create runtime
profiles, or send actuator commands.

Options:
  --contract PATH   Contract JSON to validate
  --output-dir DIR  Temporary output directory
  --skip-pytest     Skip focused pytest checks
  -h, --help        Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --contract)
      contract="${2:?--contract requires a value}"
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

echo "Soridormi WBC clearance contract validation"
echo "==========================================="
echo "Contract:   ${contract}"
echo "Output dir: ${output_dir}"
echo

echo "Checking wrappers..."
bash -n scripts/plan_wbc_clearance_experiment.sh
bash -n scripts/validate_pre_wbc_scenario_surface.sh
bash -n scripts/validate_wbc_clearance_contract.sh

echo "Compiling WBC clearance module..."
python -m compileall -q src/soridormi_runtime/wbc_clearance_contract.py

echo "Planning bounded WBC clearance experiments..."
mkdir -p "${output_dir}"
./scripts/plan_wbc_clearance_experiment.sh \
  --contract "${contract}" \
  --output-dir "${output_dir}/plan" \
  --json \
  --strict >"${output_dir}/wbc_clearance_plan_stdout.json"
python -m json.tool "${output_dir}/wbc_clearance_plan_stdout.json" >/dev/null

echo "Checking pre-WBC scenario surface..."
./scripts/validate_pre_wbc_scenario_surface.sh \
  --contract "${contract}" \
  --scenario-manifest configs/scenarios/open_duck_mini_v2_scenarios.json \
  --output-dir "${output_dir}/pre_wbc_surface" \
  --skip-pytest

echo "Checking docs..."
rg -n "plan_wbc_clearance_experiment.sh|validate_wbc_clearance_contract.sh" \
  docs/README.md \
  docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md \
  docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md \
  docs/SORIDORMI_TARGET_AND_ROADMAP.md >/dev/null
python - <<'PY'
from pathlib import Path

paths = [
    Path("docs/README.md"),
    Path("docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md"),
    Path("docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md"),
    Path("docs/SORIDORMI_TARGET_AND_ROADMAP.md"),
]
bad = [str(path) for path in paths if path.read_text(encoding="utf-8").count("```") % 2]
assert not bad, bad
PY

if [ "${run_pytest}" = "1" ]; then
  echo "Running focused WBC clearance tests..."
  pytest -q \
    tests/test_wbc_clearance_contract.py \
    tests/test_pre_wbc_scenario_surface.py
else
  echo "Skipping focused pytest gate (--skip-pytest)."
fi

echo
echo "WBC clearance contract validation: PASS"
