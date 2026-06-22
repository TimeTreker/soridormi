#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/build_clearance_evidence_package.sh [options]

Build a Soridormi clearance evidence-package manifest from clearance readiness,
follow-camera visual-inspection planning, and an optional filled visual-review
JSON. The command does not launch MuJoCo; it packages evidence paths, blockers,
next steps, and review templates.

Options:
  --profile-name NAME            Policy profile to package
  --scenario ID                  Required scenario id; repeat or comma-separate
  --scenario-manifest PATH       Scenario manifest path
  --output-dir DIR               Directory for package JSON/Markdown artifacts
  --readiness-report PATH        Clearance readiness JSON path
  --visual-plan PATH             Visual inspection plan JSON path
  --visual-review PATH           Filled visual review JSON path
  --no-require-clearance-ready   Do not block when clearance readiness is missing/failing
  --no-require-visual-plan       Do not block when visual plan is missing/failing
  --require-visual-pass          Block unless visual review passes every required field
  --json                         Print machine-readable JSON to stdout
  --strict                       Exit nonzero when package is blocked
  -h, --help                     Show this help

Example:
  ./scripts/build_clearance_evidence_package.sh \
    --profile-name context_stage1_three_scenario_10ep_e80 \
    --output-dir artifacts/clearance_evidence/context_stage1_three_scenario_10ep_e80 \
    --json
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

PYTHONPATH=src python -m soridormi_runtime.m10_evidence_package "$@"
