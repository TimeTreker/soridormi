#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/analyze_m10_clearance_readiness.sh [options]

Analyze M10 swing-foot clearance readiness from required scenario rollout reports.
By default it reads:
  artifacts/scenario_eval/<profile>_suite/<scenario>/scenario_rollout_report.json

Options:
  --profile-name NAME       Profile name for clearance analysis
  --suite-dir DIR           Directory containing per-scenario report subdirectories
  --report PATH             Explicit scenario_rollout_report.json path; repeat as needed
  --scenario ID             Required scenario id; repeat or comma-separate
  --scenario-manifest PATH  Scenario manifest path
  --output-dir DIR          Output directory for JSON and Markdown reports
  --json                    Print machine-readable JSON to stdout
  --strict                  Exit nonzero if readiness fails
  -h, --help                Show this help

Example:
  ./scripts/analyze_m10_clearance_readiness.sh \
    --profile-name context_stage1_three_scenario_10ep_e80 \
    --output-dir artifacts/m10_clearance_readiness/context_stage1_three_scenario_10ep_e80
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

PYTHONPATH=src python -m soridormi_runtime.m10_clearance_readiness "$@"
