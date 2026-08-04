#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/report_clearance_candidate_history.sh [options]

Summarize existing clearance qualification clearance scenario-evaluation artifacts and compare each
candidate against the retained s143 reference. This is an offline report; it
does not launch MuJoCo or train a policy.

Options:
  --scenario-eval-root DIR       Root containing artifacts/scenario_eval suites
  --profile NAME                 Profile/suite directory to include; repeat or
                                 comma-separate
  --scenario ID                  Required scenario id; repeat or comma-separate
  --scenario-manifest PATH       Scenario manifest path
  --reference-profile-name NAME  Retained reference profile name
  --reference-suite-dir DIR      Retained reference scenario suite directory
  --output-dir DIR               Output directory for JSON and Markdown reports
  --output PATH                  Markdown output path
  --json-output PATH             JSON output path
  --json                         Print machine-readable JSON to stdout
  --strict                       Exit nonzero when no candidate is ready for
                                 visual inspection
  -h, --help                     Show this help

Example:
  ./scripts/report_clearance_candidate_history.sh --json
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

PYTHONPATH=src python -m soridormi_runtime.clearance_candidate_history "$@"
