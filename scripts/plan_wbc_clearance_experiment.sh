#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/plan_wbc_clearance_experiment.sh [options]

Plan bounded sim-only WBC clearance parameter experiments. This validates the
contract and writes JSON/Markdown artifacts. It does not train, launch MuJoCo,
create runtime profiles, or send actuator commands.

Options:
  --contract PATH                WBC clearance contract JSON
  --scenario-manifest PATH       Scenario manifest path
  --scenario-eval-root DIR       Scenario-evaluation artifact root
  --baseline-profile NAME        Override baseline profile from the contract
  --reference-profile-name NAME  Override retained reference profile
  --output-dir DIR               Output directory for JSON/Markdown artifacts
  --output PATH                  Markdown output path
  --json-output PATH             JSON output path
  --json                         Print machine-readable JSON to stdout
  --strict                       Exit nonzero when the contract is invalid
  -h, --help                     Show this help

Example:
  ./scripts/plan_wbc_clearance_experiment.sh --json
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

PYTHONPATH=src python -m soridormi_runtime.wbc_clearance_contract "$@"
