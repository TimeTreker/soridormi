#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/plan_m10_visual_inspection.sh [options]

Create a repeatable M10 follow-camera visual-inspection plan. The command does
not launch MuJoCo by itself; it writes JSON/Markdown artifacts containing the
follow-camera server command, per-scenario rollout commands, and visual evidence
checklist.

Options:
  --profile-name NAME          Policy profile to inspect
  --scenario ID                Required scenario id; repeat or comma-separate
  --scenario-manifest PATH     Scenario manifest path
  --output-dir DIR             Directory for JSON and Markdown plan artifacts
  --readiness-report PATH      M10 clearance readiness JSON path
  --require-clearance-ready    Exit nonzero unless readiness report exists and passes
  --camera-distance N          Follow-camera distance
  --camera-azimuth DEG         Follow-camera azimuth
  --camera-elevation DEG       Follow-camera elevation
  --control-hz HZ              Rollout control frequency
  --duration-s S               Optional rollout duration override
  --steps N                    Optional rollout step-count override
  --json                       Print machine-readable JSON to stdout
  --strict                     Exit nonzero when the plan is blocked
  -h, --help                   Show this help

Example:
  ./scripts/plan_m10_visual_inspection.sh \
    --profile-name context_stage1_three_scenario_10ep_e80 \
    --output-dir artifacts/m10_visual_inspection/context_stage1_three_scenario_10ep_e80 \
    --json
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

PYTHONPATH=src python -m soridormi_runtime.m10_visual_inspection "$@"
