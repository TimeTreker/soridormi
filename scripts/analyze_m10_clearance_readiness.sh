#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/analyze_m10_clearance_readiness.sh [options]

Analyze swing-foot clearance metrics from the context_stage1_three_scenario_10ep_e80
evaluation suite and assess M10 gate readiness.

Options:
  --profile-name NAME       Profile name for clearance analysis
  --output-dir DIR          Output directory for clearance report
  --json                    Print machine-readable JSON to stdout
  -h, --help                Show this help

Example:
  ./scripts/analyze_m10_clearance_readiness.sh \
    --profile-name context_stage1_three_scenario_10ep_e80 \
    --output-dir artifacts/m10_clearance_readiness
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

profile_name='context_stage1_three_scenario_10ep_e80'
output_dir=''
json_flag=''

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile-name) profile_name="$2"; shift 2;;
    --output-dir) output_dir="$2"; shift 2;;
    --json) json_flag='--json'; shift;;
    --help|-h) usage; exit 0;;
    *) shift;;
  esac
done

if [ -z "$output_dir" ]; then
  output_dir="artifacts/m10_clearance_readiness/${profile_name}"
fi

mkdir -p "$output_dir"

# Extract and summarize clearance metrics from the three-scenario suite
python3 - "$profile_name" "$output_dir" <<'PYTHON_END'
import json
import sys
from pathlib import Path

profile = sys.argv[1] if len(sys.argv) > 1 else "context_stage1_three_scenario_10ep_e80"
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/m10_clearance_readiness") / profile

scenarios = [
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
]

suite_dir = Path("artifacts/scenario_eval") / f"{profile}_suite"
clearance_summary = {
    "profile": profile,
    "ok": True,
    "scenarios": [],
    "blockers": [],
    "recommendations": [],
}

for scenario in scenarios:
    report_json = suite_dir / scenario / "scenario_rollout_report.json"
    if not report_json.exists():
        clearance_summary["ok"] = False
        clearance_summary["blockers"].append(f"Scenario report not found: {report_json}")
        continue
    
    try:
        report = json.loads(report_json.read_text())
        metrics = report.get("metrics", {})
        
        scenario_data = {
            "scenario": scenario,
            "status": "unknown",
            "swing_clearance_p50_m": metrics.get("swing_clearance_p50_m"),
            "low_clearance_swing_ratio": metrics.get("low_clearance_swing_ratio"),
            "min_base_z_m": metrics.get("min_base_z_m"),
        }
        
        swing_clear = metrics.get("swing_clearance_p50_m")
        low_ratio = metrics.get("low_clearance_swing_ratio")
        
        if swing_clear is not None and swing_clear < 0.015:
            scenario_data["status"] = "FAIL_LOW_CLEARANCE"
            clearance_summary["ok"] = False
            clearance_summary["blockers"].append(
                f"{scenario}: median swing clearance {swing_clear:.4f}m < 0.015m"
            )
        elif low_ratio is not None and low_ratio > 0.35:
            scenario_data["status"] = "WARN_LOW_RATIO"
            clearance_summary["recommendations"].append(
                f"{scenario}: high low-clearance ratio {low_ratio:.2%} (threshold 0.35)"
            )
        else:
            scenario_data["status"] = "PASS"
        
        clearance_summary["scenarios"].append(scenario_data)
    except Exception as e:
        clearance_summary["ok"] = False
        clearance_summary["blockers"].append(f"Error reading {scenario}: {e}")

if not clearance_summary["blockers"] and all(s.get("status") == "PASS" for s in clearance_summary["scenarios"]):
    clearance_summary["ok"] = True
    clearance_summary["gate_status"] = "READY_FOR_VISUAL_INSPECTION"
else:
    clearance_summary["gate_status"] = "BLOCKED_BY_CLEARANCE_ISSUES"

if not clearance_summary["recommendations"]:
    if clearance_summary["ok"]:
        clearance_summary["recommendations"] = [
            "All clearance metrics pass. Next: perform follow-camera visual inspection.",
            "Compare swing foot height with official teacher rollouts.",
            "Measure foot trajectory ground clearance at various speeds and turns.",
        ]
    else:
        clearance_summary["recommendations"] = [
            "Collect clearance-focused training data with explicit swing-height rewards.",
            "Add foot-clearance thresholds to training context or task_context.",
            "Consider targeted BC or residual RL with clearance penalty.",
        ]

output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / "m10_clearance_readiness.json"
json_path.write_text(json.dumps(clearance_summary, indent=2))

print(json.dumps(clearance_summary, indent=2))
PYTHON_END
