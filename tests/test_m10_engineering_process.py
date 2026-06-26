from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _report_payload(*, scenario_id: str, low_ratio: float) -> dict:
    ok = low_ratio <= 0.25
    return {
        "ok": ok,
        "scenario_id": scenario_id,
        "acceptance_thresholds": {
            "require_foot_metrics": True,
            "min_swing_clearance_m": 0.015,
            "max_low_clearance_ratio": 0.25,
        },
        "metrics": {
            "samples_with_feet": 20,
            "swing_clearance_p05_m": 0.012,
            "swing_clearance_p50_m": 0.018,
            "low_clearance_swing_ratio": low_ratio,
            "forward_distance_m": 0.70,
            "stuck_ratio": 0.0,
            "fallen": False,
        },
        "checks": [
            {
                "name": "foot_metrics_present",
                "ok": True,
                "value": 20,
                "threshold": "present",
                "severity": "error",
            },
            {
                "name": "low_clearance_swing_ratio",
                "ok": ok,
                "value": low_ratio,
                "threshold": 0.25,
                "severity": "error",
            },
            {
                "name": "swing_clearance_p50_m",
                "ok": True,
                "value": 0.018,
                "threshold": 0.015,
                "severity": "error",
            },
        ],
        "stride_step_report": {"samples_with_feet": 20},
        "errors": [] if ok else ["clearance failed"],
        "warnings": [],
    }


def _write_suite(root: Path, profile: str) -> Path:
    low_ratios = {
        "flat_walk_varied_speed_v1": 0.268,
        "start_stop_velocity_ramp_v1": 0.257,
        "curve_turn_walk_v1": 0.308,
    }
    for scenario_id, low_ratio in low_ratios.items():
        report_path = root / profile / scenario_id / "scenario_rollout_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_report_payload(scenario_id=scenario_id, low_ratio=low_ratio)),
            encoding="utf-8",
        )
    return root / profile


def test_m10_engineering_process_script_parse_and_stays_offline() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "validate_m10_engineering_process.sh"
    source = script.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "report_clearance_candidate_history.sh" in source
    assert "analyze_clearance_readiness.sh" in source
    assert "plan_policy_visual_inspection.sh" in source
    assert "build_clearance_evidence_package.sh" in source
    assert "tests/test_m10_clearance_history.py" in source
    assert "run_sim_server.sh" not in source
    assert "train_clearance_residual_policy.sh" not in source


def test_m10_engineering_process_script_smoke_without_pytest(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    profile = "reference_profile"
    scenario_root = tmp_path / "scenario_eval"
    suite_dir = _write_suite(scenario_root, profile)
    output_dir = tmp_path / "process"

    result = subprocess.run(
        [
            "bash",
            "scripts/validate_m10_engineering_process.sh",
            "--profile-name",
            profile,
            "--scenario-eval-root",
            str(scenario_root),
            "--suite-dir",
            str(suite_dir),
            "--output-dir",
            str(output_dir),
            "--skip-pytest",
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "M10 engineering-process validation: PASS" in result.stdout
    assert (output_dir / "history" / "clearance_candidate_history.json").exists()
    assert (output_dir / "readiness" / "clearance_readiness.json").exists()
    assert (
        output_dir / "visual_inspection" / "policy_visual_inspection_plan.json"
    ).exists()
    assert (output_dir / "evidence" / "clearance_evidence_package.json").exists()
