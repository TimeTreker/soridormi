from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.scenario_curriculum import get_scenario_definition
from soridormi_runtime.scenario_rollout_eval import (
    evaluate_scenario_rollout,
    thresholds_from_scenario_manifest,
)
from soridormi_runtime.scenario_suite_eval import build_scenario_suite_report

CLEARANCE_PROMOTION_SCENARIOS = (
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
)


def _row(
    step: int,
    *,
    base_x: float,
    swing_z_m: float = 0.03,
    include_feet: bool = True,
    all_contact: bool = False,
    scenario_id: str = "flat_walk_varied_speed_v1",
    skill_id: str = "walk_velocity",
) -> dict:
    if all_contact:
        contacts = [1.0, 1.0]
    else:
        contacts = [1.0, 0.0] if step % 2 == 0 else [0.0, 1.0]
    left_contact = contacts[0] >= 0.5
    right_contact = contacts[1] >= 0.5
    state = {
        "time": step * 0.1,
        "base_position_xyz": [base_x, 0.0, 0.30],
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    if include_feet:
        state.update(
            {
                "feet_position_xyz": [
                    [base_x, 0.04, 0.0 if left_contact else swing_z_m],
                    [base_x + 0.05, -0.04, 0.0 if right_contact else swing_z_m],
                ],
                "feet_contacts": contacts,
            }
        )
    return {
        "type": "runtime_step",
        "step_index": step,
        "scenario_id": scenario_id,
        "skill_id": skill_id,
        "state": state,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _progress_rows(
    *, swing_z_m: float = 0.03, include_feet: bool = True, all_contact: bool = False
) -> list[dict]:
    return [
        _row(
            step,
            base_x=0.04 * step,
            swing_z_m=swing_z_m,
            include_feet=include_feet,
            all_contact=all_contact,
        )
        for step in range(8)
    ]


def _failed_check_names(report: dict) -> set[str]:
    return {check["name"] for check in report["checks"] if check.get("ok") is False}


def test_clearance_promotion_scenarios_require_hard_clearance_thresholds() -> None:
    for scenario_id in CLEARANCE_PROMOTION_SCENARIOS:
        thresholds = thresholds_from_scenario_manifest(get_scenario_definition(scenario_id))

        assert thresholds.require_foot_metrics is True, scenario_id
        assert thresholds.min_swing_clearance_m >= 0.015, scenario_id
        assert thresholds.max_low_clearance_ratio <= 0.25, scenario_id
        assert thresholds.swing_boundary_exclusion_samples == 1, scenario_id


def test_required_clearance_failure_is_rollout_error_not_warning(tmp_path: Path) -> None:
    path = tmp_path / "low_clearance.jsonl"
    _write_jsonl(path, _progress_rows(swing_z_m=0.005))

    report = evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1")

    assert not report.ok
    checks = {check["name"]: check for check in report.checks}
    assert checks["low_clearance_swing_ratio"]["severity"] == "error"
    assert checks["swing_clearance_p50_m"]["severity"] == "error"
    assert checks["low_clearance_swing_ratio"]["ok"] is False
    assert checks["swing_clearance_p50_m"]["ok"] is False
    assert any("low_clearance_swing_ratio" in error for error in report.errors)
    assert any("swing_clearance_p50_m" in error for error in report.errors)


def test_required_clearance_fails_when_swing_metrics_are_missing(tmp_path: Path) -> None:
    path = tmp_path / "no_swing.jsonl"
    _write_jsonl(path, _progress_rows(all_contact=True))

    report = evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1")

    assert not report.ok
    checks = {check["name"]: check for check in report.checks}
    assert checks["foot_metrics_present"]["ok"] is True
    assert checks["low_clearance_swing_ratio"]["ok"] is False
    assert checks["swing_clearance_p50_m"]["ok"] is False
    assert checks["low_clearance_swing_ratio"]["severity"] == "error"
    assert checks["swing_clearance_p50_m"]["severity"] == "error"


def test_suite_summary_aggregates_clearance_failures(tmp_path: Path) -> None:
    low_path = tmp_path / "low.jsonl"
    high_path = tmp_path / "high.jsonl"
    low_report_path = tmp_path / "low_report.json"
    high_report_path = tmp_path / "high_report.json"
    _write_jsonl(low_path, _progress_rows(swing_z_m=0.005))
    _write_jsonl(high_path, _progress_rows(swing_z_m=0.03))
    low_report = evaluate_scenario_rollout(low_path, scenario_id="flat_walk_varied_speed_v1")
    high_report = evaluate_scenario_rollout(high_path, scenario_id="flat_walk_varied_speed_v1")
    low_report_path.write_text(json.dumps(low_report.to_dict()), encoding="utf-8")
    high_report_path.write_text(json.dumps(high_report.to_dict()), encoding="utf-8")

    suite = build_scenario_suite_report([low_report_path, high_report_path])

    assert not suite.ok
    assert suite.summary_metrics["clearance_failed_count"] == 1
    assert suite.summary_metrics["foot_metrics_missing_count"] == 0
    assert suite.summary_metrics["min_swing_clearance_p50_m"] == 0.005
    assert suite.summary_metrics["max_low_clearance_ratio"] == 1.0


def test_rollout_cli_functionally_fails_low_clearance_json(tmp_path: Path) -> None:
    path = tmp_path / "low_clearance_cli.jsonl"
    _write_jsonl(path, _progress_rows(swing_z_m=0.005))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scenario_rollout_eval",
            "--scenario",
            "flat_walk_varied_speed_v1",
            "--log",
            str(path),
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["acceptance_thresholds"]["require_foot_metrics"] is True
    assert payload["metrics"]["low_clearance_swing_ratio"] == 1.0
    assert {"low_clearance_swing_ratio", "swing_clearance_p50_m"}.issubset(
        _failed_check_names(payload)
    )


def test_suite_cli_functionally_reports_clearance_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "low_clearance_suite.jsonl"
    report_path = tmp_path / "scenario_rollout_report.json"
    json_output = tmp_path / "suite_summary.json"
    _write_jsonl(log_path, _progress_rows(swing_z_m=0.005))
    rollout_report = evaluate_scenario_rollout(log_path, scenario_id="flat_walk_varied_speed_v1")
    report_path.write_text(json.dumps(rollout_report.to_dict()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scenario_suite_eval",
            "--reports",
            str(report_path),
            "--expected-scenario",
            "flat_walk_varied_speed_v1",
            "--json-output",
            str(json_output),
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["summary_metrics"]["clearance_failed_count"] == 1
    assert payload["summary_metrics"]["min_swing_clearance_p50_m"] == 0.005
    assert json.loads(json_output.read_text())["summary_metrics"]["max_low_clearance_ratio"] == 1.0
