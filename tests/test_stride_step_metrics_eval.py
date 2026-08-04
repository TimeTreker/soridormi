from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.stride_step_metrics_eval import (
    StrideStepThresholds,
    evaluate_stride_step_metrics,
    render_markdown,
)


def _row(
    step: int,
    *,
    base_x: float,
    contacts: list[float],
    left_x: float,
    right_x: float,
    base_z: float = 0.30,
    scenario_id: str = "flat_walk_varied_speed_v1",
) -> dict:
    left_contact = contacts[0] >= 0.5
    right_contact = contacts[1] >= 0.5
    return {
        "type": "runtime_step",
        "step_index": step,
        "scenario_id": scenario_id,
        "skill_id": "walk_velocity",
        "state": {
            "time": step * 0.1,
            "base_position_xyz": [base_x, 0.0, base_z],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "feet_position_xyz": [
                [left_x, 0.04, 0.0 if left_contact else 0.04],
                [right_x, -0.04, 0.0 if right_contact else 0.04],
            ],
            "feet_contacts": contacts,
            "joints": {"names": ["j"], "positions": [0.0], "velocities": [0.0], "torques": [0.0]},
            "imu": {"quat_wxyz": [1, 0, 0, 0], "gyro_xyz": [0, 0, 0], "accel_xyz": [0, 0, 9.81]},
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_stride_step_metrics_reports_progress_and_touchdowns(tmp_path: Path) -> None:
    path = tmp_path / "walk.jsonl"
    rows = [
        _row(0, base_x=0.00, contacts=[1.0, 0.0], left_x=0.00, right_x=0.05),
        _row(1, base_x=0.04, contacts=[0.0, 1.0], left_x=0.04, right_x=0.10),
        _row(2, base_x=0.08, contacts=[1.0, 0.0], left_x=0.15, right_x=0.12),
        _row(3, base_x=0.12, contacts=[0.0, 1.0], left_x=0.16, right_x=0.22),
        _row(4, base_x=0.16, contacts=[1.0, 0.0], left_x=0.28, right_x=0.24),
        _row(5, base_x=0.20, contacts=[0.0, 1.0], left_x=0.30, right_x=0.36),
    ]
    _write_jsonl(path, rows)

    report = evaluate_stride_step_metrics(path, thresholds=StrideStepThresholds(min_forward_speed_mps=0.01))

    assert report.ok
    assert report.sample_count == 6
    assert report.samples_with_base == 6
    assert report.samples_with_feet == 6
    assert report.duration_s == 0.5
    assert report.base_motion["forward_x_m"] == 0.20
    assert report.base_motion["mean_forward_speed_mps"] == 0.4
    assert report.step_events["touchdown_count"] == 5
    assert report.step_events["left_touchdown_count"] == 2
    assert report.step_events["right_touchdown_count"] == 3
    assert report.step_events["alternating_touchdown_ratio"] == 1.0
    assert report.scenario["scenario_ids"] == ["flat_walk_varied_speed_v1"]
    assert not report.errors


def test_stride_step_metrics_detects_fall_from_low_base_height(tmp_path: Path) -> None:
    path = tmp_path / "fall.jsonl"
    rows = [
        _row(0, base_x=0.0, contacts=[1.0, 0.0], left_x=0.0, right_x=0.0, base_z=0.30),
        _row(1, base_x=0.0, contacts=[0.0, 1.0], left_x=0.0, right_x=0.0, base_z=0.08),
    ]
    _write_jsonl(path, rows)

    report = evaluate_stride_step_metrics(path, thresholds=StrideStepThresholds(min_base_z_m=0.12))

    assert not report.ok
    assert report.fall["detected"] is True
    assert report.fall["low_base_z_samples"] == 1
    assert any("fall" in error for error in report.errors)


def test_swing_boundary_exclusion_uses_stable_middle_of_swing(tmp_path: Path) -> None:
    path = tmp_path / "stable_swing.jsonl"
    rows = [
        _row(0, base_x=0.00, contacts=[1.0, 1.0], left_x=0.00, right_x=0.05),
        _row(1, base_x=0.02, contacts=[0.0, 1.0], left_x=0.02, right_x=0.05),
        _row(2, base_x=0.04, contacts=[0.0, 1.0], left_x=0.04, right_x=0.05),
        _row(3, base_x=0.06, contacts=[0.0, 1.0], left_x=0.06, right_x=0.05),
        _row(4, base_x=0.08, contacts=[0.0, 1.0], left_x=0.08, right_x=0.05),
        _row(5, base_x=0.10, contacts=[1.0, 1.0], left_x=0.10, right_x=0.05),
    ]
    rows[1]["state"]["feet_position_xyz"][0][2] = 0.005
    rows[2]["state"]["feet_position_xyz"][0][2] = 0.020
    rows[3]["state"]["feet_position_xyz"][0][2] = 0.021
    rows[4]["state"]["feet_position_xyz"][0][2] = 0.004
    _write_jsonl(path, rows)

    report = evaluate_stride_step_metrics(
        path,
        thresholds=StrideStepThresholds(
            min_forward_speed_mps=0.01,
            swing_boundary_exclusion_samples=1,
        ),
    )

    clearance = report.foot_clearance
    assert clearance["raw_swing_count"] == 4
    assert clearance["stable_swing_count"] == 2
    assert clearance["swing_boundary_exclusion_applied"] is True
    assert clearance["low_clearance_swing_ratio"] == 0.0
    assert clearance["swing"]["p50_m"] == 0.0205


def test_render_markdown_contains_stride_summary(tmp_path: Path) -> None:
    path = tmp_path / "walk.jsonl"
    _write_jsonl(
        path,
        [
            _row(0, base_x=0.00, contacts=[1.0, 0.0], left_x=0.00, right_x=0.05),
            _row(1, base_x=0.04, contacts=[0.0, 1.0], left_x=0.04, right_x=0.10),
            _row(2, base_x=0.08, contacts=[1.0, 0.0], left_x=0.15, right_x=0.12),
        ],
    )

    rendered = render_markdown(evaluate_stride_step_metrics(path))

    assert "Soridormi stride/step metrics report" in rendered
    assert "mean_forward_speed_mps" in rendered
    assert "touchdown_count" in rendered
