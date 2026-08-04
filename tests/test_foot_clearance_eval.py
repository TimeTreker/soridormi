from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.foot_clearance_eval import (
    FootClearanceThresholds,
    evaluate_foot_clearance,
    render_markdown,
)


def _record(step: int, left_z: float, right_z: float, contacts: list[float]) -> dict:
    return {
        "type": "runtime_step",
        "step_index": step,
        "state": {
            "time": step * 0.02,
            "feet_position_xyz": [[0.0, 0.04, left_z], [0.0, -0.04, right_z]],
            "feet_contacts": contacts,
            "joints": {"names": ["j"], "positions": [0.0], "velocities": [0.0], "torques": [0.0]},
            "imu": {"quat_wxyz": [1.0, 0.0, 0.0, 0.0], "gyro_xyz": [0, 0, 0], "accel_xyz": [0, 0, 9.81]},
        },
    }


def test_evaluate_foot_clearance_reports_low_swing_steps(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    rows = [
        _record(0, 0.0, 0.025, [1.0, 0.0]),
        _record(1, 0.02, 0.0, [0.0, 1.0]),
        _record(2, 0.005, 0.0, [0.0, 1.0]),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = evaluate_foot_clearance(
        path,
        thresholds=FootClearanceThresholds(min_swing_clearance_m=0.01, max_low_clearance_ratio=0.20),
    )

    assert report.ok
    assert report.sample_count == 3
    assert report.samples_with_feet == 3
    assert report.left["low_clearance_swing_steps"] == 1
    assert report.combined["low_clearance_swing_steps"] == 1
    assert report.warnings


def test_render_markdown_contains_summary(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    path.write_text(json.dumps(_record(0, 0.03, 0.02, [0.0, 0.0])) + "\n", encoding="utf-8")
    report = evaluate_foot_clearance(path)
    rendered = render_markdown(report)
    assert "Soridormi foot-clearance report" in rendered
    assert "combined" in rendered
