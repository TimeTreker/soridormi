from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.rollout_failure_diagnosis import (
    RolloutDiagnosisThresholds,
    diagnose_rollout_comparison,
    render_rollout_failure_diagnosis_report,
    write_rollout_failure_diagnosis_outputs,
)


def _comparison(
    *,
    duration_ratio: float = 1.0,
    forward_ratio: float = 1.0,
    speed_ratio: float = 1.0,
    lateral_abs: float = 0.02,
    lateral_ratio: float = 1.0,
    action_abs: float = 1.0,
    action_ratio: float = 1.0,
    resets: int = 0,
    errors: list[str] | None = None,
) -> dict:
    return {
        "candidate": {
            "policy_records": 500,
            "reset_count": resets,
            "lateral_abs": lateral_abs,
            "action_abs_max": action_abs,
        },
        "comparison": {
            "duration_ratio": duration_ratio,
            "forward_ratio": forward_ratio,
            "forward_speed_ratio": speed_ratio,
            "lateral_abs_ratio": lateral_ratio,
            "action_abs_ratio": action_ratio,
        },
        "errors": errors or [],
        "warnings": [],
    }


def test_diagnosis_passes_for_healthy_rollout() -> None:
    result = diagnose_rollout_comparison(_comparison())

    assert result.ok
    assert result.primary_failure_modes == ["no_major_rollout_failure_detected"]
    assert "healthy" in result.summary


def test_diagnosis_identifies_core_failure_modes() -> None:
    result = diagnose_rollout_comparison(
        _comparison(
            duration_ratio=0.3,
            forward_ratio=0.2,
            speed_ratio=0.25,
            lateral_abs=0.5,
            lateral_ratio=5.0,
            action_abs=8.0,
            action_ratio=4.0,
            resets=1,
        ),
        thresholds=RolloutDiagnosisThresholds(),
    )

    assert not result.ok
    assert result.primary_failure_modes[:2] == ["stability_or_fall", "early_termination"]
    assert "weak_forward_locomotion" in result.primary_failure_modes
    assert "lateral_drift" in result.primary_failure_modes
    assert "action_saturation" in result.primary_failure_modes
    assert any("retrain" in step for step in result.recommended_next_steps)


def test_diagnosis_writes_json_and_report(tmp_path: Path) -> None:
    result = diagnose_rollout_comparison(_comparison(forward_ratio=0.1))
    write_rollout_failure_diagnosis_outputs(result, tmp_path)

    payload = json.loads((tmp_path / "rollout_failure_diagnosis.json").read_text())
    report = (tmp_path / "rollout_failure_diagnosis_report.md").read_text()

    assert payload["ok"] is False
    assert "weak_forward_locomotion" in payload["primary_failure_modes"]
    assert "Soridormi rollout failure diagnosis" in report


def test_report_contains_actionable_next_steps() -> None:
    result = diagnose_rollout_comparison(_comparison(action_abs=10.0, action_ratio=4.0))
    report = render_rollout_failure_diagnosis_report(result)

    assert "action_saturation" in report
    assert "normalization" in report
