from __future__ import annotations
import pytest

from pathlib import Path

from soridormi_runtime.rollout_compare import (
    RolloutComparisonThresholds,
    compare_rollout_summaries,
    render_rollout_comparison_report,
    write_rollout_comparison_outputs,
)


def _summary(*, path: str, duration: float, records: int, resets: int, forward: float, lateral: float, action_abs: float) -> dict:
    return {
        "path": path,
        "policy_records": records,
        "robot_time": {"duration": duration},
        "reset_cycles": {"count": resets},
        "action": {"abs_max": action_abs},
        "base_displacement": {"forward_x": forward, "lateral_y": lateral},
        "diagnosis": [f"diagnosis for {path}"],
    }


def test_rollout_comparison_passes_for_good_candidate() -> None:
    reference = _summary(path="teacher.mcap", duration=10.0, records=500, resets=0, forward=1.0, lateral=0.02, action_abs=1.5)
    candidate = _summary(path="candidate.mcap", duration=10.0, records=500, resets=0, forward=0.75, lateral=0.03, action_abs=1.8)

    result = compare_rollout_summaries(
        reference,
        candidate,
        thresholds=RolloutComparisonThresholds(
            min_candidate_policy_records=400,
            min_candidate_duration=9.0,
            min_forward_ratio=0.5,
            max_lateral_abs=0.1,
            max_lateral_ratio=2.0,
            max_action_abs=5.0,
        ),
    )

    assert result.ok
    assert result.comparison["forward_ratio"] == 0.75
    assert result.comparison["forward_speed_ratio"] == pytest.approx(0.75)
    assert not result.errors


def test_rollout_comparison_fails_for_bad_candidate() -> None:
    reference = _summary(path="teacher.mcap", duration=10.0, records=500, resets=0, forward=1.0, lateral=0.02, action_abs=1.5)
    candidate = _summary(path="candidate.mcap", duration=2.0, records=80, resets=1, forward=0.05, lateral=0.4, action_abs=8.0)

    result = compare_rollout_summaries(
        reference,
        candidate,
        thresholds=RolloutComparisonThresholds(
            min_candidate_policy_records=400,
            min_candidate_duration=9.0,
            max_candidate_resets=0,
            min_forward_ratio=0.5,
            max_lateral_abs=0.1,
            max_action_abs=5.0,
        ),
    )

    assert not result.ok
    assert any("policy_records" in item for item in result.errors)
    assert any("duration" in item for item in result.errors)
    assert any("reset count" in item for item in result.errors)
    assert any("forward ratio" in item for item in result.errors)
    assert any("lateral abs" in item for item in result.errors)
    assert any("action abs_max" in item for item in result.errors)


def test_rollout_comparison_writes_artifacts(tmp_path: Path) -> None:
    reference = _summary(path="teacher.mcap", duration=5.0, records=250, resets=0, forward=0.5, lateral=0.0, action_abs=1.0)
    candidate = _summary(path="candidate.mcap", duration=5.0, records=250, resets=0, forward=0.4, lateral=0.0, action_abs=1.1)
    result = compare_rollout_summaries(reference, candidate)

    write_rollout_comparison_outputs(result, tmp_path)

    payload = (tmp_path / "rollout_comparison.json").read_text(encoding="utf-8")
    report = (tmp_path / "rollout_comparison_report.md").read_text(encoding="utf-8")
    assert "forward_ratio" in payload
    assert "Soridormi policy rollout comparison" in report
    assert "PASS" in report


def test_rollout_comparison_report_contains_diagnosis() -> None:
    reference = _summary(path="teacher.mcap", duration=5.0, records=250, resets=0, forward=0.5, lateral=0.0, action_abs=1.0)
    candidate = _summary(path="candidate.mcap", duration=5.0, records=250, resets=0, forward=0.4, lateral=0.0, action_abs=1.1)
    result = compare_rollout_summaries(reference, candidate)

    report = render_rollout_comparison_report(result)

    assert "Reference diagnosis" in report
    assert "Candidate diagnosis" in report
    assert "diagnosis for teacher.mcap" in report
    assert "diagnosis for candidate.mcap" in report
