from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soridormi_runtime.m10_clearance_readiness import (
    build_m10_clearance_readiness,
    render_markdown,
)


def _report_payload(
    *,
    scenario_id: str,
    ok: bool,
    swing_p50: float | None = 0.03,
    swing_p05: float | None = 0.02,
    low_ratio: float | None = 0.0,
    samples_with_feet: int | None = 20,
    forward_distance_m: float = 0.2,
    stuck_ratio: float = 0.0,
    fallen: bool = False,
) -> dict:
    checks = [
        {
            "name": "foot_metrics_present",
            "ok": samples_with_feet is not None and samples_with_feet > 0,
            "value": samples_with_feet,
            "threshold": "present",
            "severity": "error",
        },
        {
            "name": "low_clearance_swing_ratio",
            "ok": low_ratio is not None and low_ratio <= 0.25,
            "value": low_ratio,
            "threshold": 0.25,
            "severity": "error",
        },
        {
            "name": "swing_clearance_p50_m",
            "ok": swing_p50 is not None and swing_p50 >= 0.015,
            "value": swing_p50,
            "threshold": 0.015,
            "severity": "error",
        },
    ]
    return {
        "ok": ok,
        "scenario_id": scenario_id,
        "scenario_title": scenario_id.replace("_", " "),
        "scenario_status": "mujoco_registry_ready",
        "scenario_family": "locomotion_flat",
        "expected_skill_id": "walk_velocity",
        "sample_count": 20,
        "duration_s": 2.0,
        "acceptance_thresholds": {
            "require_foot_metrics": True,
            "min_swing_clearance_m": 0.015,
            "max_low_clearance_ratio": 0.25,
        },
        "metrics": {
            "samples_with_feet": samples_with_feet,
            "swing_clearance_p05_m": swing_p05,
            "swing_clearance_p50_m": swing_p50,
            "low_clearance_swing_ratio": low_ratio,
            "forward_distance_m": forward_distance_m,
            "mean_forward_speed_mps": 0.1,
            "stuck_ratio": stuck_ratio,
            "fallen": fallen,
        },
        "checks": checks,
        "stride_step_report": {"samples_with_feet": samples_with_feet},
        "errors": [] if ok else ["clearance failed"],
        "warnings": [],
    }


def _write_suite_report(suite_dir: Path, scenario_id: str, payload: dict) -> Path:
    report_path = suite_dir / scenario_id / "scenario_rollout_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def test_readiness_blocks_low_clearance_with_manifest_thresholds(tmp_path: Path) -> None:
    suite_dir = tmp_path / "candidate_suite"
    _write_suite_report(
        suite_dir,
        "flat_walk_varied_speed_v1",
        _report_payload(
            scenario_id="flat_walk_varied_speed_v1",
            ok=False,
            swing_p50=0.010,
            swing_p05=0.006,
            low_ratio=0.50,
        ),
    )
    _write_suite_report(
        suite_dir,
        "start_stop_velocity_ramp_v1",
        _report_payload(scenario_id="start_stop_velocity_ramp_v1", ok=True),
    )
    _write_suite_report(
        suite_dir,
        "curve_turn_walk_v1",
        _report_payload(scenario_id="curve_turn_walk_v1", ok=True),
    )

    report = build_m10_clearance_readiness(profile="candidate", suite_dir=suite_dir)

    assert not report.ok
    assert report.gate_status == "BLOCKED_BY_CLEARANCE_GATE"
    assert report.summary_metrics["clearance_failed_count"] == 1
    assert report.summary_metrics["min_swing_clearance_p50_m"] == 0.010
    assert report.summary_metrics["max_low_clearance_ratio"] == 0.50
    assert report.summary_metrics["min_swing_clearance_p50_margin_m"] == pytest.approx(-0.005)
    assert report.summary_metrics["max_low_clearance_ratio_excess"] == 0.25
    assert report.summary_metrics["total_forward_distance_m"] == pytest.approx(0.6)
    assert any("flat_walk_varied_speed_v1" in blocker for blocker in report.blockers)
    flat = {item["scenario_id"]: item for item in report.scenarios}["flat_walk_varied_speed_v1"]
    assert flat["thresholds"]["max_low_clearance_ratio"] == 0.25
    assert any("phase/state-conditioned residual" in item for item in report.recommendations)
    assert flat["status"] == "FAIL_CLEARANCE_GATE"


def test_readiness_reports_missing_required_scenarios(tmp_path: Path) -> None:
    suite_dir = tmp_path / "partial_suite"
    _write_suite_report(
        suite_dir,
        "flat_walk_varied_speed_v1",
        _report_payload(scenario_id="flat_walk_varied_speed_v1", ok=True),
    )

    report = build_m10_clearance_readiness(profile="candidate", suite_dir=suite_dir)

    assert not report.ok
    assert report.missing_count == 2
    assert report.summary_metrics["foot_metrics_missing_count"] == 2
    assert any("missing scenario rollout report" in blocker for blocker in report.blockers)


def test_readiness_passes_clearance_before_visual_inspection(tmp_path: Path) -> None:
    suite_dir = tmp_path / "passing_suite"
    for scenario_id in (
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    ):
        _write_suite_report(
            suite_dir,
            scenario_id,
            _report_payload(scenario_id=scenario_id, ok=True, swing_p50=0.03, swing_p05=0.02, low_ratio=0.0),
        )

    report = build_m10_clearance_readiness(profile="candidate", suite_dir=suite_dir)
    rendered = render_markdown(report)

    assert report.ok
    assert report.gate_status == "READY_FOR_VISUAL_INSPECTION"
    assert report.summary_metrics["clearance_failed_count"] == 0
    assert "follow-camera visual inspection" in rendered
    assert "official-teacher comparison" in rendered


def test_readiness_compares_candidate_against_retained_reference(tmp_path: Path) -> None:
    candidate_suite = tmp_path / "candidate_suite"
    reference_suite = tmp_path / "reference_suite"
    scenarios = (
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    )
    reference_low_ratios = {
        "flat_walk_varied_speed_v1": 0.268,
        "start_stop_velocity_ramp_v1": 0.257,
        "curve_turn_walk_v1": 0.308,
    }
    candidate_low_ratios = {
        "flat_walk_varied_speed_v1": 0.258,
        "start_stop_velocity_ramp_v1": 0.252,
        "curve_turn_walk_v1": 0.285,
    }
    for scenario_id in scenarios:
        _write_suite_report(
            reference_suite,
            scenario_id,
            _report_payload(
                scenario_id=scenario_id,
                ok=False,
                swing_p50=0.018,
                low_ratio=reference_low_ratios[scenario_id],
                forward_distance_m=0.70,
            ),
        )
        _write_suite_report(
            candidate_suite,
            scenario_id,
            _report_payload(
                scenario_id=scenario_id,
                ok=False,
                swing_p50=0.019,
                low_ratio=candidate_low_ratios[scenario_id],
                forward_distance_m=0.68,
            ),
        )

    report = build_m10_clearance_readiness(
        profile="candidate",
        suite_dir=candidate_suite,
        reference_profile="reference",
        reference_suite_dir=reference_suite,
    )
    rendered = render_markdown(report)

    assert not report.ok
    assert report.candidate_beats_reference is True
    assert report.reference_profile == "reference"
    assert report.reference_summary_metrics is not None
    assert report.reference_summary_metrics["max_low_clearance_ratio"] == 0.308
    assert report.reference_comparison is not None
    assert report.reference_comparison["delta_max_low_clearance_ratio_excess"] == pytest.approx(-0.023)
    assert report.reference_comparison["movement_distance_preserved"] is True
    curve = {item["scenario_id"]: item for item in report.scenario_comparisons}["curve_turn_walk_v1"]
    assert curve["delta_low_clearance_ratio"] == pytest.approx(-0.023)
    assert "Reference comparison" in rendered
    assert "Candidate beats reference: yes" in rendered


def test_readiness_can_require_reference_improvement_from_cli(tmp_path: Path) -> None:
    candidate_suite = tmp_path / "candidate_suite"
    reference_suite = tmp_path / "reference_suite"
    output_dir = tmp_path / "readiness"
    for scenario_id in (
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    ):
        _write_suite_report(
            reference_suite,
            scenario_id,
            _report_payload(scenario_id=scenario_id, ok=False, swing_p50=0.018, low_ratio=0.30),
        )
        _write_suite_report(
            candidate_suite,
            scenario_id,
            _report_payload(
                scenario_id=scenario_id,
                ok=False,
                swing_p50=0.017,
                low_ratio=0.32,
                forward_distance_m=0.05,
            ),
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.m10_clearance_readiness",
            "--profile-name",
            "candidate",
            "--suite-dir",
            str(candidate_suite),
            "--reference-profile-name",
            "reference",
            "--reference-suite-dir",
            str(reference_suite),
            "--require-reference-improvement",
            "--output-dir",
            str(output_dir),
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
    assert payload["candidate_beats_reference"] is False
    assert payload["reference_comparison"]["blockers"]


def test_m10_clearance_readiness_cli_functionally_writes_reports(tmp_path: Path) -> None:
    suite_dir = tmp_path / "cli_suite"
    output_dir = tmp_path / "readiness"
    _write_suite_report(
        suite_dir,
        "flat_walk_varied_speed_v1",
        _report_payload(
            scenario_id="flat_walk_varied_speed_v1",
            ok=False,
            swing_p50=0.009,
            swing_p05=0.004,
            low_ratio=0.75,
        ),
    )
    _write_suite_report(
        suite_dir,
        "start_stop_velocity_ramp_v1",
        _report_payload(scenario_id="start_stop_velocity_ramp_v1", ok=True),
    )
    _write_suite_report(
        suite_dir,
        "curve_turn_walk_v1",
        _report_payload(scenario_id="curve_turn_walk_v1", ok=True),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.m10_clearance_readiness",
            "--profile-name",
            "candidate",
            "--suite-dir",
            str(suite_dir),
            "--output-dir",
            str(output_dir),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["gate_status"] == "BLOCKED_BY_CLEARANCE_GATE"
    assert payload["summary_metrics"]["clearance_failed_count"] == 1
    assert (output_dir / "clearance_readiness.json").exists()
    assert "Soridormi clearance readiness report" in (output_dir / "clearance_readiness.md").read_text()


def test_m10_clearance_readiness_script_functionally_preserves_wrapper(tmp_path: Path) -> None:
    suite_dir = tmp_path / "script_suite"
    output_dir = tmp_path / "script_readiness"
    for scenario_id in (
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    ):
        _write_suite_report(
            suite_dir,
            scenario_id,
            _report_payload(scenario_id=scenario_id, ok=True, swing_p50=0.03, swing_p05=0.02, low_ratio=0.0),
        )

    result = subprocess.run(
        [
            "bash",
            "scripts/analyze_clearance_readiness.sh",
            "--profile-name",
            "candidate",
            "--suite-dir",
            str(suite_dir),
            "--output-dir",
            str(output_dir),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["gate_status"] == "READY_FOR_VISUAL_INSPECTION"
    assert (output_dir / "clearance_readiness.json").exists()
    assert (output_dir / "clearance_readiness.md").exists()
